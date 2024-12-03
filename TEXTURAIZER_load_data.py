import comfy
import nodes
import folder_paths

import traceback
import json
import os
import torch
import numpy as np
from PIL import Image, ImageSequence, ImageOps
import requests
from io import BytesIO
import hashlib
import base64

# Initialize a blank 64x64 black image tensor in (B, H, W, C) format
blank_image = torch.zeros((1, 64, 64, 3))

def create_black_image_base64(width=64, height=64):
    img = Image.new('RGB', (width, height), color='black')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64

black_pixel_base64 = create_black_image_base64()

SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS + ['AYS SDXL', 'AYS SD1', 'AYS SVD', "GITS[coeff=1.2]"]

def parse_sampler_data(data):
    """
    Parses sampler configuration data from a dictionary, extracting
    seed, configuration values, sampler types, steps, and other parameters.
    """
    scene_data = data.get("scene_info", {})

    seed = int(scene_data['seed'])
    cfg = scene_data['cfg']
    sampler = scene_data['sampler']
    scheduler = scene_data['scheduler']
    steps = int(scene_data['steps'])
    denoise = scene_data['denoise']
    adv_step_end = int(scene_data['step_end'])
    adv_step_start = int(scene_data['step_start'])
    batch_size = int(scene_data['batch_size'])
    use_empty_latent = scene_data['use_empty_latent']

    return seed, cfg, sampler, scheduler, steps, denoise, adv_step_end, adv_step_start, batch_size, use_empty_latent,

def pil2tensor(img):
    """
    Converts a PIL image to a PyTorch tensor. Handles multi-frame images (e.g., GIFs)
    by iterating through each frame and stacking them into a single tensor.
    """
    output_images = []
    for i in ImageSequence.Iterator(img):
        i = ImageOps.exif_transpose(i)
        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        output_images.append(image)

    output_image = torch.cat(output_images, dim=0) if len(output_images) > 1 else output_images[0]
    return output_image

def load_image(image_source):
    """
    Loads an image from a URL or local file path. Returns a PIL image object.
    """
    if image_source.startswith('http'):
        response = requests.get(image_source)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_source)
    return img

def get_image_from_path(image_path):
    """
    Loads an image from a specified path, converting it to a PyTorch tensor.
    Returns a blank image if an error occurs.
    """
    try:
        img = load_image(image_path)
        img_out = pil2tensor(img)
    except Exception as e:
        img_out = blank_image
    return img_out

def get_image_from_base64(image_base64):
    """
    Decodes a base64-encoded image string and converts it to a PyTorch tensor.
    Returns a blank image if decoding fails.
    """
    try:
        imgdata = base64.b64decode(image_base64)
        i = Image.open(BytesIO(imgdata))
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        img_out = torch.from_numpy(image)[None,]
    except Exception as e:
        img_out = blank_image
    return img_out

#--------------------------------------# MEGA JSON READER #--------------------------------------#

class Texturaizer_SetGlobalDir:
    """
    Node to set and retrieve a global directory path for use in other nodes.
    Stores the path as a class variable accessible via get_global_dir_path().
    """
    global_dir_path = None  # Class variable to store the global directory path

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("global_dir_path",)
    FUNCTION = "set_dir_path"
    CATEGORY = "Texturaizer"
    OUTPUT_NODE = True

    def set_dir_path(self, directory):
        """
        Sets the global directory path for other nodes to use.
        """
        Texturaizer_SetGlobalDir.global_dir_path = directory
        print(f"[DEBUG] Global directory path set to: {directory}")
        return (directory,)

    @staticmethod
    def get_global_dir_path():
        """
        Returns the global directory path.
        """
        return Texturaizer_SetGlobalDir.global_dir_path

def read_json_from_directory(directory):
    """
    Reads a JSON file (ai_data.json) from the specified directory or from the global
    directory if none is provided. Returns the JSON data as a dictionary.
    """
    if not directory:
        directory = Texturaizer_SetGlobalDir.get_global_dir_path()
    
    json_file_path = os.path.join(directory, "data", "ai_data.json")

    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}
        
    data["texturaizer_save_dir"] = directory

    return data
def calculate_data_hash(data):
    """
    Calculates a hash for the given data, handling types such as strings,
    integers, booleans, lists, dictionaries, numpy arrays, and bytes.
    Returns an MD5 hash as a hexadecimal string.
    """
    try:
        if isinstance(data, (str, int, float, bool)):
            serialized_data = str(data).encode('utf-8')
        elif isinstance(data, (list, tuple)):
            serialized_data = json.dumps([calculate_data_hash(item) for item in data], sort_keys=True).encode('utf-8')
        elif isinstance(data, dict):
            serialized_data = json.dumps({k: calculate_data_hash(v) for k, v in data.items()}, sort_keys=True).encode('utf-8')
        elif isinstance(data, np.ndarray):
            serialized_data = data.tobytes()
        elif isinstance(data, bytes):
            serialized_data = data
        else:
            serialized_data = repr(data).encode('utf-8')

        data_hash = hashlib.md5(serialized_data).hexdigest()
        return data_hash

    except Exception as e:
        print(f"[ERROR] Failed to hash data: {e}")
        return None

def calculate_image_hash(image):
    """
    Calculates an MD5 hash for a single image, represented as a numpy array or tensor.
    """
    image_bytes = image.detach().cpu().numpy().tobytes()
    image_hash = hashlib.md5(image_bytes).hexdigest()
    return image_hash

def combine_hashes(hashes):
    """
    Combines multiple individual hashes into a single hash by concatenating
    the hash strings and rehashing the combined string.
    """
    combined_string = ''.join(hashes)
    combined_hash = hashlib.md5(combined_string.encode('utf-8')).hexdigest()
    return combined_hash

def combo_image_hash(*args):
    """
    Generates a combined hash for multiple images by calculating individual hashes
    and then combining them into a single hash.
    """
    hashes = [calculate_image_hash(image) for image in args]
    combined_hash = combine_hashes(hashes)
    return combined_hash

def get_data(directory_optional, data_optional):
    """
    Retrieves JSON data from either the provided directory or the default directory.
    """
    return data_optional if data_optional else read_json_from_directory(directory_optional)

class Texturaizer_GetJsonData:
    """
    Node for retrieving JSON data from a specified directory or global directory.
    Computes and returns the data hash to detect changes in the data.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "directory_optional": ("STRING", {"default": ""}),
                "data_optional": ("DICTIONARY", {"default": {}}),
            }
        }

    RETURN_TYPES = ("DICTIONARY", "STRING")
    RETURN_NAMES = ("json_data", "data_hash")
    OUTPUT_TOOLTIPS = (
        "Texturaizer JSON data.",
        "Hash value for debugging purposes."
    )
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Retrieves JSON data from the specified directory or the global directory if not specified."

    def read_json_data(self, directory_optional="", data_optional={}):
        """
        Reads JSON data from the specified directory or the default directory if not provided.
        Calculates a hash to detect changes in the data.
        """
        data = get_data(directory_optional, data_optional)
        data_hash = calculate_data_hash(data)
        return (data, data_hash)

    @staticmethod
    def IS_CHANGED(directory_optional="", data_optional={}):
        """
        Checks for changes by returning the hash of the JSON data.
        """
        data = get_data(directory_optional, data_optional)
        data_hash = calculate_data_hash(data)
        return (data_hash,)

checkpoint_names = folder_paths.get_filename_list("checkpoints")
try:
    unet_names = folder_paths.get_filename_list("unet_gguf")
except:
    from .any_type import any
    unet_names = any

class Texturaizer_GetModelName(Texturaizer_GetJsonData):
    """
    Node for retrieving model checkpoint and unet names from JSON data.
    Extends the JSON retrieval functionality to return model-specific information.
    """

    RETURN_TYPES = (checkpoint_names, unet_names, "STRING")
    RETURN_NAMES = ("checkpoint_name", "unet_name", "data_hash")
    OUTPUT_TOOLTIPS = (
        "Diffusion checkpoint model name (stable diffusion).",
        "Diffusion unet model name (flux).",
        "Hash value for debugging purposes."
    )
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Retrieves the diffusion model name from the specified directory or the global directory."

    def read_json_data(self, directory_optional="", data_optional={}):
        """
        Reads the model name from JSON data and returns it with a data hash.
        """
        data = get_data(directory_optional, data_optional)
        scene_data = data.get("scene_info", {})
        model = scene_data["ai_model"]
        data_hash = calculate_data_hash(model)
        return (model, model, data_hash)

    @staticmethod
    def IS_CHANGED(directory_optional="", data_optional={}):
        """
        Checks for changes by hashing the model name in the JSON data.
        """
        data = get_data(directory_optional, data_optional)
        scene_data = data.get("scene_info", {})
        model = scene_data["ai_model"]
        data_hash = calculate_data_hash(model)
        return (data_hash,)

try:
    clip_names = folder_paths.get_filename_list("text_encoders") + folder_paths.get_filename_list("clip_gguf")
except:
    clip_names = folder_paths.get_filename_list("text_encoders")
clip_names = sorted(clip_names)

class Texturaizer_GetClipModelName(Texturaizer_GetJsonData):
    """
    Node for retrieving the names of two CLIP models from JSON data.
    Extends JSON retrieval to provide clip-specific information.
    """

    RETURN_TYPES = (clip_names, clip_names, "STRING")
    RETURN_NAMES = ("clip_model_1", "clip_model_2", "data_hash")
    OUTPUT_TOOLTIPS = (
        "First flux CLIP model name.",
        "Second flux CLIP model name.",
        "Hash value for debugging purposes."
    )
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Retrieves CLIP model names from the specified directory or the global directory."

    def read_json_data(self, directory_optional="", data_optional={}):
        """
        Reads CLIP model names from JSON data and returns them with a combined hash.
        """
        data = get_data(directory_optional, data_optional)
        scene_data = data.get("scene_info", {})
        clip1 = scene_data["clip_1"]
        clip2 = scene_data["clip_2"]
        data_hash = calculate_data_hash([clip1, clip2])
        return (clip1, clip2, data_hash)

    @staticmethod
    def IS_CHANGED(directory_optional="", data_optional={}):
        """
        Checks for changes by hashing the CLIP model names in the JSON data.
        """
        data = get_data(directory_optional, data_optional)
        scene_data = data.get("scene_info", {})
        clip1 = scene_data["clip_1"]
        clip2 = scene_data["clip_2"]
        data_hash = calculate_data_hash([clip1, clip2])
        return (data_hash,)
    
vae_names = folder_paths.get_filename_list("vae")

class Texturaizer_GetVAEName(Texturaizer_GetJsonData):
    """
    Node for retrieving the VAE model name from JSON data.
    Computes a hash for the VAE name to detect changes.
    """

    RETURN_TYPES = (vae_names, "STRING")
    RETURN_NAMES = ("vae_name", "data_hash")
    OUTPUT_TOOLTIPS = (
        "The VAE name.",
        "Hash value for debugging purposes."
    )
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Retrieves the VAE name from the provided directory or from the global directory if not specified."

    def read_json_data(self, directory_optional="", data_optional={}):
        """
        Reads the VAE model name from JSON data and calculates a data hash.
        """
        data = get_data(directory_optional, data_optional)
        scene_data = data.get("scene_info", {})
        vae = scene_data["vae"]
        data_hash = calculate_data_hash(vae)
        return (vae, data_hash)

    @staticmethod
    def IS_CHANGED(directory_optional="", data_optional={}):
        """
        Checks for changes by hashing the VAE model name in the JSON data.
        """
        data = get_data(directory_optional, data_optional)
        scene_data = data.get("scene_info", {})
        vae = scene_data["vae"]
        data_hash = calculate_data_hash(vae)
        return (data_hash,)

def get_images(data):
    """
    Retrieves multiple images from JSON data based on specified keys.
    Supports both embedded base64 data and file paths for image layers.
    """
    scene_data = data.get("scene_info", {})
    keys = [
        'image_path_base', 'image_path_seg', 'image_path_depth', 
        'image_path_normal', 'image_path_edge', 
        'image_path_seg_obj', 'image_path_seg_mat'
    ]
    if not scene_data["embed_data"]:
        images_dir = os.path.join(data["texturaizer_save_dir"], "image layers")
        try:
            image_paths = [os.path.join(images_dir, scene_data[key]) for key in keys]
            images = [get_image_from_path(img) for img in image_paths]
            return tuple(images)
        except Exception as e:
            print(f"Error parsing JSON file: {e}")
            traceback.print_exc()
            return (blank_image,) * 7
    else:
        images = [get_image_from_base64(scene_data.get(key, black_pixel_base64)) for key in keys]
        return tuple(images)

class Texturaizer_GetImageData(Texturaizer_GetJsonData):
    """
    Node for retrieving various image layers (e.g., depth, normal, edge) from JSON data.
    Computes a combined hash for all images to detect changes.
    """

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("base", "segment active", "depth", "normal", "edge", "segment obj", "segment mat", "data_hash")
    OUTPUT_TOOLTIPS = (
        "Base image",
        "Active Segment image",
        "Depth pass",
        "Normal pass",
        "Edge pass",
        "Object segment image",
        "Material segment image",
        "Hash value for debugging purposes."
    )
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Retrieves image layers from the provided directory or the global directory if not specified."

    def read_json_data(self, directory_optional="", data_optional={}):
        """
        Retrieves and returns image layers with a hash for change detection.
        """
        data = get_data(directory_optional, data_optional)
        base, segment, depth, normal, edge, seg_obj, seg_mat = get_images(data)
        data_hash = combo_image_hash(base, segment, depth, normal, edge, seg_obj, seg_mat)
        return (base, segment, depth, normal, edge, seg_obj, seg_mat, data_hash)

    @staticmethod
    def IS_CHANGED(directory_optional="", data_optional={}):
        """
        Checks for changes by hashing the image layers in JSON data.
        """
        data = get_data(directory_optional, data_optional)
        base, segment, depth, normal, edge, seg_obj, seg_mat = get_images(data)
        data_hash = combo_image_hash(base, segment, depth, normal, edge, seg_obj, seg_mat)
        return (data_hash,)

def get_ip(data):
    """
    Retrieves IP adapter images and related parameters from JSON data.
    Supports both embedded base64 data and file paths.
    """
    ip_adapter_data = data.get("ip_adapters", {})
    scene_data = data.get("scene_info", {})
    
    def process_ip(ip_key, embed_data):
        ip = ip_adapter_data.get(ip_key, '')
        try:
            return get_image_from_path(ip) if not embed_data else get_image_from_base64(ip)
        except:
            return blank_image

    embed_data = scene_data["embed_data"]
    
    ip1 = process_ip('image_path_ipadapter_1', embed_data)
    ip_weight_Encode_1 = ip_adapter_data.get("ip_weight_Encode_1", 0.0)

    ip2 = process_ip('image_path_ipadapter_2', embed_data)
    ip_weight_Encode_2 = ip_adapter_data.get("ip_weight_Encode_2", 0.0)

    ip_loader_preset = ip_adapter_data.get("ip_loader_preset", "STANDARD (medium strength)")
    use_ipadapter = ip_adapter_data.get("use_ipadapter", False)

    return (ip1, ip_weight_Encode_1, ip2, ip_weight_Encode_2, ip_loader_preset, use_ipadapter, ip_adapter_data)

IP_PRESETS = [
    'LIGHT - SD1.5 only (low strength)', 'STANDARD (medium strength)', 'VIT-G (medium strength)',
    'PLUS (high strength)', 'PLUS FACE (portraits)', 'FULL FACE - SD1.5 only (portraits stronger)'
]
WEIGHT_TYPES = [
    "linear", "ease in", "ease out", 'ease in-out', 'reverse in-out', 'weak input',
    'weak output', 'weak middle', 'strong middle', 'style transfer', 'composition',
    'strong style transfer', 'style and composition', 'style transfer precise', 'composition precise'
]
SCALING = ['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty']

class Texturaizer_GetIPAdapterData(Texturaizer_GetJsonData):
    """
    Node for retrieving IP adapter images and weights from JSON data.
    Computes a combined hash to detect changes in IP adapter data.
    """

    RETURN_TYPES = ("IMAGE", "FLOAT", "IMAGE", "FLOAT", IP_PRESETS, "BOOLEAN", "DICTIONARY", "STRING")
    RETURN_NAMES = ("ip image 1", "ip1 weight", "ip image 2", "ip2 weight", "preset", "use IP", "IP data", "data_hash")
    OUTPUT_TOOLTIPS = (
        "IP image 1",
        "Weight of first IP adapter image",
        "IP image 2",
        "Weight of second IP adapter image",
        "IP preset type",
        "Enable/Disable IP Adapter",
        "IP Adapter data",
        "Hash value for debugging purposes."
    )
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Retrieves IPAdapter data from the specified directory or the global directory if not specified."

    def read_json_data(self, directory_optional="", data_optional={}):
        """
        Retrieves IP adapter data and computes a combined hash for change detection.
        """
        data = get_data(directory_optional, data_optional)
        ip1, ip_weight_Encode_1, ip2, ip_weight_Encode_2, ip_loader_preset, use_ipadapter, ip_adapter_data = get_ip(data)
        image_hash = combo_image_hash(ip1, ip2)
        data_hash = calculate_data_hash([image_hash, ip_adapter_data])
        return (ip1, ip_weight_Encode_1, ip2, ip_weight_Encode_2, ip_loader_preset, use_ipadapter, ip_adapter_data, data_hash)

    @staticmethod
    def IS_CHANGED(directory_optional="", data_optional={}):
        """
        Checks for changes by hashing the IP adapter images and data.
        """
        data = get_data(directory_optional, data_optional)
        ip1, ip_weight_Encode_1, ip2, ip_weight_Encode_2, ip_loader_preset, use_ipadapter, ip_adapter_data = get_ip(data)
        image_hash = combo_image_hash(ip1, ip2)
        data_hash = calculate_data_hash([image_hash, ip_adapter_data])
        return (data_hash,)

class Texturaizer_IPAdapterEmbeds:
    """
    Node to retrieve IP Adapter Embeds data from a provided dictionary.
    Returns parameters such as weight, type, start, end, and scaling for IP embeds.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ip_data": ("DICTIONARY",),
            },
        }

    RETURN_TYPES = ("FLOAT", WEIGHT_TYPES, "FLOAT", "FLOAT", SCALING)
    RETURN_NAMES = ("embed weight", "weight type", "embed start", "embed end", "embeds scaling")
    FUNCTION = "execute"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Retrieves IPAdapter Embeds data from the 'Get IP Adapter Data' node."

    def execute(self, ip_data):
        """
        Extracts embed weight, type, start/end, and scaling from IP adapter data.
        """
        ip_weight_embed = ip_data["ip_weight_embed"]
        ip_weight_type = ip_data["ip_weight_type"]
        ip_start = ip_data["ip_start"]
        ip_end = ip_data["ip_end"]
        ip_embeds_scaling = ip_data["ip_embeds_scaling"]
        return (ip_weight_embed, ip_weight_type, ip_start, ip_end, ip_embeds_scaling)

class Texturaizer_GetLoraData(Texturaizer_GetJsonData):
    """
    Node to retrieve LoRA data from JSON and calculate a hash for change detection.
    """

    RETURN_TYPES = ("DICTIONARY", "STRING")
    RETURN_NAMES = ("LoRAs", "data_hash")
    OUTPUT_TOOLTIPS = ("LoRA data", "Hash value for debugging purposes.")
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Retrieves LoRA data from the provided directory or the global directory if not specified."

    def read_json_data(self, directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)
        loras = data.get("loras", {})
        data_hash = calculate_data_hash(loras)
        return (loras, data_hash)

    @staticmethod
    def IS_CHANGED(directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)
        loras = data.get("loras", {})
        data_hash = calculate_data_hash(loras)
        return (data_hash,)

class Texturaizer_GetSamplerData(Texturaizer_GetJsonData):
    """
    Node to retrieve sampler data (e.g., seed, steps, scheduler) from JSON data.
    Computes a hash to detect changes in sampler configuration.
    """

    RETURN_TYPES = ("INT", "FLOAT", comfy.samplers.KSampler.SAMPLERS, SCHEDULERS, "INT", "FLOAT", "INT", "INT", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("seed", "cfg", "sampler", "scheduler", "steps", "denoise", "adv steps", "adv steps start", "batch size", "use empty latent", "data_hash")
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Retrieves sampler data from the provided directory or the global directory if not specified."

    def read_json_data(self, directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)
        seed, cfg, sampler, scheduler, steps, denoise, adv_step_end, adv_step_start, batch_size, use_empty_latent = parse_sampler_data(data)
        data_hash = calculate_data_hash((seed, cfg, sampler, scheduler, steps, denoise, adv_step_end, adv_step_start, batch_size, use_empty_latent))
        return (seed, cfg, sampler, scheduler, steps, denoise, adv_step_end, adv_step_start, batch_size, use_empty_latent, data_hash)

    @staticmethod
    def IS_CHANGED(directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)
        seed, cfg, sampler, scheduler, steps, denoise, adv_step_end, adv_step_start, batch_size, use_empty_latent = parse_sampler_data(data)
        data_hash = calculate_data_hash((seed, cfg, sampler, scheduler, steps, denoise, adv_step_end, adv_step_start, batch_size, use_empty_latent))
        return (data_hash,)

class Texturaizer_GetRenderData(Texturaizer_GetJsonData):
    """
    Node to retrieve rendering width and height settings from JSON data.
    Calculates a hash for detecting changes in render dimensions.
    """

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "data_hash")
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Retrieves render settings from the specified directory or the global directory if not specified."

    def read_json_data(self, directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)
        scene_data = data.get("scene_info", {})
        width = int(scene_data['width'])
        height = int(scene_data['height'])
        data_hash = calculate_data_hash((width, height))
        return (width, height, data_hash)

    @staticmethod
    def IS_CHANGED(directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)
        scene_data = data.get("scene_info", {})
        width = int(scene_data['width'])
        height = int(scene_data['height'])
        data_hash = calculate_data_hash((width, height))
        return (data_hash,)

def get_prompt(data):
    """
    Retrieves the positive and negative prompts from JSON data.
    """
    scene_data = data.get("scene_info", {})
    pos_g = scene_data['positive_prompt_g']
    pos_l = scene_data['positive_prompt_l']
    neg = scene_data['negative_prompt']
    return pos_g, pos_l, neg

class Texturaizer_GetPromptData(Texturaizer_GetJsonData):
    """
    Node to retrieve prompt data (positive and negative) from JSON data.
    Computes a hash to detect changes in the prompt information.
    """

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("pos_g", "pos_l", "neg", "data_hash")
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Retrieves prompt data from the provided directory or the global directory if not specified."

    def read_json_data(self, directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)
        pos_g, pos_l, neg = get_prompt(data)
        data_hash = calculate_data_hash((pos_g, pos_l, neg))
        return (pos_g, pos_l, neg, data_hash)

    @staticmethod
    def IS_CHANGED(directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)
        pos_g, pos_l, neg = get_prompt(data)
        data_hash = calculate_data_hash((pos_g, pos_l, neg))
        return (data_hash,)

def get_style(data):
    """
    Retrieves style settings from JSON data.
    """
    scene_data = data.get("scene_info", {})
    use_style = scene_data['use_style']
    style = scene_data['style']
    style_pos = scene_data['style_pos']
    style_neg = scene_data['style_neg']
    return use_style, style, style_pos, style_neg

class Texturaizer_GetStyleData(Texturaizer_GetJsonData):
    """
    Node to retrieve style data from JSON data.
    Calculates a hash to detect changes in style settings.
    """

    RETURN_TYPES = ("BOOLEAN", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("use_style", "style", "style_pos", "style_neg", "data_hash")
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Retrieves style data from the specified directory or the global directory if not specified."

    def read_json_data(self, directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)
        use_style, style, style_pos, style_neg = get_style(data)
        data_hash = calculate_data_hash((use_style, style, style_pos, style_neg))
        return (use_style, style, style_pos, style_neg, data_hash)

    @staticmethod
    def IS_CHANGED(directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)
        use_style, style, style_pos, style_neg = get_style(data)
        data_hash = calculate_data_hash((use_style, style, style_pos, style_neg))
        return (data_hash,)

def get_seg_data(data):
    """
    Retrieves segmentation-related data, including prompts, segment dimensions,
    and additional settings from JSON data.
    """
    scene_info = data['scene_info']
    positive_l = scene_info['positive_prompt_l']
    positive_g = scene_info['positive_prompt_g']
    use_other_prompt = scene_info['use_other_prompt']
    other_prompt = scene_info['other_prompt']
    prepend_pos_prompt_g = scene_info['prepend_pos_prompt_g']
    delimiter = scene_info['delimiter']
    append_pos_prompt_l = scene_info['append_pos_prompt_l']
    width = scene_info['width']
    height = scene_info['height']
    use_segment_data = scene_info['use_segment_data']
    strength = scene_info['condition_strength']
    expand = scene_info['mask_expand']
    blur = scene_info['mask_blur']
    version_select = scene_info['version_select']
    segment_type = scene_info['segment_type']
    use_style = scene_info['use_style']
    style_pos = scene_info['style_pos']
    seg = data[segment_type]

    segment_data = [
        positive_l, positive_g, use_other_prompt, other_prompt,
        prepend_pos_prompt_g, delimiter, append_pos_prompt_l,
        width, height, use_segment_data, segment_type, strength,
        expand, blur, version_select, use_style, style_pos, seg,
    ]

    return segment_data

class Texturaizer_GetSegData(Texturaizer_GetJsonData):
    """
    Node for retrieving segmentation data from JSON, computing a hash to detect changes.
    """

    RETURN_TYPES = ("DICTIONARY", "STRING")
    RETURN_NAMES = ("data", "data_hash")
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Retrieves segment data from the specified directory or the global directory if not specified."

    def read_json_data(self, directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)
        seg_data = get_seg_data(data)
        data_hash = calculate_data_hash(seg_data)
        return (data, data_hash)

    @staticmethod
    def IS_CHANGED(directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)
        seg_data = get_seg_data(data)
        data_hash = calculate_data_hash(seg_data)
        return (data_hash,)

def cn_add_preprocessed_image(controlnets, data):
    """
    Adds preprocessed images to each controlnet in the dictionary. If images are
    not embedded, loads them from specified paths; otherwise, uses base64 data.
    Returns the updated controlnets and a combined image hash.
    """
    scene_info = data["scene_info"]
    embed_data = scene_info["embed_data"]
    preprocessed_images = []

    for cn_key, cn in controlnets.items():
        if not embed_data:
            base_dir = data["texturaizer_save_dir"]
            images_dir = os.path.join(base_dir, "image layers")
            preprocess_image_path = cn.get('preprocess_image_path', "none")
            full_preprocess_image_path = os.path.join(images_dir, preprocess_image_path)
            preprocessed_image = get_image_from_path(full_preprocess_image_path)
            cn['preprocessed_image'] = preprocessed_image
            preprocessed_images.append(preprocessed_image)

    images_hash = combo_image_hash(*preprocessed_images)
    return controlnets, images_hash

class Texturaizer_GetCNData(Texturaizer_GetJsonData):
    """
    Node to retrieve ControlNet data from JSON, with optional preprocessed images.
    Computes a hash combining ControlNet data and image hashes to detect changes.
    """

    RETURN_TYPES = ("DICTIONARY", "STRING")
    RETURN_NAMES = ("controlnets", "data_hash")
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Retrieves ControlNet data from the specified directory or the global directory if not specified."

    def read_json_data(self, directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)
        controlnets = data.get("controlnets", {})
        controlnets_processed, images_hash = cn_add_preprocessed_image(controlnets, data)
        data_hash = calculate_data_hash((controlnets, images_hash))
        return (controlnets_processed, data_hash)

    @staticmethod
    def IS_CHANGED(directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)
        controlnets = data.get("controlnets", {})
        controlnets_processed, images_hash = cn_add_preprocessed_image(controlnets, data)
        data_hash = calculate_data_hash((controlnets, images_hash))
        return (data_hash,)

class Texturaizer_UseSDXL(Texturaizer_GetJsonData):
    """
    Node to determine if SDXL version is used. Computes a hash for change detection.
    """

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("use sdxl", "data_hash")
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Retrieves the use_sdxl property from the specified directory or the global directory if not specified."

    def read_json_data(self, directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)
        scene_data = data.get("scene_info", {})
        use_sdxl = (scene_data["version_select"] == 'SDXL')
        data_hash = calculate_data_hash(use_sdxl)
        return (use_sdxl, data_hash)

    @staticmethod
    def IS_CHANGED(directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)
        scene_data = data.get("scene_info", {})
        use_sdxl = (scene_data["version_select"] == 'SDXL')
        data_hash = calculate_data_hash(use_sdxl)
        return (data_hash,)

class Texturaizer_GetFluxGuidance(Texturaizer_GetJsonData):
    """
    Node to retrieve flux guidance setting from JSON. Computes a hash for change detection.
    """

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("flux guidance", "data_hash")
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Retrieves the flux_guidance property from the specified directory or the global directory if not specified."

    def read_json_data(self, directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)
        scene_data = data.get("scene_info", {})
        flux_guidance = (scene_data["flux_guidance"])
        data_hash = calculate_data_hash(flux_guidance)
        return (flux_guidance, data_hash)

    @staticmethod
    def IS_CHANGED(directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)
        scene_data = data.get("scene_info", {})
        flux_guidance = (scene_data["flux_guidance"])
        data_hash = calculate_data_hash(flux_guidance)
        return (data_hash,)

NODE_CLASS_MAPPINGS = {
    "Texturaizer_SetGlobalDir": Texturaizer_SetGlobalDir,  
    "Texturaizer_GetJsonData": Texturaizer_GetJsonData,  
    "Texturaizer_GetModelName": Texturaizer_GetModelName,
    "Texturaizer_GetClipModelName": Texturaizer_GetClipModelName,
    "Texturaizer_GetVAEName": Texturaizer_GetVAEName,
    "Texturaizer_GetImageData": Texturaizer_GetImageData,
    "Texturaizer_GetIPAdapterData": Texturaizer_GetIPAdapterData,   
    "Texturaizer_IPAdapterEmbeds": Texturaizer_IPAdapterEmbeds,   
    "Texturaizer_GetLoraData": Texturaizer_GetLoraData,
    "Texturaizer_GetSamplerData": Texturaizer_GetSamplerData,
    "Texturaizer_GetRenderData": Texturaizer_GetRenderData,
    "Texturaizer_GetPromptData": Texturaizer_GetPromptData,
    "Texturaizer_GetStyleData": Texturaizer_GetStyleData,
    "Texturaizer_GetSegData": Texturaizer_GetSegData,
    "Texturaizer_GetCNData": Texturaizer_GetCNData,
    "Texturaizer_UseSDXL": Texturaizer_UseSDXL,
    "Texturaizer_GetFluxGuidance": Texturaizer_GetFluxGuidance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Texturaizer_SetGlobalDir": "Set Global Dir (Texturaizer)",
    "Texturaizer_GetJsonData": "Get Json Data (Texturaizer)",
    "Texturaizer_GetModelName": "Get Model Name (Texturaizer)",
    "Texturaizer_GetClipModelName": "Get Clip Model Names (Texturaizer)",
    "Texturaizer_GetVAEName": "Get VAE Name (Texturaizer)",
    "Texturaizer_GetImageData": "Get Image Data (Texturaizer)",
    "Texturaizer_GetIPAdapterData": "Get IPAdapter Data (Texturaizer)",
    "Texturaizer_IPAdapterEmbeds": "IPAdapter Embeds (Texturaizer)",
    "Texturaizer_GetLoraData": "Get LoRA Data (Texturaizer)",
    "Texturaizer_GetSamplerData": "Get Sampler Data (Texturaizer)",
    "Texturaizer_GetRenderData": "Get Render Data (Texturaizer)",
    "Texturaizer_GetPromptData": "Get Prompt Data (Texturaizer)",
    "Texturaizer_GetStyleData": "Get Style Data (Texturaizer)",
    "Texturaizer_GetSegData": "Get Segment Data (Texturaizer)",
    "Texturaizer_GetCNData": "Get ControlNet Data (Texturaizer)",
    "Texturaizer_UseSDXL": "Use SDXL? (Texturaizer)",
    "Texturaizer_GetFluxGuidance": "Get Flux Guidance (Texturaizer)",
}
