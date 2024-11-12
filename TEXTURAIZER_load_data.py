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



blank_image = torch.zeros((1, 1, 1, 3))   # 1x1 black image tensor
SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS + ['AYS SDXL', 'AYS SD1', 'AYS SVD', "GITS[coeff=1.2]"]
        
def parse_sampler_data(data):
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
    output_images = []
    for i in ImageSequence.Iterator(img):
        i = ImageOps.exif_transpose(i)
        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        output_images.append(image)

    if len(output_images) > 1:
        output_image = torch.cat(output_images, dim=0)
    else:
        output_image = output_images[0]

    return output_image

def load_image(image_source):
    if image_source.startswith('http'):
        response = requests.get(image_source)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_source)
    return img

def get_image_from_path(image_path):
    try:
        img = load_image(image_path)
        img_out = pil2tensor(img)
    except Exception as e:
        # print(f"Error loading image from path {image_path}: {e}")
        img_out = blank_image
    return img_out

black_pixel_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wcAAwAB/ccf8AAAAABJRU5ErkJggg=="
def get_image_from_base64(image_base64):
    # Decode the base64 string
    try:
        imgdata = base64.b64decode(image_base64)

        # Open the image from memory
        i = Image.open(BytesIO(imgdata))
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        img_out = torch.from_numpy(image)[None,]
    except Exception as e:
        # print(f"Error loading image from path {image_path}: {e}")
        img_out = blank_image
    return img_out

#--------------------------------------# MEGA JSON READER #--------------------------------------#

class Texturaizer_SetGlobalDir:
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
    Reads the JSON file from the specified directory.
    """
    if not directory:  # Use the provided directory
        directory = Texturaizer_SetGlobalDir.get_global_dir_path()
        # if not directory:
        #     print("[DEBUG] No directory found")
    
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
    Calculates the hash of the given data, handling various types (strings, ints, booleans, lists, tensors, etc.)
    """
    try:
        # For basic types (str, int, float, bool), just convert to string and hash
        if isinstance(data, (str, int, float, bool)):
            serialized_data = str(data).encode('utf-8')

        # For lists or tuples, hash each element recursively
        elif isinstance(data, (list, tuple)):
            serialized_data = json.dumps([calculate_data_hash(item) for item in data], sort_keys=True).encode('utf-8')

        # For dictionaries, serialize and hash the keys and values
        elif isinstance(data, dict):
            serialized_data = json.dumps({k: calculate_data_hash(v) for k, v in data.items()}, sort_keys=True).encode('utf-8')

        # For numpy arrays (used for images), hash the array's bytes
        elif isinstance(data, np.ndarray):
            serialized_data = data.tobytes()

        # For binary data (e.g., images in binary format)
        elif isinstance(data, bytes):
            serialized_data = data

        # For unsupported or complex types, fallback to repr
        else:
            serialized_data = repr(data).encode('utf-8')

        # Calculate and return the MD5 hash
        data_hash = hashlib.md5(serialized_data).hexdigest()
        return data_hash

    except Exception as e:
        print(f"[ERROR] Failed to hash data: {e}")
        return None


def calculate_image_hash(image):
    # Calculates the hash for an individual image (as a numpy array).
    image_bytes = image.detach().cpu().numpy().tobytes() 
    # image_bytes = image.tobytes()  # Convert the image (numpy array) to bytes
    image_hash = hashlib.md5(image_bytes).hexdigest()
    return image_hash

def combine_hashes(hashes):
    # Combines a list of individual hashes into a single combined hash.
    combined_string = ''.join(hashes)  # Concatenate all the individual hashes
    combined_hash = hashlib.md5(combined_string.encode('utf-8')).hexdigest()  # Calculate the combined hash
    return combined_hash

def combo_image_hash(*args):
    hashes = []
    for image in args:  # Iterate over the values of kwargs (the images)
        hashes.append(calculate_image_hash(image))
    combined_hash = combine_hashes(hashes)
    return combined_hash

def get_data(directory_optional, data_optional):
    if data_optional:
        return data_optional
    else:
        return read_json_from_directory(directory_optional)
        
class Texturaizer_GetJsonData:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
        },
        "optional": {
            "directory_optional": ("STRING", {"default": ""}),  # Optional directory input
            "data_optional": ("DICTIONARY", {"default": {}}),  # Optional directory input
        }}

    RETURN_TYPES = ("DICTIONARY", "STRING")
    RETURN_NAMES = ("json_data", "data_hash")
    OUTPUT_TOOLTIPS = ("Texturaizer json data.", 
                       "The hash value for debuging purposes.")
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Gets the JSON data from the provided directory or from 'Texturaizer set dir' node if not specified"

    def read_json_data(self, directory_optional="", data_optional={}):
        """
        Reads the JSON data from the provided directory or the global directory path if not specified.
        Only triggers downstream execution if the data has changed.
        """
        data = get_data(directory_optional, data_optional)
        data_hash = calculate_data_hash(data)

        return (data, data_hash)

    @staticmethod
    def IS_CHANGED(directory_optional="", data_optional={}):
        """
        Returns the hash of the JSON data to detect changes.
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
    RETURN_TYPES = (checkpoint_names, unet_names, "STRING")
    RETURN_NAMES = ("checkpoint_name", "unet_name", "data_hash")
    OUTPUT_TOOLTIPS = ("The diffusion checkpoint model name (stable diffusion).", 
                       "The diffusion unet model name (flux).", 
                       "The hash value for debuging purposes.")
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Gets the diffusion model name from the provided directory or from 'Texturaizer set dir' node if not specified"
    
    def read_json_data(self, directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)

        scene_data = data.get("scene_info", {})
        model = scene_data["ai_model"]
        
        data_hash = calculate_data_hash(model)
        return (model, model, data_hash)

    @staticmethod
    def IS_CHANGED(directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)
        
        scene_data = data.get("scene_info", {})
        model = scene_data["ai_model"]

        data_hash = calculate_data_hash(model)

        return (data_hash,)
    

clip_names = folder_paths.get_filename_list("clip") + folder_paths.get_filename_list("clip_gguf")
class Texturaizer_GetClipModelName(Texturaizer_GetJsonData):
    RETURN_TYPES = (clip_names, clip_names, "STRING")
    RETURN_NAMES = ("clip_model_1", "clip_model_2", "data_hash")
    OUTPUT_TOOLTIPS = ("The first flux clip model name.", 
                       "The second flux clip model name.", 
                       "The hash value for debuging purposes.")
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Gets the diffusion model name from the provided directory or from 'Texturaizer set dir' node if not specified"
    
    def read_json_data(self, directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)

        scene_data = data.get("scene_info", {})
        clip1 = scene_data["clip_1"]
        clip2 = scene_data["clip_2"]
        data_hash = calculate_data_hash([clip1, clip2])

        return (clip1, clip2, data_hash)

    @staticmethod
    def IS_CHANGED(directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)
        
        scene_data = data.get("scene_info", {})
        clip1 = scene_data["clip_1"]
        clip2 = scene_data["clip_2"]
        data_hash = calculate_data_hash([clip1, clip2])

        return (data_hash,)
    
vae_names = folder_paths.get_filename_list("vae")
class Texturaizer_GetVAEName(Texturaizer_GetJsonData):
    RETURN_TYPES = (vae_names, "STRING")
    RETURN_NAMES = ("vae_name", "data_hash")
    OUTPUT_TOOLTIPS = ("The VAE name.", 
                       "The hash value for debuging purposes.")
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Gets the VAE name from the provided directory or from 'Texturaizer set dir' node if not specified"
    
    def read_json_data(self, directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)

        scene_data = data.get("scene_info", {})
        vae = scene_data["vae"]
        data_hash = calculate_data_hash(vae)

        return (vae, data_hash)

    @staticmethod
    def IS_CHANGED(directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)
        
        scene_data = data.get("scene_info", {})
        vae = scene_data["vae"]
        data_hash = calculate_data_hash(vae)

        return (data_hash,)
    

def get_images(data):
    scene_data = data.get("scene_info", {})
    keys = ['image_path_base', 'image_path_seg', 'image_path_depth', 
            'image_path_normal', 'image_path_edge', 
            'image_path_seg_obj', 'image_path_seg_mat']
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
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("base", "segment active", "depth", "normal", "edge", "segment obj", "segmet mat", "data_hash")
    OUTPUT_TOOLTIPS = ("Base image", 
                       "Active Segment image",
                       "Depth pass",
                       "Normal pass", 
                       "Edge pass", 
                       "Object segment image",
                       "Material segment image",
                       "The hash value for debuging purposes.")
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Gets the image layers from the provided directory or from 'Texturaizer set dir' node if not specified"
    
    def read_json_data(self, directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)

        base, segment, depth, normal, edge, seg_obj, seg_mat= get_images(data)
        data_hash = combo_image_hash(base, segment, depth, normal, edge, seg_obj, seg_mat)

        return (base, segment, depth, normal, edge, seg_obj, seg_mat, data_hash)

    @staticmethod
    def IS_CHANGED(directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)

        base, segment, depth, normal, edge, seg_obj, seg_mat = get_images(data)
        data_hash = combo_image_hash(base, segment, depth, normal, edge, seg_obj, seg_mat)

        return (data_hash,)
        
def get_ip(data):
    ip_adapter_data = data.get("ip_adapters", {})
    scene_data = data.get("scene_info", {})
    
    def process_ip(ip_key, embed_data):
        ip = ip_adapter_data.get(ip_key, '')
        if not embed_data:
            try: return get_image_from_path(ip)
            except: return blank_image
        else:
            try: return get_image_from_base64(ip)
            except: return blank_image

    embed_data = scene_data["embed_data"]
    
    ip1 = process_ip('image_path_ipadapter_1', embed_data)
    ip_weight_Encode_1 = ip_adapter_data.get("ip_weight_Encode_1", 0.0)

    ip2 = process_ip('image_path_ipadapter_2', embed_data)
    ip_weight_Encode_2 = ip_adapter_data.get("ip_weight_Encode_2", 0.0)

    ip_loader_preset = ip_adapter_data.get("ip_loader_preset", "STANDARD (medium strength)")
    use_ipadapter = ip_adapter_data.get("use_ipadapter", False)

    return (ip1, ip_weight_Encode_1, ip2, ip_weight_Encode_2, ip_loader_preset, use_ipadapter, ip_adapter_data)


IP_PRESETS = ['LIGHT - SD1.5 only (low strength)', 'STANDARD (medium strength)', 'VIT-G (medium strength)', 'PLUS (high strength)', 'PLUS FACE (portraits)', 'FULL FACE - SD1.5 only (portraits stronger)']
WEIGHT_TYPES = ["linear", "ease in", "ease out", 'ease in-out', 'reverse in-out', 'weak input', 'weak output', 'weak middle', 'strong middle', 'style transfer', 'composition', 'strong style transfer', 'style and composition', 'style transfer precise', 'composition precise']
SCALING = ['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty']

class Texturaizer_GetIPAdapterData(Texturaizer_GetJsonData):
    RETURN_TYPES = ("IMAGE", "FLOAT", "IMAGE", "FLOAT", IP_PRESETS, "BOOLEAN", "DICTIONARY", "STRING")
    RETURN_NAMES = ("ip image 1", "ip1 weight", "ip image 2", "ip2 weight", "preset", "use IP", "IP data", "data_hash")
    OUTPUT_TOOLTIPS = ("IP image 1",
                       "weight of first ip adapter image",
                       "IP image 2",
                       "weight of second ip adapter image",
                       "IP preset type",
                       "enable/dissable IP Adapter",
                       "IP Adapter data",
                       "The hash value for debuging purposes.")
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Gets the IPAdapter data from the provided directory or from 'Texturaizer set dir' node if not specified"
    
    def read_json_data(self, directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)

        ip1, ip_weight_Encode_1, ip2, ip_weight_Encode_2, ip_loader_preset, use_ipadapter, ip_adapter_data = get_ip(data)
        image_hash = combo_image_hash(ip1, ip2)
        data_hash = calculate_data_hash([image_hash, ip_adapter_data])

        return (ip1, ip_weight_Encode_1, ip2, ip_weight_Encode_2, ip_loader_preset, use_ipadapter, ip_adapter_data, data_hash)

    @staticmethod
    def IS_CHANGED(directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)

        ip1, ip_weight_Encode_1, ip2, ip_weight_Encode_2, ip_loader_preset, use_ipadapter, ip_adapter_data = get_ip(data)
        image_hash = combo_image_hash(ip1, ip2)
        data_hash = calculate_data_hash([image_hash, ip_adapter_data])

        return (data_hash,)
    
class Texturaizer_IPAdapterEmbeds:
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
    DESCRIPTION = "Gets the IPAdapter Embeds data from the 'Get IP Adapter Data' node"

    def execute(self, ip_data):

        ip_weight_embed = ip_data["ip_weight_embed"]
        ip_weight_type = ip_data["ip_weight_type"]
        ip_start = ip_data["ip_start"]
        ip_end = ip_data["ip_end"]
        ip_embeds_scaling = ip_data["ip_embeds_scaling"]

        return (ip_weight_embed, ip_weight_type, ip_start, ip_end, ip_embeds_scaling)

class Texturaizer_GetLoraData(Texturaizer_GetJsonData):
    RETURN_TYPES = ("DICTIONARY", "STRING")
    RETURN_NAMES = ("LoRAs", "data_hash")
    OUTPUT_TOOLTIPS = ("LoRA data",
                       "The hash value for debuging purposes.")
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Gets the LoRA data from the provided directory or from 'Texturaizer set dir' node if not specified"

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
    RETURN_TYPES = ("INT", "FLOAT", comfy.samplers.KSampler.SAMPLERS, SCHEDULERS, "INT", "FLOAT", "INT", "INT", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("seed", "cfg", "sampler", "scheduler", "steps", "denoise", "adv steps", "adv steps start", "batch size", "use empty latent", "data_hash")
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Gets the sampler data from the provided directory or from 'Texturaizer set dir' node if not specified"

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
    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "data_hash")
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Gets the render settings data from the provided directory or from 'Texturaizer set dir' node if not specified"
    
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
    scene_data = data.get("scene_info", {})
    pos_g = scene_data['positive_prompt_g']
    pos_l = scene_data['positive_prompt_l']
    neg = scene_data['negative_prompt']

    return pos_g, pos_l, neg

class Texturaizer_GetPromptData(Texturaizer_GetJsonData):
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("pos_g", "pos_l", "neg", "data_hash")
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Gets the prompt data from the provided directory or from 'Texturaizer set dir' node if not specified"
    
    def read_json_data(self, directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)

        pos_g, pos_l, neg= get_prompt(data)
        data_hash = calculate_data_hash((pos_g, pos_l, neg))

        return (pos_g, pos_l, neg, data_hash)

    @staticmethod
    def IS_CHANGED(directory_optional="", data_optional={}):
        data = get_data(directory_optional, data_optional)

        pos_g, pos_l, neg= get_prompt(data)
        data_hash = calculate_data_hash((pos_g, pos_l, neg))

        return (data_hash,)
    
def get_style(data):
    scene_data = data.get("scene_info", {})
    use_style = scene_data['use_style']
    style = scene_data['style']
    style_pos = scene_data['style_pos']
    style_neg = scene_data['style_neg']

    return use_style, style, style_pos, style_neg

class Texturaizer_GetStyleData(Texturaizer_GetJsonData):
    RETURN_TYPES = ("BOOLEAN", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("use_style", "style", "style_pos", "style_neg", "data_hash")
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Gets the style data from the provided directory or from 'Texturaizer set dir' node if not specified"
    
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
    RETURN_TYPES = ("DICTIONARY", "STRING")
    RETURN_NAMES = ("data", "data_hash")
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Gets the segment data from the provided directory or from 'Texturaizer set dir' node if not specified"
    
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
    scene_info = data["scene_info"]
    embed_data = scene_info["embed_data"]
    
    preprocessed_images = []
    # Loop through each controlnet and add the preprocessed image
    for cn_key, cn in controlnets.items():
        if not embed_data:
            base_dir = data["texturaizer_save_dir"]
            images_dir = os.path.join(base_dir, "image layers")
            preprocess_image_path = cn.get('preprocess_image_path', "none")
            full_preprocess_image_path = os.path.join(images_dir, preprocess_image_path)
            # Get the preprocessed image and add it to the controlnet
            preprocessed_image = get_image_from_path(full_preprocess_image_path)
            cn['preprocessed_image'] = preprocessed_image  # Add the image to the controlnet data
            preprocessed_images.append(preprocessed_image)
        
    images_hash = combo_image_hash(*preprocessed_images)
    return controlnets, images_hash

class Texturaizer_GetCNData(Texturaizer_GetJsonData):
    RETURN_TYPES = ("DICTIONARY", "STRING")
    RETURN_NAMES = ("controlnets", "data_hash")
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Gets the ControlNet data from the provided directory or from 'Texturaizer set dir' node if not specified"
    
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
    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("use sdxl", "data_hash")
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Gets the use_sdxl property from the provided directory or from 'Texturaizer set dir' node if not specified"
    
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
    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("flux guidance", "data_hash")
    FUNCTION = "read_json_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Gets the flux_guidance property from the provided directory or from 'Texturaizer set dir' node if not specified"
    
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