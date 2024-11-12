import nodes
import folder_paths
import comfy.controlnet
import comfy.model_management
from kornia.filters import canny

from .TEXTURAIZER_load_data import get_image_from_base64

def create_canny(image, low_threshold, high_threshold):
    output = canny(
        image.to(comfy.model_management.get_torch_device()).movedim(-1, 1),
        low_threshold,
        high_threshold
    )
    img_out = output[1].to(comfy.model_management.intermediate_device()).repeat(1, 3, 1, 1).movedim(1, -1)
    return img_out

class Texturaizer_ApplyControlNets(nodes.ControlNetApplyAdvanced):
    def __init__(self):
        # Initialize an instance-level cache dictionary for models
        self.model_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cn_data": ("DICTIONARY",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            },
            "optional": {
                "vae": ("VAE", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnets"
    CATEGORY = "Texturaizer"

    def apply_controlnets(self, cn_data, positive, negative, vae=None):
        current_positive = positive
        current_negative = negative

        # Temporary cache for models used in this run
        models_used_in_run = {}

        for cn_key in cn_data:
            cn = cn_data[cn_key]

            # Check if ControlNet is enabled
            use_cn = cn.get('enabled', False)
            if not use_cn:
                continue  # Skip if not enabled

            # Extract parameters
            cn_type = cn.get('cn_type', "unknown")
            model_name = cn.get('model_name', "unknown_model")
            strength = cn.get('strength', 0.0)
            start_percent = cn.get('cn_start', 0.0)
            end_percent = cn.get('cn_end', 1.0)
            preprocessed_image = cn.get('preprocessed_image', None)
            if isinstance(preprocessed_image, str):  # Check if preprocessed_image is of type string (base64)
                preprocessed_image = get_image_from_base64(preprocessed_image)

            # Handle 'canny' type ControlNets
            if cn_type == 'canny':
                low_threshold = cn.get('low_threshold', 1) / 255
                high_threshold = cn.get('high_threshold', 255) / 255
                preprocessed_image = create_canny(preprocessed_image, low_threshold, high_threshold)

            # Resolve the control net model's full path
            try:
                controlnet_path = folder_paths.get_full_path_or_raise("controlnet", model_name)
            except Exception as e:
                print(f"Error resolving ControlNet model path for '{model_name}': {e}")
                continue

            # Load or retrieve the ControlNet model
            if model_name in self.model_cache:
                control_net = self.model_cache[model_name]
                print(f"Using cached ControlNet model '{model_name}'")
            else:
                # Load the ControlNet model
                try:
                    control_net_tuple = self.load_controlnet(model_name)
                    control_net = control_net_tuple[0]  # Unpack the tuple
                    print(f"Loaded ControlNet model '{model_name}'")
                except Exception as e:
                    print(f"Error loading ControlNet model '{model_name}': {e}")
                    continue

            # Add the model to models_used_in_run
            models_used_in_run[model_name] = control_net

            # Apply the ControlNet using the parent class's method
            current_positive, current_negative = super().apply_controlnet(
                current_positive,
                current_negative,
                control_net,
                preprocessed_image,
                strength,
                start_percent,
                end_percent,
                vae=vae,
            )

        # Update model_cache to contain only the models used in this run
        self.model_cache = models_used_in_run

        # Return the modified conditionings
        return current_positive, current_negative

    def load_controlnet(self, control_net_name):
        # Use the same method as ControlNetLoader to load the control net
        controlnet_path = folder_paths.get_full_path_or_raise("controlnet", control_net_name)
        controlnet = comfy.controlnet.load_controlnet(controlnet_path)
        return (controlnet,)
    

class Texturaizer_ExtractCNData:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cn_data": ("DICTIONARY",),
                "index": ("INT", )
            },
        }
    
    RETURN_TYPES = ("STRING",  folder_paths.get_filename_list("controlnet"), "IMAGE", "FLOAT", "FLOAT", "FLOAT",)
    RETURN_NAMES = ("type", "model", "preprocesed image", "strength", "start", "end", "MODEL")
    FUNCTION = "read_cn_data"
    CATEGORY = "Texturaizer"
    DESCRIPTION = "Extracts the ControlNet data from the provided dictionary"
    
    def read_cn_data(self, cn_data, index):
        # Ensure the index is within the valid range
        if index < 0 or index >= len(cn_data):
            print(f"Index {index} is out of range. Returning default values.")
            model = folder_paths.get_filename_list("controlnet")[0]
            return ("unknown", model, "none", 0.0, 0.0, 1.0)

        cn_key = list(cn_data.keys())[index]
        cn = cn_data[cn_key]

        # Assign values without trailing commas to avoid creating single-element tuples
        use_cn = cn.get('enabled', False)
        cn_type = cn.get('cn_type', "unknown")
        model = cn.get('model_name', "unknown_model")
        strength = cn.get('strength', 0.0)
        start = cn.get('cn_start', 0.0)
        end = cn.get('cn_end', 1.0)
        preprocessed_image = cn.get('preprocessed_image', None)
        if isinstance(preprocessed_image, str):  # Check if preprocessed_image is of type string (base64)
            preprocessed_image = get_image_from_base64(preprocessed_image)

        if cn_type == 'canny':
            low_threshold = cn.get('low_threshold',1)/255
            high_threshold = cn.get('high_threshold',255)/255
            preprocessed_image = create_canny(preprocessed_image, low_threshold, high_threshold)
        
        # Logic to adjust strength if ControlNet is not enabled
        if not use_cn:
            strength = 0.0

        # Return the desired values
        return (cn_type, model, preprocessed_image, strength, start, end)


NODE_CLASS_MAPPINGS = {
    "Texturaizer_ExtractCNData": Texturaizer_ExtractCNData,
    "Texturaizer_ApplyControlNets": Texturaizer_ApplyControlNets,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Texturaizer_ExtractCNData": "Set ControlNet Data (Texturaizer)",
    "Texturaizer_ApplyControlNets": "Apply ControlNets (Texturaizer)",
}
