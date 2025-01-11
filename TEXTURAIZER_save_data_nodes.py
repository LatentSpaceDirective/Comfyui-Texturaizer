from nodes import SaveImage

import random
import folder_paths

class Texturaizer_SendImage(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ), },
                "optional":
                    {"filename_prefix": ("STRING", {"default": "Texturaizer", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }
    

NODE_CLASS_MAPPINGS = {
    "Texturaizer_SendImage": Texturaizer_SendImage,  
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Texturaizer_SendImage": "Send Image (Texturaizer)",
}