import folder_paths
from nodes import LoraLoader

import os

def get_lora_by_filename(file_path, lora_paths=None):
    """
    Returns the path of a LoRA file matching the provided filename. Attempts
    exact path matches, filename-only matches, and extension-less matches.
    If no exact match is found, performs a fuzzy search on available LoRAs.
    """
    lora_paths = lora_paths if lora_paths is not None else folder_paths.get_filename_list('loras')

    if file_path in lora_paths:
        return file_path

    lora_paths_no_ext = [os.path.splitext(x)[0] for x in lora_paths]

    # Check for exact path without extension
    if file_path in lora_paths_no_ext:
        found = lora_paths[lora_paths_no_ext.index(file_path)]
        return found

    # Remove extension from file_path
    file_path_no_ext = os.path.splitext(file_path)[0]
    if file_path_no_ext in lora_paths_no_ext:
        found = lora_paths[lora_paths_no_ext.index(file_path_no_ext)]
        return found

    # Check if we passed just the filename without path
    lora_filenames_only = [os.path.basename(x) for x in lora_paths]
    if file_path in lora_filenames_only:
        found = lora_paths[lora_filenames_only.index(file_path)]
        return found

    # Force input to be filename without path
    file_path_filename = os.path.basename(file_path)
    if file_path_filename in lora_filenames_only:
        found = lora_paths[lora_filenames_only.index(file_path_filename)]
        return found

    # Check filenames without extension
    lora_filenames_no_ext = [os.path.splitext(os.path.basename(x))[0] for x in lora_paths]
    file_path_filename_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    if file_path_filename_no_ext in lora_filenames_no_ext:
        found = lora_paths[lora_filenames_no_ext.index(file_path_filename_no_ext)]
        return found

    # Fuzzy matching
    for index, lora_path in enumerate(lora_paths):
        if file_path in lora_path:
            found = lora_paths[index]
            return found

    return None

class Texturaizer_PowerLoraLoader:
    """
    Node for loading and applying multiple LoRAs to a model and clip.
    Loops through each LoRA configuration and applies valid, enabled LoRAs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
            },
            "optional": {
                "loras": ("DICTIONARY", ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_loras"
    CATEGORY = "Texturaizer"

    def load_loras(self, model, clip, loras=None, **kwargs):
        """
        Applies each enabled LoRA to the model and clip with specified strengths.
        Skips LoRAs if disabled, not found, or with zero strength.
        
        loras should be a dictionary where each key is a unique identifier,
        and each value is a dictionary with 'enabled', 'lora', and 'strength' keys.
        """
        if loras is None:
            loras = {}

        for lora_name, lora_config in loras.items():
            if not isinstance(lora_config, dict):
                print(f"[WARNING] LoRA '{lora_name}' configuration is not a dictionary. Skipping.")
                continue

            enabled = lora_config.get('enabled', True)
            lora_file = lora_config.get('lora')
            strength_model = lora_config.get('strength', 1.0)
            strength_clip = lora_config.get('strengthTwo', strength_model)

            if enabled and lora_file and (strength_model != 0 or strength_clip != 0):
                lora_path = get_lora_by_filename(lora_file)
                if lora_path is not None:
                    model, clip = LoraLoader().load_lora(model, clip, lora_path, strength_model, strength_clip)
                else:
                    print(f"[WARNING] LoRA file '{lora_file}' not found.")
            else:
                print(f"[INFO] Skipping LoRA '{lora_name}' due to 'enabled' being False, zero strength, or missing file.")

        return (model, clip)


NODE_CLASS_MAPPINGS = {
    "Texturaizer_PowerLoraLoader": Texturaizer_PowerLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Texturaizer_PowerLoraLoader": "Lora Loader (Texturaizer)",
}
