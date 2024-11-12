"""
Texturaizer - Harnessing AI for Design Intent.
These are companion nodes for Texturaizer, A Blender plugin to connect complex 3D data to ComfyUI. 
www.texturaizer.com
https://github.com/LatentSpaceDirective/Comfyui-Texturaizer.git

These Nodes are brought to you by Luke Kratsios and the LatentSpaceDirective. www.lukekratsios.com
"""

__version__ = "0.0.0"

import importlib

node_list = [
    "TEXTURAIZER_combine_conditionings_node",
    "TEXTURAIZER_sampler_node",
    "TEXTURAIZER_lora_node",
    "TEXTURAIZER_load_data",
    "TEXTURAIZER_cached_models",
    "TEXTURAIZER_controlnet_node",
    "TEXTURAIZER_other_nodes"
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_name in node_list:
    # Use a leading dot to indicate a relative import
    imported_module = importlib.import_module(f".{module_name}", package=__name__)
    # Update the mappings using the imported module's attributes
    NODE_CLASS_MAPPINGS.update(imported_module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(imported_module.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
