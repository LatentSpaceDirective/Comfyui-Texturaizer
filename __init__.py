"""
Texturaizer - AI-Driven Design Integration.
These are companion nodes for Texturaizer, a Blender plugin that connects complex 3D data to ComfyUI.
For more details, visit:
- Website: www.texturaizer.com
- Repository: https://github.com/LatentSpaceDirective/Comfyui-Texturaizer.git

Developed by Luke Kratsios and the LatentSpaceDirective.
More about the creator: www.lukekratsios.com
"""

__version__ = "0.0.0"

import importlib

# Define node modules used by Texturaizer
node_list = [
    "TEXTURAIZER_combine_conditionings_node",
    "TEXTURAIZER_sampler_node",
    "TEXTURAIZER_lora_node",
    "TEXTURAIZER_load_data",
    "TEXTURAIZER_cached_models",
    "TEXTURAIZER_controlnet_node",
    "TEXTURAIZER_other_nodes"
]

# Initialize mappings for node classes and display names
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Import each module and update the mappings
for module_name in node_list:
    imported_module = importlib.import_module(f".{module_name}", package=__name__)
    NODE_CLASS_MAPPINGS.update(imported_module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(imported_module.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
