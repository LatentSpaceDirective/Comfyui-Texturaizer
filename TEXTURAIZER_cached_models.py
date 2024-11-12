import nodes
import folder_paths

class Texturaizer_CachedCheckpointLoader(nodes.CheckpointLoaderSimple):
    def __init__(self):
        # Initialize instance variables for each instance
        self.last_loaded_model = None
        self.cached_model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "doit"

    CATEGORY = "Texturaizer"

    def doit(self, ckpt_name):
        # Use instance variables instead of class variables
        if self.last_loaded_model != ckpt_name:
            res = self.load_checkpoint(ckpt_name)
            self.last_loaded_model = ckpt_name
            self.cached_model = res

        return self.cached_model

    @staticmethod
    def IS_CHANGED(ckpt_name):
        return (ckpt_name,)

    
class Texturaizer_CachedCNLoader(nodes.ControlNetLoader):
    def __init__(self):
        # Initialize instance variables for each instance
        self.last_loaded_model = None
        self.cached_model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net_name": (folder_paths.get_filename_list("controlnet"), ),
            }
        }

    RETURN_TYPES = ("CONTROL_NET", )
    FUNCTION = "doit"
    CATEGORY = "Texturaizer"

    def doit(self, control_net_name):
        # Use instance variables instead of class variables
        if self.last_loaded_model != control_net_name:
            print("NEW MODEL: ", control_net_name)
            res = self.load_controlnet(control_net_name)
            self.last_loaded_model = control_net_name
            self.cached_model = res
        else:
            print("SAME AS LAST LOADED MODEL: ", control_net_name)

        return self.cached_model

    @staticmethod
    def IS_CHANGED(control_net_name):
        return (control_net_name,)


NODE_CLASS_MAPPINGS = {
    "Texturaizer_CachedCheckpointLoader": Texturaizer_CachedCheckpointLoader,
    "Texturaizer_CachedCNLoader": Texturaizer_CachedCNLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Texturaizer_CachedCheckpointLoader": "Cached Checkpoint (Texturaizer)",
    "Texturaizer_CachedCNLoader": "Cached ControlNet (Texturaizer)",
}