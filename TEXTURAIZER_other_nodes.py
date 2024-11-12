from .any_type import any

class Texturaizer_SwitchAny:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "on_true": (any, {}),
                "on_false": (any, {}),
                "boolean": ("BOOLEAN", {"default": True}),
            }
        }

    CATEGORY = "Texturaizer"
    RETURN_TYPES = (any,)

    FUNCTION = "execute"

    def execute(self, on_true, on_false, boolean=True):
        if boolean:
            return (on_true,)
        else:
            return (on_false,)

NODE_CLASS_MAPPINGS = {
    "Texturaizer_SwitchAny": Texturaizer_SwitchAny,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Texturaizer_SwitchAny": "Switch Any (Texturaizer)",
}
