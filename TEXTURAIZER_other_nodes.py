from .any_type import any

class Texturaizer_SwitchAny:
    """
    Node that switches between two inputs based on a boolean condition.
    Returns 'on_true' if boolean is True, otherwise returns 'on_false'.
    """

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
        """
        Executes the switch logic based on the boolean value.
        Returns 'on_true' if True, 'on_false' otherwise.
        """
        return (on_true,) if boolean else (on_false,)


NODE_CLASS_MAPPINGS = {
    "Texturaizer_SwitchAny": Texturaizer_SwitchAny,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Texturaizer_SwitchAny": "Switch Any (Texturaizer)",
}
