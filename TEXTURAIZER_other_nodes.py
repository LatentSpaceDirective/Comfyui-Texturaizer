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


class Texturaizer_SwitchLazy:
    """
    Node that switches between three inputs based on an index.
    Returns the selected input and blocks others.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 1, "min": 1, "max": 3, "tooltip": "Select which input to output (1-3)."}),
            },
            "optional": {
                "input1": (any, {"lazy": True}),
                "input2": (any, {"lazy": True}),
                "input3": (any, {"lazy": True}),
            }
        }

    CATEGORY = "Texturaizer"
    RETURN_TYPES = (any, 'INT')
    RETURN_NAMES = ("selected", 'index')
    FUNCTION = "execute"

    def check_lazy_status(self, *args, **kwargs):
        """
        Determines which input needs to be evaluated based on the index.
        """
        selected_index = int(kwargs['index'])
        selected_input = f"input{selected_index}"
        return [selected_input]

    @staticmethod
    def execute(*args, **kwargs):
        selected_index = int(kwargs['index'])
        selected_input = f"input{selected_index}"

        if selected_input in kwargs and kwargs[selected_input] is not None:
            return kwargs[selected_input], selected_index
        else:
            print(f"Execution blocked for unselected input: {selected_input}")
            return None, selected_index

class Texturaizer_Placeholder:
    """
    A placeholder node that optionally prints a message and returns five any-type outputs as None.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "message": ("STRING", {"default": "", "multiline": False, "placeholder": "Enter a message to print..."})
            }
        }

    CATEGORY = "Texturaizer"
    RETURN_TYPES = (any, any, any, any, any)
    RETURN_NAMES = ("output1", "output2", "output3", "output4", "output5")
    FUNCTION = "execute"

    @staticmethod
    def execute(message):
        if message:
            print(message)

        # Return five None values as the outputs
        return None, None, None, None, None


NODE_CLASS_MAPPINGS = {
    "Texturaizer_SwitchAny": Texturaizer_SwitchAny,
    "Texturaizer_SwitchLazy": Texturaizer_SwitchLazy,
    "Texturaizer_Placeholder": Texturaizer_Placeholder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Texturaizer_SwitchAny": "Switch Any (Texturaizer)",
    "Texturaizer_SwitchLazy": "Switch Lazy (Texturaizer)",
    "Texturaizer_Placeholder": "Placeholder (Texturaizer)",
}
