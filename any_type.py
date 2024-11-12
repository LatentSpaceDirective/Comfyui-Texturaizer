class AnyType(str):
    """
    A special class that evaluates as equal in any comparison.
    Useful for cases where a wildcard-type behavior is needed.

    Credit for the concept goes to: pythongosssss
    """

    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False

# Instantiate a wildcard object
any = AnyType("*")
