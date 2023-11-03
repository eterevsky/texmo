from .layers.input import Input

class Model3(object):
    def __init__(self, spec: str):
        spec_parts = spec.split("|")
        
        if len(spec_parts) == 1:
            input_spec = ""
            layers_spec = spec_parts[0]
        elif len(spec_parts) == 2:
            input_spec, layers_spec = spec_parts
        else:
            raise AssertionError("Model spec can't contain more than one |")
        
        self.input = Input.from_spec(input_spec)