from omegaconf import OmegaConf
import os

def output_dir(output_dir: str, debug: bool):
    assert isinstance(debug, bool), type(debug)
    if debug:
        return "debug"
    else:
        return output_dir
    
def int_divide(val1: int, val2: int):
    if val1 % val2 != 0:
        raise ValueError(f"Values ({val1},{val2}) are not integer divisible.")
    return int(val1//val2)

def mkdirs(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir
    
OmegaConf.register_new_resolver("output_dir", output_dir)
OmegaConf.register_new_resolver("int_divide", int_divide)
OmegaConf.register_new_resolver("len", lambda x: len(x) if isinstance(x, list) else 1)
OmegaConf.register_new_resolver("mult", lambda x, y: x * y)
OmegaConf.register_new_resolver("abspath", lambda x: os.path.abspath(x))
OmegaConf.register_new_resolver("mkdirs", mkdirs)
OmegaConf.register_new_resolver("prepend_underscore", lambda x: "" if len(x)==0 else "_"+x)