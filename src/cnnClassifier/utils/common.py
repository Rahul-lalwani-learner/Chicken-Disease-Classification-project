import os
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a yaml file and returns a ConfigBox object
    
    Args: 
        path_to_yaml (Path): Path to yaml file
    Returns:
        ConfigBox: ConfigBox object
    Raises:
        BoxValueError: If path_to_yaml does not exist
        e: empty file
    """
    
    try: 
        with open(path_to_yaml, "r") as yaml_file:
            yaml_dict = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(yaml_dict)
    except BoxValueError: 
        logger.info("yaml file does not exist")
        raise ValueError("yaml file does not exist")
    except Exception as e:
        logger.info(e)
        raise e
    

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True): 
    """
    Create list of directories

    Args:
        path_to_directories (list): list of paths to directories
        verbose (bool, optional): Defaults to True.
    """

    for path in path_to_directories: 
        os.makedirs(path, exist_ok=True)
        if verbose: 
            logger.info(f"directory created at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict): 
    """
    save json file

    Args:
        path (Path): path to json file
        data (dict): dictionary to save
    """
    
    with open(path, "w") as json_file: 
        json.dump(data, json_file, indent=4)

    logger.info(f"json file saved at: {path}")

@ensure_annotations
def load_json(path:Path) -> ConfigBox: 
    """
    load json file

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: ConfigBox object
    """
    
    with open(path, "r") as json_file: 
        data = json.load(json_file)

    logger.info(f"json file loaded from: {path}")
    return ConfigBox(data)


@ensure_annotations
def save_bin(data:Any, path: Path): 
    """
    save binary file

    Args:
        data (Any): data to save
        path (Path): path to save
    """
    
    joblib.dump(value = data,filename =  path)

    logger.info(f"binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    load binary file

    Args:
        path (Path): path to load

    Returns:
        Any: data
    """
    
    data = joblib.load(filename = path)

    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path:Path) -> str: 
    """
    get size of file in KB

    Args:
        path (Path): path to file

    Returns:
        str: size of file
    """
    
    size_in_kb = round(os.path.getsize(path)/1024)
    return size_in_kb

def decodeImage(imgstring, filename): 
    imgdata = base64.b64decode(imgstring)
    with open(filename, 'wb') as f:
        f.write(imgdata)
        f.close()

def encodeImageIntoBase64(image_path): 
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string