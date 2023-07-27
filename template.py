# This file is use to create the folder structure automatically

import os
import sys
from pathlib import Path
import logging

#* Create a logger
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(message)s:')

project_name = "cnnClassifier"

list_of_folders = [
    ".github/workflows/.gitkeep", # .gitkeep is to keep the folder in github even if it is empty when w'll be using this folder we will remove this file
    f"src/{project_name}/__init__.py", # this is going to be the local package
    f"src/{project_name}/components/__init__.py", # this is going to be the local package folder
    f"src/{project_name}/utils/__init__.py", # All the utility functions will be here
    f"src/{project_name}/config/__init__.py", # All the configuration files will be here 
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py", # All the pipeline files will be here
    f"src/{project_name}/entity/__init__.py", # All the entity files will be here
    f"src/{project_name}/constants/__init__.py", # All the constants files will be here
    f"config/config.yaml", # github action file 
    "dvc.yaml", # integrating MLops to DVC (Data Version Control)
    "params.yaml",
    "requirements.txt", # all the dependencies will be here
    "setup.py", # this is the file which will be used to install the package
    "research/trails.ipnyb", # this is the file which will be used to do some research and other stuff
]

for filepath in list_of_folders: 
    filepath = Path(filepath) # converting the string to path object
    filedir, filename = os.path.split(filepath) # splitting the path into directory and filename

    if filedir != "": 
        os.makedirs(filedir, exist_ok=True) # creating the directory if it does not exist
        logging.info(f"Created the directory {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0): 
        with open(filepath, "w") as f:
            logging.info(f"Created the file: {filepath}")

    else: 
        logging.info(f"File already exists: {filepath}")