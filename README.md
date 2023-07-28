# Chicken-Disease-Classification-project

This project is an **end-to-end Deep learning Classifier** project which is the part of a video instructed by Krish naik team https://youtu.be/p1bfK8ZJgkE

## Here is the Step by Step guide to recreate this project 

### 1. Setup part

#### 1.1 Start by creating a new github repository

Create a new github repo of name of your choice like I have chosen "Chicken-Disease-Classification-project" include the readme and .gitignore files and choose `python` as language in gitignore

Start project locally by cloning this repository on your local computer 

```
git clone <link copied from http in code section>
```

#### 1.2 Create template.py to Automate the process of Creating project structure

In side you we will be writing script by which we can automate the project of creating project structure( required folders and file) Automatically you can create all this new on your own is you want but for now you can check out template.py file
```
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
    "templates/index.html"
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

```

#### 1.3 Construct setup.py file to use project as local package 

this setup.py file will help us to host this project as package later on site like pypi or you can just use it locally to organize all of your work

Inside setup.py I have given all the neccessary code by which it can hold all the information it needs for any one to understand the code properly (well it is just display this readme.md file if this package is hosted)

```

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

__version__ = "0.0.0"

REPO_NAME = "Chicken-Disease-Classification-project"
AUTHOR_USER_NAME = "Rahul-lalwani-learner"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "itsrahullalwani@gmail.com"

setuptools.setup(
    name = SRC_REPO,
    version = __version__,
    author = AUTHOR_USER_NAME,
    author_email = AUTHOR_EMAIL,
    description="A small Pythn package for CNN Classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        'Bug Tracker': f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
)

```
