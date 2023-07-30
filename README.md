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
#### 1.4 Write all the required modules in requirements.txt

All the neccessary modules name are written here for combine installation of the modules. 

you will see why we have written name of these modules in this project as you go along with it

```
tensorflow
pandas 
dvc
notebook
numpy
matplotlib
seaborn
python-box==6.0.2
pyYAML
tqdm
ensure==1.0.2
joblib
types-pyYAML
scipy
Flask
Flask-Cors
ipykernel
-e . 

```
**-e . it also present here which is not a module it will run setup.py file automatically to build package of you run requirements.txt file and setup.py will extract all the folder which has `__init__.py`**

#### 1.5 Create new environment and install all the modules

Here i will be using python version *3.8* for project, you can create a new virtual environment by writting (chicken is name of Environment)

```
conda create -n chicken python=3.8 -y
```
**Make sure you have miniconda or anaconda installed on your computer**


After Creating this environment simply activate this

```
conda activate chicken
```

Now, you can simply install all the requirements

```
pip install -r requirements.txt
```

automatically install all the requirements fot his project

🔑**Note** : Don't Forget to commit you source code time to time for proper management of code

Now your step is complete go ahead to create logging to create logs for each process you perform

Here i have not created Expection.py module for expection handling since i'll be using box-execption module using this you can also handle exceptions But if you you can write

### 2. Logger and utils file

#### 2.1 Logger - for logging our process

Here we have created Logger inside `__init__.py` of src.cnnClassifier so that it will be easily acessible to file by using "from cnnClassifier import logger"

In logging i have set stream out since i also want to print output of logging to the terminal (console)

You can checkout this src.cnnClassifier.__init__.py file

```
import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"

log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format=logging_str,

    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout) # print to console
    ]

)

logger = logging.getLogger("cnnClassifierLogger")

```

We can check that logging is working properly in main.py file where i have print a simple Custom logging to console 

This log will create a running_logs.log file in logs folder where we can check all of our logs and if something when wrong we can easily go through it. 

#### 2.2 Creating utiliy functions

I have created utility functions under common.py in utils folder *This are the functions that we are going to use frequently in this project* 

Several methods inside utils are: 

* read_yaml -> helps us read yaml file while CI/CD pipeline and github actions
* Create_directory -> simply Create the directory at given path
* save_json -> There results of the prediction w'll be saving in Json formate
* Load_json -> To access those predictions and JSON file
* Save_bin -> save binary files
* load_bin -> Load binary files
* get_size -> to get the size of particular file
* decodeImage -> for decoding stringImage format to int
* encodeImageIntoBase64 - encoding the binary image to base64

```
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

```

Testing of this function and questions like why configbox is used instead of dictionary and what ensure_annotation decorator does are resolved in [research/trail.ipynb](https://github.com/Rahul-lalwani-learner/Chicken-Disease-Classification-project/blob/main/research/trails.ipynb) File. Do check it out

## 3. Project workFlows

1. Update config.yaml 
> All filepaths related things of dataingestion and other pipelines are written here 
2. Update secrets.yaml [Optional]
> If you have some ceredentials or some secret information you can write it in secrets.yaml
3. Update params.yaml
> During model Configuration i'll we updating this params
4. Update the Entity
> It is the return type of the function (if you don't have any inbuilt return type you can create your custom return type)
5. Update the configuration manger in src config
> This will help us properly connect yaml file and read content from them 
6. Update the components
> Writing different components like data ingestion data prediction
7. Update the pipeline
> Creating prediction pipelines and training pipelines
8. Update the main.py
> After writing code modulary udpate main.py for CI/CD implementation 
9. Update the dvc.yaml
> for pipeline tracking

### 3.1 Data Ingestion 

**Firstly we will perform this on *01_data_ingestion.ipynb* in research folder so that we all configer that everything is running properly then we will upgrade it to modular programing fashion**

Go through workflow to properly do it

#### 3.1.1 udpate config.yaml

for data ingestion we will store all the links and path of data and folders inside this data is stored inside this folder

you can look as the code

```
artifacts_root: artifacts


data_ingestion:
  root_dir: artifacts/data_ingestion
  source_URL: https://github.com/entbappy/Branching-tutorial/raw/master/Chicken-fecal-images.zip
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion

```

This yaml file will be read as ConfigBox format mean somewhat looks like dictionary format for more convenience 
*means we can use this as variables inside different files some what like global variables for paths*

#### 3.1.2 update Entity (Create the new Datatype for Storing filepaths)

we can skip updating params.yaml and secrets.yaml for now since this we are working in testing phase

Here New entity we required is DataIngestionconfig will have attributes same as config.yaml file

```
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig: 
    root_dir : Path
    source_URL : str
    local_data_file : Path
    unzip_dir: Path

```
#### 3.1.3 Update Configmanger in src.config and updating constants
This is going to extract all the data from yaml file and return us as a ConfigBox format which we can use in our next step for components.dataingestion before this we also have to update the constants.__init__.py inside src which will hold the path of Config.yaml and params.yaml

```
from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
```

Now we can import constants form cnnClassifier and use them  
```
from cnnClassifier.constants import * 
from cnnClassifier.utils.common import read_yaml, create_directories

class ConfigurationManager: 
    def __init__(
            self, 
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH
    ): 
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig: 
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_URL = config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )

        return data_ingestion_config

```

#### 3.1.4 Update the components (Data Ingsetion)
While modular programing we will be Creating a file inside Componetns for data ingestion Load all the data and extraction it to artifacts folder 

```
import os
import urllib.request as request
import zipfile
from cnnClassifier.utils.common import get_size
from cnnClassifier import logger

class DataIngestion: 
    def __init__(self, config: DataIngestionConfig): 
        self.config = config

    def download_file(self): 
        if not os.path.exists(self.config.local_data_file): 
            filename, headers = request.urlretrieve(
                url = self.config.source_URL, 
                filename = self.config.local_data_file
            )
            logger.info(f"File downloaded at: {filename} with following headers: {headers}")
        else: 
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")
        
    
    def extract_zip_file(self): 
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

```
Here we have downloaded data from url using `urllib.request.urlretrieve(url, filename)` and after that created a function that will extraction it to local storages as <local_data_file> name

Now you can run this code under a try except statement to check does everything is running properly

```
try: 
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion.download_file()
    data_ingestion.extract_zip_file()
except Exception as e: 
    raise e
```

In theory this should download our data and extract it to local storage you can check it in your file explorer.

#### 3.1.5 converting to Modular programing approach

You just have to follow all the steps one by one that are mentioned above 
let me write them for you

1. update config.yaml (that if already updated)
2. update params.yaml and secrets.yaml (for now you can skip them)
3. update entity 
Inside entity create a file named config_entity and paste the code where you have created the DataIngestionConfig entity
4. Update configurationManger in src.config.configuration.py 
paste the code of `ConfigurationManager` class
5. update the components
Inside components create the data_ingestion.py

copy the dataIngestion class code and paste it here

Here is all your code converted to Modular approach

#### 3.1.6 Update Pipeline
Create `stage_01_data_ingestion.py` inside pipeline folder Here we will be importing DataIngestion from Components and Create class to run them 

```
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier import logger

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline: 
    def __init__(self): 
        pass

    def main(self): 
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == "__main__":
    try: 
        logger.info(">>> Stage {} started <<<".format(STAGE_NAME))
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(">>> Stage {} completed <<<\n\n X===========X".format(STAGE_NAME))
    
    except Exception as e:
        logger.exception(e)
        raise e
```

#### 3.1.7 Update the main.py 

until we are not using DVC w'll be using main.py to run pipelines and components

Here is the simple Code to run DataIngestionTrainingPipeline

```
from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"
try: 
    logger.info(">>> Stage {} started <<<".format(STAGE_NAME))
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(">>> Stage {} completed <<<\n\n X===========X".format(STAGE_NAME))

except Exception as e:
    logger.exception(e)
    raise e
```

✅ **Data Ingestion Completed**

### 4. Prepare Base model 

Firstly we will perform our experiments in [research/02_prepare_base_model.ipynb](https://github.com/Rahul-lalwani-learner/Chicken-Disease-Classification-project/blob/main/research/02_prepare_base_model.ipynb)

Firstly for model prepration we have update the config.yaml file
#### 4.1 update config.yaml
it include all the neccessary file paths for base models and updated model to save

```
prepare_base_model: 
  root_dir: artifacts/prepare_base_model
  base_model_path: artifacts/prepare_base_model/base_model.h5
  updated_base_model_path: artifacts/prepare_base_model/base_model_udpated.h5
```

🔑**Note**: The base model here is VGG16 from tf.keras.application and updated model is the model with changed top and output layer

#### 4.2 Update params.yaml
In params.yaml i will give all the parameters related to model this are going to work as global parameter that i can use in any file and update Globally

```
AUGMENTATION: True
IMAGE_SIZE: [224,224,3] # as per the VGG 16 model 
BATCH_SIZE: 16
INCLUDE_TOP: False
EPOCHS: 1
CLASSES: 2
WEIGHTS: imagenet
LEARNING_RATE: 0.01
```

#### 4.3 Update Entity
I have created a new entity inside config_entity.py for prepare base model stage Where it will be mix of params.yaml and config.yaml 
```
@dataclass(frozen=True)
class PrepareBaseModelConfig: 
    root_dir : Path
    base_model_path : Path
    updated_base_model_path : Path
    params_image_size : list
    params_learning_rate : float
    params_include_top : bool
    params_weights : str
    params_classes: int
```

#### 4.4 Update configuration Manger
Here in configuration manager i have to Create a methods that will help read the yaml files (grab data from them) and return to Components as `ConfigBox`

```
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig: 
        config  = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
        )

        return prepare_base_model_config
```

*As you have seen earlier that constructor of this Configuration Manager read the data form yaml files and create root directories*

#### 4.5 Create new Component prepare_base_model.py

Here we will create a new Class PrepareBaseModel which will have 3 methods

1. `get_base_model` - This method is going to load VGG16 model from keras.application on imagenet weights and also saves it in H5 format
2. `_prepare_full_model` - This methods is going to help in Updating our model by adding extra flatten and Dense layer to the end of the VGG16 model
3. `Update_base_model` - This function will finally update the basemodel with extra layer and save it in H5 format inside artifacts folder

```
mport os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from cnnClassifier import logger

class PrepareBaseModel: 
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    
    def get_base_model(self): 
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size, 
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        logger.info('Base model loaded successfully')
        self.save_model(path=self.config.base_model_path, model=self.model)
    

    @staticmethod
    def  _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate): 
        if freeze_all: 
            for layer in model.layers: 
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0): 
            for layer in model.layers[:-freeze_till]: 
                model.trainable = False

        flatten_layer = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(units=classes, activation='softmax')(flatten_layer)

        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)

        full_model.compile(
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss = tf.keras.losses.CategoricalCrossentropy(), 
            metrics = ['accuracy']
        )

        logger.info("Full model Compiled successfully")
        full_model.summary()

        return full_model
    
    def update_base_model(self): 
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True, 
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
        logger.info("Updated base model saved successfully")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
```

#### 4.6 Create new pipeline to prepare base model

Similar to Data ingestion pipeline this will call call the methods from components and Configuration manger to Create new Class `PrepareBaseModelTrainingPipeline`

```
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger

STAGE_NAME = 'prepare base model'

class PrepareBaseModelTrainingPipeline:
    def __init__(self): 
        pass

    def main(self): 
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()
```

#### 4.7 update main.py 
Now finally Append new lines of code to main.py to Check whether everything is working properly or not after this a new folder will be created in artifacts and both base model and updated model are also going to be present their.

```
STAGE_NAME = "Prepare Base Model Stage"
try: 
    logger.info("************************")
    logger.info(f">>>>>>>>> Running stage: {STAGE_NAME} started <<<<<<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>> stage: {STAGE_NAME} completed <<<<<<<<<\n\nX================X")
except Exception as e:
    logger.error(f"Error while running stage: {STAGE_NAME} - Error message: {e}")
    raise e
```

✅ **Preparing Base Model Completed**

### 5. Prepare Callbacks 

Procedure for this also going to be same Here we will create till components since Callbacks standalone can't perform in pipeline so they will be helping us while training the model. 

Same workflow
#### 5.1 Update Config.yaml
Insert new paths and location for Checkpoint_log_dir and tensorboard_log_dir

```
prepare_callbacks:
  root_dir: artifacts/prepare_callbacks
  tensorboard_root_log_dir: artifacts/prepare_callbacks/tensorboard_log_dir
  checkpoint_model_filepath: artifacts/prepare_callbacks/checkpoint_dir/model.h5
```

#### 5.2 Update Entity (Create new entity for PrepareCallbackconfig)

No need to update params since that will only be used will using model and training it

Create new entity which follow same format as Config.yaml > prepare_callbacks

```
@dataclass(frozen=True)
class PrepareCallbacksConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path
```

#### 5.3 Update Configuration manager
Now you again have to update the Configuration manger that will help us read the data from Config.yaml and return that as the ConfigBox format that we can again use it in Components

```
def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([
            Path(model_ckpt_dir),
            Path(config.tensorboard_root_log_dir)
        ])

        prepare_callback_config = PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath)
        )

        return prepare_callback_config
```

#### 5.4 Create new Components
Here we have to create new Component as prepareCallback where we will Create checkpoints and tensorboard that will be returning as the list that we can use will fitting the model 

```
class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config


    
    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
    

    @property
    def _create_ckpt_callbacks(self):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.config.checkpoint_model_filepath),
            save_best_only=True
        )


    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks
        ]
```

✅**Till here Prepare Model callbacks is Completed**
You can check all this in [research/03_prepare_callbacks.ipynb](https://github.com/Rahul-lalwani-learner/Chicken-Disease-Classification-project/blob/main/research/03_prepare_callbacks.ipynb) To check whether how this all is going to create new directory in artifacts folder

### 6. Training the model 
Wow!! Excited we have reached to the stage of Model training Now you can see everything you have done earlier in working stage

Here also we will follow the same Workfloww mean first we have to Update the config.yaml and also we don't have to worry about params.yaml and Secret.yaml since they both are already updated

#### 6.1 Update the Config.yaml
This time we don't have to do much in it just training path and updated model path

```
training: 
  root_dir: artifacts/training
  trained_model_path: artifacts/training/model.h5
```

#### 6.2 Update the Entity/config_entity.py
Here again w'll create new entity that will hold the content from both config.yaml and params.yaml according to requirements of training configuration manager

```
@dataclass(frozen=True)
class TrainingConfig: 
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
```

#### 6.3 Update Configuration Manager in Configuration.py

Here i haved added a new components that will Extract information from config.yaml and params.yaml and return all this information in TrainingConfig format

```
    def get_training_config(self) -> TrainingConfig: 
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "Chicken-fecal-images")
        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir = Path(training.root_dir), 
            trained_model_path = Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data = Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
        )

        return training_config
```

#### 6.4 Create new Components
Create new trainig Components that will have the functions to load updated base model, training the model and saving the model that will be used in pipeline

**train_valid_generator** is an interesting function this will perform and DataAugmentation and also do Data rescale while simulaneously Loading the Data from training directories in Artifacts

```
class Training: 
    def __init__(self, config: TrainingConfig): 
        self.config = config
    
    def get_base_model(self): 
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self): 
        
        datagenerator_kawrgs = dict(
            rescale = 1./255,
            validation_split = 0.20
        )

        dataflow_kwargs = dict(
            target_size = self.config.params_image_size[:-1],
            batch_size = self.config.params_batch_size,
            interpolation = "bilinear"
        )

        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kawrgs
        )

        self.valid_generator = valid_datagen.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False, 
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation: 
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2, 
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kawrgs
            )
        else: 
            train_datagen = valid_datagen

        self.train_generator = train_datagen.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model): 
        model.save(path)

    def train(self, callback_list: list): 
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.history = self.model.fit(
            self.train_generator, 
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.valid_generator,
            validation_steps = self.validation_steps,
            callbacks = callback_list
        )

        self.save_model(
            path=self.config.trained_model_path,
            model = self.model
        )
```

#### 6.5 Create new pipeline
Now i have created a new Pipeline that will properly Use training and Callbacks components and basically trains the model and saves it

```
class ModelTrainingPipeline: 
    def __init__(self): 
        pass

    def main(self): 
        config = ConfigurationManager()
        prepare_callbacks_config = config.get_prepare_callback_config()
        prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()

        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train(callback_list = callback_list)

if __name__ == "__main__": 
    try: 
        logger.info(f"***********************")
        logger.info(f">>>>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>>>>>> Stage {STAGE_NAME} completed <<<<<<<<<<<")
    except Exception as e: 
        logger.exception(f"Error in {STAGE_NAME} pipeline: {e}")
        raise e
```

#### 6.6 Update the main.py
Now you see all this in action by updating main.py and running TrainingPipeline

```
STAGE_NAME = "Training Stage"
try: 
    logger.info(f"***********************")
    logger.info(f">>>>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<<")
except Exception as e: 
    logger.exception(f"Error in {STAGE_NAME} pipeline: {e}")
    raise e
```

🔑**Note**: Here are some thing you have to remember 

1. You can do check logs at any time if you find any trouble in execution 
2. You can also check all of this in [research/04_training.ipynb](https://github.com/Rahul-lalwani-learner/Chicken-Disease-Classification-project/blob/main/research/04_training.ipynb) to see everything thing under one shed

✅ **Model Training Stage Completed** Finally the model training is done now w'll go ahead with model evaluation part

### 7. Model Evaluation 
Model Evaluation is also a very important part of any machine learning project Here we'll be able to see how our model is performing and does everything is working as expected or not

**Here Also w'll follow same Workflow** 

IN model Evaluation we don't have to update any yaml file we can start with updating the entity directly

#### 7.1 Update the entity/config_entity.py
similar to above create new entity that will satisfy the requirement of evaluation classes

```
@dataclass(frozen=True)
class EvaluationConfig: 
    path_of_model: Path
    training_data: Path
    all_params: dict
    params_image_size: list
    params_batch_size: int
```

#### 7.2 Update the Configuration manager
Updating the Configuration manager to load imformation from yaml files to entity

```
def get_validation_config(self) -> EvaluationConfig: 
        eval_config = EvaluationConfig(
            path_of_model=Path("artifacts/training/model.h5"), 
            training_data=Path("artifacts/data_ingestion/Chicken-fecal-images"), 
            all_params=self.params, 
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )

        return eval_config
```

#### 7.3 Create new Component (Evaluation)

In there component w'll write methods to Evalute our model and save the results of that model as the `JSON` file

🔑**Note**: This class also uses the ImageDataGenerator to rescale and load data from directories as above.

```
class Evaluation: 
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def _valid_generator(self): 
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split = 0.30
        )

        dataflow_kwargs = dict(
            target_size = self.config.params_image_size[:-1], 
            batch_size = self.config.params_batch_size, 
            interpolation = "bilinear"
        )

        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagen.flow_from_directory(
            directory=self.config.training_data, 
            subset="validation",
            shuffle=False, 
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model: 
        return tf.keras.models.load_model(path)
    
    def evaluation(self): 
        model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = model.evaluate(self.valid_generator)

    def save_score(self): 
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path = Path("scores.json"), data = scores)
```

#### 7.4 Create new pipeline
Now we have create new pipeline to Run above components with data

```
STAGE_NAME = "Evaluation Stage"

class EvaluationPipeline: 
    def __init__(self):
        pass

    def main(self): 
        config = ConfigurationManager()
        val_config = config.get_validation_config()
        evaluation = Evaluation(val_config)
        evaluation.evaluation()
        evaluation.save_score()

if __name__ == "__main__":
    try: 
        logger.info(f"******************")
        logger.info(f">>>>> {STAGE_NAME} started <<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<\n\nX=====================X")

    except Exception as e:
        logger.exception(f"Exception occured in {STAGE_NAME} : {e}")
        raise e
```

#### 7.5 Updating the main.py 
Finally update the main.py to see the effect the result on validation set using try except block

```
STAGE_NAME = "Evaluation Stage"
try: 
    logger.info(f"******************")
    logger.info(f">>>>> {STAGE_NAME} started <<<<<")
    obj = EvaluationPipeline()
    obj.main()
    logger.info(f">>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<\n\nX=====================X")

except Exception as e:
    logger.exception(f"Exception occured in {STAGE_NAME} : {e}")
    raise e
```

✅ **Booommm!! Model Evaluation is Completed**
Do Check out [research/05_model_evaluation.ipynb](https://github.com/Rahul-lalwani-learner/Chicken-Disease-Classification-project/blob/main/research/05_model_evaluation.ipynb)

## `With that our Training Pipeline is also Completed`