from pathlib import Path
from typing import List, Union
import os
import yaml
from box import ConfigBox
import sys
import joblib

from fake_job_detector.exception import CustomException
from fake_job_detector.logger import logger


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its contents as a ConfigBox.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)

        return ConfigBox(content)

    except Exception as e:
        raise CustomException(e, sys)


def create_directories(
    path_to_directories: List[Union[str, Path]],
    verbose=True
):

    try:
        for path in path_to_directories:

            path = Path(path)

            path.mkdir(parents=True, exist_ok=True)

            if verbose:
                logger.info(f"Created directory at: {path}")

    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path: Path, obj):

    """
    Saves a Python object using joblib.
    """

    try:

        os.makedirs(file_path.parent, exist_ok=True)

        joblib.dump(obj, file_path)

        logger.info(f"Object saved at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path: Path):

    """
    Loads a Python object using joblib.
    """

    try:

        return joblib.load(file_path)

    except Exception as e:
        raise CustomException(e, sys)