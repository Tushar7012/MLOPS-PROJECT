from setuptools import setup, find_packages
from typing import List
HYPEN_E_DOT = "-e ."

def get_requirements(file_path:str)->List[str]:

    """
    The function is used return the list of requirements.

    """

    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT == "-e .":
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
name = "mlproject",
version = "0.0.1",
author = "Tushar",
author_email = "td220627@gmail.com",
packages = find_packages(),
inatall_requires = get_requirements("requirements.txt")
)