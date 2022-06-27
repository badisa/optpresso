import re
import ast
from setuptools import setup, find_packages

_version_re = re.compile(r"__version__\s+=\s+(.*)")

with open("optpresso/__init__.py", "rb") as f:
    version = str(
        ast.literal_eval(_version_re.search(f.read().decode("utf-8")).group(1))
    )

setup(
    name="optpresso",
    version=version,
    install_requires=[
        "wandb==0.12.15",
        "pandas==1.1.4",
        "scikit-learn==1.0.2",
        "tensorflow==2.8.*",
        "numpy==1.21.*",
        "matplotlib==3.3.3",
        "scipy==1.7.1",
        "opencv-python==4.4.0.46",
        "astropy==4.3.1",
        "flask==1.1.2",
    ],
    packages=find_packages(),
    entry_points={"console_scripts": ["optpresso=optpresso.commands:main"]},
    author="Forrest York",
    description="Optpresso: ML for espresso",
    classifiers=[
        "Programming Language :: Python",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
    ],
)
