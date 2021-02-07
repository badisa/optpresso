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
        "pandas==1.1.4",
        "scikit-learn==0.23.2",
        "tensorflow==2.4.1",
        "numpy==1.19.*",
        "matplotlib==3.3.3",
        "keras==2.4.3",
        "pydot==1.4.1",
        "opencv-python==4.4.0.46",
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
