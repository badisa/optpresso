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
        "pandas>=1.4.0,<2.0.0",
        "scikit-learn==1.3.*",
        "tensorflow==2.13.*",
        "numpy",
        "matplotlib",
        "scipy==1.11.*",
        "astropy",
        "flask",
        "jinja2",
    ],
    extra_requires={
        "dev": [
            "wandb==0.12.*",
        ],
    },
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
