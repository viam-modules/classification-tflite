from setuptools import find_packages, setup

setup(
    name="model",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=[
        "tensorflow==2.21.0",
        "keras>=3.15,<4",
    ],
)
