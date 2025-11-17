from setuptools import find_packages, setup

setup(
    name="model",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "google-cloud-aiplatform",
        "google-cloud-storage",
        "tensorflow==2.16.2",
        "tf-keras==2.16.*",
        "keras-cv",
        "Keras-Preprocessing==1.1.2",
    ],
)
