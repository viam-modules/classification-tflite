from setuptools import find_packages, setup

setup(
    name="model",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "google-cloud-aiplatform",
        "google-cloud-storage",
        "tensorflow==2.11.0"
        "keras==2.11.0",
        "keras-cv==0.5.0",
        "Keras-Preprocessing==1.1.2",
        "tflite-support",
    ],
)
