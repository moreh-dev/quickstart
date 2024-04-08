import setuptools
import subprocess
from setuptools.command.develop import develop


def install_precommit():
    result = subprocess.call("pre-commit install", shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

    if result != 0:
        print("pre-commit build fail")
        exit(1)


class CustomDevelopCommand(develop):

    def run(self):
        develop.run(self)
        install_precommit()


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(name="moreh_model_hub",
                 version="0.0.1",
                 author="Team Model",
                 author_email="",
                 description="A framework for training various models",
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url="",
                 packages=setuptools.find_packages(exclude=["cache"]),
                 package_data={},
                 dependency_links=[],
                 include_package_data=True,
                 classifiers=[
                     "Development Status :: 3 - Alpha",
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent",
                 ],
                 python_requires="==3.8.*",
                 install_requires=[
                     "torch==1.13.1", "torchaudio==0.13.1", "torchvision==0.14.1",
                     "transformers==4.36.2", "tokenizers>=0.14", "datasets==2.14.5", "loguru==0.5.3", "sentencepiece",
                     "jsonstream==0.0.1", "tqdm>=4.27", "numpy==1.23.0",
                     "timm==0.4.12", "evaluate", "albumentations"
                     "packaging==23.1",
                     "pre-commit", "webdataset",
                     "protobuf==3.13.0"
                 ],
                 cmdclass={"develop": CustomDevelopCommand})
