import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setuptools.setup(
    name="datasets_for_haystack",
    version="0.0.1",
    author="Giguru Scheuer",
    author_email="giguru.scheuer@gmail.com",
    description="A user-friendly interface for using datasets commonly used in Conversational Question Answering and Conversational Search research with the framework Haystack",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
)