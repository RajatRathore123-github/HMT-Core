from setuptools import setup, find_packages

setup(
    name="hmt-core",
    version="0.1.0",
    author="Rajat Rathore",
    author_email="rajat.rathore.research@gmail.com",
    description="Homeostatic Manifold Transformers for Non-Singular, Interpretable AI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
        "matplotlib>=3.5.0",
        "requests>=2.28.0",
    ],
)
