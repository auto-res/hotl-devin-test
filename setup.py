from setuptools import setup, find_packages

setup(
    name="wrn-improvements",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "tqdm>=4.65.0",
    ],
    author="Research Team",
    description="Wide Residual Networks Implementation with Accuracy Improvements",
    python_requires=">=3.8",
)
