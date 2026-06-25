from setuptools import setup, find_packages

setup(
    name="ddp-training-pipeline",
    version="1.0.0",
    description="Production-ready distributed PyTorch DDP training pipeline with 92% scaling efficiency",
    author="Mamidi Sri Teja",
    author_email="sri.teja@example.com",
    url="https://github.com/yourusername/DDP-Training-Pipeline",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "mlflow>=2.0.0",
        "tensorboard>=2.10.0",
        "pyyaml>=6.0",
        "tqdm>=4.60.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
