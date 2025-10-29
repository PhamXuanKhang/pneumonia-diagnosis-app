"""Setup script for pneumonia diagnosis ML package"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pneumonia-diagnosis-ml",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Machine Learning package for pneumonia diagnosis from X-ray images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pneumonia-diagnosis-app",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.15.0",
        "numpy>=1.24.0",
        "pandas>=2.1.0",
        "pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=7.0.0",
            "jupyter>=1.0.0",
        ],
    },
)
