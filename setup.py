"""Setup file for the llm_nexus package."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-nexus",  # This is the PyPI distribution name
    version="0.1.2",
    author="Daniel Tedesco",
    author_email="dtedesco1@gmail.com",
    description="A unified interface for multiple LLM providers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dtedesco1/llm_nexus",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pydantic",
        "anthropic",
        "google-generativeai",
        "openai",
    ],
    extras_require={
        "dev": ["pytest", "black", "isort", "pylint"],
    },
)
