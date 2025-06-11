"""
Setup script for Legal-AI package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="legal-ai",
    version="1.0.0",
    author="Legal-AI Team",
    author_email="support@legal-ai.com",
    description="AI-powered platform for legal document analysis using LlamaIndex and RAG",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/legal-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Legal Industry",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "api": [
            "fastapi>=0.104.1",
            "uvicorn>=0.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "legal-ai=legal_ai.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "legal_ai": ["*.yaml", "*.yml"],
    },
)
