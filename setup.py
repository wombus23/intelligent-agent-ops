"""
Setup configuration for Intelligent Agent Ops
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="intelligent-agent-ops",
    version="0.1.0",
    author="Muhammad Noor Ullah Ejaz",
    author_email="noorejaz576@gmail.com",
    description="A comprehensive LLMOps platform for multi-agent orchestration, prompt versioning, and intelligent monitoring",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wombus23/intelligent-agent-ops",
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "openai>=1.12.0",
        "anthropic>=0.18.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.0",
        "langchain>=0.1.0",
        "langchain-openai>=0.0.5",
        "langsmith>=0.0.87",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "aiohttp>=3.9.0",
        "streamlit>=1.30.0",
        "plotly>=5.18.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=23.12.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
            "pylint>=3.0.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "iaops=cli.main:main",  # If you add a CLI
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "llm",
        "llmops",
        "ai",
        "agents",
        "multi-agent",
        "prompt-management",
        "cost-tracking",
        "langchain",
        "openai",
    ],
    project_urls={
        "Bug Reports": "https://github.com/wombus23/intelligent-agent-ops/issues",
        "Source": "https://github.com/wombus23/intelligent-agent-ops",
    },
)
