from setuptools import setup

setup(
    name="page2pod",
    version="0.1.0",
    description="Convert web pages to chapter-based podcasts",
    author="Robert Melton",
    license="MIT",
    url="https://github.com/robertmeta/page2pod",
    py_modules=["page2pod"],
    install_requires=[
        "openai",
        "mutagen",
        "beautifulsoup4",
        "requests",
    ],
    entry_points={
        "console_scripts": [
            "page2pod=page2pod:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
