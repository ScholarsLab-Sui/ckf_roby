from setuptools import setup, find_packages

setup(
    name="roby",
    version="0.0.1",
    author="Jingyi Tian",
    author_email="jingyitian9@gmail.com",
    description="Yet Another Robot Learning Toolkit",
    packages=find_packages(exclude=["tests", "docs"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21.0",
        "draccus>=0.1.0",
    ],
)