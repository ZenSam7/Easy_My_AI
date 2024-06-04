from setuptools import setup

# Это для библиотеки easymyai
# Как собрать библиотеку:
"""
pip install --upgrade twine
python -m twine upload dist/*
python setup.py sdist
twine upload dist/*
"""


setup(
    name="easymyai",
    version="6.0",
    description="Easy creation of your own simple neural network",
    long_description=open("./easymyai/README.md", "r").read(),
    long_description_content_type="text/markdown",
    packages=["easymyai"],
    install_requires=["numpy>=1.26.2"],
    url="https://github.com/ZenSam7/My_AI",
    author="ZenSam7",
    zip_safe=False,
)