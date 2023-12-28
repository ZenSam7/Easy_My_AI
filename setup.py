from setuptools import setup

# Это для библиотеки easymyai

setup(
    name="easymyai",
    version="4.2",
    description="Easy creation of your own simple neural network",
    long_description=open("../My_AI/easymyai/README.md", "r").read(),
    long_description_content_type="text/markdown",
    packages=["easymyai"],
    install_requires=["numpy>=1.26.2"],
    url="https://github.com/ZenSam7/My_AI",
    author="ZenSam7",
    author_email="samkirich@yandex.ru",
    zip_safe=False,
)
