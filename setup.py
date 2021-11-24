import setuptools

with open("Readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tst",
    version="0.0.1",
    author="",
    author_email="",
    description="TeST: Temporal-Stable Thresholding for Semi-supervised Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=None,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
    entry_points={
        "console_scripts": [
            "tst_train=tst.tools.train:main",
        ]
    },
)
