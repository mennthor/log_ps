from setuptools import setup, find_packages


setup(
    name="log_ps",
    version="0.1",
    description="ps logger for monitoring script ressource usage.",
    author="Thorben Menne",
    author_email="mennthor@aol.com",
    url="",
    packages=find_packages(),
    install_requires=[],
    entry_points={
        "console_scripts": [
            "log_ps=log_ps:_main",
        ],
    }
)
