from setuptools import setup, find_packages

setup(
    name="fake_job_detector",
    version="0.1.0",
    author="Rusheenddra Basani",
    author_email="",
    description="Production-ready NLP Fake Job Posting Detection System",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[],
    python_requires=">=3.10",
)