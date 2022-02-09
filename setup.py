from setuptools import setup, find_packages

setup(
    name="pytorch3dunet",
    packages=find_packages(exclude=["tests"]),
    version="1.0.0",
    author="Alex Matskevych, Adrian Wolny, Lorenzo Cerrone",
    url="https://github.com/kreshuklab/shallow2deep",
    license="MIT",
    python_requires='>=3.7', 
    entry_points={'console_scripts': [
        'train3dunet=shallow2deep.train:main',
        'predict3dunet=shallow2deep.predict:main']
        }
)
