from setuptools import setup, find_packages

setup(
    name="shallow2deep",
    packages=find_packages(exclude=["tests"]),
    version="1.0.0",
    author="Alex Matskevych, Adrian Wolny, Lorenzo Cerrone",
    url="https://github.com/wolny/pytorch-3dunet",
    license="MIT",
    python_requires='>=3.7', 
    entry_points={'console_scripts': [
        'shallow2deep_create_filters=shallow2deep.rf.create_filters:main',
        'shallow2deep_rf_generation=shallow2deep.rf.rf_generation:main',
        'shallow2deep_rf_prediction=shallow2deep.rf.rf_prediction:main',
        'shallow2deep_extract_single_axis=shallow2deep.helpers.extract_single_axis:main',
        'shallow2deep_add_volume_to_dataset=shallow2deep.helpers.add_volume_to_dataset:main',
        'shallow2deep_train=shallow2deep.train:main',
        'shallow2deep_predict=shallow2deep.predict:main']
        }
)
