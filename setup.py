from setuptools import setup, find_packages

setup(
    name='MLProjectTemplate',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'h3==3.7.6',
        'joblib==1.3.2',
        'numpy==1.26.4',
        'pandas==2.2.1',
        'python-dateutil==2.9.0.post0',
        'pytz==2024.1',
        'scikit-learn==1.4.1.post1',
        'scipy==1.12.0',
        'six==1.16.0',
        'threadpoolctl==3.3.0',
        'tzdata==2024.1'
    ]
)