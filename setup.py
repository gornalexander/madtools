from setuptools import setup

REQUIREMENTS = ['numpy',
                'pandas',
                'matplotlib',
                'holoviews'
                ]

LINKS = [
]

setup(
    name='madtools',
    version="0.0.1",
    packages=['madtools'],
    url='',
    license='CERN',
    author='A. Gorn',
    author_email='aleksandr.gorn@cern.ch',
    description='MADX tools',
    #python_requires='>=3.6',
    install_requires=REQUIREMENTS + LINKS,
)
