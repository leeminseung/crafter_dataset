import setuptools
import pathlib


setuptools.setup(
    name='mini_crafter',
    version='1.7.1',
    description='Open world survival game for reinforcement learning.',
    url='http://github.com/danijar/crafter',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=['mini_crafter'],
    package_data={'mini_crafter': ['data.yaml', 'assets/*']},
    entry_points={'console_scripts': ['mini_crafter=mini_crafter.run_gui:main']},
    install_requires=[
        'numpy', 'imageio', 'pillow', 'opensimplex', 'ruamel.yaml'],
    extras_require={'gui': ['pygame']},
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Games/Entertainment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
