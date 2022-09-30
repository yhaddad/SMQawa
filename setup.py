from setuptools import setup

requirements = [
    'numpy','coffea','awkward', 'onnxruntime'
]

setup(
    name='ristretto',
    version='0.0.1',
    description="In algeria Coffea is called Qawa",
    author="Yacine Haddad",
    author_email='yhaddad@cern.ch',
    packages=[
        'qawa',
    ],
    package_dir={'qawa':
                 'qawa'},
    include_package_data=True,
    install_requires=requirements,
    license="GNU General Public License v3",
    zip_safe=False,
    keywords='binopt',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)