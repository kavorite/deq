from setuptools import find_namespace_packages, setup

setup(
    name="deq",
    version="0.0.2",
    description="deep equilibrium networks for dm-haiku",
    author="kavorite",
    url="https://github.com/kavorite/deq",
    install_requires=[
        "dm_haiku>=0.0.7",
        "jax>=0.3",
        "diffrax>=0.2",
        "reparam @ git+https://github.com/kavorite/reparam@v0.0.1#egg=reparam",
    ],
    packages=find_namespace_packages(),
)
