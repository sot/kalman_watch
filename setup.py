# Licensed under a 3-clause BSD style license - see LICENSE.rst
from setuptools import setup

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}

entry_points = {
    "console_scripts": [
        "kalman_watch_reacqs = kalman_watch.reacqs:main",
        "kalman_watch_perigee = kalman_watch.perigee:main",
    ]
}

setup(
    name="kalman_watch",
    description="Kalman stars monitor",
    author="Tom Aldcroft",
    author_email="taldcroft@cfa.harvard.edu",
    url="https://sot.github.io/kalman_watch",
    packages=["kalman_watch"],
    tests_require=["pytest"],
    use_scm_version=True,
    setup_requires=["setuptools_scm", "setuptools_scm_git_archive"],
    cmdclass=cmdclass,
    entry_points=entry_points,
)
