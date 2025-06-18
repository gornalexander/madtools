"""
Tests for the madtools package.

"""

import madtools


def test_version():
    # Check tha the package has a __version__ attribute.
    assert madtools.__version__ is not None
