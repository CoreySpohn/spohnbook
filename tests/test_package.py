import spohnbook


def test_version_is_a_string():
    assert isinstance(spohnbook.__version__, str)
    assert spohnbook.__version__
