"""Shared fixtures use the same artificial world as the offline demo."""
import pytest
from waterdet.synthetic import make_world, make_bands, write_stack, demo_config

@pytest.fixture(scope="session")
def world():
    return make_world(0)


@pytest.fixture(scope="session")
def dry_bands(world):
    return make_bands(world, "dry")


@pytest.fixture(scope="session")
def wet_bands(world):
    return make_bands(world, "wet")


@pytest.fixture()
def data_dir(tmp_path, dry_bands, wet_bands):
    d = tmp_path / "data"
    d.mkdir()
    write_stack(d / "s2_dry.tif", dry_bands)
    write_stack(d / "s2_wet.tif", wet_bands)
    return d


@pytest.fixture()
def small_cfg():
    """Config sized for the 400 x 500 px synthetic world."""
    return demo_config()
