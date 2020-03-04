import pytest
from pathlib import Path


@pytest.fixture
def data_dir(request):
    path = Path(request.module.__file__)
    data_path: Path = path.parent.joinpath('data')
    print(data_path)
    assert data_path.is_dir()
    return data_path
