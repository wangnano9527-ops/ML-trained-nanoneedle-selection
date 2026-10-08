import hashlib
import importlib.util
from pathlib import Path

from needle_select import lattice


def api():
    assert importlib.util.find_spec('needle_select.lattice_api'), 'Public lattice API is required'
    from needle_select import lattice_api
    return lattice_api


def test_public_api_preserves_all_existing_functions_and_types():
    public = api()
    names = [name for name, item in vars(lattice).items()
             if getattr(item, '__module__', '') == lattice.__name__]
    assert len(names) == 20  # 18 functions, two dataclasses
    assert public.API_VERSION == 1
    assert all(getattr(public, name) is getattr(lattice, name) for name in names)
    assert all(name in public.__all__ for name in names)


def test_source_is_actual_calculation_file_not_api_wrapper():
    source = api().lattice_source()
    assert source['provider'] == 'needle-selection'
    assert source['capability'] == '3-4-a'
    assert Path(source['module_file']) == Path(lattice.__file__).resolve()
    assert source['sha256'] == hashlib.sha256(Path(lattice.__file__).read_bytes()).hexdigest()
