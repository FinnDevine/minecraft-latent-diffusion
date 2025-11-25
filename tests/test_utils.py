import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("nbtlib")

from utils.palette import invert_palette, normalize_block_name
from utils.prompts import build_prompt
from utils.schem_io import _center_grid, _resize_axis, save_voxel_as_schem, schem_to_voxel
from utils.voxelize import _parse_schem_paths


def test_invert_and_normalize_palette():
    block2id = {"minecraft:Stone": 1, " minecraft:Oak_Planks ": 2}
    normalized = {normalize_block_name(k): v for k, v in block2id.items()}
    inverted = invert_palette(normalized)

    assert normalized == {"minecraft:stone": 1, "minecraft:oak_planks": 2}
    assert inverted[1] == "minecraft:stone"
    assert inverted[2] == "minecraft:oak_planks"


def test_build_prompt_from_url_and_tags():
    row = {"PAGE_URL": "https://example.com/builds/castle-fortress", "TAGS": "castle, medieval"}
    prompt = build_prompt(row)

    assert "castle fortress" in prompt
    assert "castle, medieval" in prompt
    assert "|" in prompt


def test_resize_and_center_grid():
    grid = np.array([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
    resized = _resize_axis(grid, target=2, axis=1)
    assert resized.shape == (2, 2)
    # First and last columns should map to original ends due to linspace indexing
    assert resized[:, 0].tolist() == [1, 4]
    assert resized[:, 1].tolist() == [3, 6]

    cube = np.ones((2, 2, 2), dtype=np.int16)
    centered = _center_grid(cube, resolution=4)
    assert centered.shape == (4, 4, 4)
    assert centered.sum() == cube.sum()
    assert np.all(centered[1:3, 1:3, 1:3] == 1)
    assert np.all(centered[:1] == 0) and np.all(centered[-1:] == 0)


def test_schem_roundtrip(tmp_path):
    # Original grid with two block types
    voxel = np.array(
        [
            [[0, 1], [1, 0]],
            [[1, 0], [0, 1]],
        ],
        dtype=np.int16,
    )
    id2block = {0: "minecraft:air", 1: "minecraft:stone"}
    schem_path = tmp_path / "sample.schem"

    save_voxel_as_schem(voxel, id2block=id2block, out_path=schem_path)

    block2id = {"minecraft:air": 0}
    loaded = schem_to_voxel(str(schem_path), resolution=4, block2id=block2id)

    # Centered placement should preserve original layout in the middle of the cube
    assert loaded.shape == (4, 4, 4)
    assert block2id["minecraft:stone"] == 1
    assert np.array_equal(loaded[1:3, 1:3, 1:3], voxel)

    # Ensure saved schem is gzipped NBT and palette persisted
    with schem_path.open("rb") as f:
        assert f.read(2) == b"\x1f\x8b"
    nbt = schem_path.read_bytes()
    assert b"minecraft:stone" in nbt


def test_parse_schem_paths_handles_invalid_strings():
    assert _parse_schem_paths("['a.schem', 'b.schem']") == ["a.schem", "b.schem"]
    assert _parse_schem_paths("not a list") == []
    assert _parse_schem_paths(123) == []
