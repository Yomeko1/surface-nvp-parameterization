import pytest

from surface_nvp.training.config import load_config


def test_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("train:\n  learning_rage: 0.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="train.learning_rage"):
        load_config(str(path))


@pytest.mark.parametrize("content", ["train: []\n", "train:\n  iters: one\n"])
def test_malformed_config_types_are_rejected(tmp_path, content):
    path = tmp_path / "bad-type.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(str(path))
