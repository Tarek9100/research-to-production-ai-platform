from src.utils.runtime_info import get_container_runtime_metadata


def test_container_runtime_metadata(monkeypatch):
    monkeypatch.setenv(
        "RUNTIME_IMAGE_TAG",
        "research-ai-training:abc1234",
    )
    monkeypatch.setenv(
        "RUNTIME_IMAGE_ID",
        "sha256:deadbeef",
    )

    metadata = get_container_runtime_metadata()

    assert metadata == {
        "container.image_tag":
            "research-ai-training:abc1234",
        "container.image_id":
            "sha256:deadbeef",
    }
