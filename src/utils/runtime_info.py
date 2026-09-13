import os


def get_container_runtime_metadata():
    metadata = {}

    image_tag = os.getenv(
        "RUNTIME_IMAGE_TAG"
    )

    image_id = os.getenv(
        "RUNTIME_IMAGE_ID"
    )

    if image_tag:
        metadata[
            "container.image_tag"
        ] = image_tag

    if image_id:
        metadata[
            "container.image_id"
        ] = image_id

    return metadata
