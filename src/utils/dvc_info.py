from pathlib import Path

import yaml


def get_dvc_metadata(
    data_path: str | Path,
):
    data_path = Path(data_path)

    dvc_path = Path(
        f"{data_path}.dvc"
    )

    if not dvc_path.exists():
        return None

    with dvc_path.open("r") as f:
        metadata = yaml.safe_load(f)

    outputs = metadata.get(
        "outs",
        [],
    )

    if not outputs:
        return None

    output = outputs[0]

    hash_type = output.get(
        "hash",
        "md5",
    )

    content_hash = output.get(
        hash_type
    )

    return {
        "dvc_file": str(dvc_path),
        "hash_type": hash_type,
        "hash": content_hash,
        "size": output.get("size"),
    }
