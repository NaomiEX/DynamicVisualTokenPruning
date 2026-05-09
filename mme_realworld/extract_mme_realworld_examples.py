from pathlib import Path
import json
import random
import shutil
import tarfile

from huggingface_hub import hf_hub_download


REPO_ID = "yifanzhang114/MME-RealWorld"
OUT_DIR = Path("examples")
DOWNLOAD_DIR = Path("downloads")
N = 20
SEED = 0

# Camera-style direct-perception subsets only. This excludes OCR, diagram/table,
# remote-sensing, intention/relation reasoning, and large crowd-counting tasks.
ALLOWED_SUBTASKS = {"Autonomous_Driving", "Monitoring"}
ALLOWED_CATEGORIES = {
    "Objects_Identify",
    "Object_Count",
    "Attribute_Visual_TrafficSignal",
    "Attribute_Motion_Vehicle",
    "Attribute_Motion_Pedestrain",
    "vehicle/attribute/orientation",
    "person/attribute/color",
    "vehicle/attribute/color",
}


def extract_one(archive_path, member_name, out_path):
    with tarfile.open(archive_path, "r|*") as tar:
        try:
            for member in tar:
                if member.name == member_name or member.name.endswith(member_name):
                    src = tar.extractfile(member)
                    if src is None:
                        raise FileNotFoundError(member_name)
                    with open(out_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    return
        except tarfile.ReadError as exc:
            raise FileNotFoundError(member_name) from exc
    raise FileNotFoundError(member_name)


def archive_for_image(image_path):
    if image_path.startswith("AutonomousDriving/"):
        return "AutonomousDriving.tar.gz"
    if image_path.startswith("monitoring_images/") or image_path.startswith("Monitoring/"):
        return "monitoring_images.tar.gz.part_aa"
    raise ValueError(f"Unknown image folder for {image_path}")


def download_once(filename):
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    local_path = DOWNLOAD_DIR / filename
    if local_path.exists():
        print(f"Using existing {local_path}")
        return local_path

    print(f"Downloading {filename}...")
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
    )
    shutil.copyfile(downloaded, local_path)
    print(f"Saved {local_path}")
    return local_path


def available_members(archive_path):
    cache_path = archive_path.with_suffix(archive_path.suffix + ".members.json")
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return set(json.load(f))

    names = []
    print(f"Indexing available files in {archive_path}...")
    with tarfile.open(archive_path, "r|*") as tar:
        try:
            for member in tar:
                if member.isfile():
                    names.append(member.name)
        except tarfile.ReadError:
            # Some MME-RealWorld archives are published as partial tar chunks.
            # Keep the files that are actually present in the downloaded chunk.
            pass

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(names, f)
    return set(names)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    image_out = OUT_DIR / "images"
    image_out.mkdir(exist_ok=True)

    meta_path = download_once("MME_RealWorld.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    rows = [
        row
        for row in rows
        if row["Task"] == "Perception"
        and row["Subtask"] in ALLOWED_SUBTASKS
        and row["Category"] in ALLOWED_CATEGORIES
    ]

    archive_members = {}
    for archive_name in ("AutonomousDriving.tar.gz", "monitoring_images.tar.gz.part_aa"):
        archive_members[archive_name] = available_members(download_once(archive_name))

    rows = [
        row
        for row in rows
        if row["Image"] in archive_members[archive_for_image(row["Image"])]
    ]

    if len(rows) < N:
        raise ValueError(f"Only found {len(rows)} matching rows, need {N}")

    random.seed(SEED)
    random.shuffle(rows)
    examples = rows
    print(f"Selected up to {N} examples from {len(rows)} matching rows")

    archive_paths = {}
    for archive_name in sorted({archive_for_image(ex["Image"]) for ex in examples}):
        archive_paths[archive_name] = download_once(archive_name)

    records = []
    for ex in examples:
        if len(records) >= N:
            break

        i = len(records)
        image_rel = ex["Image"]
        archive_name = archive_for_image(image_rel)
        suffix = Path(image_rel).suffix or ".jpg"
        out_img = image_out / f"example_{i:02d}{suffix}"

        print(f"Extracting {i + 1}/{N}: {image_rel}")
        try:
            extract_one(archive_paths[archive_name], image_rel, out_img)
        except FileNotFoundError:
            print(f"Skipping missing/unreadable archive member: {image_rel}")
            continue

        records.append(
            {
                "example_id": i,
                "question_id": ex["Question_id"],
                "task": ex["Task"],
                "subtask": ex["Subtask"],
                "category": ex["Category"],
                "dataset": ex.get("Dataset", ""),
                "image_source": image_rel,
                "image_file": str(out_img),
                "question": ex["Text"],
                "answer_choices": ex["Answer choices"],
                "ground_truth": ex["Ground truth"],
            }
        )

    if len(records) < N:
        raise RuntimeError(f"Only extracted {len(records)} examples out of requested {N}")

    metadata_path = OUT_DIR / "examples.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} examples to {OUT_DIR}")
    print(f"Metadata: {metadata_path}")
    print(f"Images:   {image_out}")


if __name__ == "__main__":
    main()
