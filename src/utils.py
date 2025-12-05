import shutil, os, json
import csv
from pathlib import Path
from typing import Optional, Dict, Any, List


def write_entries_jsonl(output_path, configurations):
    with open(output_path, "w", encoding="utf-8") as f:
        for configuration in configurations:
            f.write(json.dumps(configuration) + "\n")


    print(f"Wrote {len(configurations)} entries → {output_path}")




def configuration_utils(
    generation_location="outputs/av_realistic",
    weather_configs=["fog", "morning_sun", "night", "rain", "snow", "wooden_road"]
    ):

    # Ensure absolute/normalized path
    base_dir = os.path.abspath(generation_location)
    new_path = f"{generation_location}_results"
    exclude_tokens = ["_edge", "_vis", "_depth", "_seg"]

    for filename in os.listdir(base_dir):
        if not filename.endswith(".mp4"):
            continue

        lower = filename.lower()

        # Skip files with excluded substrings
        if any(tok in lower for tok in exclude_tokens):
            continue

        # Determine weather tag from filename
        target_weather = None
        for w in weather_configs:
            if w in lower:
                target_weather = w
                break
        if target_weather is None:
            continue  # no weather tag found, skip

        # Create weather directory
        dest_dir = os.path.join(new_path, target_weather)
        os.makedirs(dest_dir, exist_ok=True)

        # Copy file
        src_path = os.path.join(base_dir, filename)
        dst_path = os.path.join(dest_dir, filename)
        shutil.copy(src_path, dst_path)

        print(f"Copied {filename} → {dest_dir}")



def parse_config_from_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Parse a video filename according to the naming convention:

        NAME_WEATHER_GUIDANCE_EDGE_SEG_SEG_MASK_FLAG_VIS_DEPTH.mp4

    Where:
        - NAME can contain underscores
        - WEATHER is a string token (e.g., 'fog', 'night', 'rain')
        - GUIDANCE is numeric (e.g., '3')
        - EDGE, SEG, VIS, DEPTH are ints representing weight * 10
        - SEG_MASK_FLAG is 't' or 'f'

    Returns a dict with parsed fields, or None if parsing fails.
    """
    stem = Path(filename).stem  # remove .mp4
    parts = stem.split("_")

    # Need at least 8 parts: NAME + 7 config tokens
    if len(parts) < 8:
        print(f"[WARN] Skipping '{filename}': not enough tokens for config.")
        return None

    # Last 7 tokens are config, the rest is NAME
    name_parts = parts[:-7]
    weather, guidance_str, edge10_str, seg10_str, segmask_flag, vis10_str, depth10_str = parts[-7:]

    name = "_".join(name_parts)

    try:
        guidance = float(guidance_str)
        edge = int(edge10_str) / 10.0
        seg = int(seg10_str) / 10.0
        vis = int(vis10_str) / 10.0
        depth = int(depth10_str) / 10.0
        seg_mask = segmask_flag.lower() == "t"
    except ValueError:
        print(f"[WARN] Skipping '{filename}': numeric parse error.")
        return None

    return {
        "filename": filename,
        "name": name,
        "weather": weather,
        "guidance": guidance,
        "edge": edge,
        "seg": seg,
        "seg_mask": seg_mask,
        "vis": vis,
        "depth": depth,
    }


def index_videos_to_csv(
    generation_location: str = "outputs/omniverse_generations_av",
    output_csv: str = "outputs/omniverse_generations_av/video_index.csv",
    recursive: bool = False,
) -> None:
    """
    Walk through `generation_location`, parse each video filename according to the
    naming convention, and write the results to a CSV.

    The CSV columns are:
        filename, path, name, weather, guidance, edge, seg, seg_mask, vis, depth
    """
    base_dir = Path(generation_location).resolve()
    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not base_dir.exists():
        raise FileNotFoundError(f"Generation directory not found: {base_dir}")

    if recursive:
        video_iter = base_dir.rglob("*.mp4")
    else:
        video_iter = (p for p in base_dir.iterdir() if p.suffix == ".mp4")

    rows: List[Dict[str, Any]] = []

    for video_path in video_iter:
        parsed = parse_config_from_filename(video_path.name)
        if parsed is None:
            continue

        parsed["path"] = str(video_path.resolve())
        rows.append(parsed)

    fieldnames = ["filename", "path", "name", "weather", "guidance",
                  "edge", "seg", "seg_mask", "vis", "depth"]

    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} video entries → {output_csv_path}")





if __name__ == "__main__":
    filename = "av_realistic_wooden_road_7_6_4_f_0_0"
    print(parse_config_from_filename(filename))

    # configuration_utils()
    configuration_utils()

    # After you’ve generated videos with the naming convention:
    # index_videos_to_csv(
    #     generation_location="outputs/omniverse_generations_av",
    #     output_csv="outputs/omniverse_generations_av/video_index.csv",
    #     recursive=True,  # set False if you only want the top-level folder
    # )