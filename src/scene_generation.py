import json
import os
import shutil
from src.utils import *

def create_entry_jsonl(
    name,
    guidance,
    prompt_location,
    prompt_type,
    video_location,
    control_weight_location,
    edge = 0,
    edge_path = False,
    seg = 0,
    seg_mask = None,
    vis = 0,
    vis_path = False,
    depth = 0,
    depth_path = False,
    generation_output_dir = None
    ):

    '''
    NAMING CONVENTION
    NAME_WEATHER_GUIDANCE_EDGE_SEG_SEG MASK (t or f)_VIS_DEPTH
    '''

    entry_name = f"{name}_{prompt_type}_{guidance}_{int(edge*10)}_{int(seg*10)}_{'t' if seg_mask else 'f'}_{int(vis*10)}_{int(depth*10)}"
    if generation_output_dir:
        if os.path.exists(os.path.join(generation_output_dir, f"{entry_name}.mp4")):
            print(f"Entry {entry_name} already exists, skipping...")
            return None, None

    entry = {
        "name": entry_name,
        "prompt_path": os.path.join(prompt_location, f"{prompt_type}.txt"),
        "video_path": video_location,
        "guidance": guidance
    }

    if edge:
        entry["edge"] = {
            "control_weight": edge
        }
        if edge_path:
            entry["edge"]["control_path"] = os.path.join(control_weight_location, "edge.mp4")


    if seg:
        entry["seg"] = {
            "control_weight": seg,
            "control_path": os.path.join(control_weight_location, "seg.mp4")
        }
        if seg_mask:
            entry["seg"]["mask_path"] = os.path.join(control_weight_location, "seg_mask.mp4")
    

    if depth:
        entry["depth"] = {
            "control_weight": depth
        }
        if depth_path:
            entry["depth"]["control_path"] = os.path.join(control_weight_location, "depth.mp4")
    
    if vis:
        entry["vis"] = {
            "control_weight": vis
        }

        if vis_path:
            entry["vis"]["control_path"] = os.path.join(control_weight_location, "vis.mp4")


    return entry, entry_name




def generate_omniverse_configurations(
    video_location="../simulation_data/simulator_rgb_input.mp4", 
    location="../simulation_data", 
    name="omniverse_generations_av"):
    
    '''
    Running the script:
    - For single GPU: python examples/inference.py -i scripts/omniverse_av_configs.jsonl -o outputs/omniverse_generations_av --disable-guardrails
    - For Multi GPU: torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py -i scripts/omniverse_av_configs.jsonl -o outputs/omniverse_generations_av --disable-guardrails
    '''

    scripts_output_dir = os.path.join("scripts/omniverse_av_configs.jsonl")
    generation_output_dir = os.path.join("outputs/", name)
    os.makedirs(generation_output_dir, exist_ok=True)


    weather = ["fog", "morning_sun", "night", "rain", "snow", "wooden_road"]
    # weather = ["fog", "morning_sun"]
    guidance = [3, 7]
    configurations = [
        {"vis": 1.0},
        {"depth": 1.0, "depth_path":True},
        {"edge": 1.0, "edge_path": True},
        {"edge": 1.0,"edge_path":True, "seg": 0.6},
        {"edge": 1.0,"edge_path":True, "seg": 0.9, "depth": 0.9, "depth_path": True},
        {"edge": 1.0,"edge_path":True, "depth": 0.9, "depth_path": True},

        {"edge": 1.0,"edge_path":True, "depth": 0.9, "seg": 1},
        {"edge": 1.0,"edge_path":True, "depth": 0.9, "seg": 0.5},
        {"edge": 0.9,"edge_path":True, "depth": 1.0, "depth_path":True},
        {"edge": 0.5,"edge_path":True, "depth": 1.0, "depth_path":True},
        {"edge": 0.5,"edge_path":True, "depth": 1.0, "depth_path":True, "seg": 0.4},
        {"edge": 0.4,"edge_path":True, "depth": 1.0, "depth_path":True},
        {"edge": 1.0,"edge_path":True, "depth": 0.5, "depth_path":True},
        {"edge": 1.0,"edge_path":True, "vis": 0.2},
        {"edge": 1.0,"edge_path":True, "vis": 0.5},
        {"edge": 0.6,"edge_path":True, "seg": 0.4},
    ]


    full_configs = []
    name_overlap = []

    for w in weather:
        for g in guidance:
            for c in configurations:
                new_config, entry_name = create_entry_jsonl(
                        name=name,
                        guidance=g,
                        prompt_location=location,
                        prompt_type=w,
                        video_location=video_location,
                        control_weight_location=location,
                        edge=c.get("edge", 0),
                        edge_path=c.get("edge_path", None),
                        seg=c.get("seg", 0),
                        seg_mask=c.get("seg_mask", None),
                        vis=c.get("vis", 0),
                        depth=c.get("depth", 0),
                        generation_output_dir=generation_output_dir
                    )

                if new_config:
                    full_configs.append(new_config)
                    assert entry_name not in name_overlap, f"Error: entry name {name_overlap} already exists"
                    name_overlap.append(entry_name)
                
    
    write_entries_jsonl(scripts_output_dir, full_configs)



def av_generate_realistic_configurations(
    video_location="../milestone_data/output_fixed.mp4", 
    location="../milestone_data", 
    name="av_realistic"):
    
    '''
    Running the script:
    - For single GPU: python cosmos_transfer2_5/examples/inference.py -i scripts/av_configs.jsonl -o outputs/av_realistic --disable-guardrails
    - For Multi GPU: torchrun --nproc_per_node=8 --master_port=12341 cosmos_transfer2_5/examples/inference.py -i scripts/av_configs.jsonl -o outputs/av_realistic --disable-guardrails
    '''

    scripts_output_dir = os.path.join("scripts/av_configs.jsonl")
    generation_output_dir = os.path.join("outputs/", name)
    os.makedirs(generation_output_dir, exist_ok=True)


    weather = ["fog", "morning_sun", "night", "rain", "no_snow", "wooden_road"]
    # weather = ["fog", "morning_sun"]
    guidance = [3, 7]
    configurations = [
        {"depth": 1},
        {"edge": 1},
        {"vis": 1},
        {"seg": 1},
        {"edge": 1, "depth": 0.9},
        {"edge": 1, "depth": 0.9, "seg": 1},
        {"edge": 1, "depth": 0.9, "seg": 0.5},
        {"edge": 0.9, "depth": 1.0},
        {"edge": 0.5, "depth": 1.0},
        {"edge": 0.5, "depth": 1.0, "seg": 0.4},
        {"edge": 0.4, "depth": 1.0},
        {"edge": 1.0, "depth": 0.5},
        {"edge": 1.0, "depth": 0.9, "vis": 0.2},
        {"edge": 1.0, "vis": 0.2},
        {"edge": 1.0, "vis": 0.5},
        {"edge": 0.6, "seg": 0.4},
    ]

    full_configs = []
    name_overlap = []

    for w in weather:
        for g in guidance:
            for c in configurations:
                new_config, entry_name = create_entry_jsonl(
                        name=name,
                        guidance=g,
                        prompt_location=os.path.join(location, "clip_0_easier_prompts/"),
                        prompt_type=w,
                        video_location=video_location,
                        control_weight_location=location,
                        edge=c.get("edge", 0),
                        edge_path=c.get("edge_path", None),
                        seg=c.get("seg", 0),
                        seg_mask=c.get("seg_mask", None),
                        vis=c.get("vis", 0),
                        depth=c.get("depth", 0),
                        generation_output_dir=generation_output_dir
                    )
                if new_config:
                    full_configs.append(new_config)
                    assert entry_name not in name_overlap, f"Error: entry name {name_overlap} already exists"
                    name_overlap.append(entry_name)
                
    
    write_entries_jsonl(scripts_output_dir, full_configs)
    


if __name__ == "__main__":
    generate_omniverse_configurations()
    # av_generate_realistic_configurations()


    ## python -m src.scene_generation