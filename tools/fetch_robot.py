# BrainAI - built by Mohammed Zahid.
"""Download LeRobot robot-demonstration datasets (camera video + joint state + actions).

    python tools/fetch_robot.py --out robot_data
    python tools/fetch_robot.py --out robot_data --skip koch_pick_place_1_lego   # save 2 GB

Each dataset is parquet (state/action per timestep) plus mp4 camera streams, in
the LeRobot format. state -> BrainAI touch/proprioception input; action -> the
actuator targets; video -> the vision/video encoders.
"""
import argparse
from pathlib import Path

DATASETS = [
    "lerobot/pusht",                              # 2D pushing, tiny
    "lerobot/xarm_lift_medium",                   # sim arm lift
    "lerobot/aloha_sim_insertion_human",          # bimanual sim, human demos
    "lerobot/aloha_sim_transfer_cube_human",
    "lerobot/droid_100",                          # real Franka, 100 episodes of DROID
    "lerobot/koch_pick_place_1_lego",             # real low-cost arm (~2 GB)
]

# Humanoid (Unitree H1, 19-DoF humanoid, teleoperated), ~1.4 GB total.
HUMANOID = [
    "lerobot/unitreeh1_fold_clothes",
    "lerobot/unitreeh1_rearrange_objects",
    "lerobot/unitreeh1_two_robot_greeting",
    "lerobot/unitreeh1_warehouse",
]
# Much larger humanoid sets (Unitree G1, 9-18 GB each); pull these on Colab/Drive only:
HUMANOID_BIG = [
    "unitreerobotics/G1_Dex1_HangCup",        # 8.8 GB
    "unitreerobotics/G1_Dex1_Stack_Block",    # 18 GB
    "unitreerobotics/G1_Dex1_Wipe_Table",     # 18 GB
]

if __name__ == "__main__":
    from huggingface_hub import snapshot_download
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="robot_data")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--humanoid", action="store_true", help="humanoid sets only")
    ap.add_argument("--humanoid-big", action="store_true", help="also the 9-18 GB G1 sets")
    a = ap.parse_args()
    repos = HUMANOID + (HUMANOID_BIG if a.humanoid_big else []) if a.humanoid else DATASETS
    for repo in repos:
        name = repo.split("/")[1]
        if name in a.skip:
            continue
        try:
            p = snapshot_download(repo, repo_type="dataset", local_dir=Path(a.out) / name)
            print(f"[done] {repo} -> {p}")
        except Exception as e:
            print(f"[fail] {repo}: {e}")
