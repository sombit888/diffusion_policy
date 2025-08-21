"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import requests
from flask import Flask, request, jsonify
import os
from PIL import Image
from torchvision import transforms

to_tensor = transforms.ToTensor()
policy = None
device = None
app = Flask(__name__)


def load_model(checkpoint_path: str, output_dir: str, device_str: str):
    global policy, device

    if os.path.exists(output_dir):
        print(f"Output path {output_dir} already exists!")

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    _policy = workspace.model
    if cfg.training.use_ema:
        _policy = workspace.ema_model

    device = torch.device(device_str)
    _policy.to(device)
    _policy.eval()

    policy = _policy
    print("Model loaded and ready.")


# @app.route("/app/predict_action", methods=["POST"])
# def predict_action():
#     global policy, device
#     if policy is None:
#         return jsonify({"error": "Model not loaded."}), 500

#     image_main_file = request.files["image_main"]  # type: FileStorage
#     image_main = Image.open(image_main_file.stream)

#     image_secondary_file = request.files["image_secondary"]
#     image_secondary = Image.open(image_secondary_file.stream)

#     joint_position = torch.tensor(json.loads(request.form['joint_position'])).to("cuda")
#     gripper_position = torch.tensor(json.loads(request.form['gripper_position'])).to("cuda")
#     agent_pos = torch.cat((joint_position,gripper_position)).unsqueeze(0)
#     image_main = to_tensor(image_main).unsqueeze(0)
#     image_secondary = to_tensor(image_secondary).unsqueeze(0)

#     nobs = {
#         "image_main": image_main.unsqueeze(1).cuda(),  # [B, Ta, C, H, W]
#         "image_secondary": image_secondary.unsqueeze(1).cuda(),
#         "agent_pos": agent_pos.unsqueeze(1).cuda()/2*torch.pi,  # normalize to [-0.5, 0.5]
#     }

#     with torch.no_grad():
#         action = policy.predict_action(nobs)
#     # You might need to post-process action, depending on model output
#     action_first = action['action'].cpu()[0,1].tolist()
#     arm_action = action_first[:7]  # Assuming first 7 are joint positions
#     gripper_action = action_first[7]  # Assuming the 8th is gripper position
#     return jsonify({
#             "arm_action": arm_action,
#             "gripper_action": gripper_action,
#         })


# Starting position: [ 3.67500335e-01 8.79731495e-04 4.88036543e-01 3.11467892e+00 -1.03698627e-02 -2.41391687e-02]
@app.route("/app/predict_action", methods=["POST"])
def predict_action():
    global policy, device
    if policy is None:
        return jsonify({"error": "Model not loaded."}), 500

    image_main_file = request.files["image_main"]  # type: FileStorage
    image_main = Image.open(image_main_file.stream)

    image_secondary_file = request.files["image_secondary"]
    image_secondary = Image.open(image_secondary_file.stream)

    image_wrist = request.files.get("image_wrist")
    image_wrist = Image.open(image_wrist.stream)

    cartesian_position = torch.tensor(json.loads(request.form["robot_pose"])).to("cuda")
    gripper_position = torch.tensor(json.loads(request.form["gripper_position"])).to(
        "cuda"
    )

    image_main = to_tensor(image_main).unsqueeze(0)
    image_secondary = to_tensor(image_secondary).unsqueeze(0)
    image_wrist = to_tensor(image_wrist).unsqueeze(0)

    agent_pos = torch.cat(
        (cartesian_position[:3], gripper_position), dim=-1
    )  # [B, Ta, Da])
    agent_pos = agent_pos.unsqueeze(0)
    nobs = {
        "image_main": image_main.unsqueeze(1).cuda(),  # [B, Ta, C, H, W]
        "image_secondary": image_secondary.unsqueeze(1).cuda(),
        "image_wrist": image_wrist.unsqueeze(1).cuda(),
        "agent_pos": agent_pos.unsqueeze(0).cuda(),
    }

    with torch.no_grad():
        action = policy.predict_action(nobs)
    # You might need to post-process action, depending on model output
    action_first = action["action"].cpu()[0, 1].tolist()
    trans = action_first[:3]  # Assuming first 7 are joint positions
    print(trans)
    # import ipdb
    # ipdb.set_trace()
    rot = [0, 0, 0]
    arm_action = trans + rot
    gripper_action = action_first[3]  # Assuming the 8th is gripper position
    return jsonify(
        {
            "arm_action": arm_action,
            "gripper_action": gripper_action,
        }
    )


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
def main(checkpoint, output_dir, device):
    load_model(checkpoint, output_dir, device)
    app.run(host="0.0.0.0", port=8889)


if __name__ == "__main__":
    main()
