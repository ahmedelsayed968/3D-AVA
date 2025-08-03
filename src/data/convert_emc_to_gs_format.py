
import argparse
from pathlib import Path
import torch
import numpy as np

from data.schema import SMPLModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path",type=str,help="path to the experiment")
    args = parser.parse_args()

    exp_path = Path(args.exp_path)
    smpl_dir = exp_path / "smpl"
    smpl_paths = list(smpl_dir.iterdir())

    smpl_data = [
        SMPLModel.from_path(path)
        for path in smpl_paths
    ]
    
    trans = [
        torch.tensor(data.Th).to(torch.float32)
        for data in smpl_data
    ]
    trans = torch.concat(trans)

    beta = [
        torch.tensor(data.shapes).to(torch.float32)
        for data in smpl_data
    ]
    beta = torch.concat(beta)

    body_pose = [
        torch.tensor(data.poses).to(torch.float32)
        for data in smpl_data
    ]
    body_pose = torch.concat(body_pose)

    r = [
        torch.tensor(data.Rh).to(torch.float32)
        for data in smpl_data
    ]
    r = torch.concat(r)

    body_pose = torch.concat([r,body_pose],dim=1)
    smpl_parms ={
        "betas":beta.numpy(),
        "transl":trans.numpy(),
        "thetas":body_pose.numpy(),
    }
    np.savez(exp_path/"poses_optimized.npz",**smpl_parms)