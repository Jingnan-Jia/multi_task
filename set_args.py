# -*- coding: utf-8 -*-
# @Time    : 11/15/20 2:43 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import argparse
ex_id = 0
ex_dir = "ex"+str(ex_id)

parser = argparse.ArgumentParser(description="Run a basic UNet segmentation baseline.")
parser.add_argument("--mode", type=str, default="infer", choices=("train", "infer"), help="mode of workflow")
parser.add_argument("--data_folder", type=str, default="", help="training data folder")
parser.add_argument("--model_folder", type=str, default="models/"+ex_dir, help="model folder")
parser.add_argument("--result_folder", type=str, default="results/"+ex_dir, help="model folder")

parser.add_argument("--patch_xy", type=int, default="192", help="patch size along x and y axis")
parser.add_argument("--patch_z", type=int, default="16", help="patch size along z axis")
parser.add_argument("--space_xy", type=float, default="1.25", help="spacing along x and y axis")
parser.add_argument("--space_z", type=float, default="5", help="spacing along z axis")

args = parser.parse_args()
