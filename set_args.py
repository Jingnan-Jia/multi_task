# -*- coding: utf-8 -*-
# @Time    : 11/15/20 2:43 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import argparse
ex_id = '2_0'
ex_dir = "ex"+str(ex_id)

parser = argparse.ArgumentParser(description="Run a basic UNet segmentation baseline.")
parser.add_argument("--mode", type=str, default="train", choices=("train", "infer"), help="mode of workflow")
parser.add_argument("--data_folder", type=str, default="COVID-19-20_v2/Train", help="training data folder")
parser.add_argument("--model_folder", type=str, default="models/"+ex_dir, help="model folder")
parser.add_argument("--result_folder", type=str, default="results/"+ex_dir, help="model folder")

parser.add_argument("--patch_xy", type=int, default=256, help="patch size along x and y axis")
parser.add_argument("--patch_z", type=int, default=16, help="patch size along z axis")
parser.add_argument("--space_xy", type=float, default=0.77, help="spacing along x and y axis")
parser.add_argument("--space_z", type=float, default=5, help="spacing along z axis")
parser.add_argument("--epochs", type=int, default=205, help="max epoch number")

parser.add_argument("--ld_model", type=str, default="models/ex2", help="loaded model folder")


args = parser.parse_args()
