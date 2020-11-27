# -*- coding: utf-8 -*-
# @Time    : 11/15/20 2:43 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import argparse
ex_id = '14'
ex_dir = "ex"+str(ex_id)

parser = argparse.ArgumentParser(description="Run a basic UNet segmentation baseline.")
parser.add_argument("--mode", type=str, default="infer", choices=("train", "infer"), help="mode of workflow")
parser.add_argument("--data_folder", type=str, default="COVID-19-20_v2/Validation", help="data folder")
#COVID-19-20_v2/Train  /data/jjia/all_ct_lesion_699
parser.add_argument("--model_folder", type=str, default="models/"+ex_dir, help="model folder")
#"models/"+ex_dir  #"models/lung/1606324643_742"
parser.add_argument("--result_folder", type=str, default="results/"+ex_dir, help=" folder")

parser.add_argument("--patch_xy", type=int, default=256, help="patch size along x and y axis")
parser.add_argument("--patch_z", type=int, default=16, help="patch size along z axis")

parser.add_argument("--space_xy", type=float, default=0.77, help="spacing along x and y axis")
parser.add_argument("--space_z", type=float, default=5, help="spacing along z axis")
parser.add_argument("--epochs", type=int, default=500, help="max epoch number")
parser.add_argument("--boost_5", type=int, default=3, help="max epoch number")

parser.add_argument("--ld_model", type=str, default="models/lung/1606324643_742", help="loaded model folder")
parser.add_argument('--pps', help='patches_per_scan', type=int, default=4)
parser.add_argument('--batch_size', help='batch_size', type=int, default=1)


args = parser.parse_args()
