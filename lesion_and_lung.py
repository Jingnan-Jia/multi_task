# -*- coding: utf-8 -*-
# @Time    : 12/10/20 12:39 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import jjnutils.util as cu

# from torch.utils.tensorboard import SummaryWriter

# from monai.handlers import CheckpointSaver, MeanDice, StatsHandler, ValidationHandler

if __name__ == '__main__':
    pass

else:
    pass

lesion_dir = "/data/jjia/monai/models/lesion/1606762984_399/infer_pred/COVID-19-20_TestSet/to_submit_multitask_deepsupervision"
lesion_names = cu.get_all_ct_names(lesion_dir)
lung_dir = "/data/jjia/monai/models/body_masks"
    # "/data/jjia/monai/models/lung/1607203020_836/infer_pred/COVID-19-20_TestSet/to_submit_testset_lung/biggest_parts"
lung_names = cu.get_all_ct_names(lung_dir)

for lesion_name, lung_name in zip(lesion_names, lung_names):
    print(f"lesion name: {lesion_name}")
    print(f"lung name: {lung_name}")
    lesion, origin, spacing = cu.load_itk(lesion_name)
    lung, _, _ = cu.load_itk(lung_name)
    new_img = lesion * lung

    write_fpath = "/data/jjia/monai/lesion_filtered_body/" + lesion_name.split("/")[-1]
    cu.save_itk(write_fpath, new_img, origin, spacing)
    print(f"save successfully at: {write_fpath}")


