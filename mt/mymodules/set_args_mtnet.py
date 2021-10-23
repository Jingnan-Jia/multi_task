# -*- coding: utf-8 -*-
# @Time    : 11/15/20 2:43 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import argparse

def get_args() -> argparse.Namespace:
    """Return arguments for multi-task learning. """

    parser = argparse.ArgumentParser(description="Run multi-task UNet segmentation.")

    # model_choices = ["net_lobe", "net_vessel", "net_recon", "net_lesion"]
    parser.add_argument('--mode', choices=("train", "infer"), help='main model ', type=str, default='train')
    parser.add_argument('--train_mode', choices=("stepbystep", "onetime"), help='main model ', type=str, default='stepbystep')
    parser.add_argument('--net_names', help='model names', type=str,
                        choices=("net_lobe_itgt", "net_lobe",
                                 "net_vessel_itgt", "net_vessel",
                                 "net_airway_itgt", "net_airway",
                                 "net_lung_itgt", "net_lung",
                                 "net_lesion_itgt", "net_lesion"
                                 "net_recon"
                                 ),default='net_lobe')
    parser.add_argument('--main_net_name', help='main model ', type=str, default='net_lobe')
    parser.add_argument('--data_path', help='change main task will change it ', type=str, default='data_ori_space')  # data_xy77_z5
    parser.add_argument('--loss', help='loss function', choices=('dice', 'CE', 'dice_CE', 'weighted_dice',
                                                                 'weighted_CE_fnfp', 'weighted_CE_fn'),
                        type=str, default='dice')  # balanced_dice, balanced_ce

    parser.add_argument('--pps', help='patches_per_scan', type=int, default=10)
    parser.add_argument('--amp', help='amp', type=bool, default=True)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=1)
    parser.add_argument('--base', help='base', type=int, default=1)
    parser.add_argument("--patch_xy", type=int, default=192, help="patch size along x and y axis")
    parser.add_argument("--patch_z", type=int, default=96, help="patch size along z axis")
    parser.add_argument('--ad_lr', help='adaptive learning rate', type=int, default=0)
    parser.add_argument('--ratio_norm_gradients', help='ratio of norm of gradients to main net', type=float, default=0)
    parser.add_argument('--fluent_ds', help='fluent_ds', type=int, default=1)
    parser.add_argument('--save_w', help='save weights magnitude', type=int, default=0)


    parser.add_argument('-step_nb', '--step_nb', help='training step', type=int, default=144001)  # 144001
    parser.add_argument('--valid_period1', help='valid_period', type=int, default=1)
    parser.add_argument('--valid_period2', help='valid_period', type=int, default=2)
    parser.add_argument('--cache', help=' cache data or not', type=int, default=1)
    parser.add_argument('--smartcache', help='smart cache data', type=int, default=0)
    parser.add_argument('-fat', '--fat', help='focus_alt_train', type=int, default=1)

    parser.add_argument('-pad', '--pad', help='padding number outside original image', type=int, default=0)
    parser.add_argument('-cntd_pts', '--cntd_pts', help='connected parts for postprocessing', type=int, default=0)

    # learning rate, normally lr of main model is greater than others
    parser.add_argument('-lr_ls', '--lr_ls', type=float, default=0.00001)
    parser.add_argument('-lr_lb', '--lr_lb', type=float, default=0.0001)
    parser.add_argument('-lr_vs', '--lr_vs', type=float, default=0.00001)
    parser.add_argument('-lr_aw', '--lr_aw', type=float, default=0.00001)
    parser.add_argument('-lr_lu', '--lr_lu', type=float, default=0.00001)
    parser.add_argument('-lr_rc', '--lr_rc', type=float, default=0.00001)


    # Number of Deep Supervisors
    parser.add_argument('--ds_ls', type=int, default=0)
    parser.add_argument('--ds_lb', type=int, default=0)
    parser.add_argument('--ds_vs', type=int, default=0)
    parser.add_argument('--ds_aw', type=int, default=0)
    parser.add_argument('--ds_lu', type=int, default=0)
    parser.add_argument('--ds_rc', type=int, default=0)

    # target spacing along (x, y) and z, format: m_n
    parser.add_argument('--tsp_ls', type=str, default='1.4_2.5')  # space along z of SSc is 0.3, of EXACT09 is 0.8
    parser.add_argument('--tsp_lb', type=str, default='1.4_2.5')  # 0.64_0.6
    parser.add_argument('--tsp_vs', type=str, default='1.4_2.5')
    parser.add_argument('--tsp_aw', type=str, default='1.4_2.5')
    parser.add_argument('--tsp_lu', type=str, default='1.4_2.5')
    parser.add_argument('--tsp_rc', type=str, default='1.4_2.5')

    # number of training images, 0 means "all"
    parser.add_argument('--tr_nb_ls', type=int, default=0)
    parser.add_argument('--tr_nb_lb', type=int, default=0)
    parser.add_argument('--tr_nb_vs', type=int, default=0)
    parser.add_argument('--tr_nb_aw', type=int, default=0)
    parser.add_argument('--tr_nb_lu', type=int, default=0)
    parser.add_argument('--tr_nb_rc', type=int, default=0)

    # number of training images, 0 means "all"
    parser.add_argument('--sub_dir_ls', type=str, default='Covid_lesion')
    parser.add_argument('--sub_dir_lb', type=str, default='GLUCOLD')
    parser.add_argument('--sub_dir_vs', type=str, default='SSc')
    parser.add_argument('--sub_dir_aw', type=str, default='None')
    parser.add_argument('--sub_dir_lu', type=str, default='None')
    parser.add_argument('--sub_dir_rc', type=str, default='LUNA16')  # or all_ct_lesion_699, LUNA16

    # name of loaded trained model for single-task net
    parser.add_argument('--ld_ls', type=str, default='')
    parser.add_argument('--ld_lb', type=str, default='')  # 1631188707_500
    parser.add_argument('--ld_vs', type=str, default='')
    parser.add_argument('--ld_aw', type=str, default='')
    parser.add_argument('--ld_lu', type=str, default='')
    parser.add_argument('--ld_rc', type=str, default='')

    # name of loaded trained model for single-task net
    parser.add_argument('--infer_data_dir', type=str, default="/data/jjia/multi_task/mt/scripts/data/data_ori_space/lobe/valid")

    #'/data/jjia/monai/data_ori_space/lola11'
    #/data/jjia/monai/data_xy77_z5/lesion
    #/data/jjia/monai/COVID-19-20_v2/Train
    # /data/jjia/monai/COVID-19-20_v2/Validation
    #/data/jjia/monai/data_ori_space/lobe
    #/data/jjia/mt/data/lobe/valid/ori_ct/LOLA11
    #/data/jjia/monai/data_xy77_z5/lesion/valid_6samples
    args = parser.parse_args()

    return args
