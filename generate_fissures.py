# -*- coding: utf-8 -*-
# @Time    : 2/8/21 1:50 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

from futils.generate_fissure_from_masks import gntFissure

gntFissure(mypath.pred_path("valid", sub_dir=sub_dir, biggest_5_lobe=biggest_5_lobe), radiusValue=1, workers=10)
