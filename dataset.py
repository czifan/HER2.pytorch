from torch.utils.data import Dataset
from PIL import Image 
import torch 
import numpy as np 
import os 
from torchvision.transforms import transforms
from glob import glob
import numpy as np
from datetime import datetime
import pandas as pd 
import json 
from datetime import datetime

# mean_std_dict = {
# "PKUCancer": {
#     "a_source": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "a_target_LN1": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "v_source": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "v_target_LN1": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "l_target_LN1": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "a_target_LN2": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "a_target_Liver1": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "a_target_Liver2": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "v_target_LN2": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "v_target_Liver1": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "v_target_Liver2": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "a_target_L_Aden": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "a_target_R_Aden": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "v_target_L_Aden": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "v_target_R_Aden": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "a_target_Peritoneum": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "v_target_Peritoneum": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "v_target_Soft": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "v_target_Other": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "l_target_LN2": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "a_target_Bone1": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "a_target_Bone2": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "v_target_Bone1": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "v_target_Bone2": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "a_target_Spleen": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "v_target_Spleen": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "a_target_Peritoneum2": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "v_target_Peritoneum2": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
#     "a_target_Soft": [[0.330,0.330,0.330], [0.211,0.211,0.211]],
# },
# "Drum": {
#     "a_source": [[0.401,0.401,0.401], [0.200,0.200,0.200]],
#     "v_source": [[0.401,0.401,0.401], [0.200,0.200,0.200]],
#     "a_target_Liver1": [[0.401,0.401,0.401], [0.200,0.200,0.200]],
#     "a_target_Liver2": [[0.401,0.401,0.401], [0.200,0.200,0.200]],
#     "v_target_Liver2": [[0.401,0.401,0.401], [0.200,0.200,0.200]],
#     "v_target_Liver1": [[0.401,0.401,0.401], [0.200,0.200,0.200]],
#     "a_target_LN1": [[0.401,0.401,0.401], [0.200,0.200,0.200]],
#     "a_target_LN2": [[0.401,0.401,0.401], [0.200,0.200,0.200]],
#     "v_target_LN1": [[0.401,0.401,0.401], [0.200,0.200,0.200]],
#     "v_target_LN2": [[0.401,0.401,0.401], [0.200,0.200,0.200]],
#     "v_target_Peritoneum": [[0.401,0.401,0.401], [0.200,0.200,0.200]],
#     "a_target_Peritoneum": [[0.401,0.401,0.401], [0.200,0.200,0.200]],
# },
# "Ruijin": {
#     "a_target_L_Aden": [[0.430,0.430,0.430], [0.205,0.205,0.205]],
#     "a_target_LN1": [[0.430,0.430,0.430], [0.205,0.205,0.205]],
#     "a_source": [[0.430,0.430,0.430], [0.205,0.205,0.205]],
#     "v_target_L_Aden": [[0.430,0.430,0.430], [0.205,0.205,0.205]],
#     "v_source": [[0.430,0.430,0.430], [0.205,0.205,0.205]],
#     "a_target_LN2": [[0.430,0.430,0.430], [0.205,0.205,0.205]],
#     "v_target_LN1": [[0.430,0.430,0.430], [0.205,0.205,0.205]],
#     "v_target_LN2": [[0.430,0.430,0.430], [0.205,0.205,0.205]],
#     "v_target_Liver2": [[0.430,0.430,0.430], [0.205,0.205,0.205]],
#     "v_target_Liver1": [[0.430,0.430,0.430], [0.205,0.205,0.205]],
# },
# "ZDYFY": {
#     "a_target_LN1": [[0.372,0.372,0.372], [0.213,0.213,0.213]],
#     "a_source": [[0.372,0.372,0.372], [0.213,0.213,0.213]],
#     "v_source": [[0.372,0.372,0.372], [0.213,0.213,0.213]],
#     "a_target_LN2": [[0.372,0.372,0.372], [0.213,0.213,0.213]],
#     "v_target_LN1": [[0.372,0.372,0.372], [0.213,0.213,0.213]],
#     "v_target_LN2": [[0.372,0.372,0.372], [0.213,0.213,0.213]],
#     "a_target_L_Aden": [[0.372,0.372,0.372], [0.213,0.213,0.213]],
#     "v_target_L_Aden": [[0.372,0.372,0.372], [0.213,0.213,0.213]],
#     "a_target_Liver1": [[0.372,0.372,0.372], [0.213,0.213,0.213]],
#     "a_target_Liver2": [[0.372,0.372,0.372], [0.213,0.213,0.213]],
#     "v_target_Liver2": [[0.372,0.372,0.372], [0.213,0.213,0.213]],
#     "v_target_Liver1": [[0.372,0.372,0.372], [0.213,0.213,0.213]],
#     "l_target_LN1": [[0.372,0.372,0.372], [0.213,0.213,0.213]],
#     "l_target_LN2": [[0.372,0.372,0.372], [0.213,0.213,0.213]],
#     "v_target_Peritoneum": [[0.372,0.372,0.372], [0.213,0.213,0.213]],
#     "a_target_Peritoneum": [[0.372,0.372,0.372], [0.213,0.213,0.213]],
# },
# "HPD1": {
#     "a_target_LN1": [[0.325,0.325,0.325], [0.221,0.221,0.221]],
#     "a_source": [[0.325,0.325,0.325], [0.221,0.221,0.221]],
#     "v_source": [[0.325,0.325,0.325], [0.221,0.221,0.221]],
#     "a_target_LN2": [[0.325,0.325,0.325], [0.221,0.221,0.221]],
#     "v_target_LN1": [[0.325,0.325,0.325], [0.221,0.221,0.221]],
#     "v_target_LN2": [[0.325,0.325,0.325], [0.221,0.221,0.221]],
#     "v_target_Liver2": [[0.325,0.325,0.325], [0.221,0.221,0.221]],
#     "v_target_Liver1": [[0.325,0.325,0.325], [0.221,0.221,0.221]],
#     "a_target_L_Aden": [[0.325,0.325,0.325], [0.221,0.221,0.221]],
#     "v_target_L_Aden": [[0.325,0.325,0.325], [0.221,0.221,0.221]],
#     "v_target_Other": [[0.325,0.325,0.325], [0.221,0.221,0.221]],
#     "l_target_LN1": [[0.325,0.325,0.325], [0.221,0.221,0.221]],
#     "l_target_LN2": [[0.325,0.325,0.325], [0.221,0.221,0.221]],
#     "v_target_Peritoneum": [[0.325,0.325,0.325], [0.221,0.221,0.221]],
#     "a_target_Other": [[0.325,0.325,0.325], [0.221,0.221,0.221]],
#     "a_target_R_Aden": [[0.325,0.325,0.325], [0.221,0.221,0.221]],
#     "v_target_R_Aden": [[0.325,0.325,0.325], [0.221,0.221,0.221]],
#     "a_target_Peritoneum": [[0.325,0.325,0.325], [0.221,0.221,0.221]],
#     "v_target_Peritoneum2": [[0.325,0.325,0.325], [0.221,0.221,0.221]],
#     "a_target_Peritoneum2": [[0.325,0.325,0.325], [0.221,0.221,0.221]],
#     "a_target_Spleen": [[0.325,0.325,0.325], [0.221,0.221,0.221]],
#     "v_target_Spleen": [[0.325,0.325,0.325], [0.221,0.221,0.221]],
#     "a_target_Liver1": [[0.325,0.325,0.325], [0.221,0.221,0.221]],
#     "a_target_Liver2": [[0.325,0.325,0.325], [0.221,0.221,0.221]],
# },
# }

mean_std_dict = {
"PKUCancer": {
    "a_source": [[0.250,0.250,0.250], [0.273,0.273,0.273]],
    "a_target_LN1": [[0.378,0.378,0.378], [0.234,0.234,0.234]],
    "v_source": [[0.254,0.254,0.254], [0.273,0.273,0.273]],
    "v_target_LN1": [[0.389,0.389,0.389], [0.228,0.228,0.228]],
    "l_target_LN1": [[0.293,0.293,0.293], [0.223,0.223,0.223]],
    "a_target_LN2": [[0.395,0.395,0.395], [0.232,0.232,0.232]],
    "a_target_Liver1": [[0.482,0.482,0.482], [0.206,0.206,0.206]],
    "a_target_Liver2": [[0.489,0.489,0.489], [0.210,0.210,0.210]],
    "v_target_LN2": [[0.401,0.401,0.401], [0.222,0.222,0.222]],
    "v_target_Liver1": [[0.525,0.525,0.525], [0.217,0.217,0.217]],
    "v_target_Liver2": [[0.539,0.539,0.539], [0.220,0.220,0.220]],
    "a_target_L_Aden": [[0.413,0.413,0.413], [0.237,0.237,0.237]],
    "a_target_R_Aden": [[0.436,0.436,0.436], [0.238,0.238,0.238]],
    "v_target_L_Aden": [[0.383,0.383,0.383], [0.258,0.258,0.258]],
    "v_target_R_Aden": [[0.466,0.466,0.466], [0.219,0.219,0.219]],
    "a_target_Peritoneum": [[0.285,0.285,0.285], [0.227,0.227,0.227]],
    "v_target_Peritoneum": [[0.292,0.292,0.292], [0.232,0.232,0.232]],
    "v_target_Soft": [[0.303,0.303,0.303], [0.233,0.233,0.233]],
    "v_target_Other": [[0.342,0.342,0.342], [0.301,0.301,0.301]],
    "l_target_LN2": [[0.190,0.190,0.190], [0.224,0.224,0.224]],
    "a_target_Bone1": [[0.765,0.765,0.765], [0.297,0.297,0.297]],
    "a_target_Bone2": [[0.631,0.631,0.631], [0.379,0.379,0.379]],
    "v_target_Bone1": [[0.653,0.653,0.653], [0.396,0.396,0.396]],
    "v_target_Bone2": [[0.746,0.746,0.746], [0.300,0.300,0.300]],
    "a_target_Spleen": [[0.609,0.609,0.609], [0.111,0.111,0.111]],
    "v_target_Spleen": [[0.474,0.474,0.474], [0.286,0.286,0.286]],
    "a_target_Peritoneum2": [[0.343,0.343,0.343], [0.245,0.245,0.245]],
    "v_target_Peritoneum2": [[0.348,0.348,0.348], [0.248,0.248,0.248]],
    "a_target_Soft": [[0.427,0.427,0.427], [0.243,0.243,0.243]],
},
"Drum": {
    "a_source": [[0.381,0.381,0.381], [0.253,0.253,0.253]],
    "v_source": [[0.350,0.350,0.350], [0.271,0.271,0.271]],
    "a_target_Liver1": [[0.527,0.527,0.527], [0.200,0.200,0.200]],
    "a_target_Liver2": [[0.547,0.547,0.547], [0.182,0.182,0.182]],
    "v_target_Liver2": [[0.601,0.601,0.601], [0.137,0.137,0.137]],
    "v_target_Liver1": [[0.557,0.557,0.557], [0.185,0.185,0.185]],
    "a_target_LN1": [[0.418,0.418,0.418], [0.223,0.223,0.223]],
    "a_target_LN2": [[0.448,0.448,0.448], [0.228,0.228,0.228]],
    "v_target_LN1": [[0.414,0.414,0.414], [0.236,0.236,0.236]],
    "v_target_LN2": [[0.439,0.439,0.439], [0.234,0.234,0.234]],
    "v_target_Peritoneum": [[0.303,0.303,0.303], [0.132,0.132,0.132]],
    "a_target_Peritoneum": [[0.222,0.222,0.222], [0.154,0.154,0.154]],
},
"Ruijin": {
    "a_target_L_Aden": [[0.425,0.425,0.425], [0.235,0.235,0.235]],
    "a_target_LN1": [[0.436,0.436,0.436], [0.198,0.198,0.198]],
    "a_source": [[0.280,0.280,0.280], [0.266,0.266,0.266]],
    "v_target_L_Aden": [[0.587,0.587,0.587], [0.195,0.195,0.195]],
    "v_source": [[0.309,0.309,0.309], [0.293,0.293,0.293]],
    "a_target_LN2": [[0.515,0.515,0.515], [0.259,0.259,0.259]],
    "v_target_LN1": [[0.442,0.442,0.442], [0.251,0.251,0.251]],
    "v_target_LN2": [[0.602,0.602,0.602], [0.231,0.231,0.231]],
    "v_target_Liver2": [[0.650,0.650,0.650], [0.061,0.061,0.061]],
    "v_target_Liver1": [[0.606,0.606,0.606], [0.195,0.195,0.195]],
},
"ZDYFY": {
    "a_target_LN1": [[0.407,0.407,0.407], [0.234,0.234,0.234]],
    "a_source": [[0.346,0.346,0.346], [0.262,0.262,0.262]],
    "v_source": [[0.361,0.361,0.361], [0.263,0.263,0.263]],
    "a_target_LN2": [[0.360,0.360,0.360], [0.264,0.264,0.264]],
    "v_target_LN1": [[0.399,0.399,0.399], [0.245,0.245,0.245]],
    "v_target_LN2": [[0.416,0.416,0.416], [0.247,0.247,0.247]],
    "a_target_L_Aden": [[0.393,0.393,0.393], [0.252,0.252,0.252]],
    "v_target_L_Aden": [[0.401,0.401,0.401], [0.268,0.268,0.268]],
    "a_target_Liver1": [[0.470,0.470,0.470], [0.213,0.213,0.213]],
    "a_target_Liver2": [[0.470,0.470,0.470], [0.208,0.208,0.208]],
    "v_target_Liver2": [[0.515,0.515,0.515], [0.217,0.217,0.217]],
    "v_target_Liver1": [[0.525,0.525,0.525], [0.209,0.209,0.209]],
    "l_target_LN1": [[0.363,0.363,0.363], [0.250,0.250,0.250]],
    "l_target_LN2": [[0.248,0.248,0.248], [0.281,0.281,0.281]],
    "v_target_Peritoneum": [[0.213,0.213,0.213], [0.219,0.219,0.219]],
    "a_target_Peritoneum": [[0.183,0.183,0.183], [0.206,0.206,0.206]],
},
"HPD1": {
    "a_target_LN1": [[0.397,0.397,0.397], [0.232,0.232,0.232]],
    "a_source": [[0.253,0.253,0.253], [0.276,0.276,0.276]],
    "v_source": [[0.257,0.257,0.257], [0.276,0.276,0.276]],
    "a_target_LN2": [[0.394,0.394,0.394], [0.245,0.245,0.245]],
    "v_target_LN1": [[0.406,0.406,0.406], [0.227,0.227,0.227]],
    "v_target_LN2": [[0.394,0.394,0.394], [0.239,0.239,0.239]],
    "v_target_Liver2": [[0.491,0.491,0.491], [0.247,0.247,0.247]],
    "v_target_Liver1": [[0.519,0.519,0.519], [0.230,0.230,0.230]],
    "a_target_L_Aden": [[0.454,0.454,0.454], [0.246,0.246,0.246]],
    "v_target_L_Aden": [[0.424,0.424,0.424], [0.249,0.249,0.249]],
    "v_target_Other": [[0.390,0.390,0.390], [0.247,0.247,0.247]],
    "l_target_LN1": [[0.278,0.278,0.278], [0.274,0.274,0.274]],
    "l_target_LN2": [[0.188,0.188,0.188], [0.284,0.284,0.284]],
    "v_target_Peritoneum": [[0.326,0.326,0.326], [0.228,0.228,0.228]],
    "a_target_Other": [[0.426,0.426,0.426], [0.200,0.200,0.200]],
    "a_target_R_Aden": [[0.564,0.564,0.564], [0.185,0.185,0.185]],
    "v_target_R_Aden": [[0.555,0.555,0.555], [0.208,0.208,0.208]],
    "a_target_Peritoneum": [[0.354,0.354,0.354], [0.218,0.218,0.218]],
    "v_target_Peritoneum2": [[0.265,0.265,0.265], [0.232,0.232,0.232]],
    "a_target_Peritoneum2": [[0.216,0.216,0.216], [0.234,0.234,0.234]],
    "a_target_Spleen": [[0.412,0.412,0.412], [0.266,0.266,0.266]],
    "v_target_Spleen": [[0.460,0.460,0.460], [0.262,0.262,0.262]],
    "a_target_Liver1": [[0.514,0.514,0.514], [0.230,0.230,0.230]],
    "a_target_Liver2": [[0.428,0.428,0.428], [0.269,0.269,0.269]],
},
}

# mean_std_dict = {
# "PKUCancer": {
#     "a_source": [[0.207,0.207,0.207], [0.236,0.236,0.236]],
#     "a_target_LN1": [[0.325,0.325,0.325], [0.204,0.204,0.204]],
#     "v_source": [[0.211,0.211,0.211], [0.234,0.234,0.234]],
#     "v_target_LN1": [[0.334,0.334,0.334], [0.194,0.194,0.194]],
#     "l_target_LN1": [[0.245,0.245,0.245], [0.200,0.200,0.200]],
#     "a_target_LN2": [[0.342,0.342,0.342], [0.207,0.207,0.207]],
#     "a_target_Liver1": [[0.441,0.441,0.441], [0.188,0.188,0.188]],
#     "a_target_Liver2": [[0.450,0.450,0.450], [0.191,0.191,0.191]],
#     "v_target_LN2": [[0.343,0.343,0.343], [0.192,0.192,0.192]],
#     "v_target_Liver1": [[0.497,0.497,0.497], [0.202,0.202,0.202]],
#     "v_target_Liver2": [[0.510,0.510,0.510], [0.206,0.206,0.206]],
#     "a_target_L_Aden": [[0.367,0.367,0.367], [0.205,0.205,0.205]],
#     "a_target_R_Aden": [[0.396,0.396,0.396], [0.213,0.213,0.213]],
#     "v_target_L_Aden": [[0.340,0.340,0.340], [0.225,0.225,0.225]],
#     "v_target_R_Aden": [[0.427,0.427,0.427], [0.192,0.192,0.192]],
#     "a_target_Peritoneum": [[0.224,0.224,0.224], [0.186,0.186,0.186]],
#     "v_target_Peritoneum": [[0.228,0.228,0.228], [0.187,0.187,0.187]],
#     "v_target_Soft": [[0.249,0.249,0.249], [0.190,0.190,0.190]],
#     "v_target_Other": [[0.287,0.287,0.287], [0.269,0.269,0.269]],
#     "l_target_LN2": [[0.169,0.169,0.169], [0.229,0.229,0.229]],
#     "a_target_Bone1": [[0.785,0.785,0.785], [0.291,0.291,0.291]],
#     "a_target_Bone2": [[0.640,0.640,0.640], [0.383,0.383,0.383]],
#     "v_target_Bone1": [[0.670,0.670,0.670], [0.402,0.402,0.402]],
#     "v_target_Bone2": [[0.751,0.751,0.751], [0.306,0.306,0.306]],
#     "a_target_Spleen": [[0.597,0.597,0.597], [0.080,0.080,0.080]],
#     "v_target_Spleen": [[0.443,0.443,0.443], [0.266,0.266,0.266]],
#     "a_target_Peritoneum2": [[0.275,0.275,0.275], [0.205,0.205,0.205]],
#     "v_target_Peritoneum2": [[0.287,0.287,0.287], [0.212,0.212,0.212]],
#     "a_target_Soft": [[0.374,0.374,0.374], [0.212,0.212,0.212]],
# },
# "Drum": {
#     "a_source": [[0.339,0.339,0.339], [0.227,0.227,0.227]],
#     "v_source": [[0.306,0.306,0.306], [0.240,0.240,0.240]],
#     "a_target_Liver1": [[0.495,0.495,0.495], [0.188,0.188,0.188]],
#     "a_target_Liver2": [[0.513,0.513,0.513], [0.165,0.165,0.165]],
#     "v_target_Liver2": [[0.569,0.569,0.569], [0.118,0.118,0.118]],
#     "v_target_Liver1": [[0.529,0.529,0.529], [0.167,0.167,0.167]],
#     "a_target_LN1": [[0.374,0.374,0.374], [0.197,0.197,0.197]],
#     "a_target_LN2": [[0.415,0.415,0.415], [0.205,0.205,0.205]],
#     "v_target_LN1": [[0.366,0.366,0.366], [0.206,0.206,0.206]],
#     "v_target_LN2": [[0.395,0.395,0.395], [0.208,0.208,0.208]],
#     "v_target_Peritoneum": [[0.234,0.234,0.234], [0.108,0.108,0.108]],
#     "a_target_Peritoneum": [[0.148,0.148,0.148], [0.105,0.105,0.105]],
# },
# "Ruijin": {
#     "a_target_L_Aden": [[0.388,0.388,0.388], [0.214,0.214,0.214]],
#     "a_target_LN1": [[0.394,0.394,0.394], [0.168,0.168,0.168]],
#     "a_source": [[0.229,0.229,0.229], [0.218,0.218,0.218]],
#     "v_target_L_Aden": [[0.595,0.595,0.595], [0.170,0.170,0.170]],
#     "v_source": [[0.286,0.286,0.286], [0.271,0.271,0.271]],
#     "a_target_LN2": [[0.500,0.500,0.500], [0.248,0.248,0.248]],
#     "v_target_LN1": [[0.437,0.437,0.437], [0.243,0.243,0.243]],
#     "v_target_LN2": [[0.623,0.623,0.623], [0.217,0.217,0.217]],
#     "v_target_Liver2": [[0.611,0.611,0.611], [0.040,0.040,0.040]],
#     "v_target_Liver1": [[0.577,0.577,0.577], [0.181,0.181,0.181]],
# },
# "ZDYFY": {
#     "a_target_LN1": [[0.361,0.361,0.361], [0.209,0.209,0.209]],
#     "a_source": [[0.300,0.300,0.300], [0.232,0.232,0.232]],
#     "v_source": [[0.318,0.318,0.318], [0.232,0.232,0.232]],
#     "a_target_LN2": [[0.315,0.315,0.315], [0.236,0.236,0.236]],
#     "v_target_LN1": [[0.352,0.352,0.352], [0.214,0.214,0.214]],
#     "v_target_LN2": [[0.367,0.367,0.367], [0.218,0.218,0.218]],
#     "a_target_L_Aden": [[0.349,0.349,0.349], [0.220,0.220,0.220]],
#     "v_target_L_Aden": [[0.365,0.365,0.365], [0.238,0.238,0.238]],
#     "a_target_Liver1": [[0.426,0.426,0.426], [0.193,0.193,0.193]],
#     "a_target_Liver2": [[0.427,0.427,0.427], [0.187,0.187,0.187]],
#     "v_target_Liver2": [[0.482,0.482,0.482], [0.201,0.201,0.201]],
#     "v_target_Liver1": [[0.496,0.496,0.496], [0.189,0.189,0.189]],
#     "l_target_LN1": [[0.345,0.345,0.345], [0.239,0.239,0.239]],
#     "l_target_LN2": [[0.238,0.238,0.238], [0.283,0.283,0.283]],
#     "v_target_Peritoneum": [[0.148,0.148,0.148], [0.156,0.156,0.156]],
#     "a_target_Peritoneum": [[0.122,0.122,0.122], [0.144,0.144,0.144]],
# },
# "HPD1": {
#     "a_target_LN1": [[0.347,0.347,0.347], [0.206,0.206,0.206]],
#     "a_source": [[0.211,0.211,0.211], [0.239,0.239,0.239]],
#     "v_source": [[0.215,0.215,0.215], [0.239,0.239,0.239]],
#     "a_target_LN2": [[0.337,0.337,0.337], [0.212,0.212,0.212]],
#     "v_target_LN1": [[0.354,0.354,0.354], [0.198,0.198,0.198]],
#     "v_target_LN2": [[0.337,0.337,0.337], [0.201,0.201,0.201]],
#     "v_target_Liver2": [[0.453,0.453,0.453], [0.232,0.232,0.232]],
#     "v_target_Liver1": [[0.484,0.484,0.484], [0.215,0.215,0.215]],
#     "a_target_L_Aden": [[0.419,0.419,0.419], [0.229,0.229,0.229]],
#     "v_target_L_Aden": [[0.387,0.387,0.387], [0.228,0.228,0.228]],
#     "v_target_Other": [[0.323,0.323,0.323], [0.224,0.224,0.224]],
#     "l_target_LN1": [[0.259,0.259,0.259], [0.280,0.280,0.280]],
#     "l_target_LN2": [[0.179,0.179,0.179], [0.291,0.291,0.291]],
#     "v_target_Peritoneum": [[0.263,0.263,0.263], [0.192,0.192,0.192]],
#     "a_target_Other": [[0.327,0.327,0.327], [0.156,0.156,0.156]],
#     "a_target_R_Aden": [[0.570,0.570,0.570], [0.170,0.170,0.170]],
#     "v_target_R_Aden": [[0.564,0.564,0.564], [0.196,0.196,0.196]],
#     "a_target_Peritoneum": [[0.281,0.281,0.281], [0.179,0.179,0.179]],
#     "v_target_Peritoneum2": [[0.206,0.206,0.206], [0.184,0.184,0.184]],
#     "a_target_Peritoneum2": [[0.164,0.164,0.164], [0.181,0.181,0.181]],
#     "a_target_Spleen": [[0.386,0.386,0.386], [0.244,0.244,0.244]],
#     "v_target_Spleen": [[0.431,0.431,0.431], [0.240,0.240,0.240]],
#     "a_target_Liver1": [[0.480,0.480,0.480], [0.211,0.211,0.211]],
#     "a_target_Liver2": [[0.404,0.404,0.404], [0.253,0.253,0.253]],
# },
# }

class SurvDataset(Dataset):
    def __init__(self,
                data_dir,
                split_file,
                split,
                anno_file,
                printer=print,
                median=365,
                num_time=1,
                num_lesion=1,
                discard=True,
                target_size=224,
                black_lst=['a_target_Liver1.jpg', 'a_target_Liver2.jpg'],
                prefix='0',
                strong_aug=True,
                ):
        super().__init__()
        self.data_dir = data_dir 
        self.median = median 
        #self.median = round(median / 30, 0)
        self.split = split 
        self.num_time = num_time 
        self.num_lesion = num_lesion
        self.black_lst = black_lst
        self.target_size = target_size
        self.prefix = prefix
        self.discard = discard

        anno_data = pd.read_csv(anno_file).values
        self.pid_to_os = {str(d[1]):int(d[4]) for d in anno_data}
        self.pid_to_event = {str(d[1]):int(d[5]) for d in anno_data}

        # if self.discard:
        #     self.pid_to_orr = {}
        #     RECIST_data = pd.read_csv(os.path.join('RECIST', split+'.csv'))
        #     for pid, RECIST in zip(RECIST_data['pid'], RECIST_data['RECIST']):
        #         pid = str(pid)
        #         if RECIST in ['CR', 'PR']:
        #             self.pid_to_orr[pid] = 1
        #         else:
        #             self.pid_to_orr[pid] = 0
                    
        # self.hos_lst = []
        # self.pid_lst = []
        # self.os_lst = []
        # self.event_lst = []
        # self.y_lst = []
        # for line in open(split_file, 'r').readlines():
        #     hos, pid = line.strip().split(',')
        #     OS = round(self.pid_to_os[pid] / 30, 0)
        #     event = self.pid_to_event[pid]

        #     self.hos_lst.append(hos)
        #     self.pid_lst.append(pid)
        #     self.os_lst.append(OS)
        #     self.event_lst.append(event)
        #     if self.discard:
        #         self.y_lst.append(self.pid_to_orr[pid])
        #     else:
        #         self.y_lst.append(-1)

        self.hos_lst = []
        self.pid_lst = []
        self.os_lst = []
        self.event_lst = []
        self.y_lst = []
        self.median = round(1.0 * self.median / 30, 0)
        for line in open(split_file, 'r').readlines():
            hos, pid = line.strip().split(',')
            OS = self.pid_to_os[pid]
            OS = round(OS / 30, 0)
            event = self.pid_to_event[pid]
            y = int(OS > self.median)
            if discard and event == 0 and OS <= self.median:
                printer(f'discard {pid} t={OS} e={event}')
                continue
            self.hos_lst.append(hos)
            self.pid_lst.append(pid)
            self.os_lst.append(OS)
            self.event_lst.append(event)
            self.y_lst.append(y)

        self.strong_aug = strong_aug
        if self.strong_aug:
            self.transforms = transforms.Compose([
                                                transforms.Resize(target_size),
                                                #transforms.Pad(32, (0, 0, 0)),  
                                                #transforms.RandomCrop(target_size),
                                                transforms.RandomRotation(30),
                                                #transforms.RandomHorizontalFlip(p=0.5),
                                                #transforms.RandomAffine(0, (0.1, 0.1)),
                                                transforms.RandomVerticalFlip(p=1.0),
                                                transforms.ToTensor(),
                                                ])
        else:
            self.transforms = transforms.Compose([
                                                transforms.Resize(target_size),
                                                transforms.RandomVerticalFlip(p=0.0 if self.split in ['test'] else 1.0),
                                                transforms.ToTensor()
                                                ])
        self._summary(printer)

    def _summary(self, printer):
        printer(f'[{self.split}]\tLoaded {self.__len__()} samples')
        printer(f'{len(self.y_lst)-sum(self.y_lst)}({100.0*(len(self.y_lst)-sum(self.y_lst))/len(self.y_lst):.2f}%) (y=0)')
        printer(f'{sum(self.y_lst)}({100.0*sum(self.y_lst)/len(self.y_lst):.2f}%) (y=1)')
        printer('=' * 25)
    
    def __len__(self):
        assert len(self.hos_lst) == len(self.pid_lst)
        return len(self.pid_lst)

    def __getitem__(self, index):
        hos = self.hos_lst[index]
        pid = self.pid_lst[index]

        pid_data_dir = os.path.join(self.data_dir, self.prefix, pid)
        times = sorted(os.listdir(pid_data_dir))
        times = [t for t in times if not (t.startswith('.') or t.endswith('.'))]
        t0 = datetime.strptime(times[0], "%Y%m%d")
        files = [file for file in os.listdir(os.path.join(pid_data_dir, times[0])) if file not in self.black_lst]
        # files = [file for file in os.listdir(os.path.join(pid_data_dir, times[0])) if 'source' not in file and file not in self.black_lst] # only target
        # files = [file for file in os.listdir(os.path.join(pid_data_dir, times[0])) if 'source' in file and file not in self.black_lst] # only source

        if self.strong_aug:
            files = np.random.choice(files, self.num_lesion)

        #T = 5 if self.strong_aug else self.num_time
        T = 3 if self.strong_aug else self.num_time
        N = len(files)
        X = torch.zeros(T, N, 3, self.target_size, self.target_size).float() # (T, N, 3, H, W)
        M = torch.zeros(T, N).float()
        S = torch.zeros(T, N).float()
        Mean = torch.zeros(T, N, 3).float()
        Std = torch.zeros(T, N, 3).float()
        time_key_lst = []
        for ti, t in enumerate(times[:T]):
            time_key_lst.append(str((datetime.strptime(t, "%Y%m%d")-t0).days))
            #if ti and self.strong_aug and np.random.uniform(0, 1) < 0.5: continue
            for ni, file in enumerate(files):
                t_file = os.path.join(pid_data_dir, t, file)
                if os.path.isfile(t_file):
                    image = Image.fromarray(np.stack([
                        np.asarray(Image.open(t_file.replace('/0', '/-1')).convert('L')),
                        np.asarray(Image.open(t_file).convert('L')),
                        np.asarray(Image.open(t_file.replace('/0', '/1')).convert('L')),
                    ], axis=-1))
                    #image = Image.open(t_file).convert('RGB')
                    image = self.transforms(image)
                    key = os.path.basename(file).replace('.jpg', '')
                    mean = mean_std_dict[hos][key][0]
                    std = mean_std_dict[hos][key][1]
                    self.norm = transforms.Normalize(mean, std)
                    image = self.norm(image)
                    X[ti, ni] = image
                    M[ti, ni] = 1.0
                    Mean[ti, ni] = torch.FloatTensor(mean)
                    Std[ti, ni] = torch.FloatTensor(std)
                    if 'source' in t_file:
                        S[ti, ni] = 1.0
        while len(time_key_lst) < T:
            time_key_lst.append(str(730))

        return {
            'X': X,
            'M': M,
            'S': S,
            'OS': self.os_lst[index],
            'y': self.y_lst[index],
            'event': self.event_lst[index],
            'pid': self.pid_lst[index],
            'hos': self.hos_lst[index],
            'Mean': Mean,
            'Std': Std,
            #'time_key_lst': [f'{ti}F' for ti in range(T)],
            'time_key_lst': time_key_lst,
            'lesion_key_lst': [os.path.basename(file).split('.')[0] for file in files],
        }
    
class ClinicalDataset(Dataset):
    def __init__(self,
                data_dir,
                split_file,
                split,
                anno_file,
                printer=print,
                median=365,
                num_time=1,
                num_lesion=1,
                discard=True,
                target_size=224,
                black_lst=['a_target_Liver1.jpg', 'a_target_Liver2.jpg'],
                prefix='0',
                strong_aug=True,
                ):
        super().__init__()
        
        self.data_dir = data_dir 
        self.median = median
        self.split = split 
        self.num_time = num_time 
        self.num_lesion = num_lesion
        self.black_lst = black_lst
        self.target_size = target_size
        self.prefix = prefix
        self.discard = discard
        
        anno_data = pd.read_csv(anno_file).values
        self.pid_to_os = {str(d[1]):int(d[4]) for d in anno_data}
        self.pid_to_event = {str(d[1]):int(d[5]) for d in anno_data}
        
        self.hos_lst = []
        self.pid_lst = []
        self.os_lst = []
        self.event_lst = []
        self.y_lst = []
        self.median = round(1.0 * self.median / 30, 0)
        self.X_lst = []
        self.M_lst = []
        self.TimePoint_lst = []
        self.t0_lst = []
        for line in open(split_file, 'r').readlines():
            hos, pid = line.strip().split(',')
            OS = self.pid_to_os[pid]
            OS = round(OS / 30, 0)
            event = self.pid_to_event[pid]
            y = int(OS > self.median)
            if discard and event == 0 and OS <= self.median:
                printer(f'discard {pid} t={OS} e={event}')
                continue
                
            npz_file = os.path.join(data_dir, pid+'.npz')
            npz_data = np.load(npz_file, allow_pickle=True)
                
            self.hos_lst.append(hos)
            self.pid_lst.append(pid)
            self.os_lst.append(OS)
            self.event_lst.append(event)
            self.y_lst.append(y)
            self.X_lst.append(npz_data['X'])
            self.M_lst.append(npz_data['M'])
            self.TimePoint_lst.append(npz_data['TimePoint'])
            self.t0_lst.append(npz_data['t0'])
        
        self._summary(printer)

    def _summary(self, printer):
        printer(f'[{self.split}]\tLoaded {self.__len__()} samples')
        printer(f'{len(self.y_lst)-sum(self.y_lst)}({100.0*(len(self.y_lst)-sum(self.y_lst))/len(self.y_lst):.2f}%) (y=0)')
        printer(f'{sum(self.y_lst)}({100.0*sum(self.y_lst)/len(self.y_lst):.2f}%) (y=1)')
        printer('=' * 25)
        
    def __len__(self):
        return len(self.X_lst)

    def __getitem__(self, index):
        pid = self.pid_lst[index]
        X = self.X_lst[index]
        M = self.M_lst[index]
        TimePoint = self.TimePoint_lst[index]
        #print(pid, TimePoint, '+++++++++++++')
#         t0 = self.t0_lst[index]
#         time_key_lst = []
#         for i in range(0, len(TimePoint)):
#             time_key_lst.append(str((TimePoint[i]-t0).days))
#             if (TimePoint[i]-t0).days < 0:
#                 M[i, :] *= 0
#                 time_key_lst[-1] = str(0)
        t0 = TimePoint[0]
        time_key_lst = [str(0),]
        for i in range(1, len(TimePoint)):
            time_key_lst.append(str((TimePoint[i]-t0).days))
        while len(time_key_lst) < X.shape[0]:
            time_key_lst.append(str(730))
        return {
            'X': X,
            'M': M,
            'S': M,
            'OS': self.os_lst[index],
            'y': self.y_lst[index],
            'event': self.event_lst[index],
            'pid': self.pid_lst[index],
            'hos': self.hos_lst[index],
            'Mean': [0., 0., 0.],
            'Std': [0., 0., 0.],
            'time_key_lst': time_key_lst,
            'lesion_key_lst': ['LDH', 'NSE', 'CEA', 'CA125', 'CA199', 'CA724', 'AFP'],
        }
    
if __name__ == '__main__':
    dataset = ClinicalDataset('SurvData/PKCancerClinicalData', 'SurvData/Annotations/train_hos_0519.txt', 'train')
    for data in dataset:
        X = data['X']
        M = data['M']
        T = data['T']
        print(X)
        print(M)
        print(T)
    
    
    # # split data
    # all_file = 'SurvData/Annotations/PKCancerCohort.csv'
    # all_data = pd.read_csv(all_file)
    # pid_lst = [d[1] for d in all_data.values]
    # train_pid_lst = np.random.choice(pid_lst, int(2*len(pid_lst)/3), replace=False)
    # valid_pid_lst = [pid for pid in pid_lst if pid not in train_pid_lst]
    # with open('SurvData/Annotations/train.txt', 'w') as f:
    #     for pid in train_pid_lst:
    #         f.write(f'{pid}\n')
    # with open('SurvData/Annotations/valid.txt', 'w') as f:
    #     for pid in valid_pid_lst:
    #         f.write(f'{pid}\n')

#     test_file = 'SurvData/Annotations/OthersCohort.csv'
#     test_data = open(test_file, 'r').readlines()
#     with open('SurvData/Annotations/test.txt', 'w') as f:
#         for d in test_data[1:]:
#             f.write(f'{d.split(",")[1]}\n')

    # train_ds = SurvDataset(
    #             data_dir='SurvData/PKCancerCohortCropData',
    #             split_file='SurvData/Annotations/train.txt',
    #             split='train',
    #             anno_file='SurvData/Annotations/PKCancerCohort.csv',
    #             printer=print,
    #             median=365,
    #             num_time=3,
    #             num_lesion=5,
    #             discard=True,
    #             target_size=224,
    #             black_lst=['a_target_Liver1.jpg', 'a_target_Liver2.jpg'])
    # for data in train_ds:
    #     for key, value in data.items():
    #         if isinstance(value, torch.Tensor):
    #             print(key, value.shape)
    #         else:
    #             print(key, value)
    #     break

    # valid_ds = SurvDataset(
    #             data_dir='SurvData/PKCancerCohortCropData',
    #             split_file='SurvData/Annotations/valid.txt',
    #             split='valid',
    #             anno_file='SurvData/Annotations/PKCancerCohort.csv',
    #             printer=print,
    #             median=365,
    #             num_time=3,
    #             num_lesion=5,
    #             discard=True,
    #             target_size=224,
    #             black_lst=['a_target_Liver1.jpg', 'a_target_Liver2.jpg'])
    # for data in valid_ds:
    #     for key, value in data.items():
    #         if isinstance(value, torch.Tensor):
    #             print(key, value.shape)
    #         else:
    #             print(key, value)
    #     break