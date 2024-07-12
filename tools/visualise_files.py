import pickle
import pprint

import sys, os.path as osp

sys.path.append(osp.abspath('.'))
sys.path.append(osp.abspath('/home/cristian/Projects/Pillar_R-CNN/'))
print(f'Current system paths: {sys.path}')

if __name__ == "__main__":
    dbinfos = pickle.load(open("data/Waymo/dbinfos_train_1sweeps_withvelo.pkl", "rb"))
    infos_test = pickle.load(open("data/Waymo/infos_test_01sweeps_filter_zero_gt.pkl", "rb"))
    infos_train = pickle.load(open("data/Waymo/infos_train_01sweeps_filter_zero_gt.pkl", "rb"))
    infos_val = pickle.load(open("data/Waymo/infos_val_01sweeps_filter_zero_gt.pkl", "rb"))

    print("dbinfos")
    pprint.pprint(dbinfos)
    print("infos_test")
    pprint.pprint(infos_test)
    print("infos_train")
    pprint.pprint(infos_train)
    print("infos_val")
    pprint.pprint(infos_val)
