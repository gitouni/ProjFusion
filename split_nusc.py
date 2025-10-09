from dataset import NuSceneDataset
import numpy as np
val_ratio = 0.2
seed = 0
np.random.seed(seed)
dataset = NuSceneDataset(daylight=False)
N = len(dataset.scene_name_list)
val_N = int(N*val_ratio)
train_N = N - val_N
index = np.random.permutation(N)
tr_index = index[:train_N]
val_index = index[train_N:]
tr_scene_names = [dataset.scene_name_list[idx] for idx in tr_index]
val_scene_names = [dataset.scene_name_list[idx] for idx in val_index]
np.savetxt("cache/nuscenes_split/train.txt", tr_scene_names, fmt="%s")
np.savetxt("cache/nuscenes_split/val.txt", val_scene_names, fmt="%s")