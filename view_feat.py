import numpy as np
from matplotlib import pyplot as plt
import os
debug_file = 'debug_data.npz'
data = np.load(debug_file)
proj_uv = data['proj_uv']  # (N, 2)
u, v = proj_uv[:,0], proj_uv[:,1]
feat_2d = data['feat_2d'].mean(axis=-1)  # (h, w, )
feat_3d = data['feat_3d'].mean(axis=-1)  # (N, )
interp_2d = data['interp_2d'].mean(axis=-1)  # (h, w, )

H, W = interp_2d.shape
plt.imshow(feat_2d)
plt.axis('off')
plt.savefig(os.path.join('fig/debug','img_feat.png'), bbox_inches='tight')
plt.cla()
plt.imshow(interp_2d)
plt.axis('off')
plt.savefig(os.path.join('fig/debug','interp_feat.png'), bbox_inches='tight')
plt.cla()
# plt.axis([0,W,H,0])
rev = (0 <= u) * (u < W) * (0 <= v) * (v < H)
u = u[rev]
v = v[rev]
feat_3d = feat_3d[rev]
vmin = feat_3d.min()
vmax = feat_3d.max()
plt.scatter([u],[v],c=[feat_3d], vmin=vmin, vmax=vmax)
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.savefig(os.path.join('fig/debug','proj_feat.png'), bbox_inches='tight')
plt.scatter([u],[v], c=np.zeros([len(u),3]) )
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.savefig(os.path.join('fig/debug','proj_pts.png'), bbox_inches='tight')
plt.close()
# # plt.axes([0,h,w,0])
# plt.show()