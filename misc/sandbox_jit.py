import numpy as np

import torch
experiment_folder = "inference/generated/transflower_expmap_cr4_bs5_og2_futureN_gauss4_tw3/"
model = torch.jit.load(experiment_folder+'compiled_jit.pth', map_location=torch.device('cuda:0'))
#model = torch.jit.load(experiment_folder+'compiled_jit.pth')

data_folder="/gpfsscratch/rech/imi/usc19dv/data/UR5_processed/"
acts = np.load(data_folder+"UR5_Alex_obs_act_etc_100_data.npz.acts_scaled.npy")[:120]
obs = np.load(data_folder+"UR5_Alex_obs_act_etc_100_data.npz.obs_scaled.npy")[:120]
anns = np.load(data_folder+"UR5_Alex_obs_act_etc_100_data.annotation.npy")

inputs = [torch.from_numpy(anns).unsqueeze(1).unsqueeze(1).long().cuda(), torch.from_numpy(obs).unsqueeze(1).float().cuda(), torch.from_numpy(acts).unsqueeze(1).float().cuda()]
#inputs = [torch.from_numpy(anns).unsqueeze(1).unsqueeze(1).long(), torch.from_numpy(obs).unsqueeze(1).float(), torch.from_numpy(acts).unsqueeze(1).float()]
eps_std=1.0
z_shape = (1, 8, 1)
noise = torch.normal(mean=torch.zeros(z_shape), std=torch.ones(z_shape)*eps_std).cuda()
#noise = noise.unsqueeze(-1)
#l_shape = (1,800,1,1)
l_shape = (1,1,800)
latent = torch.normal(mean=torch.zeros(l_shape), std=torch.ones(l_shape)*eps_std).cuda()
sldj = torch.zeros(noise.size(0), device=noise.device)

#print(model(noise,latent))
#print(model(noise,latent))
#print(model(latent))
#print(model(latent))
#print(model(noise,latent,sldj))
#print(model(noise,latent,sldj))
print(model(inputs))
print(model(inputs))
print(model(inputs))
#print(model(inputs,noise))
#print(model(inputs,noise))
#print(model(inputs,noise))
