import numpy as np
import pprint

### GETTING THE MODEL ###
import os
if "ROOT_DIR_MODEL" not in os.environ:
    root_dir_model = "/home/guillefix/code/multimodal-transflower"
else:
    root_dir_model = os.environ["ROOT_DIR_MODEL"]
import sys
sys.path.append(root_dir_model)
from inference.utils import load_model_from_logs_path
pretrained_folder = "/home/guillefix/code/inria/pretrained/"
pretrained_name="transflower_zp5_short_single_obj_nocol_trim_tw_single_more_filtered"
# pretrained_name="transflower_zp5_short_single_obj_nocol_trim_tw_single_more_filtered_nodp"
default_save_path = pretrained_folder+pretrained_name
logs_path = default_save_path
model, opt = load_model_from_logs_path(logs_path, no_grad=False);
# latest_checkpoint = get_latest_checkpoint(logs_path)
# latest_state_dict = latest_checkpoint["state_dict"]
saved_params = [p.data.clone() for p in model.parameters()]

model.opt.input_modalities
model.opt.input_lengths
model.opt.dins

import torch
import time
data = [torch.randint(73,size=(10,1,1)).cuda(), torch.randn(1,1,18).cuda(), torch.randn(1,1,8).cuda()]

### GETTING THE "actor" and distribution
latents=model.get_latents(data) #this will be the actor

latents[0].shape

dists = model.get_dists(latents) #this will be the distribution function
#%%

# testing it works
# dists
# a = dists[0].sample()
# dists[0].log_prob(a)
# c=dists[0].cond
# a=a.permute(0,2,1).unsqueeze(3)
# c=c.permute(0,2,1).unsqueeze(3)
# sldj = torch.zeros(a.size(0), device=a.device)
# model.output_mod_glows[0].flows(a,c,sldj, False)
# from models.util import channelwise, checkerboard, Flip, safe_log, squeeze, unsqueeze
# asplit=channelwise(a)
# sldj = torch.zeros(a.size(0), device=a.device)
# model.output_mod_glows[0].flows.channels[1]
# # for m in model.output_mod_glows[0].flows.channels[2].modules():
# # for m in model.output_mod_glows[0].flows.channels[2].modules()
# from models.flowplusplus.act_norm import BatchNorm

# def eval_dropouts(model):
#     for m in model.children():
#         if isinstance(m, torch.nn.Dropout):
#             m.p=0.0
#             # m.eval()
#         elif isinstance(m, torch.nn.MultiheadAttention):
#             m.dropout=0.0
#             # m.eval()
#         elif isinstance(m, torch.nn.Module):
#             eval_dropouts(m)
#
# eval_dropouts(model)

#
# eval_dropouts(model.output_mod_glows[0].flows.channels[2])
# for p in model.output_mod_glows[0].flows.channels[2].parameters():
#     print(p)
# # model.output_mod_glows[0].flows.channels[2] = model.output_mod_glows[0].flows.channels[2].eval()
# asplit=channelwise(a)
# sldj = torch.zeros(a.size(0), device=a.device)
# asplit,sldj = model.output_mod_glows[0].flows.channels[2](asplit,c,sldj, False); print(sldj)
# model = model.eval()
# model.output_mod_glows[0].flows.channels[2].eval()
# asplit=channelwise(a)
# sldj = torch.zeros(a.size(0), device=a.device)
# for i in range(3):
#     asplit,sldj = model.output_mod_glows[0].flows.channels[i](asplit,c,sldj, False); print(sldj)

# model.eval()

# from models.transflower_model import NormalizingFlow
#
# dists = []
# for i,latent in enumerate(latents):
#     dist = NormalizingFlow(model.output_mod_glows[i], latent, validate_args=False)
#     # dist = None
#     dists.append(dist)
#
# m = NormalizingFlow(model.output_mod_glows[i], latent, validate_args=False)

model.input_lengths

#%%

### NOW SETTING UP TIANSHOU ###

from torch import nn
from tianshou.utils.net.continuous import ActorProb
from agents.utils import Critic

device = model.device
class Actor(nn.Module):
    def __init__(self, model, index):
        super().__init__()
        self.model = model
        self.index = index

    def forward(self, data, state=None, info={}):
        data = (torch.from_numpy(data["ann"]).permute(1,0,2).long().to(device),
                torch.from_numpy(data["obs"]).permute(1,0,2).float().to(device),
                torch.from_numpy(data["acts"]).permute(1,0,2).float().to(device))
        return self.model.get_latents(data)[self.index], state

actor = Actor(model, 0)

class NetCritic(nn.Module):
    def __init__(self, model, index):
        super().__init__()
        self.model = model
        self.index = index

    def forward(self, data):
        data = (torch.from_numpy(data["ann"]).permute(1,0,2).long().to(device),
                torch.from_numpy(data["obs"]).permute(1,0,2).float().to(device),
                torch.from_numpy(data["acts"]).permute(1,0,2).float().to(device))
        result = self.model.get_latents(data, latent_chunk_index=1)[self.index]
        result = result.view(result.shape[0],-1)
        return result, None

net_c = NetCritic(model, 0)

critic = Critic(net_c, device=device, preprocess_net_output_dim=800*net_c.model.conditioning_seq_lens[0], flatten_obs=False, convert_to_torch=False)
critic.to(device)

print("AAAAAAAAAAAAAAAAA", list(actor.model.output_mod_glows[0].parameters())[-10])
# lr = 0.0
# lr = 0.9695e-3
# lr = 1e-7
lr = 5e-7
optim = torch.optim.Adam(
    list(actor.parameters()) + list(critic.parameters()), lr=lr
)
#%%

import sys
root_env="/home/guillefix/code/inria/RobotLangEnv/"
sys.path.append(root_env)


from src.envs.envList import ExtendedUR5PlayAbsRPY1Obj

from constants import *

#%%
input_dims = [int(x) for x in opt.dins.split(",")]
output_dims = [int(x) for x in str(opt.douts).split(",")]
input_lengths = [int(x) for x in opt.input_lengths.split(",")]
input_mods = opt.input_modalities.split(",")
output_mods = opt.output_modalities.split(",")

obs_mod = None
obs_mod_idx = None
acts_mod = None
acts_mod_idx = None
ann_mod = None
ann_mod_idx = None
ttg_mod = None
ttg_mod_idx = None
for i,mod in enumerate(input_mods):
    if "obs" in mod:
        obs_mod = mod
        obs_mod_idx = i
    elif "acts" in mod:
        acts_mod = mod
        acts_mod_idx = i
    elif "annotation" in mod:
        ann_mod = mod
        ann_mod_idx = i
    elif "times_to_go" in mod:
        ttg_mod = mod
        ttg_mod_idx = i

if ttg_mod is None:
    times_to_go = None
else:
    times_to_go = np.array(range(times_to_go_start+input_lengths[ttg_mod_idx]-1, times_to_go_start-1, -1))
    times_to_go = np.expand_dims(times_to_go, 1)

context_size_obs=input_lengths[obs_mod_idx]
context_size_acts=input_lengths[acts_mod_idx]
if ttg_mod is not None:
    context_size_ttg=input_lengths[ttg_mod_idx]

import pickle
obs_scaler = pickle.load(open(processed_data_folder+obs_mod+"_scaler.pkl", "rb"))
acts_scaler = pickle.load(open(processed_data_folder+acts_mod+"_scaler.pkl", "rb"))

prev_obs = obs_scaler.inverse_transform(np.zeros((context_size_obs,input_dims[obs_mod_idx])))
# print(prev_obs.shape)
prev_acts = acts_scaler.inverse_transform(np.zeros((context_size_acts,output_dims[0])))
#%%
save_relabelled_trajs = False
args = {}

import importlib
import src.envs.envList; importlib.reload(src.envs.envList); from src.envs.envList import ExtendedUR5PlayAbsRPY1Obj

# goal_str = "Put green dog on the shelf"
goal_str = None
env_fn = lambda: ExtendedUR5PlayAbsRPY1Obj(goal_str = goal_str, obs_scaler = obs_scaler, acts_scaler = acts_scaler, prev_obs = prev_obs, save_relabelled_trajs = save_relabelled_trajs,
                                check_completed_goals = False, use_dict_space = True, max_episode_length = 3000, sample_random_goal = True, sample_goal_from_train_set = True,
                                prev_acts = prev_acts, times_to_go = times_to_go, desc_max_len = input_lengths[ann_mod_idx], obs_mod = obs_mod, args=args)

env = env_fn()

# env.action_space
# env.observation_space
#%%
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

lr_scheduler = None
lr_decay = False
# step_per_epoch = 2048
step_per_epoch = 30000
step_per_collect = 2048
# step_per_collect = 256
epoch = 200
if lr_decay:
    # decay learning rate to 0 linearly
    max_update_num = np.ceil(
        step_per_epoch / step_per_collect
    ) * epoch

    lr_scheduler = LambdaLR(
        optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
    )

def dist(latent):
    dists = model.get_dists([latent])
    return dists[0]

gamma = 0.99
gae_lambda = 0.95
max_grad_norm = 1.0
vf_coef = 0.25
ent_coef = 0.0
rew_norm = False
bound_action_method = "clip"
# eps_clip = 0.2
eps_clip = 20000
value_clip = None
dual_clip = 2.0
# dual_clip = 20000
# dual_clip = None
norm_adv = 0
recompute_adv = 1
from tianshou.policy import PPOPolicy

policy = PPOPolicy(
    actor,
    critic,
    optim,
    dist,
    discount_factor=gamma,
    gae_lambda=gae_lambda,
    max_grad_norm=max_grad_norm,
    vf_coef=vf_coef,
    ent_coef=ent_coef,
    reward_normalization=rew_norm,
    action_scaling=False,
    action_bound_method=bound_action_method,
    lr_scheduler=lr_scheduler,
    action_space=env.action_space,
    eps_clip=eps_clip,
    value_clip=value_clip,
    dual_clip=dual_clip,
    advantage_normalization=norm_adv,
    recompute_advantage=recompute_adv
)
#%%

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv

training_num = 8
train_envs = SubprocVectorEnv(
    [env_fn for _ in range(training_num)]
)
train_envs.seed([int(time.time())+i*training_num for i in range(training_num)])

test_num = 8
test_envs = SubprocVectorEnv(
    [env_fn for _ in range(test_num)],
)
train_envs.seed([6969*training_num+int(time.time())+i*training_num for i in range(training_num)])

buffer_size = 4096
# buffer_size = 256
# collector
if training_num > 1:
    buffer = VectorReplayBuffer(buffer_size, len(train_envs))
else:
    buffer = ReplayBuffer(buffer_size)
#%%
train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
# test_collector = Collector(policy, train_envs)
test_collector = Collector(policy, test_envs)
# test_collector = None

# policy.eval()
# test_envs.seed([6969+i*test_num for i in range(test_num)])
# test_collector.reset()
# result = test_collector.collect(n_episode=test_num*10, render=False)
# print(result)
# print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')

log_path = "./tmp"
def save_best_fn(policy):
    torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

# repeat_per_collect = 2048
repeat_per_collect = 2
# batch_size = 128
batch_size = 64
from tianshou.trainer import onpolicy_trainer
# env.observation_space.sample()[1].shape
# trainer
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter(log_path)
logger = TensorboardLogger(writer, update_interval=100, train_interval=100)
result = onpolicy_trainer(
    policy,
    train_collector,
    test_collector,
    epoch,
    step_per_epoch,
    repeat_per_collect,
    test_num,
    batch_size,
    step_per_collect=step_per_collect,
    save_best_fn=save_best_fn,
    logger=logger,
    test_in_train=False,
)
pprint.pprint(result)

# print("AAAAAAAAAAAAAAAAA", list(actor.model.output_mod_glows[0].parameters())[-10])
# for i,p in enumerate(policy.actor.model.parameters()):
#     print(p.data == saved_params[i])
# Let's watch its performance!
policy.eval()
test_collector.reset()
result = test_collector.collect(n_episode=test_num*10, render=False)
print(result)
print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')
