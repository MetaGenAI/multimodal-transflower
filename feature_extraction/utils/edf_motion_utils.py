import numpy as np
from scipy.spatial.transform import Rotation as R
from pymo.rotation_tools import Rotation, euler2expmap, euler2expmap2, expmap2euler, euler_reorder, unroll
from pymo.Quaternions import Quaternions

import scipy.ndimage
# import splines.quaternion

import argparse
parser = argparse.ArgumentParser(description='Convert motion features to relative or absolute.')
parser.add_argument('path', metavar='path', type=str, help='path to the features file')
parser.add_argument('--relative', dest='relative', action='store_true')
parser.add_argument('--absolute', dest='relative', action='store_false')
parser.add_argument('--init_theta_path', type=str, help='path to the init theta file')
parser.add_argument('--init_pos_xz_path', type=str, help='path to the init pos_xz file')
parser.add_argument('--position_smoothing', dest='position_smoothing', type=float, default=0.0)
parser.add_argument('--rotation_smoothing', dest='rotation_smoothing', type=float, default=0.0)

NODE_TYPES_LIST = ["Head", "LeftHand", "RightHand"]

ROT_ORDER = "xyz"

def detect_quaternion_flips(quats):
    flips = []

    for i in range(1, len(quats)):
        dot_product = np.dot(quats[i-1], quats[i])
        dot_product_flipped = np.dot(quats[i-1], -quats[i])

        if dot_product_flipped > dot_product:
            flips.append(i)
    
    return flips

def make_relative(features, node_types_list = NODE_TYPES_LIST, position_smoothing=0.0, rotation_smoothing=0.0):

    rots_y_head_inv = None
    head_forwards = None
    head_rights = None
    head_theta_diffs = None
    pos_head = None

    init_theta = None
    init_pos_xz = None

    # make sure the head is the first node type
    assert "Head" in node_types_list
    idx = node_types_list.index("Head")
    if idx != 0:
        other = node_types_list[0]
        node_types_list[0] = "Head"
        node_types_list[idx] = other

    # make rotations along the y axis relative to the head
    for i, node_type in enumerate(node_types_list):
        rot_stream = features[:,i*3:(i+1)*3]
        # Rs = R.from_quat(rot_stream)
        # eulers = expmap2euler(rot_stream, ROT_ORDER, True)
        # quats = Quaternions.from_euler(eulers, order=ROT_ORDER, world=True)
        Rs = R.from_rotvec(rot_stream, degrees=False)
        if rotation_smoothing > 0:
            filter_width = rotation_smoothing
            # quats = Rs.as_quat()
            # flips = detect_quaternion_flips(quats)
            # print(node_type)
            # print(flips)
            # for flip in flips:
            #     quats[flip:] = -quats[flip:]
            # flips = detect_quaternion_flips(quats)
            # print(flips)
            # quats = scipy.ndimage.gaussian_filter1d(quats, filter_width, axis=0, mode='nearest')
            # quats = quats/np.linalg.norm(quats, axis=1, keepdims=True)
            # Rs = R.from_quat(quats)
            mats = Rs.as_matrix()
            mats = scipy.ndimage.gaussian_filter1d(mats, filter_width, axis=0, mode='nearest')
            Rs = R.from_matrix(mats)
            # s = splines.quaternion.PiecewiseSlerp([splines.quaternion.Quaternion(q[0], q[1:]) for q in quats], grid=np.arange(len(quats)))
            # quats = s.evaluate(np.arange(len(quats)))
            # rot_stream = scipy.ndimage.gaussian_filter1d(rot_stream, filter_width, axis=0, mode='nearest')
            # Rs = R.from_rotvec(rot_stream, degrees=False)
        # Rs = R.from_quat(quats)
        if node_type == "Head":
            vs = Rs.apply(np.array([0,0,1]).T)
            vs[:,1] = 0
            vs = vs/np.linalg.norm(vs, axis=1, keepdims=True)
            head_forwards = vs.copy()
            head_rights = vs.copy()
            head_rights[:,0] = vs[:,2]
            head_rights[:,2] = -vs[:,0]
            # print(vs.shape)
            thetas = np.arctan(vs[:,0]/vs[:,2])
            thetas += np.pi*(np.sign(vs[:,2])+1)/2
            print(thetas)
            Rys = R.from_euler(seq="y", angles=thetas)
            init_theta = thetas[0:1]
            # thetas = np.concatenate([np.array([0]), thetas])
            thetas = np.concatenate([thetas[0:1], thetas])
            thetas_diff = np.diff(thetas)
            for diff in thetas_diff:
                if np.abs(diff+2*np.pi) < np.abs(diff):
                    diff = diff+2*np.pi
                elif np.abs(diff-2*np.pi) < np.abs(diff):
                    diff = diff-2*np.pi
            rots_y_head_inv = Rys.inv()
            head_theta_diffs = np.expand_dims(thetas_diff,1)
        
        # we rotate all the nodes by the opposite of the head rotation in the vertical axis
        Rs = rots_y_head_inv * Rs
        # quats = Rs.as_quat()
        # q = Quaternions(quats)
        # eulers = q.euler(order=ROT_ORDER)
        # rot_stream = euler2expmap(eulers, ROT_ORDER, True)
        rot_stream = Rs.as_rotvec(degrees=False)
        print(node_type)
        rot_stream = unroll(rot_stream)
        # rot_stream = unroll(rot_stream)
        # rot_stream = unroll(rot_stream)
        features[:,i*3:(i+1)*3] = rot_stream
        # if node_type == "Head":
        #     features[:,i*3+1:i*3+2] = head_theta_diffs
    
    head_pos_y = None
    head_deltas_rel = None
    for i, node_type in enumerate(node_types_list):
        pos_stream = features[:,(len(node_types_list)*3)+i*3:(len(node_types_list)*3)+(i+1)*3]
        if position_smoothing > 0:
            filter_width = position_smoothing
            pos_stream = scipy.ndimage.gaussian_filter1d(pos_stream, filter_width, axis=0, mode='nearest')
        if node_type == "Head":
            pos_head = pos_stream.copy()
            pos_xz = np.stack([pos_head[:,0],pos_head[:,2]],axis=1)
            init_pos_xz = pos_xz[0:1]
            # pos_xz = np.concatenate([np.zeros((1,2)), pos_xz])
            pos_xz = np.concatenate([pos_xz[0:1], pos_xz])
            deltas = np.diff(pos_xz,axis=0)
            deltas_rel = np.einsum("ijk,ik->ij",np.stack([head_forwards[:,[0,2]], head_rights[:,[0,2]]],axis=1),deltas)
            head_pos_y = pos_stream[:,1:2]
            head_deltas_rel = deltas_rel
            features[:,len(node_types_list)*3+3*i:len(node_types_list)*3+3*(i+1)] = np.concatenate([head_deltas_rel[:,0:1], head_pos_y, head_deltas_rel[:,1:2]],axis=1)
        else:
            pos_stream[:, [0, 2]] -= pos_head[:, [0, 2]]
            pos_stream = rots_y_head_inv.apply(pos_stream)
            features[:,len(node_types_list)*3+3*i:len(node_types_list)*3+3*(i+1)] = pos_stream
    
    features = np.concatenate([head_theta_diffs, features],axis=1)
    return features, init_theta, init_pos_xz


def make_absolute(features, node_types_list = NODE_TYPES_LIST, init_thetas = None, init_pos_xz = None):
    rots_y_head = None
    head_forwards = None
    head_rights = None
    pos_head = None

    if init_thetas is None:
        init_thetas = np.zeros((1,1))
    if init_pos_xz is None:
        init_pos_xz = np.zeros((1,2))

    assert "Head" in node_types_list
    idx = node_types_list.index("Head")
    if idx != 0:
        other = node_types_list[0]
        node_types_list[0] = "Head"
        node_types_list[idx] = other

    thetas_diff = features[:,0:1]
    features = features[:,1:]

    for i, node_type in enumerate(node_types_list):
        rot_stream = features[:, i*3:(i+1)*3]
        # Rs = R.from_rotvec(rot_stream)
        # eulers = expmap2euler(rot_stream, ROT_ORDER, True)
        # quats = Quaternions.from_euler(eulers, order=ROT_ORDER, world=True)
        # Rs = R.from_quat(quats)
        Rs = R.from_rotvec(rot_stream, degrees=False)
        
        if node_type == "Head":
            # thetas_diff = features[:, i*3+1:(i+1)*3:2]
            thetas = init_thetas + np.cumsum(thetas_diff, axis=0)
            rots_y_head = R.from_euler(seq="y", angles=thetas)
            Rs = rots_y_head * Rs
            vs = Rs.apply(np.array([0,0,1]).T)
            vs[:,1] = 0
            vs = vs/np.linalg.norm(vs,axis=1,keepdims=True)
            head_forwards = vs.copy()
            head_rights = vs.copy()
            head_rights[:,0] = vs[:,2]
            head_rights[:,2] = -vs[:,0]
            
        else:
            Rs = rots_y_head * Rs

        rot_stream = Rs.as_rotvec(degrees=False)
        rot_stream = unroll(rot_stream)
        # quats = Rs.as_quat()
        # q = Quaternions(quats)
        # eulers = q.euler(order=ROT_ORDER)
        features[:, i*3:(i+1)*3] = rot_stream

    for i, node_type in enumerate(node_types_list):
        pos_stream = features[:, len(node_types_list)*3 + i*3:len(node_types_list)*3 + (i+1)*3]
        
        if node_type == "Head":
            pos_stream_y = pos_stream[:, 1:2]
            deltas_rel = pos_stream[:, [0, 2]]
            deltas = np.einsum("ijk,ik->ij",np.stack([head_forwards[:,[0,2]], head_rights[:,[0,2]]],axis=1).transpose(0,2,1),deltas_rel)
            pos_xz = init_pos_xz + np.cumsum(deltas, axis=0)
            # pos_xz = init_pos_xz + np.zeros_like(deltas)
            pos_head = np.stack([pos_xz[:,0],pos_stream_y[:,0],pos_xz[:,1]],axis=1)
            features[:, len(node_types_list)*3 + i*3:len(node_types_list)*3 + (i+1)*3] = pos_head
            
        else:
            pos_stream = rots_y_head.apply(pos_stream)
            pos_stream[:, [0, 2]] += pos_head[:, [0, 2]]
            features[:, len(node_types_list)*3 + i*3:len(node_types_list)*3 + (i+1)*3] = pos_stream
            
    return features, thetas[-1], pos_xz[-1]


## main function

if __name__ == '__main__':
    path = parser.parse_args().path
    relative = parser.parse_args().relative

    features = np.load(path)

    if relative:
        position_smoothing = parser.parse_args().position_smoothing
        rotation_smoothing = parser.parse_args().rotation_smoothing
        features, init_theta, init_pos_xz = make_relative(features, position_smoothing=position_smoothing, rotation_smoothing=rotation_smoothing)
        new_path = path[:-4] + "_relative.npy"
        new_path_init_theta = path[:-4] + "_relative_init_theta.npy"
        new_path_init_pos_xz = path[:-4] + "_relative_init_pos_xz.npy"
        np.save(new_path, features)
        np.save(new_path_init_theta, init_theta)
        np.save(new_path_init_pos_xz, init_pos_xz)
    else:
        # path_init_theta = path[:-4] + "_init_theta.npy"
        # path_init_pos_xz = path[:-4] + "_init_pos_xz.npy"
        path_init_theta = parser.parse_args().init_theta_path
        path_init_pos_xz = parser.parse_args().init_pos_xz_path
        init_theta = np.load(path_init_theta)
        init_pos_xz = np.load(path_init_pos_xz)
        if len(features.shape) == 3:
            features = features[:,0,:]
        features, new_theta, new_pos_xz = make_absolute(features, init_thetas=init_theta, init_pos_xz=init_pos_xz)
        new_path = path[:-4] + "_absolute.npy"
        np.save(new_path, features)
