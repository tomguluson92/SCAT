import sys
import torch
import torch.nn as nn
import numpy as np

import json
#from util import , batch_rodrigues, batch_lrotmin
#from config import args
from torch.autograd import Variable
import torch.nn.functional as F

def batch_rodrigues(theta):
    batch_size = theta.shape[0]
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)

    return quat2mat(quat)

def quat2mat(quat):
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def batch_global_rigid_transformation(Rs, Js, parent, rotate_base=False):
    N = Rs.shape[0]
    if rotate_base:
        np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float)
        np_rot_x = np.reshape(np.tile(np_rot_x, [N, 1]), [N, 3, 3])
        rot_x = Variable(torch.from_numpy(np_rot_x).float()).cuda()
        root_rotation = torch.matmul(Rs[:, 0, :, :], rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]
    Js = torch.unsqueeze(Js, -1)

    def make_A(R, t):
        R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
        t_homo = torch.cat([t, Variable(torch.ones(N, 1, 1)).cuda()], dim=1)
        return torch.cat([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]

    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = torch.matmul(results[parent[i]], A_here)
        results.append(res_here)

    results = torch.stack(results, dim=1)

    new_J = results[:, :, :3, 3]
    Js_w0 = torch.cat([Js, Variable(torch.zeros(N, 16, 1, 1)).cuda()], dim=2)
    init_bone = torch.matmul(results, Js_w0)
    init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
    A = results - init_bone

    return new_J, A

def batch_lrotmin(theta):
    theta = theta[:,3:].contiguous()
    Rs = batch_rodrigues(theta.reshape(-1, 3))
    e = Variable(torch.eye(3).float())
    Rs = Rs.sub(1.0, e)

    return Rs.reshape(-1, 23 * 9)

class ManoHand(nn.Module):
    def __init__(self, model_path,batch_size, obj_saveable = False):
        super(ManoHand, self).__init__()
        self.pose_param_count = 12 #args.pose_param_count

        self.finger_index = [734, 333, 443, 555, 678]

        batch_size = batch_size #args.batch_size
        self.model_path = model_path
        with open(self.model_path) as reader:
            model = json.load(reader)
        if obj_saveable:
            self.faces = model['f']
        else:
            self.faces = None
    
        np_v_template = np.array(model['v_template'], dtype = np.float)
        vertex_count, vertex_component = np_v_template.shape[0], np_v_template.shape[1]
        self.size = [vertex_count, 3]

        if True: #args.predict_shape:
            self.register_buffer('v_template', torch.from_numpy(np_v_template).float())
        else:
            np_v_template = np.tile(np_v_template, (batch_size, 1))
            self.register_buffer('v_template', torch.from_numpy(np_v_template).float().reshape(-1, vertex_count, vertex_component))
        
        np_J_regressor = np.array(model['J_regressor'], dtype = np.float).T
        self.register_buffer('J_regressor', torch.from_numpy(np_J_regressor).float())

        np_shapedirs = np.array(model['shapedirs'], dtype = np.float)
        num_shape_basis = np_shapedirs.shape[-1]
        np_shapedirs = np.reshape(np_shapedirs, [-1, num_shape_basis]).T
        self.register_buffer('shapedirs', torch.from_numpy(np_shapedirs).float())

        np_posedirs = np.array(model['posedirs'], dtype = np.float)
        num_pose_basis = np_posedirs.shape[-1]
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.from_numpy(np_posedirs).float())

        self.parents = np.array(model['kintree_table'])[0].astype(np.int32)

        np_weight = np.array(model['weights'], dtype = np.float)
        vertex_count, vertex_component = np_weight.shape[0], np_weight.shape[1]
        np_weight = np.tile(np_weight, (batch_size, 1))
        self.register_buffer('weight', torch.from_numpy(np_weight).float().reshape(-1, vertex_count, vertex_component))

        hands_mean = np.array(model['hands_mean'], dtype = np.float)
        self.register_buffer('hands_mean', torch.from_numpy(hands_mean).float())

        hands_components = np.array(model['hands_components'], dtype = np.float)
        self.register_buffer('hands_components', torch.from_numpy(hands_components).float())

        self.register_buffer('o1', torch.ones(batch_size, vertex_count).float())
        self.register_buffer('e3', torch.eye(3).float())
        self.cur_device = None

    def save_obj(self, verts, obj_mesh_name):
        if not self.faces:
            msg = 'obj not saveable!'
            sys.exit(msg)

        with open(obj_mesh_name, 'w') as fp:
            for v in verts:
                fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

            for f in self.faces:
                fp.write( 'f %d %d %d\n' %  (f[0] + 1, f[1] + 1, f[2] + 1) )

    def forward(self, beta, theta, get_skin = False):
        def _get_full_pose(theta):
            g_rot, partial_local_rot = theta[:, :3], theta[:, 3:]
            full_local_rot = torch.matmul(partial_local_rot, self.hands_components[:self.pose_param_count, :]) + self.hands_mean
            return torch.cat((g_rot, full_local_rot), dim = 1)

        if not self.cur_device:
            if theta is not None:
                device = theta.device
            else:
                device = beta.device
            self.cur_device = torch.device(device.type, device.index)
        
        theta = _get_full_pose(theta)
        num_batch = beta.shape[0] if beta is not None else theta.shape[0]
        if True: #args.predict_shape:
            v_shaped = torch.matmul(beta, self.shapedirs).reshape(-1, self.size[0], self.size[1]) + self.v_template
        else:
            v_shaped = self.v_template[:num_batch]

        Jx = torch.matmul(v_shaped[:,:,0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:,:,1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:,:,2], self.J_regressor)

        J = torch.stack([Jx, Jy, Jz], dim = 2)
        Rs = batch_rodrigues(theta.reshape(-1, 3)).reshape(-1, 16, 3, 3)
        pose_feature = Rs[:,1:,:,:].sub(1.0, self.e3).reshape(-1, 135)
        v_posed = torch.matmul(pose_feature, self.posedirs).reshape(-1, self.size[0], self.size[1]) + v_shaped
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        weight = self.weight[:num_batch]
        W = weight.reshape(num_batch, -1, 16)
        T = torch.matmul(W, A.reshape(num_batch, 16, 16)).reshape(num_batch, -1, 4, 4)
        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        verts = v_homo[:,:,:3,0]
        joint_x = torch.matmul(verts[:,:,0], self.J_regressor)
        joint_y = torch.matmul(verts[:,:,1], self.J_regressor)
        joint_z = torch.matmul(verts[:,:,2], self.J_regressor)

        finger_verts = verts[:, self.finger_index, :]

        joints = torch.stack([joint_x, joint_y, joint_z], dim = 2)

        joints = torch.cat((joints, finger_verts), dim = 1)

        if get_skin:
            return verts, joints, Rs
        else:
            return joints


import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import pickle
import numpy as np

# -------------------
# Mano in Pytorch
# -------------------

bases_num = 10
pose_num = 6
mesh_num = 778
keypoints_num = 16

dd = pickle.load(open('extra_data/MANO_RIGHT.pkl', 'rb'),encoding='latin1')
kintree_table = dd['kintree_table']
id_to_col = {kintree_table[1, i]: i for i in range(kintree_table.shape[1])}
parent = {i: id_to_col[kintree_table[0, i]] for i in range(1, kintree_table.shape[1])}

mesh_mu = Variable(torch.from_numpy(np.expand_dims(dd['v_template'], 0).astype(np.float32)).cuda())  # zero mean
mesh_pca = Variable(torch.from_numpy(np.expand_dims(dd['shapedirs'], 0).astype(np.float32)).cuda())
posedirs = Variable(torch.from_numpy(np.expand_dims(dd['posedirs'], 0).astype(np.float32)).cuda())
J_regressor = Variable(torch.from_numpy(np.expand_dims(dd['J_regressor'].todense(), 0).astype(np.float32)).cuda())
weights = Variable(torch.from_numpy(np.expand_dims(dd['weights'], 0).astype(np.float32)).cuda())
hands_components = Variable(
    torch.from_numpy(np.expand_dims(np.vstack(dd['hands_components'][:]), 0).astype(np.float32)).cuda())
hands_mean = Variable(torch.from_numpy(np.expand_dims(dd['hands_mean'], 0).astype(np.float32)).cuda())
#root_rot = Variable(torch.FloatTensor([np.pi, 0., 0.]).unsqueeze(0).cuda())
root_rot = Variable(torch.FloatTensor([0., 0., 0.]).unsqueeze(0).cuda())

def rodrigues(r):
    theta = torch.sqrt(torch.sum(torch.pow(r, 2), 1))

    def S(n_):
        ns = torch.split(n_, 1, 1)
        Sn_ = torch.cat([torch.zeros_like(ns[0]), -ns[2], ns[1], ns[2], torch.zeros_like(ns[0]), -ns[0], -ns[1], ns[0],
                         torch.zeros_like(ns[0])], 1)
        Sn_ = Sn_.view(-1, 3, 3)
        return Sn_

    n = r / (theta.view(-1, 1))
    Sn = S(n)

    # R = torch.eye(3).unsqueeze(0) + torch.sin(theta).view(-1, 1, 1)*Sn\
    #        +(1.-torch.cos(theta).view(-1, 1, 1)) * torch.matmul(Sn,Sn)

    I3 = Variable(torch.eye(3).unsqueeze(0).cuda())

    R = I3 + torch.sin(theta).view(-1, 1, 1) * Sn \
        + (1. - torch.cos(theta).view(-1, 1, 1)) * torch.matmul(Sn, Sn)

    Sr = S(r)
    theta2 = theta ** 2
    R2 = I3 + (1. - theta2.view(-1, 1, 1) / 6.) * Sr \
         + (.5 - theta2.view(-1, 1, 1) / 24.) * torch.matmul(Sr, Sr)

    idx = np.argwhere((theta < 1e-30).data.cpu().numpy())

    if (idx.size):
        R[idx, :, :] = R2[idx, :, :]

    return R, Sn


def get_poseweights(poses, bsize):
    # pose: batch x 24 x 3
    pose_matrix, _ = rodrigues(poses[:, 1:, :].contiguous().view(-1, 3))
    # pose_matrix, _ = rodrigues(poses.view(-1,3))
    pose_matrix = pose_matrix - Variable(torch.from_numpy(
        np.repeat(np.expand_dims(np.eye(3, dtype=np.float32), 0), bsize * (keypoints_num - 1), axis=0)).cuda())
    pose_matrix = pose_matrix.view(bsize, -1)
    return pose_matrix


def rot_pose_beta_to_mesh(rots, poses, betas):
    batch_size = rots.size(0)
    #print(hands_mean.shape,poses.unsqueeze(1).shape,hands_components.shape)
    #poses = (hands_mean + torch.matmul(poses.unsqueeze(1), hands_components).squeeze(1)).view(batch_size,keypoints_num - 1, 3)  #if use pca
    poses = (hands_mean + poses).view(batch_size,keypoints_num - 1, 3)
    # poses = torch.cat((poses[:,:3].contiguous().view(batch_size,1,3),poses_),1)
    poses = torch.cat((root_rot.repeat(batch_size, 1).view(batch_size, 1, 3), poses), 1)

    v_shaped = (torch.matmul(betas.unsqueeze(1),
                             mesh_pca.repeat(batch_size, 1, 1, 1).permute(0, 3, 1, 2).contiguous().view(batch_size,
                                                                                                        bases_num,
                                                                                                        -1)).squeeze(1)
                + mesh_mu.repeat(batch_size, 1, 1).view(batch_size, -1)).view(batch_size, mesh_num, 3)

    pose_weights = get_poseweights(poses, batch_size)

    v_posed = v_shaped + torch.matmul(posedirs.repeat(batch_size, 1, 1, 1),
                                      (pose_weights.view(batch_size, 1, (keypoints_num - 1) * 9, 1)).repeat(1, mesh_num,
                                                                                                            1,
                                                                                                            1)).squeeze(
        3)

    J_posed = torch.matmul(v_shaped.permute(0, 2, 1), J_regressor.repeat(batch_size, 1, 1).permute(0, 2, 1))
    J_posed = J_posed.permute(0, 2, 1)
    J_posed_split = [sp.contiguous().view(batch_size, 3) for sp in torch.split(J_posed.permute(1, 0, 2), 1, 0)]

    pose = poses.permute(1, 0, 2)
    pose_split = torch.split(pose, 1, 0)

    angle_matrix = []
    for i in range(keypoints_num):
        out, tmp = rodrigues(pose_split[i].contiguous().view(-1, 3))
        angle_matrix.append(out)

    # with_zeros = lambda x: torch.cat((x,torch.FloatTensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(batch_size,1,1)),1)

    with_zeros = lambda x: \
        torch.cat((x, Variable(torch.FloatTensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(batch_size, 1, 1).cuda())), 1)

    pack = lambda x: torch.cat((Variable(torch.zeros(batch_size, 4, 3).cuda()), x), 2)

    results = {}
    results[0] = with_zeros(torch.cat((angle_matrix[0], J_posed_split[0].view(batch_size, 3, 1)), 2))

    for i in range(1, kintree_table.shape[1]):
        tmp = with_zeros(torch.cat((angle_matrix[i],
                                    (J_posed_split[i] - J_posed_split[parent[i]]).view(batch_size, 3, 1)), 2))
        results[i] = torch.matmul(results[parent[i]], tmp)

    results_global = results

    results2 = []

    for i in range(len(results)):
        vec = (torch.cat((J_posed_split[i], Variable(torch.zeros(batch_size, 1).cuda())), 1)).view(batch_size, 4, 1)
        results2.append((results[i] - pack(torch.matmul(results[i], vec))).unsqueeze(0))

    results = torch.cat(results2, 0)

    T = torch.matmul(results.permute(1, 2, 3, 0),
                     weights.repeat(batch_size, 1, 1).permute(0, 2, 1).unsqueeze(1).repeat(1, 4, 1, 1))
    Ts = torch.split(T, 1, 2)
    rest_shape_h = torch.cat((v_posed, Variable(torch.ones(batch_size, mesh_num, 1).cuda())), 2)
    rest_shape_hs = torch.split(rest_shape_h, 1, 2)

    v = Ts[0].contiguous().view(batch_size, 4, mesh_num) * rest_shape_hs[0].contiguous().view(-1, 1, mesh_num) \
        + Ts[1].contiguous().view(batch_size, 4, mesh_num) * rest_shape_hs[1].contiguous().view(-1, 1, mesh_num) \
        + Ts[2].contiguous().view(batch_size, 4, mesh_num) * rest_shape_hs[2].contiguous().view(-1, 1, mesh_num) \
        + Ts[3].contiguous().view(batch_size, 4, mesh_num) * rest_shape_hs[3].contiguous().view(-1, 1, mesh_num)

    # v = v.permute(0,2,1)[:,:,:3]
    Rots = rodrigues(rots)[0]

    Jtr = []

    for j_id in range(len(results_global)):
        Jtr.append(results_global[j_id][:, :3, 3:4])

    # Add finger tips from mesh to joint list
    # definition as [8] MANO
    #Jtr.insert(4, v[:, :3, 333].unsqueeze(2)) #index
    #Jtr.insert(8, v[:, :3, 444].unsqueeze(2)) #middle
    #Jtr.insert(12, v[:, :3, 672].unsqueeze(2)) #pinky
    #Jtr.insert(16, v[:, :3, 555].unsqueeze(2)) #ring
    #Jtr.insert(20, v[:, :3, 745].unsqueeze(2)) #thumb
    
    # definition as HO3D MANO
    #Jtr.append(v[:, :3, 744].unsqueeze(2)) #thumb
    #Jtr.append(v[:, :3, 320].unsqueeze(2)) #index
    #Jtr.append(v[:, :3, 443].unsqueeze(2)) #middle
    #Jtr.append(v[:, :3, 554].unsqueeze(2)) #ring
    #Jtr.append(v[:, :3, 671].unsqueeze(2)) #pinky
    
    #definition as frankmocap smplx
    Jtr.append(v[:, :3, 320].unsqueeze(2)) #index
    Jtr.append(v[:, :3, 443].unsqueeze(2)) #middle
    Jtr.append(v[:, :3, 671].unsqueeze(2)) #pinky
    Jtr.append(v[:, :3, 554].unsqueeze(2)) #ring
    Jtr.append(v[:, :3, 744].unsqueeze(2)) #thumb

    Jtr = torch.cat(Jtr, 2)  # .permute(0,2,1)

    v = torch.matmul(Rots, v[:, :3, :]).permute(0, 2, 1)  # .contiguous().view(batch_size,-1)
    Jtr = torch.matmul(Rots, Jtr).permute(0, 2, 1)  # .contiguous().view(batch_size,-1)
    
    #translate to be same as smplx
    root=Jtr[:,1].clone().unsqueeze(1)
    Jtr-=root
    v-=root
    
    
    return torch.cat((Jtr, v), 1)