import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

#order same to simple

jointsMapSMPLXToSimple = [0,
                          13, 14, 15, 20,
                          1, 2, 3, 16,
                          4, 5, 6, 17,
                          10, 11, 12, 19,
                          7, 8, 9, 18]
color_hand_joints = [[1.0, 0.0, 0.0],
                     [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                     [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                     [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                     [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                     [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little
def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf

def plot_2d_hand(axis, coords_hw, vis=None, color_fixed=None, linewidth='1', order='hw', draw_kp=True):
    """ Plots a hand stick figure into a matplotlib figure. """
    if order == 'uv':
        coords_hw = coords_hw[:, ::-1]

    colors = np.array([[0.4, 0.4, 0.4],
                       [0.4, 0.0, 0.0],
                       [0.6, 0.0, 0.0],
                       [0.8, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.4, 0.4, 0.0],
                       [0.6, 0.6, 0.0],
                       [0.8, 0.8, 0.0],
                       [1.0, 1.0, 0.0],
                       [0.0, 0.4, 0.2],
                       [0.0, 0.6, 0.3],
                       [0.0, 0.8, 0.4],
                       [0.0, 1.0, 0.5],
                       [0.0, 0.2, 0.4],
                       [0.0, 0.3, 0.6],
                       [0.0, 0.4, 0.8],
                       [0.0, 0.5, 1.0],
                       [0.4, 0.0, 0.4],
                       [0.6, 0.0, 0.6],
                       [0.7, 0.0, 0.8],
                       [1.0, 0.0, 1.0]])

    colors = colors[:, ::-1]
    colors = np.array(color_hand_joints)
    # define connections and colors of the bones
    bones = [((0, 1), colors[1, :]),
             ((1, 2), colors[2, :]),
             ((2, 3), colors[3, :]),
             ((3, 4), colors[4, :]),

             ((0, 5), colors[5, :]),
             ((5, 6), colors[6, :]),
             ((6, 7), colors[7, :]),
             ((7, 8), colors[8, :]),

             ((0, 9), colors[9, :]),
             ((9, 10), colors[10, :]),
             ((10, 11), colors[11, :]),
             ((11, 12), colors[12, :]),

             ((0, 13), colors[13, :]),
             ((13, 14), colors[14, :]),
             ((14, 15), colors[15, :]),
             ((15, 16), colors[16, :]),

             ((0, 17), colors[17, :]),
             ((17, 18), colors[18, :]),
             ((18, 19), colors[19, :]),
             ((19, 20), colors[20, :])]

    if vis is None:
        vis = np.ones_like(coords_hw[:, 0]) == 1.0

    for connection, color in bones:
        if (vis[connection[0]] == False) or (vis[connection[1]] == False):
            continue

        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 1], coords[:, 0], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 1], coords[:, 0], color_fixed, linewidth=linewidth)

    if not draw_kp:
        return

    for i in range(21):
        if vis[i] > 0.5:
            axis.plot(coords_hw[i, 1], coords_hw[i, 0], 'o', color=colors[i, :])
            #axis.text(coords_hw[i, 1], coords_hw[i, 0], '{}'.format(i), fontsize=5, color='white')

def plot_3d_hand(ax,pose_cam_xyz):
    """
    :param pose_cam_xyz: 21 x 3
    :param image_size: H, W
    :return:
    """
    assert pose_cam_xyz.shape[0] == 21

    #fig = plt.figure()
    #fig.set_size_inches(float(image_size[0]) / fig.dpi, float(image_size[1]) / fig.dpi, forward=True)

    #ax = plt.subplot(111, projection='3d')
    marker_sz = 15
    line_wd = 2

    for joint_ind in range(pose_cam_xyz.shape[0]):
        ax.plot(pose_cam_xyz[joint_ind:joint_ind + 1, 0], pose_cam_xyz[joint_ind:joint_ind + 1, 1],
                pose_cam_xyz[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
        #ax.text(pose_cam_xyz[joint_ind:joint_ind + 1, 0], pose_cam_xyz[joint_ind:joint_ind + 1, 1],
        #        pose_cam_xyz[joint_ind:joint_ind + 1, 2], '{}'.format(joint_ind), fontsize=5, color='white')
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(pose_cam_xyz[[0, joint_ind], 0], pose_cam_xyz[[0, joint_ind], 1], pose_cam_xyz[[0, joint_ind], 2],
                    color=color_hand_joints[joint_ind], lw=line_wd)
        else:
            ax.plot(pose_cam_xyz[[joint_ind - 1, joint_ind], 0], pose_cam_xyz[[joint_ind - 1, joint_ind], 1],
                    pose_cam_xyz[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
                    lw=line_wd)

    ax.axis('auto')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
#     ax.view_init(elev=-90, azim=-90)

    #ret = fig2data(fig)  # H x W x 4
    #plt.show()
    #fig.savefig('3d_joints')
    #plt.close(fig)
    #return ret

def draw_3d_skeleton(pose_cam_xyz, image_size):
    """
    :param pose_cam_xyz: 21 x 3
    :param image_size: H, W
    :return:
    """
    assert pose_cam_xyz.shape[0] == 21

    fig = plt.figure()
    fig.set_size_inches(float(image_size[0]) / fig.dpi, float(image_size[1]) / fig.dpi, forward=True)

    ax = plt.subplot(111, projection='3d')
    marker_sz = 15
    line_wd = 2

    for joint_ind in range(pose_cam_xyz.shape[0]):
        ax.plot(pose_cam_xyz[joint_ind:joint_ind + 1, 0], pose_cam_xyz[joint_ind:joint_ind + 1, 1],
                pose_cam_xyz[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
        #ax.text(pose_cam_xyz[joint_ind:joint_ind + 1, 0], pose_cam_xyz[joint_ind:joint_ind + 1, 1],
        #        pose_cam_xyz[joint_ind:joint_ind + 1, 2], '{}'.format(joint_ind), fontsize=5, color='white')
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(pose_cam_xyz[[0, joint_ind], 0], pose_cam_xyz[[0, joint_ind], 1], pose_cam_xyz[[0, joint_ind], 2],
                    color=color_hand_joints[joint_ind], lw=line_wd)
        else:
            ax.plot(pose_cam_xyz[[joint_ind - 1, joint_ind], 0], pose_cam_xyz[[joint_ind - 1, joint_ind], 1],
                    pose_cam_xyz[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
                    lw=line_wd)

    ax.axis('auto')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=-85, azim=-75)

    ret = fig2data(fig)  # H x W x 4
    plt.show()
    fig.savefig('3d_joints')
    plt.close(fig)
    return ret

    
def debug_dataset(image,joints_2d,joints_3d):
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133,projection='3d')
    ax1.imshow(image)
    ax2.imshow(image)
    plot_2d_hand(ax2, joints_2d[jointsMapSMPLXToSimple], order='uv')
    plot_3d_hand(ax3, joints_3d[jointsMapSMPLXToSimple])
    ax2.axis('off')
    print('saving debug dataset image...')
    fig.savefig('dataset_debug')
    #plt.show()
    
def debug_pred_gt(image,gt_joints_2d,gt_joints_3d,pred_joints_2d,pred_joints_3d,name):
    
    fig=plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223,projection='3d')
    ax4 = fig.add_subplot(224,projection='3d')
    if image is not None:
        image = Image.fromarray(np.uint8(image)).convert('RGB')
        ax1.imshow(image)
        ax2.imshow(image)
    plot_2d_hand(ax1, gt_joints_2d[jointsMapSMPLXToSimple], order='uv')
    ax1.set_xlabel('ground truth 2d joints', fontsize=10)
    plot_2d_hand(ax2, pred_joints_2d[jointsMapSMPLXToSimple], order='uv')
    ax2.set_xlabel('predict 2d joints', fontsize=10)
    plot_3d_hand(ax3, gt_joints_3d[jointsMapSMPLXToSimple])
    ax3.set_xlabel('ground truth 3d joints', fontsize=10)
    plot_3d_hand(ax4, pred_joints_3d[jointsMapSMPLXToSimple])
    ax4.set_xlabel('predict 3d joints', fontsize=10)
    print(f'saving debug image: debug_gt_pred_{name}.png ...')
    fig.savefig(f'debug_img/debug_gt_pred_{name}')
    plt.close('all')
    
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def vis_heatmap(img,heatmap,pred_ht):
    #img=img.cpu().detach().numpy()
    #img=img.transpose((1,2,0))
    #print(img.shape)
    img=img.permute(1,2,0).cpu().numpy()
    img=np.clip(img*127.5+127.5,0,255).astype(np.uint8)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    heatmap = heatmap.squeeze().cpu() #[21, 56, 56]
    pred_ht = pred_ht.squeeze().cpu()

    for i in range(heatmap.shape[0]):
        for j in range(2):
            if j==0:
                single_map=heatmap[i]
            else:
                single_map=pred_ht[i]
            #single_map = heatmap[i]

            hm = single_map.detach().numpy()
            
            from .heatmap_coord import transfer_target
            #print(hm.shape)
            hm_coord=np.expand_dims(hm, axis=2)
            hm_coord=np.expand_dims(hm_coord, axis=0)
            #print(hm_coord.shape)
            landmark_coord=transfer_target(hm_coord, thresh=0, n_points=21)
            #print(landmark_coord)
            hm = np.maximum(hm, 0)
            hm = hm/np.max(hm)
            hm = normalization(hm)
            hm = np.uint8(255 * hm)
            hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
            
            hm = cv2.resize(hm, (224, 224))
            #hm = cv2.resize(hm, (56, 56))
            #img = cv2.resize(img, (56, 56))
            superimposed_img = hm * 0.2 + img
            
            
            coord_x, coord_y = landmark_coord[0]
            cv2.circle(superimposed_img, (int(coord_x)*4, int(coord_y)*4), 2, (0, 0, 0), thickness=-1)
            if j==0:
                out1=superimposed_img
        out_img=np.hstack((out1,superimposed_img))
        cv2.imwrite("vis/heat_map_%02d.jpg" %i , out_img)