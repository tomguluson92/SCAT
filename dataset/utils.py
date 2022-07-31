import cv2
import time
import numpy as np
from numpy.lib.stride_tricks import as_strided
import numbers
def view_as_windows(arr_in, window_shape, step=1):
    """Rolling window view of the input n-dimensional array.

    Windows are overlapping views of the input array, with adjacent windows
    shifted by a single row or column (or an index of a higher dimension).

    Parameters
    ----------
    arr_in : ndarray
        N-d input array.
    window_shape : integer or tuple of length arr_in.ndim
        Defines the shape of the elementary n-dimensional orthotope
        (better know as hyperrectangle [1]_) of the rolling window view.
        If an integer is given, the shape will be a hypercube of
        sidelength given by its value.
    step : integer or tuple of length arr_in.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.

    Returns
    -------
    arr_out : ndarray
        (rolling) window view of the input array.

    Notes
    -----
    One should be very careful with rolling views when it comes to
    memory usage.  Indeed, although a 'view' has the same memory
    footprint as its base array, the actual array that emerges when this
    'view' is used in a computation is generally a (much) larger array
    than the original, especially for 2-dimensional arrays and above.

    For example, let us consider a 3 dimensional array of size (100,
    100, 100) of ``float64``. This array takes about 8*100**3 Bytes for
    storage which is just 8 MB. If one decides to build a rolling view
    on this array with a window of (3, 3, 3) the hypothetical size of
    the rolling view (if one was to reshape the view for example) would
    be 8*(100-3+1)**3*3**3 which is about 203 MB! The scaling becomes
    even worse as the dimension of the input array becomes larger.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Hyperrectangle

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.util.shape import view_as_windows
    >>> A = np.arange(4*4).reshape(4,4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> window_shape = (2, 2)
    >>> B = view_as_windows(A, window_shape)
    >>> B[0, 0]
    array([[0, 1],
           [4, 5]])
    >>> B[0, 1]
    array([[1, 2],
           [5, 6]])

    >>> A = np.arange(10)
    >>> A
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> window_shape = (3,)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (8, 3)
    >>> B
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5],
           [4, 5, 6],
           [5, 6, 7],
           [6, 7, 8],
           [7, 8, 9]])

    >>> A = np.arange(5*4).reshape(5, 4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19]])
    >>> window_shape = (4, 3)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (2, 2, 4, 3)
    >>> B  # doctest: +NORMALIZE_WHITESPACE
    array([[[[ 0,  1,  2],
             [ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14]],
            [[ 1,  2,  3],
             [ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15]]],
           [[[ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14],
             [16, 17, 18]],
            [[ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15],
             [17, 18, 19]]]])
    """

    # -- basic checks on arguments
    if not isinstance(arr_in, np.ndarray):
        raise TypeError("`arr_in` must be a numpy ndarray")

    ndim = arr_in.ndim
    
    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = np.array(arr_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)
    
    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")
    
    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr_in.strides)

    indexing_strides = arr_in[slices].strides

    win_indices_shape = (((np.array(arr_in.shape) - np.array(window_shape))
                          // np.array(step)) + 1)

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
    return arr_out

def split_into_chunks(vid_names, seqlen, stride):
    video_start_end_indices = []

    video_names, group = np.unique(vid_names, return_index=True)
    
    perm = np.argsort(group)
    video_names, group = video_names[perm], group[perm]
    #print(video_names,group)
    indices = np.split(np.arange(0, vid_names.shape[0]), group[1:])
    #print(indices)
    for idx in range(len(video_names)):
        indexes = indices[idx]
        
        if indexes.shape[0] < seqlen:
            continue
        chunks = view_as_windows(indexes, (seqlen,), step=stride)
        start_finish = chunks[:, (0, -1)].tolist()
        #print(start_finish)
        video_start_end_indices += start_finish

    return video_start_end_indices

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    # 限制最小的值
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    # 一个圆对应内切正方形的高斯分布
    x, y = int(center[0]), int(center[1])
    width, height = heatmap.shape
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius +
                               bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        # 将高斯分布覆盖到heatmap上，取最大，而不是叠加
    return heatmap

def motion_blur(img):
    img=np.asarray(img)
    # Specify the kernel size. 
    # The greater the size, the more the motion. 
    
    # 刷分.
#     kernel_size = random.randint(1, 10)
    # 业务实际使用. 2021.03.10 week10
#     kernel_size = random.randint(7, 15)  #v1
    kernel_size = random.randint(5, 15)  # v2

    # Create the vertical kernel. 
    kernel_v = np.zeros((kernel_size, kernel_size)) 

    # Create a copy of the same for creating the horizontal kernel. 
    kernel_h = np.copy(kernel_v) 

    # Fill the middle row with ones. 
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 

    # Normalize. 
    kernel_v /= kernel_size 
    kernel_h /= kernel_size 

    # Apply the vertical kernel. 
    vertical_mb = cv2.filter2D(img, -1, kernel_v) 

    # Apply the horizontal kernel. 
    horizonal_mb = cv2.filter2D(img, -1, kernel_h) 
    choice=random.randint(0, 1)
    if choice==0:
        blur_img=vertical_mb
    else:
        blur_img=horizonal_mb
    blur_img = Image.fromarray(np.uint8(blur_img)).convert('RGB')

    return blur_img

def grabcut(img,gt_joints_2d):
    img=img.convert('RGB')
    
    img=np.array(img)
    
    img = img[:, :, ::-1].copy()
    #img=np.clip(img*127.5+127.5,0,255).astype(np.uint8)
    
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img3=img.copy()
    #print(img.shape)
    #plt.imshow(img)

    mask = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (10,10,img.shape[0]-10,img.shape[1]-10)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img2 = img*mask2[:,:,np.newaxis]
    #plt.imshow(img2),plt.colorbar(),plt.show()


    j_2d = gt_joints_2d.reshape(21, 2)#[jointsMapSMPLXToSimple]
    newmask = np.zeros(img.shape[:2], np.uint8)
    newmask[:] = (127)
    bones = [(0, 1),(1, 2),(2, 3),(3, 4), (0, 5),(5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14),(14, 15), (15, 16),(0, 17),(17, 18),(18, 19),(19, 20),(5,17)]
    for j,k in bones:
        cv2.line(newmask,(int(j_2d[j][0]),int(j_2d[j][1])),(int(j_2d[k][0]),int(j_2d[k][1])),(255,255,255),int(img.shape[0]/25))
    #plt.imshow(newmask)

    newmask2 = np.zeros(img.shape[:2], np.uint8)
    for j,k in bones:
        cv2.line(newmask2,(int(j_2d[j][0]),int(j_2d[j][1])),(int(j_2d[k][0]),int(j_2d[k][1])),(255,255,255),int(img.shape[0]/4))
    #plt.imshow(newmask2)
    
    t1 = time.time()
    mask[newmask2 == 0] = 0
    mask[newmask == 255] = 1
    mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    bg_img= np.ones((img.shape), np.uint8)*255
    img = img*mask[:,:,np.newaxis]+ (1 - mask[:,:,np.newaxis]) * bg_img
    #plt.imshow(img),plt.colorbar(),plt.show()
    #print('Grabcut in %.2f seconds' % ( time.time() - t1))

#     fig=plt.figure()
#     ax1 = fig.add_subplot(151)
#     ax2 = fig.add_subplot(152)
#     ax3 = fig.add_subplot(153)
#     ax4 = fig.add_subplot(154)
#     ax5 = fig.add_subplot(155)
#     ax1.imshow(img3)
#     ax1.set_xticks([])
#     ax1.set_yticks([])
#     ax2.imshow(newmask)
#     ax2.set_xticks([])
#     ax2.set_yticks([])
#     ax3.imshow((1 - mask[:,:,np.newaxis]))
#     ax3.set_xticks([])
#     ax3.set_yticks([])
#     ax4.imshow(img)
#     ax4.set_xticks([])
#     ax4.set_yticks([])
    
#     #inpainting
    t2 = time.time()
    
    
    dst = cv2.inpaint(img3,mask,3,cv2.INPAINT_TELEA)
#     #dst = cv2.inpaint(img3,mask,3,cv2.INPAINT_NS)
#     ax5.imshow(dst)
#     ax5.set_xticks([])
#     ax5.set_yticks([])
    #print('Inpainting in %.2f seconds' % ( time.time() - t2))
    
    final_mask=(1 - mask[:,:,np.newaxis])
    
    return final_mask,dst


if __name__=='__main__':
    seqlen=10
    stride=seqlen
    train_list_path='train_seq.txt'
    train_list = open(train_list_path).readlines()
    train_list=np.array(train_list)
    vid_indices=split_into_chunks(train_list,seqlen,stride)
    print(f'HO3D dataset number of videos: {len(vid_indices)}')
