import math
import trimesh
import pyrender
import numpy as np
from pyrender.constants import RenderFlags

import pickle


class Renderer:
    def __init__(self, resolution=(224,224), orig_img=False, wireframe=False):
        self.resolution = resolution

        dd = pickle.load(open('extra_data/MANO_RIGHT.pkl', 'rb'),encoding='latin1')
        self.faces = dd['f']
        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=1.0
        )

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)

    def render(self, img, verts, cam, angle=None, axis=None, mesh_filename=None, color=[1.0, 1.0, 0.9]):

        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)

        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        mesh.apply_transform(Rx)

        if mesh_filename is not None:
            mesh.export(mesh_filename)
            print(f'Saved mesh at {mesh_filename}')

        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        sx,  tx, ty = cam

        camera = WeakPerspectiveCamera(
            scale=[sx, sx],
            translation=[tx, ty],
            zfar=1000.
        )

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        mesh_node = self.scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img
        image = output_img.astype(np.uint8)

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)

        return image
    
class Visualizer(object):
    def __init__(
        self,
        rendererType ='opengl_gui'          #nongui or gui
    ):
        self.renderer = meshRenderer.meshRenderer()
        self.renderer.setRenderMode('geo')
        self.renderer.offscreenMode(True)
    
    def render_pred_verts(self, img_original, pred_mesh_list):
        res_img = img_original.copy()

        pred_mesh_list_offset =[]
        for mesh in pred_mesh_list:

            # Mesh vertices have in image cooridnate (left, top origin)
            # Move the X-Y origin in image center
            mesh_offset = mesh['vertices'].copy()
            mesh_offset[:,0] -= img_original.shape[1]*0.5
            mesh_offset[:,1] -= img_original.shape[0]*0.5
            pred_mesh_list_offset.append( {'ver': mesh_offset, 'f':mesh['faces'] })# verts = mesh['vertices']
            # faces = mesh['faces']
#         if self.rendererType =="opengl_gui":
#             self._visualize_gui_naive(pred_mesh_list_offset, img_original=res_img)
#             overlaidImg = None
        
        self._visualize_screenless_naive(pred_mesh_list_offset, img_original=res_img)
        overlaidImg = self.renderout['render_camview']
        # sideImg = self.renderout['render_sideview']

        return overlaidImg
    
    def _visualize_screenless_naive(self, meshList, skelList=None, body_bbox_list=None, img_original=None, show_side = False, vis=False, maxHeight = 1080):
        
        """
            args:
                meshList: list of {'ver': pred_vertices, 'f': smpl.faces}
                skelList: list of [JointNum*3, 1]       (where 1 means num. of frames in glviewer)
                bbr_list: list of [x,y,w,h] 
            output:
                #Rendered images are saved in 
                self.renderout['render_camview']
                self.renderout['render_sideview']

            #Note: The size of opengl rendering is restricted by the current screen size. Set the maxHeight accordingly

        """
        assert self.renderer is not None

        if len(meshList)==0:
               # sideImg = cv2.resize(sideImg, (renderImg.shape[1], renderImg.shape[0]) )
            self.renderout  ={}
            self.renderout['render_camview'] = img_original.copy()

            blank = np.ones(img_original.shape, dtype=np.uint8)*255       #generate blank image
            self.renderout['render_sideview'] = blank
            
            return
        
        if body_bbox_list is not None:
            for bbr in body_bbox_list:
                viewer2D.Vis_Bbox(img_original,bbr)
        # viewer2D.ImShow(img_original)

        #Check image height
        imgHeight, imgWidth = img_original.shape[0], img_original.shape[1]
        if maxHeight <imgHeight:        #Resize
            ratio = maxHeight/imgHeight

            #Resize Img
            newWidth = int(imgWidth*ratio)
            newHeight = int(imgHeight*ratio)
            img_original_resized = cv2.resize(img_original, (newWidth,newHeight))

            #Resize skeleton
            for m in meshList:
                m['ver'] *=ratio

            if skelList is not None:
                for s in skelList:
                    s *=ratio


        else:
            img_original_resized = img_original

        self.renderer.setWindowSize(img_original_resized.shape[1], img_original_resized.shape[0])
        self.renderer.setBackgroundTexture(img_original_resized)
        self.renderer.setViewportSize(img_original_resized.shape[1], img_original_resized.shape[0])

        # self.renderer.add_mesh(meshList[0]['ver'],meshList[0]['f'])
        self.renderer.clear_mesh()
        for mesh in meshList:
            self.renderer.add_mesh(mesh['ver'],mesh['f'])
        self.renderer.showBackground(True)
        self.renderer.setWorldCenterBySceneCenter()
        self.renderer.setCameraViewMode("cam")
        # self.renderer.setViewportSize(img_original_resized.shape[1], img_original_resized.shape[0])
                
        self.renderer.display()
        renderImg = self.renderer.get_screen_color_ibgr()

        if vis:
            viewer2D.ImShow(renderImg,waitTime=1,name="rendered")

        ###Render Side View
        if show_side:
            self.renderer.setCameraViewMode("free")     
            self.renderer.setViewAngle(90,20)
            self.renderer.showBackground(False)
            self.renderer.setViewportSize(img_original_resized.shape[1], img_original_resized.shape[0])
            self.renderer.display()
            sideImg = self.renderer.get_screen_color_ibgr()        #Overwite on rawImg

            if vis:
                viewer2D.ImShow(sideImg,waitTime=0,name="sideview")
        
        # sideImg = cv2.resize(sideImg, (renderImg.shape[1], renderImg.shape[0]) )
        self.renderout  ={}
        self.renderout['render_camview'] = renderImg

        if show_side:
            self.renderout['render_sideview'] = sideImg
