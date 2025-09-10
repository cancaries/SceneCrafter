import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
import torch.nn as nn
import numpy as np
import os
from lib.config import cfg
from lib.models.gaussian_model import GaussianModel
from lib.utils.general_utils import quaternion_to_matrix, inverse_sigmoid, matrix_to_quaternion, get_expon_lr_func, quaternion_raw_multiply
from lib.utils.sh_utils import RGB2SH, IDFT
from lib.utils.graphics_utils import calculate_shadow_direction
from lib.datasets.base_readers import fetchPly
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
import random
random.seed(0)

class GaussianModelActor(GaussianModel):
    def __init__(
        self, 
        model_name, 
        obj_meta, 
    ):
        # parse obj_meta
        self.obj_meta = obj_meta
        
        self.obj_class = obj_meta['class']
        self.obj_class_label = obj_meta['class_label']
        self.deformable = obj_meta['deformable']         
        self.start_frame = obj_meta['start_frame']
        self.start_timestamp = obj_meta['start_timestamp']
        self.end_frame = obj_meta['end_frame']
        self.end_timestamp = obj_meta['end_timestamp']
        self.track_id = obj_meta['track_id']

        self.save = obj_meta['save']
        self.type = obj_meta['type']
        
        # fourier spherical harmonics
        self.fourier_dim = cfg.model.gaussian.get('fourier_dim', 1)
        self.fourier_scale = cfg.model.gaussian.get('fourier_scale', 1.)
        
        # bbox
        length, width, height = obj_meta['length'], obj_meta['width'], obj_meta['height']
        self.bbox = np.array([length, width, height]).astype(np.float32)
        xyz = torch.tensor(self.bbox).float().cuda()
        self.min_xyz, self.max_xyz =  -xyz/2., xyz/2.  
        
        extent = max(length*1.5/cfg.data.box_scale, width*1.5/cfg.data.box_scale, height) / 2.
        self.extent = torch.tensor([extent]).float().cuda()   

        num_classes = 1 if cfg.data.get('use_semantic', False) else 0
        self.num_classes_global = cfg.data.num_classes if cfg.data.get('use_semantic', False) else 0        
        super().__init__(model_name=model_name, num_classes=num_classes)
        
        self.flip_prob = cfg.model.gaussian.get('flip_prob', 0.) if not self.deformable else 0.
        self.flip_axis = 1 

        self.spatial_lr_scale = extent

        self.global_light_direction = torch.tensor([0., 0., 1.]).float().cuda()
        self.cur_yaw = None
        self.shadow_ply = {}
        self.closest_shadow_ply = None

    def get_extent(self):
        max_scaling = torch.max(self.get_scaling, dim=1).values

        extent_lower_bound = torch.topk(max_scaling, int(self.get_xyz.shape[0] * 0.1), largest=False).values[-1] / self.percent_dense
        extent_upper_bound = torch.topk(max_scaling, int(self.get_xyz.shape[0] * 0.1), largest=True).values[-1] / self.percent_dense
        
        extent = torch.clamp(self.extent, min=extent_lower_bound, max=extent_upper_bound)        
        print(f'extent: {extent.item()}, extent bound: [{extent_lower_bound}, {extent_upper_bound}]')

        return extent
    @property
    def get_semantic(self):
        semantic = torch.zeros((self.get_xyz.shape[0], self.num_classes_global)).float().cuda()
        if self.semantic_mode == 'logits':
            semantic[:, self.obj_class_label] = self._semantic[:, 0] # ubounded semantic        
        elif self.semantic_mode == 'probabilities':
            semantic[:, self.obj_class_label] = torch.nn.functional.sigmoid(self._semantic[:, 0]) # 0 ~ 1

        return semantic 
    
    def get_features_fourier(self, frame=0):
        if self.closest_shadow_ply:
            features_dc_original = self._features_dc # [N, C, 3]
            features_rest_original = self._features_rest # [N, sh, 3]
            features_dc_shadow = self.closest_shadow_ply['features_dc'] # [N, C, 3]
            features_rest_shadow = self.closest_shadow_ply['features_rest'] # [N, sh, 3]
            features_dc = torch.cat([features_dc_original, features_dc_shadow], dim=0) # [N+n, C, 3]
            features_rest = torch.cat([features_rest_original, features_rest_shadow], dim=0) # [N+n, (sh + 1) * 2, 3]
            features = torch.cat([features_dc, features_rest], dim=1) # [N, (sh + 1) * C, 3]
        else:
            features_dc = self._features_dc # [N, C, 3]
            features_rest = self._features_rest # [N, sh, 3]
            features = torch.cat([features_dc, features_rest], dim=1) # [N, (sh + 1) * C, 3]
        
        return features

    @property
    def get_scaling(self):
        if self.closest_shadow_ply:
            scaling_original = self._scaling
            scaling_shadow = self.closest_shadow_ply['scaling']
            scaling = torch.cat([scaling_original, scaling_shadow], dim=0)
        else:
            scaling = self._scaling
        return self.scaling_activation(scaling)

    @property
    def get_rotation(self):
        if self.closest_shadow_ply:
            rotation_original = self._rotation
            rotation_shadow = self.closest_shadow_ply['rotation']
            rotation = torch.cat([rotation_original, rotation_shadow], dim=0)
        else:
            rotation = self._rotation
        return rotation

    @property
    def get_xyz(self):
        if self.closest_shadow_ply:
            xyz_original = self._xyz
            xyz_shadow = self.closest_shadow_ply['xyz']
            xyz = torch.cat([xyz_original, xyz_shadow], dim=0)
        else:
            xyz = self._xyz
        return xyz

    @property
    def get_opacity(self):
        if self.closest_shadow_ply:
            opacity_original = self._opacity
            opacity_shadow = self.closest_shadow_ply['opacity']
            opacity = torch.cat([opacity_original, opacity_shadow], dim=0)
        else:
            opacity = self._opacity
        return self.opacity_activation(opacity)
    # @property

    def load_shadow_ply(self, ply_path):
        ply_data_dict = {}
        plydata = PlyData.read(ply_path)
        plydata = plydata.elements[0]

        xyz = np.stack((np.asarray(plydata["x"]),
                        np.asarray(plydata["y"]),
                        np.asarray(plydata["z"])),  axis=1)
        opacities = np.asarray(plydata["opacity"])[..., np.newaxis]
 
        base_f_names = [p.name for p in plydata.properties if p.name.startswith("f_dc_")]
        base_f_names = sorted(base_f_names, key = lambda x: int(x.split('_')[-1]))
        extra_f_names = [p.name for p in plydata.properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        features_dc = np.zeros((xyz.shape[0], len(base_f_names)))
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(base_f_names):
            features_dc[:, idx] = np.asarray(plydata[attr_name])
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata[attr_name])
        features_dc = features_dc.reshape(features_dc.shape[0], 3, -1)
        features_extra = features_extra.reshape(features_extra.shape[0], 3, -1)
       
        scale_names = [p.name for p in plydata.properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata[attr_name])

        rot_names = [p.name for p in plydata.properties if p.name.startswith("rot_")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata[attr_name])


        ply_data_dict['xyz'] =  nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        ply_data_dict['features_dc'] =  nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        ply_data_dict['features_rest'] = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        ply_data_dict['opacity'] =  nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        ply_data_dict['scaling'] =  nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        ply_data_dict['rotation'] =  nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        return ply_data_dict


    def load_custom_ply(self):
        folder_path = cfg.ply_model_path
        agent_type = self.type
        ply_candidates = [f for f in os.listdir(folder_path) if f.endswith('.ply') and 'shadow' not in f]
        ply_files = [f for f in os.listdir(folder_path) if f.startswith(f'{agent_type}') and f.endswith('.ply') and 'shadow' not in f]
        if len(ply_files) == 0:
            ply_files = ply_candidates
        try:
            random_ply = random.choice(ply_files)
            self.type = random_ply.split('.')[0]
            agent_type = self.type
            random_path = os.path.join(folder_path, random_ply)
            shadow_ply_paths = [f for f in os.listdir(folder_path) if f.startswith(f'{agent_type}') and f.endswith('.ply') and 'shadow' in f]
        except:
            random_ply = random.choice(ply_candidates)
            self.type = random_ply.split('.')[0]
            agent_type = self.type
            random_path = os.path.join(folder_path, random_ply)
            shadow_ply_paths = None
        plydata = PlyData.read(random_path)
        plydata = plydata.elements[0]

        xyz = np.stack((np.asarray(plydata["x"]),
                        np.asarray(plydata["y"]),
                        np.asarray(plydata["z"])),  axis=1)
        opacities = np.asarray(plydata["opacity"])[..., np.newaxis]
 
        base_f_names = [p.name for p in plydata.properties if p.name.startswith("f_dc_")]
        base_f_names = sorted(base_f_names, key = lambda x: int(x.split('_')[-1]))
        extra_f_names = [p.name for p in plydata.properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        features_dc = np.zeros((xyz.shape[0], len(base_f_names)))
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(base_f_names):
            features_dc[:, idx] = np.asarray(plydata[attr_name])
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata[attr_name])
        features_dc = features_dc.reshape(features_dc.shape[0], 3, -1)
        features_extra = features_extra.reshape(features_extra.shape[0], 3, -1)
       
        scale_names = [p.name for p in plydata.properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata[attr_name])

        rot_names = [p.name for p in plydata.properties if p.name.startswith("rot_")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata[attr_name])

        semantic_names = [p.name for p in plydata.properties if p.name.startswith("semantic_")]
        semantic_names = sorted(semantic_names, key = lambda x: int(x.split('_')[-1]))
        semantic = np.zeros((xyz.shape[0], len(semantic_names)))
        for idx, attr_name in enumerate(semantic_names):
            semantic[:, idx] = np.asarray(plydata[attr_name])
 
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._semantic = nn.Parameter(torch.tensor(semantic, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        if shadow_ply_paths is not None:
            for shadow_ply_name in shadow_ply_paths:
                shadow_ply_type = f"{shadow_ply_name.split('.ply')[0].split('_')[-2]}_{shadow_ply_name.split('.ply')[0].split('_')[-1]}"
                shadow_plydata_dict = self.load_shadow_ply(os.path.join(folder_path, shadow_ply_name))
                self.shadow_ply[shadow_ply_type] = shadow_plydata_dict

    def create_from_pcd(self, spatial_lr_scale):
        folder_path = cfg.ply_model_path
        agent_type = self.type
    
        if agent_type == 'Porsche-911-4s-final' or agent_type == 'Tesla_cybertruck':
            agent_type = 'Audi'
        ply_candidates = [f for f in os.listdir(folder_path) if f.endswith('.ply')]
        ply_files = [f for f in os.listdir(folder_path) if f.startswith(f'{agent_type}') and f.endswith('.ply')]
        if len(ply_files) == 0:
            ply_files = ply_candidates
        try:
            random_ply = random.choice(ply_files)
            random_path = os.path.join(folder_path, random_ply)
        except:
            random_ply = random.choice(ply_candidates)
            random_path = os.path.join(folder_path, random_ply)

        if self.save == 1:
            pointcloud_path = random_path
  
        if os.path.exists(pointcloud_path):
            pcd = fetchPly(pointcloud_path)
            pointcloud_xyz = np.asarray(pcd.points)
            if pointcloud_xyz.shape[0] < 2000:
                self.random_initialization = True
            else:
                self.random_initialization = False
        else:
            self.random_initialization = True

        if self.random_initialization is True:
            points_dim = 20
            print(f'Creating random pointcloud for {self.model_name}')
            points_x, points_y, points_z = np.meshgrid(
                np.linspace(-1., 1., points_dim), np.linspace(-1., 1., points_dim), np.linspace(-1., 1., points_dim),
            )
            
            points_x = points_x.reshape(-1)
            points_y = points_y.reshape(-1)
            points_z = points_z.reshape(-1)

            bbox_xyz_scale = self.bbox / 2.
            pointcloud_xyz = np.stack([points_x, points_y, points_z], axis=-1)
            pointcloud_xyz = pointcloud_xyz * bbox_xyz_scale            
            pointcloud_rgb = np.random.rand(*pointcloud_xyz.shape).astype(np.float32)  
        elif not self.deformable and self.flip_prob > 0.:          
            pcd = fetchPly(pointcloud_path)
            pointcloud_xyz = np.asarray(pcd.points)
            pointcloud_rgb = np.asarray(pcd.colors)
            num_pointcloud_1 = (pointcloud_xyz[:, self.flip_axis] > 0).sum()
            num_pointcloud_2 = (pointcloud_xyz[:, self.flip_axis] < 0).sum()
            if num_pointcloud_1 >= num_pointcloud_2:
                pointcloud_xyz_part = pointcloud_xyz[pointcloud_xyz[:, self.flip_axis] > 0]
                pointcloud_rgb_part = pointcloud_rgb[pointcloud_xyz[:, self.flip_axis] > 0]
            else:
                pointcloud_xyz_part = pointcloud_xyz[pointcloud_xyz[:, self.flip_axis] < 0]
                pointcloud_rgb_part = pointcloud_rgb[pointcloud_xyz[:, self.flip_axis] < 0]
            pointcloud_xyz_flip = pointcloud_xyz_part.copy()
            pointcloud_xyz_flip[:, self.flip_axis] *= -1
            pointcloud_rgb_flip = pointcloud_rgb_part.copy()
            pointcloud_xyz = np.concatenate([pointcloud_xyz, pointcloud_xyz_flip], axis=0)
            pointcloud_rgb = np.concatenate([pointcloud_rgb, pointcloud_rgb_flip], axis=0)
        else:
            pcd = fetchPly(pointcloud_path)
            pointcloud_xyz = np.asarray(pcd.points)
            pointcloud_rgb = np.asarray(pcd.colors)
            
        fused_point_cloud = torch.tensor(np.asarray(pointcloud_xyz)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pointcloud_rgb)).float().cuda())

        features_dc = torch.zeros((fused_color.shape[0], 3, self.fourier_dim)).float().cuda()
        features_rest = torch.zeros(fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1).float().cuda()
        features_dc[:, :3, 0] = fused_color

        print(f"Number of points at initialisation for {self.model_name}: ", fused_point_cloud.shape[0])
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pointcloud_xyz)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4)).cuda()
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1))).float().cuda()
        semantics = torch.zeros((fused_point_cloud.shape[0], self.num_classes)).float().cuda()
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        
        self._features_dc = nn.Parameter(features_dc.transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest.transpose(1, 2).contiguous().requires_grad_(True))
        
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._semantic = nn.Parameter(semantics.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def training_setup(self):
        args = cfg.optim

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.active_sh_degree = 0
                
        tag = 'obj'
        position_lr_init = args.get('position_lr_init_{}'.format(tag), args.position_lr_init)
        position_lr_final = args.get('position_lr_final_{}'.format(tag), args.position_lr_final)
        scaling_lr = args.get('scaling_lr_{}'.format(tag), args.scaling_lr)
        feature_lr = args.get('feature_lr_{}'.format(tag), args.feature_lr)
        semantic_lr = args.get('semantic_lr_{}'.format(tag), args.semantic_lr)
        rotation_lr = args.get('rotation_lr_{}'.format(tag), args.rotation_lr)
        opacity_lr = args.get('opacity_lr_{}'.format(tag), args.opacity_lr)
        feature_rest_lr = args.get('feature_rest_lr_{}'.format(tag), feature_lr / 20.0)

        l = [
            {'params': [self._xyz], 'lr': position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': feature_rest_lr, "name": "f_rest"},
            {'params': [self._opacity], 'lr': opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': rotation_lr, "name": "rotation"},
            {'params': [self._semantic], 'lr': semantic_lr, "name": "semantic"},
        ]
        
        self.percent_dense = args.percent_dense
        self.percent_big_ws = args.percent_big_ws
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=position_lr_init * self.spatial_lr_scale,
            lr_final=position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=args.position_lr_delay_mult,
            max_steps=args.position_lr_max_steps
        )
        
        self.densify_and_prune_list = ['xyz, f_dc, f_rest, opacity, scaling, rotation, semantic']
        self.scalar_dict = dict()
        self.tensor_dict = dict()  
            
    def densify_and_prune(self, max_grad, min_opacity, prune_big_points):
        if not (self.random_initialization or self.deformable):
            max_grad = cfg.optim.get('densify_grad_threshold_obj', max_grad)
            if cfg.optim.get('densify_grad_abs_obj', False):
                grads = self.xyz_gradient_accum[:, 1:2] / self.denom
            else:
                grads = self.xyz_gradient_accum[:, 0:1] / self.denom
        else:
            grads = self.xyz_gradient_accum[:, 0:1] / self.denom
        
        grads[grads.isnan()] = 0.0

        # Clone and Split
        # extent = self.get_extent()
        extent = self.extent
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # Prune points below opacity
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        
        if prune_big_points:
            # Prune big points in world space
            extent = self.extent
            big_points_ws = self.get_scaling.max(dim=1).values > extent * self.percent_big_ws
            
            # Prune points outside the tracking box
            repeat_num = 2
            stds = self.get_scaling
            stds = stds[:, None, :].expand(-1, repeat_num, -1) # [N, M, 1] 
            means = torch.zeros_like(self.get_xyz)
            means = means[:, None, :].expand(-1, repeat_num, -1) # [N, M, 3]
            samples = torch.normal(mean=means, std=stds) # [N, M, 3]
            rots = quaternion_to_matrix(self.get_rotation) # [N, 3, 3]
            rots = rots[:, None, :, :].expand(-1, repeat_num, -1, -1) # [N, M, 3, 3]
            origins = self.get_xyz[:, None, :].expand(-1, repeat_num, -1) # [N, M, 3]
                        
            samples_xyz = torch.matmul(rots, samples.unsqueeze(-1)).squeeze(-1) + origins # [N, M, 3]                    
            num_gaussians = self.get_xyz.shape[0]
            points_inside_box = torch.logical_and(
                torch.all((samples_xyz >= self.min_xyz).view(num_gaussians, -1), dim=-1),
                torch.all((samples_xyz <= self.max_xyz).view(num_gaussians, -1), dim=-1),
            )
            points_outside_box = torch.logical_not(points_inside_box)           
            
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
            prune_mask = torch.logical_or(prune_mask, points_outside_box)
            
        self.prune_points(prune_mask)
        
        # Reset
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        torch.cuda.empty_cache()
        
        return self.scalar_dict, self.tensor_dict
    
    def set_max_radii(self, visibility_obj, max_radii2D):
        self.max_radii2D[visibility_obj] = torch.max(self.max_radii2D[visibility_obj], max_radii2D[visibility_obj])
    
    def set_global_light_direction(self, global_light_direction):
        self.global_light_direction = global_light_direction

    def set_cur_yaw(self, cur_yaw):
        self.cur_yaw = cur_yaw
        light_vector, _ = calculate_shadow_direction(self.global_light_direction, -self.cur_yaw)
        light_x = light_vector[0]
        light_y = light_vector[1]
        keys = self.shadow_ply.keys()
        if len(keys) == 0:
            self.closest_shadow_ply = None
        else:
            x_key = [float(key.split('_')[0]) for key in keys]
            y_key = [float(key.split('_')[1]) for key in keys]

            x_key = np.array(x_key)
            y_key = np.array(y_key)
            x_idx = np.argmin(np.abs(x_key - light_x))
            y_idx = np.argmin(np.abs(y_key - light_y))
            shadow_ply_key = f"{x_key[x_idx]}_{y_key[y_idx]}"
            self.closest_shadow_ply = self.shadow_ply[shadow_ply_key]


    def box_reg_loss(self):
        scaling_max = self.get_scaling.max(dim=1).values
        scaling_max = torch.where(scaling_max > self.extent * self.percent_dense, scaling_max, 0.)
        reg_loss = (scaling_max / self.extent).mean()
        
        return reg_loss
        
        
    
    