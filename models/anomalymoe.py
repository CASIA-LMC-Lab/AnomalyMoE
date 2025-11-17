import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from sklearn.cluster import KMeans
import math

class CLUBSample(nn.Module):
    def __init__(self, dim=768):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.ReLU(),
            nn.Linear(dim//2, dim)
        )

        self.p_logvar = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.ReLU(),
            nn.Linear(dim//2, dim),
            nn.Tanh()
        )

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
     
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /2./logvar.exp()).sum(dim=1).mean(dim=0)
    

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound / 2.0

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

def get_extractor(out_channels=768):
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=7, stride=7, padding=0),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
    )


def get_autoencoder(out_channels=768):
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
        ),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=9, stride=7, padding=1
        ),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=9, stride=7, padding=1
        ),
        nn.Upsample(size=3, mode="bilinear"),
        nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=2
        ),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=7, mode="bilinear"),
        nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=2
        ),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=15, mode="bilinear"),
        nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=2
        ),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=31, mode="bilinear"),
        nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=2
        ),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=63, mode="bilinear"),
        nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=2
        ),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=31, mode="bilinear"),
        nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=2
        ),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=28, mode="bilinear"),
        nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
        ),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            in_channels=128,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        ),
    )

class ComponentAutoencoder(nn.Module):
    def __init__(self, input_dim=768, bottleneck_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts),
        )
    
    def forward(self, x):
        return self.gate(x)

def extract_component_features_from_map(feature_map, masks_list, device, dilation_kernel_size=5):

    batch_size, feat_dim, h_feat, w_feat = feature_map.shape
    
    all_component_features = []
    num_components_per_image = []

    for i in range(batch_size): 
        component_masks = masks_list[i].to(device)
        num_components = component_masks.shape[0]
        num_components_per_image.append(num_components)
        
        if num_components == 0:
            continue
            

        masks_downsampled = F.interpolate(
            component_masks.unsqueeze(1).float(), 
            size=(h_feat, w_feat), 
            mode='nearest'
        )
        

        if dilation_kernel_size > 1:

            padding = (dilation_kernel_size - 1) // 2
            masks_processed = F.max_pool2d(
                masks_downsampled,
                kernel_size=dilation_kernel_size,
                stride=1,
                padding=padding
            )
        else:

            masks_processed = masks_downsampled
        
        masked_features = feature_map[i].unsqueeze(0) * masks_processed
        
        summed_features = masked_features.sum(dim=[2, 3])
        
        mask_area = masks_processed.sum(dim=[2, 3]) + 1e-8 # 防止除以0
        
        mean_features = summed_features / mask_area
        
        all_component_features.append(mean_features)
        
    if not all_component_features:
        return torch.tensor([], device=device, dtype=feature_map.dtype), num_components_per_image

    return torch.cat(all_component_features, dim=0), num_components_per_image


import copy
class AnomalyMoE(nn.Module):
    def __init__(
        self,
        encoder,
        bottleneck,
        decoder,
        num_patch_experts=6,
        num_global_experts=6,
        num_component_experts=6,
        topk=3,
        target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
        fuse_layer_encoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
        fuse_layer_decoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
        mask_neighbor_size=0,
        remove_class_token=False,
        encoder_require_grad_layer=[],
    ) -> None:
        super(AnomalyMoE, self).__init__()
        self.encoder = encoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        self.top_k = topk

        if not hasattr(self.encoder, "num_register_tokens"):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

        # --- 实例化专家模块 ---
        self.embed_dim = self.encoder.embed_dim

        self.num_patch_experts = num_patch_experts
        self.num_global_experts = num_global_experts
        self.num_component_experts = num_component_experts

        # 1. Patch-level Experts
        self.patch_experts = nn.ModuleList()
        for _ in range(num_patch_experts):
            self.patch_experts.append(
                nn.ModuleDict(
                    {
                        "bottleneck": copy.deepcopy(bottleneck),
                        "decoder": copy.deepcopy(decoder),
                    }
                )
            )

        self.global_experts = nn.ModuleList()
        for _ in range(num_global_experts):
            self.global_experts.append(
                nn.ModuleDict(
                    {
                        "autoencoder": get_autoencoder(out_channels=self.embed_dim),
                        "student": get_extractor(
                            out_channels=self.embed_dim
                        )
                    }
                )
            )
        self.component_experts = nn.ModuleList(
            [
                ComponentAutoencoder(input_dim=self.embed_dim)
                for _ in range(num_component_experts)
            ]
        )

        self.total_experts = (
            num_patch_experts + num_global_experts + num_component_experts
        )
        self.gating_network = GatingNetwork(
            input_dim=self.embed_dim, num_experts=self.total_experts
        )


    def forward(self, image_st, image_ae, masks=None):

        x = self.encoder.prepare_tokens(image_st)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            with torch.no_grad():
                x = blk(x)
            if i in self.target_layers:
                en_list.append(x)

        en_list_ae = []
        with torch.no_grad():
            temp_x_ae = self.encoder.prepare_tokens(image_ae)
            for i, blk in enumerate(self.encoder.blocks):
                with torch.no_grad():
                    temp_x_ae = blk(temp_x_ae)
                if i in self.target_layers:
                    en_list_ae.append(temp_x_ae)

        side = int(
            math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens)
        )

        cls_token = x[:, 0, :]

        raw_logits = self.gating_network(cls_token) 

        # Top-K routing
        top_k_logits, top_k_indices = torch.topk(raw_logits, self.top_k, dim=1)
        expert_mask = F.one_hot(top_k_indices, self.total_experts).sum(
            1
        )  # Binary mask of selected experts

        # Softmax is applied only to the selected experts
        routing_weights_sparse = F.softmax(top_k_logits, dim=1)

        # Scatter the sparse weights back to the full dimension for easy use
        routing_weights = torch.zeros_like(raw_logits).scatter_(
            1, top_k_indices, routing_weights_sparse
        )

        expert_outputs = {}
        patch_expert_outputs = {"recon_maps": [], "features": []}
        en_maps_fused_for_component = [
            self.fuse_feature([en_list[idx] for idx in idxs])
            for idxs in [[0, 1, 2, 3, 4, 5, 6, 7]]
        ]
        en_maps_fused = [
            self.fuse_feature([en_list[idx] for idx in idxs])
            for idxs in self.fuse_layer_encoder
        ]
        en_maps_fused_ae = [
            self.fuse_feature([en_list_ae[idx] for idx in idxs])
            for idxs in self.fuse_layer_encoder
        ]

        if not self.remove_class_token:
            current_en_maps = [
                e[:, 1 + self.encoder.num_register_tokens :, :] for e in en_maps_fused
            ]
            en_maps_fused_for_component = [
                e[:, 1 + self.encoder.num_register_tokens :, :]
                for e in en_maps_fused_for_component
            ]
            en_maps_fused_ae = [
                e[:, 1 + self.encoder.num_register_tokens :, :]
                for e in en_maps_fused_ae
            ]

        en_maps_fused_for_component = [
            e.permute(0, 2, 1).reshape(image_st.shape[0], -1, side, side)
            for e in en_maps_fused_for_component
        ]
        current_en_maps = [
            e.permute(0, 2, 1).reshape(image_st.shape[0], -1, side, side)
            for e in current_en_maps
        ]
        en_ae_maps_final = [
            e.permute(0, 2, 1).reshape(image_st.shape[0], -1, side, side)
            for e in en_maps_fused_ae
        ]

        for i, expert in enumerate(self.patch_experts):
            if True:
                x_for_decoder = self.fuse_feature(en_list)
                for blk in expert.bottleneck:
                    x_for_decoder = blk(x_for_decoder)

                de_list = []
                for blk in expert.decoder:
                    x_for_decoder = blk(x_for_decoder)
                    de_list.append(x_for_decoder)
                de_list = de_list[::-1]

                de_maps_fused = [
                    self.fuse_feature([de_list[idx] for idx in idxs])
                    for idxs in self.fuse_layer_decoder
                ]

                if not self.remove_class_token:
                    current_de_maps = [
                        d[:, 1 + self.encoder.num_register_tokens :, :]
                        for d in de_maps_fused
                    ]
                else:
                    current_de_maps = de_maps_fused

                current_de_maps = [
                    d.permute(0, 2, 1).reshape(image_st.shape[0], -1, side, side)
                    for d in current_de_maps
                ]

                patch_expert_outputs["recon_maps"].append(
                    {"en_maps": current_en_maps, "de_maps": current_de_maps}
                )
                patch_expert_outputs["features"].append(
                    current_de_maps[-1].mean(dim=[2, 3])
                ) 
        expert_outputs["patch"] = patch_expert_outputs

        global_expert_outputs = {"recon_outputs": [], "features": []}
        for i, expert in enumerate(self.global_experts):
            expert_idx = self.num_patch_experts + i
            if True:
                ae_out = expert.autoencoder(image_ae)
                student_out = expert.student(
                    image_ae
                )  
                global_expert_outputs["recon_outputs"].append(
                    {"ae_output": ae_out, "student_output": student_out}
                )
                global_expert_outputs["features"].append(student_out.mean(dim=[2, 3]))
        expert_outputs["global"] = global_expert_outputs

        # c. Component Experts
        component_expert_outputs = {"recon_feats": [], "features": []}

        source_feature_map = en_maps_fused_for_component[-1]
        orig_comp_feats, num_comps = extract_component_features_from_map(
            source_feature_map, masks, image_st.device
        )

        expert_outputs["component_orig"] = orig_comp_feats
        expert_outputs["num_components"] = num_comps

        if orig_comp_feats is not None and orig_comp_feats.numel() > 0:
            for i, expert in enumerate(self.component_experts):
                expert_idx = self.num_patch_experts + self.num_global_experts + i
                if True:
                    recon_feats = expert(orig_comp_feats)
                    component_expert_outputs["features"].append(recon_feats)
                    component_expert_outputs["recon_feats"].append(recon_feats)
        else:
            for _ in self.component_experts:
                component_expert_outputs["recon_feats"].append(
                    torch.tensor([], device=image_st.device)
                )
        expert_outputs["component"] = component_expert_outputs

        final_outputs = {
            "routing_weights": routing_weights,
            "en_ae_maps": en_ae_maps_final,
            "expert_outputs": expert_outputs,
            "raw_logits": raw_logits,
            "expert_mask": expert_mask.float(),
        }

        return final_outputs

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)
