import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import numpy as np
import random
import os
from torch.utils.data import ConcatDataset, DistributedSampler
from datetime import datetime
from PIL import Image
import cv2
import glob
from pathlib import Path


from models.anomalymoe import AnomalyMoE
from models.anomalymoe import CLUBSample
from models import vit_encoder
from models.vision_transformer import (
    Block as VitBlock,
    bMlp,
    LinearAttention2,
)

import argparse
from utils import (
    trunc_normal_,
    WarmCosineScheduler,
    calculate_hm_loss_for_expert,
)
from torch.nn import functional as F
from functools import partial
from optimizers import StableAdamW
from torchvision import transforms

from datetime import timedelta
import itertools

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator

import cv2


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.CenterCrop(isize),
            transforms.Normalize(mean=mean_train, std=std_train),
        ]
    )
    gt_transforms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((size, size)),
            transforms.CenterCrop(isize),
            transforms.ToTensor(),
        ]
    )
    return data_transforms, gt_transforms


def filter_bg_noise(sourcepath, classname, resolution):
    base_path = Path(sourcepath)
    train_file_path = base_path / f"{classname}_heat_448" / "train" / "good"

    if not train_file_path.exists():
        return list(range(5))

    trainfiles = sorted(
        glob.glob(str(train_file_path / "*")), key=lambda x: int(Path(x).name)
    )

    if not trainfiles:
        return list(range(5))

    img0_path = trainfiles[0]
    reserve_list = []

    seg_img_list = sorted(glob.glob(os.path.join(img0_path, "heatresult[0-9].jpg")))

    if not seg_img_list:
        return list(range(5))

    for i, imgpath in enumerate(seg_img_list):
        component_index = int(Path(imgpath).stem.replace("heatresult", ""))

        gray_img = cv2.imread(imgpath, 0)
        if gray_img is None:
            continue

        h, w = gray_img.shape
        if h < 448 or w < 448:
            reserve_list.append(component_index)
            continue

        gray_cal_otsu = gray_img[10 : h - 10, 10 : w - 10]
        ret, _ = cv2.threshold(gray_cal_otsu, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh2 = cv2.threshold(gray_img, ret, 1, cv2.THRESH_BINARY)

        corner1 = np.zeros_like(thresh2)
        corner1[0:20, 0:20] = 1
        corner2 = np.zeros_like(thresh2)
        corner2[h - 20 : h, 0:20] = 1
        corner3 = np.zeros_like(thresh2)
        corner3[0:20, w - 20 : w] = 1
        corner4 = np.zeros_like(thresh2)
        corner4[h - 20 : h, w - 20 : w] = 1
        ex = (
            (corner1 * thresh2).max()
            + (corner2 * thresh2).max()
            + (corner3 * thresh2).max()
            + (corner4 * thresh2).max()
        )

        kenel_size = (11, 11)
        blurred_img = cv2.blur(gray_img, kenel_size)
        maxvalue = blurred_img.max()

        if maxvalue > 64 and ex < 3:
            reserve_list.append(component_index)

    return reserve_list


class TrainMoEDataset(torch.utils.data.Dataset):

    def __init__(self, root, transform, class_name, mask_root):
        self.root = Path(root)
        self.transform = transform
        self.class_name = class_name
        self.mask_root = Path(mask_root)

        self.img_dir = self.root / "train" / "good"
        self.img_paths = sorted(
            list(self.img_dir.glob("*.png"))
            + list(self.img_dir.glob("*.JPG"))
            + list(self.img_dir.glob("*.bmp"))
        )

        self.component_indices_to_load = filter_bg_noise(
            self.mask_root, self.class_name, resolution=448
        )
        print(
            f"[{self.class_name} - Train] Keeping component indices: {self.component_indices_to_load}"
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        img = Image.open(img_path).convert("RGB")
        img_st, img_ae = self.transform(img)

        h, w = img_st.shape[-2:]
        mask_base_dir = self.mask_root / f"{self.class_name}_heat_448" / "train"

        component_masks_list = []

        for i in self.component_indices_to_load:
            component_mask_path = mask_base_dir / f"heatresult{i}.jpg"
            if component_mask_path.exists():
                mask_img = Image.open(component_mask_path).convert("L")

                intermediate_size = 448
                resized_mask = mask_img.resize(
                    (intermediate_size, intermediate_size), Image.NEAREST
                )

                left = (intermediate_size - w) // 2
                top = (intermediate_size - h) // 2
                right = left + w
                bottom = top + h

                cropped_mask_pil = resized_mask.crop((left, top, right, bottom))

                cropped_mask_np = np.array(cropped_mask_pil)
                if cropped_mask_np.size > 0:
                    _, binary_mask_np = cv2.threshold(
                        cropped_mask_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    binary_mask = (torch.from_numpy(binary_mask_np) > 0).to(torch.uint8)
                    component_masks_list.append(binary_mask)

        if component_masks_list:
            component_masks = torch.stack(component_masks_list, dim=0)
        else:
            component_masks = torch.empty(0, h, w, dtype=torch.uint8)

        label = 0

        return (img_st, img_ae), label, component_masks, self.class_name


def calculate_esb_loss(raw_logits, expert_mask, num_experts, dataset_names, args):
    batch_size = raw_logits.shape[0]

    expert_probs_full = F.softmax(raw_logits, dim=1)
    P_j = expert_probs_full.mean(0)
    importance_loss = num_experts * torch.sum(P_j * P_j)

    C_j = expert_mask.sum(0)
    capacity = (args.capacity_factor * batch_size) / num_experts

    over_capacity = torch.relu(C_j - capacity)
    load_loss = torch.sum(over_capacity * over_capacity)

    z_loss = torch.mean(raw_logits**2)

    return importance_loss + load_loss + z_loss


class MIEstimator(nn.Module):
    def __init__(self, dim=768, exp_num=6):
        super(MIEstimator, self).__init__()
        self.patch_estimator = CLUBSample(dim)
        self.global_estimator = CLUBSample(dim)
        self.normalizer = nn.LayerNorm(dim)
        self.num = exp_num

    def forward(self, expert_outputs):
        patch_features_list = expert_outputs["patch"]["features"]
        global_features_list = expert_outputs["global"]["features"]

        idx1, idx2 = random.sample(range(len(patch_features_list)), k=2)
        patch1, patch2 = self.normalizer(patch_features_list[idx1]), self.normalizer(
            patch_features_list[idx2]
        )
        idx1, idx2 = random.sample(range(len(global_features_list)), k=2)
        global1, global2 = self.normalizer(global_features_list[idx1]), self.normalizer(
            global_features_list[idx2]
        )

        mi_loss = (
            self.patch_estimator(patch1, patch2)
            + self.global_estimator(global1, global2)
        ) / 2.0
        return mi_loss

    def train_estimator(self, expert_outputs):

        patch_features_list = expert_outputs["patch"]["features"]
        global_features_list = expert_outputs["global"]["features"]

        idx1, idx2 = random.sample(range(len(patch_features_list)), k=2)
        patch1, patch2 = self.normalizer(patch_features_list[idx1]), self.normalizer(
            patch_features_list[idx2]
        )
        idx1, idx2 = random.sample(range(len(global_features_list)), k=2)
        global1, global2 = self.normalizer(global_features_list[idx1]), self.normalizer(
            global_features_list[idx2]
        )
        est_loss = (
            self.patch_estimator.learning_loss(patch1, patch2)
            + self.global_estimator.learning_loss(global1, global2)
        ) / 2.0
        return est_loss


def calculate_moe_loss(outputs, args, dataset_names):
    routing_weights = outputs["routing_weights"]
    expert_outputs = outputs["expert_outputs"]
    en_ae_maps = outputs["en_ae_maps"]
    device = routing_weights.device
    patch_loss_scale = 1.0
    global_loss_scale = 0.1
    component_loss_scale = 0.1

    patch_recon_loss = torch.tensor(0.0, device=device)
    global_recon_loss = torch.tensor(0.0, device=device)
    component_recon_loss = torch.tensor(0.0, device=device)

    num_p, num_g, num_c = (
        args.num_patch_experts,
        args.num_global_experts,
        args.num_component_experts,
    )
    patch_weights, global_weights, component_weights = torch.split(
        routing_weights, [num_p, num_g, num_c], dim=1
    )

    patch_expert_data = expert_outputs["patch"]["recon_maps"]
    for i in range(len(patch_expert_data)):
        loss_vec = calculate_hm_loss_for_expert(
            patch_expert_data[i]["en_maps"],
            patch_expert_data[i]["de_maps"],
            p=args.p_hm,
            factor=args.factor_hm,
        )
        weighted_loss = (patch_weights[:, i] * loss_vec * patch_loss_scale).mean()
        patch_recon_loss += weighted_loss

    global_expert_data = expert_outputs["global"]["recon_outputs"]
    for i in range(len(global_expert_data)):
        ae_output = global_expert_data[i]["ae_output"]
        student_output = global_expert_data[i]["student_output"]
        target_en_ae_map = en_ae_maps[-1]
        loss_stae_vec = F.mse_loss(student_output, ae_output, reduction="none").mean(
            dim=[1, 2, 3]
        )
        ae_output_resized = F.interpolate(
            ae_output, size=target_en_ae_map.shape[-2:], mode="bilinear"
        )
        loss_ae_vec = F.mse_loss(
            ae_output_resized, target_en_ae_map, reduction="none"
        ).mean(dim=[1, 2, 3])
        weighted_loss = (
            global_weights[:, i] * (loss_ae_vec + loss_stae_vec) * global_loss_scale
        ).mean()
        global_recon_loss += weighted_loss

    orig_comp_feats = expert_outputs["component_orig"]
    recon_comp_feats_list = expert_outputs["component"]["recon_feats"]
    num_components_per_image = expert_outputs["num_components"]

    if orig_comp_feats is not None and orig_comp_feats.numel() > 0:
        comp_to_img_weights = []
        for i in range(len(num_components_per_image)):
            weights_for_img = component_weights[i, :]
            comp_to_img_weights.append(
                weights_for_img.repeat(num_components_per_image[i], 1)
            )

        if comp_to_img_weights:
            all_comp_weights = torch.cat(comp_to_img_weights, dim=0)
            for i in range(len(recon_comp_feats_list)):
                loss_per_comp = 1 - F.cosine_similarity(
                    recon_comp_feats_list[i], orig_comp_feats, dim=1
                )
                weighted_loss = (
                    all_comp_weights[:, i] * loss_per_comp * component_loss_scale
                ).mean()
                component_recon_loss += weighted_loss

    recon_loss = patch_recon_loss + global_recon_loss + component_recon_loss

    return recon_loss, patch_recon_loss, global_recon_loss, component_recon_loss


def calculate_raw_anomaly_maps_moe(outputs, image_size, masks, args):
    routing_weights = outputs["routing_weights"]
    expert_outputs = outputs["expert_outputs"]
    batch_size = routing_weights.shape[0]
    device = routing_weights.device

    h_feat, w_feat = expert_outputs["patch"]["recon_maps"][0]["en_maps"][0].shape[-2:]

    patch_level_map = torch.zeros(batch_size, 1, h_feat, w_feat, device=device)
    global_level_map = torch.zeros(batch_size, 1, h_feat, w_feat, device=device)
    component_level_map = torch.zeros(
        batch_size, 1, image_size, image_size, device=device
    )

    num_p, num_g, num_c = (
        args.num_patch_experts,
        args.num_global_experts,
        args.num_component_experts,
    )
    patch_weights, global_weights, component_weights = torch.split(
        routing_weights, [num_p, num_g, num_c], dim=1
    )

    for i in range(num_p):
        en_map = expert_outputs["patch"]["recon_maps"][i]["en_maps"][-1]
        de_map = expert_outputs["patch"]["recon_maps"][i]["de_maps"][-1]
        error = (1 - F.cosine_similarity(en_map, de_map, dim=1)).unsqueeze(1)
        patch_level_map += patch_weights[:, i].view(-1, 1, 1, 1) * error

    for i in range(num_g):
        expert_data = expert_outputs["global"]["recon_outputs"][i]
        error = (expert_data["ae_output"] - expert_data["student_output"]) ** 2
        error_resized = F.interpolate(
            error.mean(dim=1, keepdim=True),
            size=(h_feat, w_feat),
            mode="bilinear",
            align_corners=False,
        )
        global_level_map += global_weights[:, i].view(-1, 1, 1, 1) * error_resized

    orig_comp_feats = expert_outputs["component_orig"]
    recon_comp_feats_list = expert_outputs["component"]["recon_feats"]
    num_components_per_image = expert_outputs["num_components"]

    if orig_comp_feats is not None and orig_comp_feats.numel() > 0:
        comp_scores = torch.zeros(orig_comp_feats.shape[0], device=device)
        comp_to_img_weights = []
        for i in range(batch_size):
            if num_components_per_image[i] > 0:
                comp_to_img_weights.append(
                    component_weights[i, :].repeat(num_components_per_image[i], 1)
                )

        if comp_to_img_weights:
            all_comp_weights = torch.cat(comp_to_img_weights, dim=0)

            for i in range(num_c):
                loss_per_comp = 1 - F.cosine_similarity(
                    recon_comp_feats_list[i], orig_comp_feats, dim=1
                )
                comp_scores += all_comp_weights[:, i] * loss_per_comp

            start_idx = 0
            for i in range(batch_size):
                num_comps = num_components_per_image[i]
                if num_comps > 0:
                    current_masks = masks[i].to(device)
                    current_scores = comp_scores[start_idx : start_idx + num_comps]
                    img_comp_map_per_comp = (
                        current_scores.view(-1, 1, 1) * current_masks.float()
                    )
                    component_level_map[i] = torch.max(
                        img_comp_map_per_comp, dim=0, keepdim=True
                    )[0]
                    start_idx += num_comps

    patch_level_map_resized = F.interpolate(
        patch_level_map, size=image_size, mode="bilinear", align_corners=False
    )
    global_level_map_resized = F.interpolate(
        global_level_map, size=image_size, mode="bilinear", align_corners=False
    )

    return patch_level_map_resized, global_level_map_resized, component_level_map


def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_ddp(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    timeout_delta = timedelta(minutes=100)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=timeout_delta
    )
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


transform_ae = transforms.RandomChoice(
    [
        transforms.ColorJitter(brightness=0.2),
        transforms.ColorJitter(contrast=0.2),
        transforms.ColorJitter(saturation=0.2),
    ]
)

data_transform, gt_transform = get_data_transforms(448, 392)


def ae_transform(image):
    return data_transform(transform_ae(image))


def train_transform(image):
    return data_transform(image), data_transform(transform_ae(image))


def custom_collate_train(batch):
    images_tuple, labels, component_masks, class_names = zip(*batch)
    imgs_st, imgs_ae = zip(*images_tuple)
    imgs_st = torch.stack(imgs_st, 0)
    imgs_ae = torch.stack(imgs_ae, 0)
    labels = torch.tensor(labels)
    return (imgs_st, imgs_ae), labels, component_masks, class_names


def custom_collate_test(batch):
    items = list(zip(*batch))
    imgs, gts, labels = (
        torch.stack(items[0], 0),
        torch.stack(items[1], 0),
        torch.tensor(items[2]),
    )
    img_paths, class_names, imgs_ae = items[3], items[4], torch.stack(items[5], 0)
    component_masks = items[6]
    return imgs, gts, labels, img_paths, class_names, imgs_ae, component_masks


def main_worker(rank, world_size, args):
    setup_ddp(rank, world_size, args.port)
    is_main_process = rank == 0

    if is_main_process:
        print(f"Experiment starting with config: {vars(args)}")

    print(f"Process {rank}/{world_size} started.")
    setup_seed(args.seed + rank)

    total_iters = args.total_iters

    item_list_mvtec = [
        "carpet",
        "grid",
        "leather",
        "tile",
        "wood",
        "bottle",
        "cable",
        "capsule",
        "hazelnut",
        "metal_nut",
        "pill",
        "screw",
        "toothbrush",
        "transistor",
        "zipper",
    ]
    item_list_visa = [
        "candle",
        "capsules",
        "chewinggum",
        "cashew",
        "fryum",
        "pipe_fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
    ]
    item_list_mvtec_loco = [
        "breakfast_box",
        "juice_bottle",
        "pushpins",
        "screw_bag",
        "splicing_connectors",
    ]
    item_list_mvtec_3d = [
        "bagel",
        "cable_gland",
        "carrot",
        "cookie",
        "dowel",
        "foam",
        "peach",
        "potato",
        "rope",
        "tire",
    ]
    item_list_mvtec_Ped2 = ["pedestrian"]
    item_list_bmad = ["BrainMRI", "LiverCT", "RESC"]

    dataset_groups = [
        ("MVTec_AD", item_list_mvtec),
        ("MVTec_LOCO", item_list_mvtec_loco),
        ("VisA", item_list_visa),
        ("MVTec_3D", item_list_mvtec_3d),
        ("BMAD", item_list_bmad),
        ("Ped2", item_list_mvtec_Ped2),
    ]

    class_to_dataset_map = {}
    for group_name, class_list in dataset_groups:
        for class_name in class_list:
            class_to_dataset_map[class_name] = group_name

    train_data_list = []
    all_items = (
        item_list_mvtec
        + item_list_mvtec_loco
        + item_list_visa
        + item_list_mvtec_3d
        + item_list_mvtec_Ped2
        + item_list_bmad
    )

    for item in all_items:
        if item in item_list_mvtec:
            data_root = "./data/mvtec_ad"
        elif item in item_list_mvtec_loco:
            data_root = "./data/mvtec_loco"
        elif item in item_list_visa:
            data_root = "./data/VisA"
        elif item in item_list_mvtec_Ped2:
            data_root = "./data/Ped2"
        elif item in item_list_bmad:
            data_root = "./data/BMAD"
        elif item in item_list_mvtec_3d:
            data_root = "./data/mvtec_ad_3d"
        else:
            print(f"Warning: Data root for item '{item}' not defined. Skipping.")
            continue

        item_root_path = os.path.join(data_root, item)

        train_data_item = TrainMoEDataset(
            root=item_root_path,
            transform=train_transform,
            class_name=item,
            mask_root=args.mask_root + "/" + data_root.split("/")[-1],
        )
        train_data_list.append(train_data_item)

    train_data = ConcatDataset(train_data_list)
    train_sampler = DistributedSampler(
        train_data, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_dataloader = InfiniteDataloader(
        DataLoaderX(
            train_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=True,
            sampler=train_sampler,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=custom_collate_train,
        )
    )

    if is_main_process:
        print("Total train image number:{}".format(len(train_data)))
        print(f"Effective batch size: {args.batch_size * world_size}")

    encoder_name, embed_dim, num_heads = "dinov2reg_vit_base_14", 768, 12
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    if is_main_process:
        print("Initializing AnomalyMoE model...")

    encoder = vit_encoder.load(encoder_name)
    template_bottleneck = nn.ModuleList(
        [bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)]
    )
    template_decoder = nn.ModuleList(
        [
            VitBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
                attn=LinearAttention2,
            )
            for _ in range(8)
        ]
    )
    model = AnomalyMoE(
        encoder=encoder,
        bottleneck=template_bottleneck,
        decoder=template_decoder,
        num_patch_experts=args.num_patch_experts,
        num_global_experts=args.num_global_experts,
        num_component_experts=args.num_component_experts,
        topk=args.top_k,
        target_layers=target_layers,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
    )

    model = model.to(rank)
    for param in model.encoder.parameters():
        param.requires_grad = False
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    club_estimator = MIEstimator(dim=embed_dim).to(rank)
    club_estimator = DDP(club_estimator, device_ids=[rank])

    trainable_modules = [
        model.module.patch_experts,
        model.module.global_experts,
        model.module.component_experts,
        model.module.gating_network,
    ]

    for module in trainable_modules:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    optimizer_params = [{"params": module.parameters()} for module in trainable_modules]
    optimizer = StableAdamW(
        optimizer_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        amsgrad=True,
        eps=1e-10,
    )
    lr_scheduler = WarmCosineScheduler(
        optimizer,
        base_value=args.lr,
        final_value=args.lr * 0.01,
        total_iters=total_iters,
        warmup_iters=100,
    )
    optimizer_mi = torch.optim.Adam(
        params=club_estimator.parameters(), lr=args.lr, weight_decay=1e-4
    )

    model.train()
    model.module.encoder.eval()

    loss_patch_list, loss_global_list, loss_comp_list = [], [], []
    loss_esb_list, loss_eir_list, loss_estimator_list = [], [], []

    tqdm_obj = tqdm(range(args.total_iters), disable=(not is_main_process))

    for it in range(total_iters):
        model.train()
        model.module.encoder.eval()
        club_estimator.eval()
        optimizer.zero_grad()

        images, _, component_masks, class_names = next(iter(train_dataloader))
        image_st, image_ae = images
        image_st = image_st.to(rank, non_blocking=True)
        image_ae = image_ae.to(rank, non_blocking=True)

        outputs = model(image_st=image_st, image_ae=image_ae, masks=component_masks)

        dataset_names = [class_to_dataset_map[c] for c in class_names]

        total_recon_loss, patch_loss, global_loss, comp_loss = calculate_moe_loss(
            outputs, args, dataset_names
        )

        esb_loss = calculate_esb_loss(
            outputs["raw_logits"],
            outputs["expert_mask"],
            model.module.total_experts,
            dataset_names,
            args,
        )

        eir_loss = club_estimator(outputs["expert_outputs"])

        loss = (
            total_recon_loss + args.lambda_eir * eir_loss + args.lambda_esb * esb_loss
        )
        loss.backward()

        all_params = itertools.chain(
            *(module.parameters() for module in trainable_modules)
        )
        nn.utils.clip_grad_norm_(all_params, max_norm=0.1)
        optimizer.step()

        club_estimator.train()
        optimizer_mi.zero_grad()
        with torch.no_grad():
            outputs = model(image_st=image_st, image_ae=image_ae, masks=component_masks)
        estimator_loss = club_estimator.module.train_estimator(
            outputs["expert_outputs"]
        )
        estimator_loss.backward()
        club_params = club_estimator.module.parameters()
        nn.utils.clip_grad_norm_(club_params, max_norm=0.1)
        optimizer_mi.step()

        loss_patch_list.append(patch_loss.item())
        loss_global_list.append(global_loss.item())
        loss_comp_list.append(comp_loss.item())
        loss_esb_list.append(esb_loss.item())
        loss_eir_list.append(eir_loss.item())
        loss_estimator_list.append(estimator_loss.item())
        lr_scheduler.step()

        if is_main_process:
            tqdm_obj.update(1)

            if (it + 1) % 100 == 0:
                avg_loss_patch = np.mean(loss_patch_list)
                avg_loss_global = np.mean(loss_global_list)
                avg_loss_comp = np.mean(loss_comp_list)
                avg_loss_esb = np.mean(loss_esb_list)
                avg_loss_eir = np.mean(loss_eir_list)
                avg_recon_loss = avg_loss_patch + avg_loss_global + avg_loss_comp
                total_avg_loss = (
                    avg_recon_loss
                    + args.lambda_esb * avg_loss_esb
                    + args.lambda_eir * avg_loss_eir
                )
                current_lr = optimizer.param_groups[0]["lr"]
                desc = (
                    f"Iter [{it+1}/{args.total_iters}], Loss:{total_avg_loss:.4f} "
                    f"(Recon:{avg_recon_loss:.3f}, ESB:{avg_loss_esb:.3f}, EIR:{avg_loss_eir:.3f}), LR:{current_lr:.6f}"
                )
                tqdm_obj.set_description(desc)
                loss_patch_list, loss_global_list, loss_comp_list = [], [], []
                loss_esb_list, loss_eir_list, loss_estimator_list = [], [], []

    if is_main_process:
        tqdm_obj.close()
        print("Training finished.")
        save_path = os.path.join(args.save_dir, args.save_name, "model_final.pth")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.module.state_dict(), save_path)
        print(f"Final model saved to {save_path}")

    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description="AnomalyMoE Detection Training")
    parser.add_argument(
        "--mask_root",
        type=str,
        default="./masks",
        help="Root directory for component masks",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./saved_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default=f"AnomalyMoE_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Specific name for this experiment",
    )

    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument(
        "--total_iters", type=int, default=100000, help="Total training iterations"
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of workers for DataLoader"
    )
    parser.add_argument(
        "--port", type=int, default=12360, help="Port for DDP communication."
    )

    parser.add_argument(
        "--num_patch_experts",
        type=int,
        default=6,
        help="Number of patch-level experts.",
    )
    parser.add_argument(
        "--num_global_experts",
        type=int,
        default=6,
        help="Number of global-level experts.",
    )
    parser.add_argument(
        "--num_component_experts",
        type=int,
        default=6,
        help="Number of component-level experts.",
    )
    parser.add_argument(
        "--p_hm",
        type=float,
        default=0.9,
        help="Hard-mining percentile for patch expert loss (top 1-p).",
    )
    parser.add_argument(
        "--factor_hm",
        type=float,
        default=0.1,
        help="Gradient modification factor for hard-mining.",
    )
    parser.add_argument(
        "--beta_comp",
        type=float,
        default=0.2,
        help="Weight for fusing the component anomaly map during evaluation.",
    )
    parser.add_argument(
        "--lambda_gating",
        type=float,
        default=0.2,
        help="Weight for the auxiliary gating loss that encourages expert specialization.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of experts to activate for each sample.",
    )
    parser.add_argument(
        "--lambda_esb",
        type=float,
        default=0.01,
        help="Weight for the Expert Selection Balancing (ESB) loss.",
    )
    parser.add_argument(
        "--lambda_eir",
        type=float,
        default=0.0001,
        help="Weight for the Expert Information Repulsion (EIR) loss.",
    )
    parser.add_argument(
        "--capacity_factor",
        type=float,
        default=4.0,
        help="Capacity factor for the ESB load loss.",
    )

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No GPUs found. DDP requires at least one GPU.")
        return

    print(f"Found {world_size} GPUs. Starting DDP training.")
    spawn_args = (world_size, args)
    mp.spawn(main_worker, args=spawn_args, nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
