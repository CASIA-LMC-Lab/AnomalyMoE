import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
from functools import partial

from models.anomalymoe import AnomalyMoE
from models import vit_encoder
from models.vision_transformer import (
    Block as VitBlock,
    bMlp,
    LinearAttention2,
)
from utils import get_gaussian_kernel

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


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


import glob
from PIL import Image
from pathlib import Path
import os
import glob
import numpy as np
import torch
import torch.utils.data
from pathlib import Path
from PIL import Image
import cv2


class MoEDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase, ae_transform, mask_root):
        self.phase = phase
        if self.phase == "train":
            self.img_path = os.path.join(root, "train")
            self.gt_path = None
        else:
            self.img_path = os.path.join(root, "test")
            self.gt_path = os.path.join(root, "ground_truth")

        self.transform = transform
        self.gt_transform = gt_transform
        self.ae_transform = ae_transform
        self.mask_root = Path(mask_root)

        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()
        self.class_name = os.path.basename(root)

        self.component_indices_to_load = filter_bg_noise(
            self.mask_root, self.class_name, resolution=448
        )

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        self.image_path_to_id = {}

        defect_types = os.listdir(self.img_path)
        for defect_type in defect_types:
            img_paths_current = (
                glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                + glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                + glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
            )
            img_paths_current.sort()

            for i, path in enumerate(img_paths_current):
                self.image_path_to_id[path] = str(i)

            if defect_type == "good":
                img_tot_paths.extend(img_paths_current)
                gt_tot_paths.extend([0] * len(img_paths_current))
                tot_labels.extend([0] * len(img_paths_current))
                tot_types.extend(["good"] * len(img_paths_current))
            else:
                if self.phase == "test":
                    gt_paths = glob.glob(
                        os.path.join(self.gt_path, defect_type) + "/*.png"
                    )
                    gt_paths.sort()
                    gt_tot_paths.extend(gt_paths)
                else:
                    gt_tot_paths.extend([0] * len(img_paths_current))
                img_tot_paths.extend(img_paths_current)
                tot_labels.extend([1] * len(img_paths_current))
                tot_types.extend([defect_type] * len(img_paths_current))
        if self.phase == "test":
            assert len(img_tot_paths) == len(
                gt_tot_paths
            ), "Something wrong with test and ground truth pair!"
        return (
            np.array(img_tot_paths),
            np.array(gt_tot_paths),
            np.array(tot_labels),
            np.array(tot_types),
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path_str, gt_path, label, _ = (
            self.img_paths[idx],
            self.gt_paths[idx],
            self.labels[idx],
            self.types[idx],
        )
        img_path = Path(img_path_str)

        img = Image.open(img_path).convert("RGB")
        image_ae = self.ae_transform(img)
        img_transformed = self.transform(img)

        if label == 0:
            gt = torch.zeros(
                [1, img_transformed.size()[-2], img_transformed.size()[-1]]
            )
        else:
            gt = Image.open(gt_path)
            gt = self.gt_transform(gt)

        h, w = img_transformed.shape[-2:]

        image_id = self.image_path_to_id[img_path_str]

        if self.phase == "train":
            mask_base_dir = (
                self.mask_root
                / f"{self.class_name}_heat_448"
                / "train"
                / "good"
                / image_id
            )
        else:
            defect_type_from_path = img_path.parent.name
            mask_base_dir = (
                self.mask_root
                / f"{self.class_name}_heat_448"
                / "test"
                / defect_type_from_path
                / image_id
            )

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

        if component_masks.shape[0] > 0:
            assert img_transformed.size()[1:] == component_masks.size()[1:]

        return (
            img_transformed,
            gt,
            label,
            img_path_str,
            self.class_name,
            image_ae,
            component_masks,
        )


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


def evaluate(model, test_dataloader, device, args):
    model.eval()

    all_gts_px, all_gts_sp = [], []
    all_am_patch, all_am_global, all_am_comp = [], [], []
    all_routing_weights = []

    for img, gt, label, _, _, img_ae, component_masks in test_dataloader:
        img = img.to(device)
        img_ae = img_ae.to(device)

        with torch.no_grad():
            outputs = model(image_st=img, image_ae=img_ae, masks=component_masks)
            am_patch_raw, am_global_raw, am_comp_raw = calculate_raw_anomaly_maps_moe(
                outputs, img.shape[-1], component_masks, args
            )

        all_gts_px.append(gt.cpu())
        all_gts_sp.append(label.cpu())
        all_am_patch.append(am_patch_raw.cpu())
        all_am_global.append(am_global_raw.cpu())
        all_am_comp.append(am_comp_raw.cpu())
        all_routing_weights.append(outputs["routing_weights"].cpu())

    all_gts_px = torch.cat(all_gts_px, dim=0)
    all_gts_sp = torch.cat(all_gts_sp, dim=0)
    all_am_patch = torch.cat(all_am_patch, dim=0)
    all_am_global = torch.cat(all_am_global, dim=0)
    all_am_comp = torch.cat(all_am_comp, dim=0)
    all_routing_weights = torch.cat(all_routing_weights, dim=0)

    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to("cpu")
    eps = 1e-11

    min_p, max_p = all_am_patch.min(), all_am_patch.max()
    min_g, max_g = all_am_global.min(), all_am_global.max()
    min_c, max_c = all_am_comp.min(), all_am_comp.max()

    all_am_patch_norm = (all_am_patch - min_p) / (max_p - min_p + eps)
    all_am_global_norm = (all_am_global - min_g) / (max_g - min_g + eps)
    all_am_comp_norm = (all_am_comp - min_c) / (max_c - min_c + eps)

    num_p, num_g, num_c = (
        args.num_patch_experts,
        args.num_global_experts,
        args.num_component_experts,
    )
    patch_w, global_w, component_w = torch.split(
        all_routing_weights, [num_p, num_g, num_c], dim=1
    )

    avg_patch_weight = patch_w.max(dim=1, keepdim=True)[0]
    avg_global_weight = global_w.max(dim=1, keepdim=True)[0]
    avg_comp_weight = component_w.max(dim=1, keepdim=True)[0]

    final_am = (
        avg_patch_weight.view(-1, 1, 1, 1) * all_am_patch_norm
        + avg_global_weight.view(-1, 1, 1, 1) * all_am_global_norm
        + avg_comp_weight.view(-1, 1, 1, 1) * all_am_comp_norm
    )

    resize_mask = 256
    am_resized = F.interpolate(
        final_am, size=resize_mask, mode="bilinear", align_corners=False
    )
    gt_resized = F.interpolate(all_gts_px, size=resize_mask, mode="nearest")

    am_smoothed = gaussian_kernel(am_resized)

    gt_px_flat = gt_resized.bool().view(-1).numpy()
    pr_px_flat = am_smoothed.view(-1).numpy()

    am_flat_per_image = am_smoothed.view(am_smoothed.shape[0], -1)
    k = max(1, int(am_flat_per_image.shape[1] * 0.01))
    pr_sp_scores = torch.topk(am_flat_per_image, k, dim=1).values.mean(dim=1).numpy()
    gt_sp_scores = all_gts_sp.numpy()

    auroc_px = roc_auc_score(gt_px_flat, pr_px_flat)
    auroc_sp = roc_auc_score(gt_sp_scores, pr_sp_scores)

    return {"Image-AUROC": auroc_sp, "Pixel-AUROC": auroc_px}


def main():
    parser = argparse.ArgumentParser(
        description="AnomalyMoE Testing Script for Full Datasets"
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained model .pth file.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of the dataset (e.g., /path/to/mvtec_ad).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["MVTec_AD", "VisA", "MVTec_LOCO", "MVTec_3D", "BMAD", "Ped2"],
        help="Name of the dataset to evaluate.",
    )
    parser.add_argument(
        "--mask_root",
        type=str,
        default="./final_masks",
        help="Root directory for component masks",
    )

    parser.add_argument("--num_patch_experts", type=int, default=6)
    parser.add_argument("--num_global_experts", type=int, default=6)
    parser.add_argument("--num_component_experts", type=int, default=6)
    parser.add_argument("--top_k", type=int, default=3)

    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--crop_size", type=int, default=392)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

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
    item_list_bmad = ["BrainMRI", "LiverCT", "RESC"]
    item_list_ped2 = ["pedestrian"]

    dataset_map = {
        "MVTec_AD": item_list_mvtec,
        "VisA": item_list_visa,
        "MVTec_LOCO": item_list_mvtec_loco,
        "MVTec_3D": item_list_mvtec_3d,
        "BMAD": item_list_bmad,
        "Ped2": item_list_ped2,
    }

    if args.dataset_name not in dataset_map:
        raise ValueError(
            f"Invalid dataset name: {args.dataset_name}. Please choose from {list(dataset_map.keys())}"
        )

    classes_to_test = dataset_map[args.dataset_name]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing AnomalyMoE model...")
    encoder_name, embed_dim, num_heads = "dinov2reg_vit_base_14", 768, 12
    target_layers, fuse_layer_encoder, fuse_layer_decoder = (
        [2, 3, 4, 5, 6, 7, 8, 9],
        [[0, 1, 2, 3, 4, 5, 6, 7]],
        [[0, 1, 2, 3, 4, 5, 6, 7]],
    )

    encoder = vit_encoder.load(encoder_name)
    template_bottleneck = torch.nn.ModuleList(
        [bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)]
    )
    template_decoder = torch.nn.ModuleList(
        [
            VitBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-8),
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

    print(f"Loading weights from checkpoint: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {args.checkpoint_path}")

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    all_class_metrics = {}

    pbar = tqdm(classes_to_test, desc=f"Evaluating Dataset: {args.dataset_name}")
    for class_name in pbar:
        pbar.set_postfix_str(f"Current Class: {class_name}")

        data_transform, gt_transform = get_data_transforms(
            args.image_size, args.crop_size
        )
        transform_ae = transforms.RandomChoice(
            [
                transforms.ColorJitter(brightness=0.2),
                transforms.ColorJitter(contrast=0.2),
                transforms.ColorJitter(saturation=0.2),
            ]
        )
        ae_transform = lambda image: data_transform(transform_ae(image))

        test_dataset = MoEDataset(
            root=os.path.join(args.data_root, class_name),
            transform=data_transform,
            gt_transform=gt_transform,
            phase="test",
            ae_transform=ae_transform,
            mask_root="./masks/" + args.data_root.split("/")[-1],
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=custom_collate_test,
            pin_memory=True,
        )

        metrics = evaluate(model, test_dataloader, device, args)
        all_class_metrics[class_name] = metrics

    print("\n" + "=" * 60)
    print(f"Final Summary for Dataset: {args.dataset_name}")
    print("=" * 60)
    print(f"{'Class':<20} | {'Image-AUROC':<15} | {'Pixel-AUROC':<15}")
    print("-" * 60)

    i_auroc_list, p_auroc_list = [], []

    for class_name, metrics in all_class_metrics.items():
        i_auroc = metrics["Image-AUROC"]
        p_auroc = metrics["Pixel-AUROC"]
        i_auroc_list.append(i_auroc)
        p_auroc_list.append(p_auroc)
        print(f"{class_name:<20} | {i_auroc:<15.4f} | {p_auroc:<15.4f}")

    print("-" * 60)

    if i_auroc_list:
        mean_i_auroc = np.mean(i_auroc_list)
        mean_p_auroc = np.mean(p_auroc_list)
        print(f"{'Mean':<20} | {mean_i_auroc:<15.4f} | {mean_p_auroc:<15.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
