import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import cv2
import PIL
import os
import glob
import argparse

from sklearn.cluster import MiniBatchKMeans
from typing import Union
import abc

from modules import DinoFeaturizer
from utils import unnorm


import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import torchvision.transforms.functional as VF


K_VALUES_PER_CLASS = {}


DEFAULT_K_VALUE = 5


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
i_m = np.array(IMAGENET_MEAN)[:, None, None]
i_std = np.array(IMAGENET_STD)[:, None, None]


MAX_ITER = 2


def dense_crf(image_tensor: torch.FloatTensor, output_logits: torch.FloatTensor):
    image = np.array(VF.to_pil_image(unnorm(image_tensor)))[:, :, ::-1]
    H, W = image.shape[:2]
    image = np.ascontiguousarray(image)

    output_logits = F.interpolate(
        output_logits.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
    ).squeeze()
    output_probs = F.softmax(output_logits, dim=0).cpu().numpy()

    c, h, w = output_probs.shape

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=1, compat=3)
    d.addPairwiseBilateral(sxy=67, srgb=3, rgbim=image, compat=4)
    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q


class BaseSampler(abc.ABC):
    def __init__(self, percentage: float):
        if not 0 < percentage <= 1:
            raise ValueError("Percentage value must be in (0, 1].")
        self.percentage = percentage

    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.to(self.features_device)


class GreedyCoresetSampler(BaseSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        dimension_to_project_features_to=128,
    ):
        super().__init__(percentage)
        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _reduce_features(self, features):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        ).to(self.device)
        features = features.to(self.device)
        return mapper(features)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        if self.percentage == 1:
            return features
        self._store_type(features)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).to(self.device)
        else:
            features = features.to(self.device)

        reduced_features = self._reduce_features(features)
        sample_indices = self._compute_greedy_coreset_indices(reduced_features)
        features = features[sample_indices]
        return self._restore_type(features)

    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)
        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        distance_matrix = self._compute_batchwise_differences(features, features)
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1)
        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)
            coreset_select_distance = distance_matrix[:, select_idx : select_idx + 1]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values
        return np.array(coreset_indices)


class BaseMVTecDataset(Dataset):
    def __init__(self, resize_shape=None):
        self.resize_shape = resize_shape
        self.image_paths = []

        self.transform_img = transforms.Compose(
            [
                transforms.Resize((self.resize_shape, self.resize_shape)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement the __getitem__ method.")


class MVTecLocoDataset(BaseMVTecDataset):
    def __init__(self, root_dir, category, resize_shape=None):
        super().__init__(resize_shape)
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, category, "*")))

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_pil = PIL.Image.open(image_path).convert("RGB")
        image_tensor_transformed = self.transform_img(image_pil)
        image_tensor_original = transforms.ToTensor()(image_pil)
        return {"image": image_tensor_transformed, "image1": image_tensor_original}


class MVTecADFormatDataset(BaseMVTecDataset):
    def __init__(self, root_dir, classname, split="train", resize_shape=None):
        super().__init__(resize_shape)
        image_dir = os.path.join(root_dir, classname, split)
        subfolders = [f.path for f in os.scandir(image_dir) if f.is_dir() and f.name != 'good']


        if "VisA" in root_dir:
            img_ext = "*.JPG"
        else:
            img_ext = "*.png"
            

        self.image_paths.extend(sorted(glob.glob(os.path.join(image_dir, img_ext))))

        good_dir = os.path.join(image_dir, 'good')
        if os.path.isdir(good_dir):
            self.image_paths.extend(sorted(glob.glob(os.path.join(good_dir, img_ext))))

        for subfolder in subfolders:
            self.image_paths.extend(sorted(glob.glob(os.path.join(subfolder, img_ext))))

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_pil = PIL.Image.open(image_path).convert("RGB")
        image_tensor_transformed = self.transform_img(image_pil)
        image_tensor_original = transforms.ToTensor()(image_pil)
        return {"image": image_tensor_transformed, "image1": image_tensor_original}


def run(dataset_path: str, dataset_name: str, classname: str, image_size: int, coreset_sampling_ratio: float) -> None:
    
    num_cluster = K_VALUES_PER_CLASS.get(classname, DEFAULT_K_VALUE)
    print(f"Using k={num_cluster} for class '{classname}'.")

    color_list = [
        [127, 123, 229], [195, 240, 251], [146, 223, 255],
        [243, 241, 230], [224, 190, 144], [178, 116, 75],
    ]
    if num_cluster > len(color_list):
        np.random.seed(0)
        new_colors = np.random.randint(0, 255, size=(num_cluster - len(color_list), 3)).tolist()
        color_list.extend(new_colors)

    global color_tensor
    color_tensor = torch.tensor(color_list[:num_cluster])[:, :, None, None]
    color_tensor = color_tensor.repeat(1, 1, image_size, image_size)

    dataloaders = {}

    num_workers = 4 

    if dataset_name == "mvtec_loco":
        dataset_root = os.path.join(dataset_path, classname)
        categories = ["train/good", "test/good", "test/logical_anomalies", "test/structural_anomalies"]
        for category in categories:
            dataset = MVTecLocoDataset(root_dir=dataset_root, category=category, resize_shape=image_size)
            if len(dataset) > 0:
                dataloaders[category] = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        dataset_train = MVTecLocoDataset(root_dir=dataset_root, category="train/good", resize_shape=image_size)
        dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=num_workers)

    elif dataset_name in ["mvtec_ad", "VisA", "Ped2", "BMAD", "mvtec_ad_3d"]:
        dataset_root = dataset_path
        dataset_train = MVTecADFormatDataset(root_dir=dataset_root, classname=classname, split="train", resize_shape=image_size)
        dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=num_workers)
        dataloaders["train/good"] = dataloader_train

        test_dir = os.path.join(dataset_root, classname, "test")
        if os.path.exists(test_dir):
            test_types = [f.name for f in os.scandir(test_dir) if f.is_dir()]
            for test_type in test_types:

                image_paths_test = sorted(glob.glob(os.path.join(test_dir, test_type, "*.png")))
                if not image_paths_test: continue
                
                dataset_test = MVTecADFormatDataset(root_dir=dataset_root, classname=classname, split="test", resize_shape=image_size)
                dataset_test.image_paths = image_paths_test
                
                if len(dataset_test) > 0:
                    dataloaders[f"test/{test_type}"] = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=num_workers)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    net = DinoFeaturizer().cuda()
    greedsampler_perimg = GreedyCoresetSampler(percentage=coreset_sampling_ratio, device="cuda")

    centers_path = f"./masks/{classname}_{dataset_name}_k{num_cluster}_{image_size}.pth"
    os.makedirs("./masks", exist_ok=True)

    if StartTrain:
        print("Starting training: Extracting features and clustering with MiniBatchKMeans...")

        kmeans = MiniBatchKMeans(n_clusters=num_cluster, random_state=0, batch_size=2048, n_init=10)

        for i, data in enumerate(dataloader_train):
            image = data["image"].cuda()
            with torch.no_grad():
                feats0, _ = net(image)
                feats = feats0.squeeze().reshape(feats0.shape[1], -1).permute(1, 0)
                feats_sample = greedsampler_perimg.run(feats)

                kmeans.partial_fit(feats_sample.cpu().numpy())
            

            print(f"Processing for clustering: {i+1}/{len(dataloader_train)}", end='\r')
        
        print("\nClustering finished.")

        cluster_centers = torch.from_numpy(kmeans.cluster_centers_)
        torch.save(cluster_centers, centers_path)
        print(f"Cluster centers saved to {centers_path}")
    else:
        print(f"Skipping training. Loading pre-computed cluster centers from {centers_path}...")
        if not os.path.exists(centers_path):
            raise FileNotFoundError(f"Cannot skip training, file not found: {centers_path}")
        cluster_centers = torch.load(centers_path)

    train_features_sampled = cluster_centers.cuda().unsqueeze(0).unsqueeze(0).permute(0, 3, 1, 2)
    base_savepath = f"./masks/{dataset_name}/{classname}_heat_{image_size}"

    for subpath, loader in dataloaders.items():
        if len(loader) == 0:
            print(f"Skipping empty category: {subpath}")
            continue
        print(f"Processing inference for: {subpath}...")
        savepath = os.path.join(base_savepath, subpath)
        os.makedirs(savepath, exist_ok=True)
        save_img(loader, train_features_sampled, net, savepath)
    print("All images processed.")


def save_img(dataloader, train_features_sampled, net, savepath):
    for i, data in enumerate(dataloader):
        image = data["image"]
        imageo = data["image1"][0]
        imageo = unloader(imageo)

        heatmap, heatmap_intra = get_heatmaps(image, train_features_sampled, net)

        img_savepath = os.path.join(savepath, str(i))
        os.makedirs(img_savepath, exist_ok=True)

        imageo.save(os.path.join(img_savepath, "imgo.jpg"))
        see_image(image, heatmap, img_savepath, heatmap_intra)


def get_heatmaps(img, query_feature, net):
    with torch.no_grad():
        feats1, _ = net(img.cuda())

    sfeats1 = query_feature
    attn_intra = torch.einsum(
        "nchw,ncij->nhwij", F.normalize(sfeats1, dim=1), F.normalize(feats1, dim=1)
    )
    attn_intra -= attn_intra.mean([3, 4], keepdims=True)
    attn_intra = attn_intra.clamp(0).squeeze(0)

    heatmap_intra = (
        F.interpolate(attn_intra, img.shape[2:], mode="bilinear", align_corners=True)
        .squeeze(0)
        .detach()
        .cpu()
    )

    img_crf = img.squeeze()
    crf_result = dense_crf(img_crf, heatmap_intra)
    heatmap_intra_crf = torch.from_numpy(crf_result)

    d = heatmap_intra_crf.argmax(dim=0).unsqueeze(0).unsqueeze(0)
    d = d.repeat(1, 3, 1, 1)

    seg_map = torch.zeros([1, 3, d.shape[2], d.shape[3]], dtype=torch.int64)
    for color_idx in range(query_feature.shape[2]):
        seg_map = torch.where(d == color_idx, color_tensor[color_idx], seg_map)

    return seg_map, heatmap_intra_crf


def see_image(data, heatmap, savepath, heatmap_intra):
    data_np = data[0].cpu().numpy()
    data_np = np.clip((data_np * i_std + i_m) * 255, 0, 255).astype(np.uint8)
    data_np = data_np.transpose(1, 2, 0)
    data_np = cv2.cvtColor(data_np, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(savepath, "img.jpg"), data_np)

    heatmap_np = heatmap[0].cpu().numpy().transpose(1, 2, 0)
    cv2.imwrite(os.path.join(savepath, "heatresult.jpg"), heatmap_np)

    for i in range(heatmap_intra.shape[0]):
        heat = heatmap_intra[i].cpu().numpy()
        heat = np.round(heat * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(savepath, f"heatresult{i}.jpg"), heat_color)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly segmentation on various industrial inspection datasets.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the root of the dataset.")
    parser.add_argument("--dataset_name", type=str, required=True,
                        choices=["mvtec_ad", "mvtec_loco", "VisA", "Ped2", "BMAD", "mvtec_ad_3d"],
                        help="Name of the dataset to use.")
    parser.add_argument("--classname", type=str, required=True,
                        help="Class category to process (e.g., 'screw' or 'BrainMRI').")
    parser.add_argument("--train", default=True,
                        help="Set this flag to run the training process; otherwise, loads existing features.")

    parser.add_argument("--image_size", type=int, default=448, 
                        help="Size to which images are resized.")
    parser.add_argument("--coreset_sampling_ratio", type=float, default=0.01,
                        help="Percentage of features to sample for coreset.")

    args = parser.parse_args()

    unloader = transforms.ToPILImage()
    StartTrain = args.train
    color_tensor = None

    run(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        classname=args.classname,
        image_size=args.image_size,
        coreset_sampling_ratio=args.coreset_sampling_ratio,
    )