import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional

# CIFAR-100 normalization constants
CIFAR100_TRAIN_MEAN = (0.5070751592371766, 0.4865488733149507, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

def get_cifar100_transforms():
    """Returns train and test transforms for CIFAR-100."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
    ])
    return train_transform, test_transform

class RemappedCIFARSubset(Dataset):
    """A subset wrapper that maps original CIFAR-100 class IDs to a contiguous 0..K-1 range."""
    def __init__(self, base_dataset: Dataset, indices: List[int], label_map: Dict[int, int]):
        self.base_dataset = base_dataset
        self.indices = indices
        self.label_map = label_map
        
    def __len__(self) -> int:
        return len(self.indices)
        
    def __getitem__(self, idx: int):
        img, original_label = self.base_dataset[self.indices[idx]]
        remapped_label = self.label_map[original_label]
        return img, remapped_label

class UnnormalizedView:
    """Helper to unnormalize tensors back to RGB [0, 1] for visualization and Grad-CAM."""
    def __init__(self, mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD):
        self.mean = np.array(mean)
        self.std = np.array(std)
        
    def __call__(self, tensor: torch.Tensor) -> np.ndarray:
        # tensor shape: (C, H, W) or (B, C, H, W)
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
        img = self.std * img + self.mean
        img = np.clip(img, 0.0, 1.0)
        return img

def get_class_name_map(root: str = "dataset") -> List[str]:
    """Retrieve full 100 class names."""
    raw_ds = torchvision.datasets.CIFAR100(root=root, train=False, download=False)
    return raw_ds.classes

class SingleClassFLDataManager:
    """
    Partitions CIFAR-100 into a 1-class-per-client federated layout.
    
    If num_classes=10, Client 0 has only Class 0 samples, Client 1 has Class 1 samples, etc.
    Provides data loaders for:
      - Each individual client
      - Retain dataset (union of all clients except the forget client)
      - Forget dataset (the single forgotten client's samples)
      - Test sets (retain test, forget test, and overall test)
    """
    def __init__(
        self,
        root: str = "dataset",
        selected_classes: Optional[List[int]] = None,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Default to 10 prominent classes if not specified
        if selected_classes is None:
            # Classes: [apple, aquarium_fish, baby, bear, beaver, bed, bee, beetle, bicycle, bottle]
            self.selected_classes = list(range(10))
        else:
            self.selected_classes = selected_classes
            
        self.num_classes = len(self.selected_classes)
        self.label_map = {orig: new for new, orig in enumerate(self.selected_classes)}
        self.inv_label_map = {new: orig for new, orig in enumerate(self.selected_classes)}
        
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
        ])
        
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
        ])
        
        # Load datasets
        self.raw_train = torchvision.datasets.CIFAR100(
            root=root, train=True, transform=train_transform, download=False
        )
        self.raw_test = torchvision.datasets.CIFAR100(
            root=root, train=False, transform=eval_transform, download=False
        )
        self.raw_train_eval = torchvision.datasets.CIFAR100(
            root=root, train=True, transform=eval_transform, download=False
        )
        
        self.all_classes = self.raw_train.classes
        self.class_names = [self.all_classes[orig_id] for orig_id in self.selected_classes]
        
        # Extract per-class indices
        self.train_class_indices = self._extract_class_indices(self.raw_train)
        self.test_class_indices = self._extract_class_indices(self.raw_test)
        
        # Build client datasets (1 class per client)
        self.client_train_datasets = {}
        for client_id, orig_class in enumerate(self.selected_classes):
            indices = self.train_class_indices[orig_class]
            self.client_train_datasets[client_id] = RemappedCIFARSubset(
                self.raw_train, indices, self.label_map
            )
            
        self.unnormalizer = UnnormalizedView()

    def _extract_class_indices(self, dataset) -> Dict[int, List[int]]:
        targets = np.array(dataset.targets)
        indices_by_class = {}
        for cls_id in self.selected_classes:
            indices_by_class[cls_id] = np.where(targets == cls_id)[0].tolist()
        return indices_by_class

    def get_client_train_loader(self, client_id: int, shuffle: bool = True) -> DataLoader:
        """Returns the DataLoader for a specific client (which holds 1 single class)."""
        ds = self.client_train_datasets[client_id]
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def get_retain_train_loader(self, forget_client_id: int, shuffle: bool = True) -> DataLoader:
        """Returns DataLoader containing data of all clients EXCEPT the forget client."""
        retain_indices = []
        for client_id, orig_class in enumerate(self.selected_classes):
            if client_id != forget_client_id:
                retain_indices.extend(self.train_class_indices[orig_class])
        ds = RemappedCIFARSubset(self.raw_train, retain_indices, self.label_map)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def get_forget_train_loader(self, forget_client_id: int, shuffle: bool = False) -> DataLoader:
        """Returns DataLoader containing data of the forget client (unlearning target D_forget)."""
        orig_class = self.selected_classes[forget_client_id]
        indices = self.train_class_indices[orig_class]
        ds = RemappedCIFARSubset(self.raw_train_eval, indices, self.label_map)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def get_full_test_loader(self) -> DataLoader:
        """Returns holdout test DataLoader across all selected classes."""
        all_test_indices = []
        for orig_class in self.selected_classes:
            all_test_indices.extend(self.test_class_indices[orig_class])
        ds = RemappedCIFARSubset(self.raw_test, all_test_indices, self.label_map)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def get_retain_test_loader(self, forget_client_id: int) -> DataLoader:
        """Returns holdout test DataLoader for retained classes only."""
        retain_indices = []
        for client_id, orig_class in enumerate(self.selected_classes):
            if client_id != forget_client_id:
                retain_indices.extend(self.test_class_indices[orig_class])
        ds = RemappedCIFARSubset(self.raw_test, retain_indices, self.label_map)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def get_forget_test_loader(self, forget_client_id: int) -> DataLoader:
        """Returns holdout test DataLoader for the forgotten class only."""
        orig_class = self.selected_classes[forget_client_id]
        indices = self.test_class_indices[orig_class]
        ds = RemappedCIFARSubset(self.raw_test, indices, self.label_map)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def get_sample_images_for_class(self, client_id: int, count: int = 5) -> Tuple[torch.Tensor, List[int]]:
        """Retrieve raw normalized tensors and labels for specific class images for Grad-CAM."""
        orig_class = self.selected_classes[client_id]
        indices = self.test_class_indices[orig_class][:count]
        ds = RemappedCIFARSubset(self.raw_test, indices, self.label_map)
        images = []
        labels = []
        for i in range(len(ds)):
            img, label = ds[i]
            images.append(img.unsqueeze(0))
            labels.append(label)
        return torch.cat(images, dim=0), labels


class MultiClassFLDataManager:
    """
    Partitions CIFAR-100 such that each client holds samples from MULTIPLE classes
    under a Non-IID Dirichlet distribution Dir(alpha).
    
    Provides data loaders for:
      - Each individual multi-class client
      - Retain dataset (union of all clients except the forgotten client)
      - Forget dataset (the forgotten client's private multi-class training data D_forget)
      - Full test loader across all classes
      - Client class distributions and sample image extraction for auditing
    """
    def __init__(
        self,
        root: str = "dataset",
        selected_classes: Optional[List[int]] = None,
        num_clients: int = 10,
        alpha: float = 0.5,
        batch_size: int = 64,
        num_workers: int = 0,
        seed: int = 42
    ):
        self.root = root
        self.num_clients = num_clients
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        if selected_classes is None:
            self.selected_classes = list(range(10))
        else:
            self.selected_classes = selected_classes

        self.num_classes = len(self.selected_classes)
        self.label_map = {orig: new for new, orig in enumerate(self.selected_classes)}
        self.inv_label_map = {new: orig for new, orig in enumerate(self.selected_classes)}

        train_transform, eval_transform = get_cifar100_transforms()

        self.raw_train = torchvision.datasets.CIFAR100(
            root=root, train=True, transform=train_transform, download=False
        )
        self.raw_test = torchvision.datasets.CIFAR100(
            root=root, train=False, transform=eval_transform, download=False
        )
        self.raw_train_eval = torchvision.datasets.CIFAR100(
            root=root, train=True, transform=eval_transform, download=False
        )

        self.all_classes = self.raw_train.classes
        self.class_names = [self.all_classes[orig_id] for orig_id in self.selected_classes]

        # Extract per-class indices
        self.train_class_indices = self._extract_class_indices(self.raw_train)
        self.test_class_indices = self._extract_class_indices(self.raw_test)

        # Partition training indices across clients using Dirichlet distribution
        np.random.seed(seed)
        self.client_indices = {i: [] for i in range(num_clients)}
        for orig_class in self.selected_classes:
            c_indices = np.array(self.train_class_indices[orig_class])
            np.random.shuffle(c_indices)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = proportions / proportions.sum()
            splits = (np.cumsum(proportions) * len(c_indices)).astype(int)[:-1]
            client_splits = np.split(c_indices, splits)
            for i in range(num_clients):
                self.client_indices[i].extend(client_splits[i].tolist())

        # Build client train datasets
        self.client_train_datasets = {}
        for cid in range(num_clients):
            self.client_train_datasets[cid] = RemappedCIFARSubset(
                self.raw_train, self.client_indices[cid], self.label_map
            )

        self.unnormalizer = UnnormalizedView()

    def _extract_class_indices(self, dataset) -> Dict[int, List[int]]:
        targets = np.array(dataset.targets)
        indices_by_class = {}
        for cls_id in self.selected_classes:
            indices_by_class[cls_id] = np.where(targets == cls_id)[0].tolist()
        return indices_by_class

    def get_client_class_distribution(self, client_id: int) -> Dict[str, int]:
        """Returns the class frequency count for a given client."""
        raw_targets = np.array(self.raw_train.targets)[self.client_indices[client_id]]
        dist = {}
        for orig_id in self.selected_classes:
            cnt = int((raw_targets == orig_id).sum())
            if cnt > 0:
                dist[self.all_classes[orig_id]] = cnt
        return dist

    def get_client_train_loader(self, client_id: int, shuffle: bool = True) -> DataLoader:
        """Returns training DataLoader for a specific multi-class client."""
        ds = self.client_train_datasets[client_id]
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def get_forget_train_loader(self, forget_client_id: int, shuffle: bool = False) -> DataLoader:
        """Returns DataLoader containing the forgotten client's private training data (D_forget)."""
        indices = self.client_indices[forget_client_id]
        ds = RemappedCIFARSubset(self.raw_train_eval, indices, self.label_map)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def get_retain_train_loader(self, forget_client_id: int, shuffle: bool = True) -> DataLoader:
        """Returns DataLoader containing all training data of retained clients."""
        retain_indices = []
        for cid in range(self.num_clients):
            if cid != forget_client_id:
                retain_indices.extend(self.client_indices[cid])
        ds = RemappedCIFARSubset(self.raw_train, retain_indices, self.label_map)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def get_full_test_loader(self) -> DataLoader:
        """Returns holdout test DataLoader across all selected classes."""
        all_test_indices = []
        for orig_class in self.selected_classes:
            all_test_indices.extend(self.test_class_indices[orig_class])
        ds = RemappedCIFARSubset(self.raw_test, all_test_indices, self.label_map)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def get_forget_test_loader(self, forget_client_id: int) -> DataLoader:
        """For compatibility with single-class interface, returns forgotten client private train data."""
        return self.get_forget_train_loader(forget_client_id)

    def get_sample_images_for_client(self, client_id: int, count: int = 6) -> Tuple[torch.Tensor, List[int]]:
        """Retrieve raw normalized tensors and labels from a client's private dataset for Grad-CAM."""
        indices = self.client_indices[client_id][:count]
        ds = RemappedCIFARSubset(self.raw_train_eval, indices, self.label_map)
        images = []
        labels = []
        for i in range(len(ds)):
            img, label = ds[i]
            images.append(img.unsqueeze(0))
            labels.append(label)
        return torch.cat(images, dim=0), labels

    def get_sample_images_for_class(self, class_id: int, count: int = 5) -> Tuple[torch.Tensor, List[int]]:
        """Retrieve raw normalized tensors and labels for specific class images for Grad-CAM."""
        orig_class = self.selected_classes[class_id]
        indices = self.test_class_indices[orig_class][:count]
        ds = RemappedCIFARSubset(self.raw_test, indices, self.label_map)
        images = []
        labels = []
        for i in range(len(ds)):
            img, label = ds[i]
            images.append(img.unsqueeze(0))
            labels.append(label)
        return torch.cat(images, dim=0), labels
