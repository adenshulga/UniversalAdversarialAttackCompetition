"""
Here i need to allocate resources, load model and dataset
"""

import typing as tp

import torch
from PIL.Image import Image
from torch import Tensor
from torch.nn import Module as TorchModel
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import TensorDataset
from torchvision import transforms

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
EPSILON = 4e-2

model: TorchModel = tp.cast(
    TorchModel,
    torch.hub.load(
        "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True
    ),
)  # resnet18 trained on CIFAR-10 dataset

model.to(DEVICE)

cast_to_tensor: tp.Callable[[Image], Tensor] = transforms.ToTensor()
cast_to_image: tp.Callable[[Tensor], Image] = transforms.ToPILImage()


def load_data(filepath: str) -> TorchDataset:
    # Load the saved dictionary
    loaded_data = torch.load(filepath)

    images_tensor = loaded_data["images"]
    labels_tensor = loaded_data["labels"]

    dataset = TensorDataset(images_tensor, labels_tensor)

    print(f"Subset loaded successfully from {filepath}")
    return dataset


cifar10_tensors = load_data("data/test_data.pt")


cifar10_dataloader = TorchDataLoader(
    dataset=cifar10_tensors, batch_size=128, shuffle=False
)


def l_inf_img_norm(img: Tensor) -> float:
    return torch.max(torch.abs(img)).item()


def check_validity_of_pertubations(
    pertubations: list[Tensor],
    norm_function: tp.Callable[[Tensor], float] = l_inf_img_norm,
    max_norm: float = EPSILON,
) -> bool:
    """
    Check whether the norm of each pertubation is within bounds
    """
    for pert in pertubations:
        if norm_function(pert) > max_norm:
            return False

    return True


def score_adversarial_attack(
    pertubations: list[Tensor],
    model: torch.nn.Module,
    dataloader: TorchDataLoader,
    device: torch.device,
) -> float:
    """
    Score is evaluated as a percent of samples on which model changed its predictions("fooling ratio")
    Attack is successful if at least one of passed pertubations changes prediction of the model
    """

    model.eval()

    num_samples = len(dataloader.dataset)  # type: ignore

    attack_results: list[Tensor] = []

    with torch.no_grad():
        original_preds_list = []
        for images, _ in dataloader:
            images = images.to(device)
            logits = model(images)  # Shape: [batch_size, num_classes]
            preds = logits.argmax(dim=1)  # Shape: [batch_size]
            original_preds_list.append(preds.cpu())

        original_preds = torch.cat(original_preds_list, dim=0)  # Shape: [num_samples]

        for pert_idx, pert in enumerate(pertubations):
            pert = pert.to(device)  # just in case

            fooled_mask = torch.zeros(num_samples, dtype=torch.bool)
            offset = 0  # Tracks position in the fooled_mask

            for images, _ in dataloader:
                batch_size = images.size(0)

                images = images.to(device)

                attacked_images = images + pert.to(device)

                attacked_logits = model(attacked_images)
                attacked_preds = attacked_logits.argmax(dim=1).cpu()  # [batch_size]

                # Compare with the corresponding original predictions
                original_batch_preds = original_preds[offset : offset + batch_size]
                fooled_batch = attacked_preds.ne(original_batch_preds)  # bool tensor

                fooled_mask[offset : offset + batch_size] = fooled_batch
                offset += batch_size

            attack_results.append(fooled_mask)

    all_fooled_tensor = torch.stack(attack_results, dim=0)
    fooled_by_any = all_fooled_tensor.any(dim=0)  # Logical OR

    num_fooled_samples = fooled_by_any.sum().item()
    fooling_ratio = num_fooled_samples / num_samples

    return fooling_ratio


def score_submission(pertubations: list[Tensor]) -> float:
    return score_adversarial_attack(pertubations, model, cifar10_dataloader, DEVICE)
