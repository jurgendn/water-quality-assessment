from typing import List, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class WaterQualityDataset(Dataset):
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target_feature_names: List[str],
        target_feature_indices: List[int],
    ) -> None:
        super(WaterQualityDataset, self).__init__()
        self.target_feature_names = target_feature_names
        self.target_feature_indices = target_feature_indices

        x = torch.FloatTensor(x)
        self.x = x.unsqueeze(dim=1)
        self.y = torch.FloatTensor(y[:, target_feature_indices])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index: int):
        x = self.x[index]
        y = self.y[index]
        return x, y


class WaterQualityDataModule(LightningDataModule):
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target_feature_names: List[str],
        target_feature_indices: List[int],
        val_ratio: float = 0.2,
        batch_size: int = 32,
    ) -> None:
        super(WaterQualityDataModule, self).__init__()
        assert 0 <= val_ratio < 1
        assert isinstance(batch_size, int)
        self.x = x
        self.y = y
        self.target_feature_indices = target_feature_indices
        self.target_feature_names = target_feature_names
        self.val_ratio = val_ratio
        self.batch_size = batch_size

        self.setup()
        self.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def prepare_data(self) -> None:
        num_records: int = len(self.x)
        train_size: int = num_records - int(num_records * self.val_ratio)

        self.train_dataset = WaterQualityDataset(
            x=self.x[:train_size],
            y=self.y[:train_size],
            target_feature_names=self.target_feature_names,
            target_feature_indices=self.target_feature_indices,
        )
        if train_size < num_records:
            self.val_dataset = WaterQualityDataset(
                x=self.x[train_size:],
                y=self.y[train_size:],
                target_feature_names=self.target_feature_names,
                target_feature_indices=self.target_feature_indices,
            )
        else:
            self.val_dataset = WaterQualityDataset(
                x=self.x[-self.batch_size :],
                y=self.y[-self.batch_size :],
                target_feature_names=self.target_feature_names,
                target_feature_indices=self.target_feature_indices,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size)
