import os
import logging
import typing as T

import pytorch_lightning as pl
import torch
from tokenizers import Tokenizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


logger = logging.getLogger(__name__)


class LMDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_file_path: str) -> None:

        all_corpus_chunk_paths = self.get_all_corpus_files(dataset_file_path)
        self.data = []
        for path in all_corpus_chunk_paths:
            with open(path, mode="r") as dataset_file:
                data = dataset_file.readlines()
            data_chunk = [line.strip() for line in data]
            self.data.extend(data_chunk)
            dataset_file.close()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> str:
        return self.data[idx]

    @staticmethod
    def get_all_corpus_files(path_to_corpus: str) -> T.List[str]:
        all_corpus_files = os.listdir(path_to_corpus)
        return [os.path.join(path_to_corpus, corpus_chunk) for corpus_chunk in all_corpus_files]


class LMDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_dataset_file_path: str,
            val_dataset_file_path: str,
            tokenizer: Tokenizer,
            batch_size: T.Optional[int] = 16,
            num_workers: T.Optional[int] = 0,
    ) -> None:
        super().__init__()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self._train_dataset_file_path = train_dataset_file_path
        self._val_dataset_file_path = val_dataset_file_path
        self._batch_size = batch_size
        self._num_workers = num_workers

        self._tokenizer = tokenizer
        self._add_pad_token_to_tokenizer()

    def prepare_data(self, stage: T.Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            pass

        if stage == "test" or stage is None:
            pass

    def setup(self, stage: T.Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            logger.info("Building Training Dataset")
            self.train_dataset = LMDataset(self._train_dataset_file_path)
            logger.info("Building Validation Dataset")
            self.val_dataset = LMDataset(self._val_dataset_file_path)

        if stage == "test" or stage is None:
            pass

    def dataloader(
            self,
            dataset: Dataset,
            shuffle: T.Optional[bool] = True,
            collate_fn: T.Optional[T.Callable] = None,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            shuffle=shuffle,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(
            self.train_dataset, shuffle=True, collate_fn=self._tokenize_and_mask_batch
        )

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(
            self.val_dataset, shuffle=False, collate_fn=self._tokenize_and_mask_batch
        )

    def test_dataloader(self) -> DataLoader:
        return self.dataloader(
            self.test_dataset, shuffle=False, collate_fn=self._tokenize_and_mask_batch
        )

    def _add_pad_token_to_tokenizer(self) -> None:
        self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def _tokenize_and_mask_batch(self, batch: T.List[str]) -> T.Dict[str, torch.Tensor]:

        # Tokenize batch of sentences
        encoded_batch = self._tokenizer(batch, padding=True)

        # Get labels, input_ids, attention_masks vectors
        labels = torch.tensor(encoded_batch["input_ids"])
        input_ids = labels.detach().clone()
        attention_mask = torch.tensor(encoded_batch["attention_mask"])

        mask_token_id = 3
        mask_probability = 0.15

        # Now we need to mask out randomly 0.15 of the tokens in input_ids
        # Create random array of floats with equal dims to input_ids
        rand = torch.rand(input_ids.shape)
        # Mask random 15% where token is not 0 [PAD], 1 [CLS], or 2 [SEP]
        mask_arr = (rand < mask_probability) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)

        # Loop through each vector in the batch
        for i in range(input_ids.shape[0]):
            # Get indices of mask positions from mask array
            selection = torch.flatten(mask_arr[i].nonzero()).tolist()
            # Mask input_ids
            input_ids[i, selection] = mask_token_id

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

