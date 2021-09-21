import os
import logging
import string

import datasets
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class AmazonCorpus:
    """
    This class load `amazon_us_reviews` dataset from :huggingface: Transformers
    For more details about this dataset, check out: https://huggingface.co/datasets/amazon_us_reviews
    """
    def __init__(
            self,
            cache_dir: str,
            dataset_type: str,
            output_dir: str,
            validation_sample_size: float
    ) -> None:

        self.cache_dir = cache_dir
        self.output_dir = output_dir

        self.validation_sample_size = validation_sample_size
        self.dataset = datasets.load_dataset(
            "amazon_us_reviews",
            dataset_type,
            cache_dir=self.cache_dir
        )

    def generate_and_dump_corpus(self, corpus_chunk_len: int = 10_000) -> None:

        corpus_chunk = []
        file_count = 0
        product_count = 0

        dataset = self.dataset["train"]
        train_sample_size = 1 - self.validation_sample_size
        number_of_train_products = int(len(dataset) * train_sample_size)
        set_name = "train"

        logger.info("Extracting sentences from dataset and saving them as a corpus")

        for product in tqdm(dataset):
            product_count += 1
            if product_count >= number_of_train_products:
                set_name = "validation"

            # In this demo, the `product title` will be the selected textual representation
            # of the sentences in the language we want to learn
            product_title = product["product_title"].replace("\n", " ")
            product_title = self.normalize_text(product_title)
            corpus_chunk.append(product_title)

            if len(corpus_chunk) == corpus_chunk_len:
                destination_dir = os.path.join(self.output_dir, set_name)
                if not os.path.isdir(destination_dir):
                    os.makedirs(destination_dir)
                chunk_file_name = os.path.join(destination_dir, f"corpus_chunk_{file_count}.txt")
                logger.info(f"Saving corpus chunk number {file_count} to {chunk_file_name}")
                with(open(chunk_file_name, "w", encoding="utf-8")) as fp:
                    fp.write("\n".join(corpus_chunk))
                corpus_chunk = []
                file_count += 1

    @staticmethod
    def normalize_text(input_string: str) -> str:
        """
        We remove punctuation and strip out multiple blank spaces
        """
        string_wo_punctuation = input_string.translate(str.maketrans('', '', string.punctuation))
        return " ".join(string_wo_punctuation.split())

