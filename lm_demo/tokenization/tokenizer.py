import os
from glob import glob
import logging
from typing import List

from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer, normalizers, pre_tokenizers, processors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BPETokenizer:
    def __init__(
            self,
            vocab_size: int,
            corpus_dir: str,
            output_dir: str
    ) -> None:

        self.vocab_size = vocab_size

        self.corpus_dir = corpus_dir
        self.output_dir = output_dir

        self.tokenizer = self.build_tokenizer()

    @staticmethod
    def build_tokenizer() -> Tokenizer:
        tokenizer = Tokenizer(BPE())

        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        # Each sentence will be lowercased before being tokenized
        # We want the same representation for `this product` and `this PRODUCT`
        tokenizer.normalizer = normalizers.Lowercase()

        tokenizer.post_processor = processors.RobertaProcessing(
            sep=("SEP", 2),
            cls=("CLS", 1)
        )

        tokenizer.enable_padding()

        return tokenizer

    def save_tokenizer(self) -> None:
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        self.tokenizer.save(os.path.join(self.output_dir, "tokenizer.json"))

    def get_corpus_files(self) -> List[str]:
        return [path_file_txt for path in os.walk(self.corpus_dir)
                for path_file_txt in glob(os.path.join(path[0], '*.txt'))]

    def train(self) -> None:
        tokenizer_trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
            show_progress=True,
        )

        files = self.get_corpus_files()
        logger.info(f"Training Tokenizer with corpus {files}")
        self.tokenizer.train(files=files, trainer=tokenizer_trainer)

        logger.info(f"Saving Tokenizer to {self.output_dir}")
        self.save_tokenizer()




