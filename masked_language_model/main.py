from argparse import ArgumentParser
from typing import Dict, Any
import logging

from pytorch_lightning import seed_everything
from transformers import PreTrainedTokenizerFast

from masked_language_model.trainer import TrainerBuilder
from masked_language_model.model import RobertaForMaskedLMModule
from masked_language_model.dataset import LMDataModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_MAX_STEPS = 1000
DEFAULT_CHECKPOINT_EVERY_N_STEPS = 10
DEFAULT_VALIDATE_EVERY_N_STEPS = 50
DEFAULT_LR = 5e-4


def parse_args_to_dict() -> Dict[str, Any]:
    parser = ArgumentParser()

    parser.add_argument(
        "--job-dir",
        help="Specify directory where checkpoints and auxiliary files will be saved",
        type=str,
        required=True
    )
    parser.add_argument(
        "--path-to-train-set",
        help="Full path pointing to the training corpus (a directory containing one or several .txt files)",
        type=str,
        required=True
    )
    parser.add_argument(
        "--path-to-val-set",
        help="Full path pointing to the validation corpus (a directory containing one or several .txt files)",
        type=str,
        required=True
    )
    parser.add_argument(
        "--path-to-tokenizer",
        help="Full path pointing to pre-trained tokenizer.json",
        type=str,
        required=True
    )
    parser.add_argument(
        "--checkpoint-every-n-steps",
        help="Specify how often (in numbers of steps) you want to checkpoint your model",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY_N_STEPS
    )
    parser.add_argument(
        "--validate-every-n-steps",
        help="Specify how often (in numbers of steps) you want to checkpoint your model",
        type=int,
        default=DEFAULT_VALIDATE_EVERY_N_STEPS
    )
    parser.add_argument(
        "--seed",
        help="Specify a random seed to make the training deterministic",
        type=int,
        default=DEFAULT_SEED
    )
    parser.add_argument(
        "--lr",
        help="Specify learning rate",
        type=float,
        default=DEFAULT_LR
    )
    parser.add_argument(
        "--max-steps",
        help="Specify the number of steps you want to train your language model",
        type=int,
        default=DEFAULT_MAX_STEPS
    )

    parsed_args = parser.parse_args()
    return vars(parsed_args)


def run() -> None:
    args = parse_args_to_dict()

    seed_everything(args["seed"])

    tokenizer_path = args["path_to_tokenizer"]
    logger.info(f"Loading Tokenizer from {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    logger.info("Setting up the Trainer obj")
    trainer = TrainerBuilder(
        job_dir=args["job_dir"],
        max_steps=args["max_steps"],
        checkpoint_every_n_steps=args["checkpoint_every_n_steps"],
        validate_every_n_steps=args["validate_every_n_steps"]
    ).build()

    logger.info("Setting up the Masked Language Model")
    model = RobertaForMaskedLMModule(
        job_dir=args["job_dir"],
        max_steps=args["max_steps"],
        learning_rate=args["lr"],
        vocab_size=tokenizer.vocab_size
    )

    logger.info("Preparing the data module")
    data_module = LMDataModule(
        train_dataset_file_path=args["path_to_train_set"],
        val_dataset_file_path=args["path_to_val_set"],
        tokenizer=tokenizer
    )

    logger.info("Training started")
    trainer.fit(model, data_module)


if __name__ == "__main__":
    run()
