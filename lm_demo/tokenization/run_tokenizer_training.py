from argparse import ArgumentParser
from typing import Dict, Any

from lm_demo.tokenization.tokenizer import BPETokenizer

DEFAULT_VOCAB_SIZE = 30_000
DEFAULT_OUTPUT_DIR = "./tokenizer"


def parse_args_to_dict() -> Dict[str, Any]:
    parser = ArgumentParser()

    parser.add_argument(
        "--corpus-dir",
        help="Specify dir where the corpus was stored",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output-dir",
        help="Specify dir where the trained tokenizer will be saved",
        type=str,
        default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--vocab-size",
        help="Specify the vocabulary size ",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
    )

    parsed_args = parser.parse_args()
    return vars(parsed_args)


if __name__ == "__main__":
    args = parse_args_to_dict()

    (
        BPETokenizer(
            corpus_dir=args["corpus_dir"],
            vocab_size=args["vocab_size"],
            output_dir=args["output_dir"]
        )
        .train()
    )
