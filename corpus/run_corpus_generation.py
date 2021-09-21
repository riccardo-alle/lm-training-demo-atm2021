from argparse import ArgumentParser
from typing import Dict, Any

from corpus import AmazonCorpus

DEFAULT_CACHE_DIR = "corpus/dataset"
AVAILABLE_DATASET_TYPES = ["Apparel_v1_00", "Automotive_v1_00", "Baby_v1_00", "Beauty_v1_00", "Books_v1_00"]
DEFAULT_DATASET_TYPE = "Baby_v1_00"
DEFAULT_OUTPUT_DIR = "corpus/preprocessed_corpus"
DEFAULT_VAL_SAMPLE_SIZE = 0.1


def parse_args_to_dict() -> Dict[str, Any]:
    parser = ArgumentParser()

    parser.add_argument(
        "--cache-dir",
        help="Specify dir where huggingface Transformers dataset will be cached after being downloaded",
        type=str,
        default=DEFAULT_CACHE_DIR
    )
    parser.add_argument(
        "--output-dir",
        help="Specify dir where the pre-processed corpus will be stored",
        type=str,
        default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--dataset-type",
        help="Specify which dataset type you want to download",
        type=str,
        default=DEFAULT_DATASET_TYPE,
        choices=AVAILABLE_DATASET_TYPES
    )
    parser.add_argument(
        "--validation-sample-size",
        help="Specify how much data to use as validation set",
        type=float,
        default=DEFAULT_VAL_SAMPLE_SIZE
    )

    parsed_args = parser.parse_args()
    return vars(parsed_args)


if __name__ == "__main__":
    args = parse_args_to_dict()

    (
        AmazonCorpus(
            cache_dir=args["cache_dir"],
            dataset_type=args["dataset_type"],
            output_dir=args["output_dir"],
            validation_sample_size=args["validation_sample_size"]
        )
        .generate_and_dump_corpus()
    )
