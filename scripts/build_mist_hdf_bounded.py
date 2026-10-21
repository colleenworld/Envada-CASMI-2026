from __future__ import print_function

import argparse
import pickle
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mist-root", required=True)
    parser.add_argument("--labels-file", required=True)
    parser.add_argument("--form-to-smi", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--database-name", required=True)
    parser.add_argument(
        "--formulas-per-batch",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )
    args = parser.parse_args()

    mist_src = str(
        Path(args.mist_root) / "src"
    )

    sys.path.insert(
        0,
        mist_src,
    )

    import mist.retrieval_lib.make_hdf5 as mh
    from mist import utils

    if args.formulas_per_batch <= 0:
        raise ValueError(
            "--formulas-per-batch must be positive"
        )

    if args.workers <= 0:
        raise ValueError(
            "--workers must be positive"
        )

    mh.CHUNKSIZE = args.formulas_per_batch

    original_chunked_parallel = (
        utils.chunked_parallel
    )

    def limited_chunked_parallel(
        items,
        function,
        *positional,
        **keywords
    ):
        keywords["max_cpu"] = args.workers

        return original_chunked_parallel(
            items,
            function,
            *positional,
            **keywords
        )

    utils.chunked_parallel = (
        limited_chunked_parallel
    )

    with open(
        args.form_to_smi,
        "rb",
    ) as handle:
        form_to_smi = pickle.load(
            handle
        )

    print(
        "HDF formulas per batch:",
        args.formulas_per_batch,
    )
    print(
        "HDF fingerprint workers:",
        args.workers,
    )
    print(
        "Formulas:",
        len(form_to_smi),
    )

    mh.make_retrieval_hdf5(
        labels_file=args.labels_file,
        form_to_smi=form_to_smi,
        output_dir=args.output_dir,
        database_name=args.database_name,
        fp_names=["morgan4096"],
    )


if __name__ == "__main__":
    main()
