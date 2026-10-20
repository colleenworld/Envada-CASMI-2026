from __future__ import print_function

import argparse
import os
import pickle

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()

    with open(args.input, "rb") as handle:
        data = pickle.load(handle)

    if not os.path.isdir(args.cache_dir):
        os.makedirs(args.cache_dir)

    formulas = sorted(data)
    total_formulas = len(formulas)

    total = 0
    kept = 0
    removed = 0
    out = {}

    for index, formula in enumerate(formulas, start=1):
        cache_path = os.path.join(
            args.cache_dir,
            "%06d.pkl" % index,
        )

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as handle:
                cached = pickle.load(handle)

            clean = cached["entries"]
            formula_total = cached["total"]
            formula_kept = cached["kept"]
            formula_removed = cached["removed"]
        else:
            clean = []
            seen = set()
            formula_total = 0
            formula_kept = 0
            formula_removed = 0

            for entry in data[formula]:
                formula_total += 1

                smi = str(entry["smi"])
                mol = Chem.MolFromSmiles(smi)

                if mol is None:
                    formula_removed += 1
                    continue

                canonical = Chem.MolToSmiles(
                    mol,
                    isomericSmiles=False,
                )

                ikey = str(entry["ikey"])
                key = (ikey, canonical)

                if key in seen:
                    continue

                seen.add(key)
                clean.append(
                    {
                        "smi": canonical,
                        "ikey": ikey,
                    }
                )
                formula_kept += 1

            payload = {
                "formula": formula,
                "entries": clean,
                "total": formula_total,
                "kept": formula_kept,
                "removed": formula_removed,
            }

            temporary = cache_path + ".tmp"

            with open(temporary, "wb") as handle:
                pickle.dump(
                    payload,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            os.replace(
                temporary,
                cache_path,
            )

        total += formula_total
        kept += formula_kept
        removed += formula_removed

        if clean:
            out[formula] = clean

        if (
            index % 25 == 0
            or index == total_formulas
        ):
            print(
                "Sanitized formulas: %d/%d | "
                "candidates=%s kept=%s removed=%s"
                % (
                    index,
                    total_formulas,
                    format(total, ","),
                    format(kept, ","),
                    format(removed, ","),
                ),
                flush=True,
            )

    temporary_output = args.output + ".tmp"

    with open(temporary_output, "wb") as handle:
        pickle.dump(
            out,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    os.replace(
        temporary_output,
        args.output,
    )

    print(
        "Candidate SMILES total: %s"
        % format(total, ",")
    )
    print(
        "Candidate SMILES kept: %s"
        % format(kept, ",")
    )
    print(
        "Candidate SMILES removed: %s"
        % format(removed, ",")
    )
    print(
        "Formulas retained: %s"
        % format(len(out), ",")
    )


if __name__ == "__main__":
    main()
