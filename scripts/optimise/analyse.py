"""
Analyse the output of the optimise.py script

Its output is a series of text files - we will need to concatenate them together

"""
import glob
import argparse


def main(
    year: str,
    sign: str,
    magnetisation: str,
    k_sign: str,
    fit: bool,
):
    """
    Read the right output files and find which hyperparameters gives the best performance

    """
    # Find what the output files are called
    output_files = glob.glob("output*.out")

    # Just print all the stuff
    for output_file in output_files:
        with open(output_file, "r") as f:
            print(f.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make plots of ampgen + MC phase space variables, and do the reweighting."
    )
    parser.add_argument(
        "sign",
        type=str,
        choices={"cf", "dcs"},
        help="type of decay, CF or DCS",
    )
    parser.add_argument(
        "k_sign",
        type=str,
        choices={"k_plus", "k_minus"},
        help="whether to read K+ or K- data",
    )
    parser.add_argument("--fit", action="store_true")

    args = parser.parse_args()
    main(
        args.sign,
        args.k_sign,
        args.fit,
    )
