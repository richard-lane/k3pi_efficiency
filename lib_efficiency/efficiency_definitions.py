"""
Definitions of stuff needed for creating/using the efficiency reweighter

"""
import pathlib

REWEIGHTER_DIR = pathlib.Path(__file__).resolve().parents[1] / "reweighter"
DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"


def mc_dump_path(year: str, sign: str, magnetisation: str) -> pathlib.Path:
    """
    Where the pickle dump for the MC dataframe lives

    """
    # TODO the rest of these
    assert year in {"2018"}
    assert sign in {"RS", "WS"}
    assert magnetisation in {"magdown"}

    return DATA_DIR / f"mc_{year}_{sign}_{magnetisation}.pkl"


def ampgen_dump_path(sign: str) -> pathlib.Path:
    """
    Where the pickle dump for the AmpGen dataframe lives

    """
    assert sign in {"RS", "WS"}

    return DATA_DIR / f"ampgen_{sign}.pkl"


def reweighter_path(year: str, sign: str, magnetisation: str) -> pathlib.Path:
    """
    Where the efficiency correction reweighter lives

    """
    # TODO the rest of these
    assert year in {"2018"}
    assert sign in {"RS", "WS"}
    assert magnetisation in {"magdown"}

    return REWEIGHTER_DIR / f"reweighter_{year}_{sign}_{magnetisation}.pkl"


def reweighter_exists(year: str, sign: str, magnetisation: str) -> bool:
    """
    Whether the reweighter has been created yet

    """
    return reweighter_path(year, sign, magnetisation).is_file()
