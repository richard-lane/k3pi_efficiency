"""
Definitions of stuff needed for creating/using the efficiency reweighter

"""
import pathlib

REWEIGHTER_DIR = pathlib.Path(__file__).resolve().parents[1] / "reweighter"

# Time below which we just throw away events - the reweighting here is too unstable
MIN_TIME = 0.3


def reweighter_path(
    year: str, sign: str, magnetisation: str, time_fit: bool = False
) -> pathlib.Path:
    """
    Where the efficiency correction reweighter lives

    """
    # TODO the rest of these
    assert year in {"2018"}
    assert sign in {"cf", "dcs"}
    assert magnetisation in {"magdown"}

    time_suffix = "_time_fit" if time_fit else ""

    return REWEIGHTER_DIR / f"{year}_{sign}_{magnetisation}{time_suffix}.pkl"


def reweighter_exists(
    year: str, sign: str, magnetisation: str, time_fit: bool = False
) -> bool:
    """
    Whether the reweighter has been created yet

    """
    return reweighter_path(year, sign, magnetisation, time_fit).is_file()
