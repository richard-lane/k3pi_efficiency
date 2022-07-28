"""
Definitions of stuff needed for creating/using the efficiency reweighter

"""
import pathlib

REWEIGHTER_DIR = pathlib.Path(__file__).resolve().parents[1] / "reweighter"


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
