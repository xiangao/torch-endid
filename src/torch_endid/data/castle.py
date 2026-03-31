"""Castle Doctrine dataset for examples and testing."""

from pathlib import Path
import pandas as pd


def load_castle() -> pd.DataFrame:
    """Load the Castle Doctrine dataset.

    50 US states observed 2000-2010. Staggered adoption of Castle Doctrine
    laws (stand-your-ground). Treatment timing varies by state.

    Returns:
        pd.DataFrame with columns including:
            sid: state identifier
            year: calendar year (2000-2010)
            lhomicide: log homicide rate (outcome)
            effyear: year Castle Doctrine enacted (NA = never-treated)
            post: binary post-treatment indicator
    """
    csv_path = Path(__file__).parent.parent.parent.parent / "data" / "castle.csv"
    df = pd.read_csv(csv_path)
    # Clean gvar: effyear with NA/empty = never-treated
    df["gvar"] = df["effyear"].copy()
    df.loc[df["gvar"].isna(), "gvar"] = float("inf")
    return df
