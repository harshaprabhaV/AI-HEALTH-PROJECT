from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

STREAMLIT_DIR = CURRENT_FILE.parent.parent
PROJECT_DIR = STREAMLIT_DIR.parent

DATA_DIR = PROJECT_DIR / "data"


# ============================================================
# LOAD HOSPITAL / LAB DATA
# ============================================================

def load_hospital_data():

    file_path = (
        DATA_DIR /
        "india_hospitals_labs.csv"
    )


    if not file_path.exists():

        raise FileNotFoundError(
            f"Hospital dataset not found: {file_path}"
        )


    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin1"
    ]


    df = None


    for encoding in encodings:

        try:

            df = pd.read_csv(
                file_path,
                encoding=encoding
            )

            break

        except UnicodeDecodeError:

            continue


    if df is None:

        raise ValueError(
            "Unable to read hospital dataset."
        )


    # Clean column names

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]


    return df


# ============================================================
# SEARCH HOSPITALS / LABS
# ============================================================

def search_hospitals(
    df,
    keyword=None,
    location=None
):

    if df is None:

        return pd.DataFrame()


    result = df.copy()


    # ========================================================
    # GENERAL KEYWORD SEARCH
    # ========================================================

    if keyword and keyword.strip():

        keyword = keyword.strip()


        text_columns = result.select_dtypes(
            include=["object"]
        ).columns


        if len(text_columns) > 0:

            mask = (
                result[text_columns]
                .astype(str)
                .apply(
                    lambda column:
                    column.str.contains(
                        keyword,
                        case=False,
                        na=False
                    )
                )
                .any(axis=1)
            )


            result = result[
                mask
            ]


    # ========================================================
    # LOCATION SEARCH
    # ========================================================

    if location and location.strip():

        location = location.strip()


        text_columns = result.select_dtypes(
            include=["object"]
        ).columns


        if len(text_columns) > 0:

            location_mask = (
                result[text_columns]
                .astype(str)
                .apply(
                    lambda column:
                    column.str.contains(
                        location,
                        case=False,
                        na=False
                    )
                )
                .any(axis=1)
            )


            result = result[
                location_mask
            ]


    return result.reset_index(
        drop=True
    )


# ============================================================
# GET UNIQUE LOCATIONS
# ============================================================

def get_locations(df):

    if df is None:

        return []


    possible_columns = [
        "city",
        "location",
        "district",
        "state"
    ]


    location_column = None


    # Find likely location column

    for column in df.columns:

        normalized = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
        )


        if normalized in possible_columns:

            location_column = column

            break


    if location_column is None:

        return []


    locations = (
        df[location_column]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )


    locations.sort()


    return locations


# ============================================================
# GET DATASET STATISTICS
# ============================================================

def get_hospital_statistics(df):

    if df is None:

        return {
            "total_records": 0,
            "total_columns": 0,
            "locations": 0
        }


    locations = get_locations(
        df
    )


    return {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "locations": len(locations)
    }