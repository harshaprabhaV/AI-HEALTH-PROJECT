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
# LOAD MEDICINE DATASET
# ============================================================

def load_medicine_data():

    file_path = (
        DATA_DIR /
        "india_medicine_data.csv"
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"Medicine dataset not found: {file_path}"
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
            "Unable to read medicine dataset."
        )

    # Clean column names

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


# ============================================================
# SEARCH MEDICINES
# ============================================================

def search_medicines(
    df,
    keyword=None
):

    if df is None:

        return pd.DataFrame()


    result = df.copy()


    if not keyword or not keyword.strip():

        return result.reset_index(
            drop=True
        )


    keyword = keyword.strip()


    # Get text columns

    text_columns = result.select_dtypes(
        include=["object"]
    ).columns


    if len(text_columns) == 0:

        return pd.DataFrame(
            columns=result.columns
        )


    # Search across all text columns

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


    return result.reset_index(
        drop=True
    )


# ============================================================
# FIND MEDICINE NAME COLUMN
# ============================================================

def find_medicine_name_column(df):

    if df is None:

        return None


    possible_names = [
        "medicine_name",
        "medicine",
        "name",
        "drug_name",
        "drug",
        "brand_name",
        "product_name"
    ]


    # --------------------------------------------------------
    # Exact match
    # --------------------------------------------------------

    for column in df.columns:

        normalized = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
        )


        if normalized in possible_names:

            return column


    # --------------------------------------------------------
    # Partial match
    # --------------------------------------------------------

    for column in df.columns:

        normalized = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
        )


        for name in possible_names:

            if name in normalized:

                return column


    return None


# ============================================================
# GET MEDICINE DETAILS
# ============================================================

def get_medicine_details(
    df,
    medicine_name
):

    if df is None:

        return None


    name_column = find_medicine_name_column(
        df
    )


    if name_column is None:

        return None


    matches = df[
        df[name_column]
        .astype(str)
        .str.strip()
        .str.lower()
        ==
        str(medicine_name)
        .strip()
        .lower()
    ]


    if matches.empty:

        return None


    return matches.iloc[0].to_dict()


# ============================================================
# GET UNIQUE MEDICINES
# ============================================================

def get_medicine_names(df):

    if df is None:

        return []


    name_column = find_medicine_name_column(
        df
    )


    if name_column is None:

        return []


    names = (
        df[name_column]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )


    names.sort()


    return names


# ============================================================
# GET DATASET STATISTICS
# ============================================================

def get_medicine_statistics(df):

    if df is None:

        return {
            "total_records": 0,
            "total_columns": 0,
            "unique_medicines": 0
        }


    names = get_medicine_names(
        df
    )


    return {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "unique_medicines": len(names)
    }