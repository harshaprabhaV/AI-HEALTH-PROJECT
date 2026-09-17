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
# READ CSV SAFELY
# ============================================================

def read_csv_flexible(file_path):

    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin1"
    ]

    last_error = None

    for encoding in encodings:

        try:

            return pd.read_csv(
                file_path,
                encoding=encoding
            )

        except UnicodeDecodeError as error:

            last_error = error


    if last_error:

        raise last_error


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(file_name):

    file_path = DATA_DIR / file_name


    if not file_path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )


    df = read_csv_flexible(
        file_path
    )


    # Clean column names

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]


    return df


# ============================================================
# LOAD OPTIONAL DATASET
# ============================================================

def load_optional_dataset(file_name):

    file_path = DATA_DIR / file_name


    if not file_path.exists():

        return None


    df = read_csv_flexible(
        file_path
    )


    df.columns = [
        str(column).strip()
        for column in df.columns
    ]


    return df


# ============================================================
# FIND DATASET FILE
# ============================================================

def find_data_file(file_names):

    for file_name in file_names:

        file_path = DATA_DIR / file_name

        if file_path.exists():

            return file_path


    return None


# ============================================================
# LOAD FIRST AVAILABLE DATASET
# ============================================================

def load_first_available(file_names):

    file_path = find_data_file(
        file_names
    )


    if file_path is None:

        return None


    df = read_csv_flexible(
        file_path
    )


    df.columns = [
        str(column).strip()
        for column in df.columns
    ]


    return df


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(
    df,
    possible_names
):

    if df is None:

        return None


    # Exact normalized match

    normalized_columns = {}

    for column in df.columns:

        normalized = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
        )

        normalized_columns[
            normalized
        ] = column


    for name in possible_names:

        normalized_name = (
            str(name)
            .strip()
            .lower()
            .replace(" ", "_")
        )


        if normalized_name in normalized_columns:

            return normalized_columns[
                normalized_name
            ]


    # Partial match

    for column in df.columns:

        column_name = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
        )


        for name in possible_names:

            name = (
                str(name)
                .strip()
                .lower()
                .replace(" ", "_")
            )


            if name in column_name:

                return column


    return None


# ============================================================
# SEARCH DATAFRAME
# ============================================================

def search_dataframe(
    df,
    keyword
):

    if df is None:

        return None


    if not keyword or not keyword.strip():

        return df.copy()


    keyword = keyword.strip()


    # Only search text columns

    text_columns = df.select_dtypes(
        include=["object"]
    ).columns


    if len(text_columns) == 0:

        return df.iloc[0:0].copy()


    mask = (
        df[text_columns]
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


    return df[mask].copy()


# ============================================================
# GET DATASET INFORMATION
# ============================================================

def get_dataset_info(df):

    if df is None:

        return {
            "rows": 0,
            "columns": 0
        }


    return {
        "rows": len(df),
        "columns": len(df.columns)
    }