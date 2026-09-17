import streamlit as st
from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

STREAMLIT_DIR = CURRENT_FILE.parent.parent
PROJECT_DIR = STREAMLIT_DIR.parent

DATA_DIR = PROJECT_DIR / "data"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Medicine Search",
    page_icon="💊",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("💊 Medicine Search")

st.write(
    """
    Search and explore medicine information available in the
    healthcare dataset.
    """
)

st.divider()


# ============================================================
# LOAD MEDICINE DATASET
# ============================================================

@st.cache_data
def load_medicine_data():

    file_path = DATA_DIR / "india_medicine_data.csv"

    if not file_path.exists():
        return None

    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin1"
    ]

    for encoding in encodings:

        try:

            return pd.read_csv(
                file_path,
                encoding=encoding
            )

        except UnicodeDecodeError:

            continue

    return None


# ============================================================
# LOAD DATA
# ============================================================

df = load_medicine_data()


if df is None:

    st.error(
        "❌ `india_medicine_data.csv` was not found."
    )

    st.info(
        "Place your medicine CSV file inside the `data/` folder."
    )

    st.stop()


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = [
    str(column).strip()
    for column in df.columns
]


# ============================================================
# DATASET INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Total Medicines",
        f"{len(df):,}"
    )

with col2:

    st.metric(
        "Dataset Columns",
        len(df.columns)
    )

with col3:

    st.metric(
        "Source",
        "Medicine Dataset"
    )


st.divider()


# ============================================================
# SEARCH
# ============================================================

st.subheader("🔎 Search Medicine")


search_text = st.text_input(
    "Medicine Search",
    placeholder=(
        "Enter medicine name, generic name, category..."
    )
)


# ============================================================
# FILTER
# ============================================================

filtered_df = df.copy()


if search_text.strip():

    # Get text columns

    text_columns = filtered_df.select_dtypes(
        include=["object"]
    ).columns


    if len(text_columns) > 0:

        search_mask = (
            filtered_df[
                text_columns
            ]
            .astype(str)
            .apply(
                lambda column:
                column.str.contains(
                    search_text,
                    case=False,
                    na=False
                )
            )
            .any(axis=1)
        )


        filtered_df = filtered_df[
            search_mask
        ]


# ============================================================
# RESULTS
# ============================================================

st.divider()

st.subheader("📋 Medicine Results")


st.write(
    f"Found **{len(filtered_df):,}** matching records."
)


if filtered_df.empty:

    st.warning(
        "No medicine information matched your search."
    )

else:

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SELECT A MEDICINE
# ============================================================

if not filtered_df.empty:

    st.divider()

    st.subheader(
        "💊 Medicine Details"
    )


    # Find a likely medicine-name column

    possible_name_columns = [
        "medicine_name",
        "medicine",
        "name",
        "drug_name",
        "drug",
        "brand_name"
    ]


    name_column = None


    for column in filtered_df.columns:

        normalized = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
        )


        if normalized in possible_name_columns:

            name_column = column

            break


    # If no obvious name column exists,
    # use the first column

    if name_column is None:

        name_column = filtered_df.columns[0]


    medicine_names = (
        filtered_df[name_column]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )


    selected_medicine = st.selectbox(
        "Select a medicine",
        medicine_names
    )


    selected_rows = filtered_df[
        filtered_df[name_column]
        .astype(str)
        ==
        selected_medicine
    ]


    if not selected_rows.empty:

        medicine_data = selected_rows.iloc[0]


        for column in selected_rows.columns:

            value = medicine_data[column]


            if pd.isna(value):

                continue


            st.markdown(
                f"**{column}:** {value}"
            )


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

if not filtered_df.empty:

    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="⬇️ Download Search Results",
        data=csv_data,
        file_name="medicine_search_results.csv",
        mime="text/csv"
    )


# ============================================================
# AVAILABLE COLUMNS
# ============================================================

st.divider()


with st.expander(
    "📌 Available Dataset Columns"
):

    for column in df.columns:

        st.write(
            f"• `{column}`"
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    """
    ⚠️ **Important**

    This page is a medicine information/search tool only.

    It does NOT prescribe medicines or recommend a dosage.

    Do not start, stop, or change medication based solely on
    information displayed here. Consult a qualified doctor
    or pharmacist for medication decisions.
    """
)