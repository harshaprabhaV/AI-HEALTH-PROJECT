import streamlit as st
from pathlib import Path
import sys
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
    page_title="Hospitals & Labs",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🏥 Hospitals & Labs")

st.write(
    """
    Search and explore hospitals and laboratory information
    available in the healthcare dataset.
    """
)

st.divider()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_hospital_data():

    file_path = DATA_DIR / "india_hospitals_labs.csv"

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
# LOAD
# ============================================================

df = load_hospital_data()


if df is None:

    st.error(
        "❌ `india_hospitals_labs.csv` was not found."
    )

    st.info(
        "Place the hospital/lab CSV file inside the `data/` folder."
    )

    st.stop()


# ============================================================
# CLEAN DATA
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
        "Total Records",
        f"{len(df):,}"
    )

with col2:

    st.metric(
        "Total Columns",
        len(df.columns)
    )

with col3:

    st.metric(
        "Data Source",
        "India Dataset"
    )


st.divider()


# ============================================================
# SEARCH SECTION
# ============================================================

st.subheader("🔎 Search Hospitals & Labs")


col1, col2 = st.columns(2)


with col1:

    search_text = st.text_input(
        "Search",
        placeholder=(
            "Hospital name, laboratory, location..."
        )
    )


with col2:

    city_search = st.text_input(
        "City / Location",
        placeholder="Example: Chennai"
    )


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df.copy()


# ============================================================
# GENERAL SEARCH
# ============================================================

if search_text.strip():

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
# CITY FILTER
# ============================================================

if city_search.strip():

    text_columns = filtered_df.select_dtypes(
        include=["object"]
    ).columns


    if len(text_columns) > 0:

        city_mask = (
            filtered_df[
                text_columns
            ]
            .astype(str)
            .apply(
                lambda column:
                column.str.contains(
                    city_search,
                    case=False,
                    na=False
                )
            )
            .any(axis=1)
        )

        filtered_df = filtered_df[
            city_mask
        ]


# ============================================================
# RESULTS
# ============================================================

st.divider()

st.subheader("📋 Search Results")


st.write(
    f"Found **{len(filtered_df):,}** matching records."
)


if filtered_df.empty:

    st.warning(
        "No hospitals or laboratories matched your search."
    )

else:

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DOWNLOAD FILTERED DATA
# ============================================================

if not filtered_df.empty:

    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="⬇️ Download Search Results",
        data=csv_data,
        file_name="hospital_lab_search_results.csv",
        mime="text/csv"
    )


# ============================================================
# DATASET COLUMNS
# ============================================================

st.divider()

with st.expander("📌 Available Dataset Columns"):

    for column in df.columns:

        st.write(
            f"• `{column}`"
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.info(
    """
    ℹ️ Hospital and laboratory information displayed here
    comes from the provided dataset. Please verify current
    availability, address, contact details and services
    directly with the healthcare provider before visiting.
    """
)