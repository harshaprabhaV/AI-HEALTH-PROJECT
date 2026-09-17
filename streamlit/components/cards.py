import streamlit as st


def metric_card(
    title,
    value,
    description=None
):
    """
    Display a reusable metric card.
    """

    st.markdown(
        f"""
        <div style="
            padding: 20px;
            border-radius: 15px;
            border: 1px solid rgba(128,128,128,0.25);
            margin-bottom: 15px;
        ">

            <h4 style="margin-bottom: 5px;">
                {title}
            </h4>

            <h2 style="margin-top: 0;">
                {value}
            </h2>

            {
                f'<p style="opacity:0.7;">{description}</p>'
                if description
                else ""
            }

        </div>
        """,
        unsafe_allow_html=True
    )


def info_card(
    icon,
    title,
    description
):
    """
    Display a reusable information card.
    """

    st.markdown(
        f"""
        <div style="
            padding: 22px;
            border-radius: 15px;
            border: 1px solid rgba(128,128,128,0.25);
            margin-bottom: 15px;
        ">

            <h2>
                {icon} {title}
            </h2>

            <p style="opacity:0.75;">
                {description}
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


def success_card(
    title,
    value
):
    """
    Display a prediction/result card.
    """

    st.success(
        f"**{title}**\n\n{value}"
    )


def warning_card(
    title,
    message
):
    """
    Display a warning card.
    """

    st.warning(
        f"**{title}**\n\n{message}"
    )