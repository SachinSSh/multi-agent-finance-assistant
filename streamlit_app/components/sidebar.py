# streamlit_app/components/sidebar.py
import streamlit as st
from config.settings import SIDEBAR_OPTIONS

def render_sidebar():
    """Render sidebar with navigation and filters"""
    
    st.sidebar.title("🚀 Navigation")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Select Page",
        SIDEBAR_OPTIONS['pages']
    )
    
    st.sidebar.markdown("---")
    
    # Filters section
    st.sidebar.subheader("🔧 Filters")
    
    filter_option = st.sidebar.selectbox(
        "Filter Data",
        SIDEBAR_OPTIONS['filter_options']
    )
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=None,
        help="Filter data by date range"
    )
    
    # Additional controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Display Options")
    
    show_raw_data = st.sidebar.checkbox("Show Raw Data", value=False)
    chart_type = st.sidebar.selectbox(
        "Chart Type",
        ["Line", "Bar", "Area", "Scatter"]
    )
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Tip**: Use the filters above to customize your data view. "
        "Upload your own data in the Data Analysis page."
    )
    
    return {
        'page': page,
        'filter_option': filter_option,
        'date_range': date_range,
        'show_raw_data': show_raw_data,
        'chart_type': chart_type
    }

