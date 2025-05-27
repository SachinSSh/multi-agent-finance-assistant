# streamlit_app/app.py
import streamlit as st
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import APP_CONFIG
from components.sidebar import render_sidebar
from components.header import render_header
from components.data_display import render_data_display, render_charts
from components.forms import render_contact_form
from utils import load_sample_data, process_data, format_currency

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title=APP_CONFIG['app_name'],
        page_icon=APP_CONFIG['page_icon'],
        layout=APP_CONFIG['layout'],
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Render header
    render_header()
    
    # Render sidebar
    sidebar_options = render_sidebar()
    
    # Main content based on sidebar selection
    if sidebar_options['page'] == "Dashboard":
        render_dashboard(sidebar_options)
    elif sidebar_options['page'] == "Data Analysis":
        render_data_analysis(sidebar_options)
    elif sidebar_options['page'] == "Settings":
        render_settings()
    elif sidebar_options['page'] == "Contact":
        render_contact_form()

def render_dashboard(options):
    """Render the main dashboard"""
    st.markdown('<div class="main-header"><h1>📊 Dashboard</h1></div>', unsafe_allow_html=True)
    
    # Load and process data
    data = load_sample_data()
    processed_data = process_data(data, options.get('filter_option', 'All'))
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Records", len(processed_data))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_value = processed_data['value'].mean() if 'value' in processed_data.columns else 0
        st.metric("Average Value", format_currency(avg_value))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        max_value = processed_data['value'].max() if 'value' in processed_data.columns else 0
        st.metric("Max Value", format_currency(max_value))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        min_value = processed_data['value'].min() if 'value' in processed_data.columns else 0
        st.metric("Min Value", format_currency(min_value))
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data display and charts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_charts(processed_data)
    
    with col2:
        render_data_display(processed_data.head(10))

def render_data_analysis(options):
    """Render data analysis page"""
    st.markdown('<div class="main-header"><h1>🔍 Data Analysis</h1></div>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload your data file", type=['csv', 'xlsx', 'json'])
    
    if uploaded_file:
        st.success("File uploaded successfully!")
        # Here you would process the uploaded file
        st.info("File processing functionality would be implemented here")
    else:
        # Use sample data
        data = load_sample_data()
        st.info("Using sample data. Upload a file to analyze your own data.")
        
        # Analysis options
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Statistical Summary", "Correlation Analysis", "Distribution Analysis"]
        )
        
        if analysis_type == "Statistical Summary":
            st.subheader("Statistical Summary")
            st.dataframe(data.describe())
        
        elif analysis_type == "Correlation Analysis":
            st.subheader("Correlation Matrix")
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr()
                st.dataframe(corr_matrix)
            else:
                st.warning("Need at least 2 numeric columns for correlation analysis")
        
        elif analysis_type == "Distribution Analysis":
            st.subheader("Data Distribution")
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select column to analyze", numeric_cols)
                st.histogram_chart(data[selected_col])
            else:
                st.warning("No numeric columns found for distribution analysis")

def render_settings():
    """Render settings page"""
    st.markdown('<div class="main-header"><h1>⚙️ Settings</h1></div>', unsafe_allow_html=True)
    
    st.subheader("Application Settings")
    
    # Theme settings
    theme = st.selectbox("Select Theme", ["Light", "Dark", "Auto"])
    st.session_state.theme = theme
    
    # Data refresh rate
    refresh_rate = st.slider("Data Refresh Rate (seconds)", 5, 300, 60)
    st.session_state.refresh_rate = refresh_rate
    
    # Notification settings
    notifications = st.checkbox("Enable Notifications", value=True)
    st.session_state.notifications = notifications
    
    # Export settings
    st.subheader("Export Settings")
    export_format = st.selectbox("Default Export Format", ["CSV", "Excel", "JSON"])
    st.session_state.export_format = export_format
    
    # Save button
    if st.button("Save Settings", type="primary"):
        st.success("Settings saved successfully!")
        # Here you would typically save settings to a database or config file

if __name__ == "__main__":
    main()


