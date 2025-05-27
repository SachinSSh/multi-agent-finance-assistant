# streamlit_app/components/header.py
import streamlit as st
from config.settings import APP_CONFIG

def render_header():
    """Render the application header"""
    
    # Main title with custom styling
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="color: #667eea; margin-bottom: 0.5rem;">
            {APP_CONFIG['page_icon']} {APP_CONFIG['app_name']}
        </h1>
        <p style="color: #666; font-size: 1.1rem; margin-top: 0;">
            {APP_CONFIG['description']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation breadcrumb (optional)
    if 'current_page' in st.session_state:
        st.markdown(f"📍 **Current Page:** {st.session_state.current_page}")
    
    st.markdown("---")


