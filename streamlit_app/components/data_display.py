# streamlit_app/components/data_display.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def render_data_display(data):
    """Render data in a formatted table"""
    
    st.subheader("📋 Data Overview")
    
    if data.empty:
        st.warning("No data to display")
        return
    
    # Display options
    col1, col2 = st.columns([3, 1])
    
    with col2:
        show_index = st.checkbox("Show Index", value=False)
        max_rows = st.number_input("Max Rows", min_value=5, max_value=100, value=10)
    
    # Display data
    display_data = data.head(max_rows)
    
    if show_index:
        st.dataframe(display_data, use_container_width=True)
    else:
        st.dataframe(display_data.reset_index(drop=True), use_container_width=True)
    
    # Data info
    st.caption(f"Showing {len(display_data)} of {len(data)} rows")

def render_charts(data):
    """Render various charts based on data"""
    
    st.subheader("📈 Data Visualization")
    
    if data.empty:
        st.warning("No data available for visualization")
        return
    
    # Chart selection
    chart_tabs = st.tabs(["Line Chart", "Bar Chart", "Histogram", "Scatter Plot"])
    
    numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    
    with chart_tabs[0]:  # Line Chart
        if len(numeric_columns) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", numeric_columns, key="line_x")
            with col2:
                y_col = st.selectbox("Y-axis", numeric_columns, key="line_y")
            
            fig = px.line(data, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns for line chart")
    
    with chart_tabs[1]:  # Bar Chart
        if len(categorical_columns) >= 1 and len(numeric_columns) >= 1:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Categories", categorical_columns, key="bar_x")
            with col2:
                y_col = st.selectbox("Values", numeric_columns, key="bar_y")
            
            fig = px.bar(data, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 1 categorical and 1 numeric column for bar chart")
    
    with chart_tabs[2]:  # Histogram
        if len(numeric_columns) >= 1:
            col = st.selectbox("Column", numeric_columns, key="hist_col")
            bins = st.slider("Number of bins", min_value=5, max_value=50, value=20)
            
            fig = px.histogram(data, x=col, nbins=bins, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 1 numeric column for histogram")
    
    with chart_tabs[3]:  # Scatter Plot
        if len(numeric_columns) >= 2:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X-axis", numeric_columns, key="scatter_x")
            with col2:
                y_col = st.selectbox("Y-axis", numeric_columns, key="scatter_y")
            with col3:
                color_col = st.selectbox("Color by", ["None"] + categorical_columns, key="scatter_color")
            
            color_param = None if color_col == "None" else color_col
            fig = px.scatter(data, x=x_col, y=y_col, color=color_param, title=f"{y_col} vs {x_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns for scatter plot")


