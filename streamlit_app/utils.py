# streamlit_app/utils.py
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import streamlit as st

def load_sample_data():
    """Load sample data for demonstration"""
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    
    data = pd.DataFrame({
        'date': np.random.choice(dates, 1000),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
        'value': np.random.normal(100, 25, 1000),
        'quantity': np.random.randint(1, 100, 1000),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
        'status': np.random.choice(['Active', 'Inactive', 'Pending'], 1000)
    })
    
    # Ensure positive values
    data['value'] = np.abs(data['value'])
    
    return data.sort_values('date').reset_index(drop=True)

def process_data(data, filter_option='All'):
    """Process and filter data based on options"""
    
    if data.empty:
        return data
    
    processed_data = data.copy()
    
    # Apply filters
    if filter_option != 'All':
        if filter_option in processed_data.columns:
            # Filter by column value
            processed_data = processed_data[processed_data[filter_option].notna()]
        elif filter_option == 'Recent':
            # Filter recent data (last 30 days)
            if 'date' in processed_data.columns:
                cutoff_date = datetime.now() - timedelta(days=30)
                processed_data = processed_data[
                    pd.to_datetime(processed_data['date']) >= cutoff_date
                ]
        elif filter_option == 'High Value':
            # Filter high value records
            if 'value' in processed_data.columns:
                threshold = processed_data['value'].quantile(0.75)
                processed_data = processed_data[processed_data['value'] >= threshold]
    
    return processed_data

def format_currency(value, currency_symbol='$'):
    """Format numeric value as currency"""
    
    if pd.isna(value):
        return f"{currency_symbol}0.00"
    
    return f"{currency_symbol}{value:,.2f}"

def format_percentage(value, decimal_places=2):
    """Format numeric value as percentage"""
    
    if pd.isna(value):
        return "0.00%"
    
    return f"{value:.{decimal_places}f}%"

def validate_email(email):
    """Validate email address format"""
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def clean_data(data):
    """Basic data cleaning operations"""
    
    if data.empty:
        return data
    
    cleaned_data = data.copy()
    
    # Remove duplicate rows
    cleaned_data = cleaned_data.drop_duplicates()
    
    # Handle missing values
    numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
    
    categorical_columns = cleaned_data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        cleaned_data[col] = cleaned_data[col].fillna('Unknown')
    
    return cleaned_data

def calculate_summary_stats(data, column):
    """Calculate summary statistics for a numeric column"""
    
    if column not in data.columns or data[column].dtype not in ['int64', 'float64']:
        return None
    
    series = data[column].dropna()
    
    if len(series) == 0:
        return None
    
    stats = {
        'count': len(series),
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'q25': series.quantile(0.25),
        'q75': series.quantile(0.75)
    }
    
    return stats

def export_data(data, format_type='csv', filename=None):
    """Export data in various formats"""
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"export_{timestamp}"
    
    try:
        if format_type.lower() == 'csv':
            csv = data.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{filename}.csv",
                mime="text/csv"
            )
        
        elif format_type.lower() == 'excel':
            # Note: This requires openpyxl
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                data.to_excel(writer, sheet_name='Data', index=False)
            
            st.download_button(
                label="📥 Download Excel",
                data=buffer.getvalue(),
                file_name=f"{filename}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        elif format_type.lower() == 'json':
            json_str = data.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"{filename}.json",
                mime="application/json"
            )
        
        return True
        
    except Exception as e:
        st.error(f"Error exporting data: {str(e)}")
        return False

def send_email(form_data):
    """Send email notification (placeholder function)"""
    
    # This is a placeholder function
    # In a real application, you would integrate with an email service
    # like SendGrid, Amazon SES, or SMTP
    
    print(f"Email would be sent with data: {form_data}")
    return True

@st.cache_data
def load_cached_data(file_path):
    """Load data with caching for better performance"""
    
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return pd.DataFrame()

def format_number(value, format_type='comma'):
    """Format numbers for display"""
    
    if pd.isna(value):
        return "N/A"
    
    if format_type == 'comma':
        return f"{value:,}"
    elif format_type == 'abbreviated':
        if abs(value) >= 1e9:
            return f"{value/1e9:.1f}B"
        elif abs(value) >= 1e6:
            return f"{value/1e6:.1f}M"
        elif abs(value) >= 1e3:
            return f"{value/1e3:.1f}K"
        else:
            return f"{value:.1f}"
    else:
        return str(value)

