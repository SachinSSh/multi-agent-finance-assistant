# config/settings.py
import os
from datetime import datetime

# Application Configuration
APP_CONFIG = {
    'app_name': 'Data Analytics Dashboard',
    'version': '1.0.0',
    'description': 'A comprehensive data analytics and visualization platform',
    'page_icon': '📊',
    'layout': 'wide',
    'author': 'Your Name',
    'created_date': '2024-01-01'
}

# Sidebar Configuration
SIDEBAR_OPTIONS = {
    'pages': [
        'Dashboard',
        'Data Analysis', 
        'Settings',
        'Contact'
    ],
    'filter_options': [
        'All',
        'Recent',
        'High Value',
        'Active',
        'Pending'
    ]
}

# Chart Configuration
CHART_CONFIG = {
    'default_theme': 'plotly_white',
    'color_palette': [
        '#667eea', '#764ba2', '#f093fb', '#f5576c',
        '#4facfe', '#00f2fe', '#43e97b', '#38f9d7'
    ],
    'default_height': 400,
    'default_width': None
}

# Data Configuration
DATA_CONFIG = {
    'max_upload_size': 200,  # MB
    'supported_formats': ['csv', 'xlsx', 'json', 'parquet'],
    'default_sample_rows': 1000,
    'cache_ttl': 3600  # seconds
}

# UI Configuration
UI_CONFIG = {
    'theme': {
        'primary_color': '#667eea',
        'background_color': '#ffffff',
        'secondary_background_color': '#f8f9fa',
        'text_color': '#262730'
    },
    'fonts': {
        'main_font': 'Inter, sans-serif',
        'code_font': 'Fira Code, monospace'
    }
}

# Email Configuration (if using email features)
EMAIL_CONFIG = {
    'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
    'smtp_port': int(os.getenv('SMTP_PORT', '587')),
    'sender_email': os.getenv('SENDER_EMAIL', ''),
    'sender_name': 'Data Analytics Dashboard'
}

# Database Configuration (if using database)
DATABASE_CONFIG = {
    'type': os.getenv('DB_TYPE', 'sqlite'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'name': os.getenv('DB_NAME', 'analytics_db'),
    'user': os.getenv('DB_USER', ''),
    'password': os.getenv('DB_PASSWORD', '')
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_path': os.getenv('LOG_FILE_PATH', 'logs/app.log')
}

# Security Configuration
SECURITY_CONFIG = {
    'session_timeout': 3600,  # seconds
    'max_login_attempts': 5,
    'password_min_length': 8,
    'enable_2fa': False
}

# Feature Flags
FEATURE_FLAGS = {
    'enable_file_upload': True,
    'enable_data_export': True,
    'enable_real_time_updates': False,
    'enable_user_authentication': False,
    'enable_data_caching': True,
    'enable_email_notifications': False
}

# API Configuration (if using external APIs)
API_CONFIG = {
    'rate_limit': 1000,  # requests per hour
    'timeout': 30,  # seconds
    'retry_attempts': 3
}

# Development Configuration
DEV_CONFIG = {
    'debug_mode': os.getenv('DEBUG', 'False').lower() == 'true',
    'show_error_details': os.getenv('SHOW_ERROR_DETAILS', 'False').lower() == 'true',
    'enable_profiling': False
}

