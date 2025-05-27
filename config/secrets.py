# config/secrets.py
import os
from typing import Optional

class SecretsManager:
    """Manage application secrets and API keys"""
    
    def __init__(self):
        self.secrets = {}
        self.load_secrets()
    
    def load_secrets(self):
        """Load secrets from environment variables or Streamlit secrets"""
        
        try:
            import streamlit as st
            
            # Try to load from Streamlit secrets first
            if hasattr(st, 'secrets'):
                self.secrets.update(st.secrets)
        
        except ImportError:
            pass
        
        # Load from environment variables (these take precedence)
        env_secrets = {
            # Database secrets
            'DB_HOST': os.getenv('DB_HOST'),
            'DB_USER': os.getenv('DB_USER'),
            'DB_PASSWORD': os.getenv('DB_PASSWORD'),
            'DB_NAME': os.getenv('DB_NAME'),
            
            # Email secrets
            'SMTP_SERVER': os.getenv('SMTP_SERVER'),
            'SMTP_PORT': os.getenv('SMTP_PORT'),
            'EMAIL_USER': os.getenv('EMAIL_USER'),
            'EMAIL_PASSWORD': os.getenv('EMAIL_PASSWORD'),
            
            # API Keys
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
            'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
            'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
            
            # Third-party services
            'SENDGRID_API_KEY': os.getenv('SENDGRID_API_KEY'),
            'TWILIO_ACCOUNT_SID': os.getenv('TWILIO_ACCOUNT_SID'),
            'TWILIO_AUTH_TOKEN': os.getenv('TWILIO_AUTH_TOKEN'),
            
            # Security keys
            'SECRET_KEY': os.getenv('SECRET_KEY'),
            'JWT_SECRET': os.getenv('JWT_SECRET'),
            'ENCRYPTION_KEY': os.getenv('ENCRYPTION_KEY'),
            
            # External APIs
            'WEATHER_API_KEY': os.getenv('WEATHER_API_KEY'),
            'NEWS_API_KEY': os.getenv('NEWS_API_KEY'),
            'STOCK_API_KEY': os.getenv('STOCK_API_KEY')
        }
        
        # Update secrets with non-None environment variables
        for key, value in env_secrets.items():
            if value is not None:
                self.secrets[key] = value
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a secret value by key"""
        return self.secrets.get(key, default)
    
    def get_required(self, key: str) -> str:
        """Get a required secret value, raise error if not found"""
        value = self.secrets.get(key)
        if value is None:
            raise ValueError(f"Required secret '{key}' not found in environment or secrets")
        return value
    
    def set(self, key: str, value: str):
        """Set a secret value (for testing purposes)"""
        self.secrets[key] = value
    
    def has(self, key: str) -> bool:
        """Check if a secret exists"""
        return key in self.secrets and self.secrets[key] is not None
    
    def get_database_url(self) -> Optional[str]:
        """Get database URL from secrets"""
        db_type = self.get('DB_TYPE', 'postgresql')
        host = self.get('DB_HOST')
        port = self.get('DB_PORT', '5432')
        user = self.get('DB_USER')
        password = self.get('DB_PASSWORD')
        name = self.get('DB_NAME')
        
        if all([host, user, password, name]):
            return f"{db_type}://{user}:{password}@{host}:{port}/{name}"
        return None
    
    def get_email_config(self) -> dict:
        """Get email configuration from secrets"""
        return {
            'smtp_server': self.get('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(self.get('SMTP_PORT', '587')),
            'email_user': self.get('EMAIL_USER'),
            'email_password': self.get('EMAIL_PASSWORD'),
            'use_tls': True
        }
    
    def get_aws_config(self) -> dict:
        """Get AWS configuration from secrets"""
        return {
            'access_key_id': self.get('AWS_ACCESS_KEY_ID'),
            'secret_access_key': self.get('AWS_SECRET_ACCESS_KEY'),
            'region': self.get('AWS_REGION', 'us-east-1')
        }
    
    def mask_secret(self, value: str, show_chars: int = 4) -> str:
        """Mask a secret value for logging/display"""
        if not value or len(value) <= show_chars:
            return '*' * len(value) if value else ''
        
        return value[:show_chars] + '*' * (len(value) - show_chars)
    
    def validate_secrets(self) -> dict:
        """Validate that required secrets are present"""
        validation_results = {
            'valid': True,
            'missing_secrets': [],
            'warnings': []
        }
        
        # Define required secrets for different features
        required_secrets = {
            'database': ['DB_HOST', 'DB_USER', 'DB_PASSWORD', 'DB_NAME'],
            'email': ['EMAIL_USER', 'EMAIL_PASSWORD'],
            'security': ['SECRET_KEY']
        }
        
        for feature, secrets in required_secrets.items():
            missing = [secret for secret in secrets if not self.has(secret)]
            if missing:
                validation_results['missing_secrets'].extend(missing)
                validation_results['warnings'].append(
                    f"{feature.title()} feature may not work properly. Missing: {', '.join(missing)}"
                )
        
        if validation_results['missing_secrets']:
            validation_results['valid'] = False
        
        return validation_results

# Create global secrets manager instance
secrets_manager = SecretsManager()

# Convenience functions for backward compatibility
def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get a secret value"""
    return secrets_manager.get(key, default)

def get_required_secret(key: str) -> str:
    """Get a required secret value"""
    return secrets_manager.get_required(key)

def has_secret(key: str) -> bool:
    """Check if a secret exists"""
    return secrets_manager.has(key)

# Example secrets.toml file for Streamlit Cloud deployment
SECRETS_TEMPLATE = """
# Streamlit secrets.toml template
# Copy this to .streamlit/secrets.toml and fill in your values

[database]
DB_HOST = "your-database-host"
DB_USER = "your-database-user"
DB_PASSWORD = "your-database-password"
DB_NAME = "your-database-name"

[email]
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = "587"
EMAIL_USER = "your-email@gmail.com"
EMAIL_PASSWORD = "your-app-password"

[api_keys]
OPENAI_API_KEY = "your-openai-api-key"
GOOGLE_API_KEY = "your-google-api-key"

[aws]
AWS_ACCESS_KEY_ID = "your-aws-access-key"
AWS_SECRET_ACCESS_KEY = "your-aws-secret-key"
AWS_REGION = "us-east-1"

[security]
SECRET_KEY = "your-secret-key-for-sessions"
JWT_SECRET = "your-jwt-secret-key"

[third_party]
SENDGRID_API_KEY = "your-sendgrid-api-key"
WEATHER_API_KEY = "your-weather-api-key"
"""

