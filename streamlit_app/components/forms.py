# streamlit_app/components/forms.py
import streamlit as st
from utils import send_email, validate_email

def render_contact_form():
    """Render contact form"""
    
    st.markdown('<div class="main-header"><h1>📧 Contact Us</h1></div>', unsafe_allow_html=True)
    
    st.markdown("""
    We'd love to hear from you! Please fill out the form below and we'll get back to you as soon as possible.
    """)
    
    with st.form("contact_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name *", placeholder="Enter your full name")
            email = st.text_input("Email Address *", placeholder="your.email@example.com")
        
        with col2:
            company = st.text_input("Company/Organization", placeholder="Optional")
            phone = st.text_input("Phone Number", placeholder="Optional")
        
        subject = st.selectbox(
            "Subject *",
            ["General Inquiry", "Technical Support", "Feature Request", "Bug Report", "Partnership"]
        )
        
        message = st.text_area(
            "Message *",
            placeholder="Please describe your inquiry in detail...",
            height=150
        )
        
        # File attachment
        attachment = st.file_uploader(
            "Attach File (Optional)",
            type=['pdf', 'doc', 'docx', 'txt', 'png', 'jpg', 'jpeg'],
            help="Maximum file size: 10MB"
        )
        
        # Priority level
        priority = st.radio(
            "Priority Level",
            ["Low", "Medium", "High"],
            horizontal=True
        )
        
        # Newsletter subscription
        newsletter = st.checkbox("Subscribe to our newsletter for updates and tips")
        
        submitted = st.form_submit_button("Send Message", type="primary")
        
        if submitted:
            # Validation
            errors = []
            
            if not name.strip():
                errors.append("Name is required")
            
            if not email.strip():
                errors.append("Email is required")
            elif not validate_email(email):
                errors.append("Please enter a valid email address")
            
            if not message.strip():
                errors.append("Message is required")
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Process form submission
                form_data = {
                    'name': name,
                    'email': email,
                    'company': company,
                    'phone': phone,
                    'subject': subject,
                    'message': message,
                    'priority': priority,
                    'newsletter': newsletter,
                    'attachment': attachment
                }
                
                if process_contact_form(form_data):
                    st.success("✅ Thank you! Your message has been sent successfully. We'll get back to you within 24 hours.")
                    st.balloons()
                else:
                    st.error("❌ Sorry, there was an error sending your message. Please try again later.")

def process_contact_form(form_data):
    """Process the contact form submission"""
    
    # Here you would typically:
    # 1. Save to database
    # 2. Send email notification
    # 3. Add to CRM system
    # etc.
    
    try:
        # Simulate processing
        st.info("Processing your message...")
        
        # You could integrate with email services here
        # send_email(form_data)
        
        # For demo purposes, we'll just return True
        return True
        
    except Exception as e:
        st.error(f"Error processing form: {str(e)}")
        return False

def render_feedback_form():
    """Render a simple feedback form"""
    
    st.subheader("📝 Quick Feedback")
    
    with st.form("feedback_form"):
        rating = st.slider("Rate your experience", 1, 5, 3)
        feedback = st.text_area("Your feedback", placeholder="Tell us what you think...")
        
        if st.form_submit_button("Submit Feedback"):
            if feedback.strip():
                st.success("Thank you for your feedback!")
                # Process feedback here
            else:
                st.warning("Please enter some feedback before submitting")


