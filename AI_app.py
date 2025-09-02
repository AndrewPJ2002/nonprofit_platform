# This is our nonprofit platform - starting VERY simple
import streamlit as st
import pandas as pd  # This helps us work with CSV files and data tables
import plotly.express as px  # This makes pretty charts and graphs

# AI/ML imports for Hugging Face
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# This line makes the page look nice (like setting the title of a website)
st.set_page_config(
    page_title="Nonprofit Community Support Platform", 
    page_icon="🤝",
    layout="wide"  # Use full width of screen
)

# Add professional styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: #f0f2f6;
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
    }
    
    /* Make the tabs look better */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        border-radius: 10px 10px 0 0;
    }
    
    /* Success/info boxes */
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Professional header
st.markdown("""
<div class="main-header">
    <h1>🤝 Nonprofit Community Support & Insights Platform</h1>
    <p>Empowering nonprofits through AI-driven community support, streamlined data management, and outcome analysis</p>
</div>
""", unsafe_allow_html=True)

# Initialize AI model (cached for performance)
@st.cache_resource
def load_ai_model():
    """Load and cache the Hugging Face AI model"""
    if not HF_AVAILABLE:
        return None
    
    try:
        # Using a lightweight but capable model for conversation
        model_name = "microsoft/DialoGPT-medium"
        
        with st.spinner("🤖 Loading AI model... (This may take a moment on first run)"):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add padding token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
        return {"model": model, "tokenizer": tokenizer}
    except Exception as e:
        st.error(f"Error loading AI model: {e}")
        return None

# Load the model
ai_model = load_ai_model()

# This function is the "brain" of our chatbot - now with AI!
def get_chatbot_answer(question):
    # Convert to lowercase so "HOURS" and "hours" both work
    question_lower = question.lower()
    
    # First, check for specific nonprofit-related questions (rule-based)
    if "hours" in question_lower or "time" in question_lower:
        return "🕒 **Our Hours:** Monday-Friday 9 AM to 6 PM, Saturday 10 AM to 4 PM. We're closed on Sundays and major holidays. For emergency assistance, please call our 24/7 helpline at (555) 999-HELP."
    
    elif "volunteer" in question_lower:
        return "🤝 **Volunteer Opportunities:** We'd love your help! Current openings include tutoring, food service, event planning, and administrative support. Please visit our website to complete the volunteer application and background check."
    
    elif "donate" in question_lower:
        return "💝 **Donations:** Thank you for your generosity! You can donate online at our secure portal, by phone (555) 123-4567, or mail checks to 123 Community St. All donations are tax-deductible!"
    
    elif "programs" in question_lower or "services" in question_lower:
        return "📋 **Our Programs:** Youth Mentoring (ages 12-18), Job Training, Food Assistance, Housing Support, Educational Workshops, and Senior Services. Which program would you like to learn more about?"
    
    elif "contact" in question_lower or "phone" in question_lower or "email" in question_lower:
        return "📞 **Contact Information:** Phone: (555) 123-4567 | Email: info@nonprofitcommunity.org | Address: 123 Community St, City, State 12345 | Emergency Helpline: (555) 999-HELP (24/7)"
    
    elif "emergency" in question_lower or "crisis" in question_lower or "urgent" in question_lower:
        return "🆘 **Emergency Resources:** Crisis Hotline: (555) 999-HELP (24/7) | Emergency Food: Available during office hours | Mental Health: Free counseling referrals | If this is life-threatening, please call 911."
    
    # If no specific rule matches, try to use AI for more natural conversation
    elif ai_model and HF_AVAILABLE:
        try:
            ai_response = generate_ai_response(question)
            if ai_response and len(ai_response.strip()) > 0:
                return f"🤖 **AI Assistant:** {ai_response}\n\n💡 *For specific information about our programs, try asking about hours, volunteering, donations, or services.*"
        except Exception as e:
            # If AI fails, fall back to helpful message
            pass
    
    # Default fallback response
    return f"Thanks for your question! I can help you with information about our **hours, volunteer opportunities, donations, programs, contact information,** and **emergency resources**. What would you like to know more about?"

def generate_ai_response(question):
    """Generate AI response using Hugging Face model"""
    if not ai_model or not HF_AVAILABLE:
        return None
    
    try:
        model = ai_model["model"]
        tokenizer = ai_model["tokenizer"]
        
        # Prepare the input for the model
        nonprofit_context = "You are a helpful assistant for a nonprofit organization that provides community services including youth mentoring, job training, food assistance, housing support, and emergency services. "
        full_input = nonprofit_context + question
        
        # Encode the input
        inputs = tokenizer.encode(full_input + tokenizer.eos_token, return_tensors='pt')
        
        # Generate response (keep it short to avoid long processing times)
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_length=inputs.shape[1] + 50,  # Limit response length
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the new part (remove the input)
        response = response[len(full_input):].strip()
        
        # Clean up the response
        if len(response) > 200:  # Keep responses reasonable
            response = response[:200] + "..."
        
        return response
        
    except Exception as e:
        return None

# Professional sidebar
with st.sidebar:
    st.markdown("## 🎯 Platform Features")
    st.markdown("""
    **💬 AI Community Support**
    - Instant FAQ responses
    - 24/7 availability simulation
    - Issue categorization
    
    **📊 Data Management**  
    - CSV upload & validation
    - Automated data cleaning
    - Quality assessments
    
    **📈 Analytics Dashboard**
    - Interactive visualizations
    - Program outcome tracking
    - Performance metrics
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Session Statistics")
    
    # Show live stats
    if 'data' in st.session_state:
        data_status = f"{len(st.session_state['data'])} records loaded"
    else:
        data_status = "No data loaded"
    
    st.info(f"📄 Data: {data_status}")
    
    st.markdown("---")
    st.markdown("*Built with Python & Streamlit*")
    st.markdown("*Designed for social impact* 💝")

# Create 3 tabs (like folders in a filing cabinet)
tab1, tab2, tab3 = st.tabs(["💬 Chatbot", "📊 Data Upload", "📈 Charts"])

# Tab 1: Chatbot area
with tab1:
    st.markdown("### 💬 AI-Powered Community Support Assistant")
    
    # Show AI status
    if HF_AVAILABLE and ai_model:
        st.success("🤖 **AI Model Active:** Enhanced conversational responses available")
    else:
        st.info("🔧 **Rule-based Mode:** Install AI libraries for enhanced responses")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("*Get instant answers powered by artificial intelligence and nonprofit expertise.*")
        
        # This creates a text box where users can type
        user_question = st.text_input(
            "Ask me anything:", 
            placeholder="Try: 'What are your hours?' or 'Tell me about volunteering' or 'Hello, how are you?'"
        )
        
        # This creates a button
        col_ask, col_clear = st.columns([1, 1])
        
        with col_ask:
            if st.button("💬 Ask AI Assistant", type="primary"):
                # Only do something if they actually typed something
                if user_question:
                    with st.spinner("🤖 AI is thinking..."):
                        # This is our enhanced "brain" with AI
                        answer = get_chatbot_answer(user_question)
                    
                    # Show the answer in a nice box
                    st.markdown(f"""
                    <div style="background: #f0f8ff; border-left: 4px solid #667eea; padding: 1rem; margin: 1rem 0; border-radius: 5px; color: #333333;">
                        <strong>🤖 AI Assistant:</strong><br>
                        {answer}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Please type a question first!")
        
        with col_clear:
            if st.button("🗑️ Clear"):
                st.rerun()  # Refresh to clear the input
    
    with col2:
        st.markdown("### 💡 Try Asking About")
        st.markdown("""
        🕒 **Operating Hours & Contact**
        - "What time are you open?"
        - "How can I reach you?"
        
        🤝 **Volunteering & Getting Involved**
        - "How can I help?"
        - "What volunteer opportunities exist?"
        
        💰 **Donations & Support**
        - "How do I donate?"
        - "Ways to support your mission?"
        
        📋 **Programs & Services**
        - "What services do you offer?"
        - "Tell me about your programs?"
        
        💬 **Natural Conversation** *(AI-powered)*
        - "Hello, how are you?"
        - "What makes your nonprofit special?"
        - "I'm interested in helping my community"
        """)
        
        st.markdown("---")
        if HF_AVAILABLE and ai_model:
            st.success("🤖 **AI Enhanced:** Natural language understanding active!")
        else:
            st.info("💡 **Tip:** Install transformers library for AI responses!")

# Tab 2: Data upload area  
with tab2:
    st.markdown("### 📊 Data Upload & Management")
    st.markdown("*Upload your nonprofit participant data for instant analysis and reporting.*")
    
    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # This creates a file uploader that only accepts CSV files
        uploaded_file = st.file_uploader(
            "📁 Choose your CSV file", 
            type="csv",
            help="Upload participant data, program information, or volunteer records"
        )
        
        # Only do something if a file was actually uploaded
        if uploaded_file is not None:
            try:
                # Read the CSV file and turn it into a table (DataFrame)
                data = pd.read_csv(uploaded_file)
                
                # Basic validation
                if len(data) == 0:
                    st.error("❌ The file appears to be empty. Please check your CSV file.")

                
                # Show success message with basic info
                st.success(f"✅ **File processed successfully!**")
                
                # Display file info in metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("📊 Rows", f"{len(data):,}")
                with col_b:
                    st.metric("📋 Columns", len(data.columns))
                with col_c:
                    file_size = uploaded_file.size / 1024
                    st.metric("💾 Size", f"{file_size:.1f} KB")
                
                # Show the first few rows so they can see what's in the file
                st.markdown("#### 👀 Data Preview")
                st.dataframe(data.head(10), use_container_width=True)
                
                # Data quality check
                st.markdown("#### 🔍 Data Quality Assessment")
                missing_values = data.isnull().sum().sum()
                completeness = ((len(data) * len(data.columns)) - missing_values) / (len(data) * len(data.columns)) * 100
                
                qual_col1, qual_col2 = st.columns(2)
                with qual_col1:
                    if missing_values == 0:
                        st.success("✅ No missing values detected")
                    else:
                        st.warning(f"⚠️ {missing_values} missing values found")
                
                with qual_col2:
                    if completeness >= 95:
                        st.success(f"✅ {completeness:.1f}% data completeness")
                    elif completeness >= 80:
                        st.info(f"ℹ️ {completeness:.1f}% data completeness")
                    else:
                        st.warning(f"⚠️ {completeness:.1f}% data completeness")
                
                # Store the data so we can use it in Tab 3
                st.session_state['data'] = data
                
            except Exception as e:
                # If something goes wrong reading the file
                st.error(f"❌ **Error processing file:** {str(e)}")
                st.info("💡 **Tip:** Make sure your file is a valid CSV with proper formatting.")
        
    with col2:
        st.markdown("#### 📋 Expected Format")
        st.markdown("Your CSV should include columns like:")
        
        sample_data = {
            'Name': ['John Smith', 'Sarah Jones', 'Mike Davis'],
            'Age': [25, 34, 19],
            'Program': ['Job Training', 'Youth Mentoring', 'Food Assistance'],
            'Status': ['Active', 'Completed', 'Active']
        }
        st.table(sample_data)
        
        st.markdown("#### 💡 Tips for Best Results")
        st.markdown("""
        - Include headers in the first row
        - Use consistent date formats
        - Avoid special characters in column names
        - Keep data clean and consistent
        """)
        
        # Generate sample data button
        if st.button("🎲 Generate Sample Data", help="Create test data to explore the platform"):
            sample_participants = []
            programs = ['Youth Mentoring', 'Job Training', 'Food Assistance', 'Housing Support']
            statuses = ['Active', 'Completed', 'On Hold']
            
            for i in range(20):
                sample_participants.append({
                    'ID': i + 1,
                    'Name': f'Participant {i + 1}',
                    'Age': 15 + (i * 3) % 60,
                    'Program': programs[i % len(programs)],
                    'Status': statuses[i % len(statuses)],
                    'Join_Date': f'2024-0{(i % 9) + 1}-15'
                })
            
            st.session_state['data'] = pd.DataFrame(sample_participants)
            st.success("✅ Sample data generated! Check the Analytics tab.")
    
    # Show current status
    if 'data' not in st.session_state:
        st.info("📤 **Ready to upload!** Choose a CSV file above to get started with data analysis.")

# Tab 3: Charts area
with tab3:
    st.write("### 📈 Data Visualizations")
    
    # Check if we have data from Tab 2
    if 'data' in st.session_state:
        data = st.session_state['data']
        
        st.success(f"✅ Using data with {len(data)} participants")
        
        # Let the user pick which column to make a chart from
        st.write("**Choose what you want to visualize:**")
        column_to_chart = st.selectbox("Pick a column:", data.columns)
        
        # Handle Age column differently (it's a number, not categories)
        if column_to_chart == 'Age' and 'Age' in data.columns:
            # For age, create a histogram showing age ranges
            fig = px.histogram(
                data, 
                x='Age',
                title="Age Distribution of Participants",
                labels={'Age': 'Age', 'count': 'Number of People'},
                nbins=10  # This groups ages into 10 ranges
            )
            st.write("📊 **Age Histogram** (shows age ranges, not individual ages)")
            
        else:
            # For other columns, count how many times each value appears
            value_counts = data[column_to_chart].value_counts()
            
            # Create a simple bar chart
            fig = px.bar(
                x=value_counts.index,  # The categories (like Active, Completed)
                y=value_counts.values,  # How many of each
                title=f"Count of {column_to_chart}",
                labels={'x': column_to_chart, 'y': 'Number of People'}
            )
        
        # Show the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Show some basic stats
        st.write("**Quick Stats:**")
        col1, col2, col3 = st.columns(3)  # Create 3 columns for stats
        
        with col1:
            st.metric("Total Records", len(data))
        
        with col2:
            if 'Status' in data.columns:
                completed = len(data[data['Status'] == 'Completed'])
                st.metric("Completed", completed)
            else:
                st.metric("Columns", len(data.columns))
        
        with col3:
            if 'Age' in data.columns:
                avg_age = round(data['Age'].mean(), 1)
                st.metric("Average Age", f"{avg_age} years")
            else:
                st.metric("Data Points", len(data) * len(data.columns))
    
    else:
        # If no data uploaded yet
        st.info("📤 Please upload data in the 'Data Upload' tab first!")
        
        # Show a sample chart so they can see what it will look like
        st.write("**Here's what your charts will look like:**")
        
        # Create sample data for demonstration
        sample_data = pd.DataFrame({
            'Program': ['Job Training', 'Youth Mentoring', 'Food Assistance', 'Housing Support'],
            'Participants': [25, 18, 32, 15]
        })
        
        # Make a sample chart
        sample_fig = px.bar(
            sample_data, 
            x='Program', 
            y='Participants',
            title="Sample: Participants by Program"
        )
        st.plotly_chart(sample_fig, use_container_width=True)