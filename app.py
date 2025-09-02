# This is our nonprofit platform - starting VERY simple
import streamlit as st
import pandas as pd
import plotly.express as px


# This line makes the page look nice (like setting the title of a website)
st.set_page_config(page_title="Nonprofit Platform", page_icon="🤝")

# This is like the header/title at the top of our page
st.title("Nonprofit Community Support Platform")
st.write("This platform helps nonprofits manage their community outreach")

# basic brain function
def get_chatbot_answer(question):
    question = question.lower()
    
    # Check what the person is asking about using simple keyword matching
    if "hours" in question or "time" in question:
        return "We're open Monday-Friday 9 AM to 6 PM, Saturday 10 AM to 4 PM"
    
    elif "volunteer" in question:
        return "We'd love your help! We need volunteers for tutoring, food service, and events. Visit our website to apply!"
    
    elif "donate" in question:
        return "Thank you! You can donate online, by phone (555-123-4567), or visit our office at 123 Community St."
    
    elif "programs" in question or "services" in question:
        return "We offer youth mentoring, job training, food assistance, and housing support. Which interests you?"
    
    else:
        return f"Thanks for asking about '{question}'. Try asking about our hours, volunteer opportunities, donations, or programs!"
 

# Create 3 tabs (like folders in a filing cabinet)
tab1, tab2, tab3 = st.tabs(["💬 Chatbot", "📊 Data Upload", "📈 Charts"])

# Tab 1: Chatbot area ------------------------------------------------------------------------------
with tab1:
    st.write("### 💬 Ask Our AI Assistant")
    
    # creates a text box where users can type
    user_question = st.text_input("How can I help you?")
    
    # create a button 
    if st.button("Ask"):
        if user_question:
            # simple brain
            answer = get_chatbot_answer(user_question)
            
            # show answer in a box
            st.success(f"**Answer:** {answer}")
        else:
            st.warning("Please type a question first!")

# Tab 2: Data upload area  --------------------------------------------------------------------------
with tab2:
    st.write("### 📊 Upload Your Data")
    
    # create a file uploader for csv files
    uploaded_file = st.file_uploader("Choose CSV file", type="csv" )
    
    if uploaded_file is not None:
        
        try:
            # read csv and turn it into table
            data = pd.read_csv(uploaded_file)
            
            # success message
            st.success(f"✅ File uploaded! It has {len(data)} rows and {len(data.columns)} columns")
            
            # show the first few columns
            st.write("**Here's a preview of your data:**")
            st.dataframe(data.head())
            
            # Store the data so we can use it in Tab 3
            st.session_state['data'] = data
        except Exception as e:
            # If something goes wrong reading the file
            st.error(f"Oops! There was a problem reading your file: {e}")
    
    else:
        # If no file uploaded, show them what to expect
        st.info("👆 Upload a CSV file to see your data here")
        st.write("**Expected format example:**")
        
        
         # Show a sample of what their data might look like
        sample_data = {
            'Name': ['John Smith', 'Sarah Jones', 'Mike Davis'],
            'Age': [25, 34, 19],
            'Program': ['Job Training', 'Youth Mentoring', 'Food Assistance'],
            'Status': ['Active', 'Completed', 'Active']
        }
        st.table(sample_data)

# Tab 3: Charts area --------------------------------------------------------------------------------
with tab3:
    st.write("### 📈 Data Visualization") 
    
    # Check if we have data from Tab 2
    if 'data' in st.session_state:
        data = st.session_state['data']
        
        # user can pick which column to make a chart from
        st.write("**Choose which column to visualize:**")
        col_to_chart = st.selectbox("Pick a column:", data.columns)
        
        # count how many times each value appeats
        value_counts = data[col_to_chart].value_counts()
        
        fig = px.bar(
            x = value_counts.index,
            y = value_counts.values,
            title = f"Count of {col_to_chart}", 
            labels = {'x': col_to_chart, 'y':'Number of People'}
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