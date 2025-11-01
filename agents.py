import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import io
import sys
import base64
from io import StringIO
import contextlib
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables(good to use env)
load_dotenv()


st.set_page_config(
    page_title="🏭 Chemical Factory Data Analyzer",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# css for styles in streamlit 
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border: 2px solid #1f77b4;
    }
    
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .query-box {
        background-color: #000000;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .code-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
    }
    
    .result-container {
        background-color: #000000;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #0d47a1;
        transform: translateY(-2px);
    }
    
    .sidebar .stSelectbox > div > div {
        background-color: #f0f8ff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'client' not in st.session_state:
    st.session_state.client = None

def setup_openai_client():
    """Initialize OpenAI client with Gemini API"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        st.session_state.client = client
        return True
    return False

def extract_code_block(text: str) -> str:
    """Extract Python code from markdown code blocks"""
    start = text.find("```python")
    if start == -1:
        return ""
    start += len("```python")
    end = text.find("```", start)
    if end == -1:
        return ""
    return text[start:end].strip()

def CodeWritingTool(query: str, df: pd.DataFrame) -> str:
    """Generate Python code based on query and dataframe"""
    if not st.session_state.client:
        return "Error: API client not initialized"
    
    data_understanding_configuration_messages = [
        {
            "role": "system",
            "content": f"""You are a data analyst assistant. Follow these steps:
            
    Get familiar with data using:
    
    Available categories: {df['sku_category'].unique().tolist()} 
    Available regions: {df['dist_region'].unique().tolist()} 
    Available sku_name: {df['sku_name'].unique().tolist()} 
    Available dist_city: {df['dist_city'].unique().tolist()}
    Available dist_state: {df['dist_state'].unique().tolist()} - states in which the industry is setup 
    Available size of industry: {df['dist_size_category'].unique().tolist()} - these are the size of the industry 

    1. Analyze the query for filters: year, sku_category, region
    2. Generate pandas code wrapped in ```python```
    3. Here amount column is calculated as quantity * sku_mrp so amount is in ₹
    4. When asked about total sales then give it in terms of amount when asked about quantity then return quantity 
    5. When asked about specific category like Ethanol 1L can or 1 L Bottle refer the Available categories 
       the user might make mistakes like ethanol 1l ca or 1LL can correct them by matching the Available Categories in sku_category and write code accordingly 

    Example response:
    ```python
    result = df[(df['order_year'] == 2020) &
       (df['sku_category'] == 'Ethanol') &
       (df['dist_region'] == 'West')]['amount'].sum()
    ```
    
    6. If the query is asking for a plot generation or visualization of the data then write code in matplotlib or plotly after understanding the query carefully 
    7. Always assign the final result to a variable named 'result'
    8. For plots, use plt.figure(figsize=(10, 6)) and plt.tight_layout()
    9. For plotting queries, set result = "Plot generated successfully" after creating the plot
    10. Don't use plt.show() in the code as Streamlit handles plot display
    follow the example response carefullly the output shall be in that exact format 
    """
        },
        {
            "role": "user",
            "content": query
        }
    ]
    
    try:
        response = st.session_state.client.chat.completions.create(
            model="gemini-2.5-pro",
            messages=data_understanding_configuration_messages,
            temperature=1.0,
            max_tokens=1024
        )
        
        llm_code_response = response.choices[0].message.content
        actual_code = extract_code_block(llm_code_response)
        
        return actual_code
    except Exception as e:
        return f"Error generating code: {str(e)}"

def CodeExecutionTool(actual_code: str, df: pd.DataFrame):
    """Execute the generated code safely"""
    # Create a safe environment for code execution
    env = {
        "pd": pd, 
        "df": df, 
        "plt": plt, 
        "sns": sns,
        "px": px,
        "go": go,
        "st": st
    }
    
    
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    try:
        
        clean_code = actual_code.replace("plt.show()", "").strip()
        
        
        exec(clean_code, {}, env)
        
     
        output = captured_output.getvalue()
        
    
        sys.stdout = old_stdout
        
        
        is_plot = 'plt.' in actual_code or 'plot' in actual_code.lower()
        
        
        if "result" in env:
            return env["result"], output, env.get("plt", None)
        elif is_plot:
            return "Plot generated successfully", output, env.get("plt", None)
        else:
            try:
                result = eval(clean_code, {}, env)
                return result, output, env.get("plt", None)
            except:
                return output if output else "Code executed successfully", output, env.get("plt", None)
                
    except Exception as exc:
        sys.stdout = old_stdout
        return f"Error executing code: {exc}", "", None

def CodeandQueryResponsiveAgent(query: str, actual_code: str, df: pd.DataFrame, code_output) -> str:
    """Generate natural language response based on query and code output"""
    if not st.session_state.client:
        return "Error: API client not initialized"
    
    prompt = f'''This is the {query} and this is code {actual_code} used to answer the query. The code output is {code_output}. Make a response in one liner accordingly'''

    final_messages = [
        {
            "role": "system", 
            "content": '''FOLLOW THIS INSTRUCTION VERY CAREFULLY:
    1) The quantity should be represented as Units 
    2) Amount should be represented in Indian Rupee Symbol ₹
    3) The output should be formatted in Indian Numeric System 
    4) Also follow the code if the code has quantity then it is in Units if the code has amount then it is in ₹
    THE code is always right the user might make a mistake lets say propanol 80 l bottle but the code has 1L bottle the make a sentence according to the code and alert the user that they gave wrong query
    CHEK EVERY QUERY WITH  RESPECT TO CODE CAREFULLY ALWAYS GIVE ALERTS TO USER WHRN REQUIRED especially when there is a spelling mistake but be sarcastic while pointing out mistakes
    5) IF THE QUERY IS NOT RELATED TO THE DATA THEN SAY "I Am Here To Help You But I Cannot Respond To That"
    
    You are a smart Data Analysis Expert who does not talk a lot and gives concise and correct thoughtful response. You will receive a string query and numeric data which is the answer to the string query, then respond accordingly in one line framing a sentence regarding query.
    '''
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    try:
        final_response = st.session_state.client.chat.completions.create(
            model="gemini-2.5-pro",
            messages=final_messages,
            temperature=1.0,
            max_tokens=512
        )
        
        return final_response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

def load_data():
    """Load and preprocess the data"""
    uploaded_file = st.file_uploader(
        "Upload your CSV file", 
        type=['csv'],
        help="Upload the chemical factory inventory data file"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Preprocessing
            df['order_date'] = pd.to_datetime(df["order_date"])
            df['order_year'] = df['order_date'].dt.year
            df['order_month'] = df['order_date'].dt.month_name()
            
            numeric_cols = ['amount', 'quantity', 'sku_mrp']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            st.session_state.df = df
            return df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    return st.session_state.df

def display_data_overview(df):
    """Display data overview and statistics"""
    st.subheader("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        st.metric("Categories", len(df['sku_category'].unique()))
    
    with col3:
        st.metric("Regions", len(df['dist_region'].unique()))
    
    with col4:
        st.metric("Years", len(df['order_year'].unique()))
    
    
    with st.expander("View Sample Data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
    
    
    with st.expander("Data Summary", expanded=False):
        st.write("**Available Categories:**", ", ".join(df['sku_category'].unique()))
        st.write("**Available Regions:**", ", ".join(df['dist_region'].unique()))
        st.write("**Available Years:**", ", ".join(map(str, sorted(df['order_year'].unique()))))

def main():
    # Header
    st.markdown('<div class="main-header">🏭 Chemical Factory Data Analyzer</div>', unsafe_allow_html=True)
    

    if not setup_openai_client():
        st.error("⚠️ Please set your GEMINI_API_KEY in the environment variables or .env file")
        st.stop()
    
  
    with st.sidebar:
        st.header("🔧 Configuration")
        
        
        st.subheader("📂 Data Loading")
        df = load_data()
        
        if df is not None:
            st.success(f"✅ Data loaded successfully! ({len(df)} records)")
            
            # Quick filters
            st.subheader("🎯 Quick Filters")
            
            selected_year = st.selectbox(
                "Select Year (Optional)",
                options=["All"] + sorted(df['order_year'].unique().tolist()),
                index=0
            )
            
            selected_category = st.selectbox(
                "Select Category (Optional)",
                options=["All"] + df['sku_category'].unique().tolist(),
                index=0
            )
            
            selected_region = st.selectbox(
                "Select Region (Optional)",
                options=["All"] + df['dist_region'].unique().tolist(),
                index=0
            )
        
    
        st.subheader("💡 Sample Queries")
        sample_queries = [
            "What are the total sales in 2023?",
            "Show me quantity sold by region",
            "Which category has highest sales?",
            "Plot sales trend by year",
            "Show top 5 products by quantity",
            "Compare sales across regions"
        ]
        
        for query in sample_queries:
            if st.button(query, key=f"sample_{query}", use_container_width=True):
                st.session_state.current_query = query
    
  
    if df is not None:
        
        display_data_overview(df)
        
       
        st.subheader("💬 Ask Your Question")
        
    
        query_input = st.text_area(
            "Enter your question about the data:",
            height=100,
            placeholder="e.g., What are the total sales in 2023? or Show me a plot of sales by region",
            value=getattr(st.session_state, 'current_query', '')
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
     
        if analyze_button and query_input.strip():
            with st.spinner("🔄 Analyzing your query..."):
            
                generated_code = CodeWritingTool(query_input, df)
                
                if generated_code and not generated_code.startswith("Error"):
                
                    result, output, plot_obj = CodeExecutionTool(generated_code, df)
                    
                
                    response = CodeandQueryResponsiveAgent(query_input, generated_code, df, result)
                    
                
                    st.session_state.chat_history.append({
                        'query': query_input,
                        'code': generated_code,
                        'result': result,
                        'response': response,
                        'output': output,
                        'has_plot': 'plt.' in generated_code or 'plot' in generated_code.lower() or 'fig' in generated_code
                    })
            
            
            if hasattr(st.session_state, 'current_query'):
                delattr(st.session_state, 'current_query')
            st.rerun()
        
        # chat history (this step is crucial to avoid extra billing of llms as the user may ask repetetive queries)
        if st.session_state.chat_history:
            st.subheader("📈 Analysis Results")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.container():
                    # Query
                    st.markdown(f'<div class="query-box"><strong>🤔 Question:</strong> {chat["query"]}</div>', unsafe_allow_html=True)
                    
                    # Response
                    st.markdown(f'<div class="result-container"><strong>💡 Answer:</strong> {chat["response"]}</div>', unsafe_allow_html=True)
                    
                    
                    with st.expander("🔍 View Details", expanded=False):
                        # Generated code
                        st.markdown("**Generated Code:**")
                        st.code(chat["code"], language="python")
                        
                        # Raw result
                        if chat["result"] is not None and str(chat["result"]).strip():
                            st.markdown("**Raw Result:**")
                            st.write(chat["result"])
                        
                        # Output
                        if chat["output"]:
                            st.markdown("**Console Output:**")
                            st.text(chat["output"])
                    
                    # Handling plots 
                    if chat["has_plot"] or 'plt.' in chat["code"] or 'plot' in chat["code"].lower():
                        try:
                           
                            env = {"pd": pd, "df": df, "plt": plt, "sns": sns, "px": px, "go": go}
                            
                           
                            plot_code = chat["code"].replace("plt.show()", "").strip()
                            exec(plot_code, {}, env)
                            
                        
                            if plt.get_fignums():  
                                st.pyplot(plt.gcf())
                                plt.clf()  # Clear the figure to  avoid extra memory usage, not really necessary 
                            elif 'fig' in env:
                                st.plotly_chart(env['fig'], use_container_width=True)
                        except Exception as e:
                            st.error(f"Error displaying plot: {str(e)}")
                    
                    st.markdown("---")
    
    else:
        st.info("Please upload a CSV file to get started with data analysis!")
        
        # Show example of expected data format
        # st.subheader("📋 Expected Data Format")
        # st.write("Your CSV file should contain columns like:")
        # example_df = pd.DataFrame({
        #     'order_date': ['2023-01-01', '2023-01-02'],
        #     'sku_category': ['Ethanol 1L Can', 'Methanol 500ml'],
        #     'dist_region': ['North', 'South'],
        #     'quantity': [100, 200],
        #     'sku_mrp': [50.0, 30.0],
        #     'amount': [5000.0, 6000.0],
        #     'dist_city': ['Delhi', 'Chennai'],
        #     'dist_state': ['Delhi', 'Tamil Nadu'],
        #     'dist_size_category': ['Large', 'Medium']
        # })
        # st.dataframe(example_df, use_container_width=True)
        
        # Can use the above code for sample presentation

if __name__ == "__main__":
    main()