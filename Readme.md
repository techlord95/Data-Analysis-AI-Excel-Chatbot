
---

````markdown
# 🏭 Chemical Factory Data Analyzer

A smart and interactive data analysis Streamlit app that allows users to explore and analyze chemical factory inventory data using natural language queries. Powered by LLMs (Gemini via OpenAI client), this tool automatically generates and executes Python code to extract insights and visualize trends from uploaded CSV files.

---

## 🔍 Features

- 📁 **Upload CSV Data**: Easily upload chemical factory inventory datasets.
- 🧠 **LLM-Powered Query Interpretation**: Ask questions in plain English, and the app will understand, generate code, and execute it.
- 📊 **Auto-Generated Visualizations**: Supports both matplotlib and Plotly for dynamic and interactive plots.
- 📦 **Smart Response Agent**: Summarizes results intelligently, alerts on query mistakes, and even responds sarcastically to wrong spellings.
- 🛠️ **Custom UI Styling**: Clean, professional, and modern UI using advanced CSS with Streamlit.
- 📌 **Session History**: Keeps a chat-style history of all your queries and results to reduce redundant LLM calls.

---

## 🏗️ Tech Stack

- [Streamlit](https://streamlit.io/) – For building interactive web UI
- [Pandas](https://pandas.pydata.org/) – For data manipulation
- [Matplotlib, Seaborn, Plotly](https://matplotlib.org/) – For plotting and visualization
- [OpenAI Python SDK](https://pypi.org/project/openai/) – For Gemini LLM API calls
- `.env` for secure API key handling

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/chemical-factory-analyzer.git
cd chemical-factory-analyzer
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the root directory:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Run the Application

```bash
streamlit run agents.py
```


---

##  Sample Queries

You can try asking:

* "What are the total sales in 2023?"
* "Show me quantity sold by region"


---

## LLM Behavior and Rules

* Code generated is always trusted; incorrect or misspelled queries will be fixed and notified (often sarcastically).
* Responses are tailored using Indian numeric format and currency symbol (₹).
* Handles plot generation seamlessly.
* If query is out-of-scope, responds gracefully.

---

## License

MIT License

---

## Acknowledgments

* Gemini (by Google) for LLMs
* Streamlit for enabling easy Python app deployment
* OpenAI Python SDK

---

## 📸 Screenshot

![alt text](image.png)



![alt text](image-1.png)


![alt text](image-2.png)

---

## 🔗 Author

**techlord95**

Feel free to fork, contribute, or open an issue if you find bugs or want enhancements.

