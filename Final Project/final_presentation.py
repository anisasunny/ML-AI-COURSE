import os
import csv
from io import BytesIO
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import google.generativeai as genai
from pathlib import Path
import time

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not found in environment variables.")
    st.stop()

# Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

# Load CSV or Excel
@st.cache_data
def load_uploaded_data(file):
    filename = file.name
    try:
        if filename.endswith(".csv"):
            sample = file.read(1024).decode('utf-8')
            file.seek(0)
            delimiter = csv.Sniffer().sniff(sample).delimiter
            return pd.read_csv(file, delimiter=delimiter)
        elif filename.endswith((".xlsx", ".xls")):
            return pd.read_excel(file)
        else:
            return None
    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
        return None

# Summarizer for Gemini
def create_data_summary(df):
    summary = f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.\n\n"
    summary += "📊 Column Overview:\n"
    for col in df.columns:
        dtype = df[col].dtype
        unique = df[col].dropna().unique()
        sample_vals = ', '.join(map(str, unique[:3]))
        summary += f"- {col} (type: {dtype}), sample: {sample_vals}\n"
    try:
        desc = df.describe(include='all').transpose()
        summary += f"\n📈 Basic Statistics:\n{desc.head(3).to_string()}"
    except Exception as e:
        summary += f"\n⚠️ Could not generate basic stats: {e}"
    return summary

# AI Agent
@st.cache_data(show_spinner=False)
def ai_agent(user_query, df):
    data_context = create_data_summary(df)
    prompt = f"""
You are a helpful Python data analyst AI.

# Dataset Summary:
{data_context}

User Question:
"{user_query}"

👉 Respond with the data output only.
Return only the pandas output from the analysis or query requested.
Do not include any code or explanation.
If the request cannot be completed, say 'Unable to retrieve result from dataset.'
"""
    response = model.generate_content(prompt)
    return response.text

# Detect plot requests
def is_plot_request(query):
    return any(word in query.lower() for word in ["plot", "show", "visualize", "chart"])

def get_plot_type(query):
    q = query.lower()
    if "pie" in q:
        return "pie"
    elif "box" in q:
        return "box"
    elif "scatter" in q:
        return "scatter"
    elif "line" in q:
        return "line"
    elif "bar" in q or "distribution" in q:
        return "bar"
    elif "relationship" in q or "between" in q:
        return "relationship"
    return "unknown"

def extract_column_names(query, columns, max_count=2):
    found = []
    for col in columns:
        if col.lower() in query.lower():
            found.append(col)
        if len(found) == max_count:
            break
    return found if found else None

# Streamlit app
st.set_page_config(page_title="📈 AI Dataset Analyzer", layout="wide")
st.title("📊 AI-Powered Dataset Analyzer")

# Sidebar
page = st.sidebar.radio("📂 Navigate", ["🧪 Ask Questions", "📚 Chat History"])

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "final_query" not in st.session_state:
    st.session_state.final_query = ""

uploaded_file = st.file_uploader("📁 Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
if uploaded_file:
    df = load_uploaded_data(uploaded_file)
    if df is None:
        st.stop()

    df_cleaned = df.dropna()

    if page == "🧪 Ask Questions":
        st.subheader("🔍 Cleaned Data Preview")
        st.dataframe(df_cleaned.head())

        st.subheader("📈 Descriptive Statistics")
        desc_df = df_cleaned.describe(include='all').transpose()
        st.dataframe(desc_df)

        st.subheader("💬 Ask a Question About Your Dataset")
        dynamic_suggestions = []
        for col in df_cleaned.columns:
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                dynamic_suggestions.append(f"What is the average value of '{col}'?")
                dynamic_suggestions.append(f"Plot distribution of '{col}'")
            else:
                unique_vals = df_cleaned[col].dropna().unique()
                if len(unique_vals) > 0:
                    dynamic_suggestions.append(f"How many entries have '{col}' = {unique_vals[0]}")
                    dynamic_suggestions.append(f"Plot pie chart of '{col}'")

        selected_question = st.selectbox("🧠 Suggested questions:", ["Select..."] + dynamic_suggestions)
        custom_input = st.text_input("✍️ Or ask your own question:")

        final_query = selected_question if selected_question != "Select..." else custom_input
        st.session_state.final_query = final_query

        if st.button("🔍 Ask Now") and st.session_state.final_query:
            with st.spinner("🔄 Processing..."):
                if is_plot_request(final_query):
                    plot_type = get_plot_type(final_query)
                    columns = extract_column_names(final_query, df_cleaned.columns, max_count=2)

                    if not columns:
                        st.warning("❗ Could not detect column(s) for plotting.")
                    else:
                        fig, ax = plt.subplots()
                        try:
                            if plot_type == "pie":
                                df_cleaned[columns[0]].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
                                ax.set_ylabel("")
                                ax.set_title(f"Pie Chart of {columns[0]}")
                            elif plot_type == "bar":
                                col = columns[0]
                                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                                    ax.hist(df_cleaned[col], bins=20, color='skyblue', edgecolor='black')
                                else:
                                    df_cleaned[col].value_counts().plot(kind='bar', ax=ax, color='orange')
                                ax.set_title(f"Bar Chart of {col}")
                                ax.set_ylabel("Frequency")
                            elif plot_type == "box" and len(columns) == 2:
                                df_cleaned.boxplot(column=columns[0], by=columns[1], ax=ax)
                                ax.set_title(f"Boxplot of {columns[0]} by {columns[1]}")
                            elif plot_type == "scatter" and len(columns) == 2:
                                ax.scatter(df_cleaned[columns[0]], df_cleaned[columns[1]], alpha=0.6)
                                ax.set_title(f"Scatter: {columns[0]} vs {columns[1]}")
                                ax.set_xlabel(columns[0])
                                ax.set_ylabel(columns[1])
                            elif plot_type == "line":
                                df_cleaned[columns[0]].plot(kind='line', ax=ax)
                                ax.set_title(f"Line Chart of {columns[0]}")
                            elif plot_type == "relationship" and len(columns) == 2:
                                x, y = columns
                                if pd.api.types.is_numeric_dtype(df_cleaned[x]) and not pd.api.types.is_numeric_dtype(df_cleaned[y]):
                                    df_cleaned.boxplot(column=x, by=y, ax=ax)
                                elif pd.api.types.is_numeric_dtype(df_cleaned[y]) and not pd.api.types.is_numeric_dtype(df_cleaned[x]):
                                    df_cleaned.boxplot(column=y, by=x, ax=ax)
                                else:
                                    pd.crosstab(df_cleaned[x], df_cleaned[y]).plot(kind="bar", stacked=True, ax=ax)
                                ax.set_title(f"Relationship between {x} and {y}")
                            st.pyplot(fig)
                            st.session_state.history.append((final_query, fig))
                        except Exception as e:
                            st.error(f"❌ Error generating plot: {e}")
                else:
                    result = ai_agent(final_query, df_cleaned)
                    if isinstance(result, pd.DataFrame):
                        st.success("📋 Data Output:")
                        st.dataframe(result)
                        st.download_button("📤 Download Result", result.to_csv(index=False), "result.csv", mime="text/csv")
                        st.session_state.history.append((final_query, result))
                    else:
                        st.warning(result)

    elif page == "📚 Chat History":
        st.subheader("📚 Previous Queries")
        if st.session_state.history:
            for i, (q, a) in enumerate(reversed(st.session_state.history[-10:]), 1):
                st.markdown(f"**Q{i}:** {q}")
                if isinstance(a, pd.DataFrame):
                    st.dataframe(a)
                elif isinstance(a, plt.Figure):
                    st.pyplot(a)
                else:
                    st.write(a)
        else:
            st.info("No history yet.")
else:
    st.info("📁 Upload a CSV or Excel file to begin.")