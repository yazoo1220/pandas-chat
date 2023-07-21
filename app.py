from langchain.agents import initialize_agent, AgentType
from langchain.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
import streamlit as st
import pandas as pd
import os

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
    "json": pd.read_json,
    "html": pd.read_html,
    "sql": pd.read_sql,
    "feather": pd.read_feather,
    "parquet": pd.read_parquet,
    "dta": pd.read_stata,
    "sas7bdat": pd.read_sas,
    "h5": pd.read_hdf,
    "hdf5": pd.read_hdf,
    "pkl": pd.read_pickle,
    "pickle": pd.read_pickle,
    "gbq": pd.read_gbq,
    "orc": pd.read_orc,
    "xpt": pd.read_sas,
    "sav": pd.read_spss,
    "gz": pd.read_csv,
    "zip": pd.read_csv,
    "bz2": pd.read_csv,
    "xz": pd.read_csv,
    "txt": pd.read_csv,
    "xml": pd.read_xml,
}


def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False


@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None

def load_sample_data():
    # Define the path to the sample data file
    sample_data_path = "./sample_data.csv"
    # Load the sample data using the appropriate file format function (assuming it's CSV)
    sample_data = pd.read_csv(sample_data_path)
    return sample_data

st.set_page_config(page_title="Balencer Pandas", page_icon="🐼")
st.title("Balencer 🐼 Pandas")

uploaded_file = st.file_uploader(
    "ファイルをアップロード",
    type=list(file_formats.keys()),
    help="Various File formats are Support",
    on_change=clear_submit,
)

# Use Streamlit to create a button to load the sample data
if st.button("サンプルデータを読む"):
    df = load_sample_data()

    if sample_data is not None:
        st.success("サンプルが正常に読み込まれました!")
        st.sidebar.dataframe(df)
    else:
        st.error("サンプルの読み込みに失敗しました。")

if uploaded_file:
    df = load_data(uploaded_file)
    st.sidebar.dataframe(df)

openai_api_key = os.environ['OPENAI_API_KEY']
if "messages" not in st.session_state or st.sidebar.button("チャット履歴を消す"):
    st.session_state["messages"] = [{"role": "assistant", "content": "データをアップして質問をしてください。"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="データの概要を説明してください"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    llm = ChatOpenAI(
        temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=openai_api_key, streaming=True
    )

    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
