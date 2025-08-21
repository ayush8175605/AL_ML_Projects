import pandas as pd
import streamlit as st
from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from io import BytesIO
import os
import matplotlib
matplotlib.use('Agg')
from sqlalchemy import create_engine
from langchain_experimental.agents import create_pandas_dataframe_agent
import base64

# custom imports
from prompts.table_info import table_info_non_contextual
from prompts.prompts_non_contextual import prefix, suffix
from config import llm, engine_string_public
from utils import get_current_quarter_and_year,get_filters,get_sample_data,get_examples


os.environ['AZURE_OPENAI_API_KEY'] = 'OPENAI_KEY'
os.environ['AZURE_OPENAI_API_TYPE'] = 'azure'
os.environ['OPENAI_API_VERSION'] = 'OPENAI_VERSION'
os.environ['AZURE_OPENAI_ENDPOINT'] = 'OPENAI_ENDPOINT'

st.title("DATA-BOT")

if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []
if 'agent' not in st.session_state:
    st.session_state['agent'] = None
if 'agent_response' not in st.session_state:
    st.session_state['agent_response'] = []
if 'generated_plot' not in st.session_state:
    st.session_state['generated_plot'] = []
if 'modified_csv' not in st.session_state:
    st.session_state['modified_csv'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'prompt' not in st.session_state:
    st.session_state['prompt'] = []

if "filters" not in st.session_state:
    st.session_state['filters']=[]

# container for the chat history
response_container = st.container()
# container for the user's text input
container = st.container()


# Function to Add Conversation History
def add_to_chat_history(user_query, bot_response, plot_data):
    st.session_state['conversation_history'].append((user_query, bot_response, plot_data))



def sql_agent():
        # Database connection
    # db = SQLDatabase.from_uri("sqlite:///your_database.db")
    sql_db=SQLDatabase.from_uri(
    engine_string_public,
    include_tables=['sales_data', 'sales_data_plant'],
    custom_table_info=table_info_non_contextual(
                        sample_data1=get_sample_data(
                                    table_name='sales_data',sample_size=4),
                        sample_data2=get_sample_data(
                            table_name='sales_data_plant',sample_size=4
                        )),view_support=True)
    toolkit = SQLDatabaseToolkit(db=sql_db, llm=llm)

    # Initialize Agents
    # sql_agent = create_sql_agent(llm, db, verbose=True)
    # Initializing SQL Agent
    sql_agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    top_k=50,
    handle_parsing_errors=True,
    agent_executor_kwargs={"return_intermediate_steps": True, "use_query_checker": True}
    )

    return sql_agent

def python_agent():
    """Create a pandas agent backed by the public SQL engine.

    The previous implementation expected separate database credentials
    (``db_host``, ``db_name`` etc.) which were never defined in this
    module.  This resulted in a ``NameError`` as soon as the function was
    called.  Instead we reuse ``engine_string_public`` – already imported
    from ``config`` – to establish the connection.  Using SQLAlchemy also
    keeps the connection logic in line with how the SQL agent is created
    above.
    """

    # Create a SQLAlchemy engine from the shared connection string
    engine = create_engine(engine_string_public)

    # Load the required tables into pandas DataFrames
    df_config = pd.read_sql("SELECT * from sales_data", engine)
    df_plant = pd.read_sql("SELECT * from sales_data_plant", engine)

    # csv_agent = create_csv_agent(llm, "your_data.csv", verbose=True)
    python_agent = create_pandas_dataframe_agent(llm, 
                                                 [df_config, df_plant], 
                                                verbose=True, 
                                                agent_type=AgentType.OPENAI_FUNCTIONS,
                                                handle_parsing_errors=True,
                                                allow_dangerous_code=True,
                                                return_intermediate_steps=True)
    
    return python_agent

def decide_next_action(user_question, context):
    """Uses LLM to decide the next action based on the question and intermediate results."""
    decision_prompt = f"""
    User Question: {user_question}
    Current Context: {context}

    Based on the data so far, which agent should be called next?
    - SQL for data extraction (KPIs, metrics, etc.)
    - Pandas for graph generation and visualisations only!
    - 'DONE' if no further processing is required.

    Respond with only one choice: SQL, Pandas, or DONE.
    """
    return llm.predict(decision_prompt).strip()

def execute_dynamic_agents(question, sql_agent, python_agent, scenario_id):
    """Dynamically routes queries between agents in a loop."""
    context = {}
    active_agent = decide_next_action(question, context)
    mapped_json = get_filters(question.lower(), scenario_id=scenario_id)  # change scenario id to dynamic

    plot_flag =0
    while active_agent != "DONE":
        print(f"[INFO] Using {active_agent} agent...")

        if active_agent == "SQL":

            result = sql_agent.invoke(
                        prefix.format(question=question, current_quarter=str(get_current_quarter_and_year()[0]),
                        current_year=str(get_current_quarter_and_year()[1]), scenario_id=str(1),
                        filters=str(mapped_json)) + "\n" + get_examples(
                        user_input=question))
            
        elif active_agent == "Pandas":
            result = python_agent.invoke(f"Based on {context}, generate the required graph or visualisation. Create aesthetical graphs with clear labels. Also Provide the ByteIO object of the graph generated")
            #Also Provide the ByteIO object of the graph generated.
            plot_flag = 1 # chart created

        else:
            result = "Invalid agent selection."

        # Store intermediate results
        context[active_agent] = result

        # Decide the next step
        active_agent = decide_next_action(question, context)

    if plot_flag == 1:
        # st.session_state['generated_plot'].append(output['Pandas']['output'])
        byte_object = context['Pandas']['intermediate_steps'][-1][1]
        # Convert BytesIO to Base64
        output = base64.b64encode(byte_object.getvalue()).decode("utf-8")

    else:
        output =  context['SQL']['output']


    # return f"Final Response: {context.get('Reasoning', 'See intermediate results')}"
    return output, plot_flag


# Display Answer
# st.chat_message("assistant").write(f"**Bot:** {analysis_result}")

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Type your query...", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        
        output, plot_flag = execute_dynamic_agents(user_input, sql_agent= sql_agent(), python_agent= python_agent(), scenario_id = 21)
        # print(output)
        st.session_state['past'].append(user_input)
        st.session_state['agent_response'].append(output)
        if plot_flag == 1:
            # st.session_state['generated_plot'].append(output['Pandas']['output'])
            # byte_object = output['Pandas']['intermediate_steps'][-1][1]
            # Convert BytesIO to Base64
            # base64_string = base64.b64encode(byte_object.getvalue()).decode("utf-8")
            image_io = BytesIO(base64.b64decode(output))


            # python_query = output['Pandas']['intermediate_steps'][0][0]['tool_inputs']['query']

        #st.session_state['filters'].append(filters)

        # To see which examples are getting picked

        # st.session_state['prompt'].append(output['input'])
if st.session_state['agent_response']:
    with response_container:
        for i in range(len(st.session_state['agent_response'])):
            bot_avatar_image = "https://raw.githubusercontent.com/SAMPLE URL"
            with st.chat_message("user"):
                st.write(st.session_state["past"][i])

            with st.chat_message("assistant", avatar=bot_avatar_image):
                # comment if you don't want to see examples.
                #st.write(st.session_state['filters'][i])
                #st.write(st.session_state['prompt'][i])

                if plot_flag == 1:
                    st.image(image_io)
                else:
                    st.write(st.session_state["agent_response"][i])
                    # st.write(st.session_state["generated_plot"][i])

