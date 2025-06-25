import streamlit as st
from streamlit_chat import message
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# from langchain.chat_models import ChatOpenAI

from langchain.chat_models import AzureChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import HumanMessage
from langchain.utilities import SQLDatabase
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
# from langchain.chains.sql_database.examples import get_synthetic_sql_database_examples


llm = AzureChatOpenAI(
    temperature=0,
    openai_api_key='OPENAI_KEY',
    openai_api_base='OPENAI_BASE',
    deployment_name='gpt-4o',
    openai_api_version="OPENAI_VERSION"  # Ensure this matches your Azure OpenAI API version
)


def conversational_chat(question):
    # Set up the SQL database
    db_host='localhost'
    db_name='postgres'
    db_user = 'postgres'
    db_password='ayush1012'
    engine_string = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}/{db_name}'
    table_info = {
        "supermarket_sales":"""
                            CREATE TABLE public.supermarket_sales (
                                invoice_id varchar(50) NULL,
                                branch varchar(50) NULL,
                                city varchar(50) NULL,
                                customer_type varchar(50) NULL,
                                gender varchar(50) NULL,
                                product_line varchar(50) NULL,
                                unit_price float4 NULL,
                                qty int4 NULL,
                                tax float4 NULL,
                                total_sales float4 NULL,
                                "date" varchar(50) NULL,
                                payment_method varchar(50) NULL,
                                cogs float4 NULL,
                                gross_margin_percentage float4 NULL,
                                gross_income float4 NULL,
                                customer_rating float4 NULL
                            );

                            /* 3 rows from supermarket_sales_table:
                            invoice_id	branch	city	customer_type	gender	product_line	unit_price	qty	tax	total_sales	date	payment_method	cogs	gross_margin_percentage	gross_income	customer_rating
                            750-67-8428	A	Yangon	Member	Female	Health and beauty	74.69	7	26.1415	548.9715	01-05-2019	Ewallet	522.83	4.761904762	26.1415	9.1
                            226-31-3081	C	Naypyitaw	Normal	Female	Electronic accessories	15.28	5	3.82	80.22	03-08-2019	Cash	76.4	4.761904762	3.82	9.6
                            631-41-3108	A	Yangon	Normal	Male	Home and lifestyle	46.33	7	16.2155	340.5255	03-03-2019	Credit card	324.31	4.761904762	16.2155	7.4
                            */
                            invoice_id: A unique identifier for each sales transaction (generated automatically as a serial number). It serves as the primary key of the table.
                            branch: Indicates the specific branch where the sale took place. This could be a location code or name of the branch.
                            city: The name of the city where the transaction occurred. It helps identify the geographical region of the sale.
                            customer_type: Describes whether the customer is a regular or a first-time customer. Typical values could be ‘Regular’ or ‘First-Time’.
                            gender: Records the gender of the customer involved in the transaction (e.g., 'Male' or 'Female').
                            product_line: The category or type of product sold in the transaction, such as 'Electronics', 'Clothing', or 'Groceries'.
                            unit_price: The price per unit of the product sold, recorded with two decimal precision for monetary values.
                            qty: The quantity of products sold in the transaction, expressed as an integer.
                            tax: The amount of tax applied to the transaction, typically calculated as a percentage of the total sale, recorded with two decimal places.
                            total_sales: The total sales amount for the transaction, including both the product price and tax, recorded with two decimal precision.
                            date: The date on which the transaction took place, stored in YYYY-MM-DD format.
                            payment_method: The method of payment used by the customer, such as 'Cash', 'Credit Card', or 'Mobile Payment'.
                            cogs: The cost of goods sold, which represents the direct costs attributable to the production of the goods sold during the transaction.
                            gross_margin_percentage: The gross margin as a percentage of total sales, calculated by the formula: ((total_sales - cogs) / total_sales) * 100.
                            gross_income: The gross profit or income from the sale after deducting the cost of goods sold, recorded with two decimal precision.
                            customer_rating: A numerical rating given by the customer post-purchase, typically on a scale from 1 to 5, recorded with two decimal places.
                            /*

                            """
    }
    db = SQLDatabase.from_uri(database_uri=engine_string, custom_table_info = table_info)

    Prompt = f"""You are an expert SQL chatbot specializing in supermarket sales data analysis. 
                Your role is to assist users by crafting accurate SQL queries based on their questions about supermarket sales.

                User Question: {question}

                You have access to a table that contains comprehensive sales data, and you must generate SQL queries to answer questions such as total sales, product performance, customer trends, and other key metrics. 
                You are expected to follow the following instructions:

                    1) Understand user inquiries and interpret them in the context of supermarket sales.
                    2) Write well-formed SQL queries that retrieve the required information efficiently. Don't use triple backticks while writing the query.
                    3) Use appropriate filtering, aggregation, and sorting functions to generate insightful results.
                    4) Ensure all queries are correctly formatted and optimized for a PostgreSQL database.
                    5) Be proactive in ensuring the accuracy of your SQL statements and handle any edge cases the user might ask about.
                    6) Show all the financial metrics in unit of Rupees. USe the correct symbols and units."""

    # Create the SQL toolkit
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Create the SQL agent
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        # agent_type='openai-tools'
    )

    return agent.run(Prompt.format(question=question))



st.set_page_config(layout="centered")

st.markdown("<h2 style='font-family:sans-serif;text-align: center; color: Black;'> Welcome to GenAI</h2>", unsafe_allow_html=True)

st.markdown("<h4 style='font-family:sans-serif;text-align: center; color: Grey;'> Lets chat about SQL DB </h4>", unsafe_allow_html=True)


if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything about the SQL DB🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

#container for the chat history
response_container = st.container()
#container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        
        user_input = st.text_input("Query:", placeholder="Type your query...", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        output = conversational_chat(user_input)
        
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")