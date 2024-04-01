# Natural Language Query (NLQ) demo using Amazon RDS for PostgreSQL and Amazon Bedrock.
# Author: Gary A. Stafford (garystaf@amazon.com)
# Date: 2024-02-21
# Usage: streamlit run app_bedrock.py --server.runOnSave true

import ast
import boto3
import json
import logging
import os
import pandas as pd
import streamlit as st
import yaml
from botocore.exceptions import ClientError
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _postgres_prompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector.semantic_similarity import (
    SemanticSimilarityExampleSelector,
)
from langchain.sql_database import SQLDatabase
from langchain_community.llms import Bedrock


# from langchain.llms.bedrock import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain_experimental.sql import SQLDatabaseChain
from langchain.utilities.sql_database import SQLDatabase
from urllib.parse import quote_plus
from sqlalchemy.engine import create_engine




# ***** CONFIGURABLE PARAMETERS *****
REGION_NAME = "us-east-1"


# anthropic.claude-v2:1
# anthropic.claude-3-sonnet-20240229-v1:0
# anthropic.claude-3-haiku-20240307-v1:0
# anthropic.claude-instant-v1

MODEL_NAME = "anthropic.claude-3-haiku-20240307-v1:0"
TEMPERATURE =  0.3
TOP_P = 1
BASE_AVATAR_URL = (
    "https://raw.githubusercontent.com/garystafford-aws/static-assets/main/static"
)
ASSISTANT_ICON = os.environ.get("ASSISTANT_ICON", "bot-64px.png")
USER_ICON = os.environ.get("USER_ICON", "human-64px.png")
# HUGGING_FACE_EMBEDDINGS_MODEL = os.environ.get(
#     "HUGGING_FACE_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
# )


def main():
    st.set_page_config(
        page_title="InsuranceLake NLQ Demo",
        page_icon="🔎",
        layout="wide",
        # initial_sidebar_state="collapsed",
    )

    # # hide the hamburger bar menu
    # hide_streamlit_style = """
    #     <style>
    #     #MainMenu {visibility: hidden;}
    #     footer {visibility: hidden;}
    #     </style>

    # """
    # st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # os.environ["TOKENIZERS_PARALLELISM"] = "false"

    NO_ANSWER_MSG = "Sorry, I was unable to answer your question."

    parameters = {
        "temperature": TEMPERATURE,
        "max_tokens": 4000
    }
    llm = BedrockChat(region_name='us-east-1',
                  model_id=MODEL_NAME,
                  model_kwargs=parameters,
                  verbose=True)
    # llm = BedrockChat(
        
    #     model_id=MODEL_NAME,
    #     model_kwargs=parameters,
        
    # )

    # define datasource uri
    athena_uri= get_athena_uri(REGION_NAME)
    # print (athena_uri)
    engine_athena = create_engine(athena_uri, echo=False)
    db = SQLDatabase(engine_athena, include_tables=['policydata'],sample_rows_in_table_info=3)
    

    # Create the prompt
    QUERY = """
    Create a syntactically correct postgresql query to run based on the question, then look at the results of the query and return the answer like a human
    Always return just the answer in a way a human can unserstand. without any sql statements or steps of reasoning.

    {question}
    """
   
    # sql_db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True,return_intermediate_steps=True)
    # sql_db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db,verbose=True,return_intermediate_steps=True)
    sql_db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db,verbose=True,return_intermediate_steps=True)

    # return SQLDatabaseChain.from_llm(
    #     llm,
    #     db,
    #     prompt=few_shot_prompt,
    #     use_query_checker=False,  # must be False for OpenAI model
    #     verbose=True,
    #     return_intermediate_steps=True,
    # )

    # store the initial value of widgets in session state
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    if "query" not in st.session_state:
        st.session_state["query"] = ""

    if "query_text" not in st.session_state:
        st.session_state["query_text"] = ""

    if "query_error" not in st.session_state:
        st.session_state["query_error"] = ""

    tab1, tab2, tab3 = st.tabs(["Chatbot", "Details", "Technologies"])

    with tab1:
        col1, col2 = st.columns([6, 1], gap="medium")

        with col1:
            with st.container():
                st.markdown("## InsuranceLake NLQ Demo")
                st.markdown(
                    "#### Query InsuranceLake using natural language."
                )
                st.markdown(" ")
                with st.expander("Click here for sample questions..."):
                    st.markdown(
                        """
                        - Simple
                            - What is the average written premium for each line of business and employer size tier?
                            - Who are the top 10 agents by total written premium?
                            - What is the average claim amount for Commercial Auto policies, grouped by employer size tier and state?
                            - What are the policies with the highest and lowest claim amounts for each line of business?
                            - What are the top 5 industries with the highest total written premium for each line of business?
                            - What are the top 5 agents with the highest written premium, and what are their distribution channels?
                            - How many policies and what is the total earned premium for each combination of line of business and distribution channel?
                            - What is the average claim amount and the average number of employees for each industry and sector?
                            - How many policies and what is the total written premium for each combination of state and employer size tier, ordered by total written premium descending?
                            - What is the average earned premium and the average claim amount for each combination of line of business and distribution channel, for policies that are in-force and have a claim amount greater than $50,000?
                        - Moderate
                            - What is the average written premium for each line of business and employer size tier?
                            - Who are the top 10 agents by total written premium?
                            - What is the average claim amount for Commercial Auto policies, grouped by employer size tier and state?
                            - What are the policies with the highest and lowest claim amounts for each line of business?
                            - What are the top 5 industries with the highest total written premium for each line of business?
                            - What are the top 5 agents with the highest written premium, and what are their distribution channels?
                            - How many policies and what is the total earned premium for each combination of line of business and distribution channel?
                            - What is the average claim amount and the average number of employees for each industry and sector?
                            - How many policies and what is the total written premium for each combination of state and employer size tier, ordered by total written premium descending?
                            - What is the average earned premium and the average claim amount for each combination of line of business and distribution channel, for policies that are in-force and have a claim amount greater than $50,000?
                        - Complex
                            - For which combinations of line of business, distribution channel, and employer size tier is the premium retention ratio (earned premium / written premium) below 0.8?
                            - What are the top 10 policies with the highest claim amounts for each combination of line of business, industry, and employer size tier?
                            - For which combinations of line of business, industry, and territory is the policy retention rate (policies in-force / total policies) below 0.7?
                            - What are the top 5 agents with the highest total written premium for each combination of line of business, industry, and territory, along with their average claim amount?
                            - What are the policies with the highest and lowest earned premium for each combination of line of business, employer size tier, and state, along with their claim ratio (claim amount / earned premium)?
                            - calculate the combined ratio for all data per each industry and state combination. return the data as a markdown table matrix
                            - calculate the combined ratio for all data per each customer and state combination for the less profitable customers. return the data as a markdown table matrix
                            - What are the top 5 territories with the highest total written premium, and what are the average claim amount and the average number of employees for each territory, ordered by total written premium descending?
                            - How many policies, what is the total earned premium, and what is the average claim amount for each combination of line of business, distribution channel, and employer size tier, for policies that are new, have a claim amount greater than $100,000, and have a revenue greater than $10,000,000, ordered by total earned premium descending?
                            - How many policies, what is the total written premium, and what is the average claim amount for each combination of industry, sector, and distribution channel, for policies that are expiring, have a claim amount between $50,000 and $200,000, and have a number of employees between 100 and 500, ordered by total written premium ascending?
                            - How many policies, what is the total earned premium, and what is the average claim amount for each combination of state, city, and line of business, for policies that have a policy effective date in the year 2020, have a written premium greater than $50,000, and have a claim amount less than $20,000, ordered by total earned premium descending?
                            - How many policies, what is the total written premium, and what is the average claim amount for each combination of agent name, line of business, and distribution channel, for policies that are new, have a policy effective date in the year 2021, have a written premium greater than $100,000, and have a claim amount greater than $75,000, ordered by total written premium descending?
                        - Unrelated to the Dataset
                            - Give me a recipe for chocolate cake.
                            - Who won the 2022 FIFA World Cup final?
                    """
                    )
                st.markdown(" ")
            with st.container():
                input_text = st.text_input(
                    "Ask a question:",
                    "",
                    key="query_text",
                    placeholder="Your question here...",
                    on_change=clear_text(),
                )
                logging.info(input_text)

                user_input = st.session_state["query"]

                if user_input:
                    with st.spinner(text="Thinking..."):
                        st.session_state.past.append(user_input)
                        try:
                            question = QUERY.format(question=user_input
                                                    # ,table_info=glue_catalog
                            )
                            print (question)
                            output = sql_db_chain(question)
                            # output = sql_db_chain(user_input)
                            st.session_state.generated.append(output)
                            logging.info(st.session_state["query"])
                            logging.info(st.session_state["generated"])
                        except Exception as exc:
                            st.session_state.generated.append(NO_ANSWER_MSG)
                            logging.error(exc)
                            st.session_state["query_error"] = exc

                # https://discuss.streamlit.io/t/streamlit-chat-avatars-not-working-on-cloud/46713/2
                if st.session_state["generated"]:
                    with col1:
                        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                            if (i >= 0) and (
                                    st.session_state["generated"][i] != NO_ANSWER_MSG
                            ):
                                with st.chat_message(
                                        "assistant",
                                        avatar=f"{BASE_AVATAR_URL}/{ASSISTANT_ICON}",
                                ):
                                    st.write(st.session_state["generated"][i]["result"])
                                with st.chat_message(
                                        "user",
                                        avatar=f"{BASE_AVATAR_URL}/{USER_ICON}",
                                ):
                                    st.write(st.session_state["past"][i])
                            else:
                                with st.chat_message(
                                        "assistant",
                                        avatar=f"{BASE_AVATAR_URL}/{ASSISTANT_ICON}",
                                ):
                                    st.write(NO_ANSWER_MSG)
                                with st.chat_message(
                                        "user",
                                        avatar=f"{BASE_AVATAR_URL}/{USER_ICON}",
                                ):
                                    st.write(st.session_state["past"][i])
        with col2:
            with st.container():
                st.button("clear chat", on_click=clear_session)
    with tab2:
        with st.container():
            st.markdown("### Details")
            st.markdown("Amazon Bedrock Model:")
            st.code(MODEL_NAME, language="text")

            position = len(st.session_state["generated"]) - 1
            if (position >= 0) and (
                    st.session_state["generated"][position] != NO_ANSWER_MSG
            ):
                st.markdown("Question:")
                st.code(
                    st.session_state["generated"][position]["query"], language="text"
                )

                st.markdown("SQL Query:")
                st.code(
                    st.session_state["generated"][position]["intermediate_steps"][1],
                    language="sql",
                )
                
                st.markdown("ALL:")
                st.code(
                    st.session_state["generated"][position], language="text"
                )
                st.markdown("BLABLA2:")
                st.code(
                    st.session_state["generated"][position]["intermediate_steps"][0],
                    language="sql",
                )

                st.markdown("Results:")
                st.code(
                    st.session_state["generated"][position]["intermediate_steps"][3],
                    language="python",
                )

                st.markdown("Answer:")
                st.code(
                    st.session_state["generated"][position]["result"], language="text"
                )
                import pprint
                from decimal import Decimal

                # pprint.pprint (st.session_state["generated"][position]["intermediate_steps"][3])
                # pprint.pprint (ast.dump(ast.parse(st.session_state["generated"][position]["intermediate_steps"][3], mode='eval'), indent=4))
                # import io
                # data=pd.read_table(st.session_state["generated"][position]["intermediate_steps"][3], encoding='utf8') 

                # Convert the list of tuples into a dataframe 
                
                # data = ast.literal_eval(
                #     st.session_state["generated"][position]["intermediate_steps"][3]
                # )
                data =st.session_state["generated"][position]["intermediate_steps"][3]
                # print (data)   
                data = eval(data)

                # Create a DataFrame from the list of tuples
                if len(data) > 0 and len(data[0]) > 1:
                    df = None
                    st.markdown("Pandas DataFrame:")
                    df = pd.DataFrame(data)
                    st.dataframe(df)
            st.markdown("Query Error:")
            st.code(
                st.session_state["query_error"], language="text"
            )
    with tab3:
        with st.container():
            st.markdown("### Technologies")
            st.markdown(" ")

            st.markdown("##### Natural Language Query (NLQ)")
            st.markdown(
                """
            [Natural language query (NLQ)](https://www.yellowfinbi.com/glossary/natural-language-query), according to Yellowfin, enables analytics users to ask questions of their data. It parses for keywords and generates relevant answers sourced from related databases, with results typically delivered as a report, chart or textual explanation that attempt to answer the query, and provide depth of understanding.
            """
            )
            st.markdown(" ")

            st.markdown("##### The MoMa Collection Datasets")
            st.markdown(
                """
            [The Museum of Modern Art (MoMA) Collection](https://github.com/MuseumofModernArt/collection) contains over 120,000 pieces of artwork and 15,000 artists. The datasets are available on GitHub in CSV format, encoded in UTF-8. The datasets are also available in JSON. The datasets are provided to the public domain using a [CC0 License](https://creativecommons.org/publicdomain/zero/1.0/).
            """
            )
            st.markdown(" ")

            st.markdown(" ")

            st.markdown("##### Amazon Bedrock")
            st.markdown(
                """
            [Amazon Bedrock](https://aws.amazon.com/bedrock/) is the easiest way to build and scale generative AI applications with foundation models (FMs).
            """
            )

            st.markdown("##### LangChain")
            st.markdown(
                """
            [LangChain](https://python.langchain.com/en/latest/index.html) is a framework for developing applications powered by language models. LangChain provides standard, extendable interfaces and external integrations.
            """
            )
            st.markdown(" ")

            st.markdown("##### Chroma")
            st.markdown(
                """
            [Chroma](https://www.trychroma.com/) is the open-source embedding database. Chroma makes it easy to build LLM apps by making knowledge, facts, and skills pluggable for LLMs.
            """
            )
            st.markdown(" ")

            st.markdown("##### Streamlit")
            st.markdown(
                """
            [Streamlit](https://streamlit.io/) is an open-source app framework for Machine Learning and Data Science teams. Streamlit turns data scripts into shareable web apps in minutes. All in pure Python. No front-end experience required.
            """
            )

        with st.container():
            st.markdown("""---""")
            st.markdown(
                "![](app/static/github-24px-blk.png) [Feature request or bug report?](https://github.com/aws-solutions-library-samples/guidance-for-natural-language-queries-of-relational-databases-on-aws/issues)"
            )
            st.markdown(
                "![](app/static/github-24px-blk.png) [The MoMA Collection datasets on GitHub](https://github.com/MuseumofModernArt/collection)"
            )
            st.markdown(
                "![](app/static/flaticon-24px.png) [Icons courtesy flaticon](https://www.flaticon.com)"
            )


def get_athena_uri(region_name):
    # SQLAlchemy 2.0 reference: https://docs.sqlalchemy.org/en/20/dialects/postgresql.html
    # URI format: postgresql+psycopg2://user:pwd@hostname:port/dbname

    connathena=f"athena.us-east-2.amazonaws.com"
    portathena='443' #Update, if port is different.
    schemaathena='syntheticgeneraldata_consume' #from Amazon Athena
    s3stagingathena=f's3://dev-insurancelake-566541803426-us-east-2-glue-temp/query-results/'#from Amazon Athena settings
    wkgrpathena='insurancelake'#Update, if the workgroup is different

    connection_string = f"awsathena+rest://{connathena}:{portathena}/{schemaathena}?s3_staging_dir={s3stagingathena}/&work_group={wkgrpathena}"
    # connection_string = f"awsathena+rest://{connathena}:{portathena}/{schemaathena}?s3_staging_dir={s3stagingathena}/"
    # conn_str = "awsathena+rest://:@athena.{region_name}.amazonaws.com:443/" \
    #        "{schema_name}?s3_staging_dir={s3_staging_dir}"

    return connection_string





    


def clear_text():
    st.session_state["query"] = st.session_state["query_text"]
    st.session_state["query_text"] = ""
    st.session_state["query_error"] = ""


def clear_session():
    for key in st.session_state.keys():
        del st.session_state[key]


if __name__ == "__main__":
    main()
