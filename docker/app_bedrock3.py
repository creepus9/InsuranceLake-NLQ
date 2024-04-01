# Natural Language Query (NLQ) demo using Amazon RDS for PostgreSQL and Amazon Bedrock.
# Author: Gary A. Stafford (garystaf@amazon.com)
# Date: 2024-02-21
# Usage: streamlit run app_bedrock.py --server.runOnSave true

import ast
import pprint
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



from langchain_community.chat_models import BedrockChat
from langchain_experimental.sql import SQLDatabaseChain
from langchain.utilities.sql_database import SQLDatabase
from urllib.parse import quote_plus
from sqlalchemy import create_engine


from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentExecutor
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# ***** CONFIGURABLE PARAMETERS *****
REGION_NAME = "us-east-1"


# anthropic.claude-v2:1
# anthropic.claude-3-sonnet-20240229-v1:0
# anthropic.claude-3-haiku-20240307-v1:0
# anthropic.claude-instant-v1
#llm = Bedrock(model_id="ai21.j2-ultra-v1", model_kwargs={"maxTokens": 1024,"temperature": 0.0})


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
    model_kwargs=parameters,
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
    db = SQLDatabase(engine_athena)
    
    #retrieve metadata from glue data catalog
    def parse_catalog():
        #Connect to Glue catalog
        #get metadata of redshift serverless tables
        columns_str=''
        
        #define glue cient
        glue_client = boto3.client('glue',region_name='us-east-2')
        
        db="syntheticgeneraldata"
        response = glue_client.get_tables(DatabaseName =db)
        print (response)
        for tables in response['TableList']:
            #classification in the response for s3 and other databases is different. Set classification based on the response location
            if tables['StorageDescriptor']['Location'].startswith('s3'):  classification='s3' 
            else:  classification = tables['Parameters']['classification']
            for columns in tables['StorageDescriptor']['Columns']:
                    dbname,tblname,colname=tables['DatabaseName'],tables['Name'],columns['Name']
                    columns_str=columns_str+f'\n{classification}|{dbname}|{tblname}|{colname}'                     
        #API
        ## Append the metadata of the API to the unified glue data catalog
        columns_str=columns_str+'\n'+('api|meteo|weather|weather')
        return columns_str

    glue_catalog = parse_catalog()
    # Create the prompt
    # QUERY = """
    # Create a syntactically correct athena query to run based on the question, then look at the results of the query and return the answer like a human
    # {question}
    # """
    
    QUERY = """\n\nHuman: Given an input question, first create a syntactically correct athena query to run, then look at the results of the query and return the answer.

    Do not append 'Query:' to SQLQuery.
    
    Display SQLResult after the query is run in plain english that users can understand. 

    Provide answer in simple english statement.
    
      Here is info about the schema (between the <schema> tags):
    <schema>
    CREATE TABLE policydata
    (
        startdate           	date                	                    
        enddate             	date                	                    
        policynumber        	int                 	                    
        effectivedate       	date                	                    
        expirationdate      	date                	                    
        lobcode             	string              	                    
        customerno          	string              	                    
        insuredcompanyname  	string              	                    
        ein                 	string              	                    
        insuredcity         	string              	                    
        insuredstatecode    	string              	                    
        insuredcontactcellphone	string              	                    
        insuredcontactemail 	string              	                    
        insuredindustry     	string              	                    
        insuredsector       	string              	                    
        insurednumberofemployees	int                 	                    
        insuredemployeetier 	int                 	                    
        insuredannualrevenue	bigint              	                    
        neworrenewal        	string              	                    
        territory           	string              	                    
        distributionchannel 	string              	                    
        producercode        	int                 	                    
        agentname           	string              	                    
        accidentyeartotalincurredamount	decimal(10,2)       	                    
        policyinforce       	int                 	                    
        expiringpolicy      	int                 	                    
        expiringpremiumamount	decimal(10,2)       	                    
        writtenpremiumamount	decimal(10,2)       	                    
        writtenpolicy       	int                 	                    
        earnedpremium       	decimal(10,2)       	                    
        claimlimit          	decimal(13,2)       	                    
        execution_id        	string              	                    
        year                	string              	                    
        month               	string              	                    
        day                 	string              	                    
            
        # Partition Information	 	 
        # col_name            	data_type           	comment             
            
        year                	string              	                    
        month               	string              	                    
        day                 	string              	       
    )

    /*
    3 rows from policydata table:
    "startdate","enddate","policynumber","effectivedate","expirationdate","lobcode","customerno","insuredcompanyname","ein","insuredcity","insuredstatecode","insuredcontactcellphone","insuredcontactemail","insuredindustry","insuredsector","insurednumberofemployees","insuredemployeetier","insuredannualrevenue","neworrenewal","territory","distributionchannel","producercode","agentname","accidentyeartotalincurredamount","policyinforce","expiringpolicy","expiringpremiumamount","writtenpremiumamount","writtenpolicy","earnedpremium","claimlimit","execution_id","year","month","day"
    "2022-09-01","2022-09-30","9072092","2021-09-24","2022-09-24","WC","****","Cultivate Magnetic Action-Items","6cfcb7ef7f2bfaa21ec6f488ff7e36d462dbd2cfadb480e9b76e3e6e5cbbba35","Denver","CO","ef4f73a9a7081f7d39ba86f5d5ff12a01fa82d215a6053808f07bb2e4211fc02","ae7dd771709ea254c689d83a30f995fffec84e0d2b190d314def149c8bc81009","Restaurants","Services","9","1","1614240","New","West","Direct Portal","48920","VBBI Online Insurance LLC.",,"1","1","202.50","0.00","0","202.50","2025.00","e7d46ede-884a-49de-87cc-aa99d6b00aa0","2024","01","24"
    "2022-09-01","2022-09-30","9267867","2022-07-02","2023-07-02","AUTO","****","Deploy Collaborative Users","4e2c51b7ec1148b14f6774594a07d525dabcb109ee3a65ea4158fca13f518426","Columbus","GA","eadb479d782ff83ab683f4fd81de645d0d060e309b5ddedb67ca488a369c5dba","e1db61c52450876161eb20893e6f2a7b05495d13a81bd8bc3bfd1ccffaa41867","Retail Apparel","Retail","475","2","94639950","Renewal","Southeast","Agent Email Not-ACORD","33375","Lawley LLC",,"1","0","0.00","0.00","0","3958.33","39583.30","e7d46ede-884a-49de-87cc-aa99d6b00aa0","2024","01","24"
    "2022-09-01","2022-09-30","9310325","2022-03-05","2023-03-05","AUTO","****","Scale Clicks-And-Mortar Web-Readiness","e27a9c24eaf61ff086da49324176f0c800cd72a3b52558dd23893accc622c132","Indianapolis","IN","693335aa32c3bf5280f8a1e43e085b1f2c39b9aaf658f6fc200f1f7ece1d6a01","5351e869640130e64c6678bbdd9dfd1534109ac14d9b61cd95bb2c7b8a985bb1","Office Supplies","Consumer Non Cyclical","16","1","5246064","Renewal","Central","Agent Portal","47201","Robertson Ryan & Associates",,"1","0","0.00","0.00","0","133.33","1333.30","e7d46ede-884a-49de-87cc-aa99d6b00aa0","2024","01","24"
    */

    CREATE VIEW syntheticgeneraldata_consume
    (
        Policy Number	integer
        Summary Date	date
        Policy Effective Date	date
        Policy Expiration Date	date
        Company	varchar
        Line of Business	varchar
        LOBCode	varchar
        New or Renewal	varchar
        Industry	varchar
        Sector	varchar
        Distribution Channel	varchar
        City	varchar
        State	varchar
        Number of Employees	integer
        Employer Size Tier	integer
        Revenue	bigint
        Territory	varchar
        Claim Amount	decimal(10,2)
        Policy In-force	integer
        Policy Expiring	integer
        Premium Expiring	decimal(10,2)
        Written Premium	decimal(10,2)
        Written Policy	integer
        Earned Premium	decimal(10,2)
        Agent Name	varchar
        Agent Code	integer
    )

    /*
    3 rows from general_insurance_quicksight_view view:
    "Policy Number","Summary Date","Policy Effective Date","Policy Expiration Date","Company","Line of Business","LOBCode","New or Renewal","Industry","Sector","Distribution Channel","City","State","Number of Employees","Employer Size Tier","Revenue","Territory","Claim Amount","Policy In-force","Policy Expiring","Premium Expiring","Written Premium","Written Policy","Earned Premium","Agent Name","Agent Code"
    "9072092","2022-09-01","2021-09-24","2022-09-24","Cultivate Magnetic Action-Items","WC","WC","New","Restaurants","Services","Direct Portal","Denver","CO","9","1","1614240","West",,"1","1","202.50","0.00","0","202.50","VBBI Online Insurance LLC.","48920"
    "9267867","2022-09-01","2022-07-02","2023-07-02","Deploy Collaborative Users","AUTO","AUTO","Renewal","Retail Apparel","Retail","Agent Email Not-ACORD","Columbus","GA","475","2","94639950","Southeast",,"1","0","0.00","0.00","0","3958.33","Lawley LLC","33375"
    "9310325","2022-09-01","2022-03-05","2023-03-05","Scale Clicks-And-Mortar Web-Readiness","AUTO","AUTO","Renewal","Office Supplies","Consumer Non Cyclical","Agent Portal","Indianapolis","IN","16","1","5246064","Central",,"1","0","0.00","0.00","0","133.33","Robertson Ryan & Associates","47201"

    */
<schema>

    Question: {question}\n\nAssistant:\n\n Here is the SQL Query: """
    QUERY = """Human: Create a syntactically correct postgresql query to run based on the question, then look at the results of the query and return the answer like a human
Question: {question}\n\nAssistant:\n\n
"""
    # Only use the following tables:
    # {table_info}
    # If someone asks for the sales, they really mean the tickit.sales table.
    # If someone asks for the sales date, they really mean the column tickit.sales.saletime.
    # Just provide the SQL query. Do not provide any preamble or additional text.
    #ORIG
    # sql_db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True,return_intermediate_steps=True)
    sql_db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db,verbose=True,return_intermediate_steps=True)

    
    tools = [
        Tool(
            name="dbchain",
            func=sql_db_chain,
            description="Chat with Athena SQLDB",
        )
    ]

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    agent = initialize_agent(
        tools, 
        llm, 
        # agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        # agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True, 
        # agent_kwargs=agent_kwargs, 
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        # memory=memory,
        top_k=10000
    )
   
    
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
                            - How many artists are there in the collection?
                            - How many pieces of artwork are there?
                            - How many artists are there whose nationality is 'Italian'?
                            - How many artworks are by the artist 'Claude Monet'?
                            - How many artworks are classified as paintings?
                            - How many artworks were created by 'Spanish' artists?
                            - How many artist names start with the letter 'M'?
                        - Moderate
                            - How many artists are deceased as a percentage of all artists?
                            - Who is the most prolific artist? What is their nationality?
                            - What nationality of artists created the most artworks?
                            - What is the ratio of male to female artists? Return as a ratio.
                        - Complex
                            - How many artworks were produced during the First World War, which are classified as paintings?
                            - What are the five oldest pieces of artwork? Return the title and date for each.
                            - What are the 10 most prolific artists? Return their name and count of artwork.
                            - Return the artwork for Frida Kahlo in a numbered list, including the title and date.
                            - What is the count of artworks by classification? Return the first ten in descending order. Don't include Not_Assigned.
                            - What are the 12 artworks by different Western European artists born before 1900? Write Python code to output them with Matplotlib as a table. Include header row and font size of 12.
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
                            # output = sql_db_chain(user_input)
                            # output = agent_executor(user_input)
                            output = agent(question)
                            # print (output)
                            # pprint (output)
                            st.session_state.generated.append(output)
                            logging.info(st.session_state["query"])
                            logging.info(st.session_state["generated"])
                            # print("Query: ",st.session_state["query"])
                            # print("Generated: ",st.session_state["generated"])
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
                                    st.write(st.session_state["generated"][i]["output"])
                                    # st.write(st.session_state["generated"][i])
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
                    st.session_state["generated"][position]["input"], language="text"
                    # st.session_state["generated"][position], language="text"
                )

                st.markdown("intermediate_steps:")
                intermediate_steps = st.session_state["generated"][position]["intermediate_steps"]
                thoughts = ""
                for action, observation in intermediate_steps:
                    thoughts += action.log
                    thoughts += f"\nObservation: {observation}\nThought: "
                # Set the agent_scratchpad variable to that value
                # thoughts
                st.code(
                    # st.session_state["generated"][position]["intermediate_steps"],
                    thoughts,
                    language="sql",
                )

                # st.markdown("Results:")
                # st.code(
                #     st.session_state["generated"][position]["intermediate_steps"][3],
                #     language="python",
                # )

                st.markdown("Answer:")
                st.code(
                    st.session_state["generated"][position]["output"], language="text"
                )
                
                # data = ast.literal_eval(
                #     st.session_state["generated"][position]["intermediate_steps"]
                # )
                # if len(data) > 0 and len(data[0]) > 1:
                #     df = None
                #     st.markdown("Pandas DataFrame:")
                #     df = pd.DataFrame(data)
                #     df
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
    schemaathena='syntheticgeneraldata' #from Amazon Athena _consume
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
