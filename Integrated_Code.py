import streamlit as st
from azure.core.credentials import AzureKeyCredential
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
import plotly.express as px
import plotly.graph_objects as go
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
import openai
import pyodbc
import urllib
from sqlalchemy import create_engine
import pandas as pd
from azure.identity import InteractiveBrowserCredential
from pandasai import SmartDataframe
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from PIL import Image
import base64
import pandasql as ps
import matplotlib.pyplot as plt
import seaborn as sns
import re
import requests
from io import BytesIO
import io
from devices_library import *
from WindowsSummaryApproach import *
global history
import os
import json

file_path = 'chat_history.json'

def load_list():
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        return []

def save_list(lst):
    with open(file_path, 'w') as file:
        json.dump(lst, file)
        
history = load_list()

print(f"I have this history : {history}")

if not history:
    print("First run, initializing the list.")
    history = []  # Initialize as empty list
else:
    print("Not the first run, list loaded from file.")
    
    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-02-01",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )


global model
model = AzureChatOpenAI(
            azure_deployment="Thruxton_R",
            api_version='2024-03-01-preview',temperature = 0.0)

#Initializing some variables for Devices
if not hasattr(st.session_state, 'display_history_devices'):
    st.session_state.display_history_devices = []
if not hasattr(st.session_state, 'context_history_devices'):
    st.session_state.context_history_devices = []
if not hasattr(st.session_state, 'curr_response'):
    st.session_state.curr_response = ""
if not hasattr(st.session_state, 'user_question'):
    st.session_state.user_question = None 
if not hasattr(st.session_state, 'devices_flag'):
    st.session_state.devices_flag = False
if not hasattr(st.session_state, 'devices_approach'):
    st.session_state.devices_approach = ""
if not hasattr(st.session_state, 'selected_sugg'):
    st.session_state.selected_sugg = None
if not hasattr(st.session_state, 'prompt_sugg'):
    st.session_state.prompt_sugg = None
if not hasattr(st.session_state, 'selected_questions'):
    st.session_state.selected_questions = ""
if not hasattr(st.session_state, 'copilot_curr_ques'):
    st.session_state.copilot_curr_ques = None
####################################################################################################################----------------Copilot-------------------#####################################################################################################

Copilot_Sentiment_Data  = pd.read_csv("Cleaned_Combined_Data.csv")

st.markdown("""
<head>
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/font-awesome/4.4.0/css/font-awesome.min.css">
</head>
""", unsafe_allow_html=True)

def Sentiment_Score_Derivation(value):
    try:
        if value == "positive":
            return 1
        elif value == "negative":
            return -1
        else:
            return 0
    except Exception as e:
        err = f"An error occurred while deriving Sentiment Score: {e}"
        return err    

#Deriving Sentiment Score and Review Count columns into the dataset
Copilot_Sentiment_Data["Sentiment_Score"] = Copilot_Sentiment_Data["Sentiment"].apply(Sentiment_Score_Derivation)
Copilot_Sentiment_Data["Review_Count"] = 1.0
Copilot_Sentiment_Data["Product"] = Copilot_Sentiment_Data["Product"].astype(str).str.upper()
Copilot_Sentiment_Data["Product_Family"] = Copilot_Sentiment_Data["Product_Family"].astype(str).str.upper()
Copilot_Sentiment_Data["Keywords"] = Copilot_Sentiment_Data["Keywords"].astype(str).str.title()

#overall_net_sentiment = round(sum(Copilot_Sentiment_Data["Sentiment_Score"])*100/sum(Copilot_Sentiment_Data["Review_Count"]),1)
#overall_review_count = sum(Copilot_Sentiment_Data["Review_Count"])


def convert_top_to_limit(sql):
    try:
        tokens = sql.upper().split()
        is_top_used = False

        for i, token in enumerate(tokens):
            if token == 'TOP':
                is_top_used = True
                if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                    limit_value = tokens[i + 1]
                    # Remove TOP and insert LIMIT and value at the end
                    del tokens[i:i + 2]
                    tokens.insert(len(tokens), 'LIMIT')
                    tokens.insert(len(tokens), limit_value)
                    break  # Exit loop after successful conversion
                else:
                    raise ValueError("TOP operator should be followed by a number")

        return ' '.join(tokens) if is_top_used else sql
    except Exception as e:
        err = f"An error occurred while converting Top to Limit in SQL Query: {e}"
        return err

def process_tablename(sql, table_name):
    try:
        x = sql.upper()
        query = x.replace(table_name.upper(), table_name)
        
        if '!=' in query or '=' in query:
            query = query.replace("!="," NOT LIKE ")
            query = query.replace("="," LIKE ")
            
            pattern = r"LIKE\s'([^']*)'"
            def add_percentage_signs(match):
                return f"LIKE '%{match.group(1)}%'"
            query = re.sub(pattern, add_percentage_signs, query)
        
        return query
    except Exception as e:
        err = f"An error occurred while processing table name in SQL query: {e}"
        return err        
 
#-------------------------------------------------------------------------------------------------------Summarization-------------------------------------------------------------------------------------------------------#
 
def get_conversational_chain_quant():
    global model
    try:
        prompt_template = """
        
            You are an AI Chatbot assistant. Understand the user question carefully and follow all the instructions mentioned below.
                1. Your job is to convert the user question to an SQL query (Follow Microsoft SQL Server SSMS syntax). You have to give the query so that it can be used on Microsoft SQL Server SSMS. You have to only return the query as a result.
                2. There is only one table with the table name `Copilot_Sentiment_Data` where each row is a user review. The table has 10 columns:
                    - Review: Review of the Copilot Product and its Competitors
                    - Data_Source: From where the review was taken. It contains different retailers. The possible values are: [chinatechnews, DigitalTrends, Engadget, clubic, g2.com, gartner, JP-ASCII, Jp-Impresswatch, Itmedia, LaptopMag, NotebookCheck, PCMag, TechAdvisor, TechRadar, TomsHardware, TechCrunch, Verge, ZDNET, PlayStore, App Store, AppStore, Reddit, YouTube, Facebook, Instagram, X, VK, Forums, News, Print, Blogs/Websites, Reviews, Wordpress, Podcast, TV, Quora, LinkedIn, Videos]
                    - Geography: From which country or region the review was given. The possible values are: [China, France, Japan, US, Brazil, Canada, Germany, India, Mexico, UK, Australia, Unknown, Venezuela, Vietnam, Cuba, Colombia, Iran, Ukraine, Northern Mariana Islands, Uruguay, Taiwan, Spain, Russia, Bolivia, Argentina, Lebanon, Finland, Saudi Arabia, Oman, United Arab Emirates, Austria, Luxembourg, Macedonia, Puerto Rico, Bulgaria, Qatar, Belgium, Italy, Switzerland, Peru, Czech Republic, Thailand, Greece, Netherlands, Romania, Indonesia, Benin, Sweden, South Korea, Poland, Portugal, Tonga, Norway, Denmark, Samoa, Ireland, Turkey, Ecuador, Guernsey, Botswana, Kenya, Chad, Bangladesh, Nigeria, Singapore, Malaysia, Malawi, Georgia, Hong Kong, Philippines, South Africa, Jordan, New Zealand, Pakistan, Nepal, Jamaica, Egypt, Macao, Bahrain, Tanzania, Zimbabwe, Serbia, Estonia, Jersey, Afghanistan, Kuwait, Tunisia, Israel, Slovakia, Panama, British Indian Ocean Territory, Comoros, Kazakhstan, Maldives, Kosovo, Ghana, Costa Rica, Belarus, Sri Lanka, Cameroon, San Marino, Antigua and Barbuda]
                    - Title: Title of the review
                    - Review_Date: Date on which the review was posted
                    - Product: Corresponding product for the review. The possible values are 'COPILOT','OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI'
                    - Product_Family: Version or type of the corresponding product for which the review was posted. The possible values are: 'Windows Copilot', 'Microsoft Copilot', 'Copilot for Security', 'Copilot Pro', 'Copilot for Microsoft 365', 'Edge Copilot', 'Github Copilot', 'Copilot for Mobile', 'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI', 'Gemini AI For Mobile', 'ChatGPT For Mobile', 'Perplexity AI For Mobile', 'Claude AI For Mobile'.
                    - Sentiment: Sentiment of the review. The possible values are: 'positive', 'neutral', 'negative'
                    - Aspect: Aspect or feature of the product being reviewed. The possible values are: 'Interface', 'Connectivity', 'Privacy', 'Compatibility', 'Generic', 'Innovation', 'Reliability', 'Productivity', 'Price', 'Text Summarization/Generation', 'Code Generation', 'Ease of Use', 'Performance', 'Personalization/Customization', 'Accessibility','End User Usecase'
                    - Keywords: Keywords mentioned in the review
                    - Review_Count: Always 1 for each review or each row
                    - Sentiment_Score: 1 for positive, 0 for neutral, and -1 for negative sentiment

                
        IMPORTANT : User won't exactly mention the exact Geography Names, Product Names, Product Families, Data Source name, Aspect names. Please make sure to change/correct to the values that you know from the context and then provide SQL Query.
                    Exmaple : User Question : "Summarize the reviews of Copilot from Chinanews"
                        We know that Chinanews in not any of the DataSource, Geography and so on.
                        So Change it to "Summarize the reviews of Copilot from Chinatechnews" as this is more relevant and faces no issues when we pull SQL Queries
                     Exmaple : User Question : "Summarize the reviews of ChatGPT"
                        We know that ChatGPT is not any of the Product, Product_Family and so on.
                        So Change it to "Summarize the reviews of OpenAi GPT" as this is more relevant and faces no issues in understanding
                    Exmaple : User Question : "Summarize the reviews of Copilot from USA"
                        We know that USA in not any of the Geography, Data Source and so on.
                        So Change it to "Summarize the reviews of Copilot from US" as this is more relevant and faces no issues in understanding
                        
                        Same goes for all the columns
                
        3. Net sentiment is calculcated by sum of Sentiment_Score divided by sum of Review_Count. It should be in percentage. Example:
                    SELECT ((SUM(Sentiment_Score)*1.0)/(SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment 
                    FROM Copilot_Sentiment_Data
                    ORDER BY Net_Sentiment DESC
        4. If an user is asking for Summarize reviews of any product. Note that user is not seeking for reviews, user is seeking for all the Quantitative results product(Net Sentiment & Review Count) and also (Aspect wise sentiment and Aspect wise review count). So choose to Provide Net Sentiment and Review Count and Aspect wise sentiment and their respective review count and Union them in single table
        
        5. IMPORTANT : REMEMBER THAT ALL THE NAMES IN PRODUCT_FAMILIES ARE NOT DIFFERENT VERSION OF COPILOT, THERE ARE SOME COMPETITORS AS WELL.
            5.1 IMPORTANT : 1. Different Product families/versions of copilot is 'Windows Copilot', 'Microsoft Copilot', 'Copilot for Security', 'Copilot Pro', 'Copilot for Microsoft 365', 'Github Copilot', 'Copilot for Mobile'.
                            2. Competitors of various versions of Copilot are : 'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI', 'Gemini AI For Mobile', 'ChatGPT For Mobile', 'Perplexity AI For Mobile', 'Claude AI For Mobile'.
            5.2 IMPORTANT : Competitors of Copilot is 'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI'
         
            IMPORTANT : User can confuse a lot, but understand the question carefully and respond:
            Example : I am a story writer , Can you tell me which AI is good for writing stories based on user reviews? -> In this case, user confuses by telling that he is a story teller and all but he just needs to know "What is the best AI for Text Generation" -> Which is again decided based on comparison.
         
        6. Example : If the user Quesiton is "Summarize reviews of Github Copilot"
        
                6.1 "Summarize reviews of CoPilot for Mobile" - User seeks for net sentiment and aspect wise net sentiment of "CoPilot for Mobile" Product Family and their respective aspect review count in a single table
                    
                    The Query has to be like this 
                        
                        SELECT 'TOTAL' AS Aspect, 
                        ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                        SUM(Review_Count) AS Review_Count
                        FROM Copilot_Sentiment_Data
                        WHERE Product_Family LIKE '%CoPilot for Mobile%'

                        UNION

                        SELECT Aspect, 
                        ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                        SUM(Review_Count) AS Review_Count
                        FROM Copilot_Sentiment_Data
                        WHERE Product_Family LIKE '%CoPilot for Mobile%'
                        GROUP BY Aspect
                        ORDER BY Review_Count DESC
                    
                6.2 And if user specifies any Geography/DataSource. Make sure to apply those filters in the SQL Query response
                
                    if Geography is included:
                    
                            SELECT 'TOTAL' AS Aspect, 
                            ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                            SUM(Review_Count) AS Review_Count
                            FROM Copilot_Sentiment_Data
                            WHERE Product_Family LIKE '%CoPilot for Mobile%'
                            GEOGRAPHY LIKE '%US%'

                            UNION

                            SELECT Aspect, 
                            ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                            SUM(Review_Count) AS Review_Count
                            FROM Copilot_Sentiment_Data
                            WHERE Product_Family LIKE '%CoPilot for Mobile%' AND
                            GEOGRAPHY LIKE '%US%'
                            GROUP BY Aspect
                            ORDER BY Review_Count DESC

                    
                    
                6.3 IMPORTANT : if any particular Aspect "Code Generation" in user question:
                
                
                    IMPORTANT : TOTAL Should always be the Overall Sentiment of a product family/Product.
                    

                        SELECT 'TOTAL' AS Aspect, 
                        ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                        SUM(Review_Count) AS Review_Count
                        FROM Copilot_Sentiment_Data
                        WHERE Product_Family LIKE '%CoPilot for Mobile%'

                        UNION

                        SELECT Aspect, 
                        ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                        SUM(Review_Count) AS Review_Count
                        FROM Copilot_Sentiment_Data
                        WHERE Product_Family LIKE '%CoPilot for Mobile%'
                        GROUP BY Aspect
                        HAVING Aspect LIKE %'Code Generation'%
                        ORDER BY Review_Count DESC
                        
                VERY IMPORTANT : Here, important thing to notice is in the "TOTAL" of aspect column, %Code Generation% is not added as it is Overall sentiment. It should be the Overall sentiment of CoPilot for Mobile in this case.


        7. Generic Queries: 
                
            7.1. Sentiment mark is calculated by sum of Sentiment_Score.
            7.2. Net sentiment is calculcated by sum of Sentiment_Score divided by sum of Review_Count. It should be in percentage. Example:
                    SELECT ((SUM(Sentiment_Score)*1.0)/(SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment 
                    FROM Copilot_Sentiment_Data
                    ORDER BY Net_Sentiment DESC
            7.3. Net sentiment across country or across region is sentiment mark of a country divided by total reviews of that country. It should be in percentage.
                Example to calculate net sentiment across country:
                    SELECT Geography, ((SUM(Sentiment_Score)*1.0) / (SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment
                    FROM Copilot_Sentiment_Data
                    GROUP BY Geography
                    ORDER BY Net_Sentiment DESC
            7.4. Net Sentiment across a column "X" is calculcated by Sentiment Mark for each "X" divided by Total Reviews for each "X".
                Example to calculate net sentiment across a column "X":
                    SELECT X, ((SUM(Sentiment_Score)*1.0) / (SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment
                    FROM Copilot_Sentiment_Data
                    GROUP BY X
                    ORDER BY Net_Sentiment DESC
            7.5. Distribution of sentiment is calculated by sum of Review_Count for each Sentiment divided by overall sum of Review_Count
                Example: 
                    SELECT Sentiment, SUM(ReviewCount)*100/(SELECT SUM(Review_Count) AS Reviews FROM Copilot_Sentiment_Data) AS Total_Reviews 
                    FROM Copilot_Sentiment_Data 
                    GROUP BY Sentiment
                    ORDER BY Total_Reviews DESC
            7.6. If the user asks for net sentiment across any country: example : Net sentiment of Windows Copilot in US geography
                   SELECT ((SUM(Sentiment_Score)) / (SUM(Review_Count))) * 100 AS Net_Sentiment
                   FROM Copilot_Sentiment_Data
                   WHERE Geography LIKE "%US%"
                   
           IMPORTANT : These are the aspects we have : 'Interface', 'Connectivity', 'Privacy','Compatibility', 'Generic', 'Innovation', 'Reliability','Productivity', 'Price', 'Text Summarization/Generation','Code Generation', 'Ease of Use', 'Performance','Personalization/Customization','Accessibility','End User Usecase', 'Image Generation'.
                    But user who writes the question might not know the exact name of the aspect we have, for example : User might write "Picture Generation" for 'Image Generarion' and "writing codes" for code generation. 
                    You should be carefull while rephrasing it.
            
            REMEBER TO USE LIKE OPERATOR whenever you use 'where' clause
            
            
                     
        8. Points to remember :  
            8.1. Convert numerical outputs to float upto 1 decimal point.
            8.2. Always include ORDER BY clause to sort the table based on the aggregate value calculated in the query.
            8.3 Top Country is based on Sentiment_Score i.e., the Country which have highest sum(Sentiment_Score)
            8.4 Always use 'LIKE' operator whenever they mention about any Country. Use 'LIMIT' operator instead of TOP operator.Do not use TOP OPERATOR. Follow syntax that can be used with pandasql.
            8.5 If you are using any field in the aggregate function in select statement, make sure you add them in GROUP BY Clause.
            8.6 Make sure to Give the result as the query so that it can be used on Microsoft SQL server SSMS.
            8.7 Important: Always show Net_Sentiment in Percentage upto 1 decimal point. Hence always make use of ROUND function while giving out Net Sentiment and Add % Symbol after it.
            8.8 Important: User can ask question about any categories including Aspects, Geograpgy, Sentiment etc etc. Hence, include the in SQL Query if someone ask it.
            8.9 Important: You Response should directly starts from SQL query nothing else.
            8.10 Important: Always use LIKE keyword instead of '=' symbol while generating SQL query.
            8.11 Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.
            8.12 Sort all Quantifiable outcomes based on review count.

        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for quantifiable review summarization: {e}"
        return err
        

#Function to convert user prompt to quantitative outputs for Copilot Review Summarization
def query_quant(user_question,history,vector_store_path="combine_indexes"):
    try:
        # Initialize the embeddings model
        embeddings = AzureOpenAIEmbeddings(azure_deployment="MV_Agusta")
        
        # Load the vector store with the embeddings model
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        
        # Rest of the function remains unchanged
        chain = get_conversational_chain_quant()
        docs = []
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        SQL_Query = response["output_text"]
        SQL_Query = convert_top_to_limit(SQL_Query)
        SQL_Query = process_tablename(SQL_Query,"Copilot_Sentiment_Data")
        print(SQL_Query)
        # st.write(SQL_Query)
        data = ps.sqldf(SQL_Query, globals())
        data_1 = data
        html_table = data.to_html(index=False)
    #     return html_table
        return data_1
    except Exception as e:
        err = f"An error occurred while generating response for quantitative review summarization: {e}"
        return err



def get_conversational_chain_aspect_wise_detailed_summary():
    global model
    try:
        prompt_template = """
        
        1. Your Job is to analyse the Net Sentiment, Aspect wise sentiment and Key word regarding the different aspect and summarize the reviews that user asks for utilizing the reviews and numbers you get. Use maximum use of the numbers and Justify the numbers using the reviews.
        
        
        Your will receive Aspect wise net sentiment of the Product. you have to concentrate on top 4 Aspects based on ASPECT_RANKING.
        For that top 4 Aspect you will get top 2 keywords for each aspect. You will receive each keywords' contribution and +ve mention % and negative mention %
        You will receive reviews of that devices focused on these aspects and keywords.

        For Each Aspect

        Condition 1 : If the aspect sentiment is higher than net sentiment, which means that particular aspect is driving the net sentiment high for that Product. In this case provide why the aspect sentiment is higher than net sentiment based on reviews.
        Condition 2 : If the aspect sentiment is Lower than net sentiment, which means that particular aspect is driving the net sentiment Lower for that Product. In this case provide why the aspect sentiment is lower than net sentiment based on reviews.

        IMPORTANT: Use only the data provided to you and do not rely on pre-trained documents.

            Your summary should justify the above conditions and tie in with the net sentiment and aspect sentiment and keywords. Mention the difference between Net Sentiment and Aspect Sentiment (e.g., -2% or +2% higher than net sentiment) in your summary and provide justification.

            IMPOPRTANT : Start with : "These are the 4 major aspects users commented about" and mention their review count contributions. These top 4 shold be based on ASPECT_RANKING Column

                           These are the 4 top ranked aspects users commented about - IMPORTANT : These top 4 should be from Aspect Ranking:
                           
                           IMPORTANT : DO NOT CONSIDER GENERIC AS ONE OF THE ASPECTS
                           
                           
                       Net Sentiment is nothing but the "TOTAL" - Sentiment in the Provided data 

                        - Total Review for CoPilot for Mobile Product is y
                        - Code Generarion: x% of the reviews mentioned this aspect
                        - Ease of Use: x% of the reviews mentioned this aspect
                        - Compatibility: x% of the reviews mentioned this aspect
                        - Interface: x% of the reviews mentioned this aspect

                        Code Generation:
                        - The aspect sentiment for price is x%, which is higher than the net sentiment of x%. This indicates that the aspect of price is driving the net sentiment higher for the Github Copilot.
                        -  The top keyword for price is "buy" with a contribution of x%. It has a positive percentage of x% and a negative percentage of x%.
                              - Users mentioned that the Github Copilot offers good value for the price and is inexpensive.
                        - Another top keyword for price is "price" with a contribution of 26.89%. It has a positive percentage of x% and a negative percentage of x%.
                            - Users praised the affordable price of the Github Copilot and mentioned that it is worth the money.

                        Ease of use:
                        - The aspect sentiment for performance is x%, which is lower than the net sentiment of x%. This indicates that the aspect of performance is driving the net sentiment lower for the Github Copilot.
                        - The top keyword for performance is "fast" with a contribution of x%. It has a positive percentage of 16.76% and a neutral percentage of x%.
                            - Users mentioned that the Github Copilot is fast and offers good speed.
                        - Another top keyword for performance is "speed" with a contribution of x%. It has a positive percentage of 9.12% and a negative percentage of x%.
                            - Users praised the speed of the Github Copilot and mentioned that it is efficient.


                        lIKE THE ABOVE ONE EXPLAIN OTHER 2 ASPECTS

                        Overall Summary:
                        The net sentiment for the Github Copilot is x%, while the aspect sentiment for price is x%, performance is x%, software is x%, and design is x%. This indicates that the aspects of price and design are driving the net sentiment higher, while the aspects of performance and software are driving the net sentiment lower for the Github Copilot. Users mentioned that the Github Copilot offers good value for the price, is fast and efficient in performance, easy to set up and use in terms of software, and has a sleek and high-quality design.

                        Some Pros and Cons of the device, 


           IMPORTANT : Do not ever change the above template of Response. Give Spaces accordingly in the response to make it more readable. In the above template, x are the numbers you should use from the data provided to you.

           A Good Response should contains all the above mentioned poniters in the example. 
               1. Net Sentiment and The Aspect Sentiment
               2. Total % of mentions regarding the Aspect
               3. A Quick Summary of whether the aspect is driving the sentiment high or low
               4. Top Keyword: "Usable" (Contribution: x%, Positive: x%, Negative: x%)
                    - Users have praised the usable experience on the Cobilot for Mobile, with many mentioning the smooth usage and easy to use
                    - Some users have reported experiencing lag while not very great to use, but overall, the gaming Ease of use is highly rated.

                Top 3 Keywords : Their Contribution, Postitive mention % and Negative mention % and one ot two positive mentions regarding this keywords in each pointer

                5. IMPORTANT : Pros and Cons in pointers (overall, not related to any aspect)
                6. Overall Summary
                    
          Enhance the model’s comprehension to accurately interpret user queries by:
          Recognizing abbreviations for country names (e.g., ‘DE’ for Germany, ‘USA’or 'usa' or 'US' for the United States of America) and expanding them to their full names for clarity.
          Understanding product family names even when written in reverse order or missing connecting words (e.g., ‘copilot in windows 11’ as ‘copilot windows’ and ‘copilot for security’ as ‘copilot security’ etc.).
          Utilizing context and available data columns to infer the correct meaning and respond appropriately to user queries involving variations in product family names or geographical references
          Please provide a comprehensive Review summary, feature comparison, feature suggestions for specific product families and actionable insights that can help in product development and marketing strategies.
          Generate acurate response only, do not provide extra information.
            
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for detailed review summarization: {e}"
        print("Its here")
        return err

# Function to handle user queries using the existing vector store
def query_aspect_wise_detailed_summary(user_question,vector_store_path="combine_indexes"):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="MV_Agusta")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_aspect_wise_detailed_summary()
        print("I have the chain")
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        print("Oh oh")
        err = f"An error occurred while getting LLM response for detailed review summarization: {e}"
        return err

#----------------------------------------------------------------------------------------------------------------------Visualization------------------------------------------------------------------------#

def get_conversational_chain_quant_classify2_compare():
    global model
    try:
        prompt_template = """

                You are an AI Chatbot assistant. Understand the user question carefully and follow all the instructions mentioned below.
                1. Your job is to convert the user question to an SQL query (Follow Microsoft SQL Server SSMS syntax). You have to give the query so that it can be used on Microsoft SQL Server SSMS. You have to only return the query as a result.
                2. There is only one table with the table name `Copilot_Sentiment_Data` where each row is a user review. The table has 10 columns:
                    - Review: Review of the Copilot Product and its Competitors
                    - Data_Source: From where the review was taken. It contains different retailers. The possible values are: [chinatechnews, DigitalTrends, Engadget, clubic, g2.com, gartner, JP-ASCII, Jp-Impresswatch, Itmedia, LaptopMag, NotebookCheck, PCMag, TechAdvisor, TechRadar, TomsHardware, TechCrunch, Verge, ZDNET, PlayStore, App Store, AppStore, Reddit, YouTube, Facebook, Instagram, X, VK, Forums, News, Print, Blogs/Websites, Reviews, Wordpress, Podcast, TV, Quora, LinkedIn, Videos]
                    - Geography: From which country or region the review was given. The possible values are: [China, France, Japan, US, Brazil, Canada, Germany, India, Mexico, UK, Australia, Unknown, Venezuela, Vietnam, Cuba, Colombia, Iran, Ukraine, Northern Mariana Islands, Uruguay, Taiwan, Spain, Russia, Bolivia, Argentina, Lebanon, Finland, Saudi Arabia, Oman, United Arab Emirates, Austria, Luxembourg, Macedonia, Puerto Rico, Bulgaria, Qatar, Belgium, Italy, Switzerland, Peru, Czech Republic, Thailand, Greece, Netherlands, Romania, Indonesia, Benin, Sweden, South Korea, Poland, Portugal, Tonga, Norway, Denmark, Samoa, Ireland, Turkey, Ecuador, Guernsey, Botswana, Kenya, Chad, Bangladesh, Nigeria, Singapore, Malaysia, Malawi, Georgia, Hong Kong, Philippines, South Africa, Jordan, New Zealand, Pakistan, Nepal, Jamaica, Egypt, Macao, Bahrain, Tanzania, Zimbabwe, Serbia, Estonia, Jersey, Afghanistan, Kuwait, Tunisia, Israel, Slovakia, Panama, British Indian Ocean Territory, Comoros, Kazakhstan, Maldives, Kosovo, Ghana, Costa Rica, Belarus, Sri Lanka, Cameroon, San Marino, Antigua and Barbuda]
                    - Title: Title of the review
                    - Review_Date: Date on which the review was posted
                    - Product: Corresponding product for the review. The possible values are 'COPILOT','OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI'
                    - Product_Family: Version or type of the corresponding product for which the review was posted. The possible values are: 'Windows Copilot', 'Microsoft Copilot', 'Copilot for Security', 'Copilot Pro', 'Copilot for Microsoft 365', 'Edge Copilot', 'Github Copilot', 'Copilot for Mobile', 'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI', 'Gemini AI For Mobile', 'ChatGPT For Mobile', 'Perplexity AI For Mobile', 'Claude AI For Mobile'
                    - Sentiment: Sentiment of the review. The possible values are: 'positive', 'neutral', 'negative'
                    - Aspect: Aspect or feature of the product being reviewed. The possible values are: 'Interface', 'Connectivity', 'Privacy', 'Compatibility', 'Generic', 'Innovation', 'Reliability', 'Productivity', 'Price', 'Text Summarization/Generation', 'Code Generation', 'Ease of Use', 'Performance', 'Personalization/Customization', 'Accessibility','End User Usecase'
                    - Keywords: Keywords mentioned in the review
                    - Review_Count: Always 1 for each review or each row
                    - Sentiment_Score: 1 for positive, 0 for neutral, and -1 for negative sentiment

                
        IMPORTANT : User won't exactly mention the exact Geography Names, Product Names, Product Families, Data Source name, Aspect names. Please make sure to change/correct to the values that you know from the context and then provide SQL Query.
                    Exmaple : User Question : "Summarize the reviews of Copilot from Chinanews"
                        We know that Chinanews in not any of the DataSource, Geography and so on.
                        So Change it to "Summarize the reviews of Copilot from Chinatechnews" as this is more relevant and faces no issues when we pull SQL Queries
                     Exmaple : User Question : "Summarize the reviews of ChatGPT"
                        We know that ChatGPT is not any of the Product, Product_Family and so on.
                        So Change it to "Summarize the reviews of OpenAi GPT" as this is more relevant and faces no issues in understanding
                    Exmaple : User Question : "Summarize the reviews of Copilot from USA"
                        We know that USA in not any of the Geography, Data Source and so on.
                        So Change it to "Summarize the reviews of Copilot from US" as this is more relevant and faces no issues in understanding
                        
                        Same goes for all the columns
                        
                        
                    IMPORTANT : REMEMBER THAT ALL THE NAMES IN PRODUCT_FAMILIES ARE NOT DIFFERENT VERSION OF COPILOT, THERE ARE SOME COMPETITORS AS WELL.
                    IMPORTANT : Different Product families/versions of copilot is 'Windows Copilot', 'Microsoft Copilot', 'Copilot for Security', 'Copilot Pro', 'Copilot for Microsoft 365', 'Github Copilot', 'Copilot for Mobile'
                    IMPORTANT : Competitors of Copilot are 'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI'
                    IMPORTANT : whenever user ask like Compare the net sentiment for github Copilot and its competitors the it should take the below competitors as Product_Family.
                                Competitors of versions of Copilot are : 'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI', 'Gemini AI For Mobile', 'ChatGPT For Mobile', 'Perplexity AI For Mobile', 'Claude AI For Mobile'.
                    
                    IMPORTANT : User can confuse a lot, but understand the question carefully and respond:
            Example : I am a story writer , Can you tell me which AI is good for writing stories based on user reviews? -> In this case, user confuses by telling that he is a story teller and all but he just needs to know "What is the best AI for Text Generation" -> Which is again decided based on comparison.
                    
                    IMPORTANT : These are the aspects we have : 'Interface', 'Connectivity', 'Security/Privacy','Compatibility', 'Generic', 'Innovation', 'Reliability','Productivity', 'Price', 'Text Summarization/Generation','Code Generation', 'Ease of Use', 'Performance','Personalization/Customization','Accessibility','End User Usecase', 'Image Generation'.
                    But user who writes the question might not know the exact name of the aspect we have, for example : User might write "Picture Generation" for 'Image Generarion' and "writing codes" for code generation. 
                    You should be carefull while rephrasing it.
                   
                    
                IMPORTANT : Out of these Product Family Names, it can be segregated into 2 things : One is Different versions of Copilot like [Microsoft Copilot, Windows CoPilot, Copilot, Github Copilot , Copilot for Security, Copilot Pro, Copilot for Microsoft 365, Copilot for Mobile] and the other ones are Competitors of copilot like ['OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI', 'Gemini AI For Mobile', 'ChatGPT For Mobile', 'Perplexity AI For Mobile', 'Claude AI For Mobile'.]
                
                So, whenever user asks for compare different versions of copilot, the user meant compare Different versions of Copilot like [Microsoft Copilot, Windows CoPilot, Copilot, Github Copilot , Copilot for Security, Copilot Pro, Copilot for Microsoft 365, Copilot for Mobile]
                
                and whenever user asks for compare copilot and its competitors, the user meant compare 'Copilot','OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI' - IMPORTANT Note : This is from Product column and not from Product_Family
                
                and whenever user asks for Compare all copilot versions with their competitors, you can go ahead with all the values in Product_Family (Including all the values in it)
                
                and whenever user asks different AIs without any mentions, you can go ahead with all the values in Product_Family (Including all the values in it)
                
                   IMPORTANT : Copilot is the Product and Product_Famiies contains are different versions of Copilot that contains 'Copilot' string in their names. Compatitors of copilot are the one without 'Copilot string' not in the Product Family column.
                    

                    1. If the user asks for Net Sentiment for column "X", the query should be exactly like this: 

                            SELECT X, ((SUM(Sentiment_Score)) / (SUM(Review_Count))) * 100 AS Net_Sentiment, SUM(Review_Count) AS Review_Count
                            FROM Copilot_Sentiment_Data
                            GROUP BY X
                            ORDER BY Review_Count DESC

                    2. If the user asks for overall review count, the query should be like this:
                            SELECT SUM(Review_Count) 
                            FROM Copilot_Sentiment_Data
                    3. If the user asks for review distribution across column 'X', the query should be like this:
                            SELECT 'X', SUM(Review_Count) * 100 / (SELECT SUM(Review_Count) FROM Copilot_Sentiment_Data) AS Review_Distribution
                            FROM Copilot_Sentiment_Data 
                            GROUP BY 'X'
                            ORDER BY Review_Distribution DESC
                    4. If the user asks for column 'X' Distribution across column 'Y', the query should be like this: 
                            SELECT 'Y', SUM('X') * 100 / (SELECT SUM('X') AS Reviews FROM Copilot_Sentiment_Data) AS Distribution_PCT
                            FROM Copilot_Sentiment_Data 
                            GROUP BY 'Y'
                            ORDER BY Distribution_PCT DESC
                    5. If the user asks for net sentiment across any country: example : Net sentiment of Windows Copilot in US geography
                               SELECT ((SUM(Sentiment_Score)) / (SUM(Review_Count))) * 100 AS Net_Sentiment
                               FROM Copilot_Sentiment_Data
                               WHERE Geography LIKE "%US%"
                               
                    6. IMPORTANT NOTE :
                   
                        THIS IS THE ONLY WAY TO CALCULATE NET SENTIMENT : ((SUM(Sentiment_Score)*1.0)/(SUM(Review_Count)*1.0)) * 100
                        
                    7. Review count mix/ Review count Percentage of a product by aspect:
                    
                    
                        SELECT ASPECT, (SUM(REVIEW_COUNT)*100/(SELECT SUM(REVIEW_COUNT) FROM Copilot_Sentiment_Data WHERE PRODUCT_FAMILY LIKE '%GITHUB COPILOT%')) AS REVIEW_COUNT_PERCENTAGE
                        FROM Copilot_Sentiment_Data
                        WHERE PRODUCT_FAMILY LIKE '%GITHUB COPILOT%'
                        GROUP BY ASPECT
                        ORDER BY REVIEW_COUNT_PERCENTAGE DESC

                    
                    For all the comparison related user query, the format should remain the same. i.e., the column names that we are giving as alias should remain the same. Do not change the schema. 

                    IMPORTANT -> Comparison SQL Queries
                    
                    IMPORTANT : User can confuse a lot, but understand the question carefully and respond:
                    Example : I am a story writer , Can you tell me which AI is good for writing stories based on user reviews? -> In this case, user confuses by telling that he is a story teller and all but he just needs to know "What is the best AI for Text Generation" -> Which is again decided based on comparison.
                    
                            1. Compare aspect wise sentiment of different Product Families / Compare different AI models / Compare aspect wise sentiment of different AI models:
                            
                                Follow this same template whenever user asks for compare different AI models across Geography, make sure to change all the aspects to Geography
                                \
                                
                                IMPORTANT : Everytime give the overall sentiment. Every PRODUCT Copilot overall and aspect should be in your response.
                                
                                IMPORTANT : ADD COPILOT_OVERALL_NETSENTIMENT EVERYTIME FOR COMPARISON

                                 WITH NETSENTIMENT AS (
                                                        SELECT
                                                            PRODUCT_FAMILY,
                                                            ASPECT,
                                                            ((SUM(SENTIMENT_SCORE) * 1.0) / (SUM(REVIEW_COUNT) * 1.0)) * 100 AS NET_SENTIMENT,
                                                            SUM(REVIEW_COUNT) AS REVIEW_COUNT
                                                        FROM
                                                            Copilot_Sentiment_Data
                                                        GROUP BY
                                                            PRODUCT_FAMILY, ASPECT
                                                    ),

                                                    COPILOT_NETSENTIMENT AS (
                                                        SELECT
                                                            'Copilot' AS PRODUCT_FAMILY,
                                                            ASPECT,
                                                            ((SUM(SENTIMENT_SCORE) * 1.0) / (SUM(REVIEW_COUNT) * 1.0)) * 100 AS NET_SENTIMENT,
                                                            SUM(REVIEW_COUNT) AS REVIEW_COUNT
                                                        FROM
                                                            Copilot_Sentiment_Data
                                                        WHERE
                                                            PRODUCT = 'Copilot'
                                                        GROUP BY
                                                            ASPECT
                                                    ),

                                                    COPILOT_OVERALL_NETSENTIMENT AS (
                                                        SELECT
                                                            'Copilot' AS PRODUCT_FAMILY,
                                                            ((SUM(SENTIMENT_SCORE) * 1.0) / (SUM(REVIEW_COUNT) * 1.0)) * 100 AS NET_SENTIMENT
                                                        FROM
                                                            Copilot_Sentiment_Data
                                                        WHERE
                                                            PRODUCT = 'Copilot'
                                                    )

                                                    SELECT
                                                        NS.PRODUCT_FAMILY,
                                                        NS.ASPECT,
                                                        NS.NET_SENTIMENT AS NET_SENTIMENT_ASPECT,
                                                        ONS.NET_SENTIMENT AS NET_SENTIMENT_OVERALL,
                                                        NS.REVIEW_COUNT
                                                    FROM
                                                        NETSENTIMENT NS
                                                    JOIN
                                                        (SELECT
                                                            PRODUCT_FAMILY,
                                                            ((SUM(SENTIMENT_SCORE) * 1.0) / (SUM(REVIEW_COUNT) * 1.0)) * 100 AS NET_SENTIMENT
                                                        FROM
                                                            Copilot_Sentiment_Data
                                                        GROUP BY
                                                            PRODUCT_FAMILY) ONS
                                                    ON
                                                        NS.PRODUCT_FAMILY  LIKE  ONS.PRODUCT_FAMILY

                                                    UNION ALL

                                                    SELECT
                                                        ONS.PRODUCT_FAMILY,
                                                        'OVERALL' AS ASPECT,
                                                        ONS.NET_SENTIMENT AS NET_SENTIMENT_ASPECT,
                                                        ONS.NET_SENTIMENT AS NET_SENTIMENT_OVERALL,
                                                        SUM(NS.REVIEW_COUNT) AS REVIEW_COUNT
                                                    FROM
                                                        (SELECT
                                                            PRODUCT_FAMILY,
                                                            ((SUM(SENTIMENT_SCORE) * 1.0) / (SUM(REVIEW_COUNT) * 1.0)) * 100 AS NET_SENTIMENT
                                                        FROM
                                                            Copilot_Sentiment_Data
                                                        GROUP BY
                                                            PRODUCT_FAMILY) ONS
                                                    JOIN
                                                        NETSENTIMENT NS
                                                    ON
                                                        ONS.PRODUCT_FAMILY  LIKE  NS.PRODUCT_FAMILY
                                                    GROUP BY
                                                        ONS.PRODUCT_FAMILY, ONS.NET_SENTIMENT

                                                    UNION ALL

                                                    SELECT
                                                        CNS.PRODUCT_FAMILY,
                                                        CNS.ASPECT,
                                                        CNS.NET_SENTIMENT AS NET_SENTIMENT_ASPECT,
                                                        CONS.NET_SENTIMENT AS NET_SENTIMENT_OVERALL,
                                                        CNS.REVIEW_COUNT
                                                    FROM
                                                        COPILOT_NETSENTIMENT CNS
                                                    JOIN
                                                        COPILOT_OVERALL_NETSENTIMENT CONS
                                                    ON
                                                        CNS.PRODUCT_FAMILY = CONS.PRODUCT_FAMILY

                                                    UNION ALL

                                                    SELECT
                                                        'Copilot' AS PRODUCT_FAMILY,
                                                        'OVERALL' AS ASPECT,
                                                        CONS.NET_SENTIMENT AS NET_SENTIMENT_ASPECT,
                                                        CONS.NET_SENTIMENT AS NET_SENTIMENT_OVERALL,
                                                        SUM(CNS.REVIEW_COUNT) AS REVIEW_COUNT
                                                    FROM
                                                        COPILOT_NETSENTIMENT CNS
                                                    JOIN
                                                        COPILOT_OVERALL_NETSENTIMENT CONS
                                                    ON
                                                        CNS.PRODUCT_FAMILY = CONS.PRODUCT_FAMILY
                                                    GROUP BY
                                                        CONS.PRODUCT_FAMILY, CONS.NET_SENTIMENT

                                                    ORDER BY
                                                        REVIEW_COUNT DESC;
                                                        
                                        
                            2. Specific Product_Family comparision with competitors : Compare Github copilot with its competitors:/ Compare [Product Family] with its competitors. It can any AI model. Change the names in the below SQL accordingly.
                            
                            IMPORTANT : ADD COPILOT_OVERALL_NETSENTIMENT EVERYTIME FOR COMPARISON
                            
                            
                            WITH NETSENTIMENT AS (
                                                    SELECT
                                                        PRODUCT_FAMILY,
                                                        ASPECT,
                                                        ((SUM(SENTIMENT_SCORE) * 1.0) / (SUM(REVIEW_COUNT) * 1.0)) * 100 AS NET_SENTIMENT,
                                                        SUM(REVIEW_COUNT) AS REVIEW_COUNT
                                                    FROM
                                                        Copilot_Sentiment_Data
                                                    WHERE
                                                        PRODUCT_FAMILY IN ('GITHUB COPILOT','OPENAI GPT', 'GEMINI AI', 'CLAUDE AI', 'VERTEX AI', 'PERPLEXITY AI', 'GEMINI AI FOR MOBILE','CHATGPT FOR MOBILE', 'PERPLEXITY AI FOR MOBILE', 'CLAUDE AI FOR MOBILE')
                                                    GROUP BY
                                                        PRODUCT_FAMILY, ASPECT
                                                ),

                                                COPILOT_NETSENTIMENT AS (
                                                    SELECT
                                                        'Copilot' AS PRODUCT_FAMILY,
                                                        ASPECT,
                                                        ((SUM(SENTIMENT_SCORE) * 1.0) / (SUM(REVIEW_COUNT) * 1.0)) * 100 AS NET_SENTIMENT,
                                                        SUM(REVIEW_COUNT) AS REVIEW_COUNT
                                                    FROM
                                                        Copilot_Sentiment_Data
                                                    WHERE
                                                        PRODUCT = 'Copilot'
                                                    GROUP BY
                                                        ASPECT
                                                ),

                                                COPILOT_OVERALL_NETSENTIMENT AS (
                                                    SELECT
                                                        'Copilot' AS PRODUCT_FAMILY,
                                                        ((SUM(SENTIMENT_SCORE) * 1.0) / (SUM(REVIEW_COUNT) * 1.0)) * 100 AS NET_SENTIMENT
                                                    FROM
                                                        Copilot_Sentiment_Data
                                                    WHERE
                                                        PRODUCT = 'Copilot'
                                                )

                                                SELECT
                                                    NS.PRODUCT_FAMILY,
                                                    NS.ASPECT,
                                                    NS.NET_SENTIMENT AS NET_SENTIMENT_ASPECT,
                                                    ONS.NET_SENTIMENT AS NET_SENTIMENT_OVERALL,
                                                    NS.REVIEW_COUNT
                                                FROM
                                                    NETSENTIMENT NS
                                                JOIN
                                                    (SELECT
                                                        PRODUCT_FAMILY,
                                                        ((SUM(SENTIMENT_SCORE) * 1.0) / (SUM(REVIEW_COUNT) * 1.0)) * 100 AS NET_SENTIMENT
                                                    FROM
                                                        Copilot_Sentiment_Data
                                                    WHERE
                                                        PRODUCT_FAMILY IN ('GITHUB COPILOT', 'OPENAI GPT', 'GEMINI AI', 'CLAUDE AI', 'VERTEX AI', 'PERPLEXITY AI', 'GEMINI AI FOR MOBILE','CHATGPT FOR MOBILE', 'PERPLEXITY AI FOR MOBILE', 'CLAUDE AI FOR MOBILE')
                                                    GROUP BY
                                                        PRODUCT_FAMILY) ONS
                                                ON
                                                    NS.PRODUCT_FAMILY = ONS.PRODUCT_FAMILY

                                                UNION ALL

                                                SELECT
                                                    ONS.PRODUCT_FAMILY,
                                                    'OVERALL' AS ASPECT,
                                                    ONS.NET_SENTIMENT AS NET_SENTIMENT_ASPECT,
                                                    ONS.NET_SENTIMENT AS NET_SENTIMENT_OVERALL,
                                                    SUM(NS.REVIEW_COUNT) AS REVIEW_COUNT
                                                FROM
                                                    (SELECT
                                                        PRODUCT_FAMILY,
                                                        ((SUM(SENTIMENT_SCORE) * 1.0) / (SUM(REVIEW_COUNT) * 1.0)) * 100 AS NET_SENTIMENT
                                                    FROM
                                                        Copilot_Sentiment_Data
                                                    WHERE
                                                        PRODUCT_FAMILY IN ('GITHUB COPILOT', 'OPENAI GPT', 'GEMINI AI', 'CLAUDE AI', 'VERTEX AI', 'PERPLEXITY AI', 'GEMINI AI FOR MOBILE','CHATGPT FOR MOBILE', 'PERPLEXITY AI FOR MOBILE', 'CLAUDE AI FOR MOBILE')
                                                    GROUP BY
                                                        PRODUCT_FAMILY) ONS
                                                JOIN
                                                    NETSENTIMENT NS
                                                ON
                                                    ONS.PRODUCT_FAMILY = NS.PRODUCT_FAMILY
                                                GROUP BY
                                                    ONS.PRODUCT_FAMILY, ONS.NET_SENTIMENT

                                                UNION ALL

                                                SELECT
                                                    CNS.PRODUCT_FAMILY,
                                                    CNS.ASPECT,
                                                    CNS.NET_SENTIMENT AS NET_SENTIMENT_ASPECT,
                                                    CONS.NET_SENTIMENT AS NET_SENTIMENT_OVERALL,
                                                    CNS.REVIEW_COUNT
                                                FROM
                                                    COPILOT_NETSENTIMENT CNS
                                                JOIN
                                                    COPILOT_OVERALL_NETSENTIMENT CONS
                                                ON
                                                    CNS.PRODUCT_FAMILY = CONS.PRODUCT_FAMILY

                                                UNION ALL

                                                SELECT
                                                    'Copilot' AS PRODUCT_FAMILY,
                                                    'OVERALL' AS ASPECT,
                                                    CONS.NET_SENTIMENT AS NET_SENTIMENT_ASPECT,
                                                    CONS.NET_SENTIMENT AS NET_SENTIMENT_OVERALL,
                                                    SUM(CNS.REVIEW_COUNT) AS REVIEW_COUNT
                                                FROM
                                                    COPILOT_NETSENTIMENT CNS
                                                JOIN
                                                    COPILOT_OVERALL_NETSENTIMENT CONS
                                                ON
                                                    CNS.PRODUCT_FAMILY = CONS.PRODUCT_FAMILY
                                                GROUP BY
                                                    CONS.PRODUCT_FAMILY, CONS.NET_SENTIMENT

                                                ORDER BY
                                                    REVIEW_COUNT DESC;

                                                          
                                    IMPORTANT NOTE : Product column only have : 'COPILOT','OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI' and Product_Family only have the versions of copilot and its competitors they are : 'Copilot for Mobile', 'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI', 'Gemini AI For Mobile', 'ChatGPT For Mobile', 'Perplexity AI For Mobile', 'Claude AI For Mobile'..
                                    So Whenever user meant specific version of copilot : Take them from Product_Family and others you can take from Product.
       
                            3. End User UserCase - Comparison of Keywords :
                            
                            If a user asks anything regarding "End User Usercase" Aspect, follow the below SQL Query Template.
                            
                            IMPORTANT : Compare "net sentiment of different Keywords" in end user use case for different Product Family Names/AI models? - Here the user doesn't want to compare different AI for Paticular Aspect.
                            User seeks to compare the keywords of particular aspect among the AI models.
                
                                WITH NETSENTIMENT AS (
                                                        SELECT
                                                            PRODUCT_FAMILY,
                                                            KEYWORDS,
                                                            ASPECT,
                                                            ((SUM(SENTIMENT_SCORE) * 1.0) / (SUM(REVIEW_COUNT) * 1.0)) * 100 AS NET_SENTIMENT,
                                                            SUM(REVIEW_COUNT) AS REVIEW_COUNT
                                                        FROM
                                                            Copilot_Sentiment_Data
                                                        WHERE
                                                            ASPECT = 'End User Usecase'
                                                        GROUP BY
                                                            PRODUCT_FAMILY, KEYWORDS, ASPECT
                                                    )

                                                    -- SELECT ONLY THE ASPECT NET SENTIMENTS WITHOUT ASPECT COLUMN
                                                    SELECT
                                                        NS.PRODUCT_FAMILY,
                                                        NS.KEYWORDS,
                                                        NS.NET_SENTIMENT AS NET_SENTIMENT_KEYWORDS,
                                                        NS.REVIEW_COUNT
                                                    FROM
                                                        NETSENTIMENT NS
                                                    ORDER BY
                                                        NS.REVIEW_COUNT DESC;

                        VERY IMPORTANT : For Comparision, Always give the Product_Family/Product column along with aspect or geography in the SQL Query
                    
                        
                    Same goes for all comparisions.

                    This is how you should calculate review count mix of Github copilot across different Aspect.
                        
                    Net sentiment is calculcated by sum of Sentiment_Score divided by sum of Review_Count. It should be in percentage. Example:
                    SELECT ((SUM(Sentiment_Score)*1.0)/(SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment 
                    FROM Copilot_Sentiment_Data
                    ORDER BY Net_Sentiment DESC

                    Important: While generating SQL query to calculate net_sentiment across column 'X' and 'Y', if 'Y' has less distinct values, keep your response like this - SELECT 'Y','X', ((SUM(Sentiment_Score)) / (SUM(Review_Count))) * 100 AS Net_Sentiment, SUM(Review_Count) AS Review_Count FROM Copilot_Sentiment_Data GROUP BY 'Y','X'
                    
                    Important: Always replace '=' operator with LIKE keyword and add '%' before and after filter value for single or multiple WHERE conditions in the generated SQL query . For example, if the query is like - 'SELCT * FROM Copilot_Sentiment_Data WHERE PRODUCT='ABC' AND GEOGRAPHY='US' ORDER BY Review_Count' , you should modify the query and share the output like this - 'SELCT * FROM Copilot_Sentiment_Data WHERE PRODUCT LIKE '%ABC%' AND GEOGRAPHY LIKE '%US%' ORDER BY Review_Count'

                    Important: Always include ORDER BY clause to sort the table based on the aggregate value calculated in the query.
                    Important: Use 'LIMIT' operator instead of TOP operator.Do not use TOP OPERATOR. Follow syntax that can be used with pandasql.
                    Important: You Response should directly start from SQL query nothing else.
                    Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.
                    
                    Enhance the model’s comprehension to accurately interpret user queries by:
                      Recognizing abbreviations for country names (e.g., ‘DE’ for Germany, ‘USA’or 'usa' or 'US' for the United States of America) and expanding them to their full names for clarity.
                      Understanding product family names even when written in reverse order or missing connecting words such as HP Laptop 15, Lenovo Legion 5 15 etc
                      Utilizing context and available data columns to infer the correct meaning and respond appropriately to user queries involving variations in product family names or geographical references
                      Please provide a comprehensive Review summary, feature comparison, feature suggestions for specific product families and actionable insights that can help in product development and marketing strategies.
                      Generate acurate response only, do not provide extra information.

                Context:\n {context}?\n
                Question: \n{question}\n

                Answer:
                """
         
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for quantifiable review summarization: {e}"
        return err

def query_quant_classify2_compare(user_question, vector_store_path="combine_indexes"):
    global history
    try:
        # Initialize the embeddings model
        embeddings = AzureOpenAIEmbeddings(azure_deployment="MV_Agusta")
        
        # Load the vector store with the embeddings model
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        
        # Rest of the function remains unchanged
        chain = get_conversational_chain_quant_classify2_compare()
        docs = []
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        SQL_Query = response["output_text"]
        # st.write(SQL_Query)
        SQL_Query = convert_top_to_limit(SQL_Query)
        SQL_Query = process_tablename(SQL_Query,"Copilot_Sentiment_Data")
        print(SQL_Query)
        # st.write(SQL_Query)
        data = ps.sqldf(SQL_Query, globals())
        data_1 = data
        html_table = data.to_html(index=False)
    #     return html_table
        return data_1
    except Exception as e:
        err = f"An error occurred while generating response for quantitative review summarization: {e}"
        return errs
    


def get_conversational_chain_quant_classify2():
    global model
    try:
        prompt_template = """

                You are an AI Chatbot assistant. Understand the user question carefully and follow all the instructions mentioned below. The data contains Copilot and the competitors of copilot. 
                The different competitors of Copilot are "Claude AI", "Gemini AI", "OpenAI GPT", "Perplexity AI", "Vertex AI".
                Competitors of various versions of Copilot are : 'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI', 'Gemini AI For Mobile', 'ChatGPT For Mobile', 'Perplexity AI For Mobile', 'Claude AI For Mobile'.

                    1. Your Job is to convert the user question to SQL Query (Follow Microsoft SQL server SSMS syntax.). You have to give the query so that it can be used on Microsoft SQL server SSMS.You have to only return query as a result.
                    2. There is only one table with table name Copilot_Sentiment_Data where each row is a user review. The table has 10 columns, they are: "Claude AI", "Gemini AI", "OpenAI GPT", "Perplexity AI", "Vertex AI"
                        Review: Review of the Copilot Product and Competitors of Copilot
                        Data_Source: From where is the review taken. It contains different retailers - It contains following values : [chinatechnews, DigitalTrends, Engadget, clubic, g2.com, gartner, JP-ASCII, Jp-Impresswatch, Itmedia, LaptopMag, NotebookCheck, PCMag, TechAdvisor, TechRadar, TomsHardware, TechCrunch, Verge, ZDNET, PlayStore, App Store, AppStore, Reddit, YouTube, Facebook, Instagram, X, VK, Forums, News, Print, Blogs/Websites, Reviews, Wordpress, Podcast, TV, Quora, LinkedIn, Videos]
                        Geography: From which Country or Region the review was given. It contains different Geography.
                                   list of Geographies in the table - Values in this column 
                                   [ 'China', 'Unknown', 'France', 'Japan', 'US', 'Australia', 'Brazil', 'Canada', 'Germany', 'India', 'Mexico', 'UK' ]
                        Title: What is the title of the review
                        Review_Date: The date on which the review was posted
                        Product: Corresponding product for the review. It contains following values: "COPILOT" and "Claude AI", "Gemini AI", "OpenAI GPT", "Perplexity AI", "Vertex AI". 
                        
         IMPORTANT: "Claude AI", "Gemini AI", "OpenAI GPT", "Perplexity AI", "Vertex AI" all are competitors of Copilot
         
                        Product_Family: Which version or type of the corresponding Product was the review posted for. Different Product Names  - It contains following Values - 'Windows Copilot', 'Microsoft Copilot', 'Copilot for Security', 'Copilot Pro', 'Copilot for Microsoft 365', 'Edge Copilot', 'Github Copilot', 'Copilot for Mobile', 'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI', 'Gemini AI For Mobile', 'ChatGPT For Mobile', 'Perplexity AI For Mobile', 'Claude AI For Mobile']
                        Sentiment: What is the sentiment of the review. It contains following values: 'positive', 'neutral', 'negative'.
                        Aspect: The review is talking about which aspect or feature of the product. It contains following values: 'Interface', 'Connectivity', 'Privacy','Compatibility', 'Generic', 'Innovation', 'Reliability','Productivity', 'Price', 'Text Summarization/Generation','Code Generation', 'Ease of Use', 'Performance','Personalization/Customization','Accessibility'.
                        Keywords: What are the keywords mentioned in the product
                        Review_Count - It will be 1 for each review or each row
                        Sentiment_Score - It will be 1, 0 or -1 based on the Sentiment.
                        
                IMPORTANT : User won't exactly mention the exact Geography Names, Product Names, Product Families, Data Source name, Aspect names. Please make sure to change/correct to the values that you know from the context and then provide SQL Query.
                            Exmaple : User Question : "Summarize the reviews of Copilot from Chinanews"
                                We know that Chinanews in not any of the DataSource, Geography and so on.
                                So Change it to "Summarize the reviews of Copilot from Chinatechnews" as this is more relevant and faces no issues when we pull SQL Queries
                            
                            Exmaple : User Question : "Summarize the reviews of Copilot from USA"
                                We know that USA in not any of the Geography, Data Source and so on.
                                So Change it to "Summarize the reviews of Copilot from US" as this is more relevant and faces no issues in understanding
                                
                User Question  : best AI based on text Generation : By this user meant : What is the best Product Families for text Generation aspect based on net sentiment?

                Consider Product_Family Column to get different AI names.


                IMPORTANT : These are the aspects we have : 'Interface', 'Connectivity', 'Security/Privacy','Compatibility', 'Generic', 'Innovation', 'Reliability','Productivity', 'Price', 'Text Summarization/Generation','Code Generation', 'Ease of Use', 'Performance','Personalization/Customization','Accessibility','End User Usecase', 'Image Generation'.
                    But user who writes the question might not know the exact name of the aspect we have, for example : User might write "Picture Generation" for 'Image Generarion' and "writing codes" for code generation. 
                    You should be carefull while rephrasing it. 

                IMPORTANT : User can confuse a lot, but understand the question carefully and respond:
                Example : I am a story writer , Can you tell me which AI is good for writing stories based on user reviews? -> In this case, user confuses by telling that he is a story teller and all but he just needs to know "What is the best AI for Text Generation" -> Which is again decided based on comparison.
                Same goes for all the columns
                                
IMPORTANT : If a user is asking about which is best/poor, everything should be based on net sentiment and Review count. So Give it as Quantifiable and Visualization.

                User Question  : best AI based on text Generation : By this user meant : What is the best Product Families for text Generation aspect based on net sentiment?

Consider Product_Family Column to get different AI names.


                IMPORTANT: If the user is asking "Give or calculate net sentiment of Copilot, the user means the product Copilot. You should always have where condition in your query to filter for Copilot Product 
                               
                               Same goes for all the columns
                               
                               

                            1. If the user asks for count of column 'X', the query should be like this:
                                    SELECT COUNT('X')
                                    FROM Copilot_Sentiment_Data
                            2. If the user asks for count of column 'X' for different values of column 'Y', the query should be like this:
                                    SELECT 'Y', COUNT('X') AS Total_Count
                                    FROM Copilot_Sentiment_Data 
                                    GROUP BY 'Y'
                                    ORDER BY TOTAL_COUNT DESC
                            3. If the user asks for Net overall sentiment the query should be like this:
                                    SELECT ((SUM(Sentiment_Score))/(SUM(Review_Count))) * 100 AS Net_Sentiment,  SUM(Review_Count) AS Review_Count
                                    FROM Copilot_Sentiment_Data
                                    ORDER BY Net_Sentiment DESC

                            4. If the user asks for Net Sentiment for column "X", the query should be exactly like this: 

                                    SELECT X, ((SUM(Sentiment_Score)) / (SUM(Review_Count))) * 100 AS Net_Sentiment, SUM(Review_Count) AS Review_Count
                                    FROM Copilot_Sentiment_Data
                                    GROUP BY X
                                    ORDER BY Review_Count DESC


                            5. If the user asks for overall review count, the query should be like this:
                                    SELECT SUM(Review_Count) 
                                    FROM Copilot_Sentiment_Data
                            6. If the user asks for review distribution across column 'X', the query should be like this:
                                    SELECT 'X', SUM(Review_Count) * 100 / (SELECT SUM(Review_Count) FROM Copilot_Sentiment_Data) AS Review_Distribution
                                    FROM Copilot_Sentiment_Data 
                                    GROUP BY 'X'
                                    ORDER BY Review_Distribution DESC
                            7. If the user asks for column 'X' Distribution across column 'Y', the query should be like this: 
                                    SELECT 'Y', SUM('X') * 100 / (SELECT SUM('X') AS Reviews FROM Copilot_Sentiment_Data) AS Distribution_PCT
                                    FROM Copilot_Sentiment_Data 
                                    GROUP BY 'Y'
                                    ORDER BY Distribution_PCT DESC
                                    
                    IMPORTANT: If the user is asking about "ChatGPT", then consider it as "Open AI GPT"

                    Important: While generating SQL query to calculate net_sentiment across column 'X' and 'Y', if 'Y' has less distinct values, keep your response like this - SELECT 'Y','X', ((SUM(Sentiment_Score)) / (SUM(Review_Count))) * 100 AS Net_Sentiment, SUM(Review_Count) AS Review_Count FROM Copilot_Sentiment_Data GROUP BY 'Y','X'
                    
                    IMPORTANT: Always replace '=' operator with LIKE keyword.
                    IMPORTANT IMPORTANT : Always add '%' before and after filter value in LIKE OPERATOR for single or multiple WHERE conditions in the generated SQL query . Example LIKE 'Performance' should be replaced by LIKE '%Performance%'
                    
                    IMPORTANT : For example, if the SQL query is like - 'SELECT * FROM Copilot_Sentiment_Data WHERE PRODUCT='ABC' AND GEOGRAPHY='US' ORDER BY Review_Count' , you should modify the query and share the output like this - 'SELECT * FROM Copilot_Sentiment_Data WHERE PRODUCT LIKE '%ABC%' AND GEOGRAPHY LIKE '%US%' ORDER BY Review_Count'

                    Important: Always include ORDER BY clause to sort the table based on the aggregate value calculated in the query.
                    Important: Use 'LIMIT' operator instead of TOP operator.Do not use TOP OPERATOR. Follow syntax that can be used with pandasql.
                    Important: You Response should directly start from SQL query nothing else.
                    Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.
                    
                    Enhance the model’s comprehension to accurately interpret user queries by:
                      Recognizing abbreviations for country names (e.g., ‘DE’ for Germany, ‘USA’or 'usa' or 'US' for the United States of America) and expanding them to their full names for clarity.
                      Understanding product family names even when written in reverse order or missing connecting words such as HP Laptop 15, Lenovo Legion 5 15 etc
                      Utilizing context and available data columns to infer the correct meaning and respond appropriately to user queries involving variations in product family names or geographical references
                      Please provide a comprehensive Review summary, feature comparison, feature suggestions for specific product families and actionable insights that can help in product development and marketing strategies.
                      Generate acurate response only, do not provide extra information.

                Context:\n {context}?\n
                Question: \n{question}\n

                Answer:
                """
                
         
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for quantifiable review summarization: {e}"
        return err

def query_quant_classify2(user_question, vector_store_path="combine_indexes"):
    try:
        # Initialize the embeddings model
        embeddings = AzureOpenAIEmbeddings(azure_deployment="MV_Agusta")
        
        # Load the vector store with the embeddings model
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        
        # Rest of the function remains unchanged
        chain = get_conversational_chain_quant_classify2()
        docs = []
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        SQL_Query = response["output_text"]
        # st.write(SQL_Query)
        SQL_Query = convert_top_to_limit(SQL_Query)
        SQL_Query = process_tablename(SQL_Query,"Copilot_Sentiment_Data")
        #st.write(SQL_Query)
        
        #Extract required filters applied on Copilot_Sentiment_Data dataframe
        sql_list=list(SQL_Query.split('LIKE'))
        col_names=[]
        filters=[]
        try:
            if len(sql_list)>2:
                for i in range(len(sql_list)): 
                    if i==0:
                        col_names.append(sql_list[i].split(' ')[-2])
                    elif i==len(sql_list)-1:
                        pattern = r'%([^%]+)%'
                        filters.append(re.findall(pattern, sql_list[i])[0])
                    else:
                        col_names.append(sql_list[i].split(' ')[-2])
                        pattern = r'%([^%]+)%'
                        filters.append(re.findall(pattern, sql_list[i])[0])
            elif len(sql_list)==2:
                for i in range(len(sql_list)): 
                    if i==0:
                        col_names.append(sql_list[i].split(' ')[-2])
                    elif i==len(sql_list)-1:
                        pattern = r'%([^%]+)%'
                        filters.append(re.findall(pattern, sql_list[i])[0])
        except:
            pass
        
        print(SQL_Query)
        data = ps.sqldf(SQL_Query, globals())
        data_1 = data
        html_table = data.to_html(index=False)
    #     return html_table
        
        #Calculate overall net-sentiment and review_count based on filtered data
        try:
            data2=Copilot_Sentiment_Data.copy()
            data2.columns = data2.columns.str.upper()
            data2=data2.fillna('Unknown')
            if len(col_names)>0 and len(filters)>0:
                for i in range(len(col_names)):
                    data2=data2[data2[col_names[i]].str.contains(f'{filters[i]}',case=False)]
            #Add top row to quant data for overall net_sentiment and review count (if applicable)
            col_list=data_1.columns
            temp_df={}
            if 'NET_SENTIMENT' in col_list and 'REVIEW_COUNT' in col_list:
                for i in col_list:
                    if i!='NET_SENTIMENT' and i!='REVIEW_COUNT':
                        temp_df[i]=['TOTAL']
                    elif i=='NET_SENTIMENT':
                        temp_df[i]=[sum(data2['SENTIMENT_SCORE'])*100/sum(data2['REVIEW_COUNT'])]
                    elif i=='REVIEW_COUNT':
                        temp_df[i]=[sum(data2['REVIEW_COUNT'])]
            temp_df=pd.DataFrame(temp_df)
            union_df = pd.concat([temp_df, data_1], ignore_index=True)
            union_df=union_df.fillna('Unknown')
            return union_df
        except:
            pass
        return data_1    
    except Exception as e:
        
        err = f"An error occurred while generating response for Quantify: {e}"
        return err


def get_conversational_chain_detailed_summary():
    global model
    try:
        #st.write("hi-inside summary func")
        #st.write(overall_net_sentiment,overall_review_count)
        prompt_template = """

                Important: You are provided with an input dataset. Also you have an Impact column with either "HIGH" or "LOW" values.
        Your Job is to analyse the Net Sentiment, Geo-Wise wise sentiment of particular product or Product-wise sentiment and summarize the reviews that user asks, utilizing the reviews and numbers you get from the input data. Ensure maximum utility of the numbers and justify them using the reviews.
        For example, if the data you receive is Geography wise net sentiment data for a particular product-
        First give an overall summary of the data like, from which Geography most of the reviews are and which geographies have the most and least net sentiment, etc. Then, with the help of the reviews, summarize reviews from each geography and provide Pros and Cons about that Product in each Geography.
                
                IMPORTANT: For summarizing for all the rows, mention " It is Driving the overall net sentiment high" , if the value in the Impact column is "HIGH", else mention "It is Driving the overall net sentiment low"
                

                Example Template Format -

                 -Overall Net sentiment and review count
                 -Summary and insight generation
                 -Some Pros and Cons in every case
                 -Summary based on the factors that are driving the overall net sentiment high and low

                For example, Geography-wise summary for a particular product -
                Based on the provided sentiment data for Github CoPilot reviews from different geographies, here is a summary:


                        - 1st Geography: The net sentiment for reviews with unknown geography is 5.2, based on 2,212 reviews. So its driving overall net sentiment low as its net_sentiment is less than overall net_sentiment.
                        Overall summary of 1st geography: Users have highly positive reviews, praising its functionality and ease of use. They find it extremely helpful in their mobile development tasks and appreciate the regular updates and additions to the toolkit.

                            Overall summary of the Product reviews from that Geography in 5 to 6 lines
                            Give Some Pros and Cons of the Product from the reviews in this Geography

                        - 2nd Geography: The net sentiment for reviews from the United States is 8.1, based on 1,358 reviews. So its driving overall net sentiment high as its net_sentiment is greater overall net_sentiment.

                            Overall summary of the Product reviews from that Geography in 5 to 6 lines
                           Give Some Pros and Cons of the Product from the reviews in this Geography

                       - 3rd Geography: The net sentiment for reviews from Japan is 20.0, based on 165 reviews. So its driving overall net sentiment high as its net_sentiment is greater than overall net_sentiment.

                            Overall summary of the Product reviews from that Geography in 5 to 6 lines
                            Give Some Pros and Cons of the Product from the reviews in this Geography
                            

                1.Ensure to include all possible insights and findings that can be extracted, which reveals vital trends and patterns in the data
                IMPORTANT: Don't mention at the end this statement - "It is important to note that the impact of each product family on the overall net sentiment is mentioned in the dataset"
                
                IMPORTANT: If only 1 row is present in the data then don't mention anything about "driving the net sentiment high or low"
                
                2.Share the findings or insights in a format which makes more sense to business oriented users, and can generate vital action items for them. 
                
                3.AT THE END OF THE SUMMARY, ALWAYS MAKE SURE TO MENTION THE FACTORS DRIVING THE OVERALL NET SENTIMENT HIGH OR LOW
                
                4.If any recommendations are possible based on the insights, share them as well - primarily focusing on the areas of concern.
                5.For values like Net_Sentiment score, positive values indicate positive sentiment, negative values indicate negative sentiment and 0 value indicate neutral sentiment. For generating insights around net_sentiment feature, consider this information.
                
                IMPORTANT: If the maximum numerical value is less than or equal to 100, then the numerical column is indicating percentage results - therefore while referring to numbers in your insights, add % at the end of the number.
                IMPORTANT : Dont provide any prompt message or example template written here in the response, this is for your understanding purpose

                Important: Ensure to Provide the overall summary for each scenario where you are providing the net sentiment value and impact
                Important: Modify the Geography, Product Family or Product names in the prompt as per given dataset values            
                Important: Enhance the model’s comprehension to accurately interpret user queries by:
                  - Recognizing abbreviations for country names (e.g., ‘DE’ for Germany, ‘USA’or 'usa' or 'US' for the United States of America) and expanding them to their full names for clarity.
                  - Understanding product family names even when written in reverse order or missing connecting words (e.g., ‘copilot in windows 11’ as ‘copilot windows’ and ‘copilot for security’ as ‘copilot security’ etc.).
                  - Utilizing context and available data columns to infer the correct meaning and respond appropriately to user queries involving variations in product family names or geographical references]
                 Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs\n
                
                  Context:\n {context}?\n
                  Question: \n{question}\n

          Answer:
          """
            
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for detailed review summarization: {e}"
        return err

# Function to handle user queries using the existing vector store
def query_detailed_summary(dataframe_as_dict,user_question, history, vector_store_path="combine_indexes"):
    try:
        #st.write("hi")
#         st.write(user_question)
        embeddings = AzureOpenAIEmbeddings(azure_deployment="MV_Agusta")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_detailed_summary()
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        #st.write(e)
        #st.write("hi2")
        err = generate_chart_insight_llm(dataframe_as_dict)
        #st.write(e)
        return err
        
def generate_chart_insight_llm(user_question):
    global model
    try:
        prompt_template = """
        
        1.Ensure to include all possible insights and findings that can be extracted, which reveals vital trends and patterns in the data. 
        2.Share the findings or insights in a format which makes more sense to business oriented users, and can generate vital action items for them. 
        3.For values like Net_Sentiment score, positive values indicate positive overall sentiment, negative values indicate negative overall sentiment and 0 value indicate neutral overall sentiment. For generating insights around net_sentiment feature, consider this information.
        IMPORTANT: If the maximum numerical value is less than or equal to 100, then the numerical column is indicating percentage results - therefore while referring to numbers in your insights, add % at the end of the number.
        IMPORTANT : Use the data from the input only and do not give information from pre-trained data.
        IMPORTANT : Dont provide any prompt message written here in the response, this is for your understanding purpose
           
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        response = chain({"input_documents": [], "question": user_question}, return_only_outputs=True)
        st.write("\n\n",response["output_text"])
        return response["output_text"]
            
    except Exception as e:
        #st.write("inside 2nd func")
        err = "Apologies, unable to generate insights based on the provided input data. Kindly refine your search query and try again!"
        return err

def quantifiable_data(user_question):
    try:
        #st.write("correct_func")
        response = query_quant_classify2(user_question)
        
        return response
    except Exception as e:
        err = f"An error occurred while generating quantitative review summarization: {e}"
        return err
        

def generate_chart(df):
    global full_response
    # Determine the data types of the columns
#     if df.shape[0] == 1:
#         #print("hi")
#         return
    df_copy=df.copy()
    df = df[~df.applymap(lambda x: x == 'TOTAL').any(axis=1)]
    #st.write("shape of df",df)
    if df.shape[0] == 1 or (df.shape[0]==2 and (df.iloc[0:1,-1]==df.iloc[1:2,-1])):
        return
    
    
    if 'REVIEW_COUNT' in df.columns:
        df.drop('REVIEW_COUNT',axis=1, inplace=True)
        #st.write(df)
        
    try:
        df=df.drop('Impact',axis=1)
        df=df.drop('REVIEW_COUNT',axis=1)
        
    except:
        pass
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    date_cols = df.select_dtypes(include=['datetime']).columns
    
    if len(num_cols)>0:
        for i in range(len(num_cols)):
            df[num_cols[i]]=round(df[num_cols[i]],1)
            
    if len(df.columns)>3:
        try:
            cols_to_drop = [col for col in df.columns if df[col].nunique() == 1]
            df.drop(columns=cols_to_drop, inplace=True)
        except:
            pass
        
        df=df.iloc[:, :3]
        
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    date_cols = df.select_dtypes(include=['datetime']).columns
    #st.write(num_cols,cat_cols,len(num_cols),len(cat_cols))
    # Simple heuristic to determine the most suitable chart
    if len(df.columns)<=2:
        
        if len(num_cols) == 1 and len(cat_cols) == 0 and len(date_cols) == 0:

            plt.figure(figsize=(10, 6))
            sns.histplot(df[num_cols[0]], kde=True)
            plt.title(f"Frequency Distribution of '{num_cols[0]}'")
            st.pyplot(plt)
            # try:
                # chart = plt.to_html()
                # full_response += chart
            # except:
                # st.write("Error in converting chart to html")


        elif len(num_cols) == 2:
   
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df[num_cols[0]], y=df[num_cols[1]])
            plt.title(f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'")
            st.pyplot(plt)
            # try:
                # chart = plt.to_html()
                # full_response += chart
            # except:
                # st.write("Error in converting chart to html")


        elif len(cat_cols) == 1 and len(num_cols) == 1:
            if df[cat_cols[0]].nunique() <= 5 and df[num_cols[0]].sum()>=99 and df[num_cols[0]].sum()<=101:
                fig = px.pie(df, names=cat_cols[0], values=num_cols[0], title=f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'")
                st.plotly_chart(fig)
                # try:
                    # chart = fig.to_html()
                    # full_response += chart
                # except:
                    # st.write("Error in converting chart to html")

            else:
                num_categories=df[cat_cols[0]].nunique()
                width = 800
                height = max(600,num_categories*50)
                df['Color'] = df[num_cols[0]].apply(lambda x: 'grey' if x < 0 else 'blue')
                bar=px.bar(df,x=num_cols[0],y=cat_cols[0],title=f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'",text=num_cols[0],color='Color')
                bar.update_traces(textposition='outside', textfont_size=12)
                bar.update_layout(width=width, height=height)
                bar.update_layout(showlegend=False)
                st.plotly_chart(bar)
                # try:
                    # chart = bar.to_html()
                    # full_response += chart
                # except:
                    # st.write("Error in converting chart to html")


        elif len(cat_cols) == 2:

            plt.figure(figsize=(10, 6))
            sns.countplot(x=df[cat_cols[0]], hue=df[cat_cols[1]], data=df)
            plt.title(f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'")
            st.pyplot(plt)
            # try:
                # chart = plt.to_html()
                # full_response += chart
            # except:
                # st.write("Error in converting chart to html")


        elif len(date_cols) == 1 and len(num_cols) == 1:
            fig = px.line(df, x=date_cols[0], y=num_cols[0], title=f'Trend Analysis:{num_cols[0]} vs {date_cols[0]}')
            st.plotly_chart(fig)
   
#             plt.figure(figsize=(10, 6))
#             sns.lineplot(x=df[date_cols[0]], y=df[num_cols[0]], data=df)
#             plt.title(f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'")
#             st.pyplot(plt)
            # try:
                # chart = plt.to_html()
                # full_response += chart
            # except:
                # st.write("Error in converting chart to html")


        else:
            sns.pairplot(df)
            st.pyplot(plt)
            
    elif len(df.columns)==3 and len(date_cols)==1 and len(num_cols)==2:
        # Create traces
        trace1 = go.Bar(
            x=df[date_cols[0]],
            y=df[num_cols[0]],
            name=f'{num_cols[0]}',
            yaxis='y1'
        )
        
        trace2 = go.Scatter(
            x=df[date_cols[0]],
            y=df[num_cols[1]],
            name=f'{num_cols[1]}',
            yaxis='y2',
            mode='lines'
        )

        # Define layout with dual y-axis
        layout = go.Layout(
            title=f'Variation of {num_cols[1]} and {num_cols[0]} with change of {date_cols[0]}',
            xaxis=dict(title=f'{date_cols[0]}'),
            yaxis=dict(
                title=f'{num_cols[0]}',
                titlefont=dict(color='blue'),
                tickfont=dict(color='blue')
            ),
            yaxis2=dict(
                title=f'{num_cols[1]}',
                titlefont=dict(color='green'),
                tickfont=dict(color='green'),
                overlaying='y',
                side='right'
            )
        )

        # Create figure
        fig = go.Figure(data=[trace1, trace2], layout=layout)
        st.plotly_chart(fig)
        
#         line_plot = go.Scatter(x=df[date_cols[0]], y=df[num_cols[1]], mode='lines', name=f'{num_cols[1]}')
#         bar_plot = go.Bar(x=df[date_cols[0]], y=df[num_cols[0]], name=f'{num_cols[0]}')
#         fig = go.Figure(data=[line_plot, bar_plot])
#         fig.update_layout(
#             title=f'Variation of {num_cols[1]} and {num_cols[0]} with change of {date_cols[0]}',
#             xaxis_title='Date',
#             yaxis_title='Value',
#             legend=dict(x=0.1, y=1.1, orientation='h')
#         )
#         st.plotly_chart(fig)
            
    elif len(df.columns)==3 and len(cat_cols)>=1:
        
        col_types = df.dtypes

#         cat_col = None
#         num_cols = []

#         for col in df.columns:
#             if col_types[col] == 'object' and df[col].nunique() == len(df):
#                 categorical_col = col
#             elif col_types[col] in ['int64', 'float64']:
#                 num_cols.append(col)
#         st.write(cat_cols,num_cols,len(cat_cols),len(num_cols))
#         st.write(type(cat_cols))
        # Check if we have one categorical and two numerical columns
        if len(cat_cols)==1 and len(num_cols) == 2:
#             df[cat_cols[0]]=df[cat_cols[0]].astype(str)
#             df[cat_cols[0]]=df[cat_cols[0]].fillna('NA')
            
            
            if df[cat_cols[0]].nunique() <= 5 and df[num_cols[0]].sum()>=99 and df[num_cols[0]].sum()<=101:
                fig = px.pie(df, names=cat_cols[0], values=num_cols[0], color_discrete_map={'TOTAL':'Green'}, title=f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'")
                fig2 = px.pie(df, names=cat_cols[0], values=num_cols[1], title=f"Distribution of '{num_cols[1]}' across '{cat_cols[0]}'")
                st.plotly_chart(fig)
                st.plotly_chart(fig2)
                # try:
                    # chart = fig.to_html()
                    # full_response += chart
                # except:
                    # st.write("Error in converting chart to html")
                # try:
                    # chart = fig2.to_html()
                    # full_response += chart
                # except:
                    # st.write("Error in converting chart to html")
                

            else:
                num_categories=df[cat_cols[0]].nunique()
                width = 800
                height = max(600,num_categories*50)
                df['Color'] = df[num_cols[0]].apply(lambda x: 'grey' if x < 0 else 'blue')
                bar=px.bar(df,x=num_cols[0],y=cat_cols[0],title=f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'",text=num_cols[0],color='Color')
                bar.update_traces(textposition='outside', textfont_size=12)
                bar.update_layout(width=width, height=height)
                bar.update_layout(showlegend=False)
                st.plotly_chart(bar)
                
                df['Color'] = df[num_cols[1]].apply(lambda x: 'grey' if x < 0 else 'blue')
                bar2=px.bar(df,x=num_cols[1],y=cat_cols[0],title=f"Distribution of '{num_cols[1]}' across '{cat_cols[0]}'",text=num_cols[1],color='Color')
                bar2.update_traces(textposition='outside', textfont_size=12)
                bar2.update_layout(width=width, height=height)
                bar2.update_layout(showlegend=False)
                st.plotly_chart(bar2)
                # try:
                    # chart = bar.to_html()
                    # full_response += chart
                # except:
                    # st.write("Error in converting chart to html")
                # try:
                    # chart = bar2.to_html()
                    # full_response += chart
                # except:
                    # st.write("Error in converting chart to html")
                
        elif len(cat_cols)==2 and len(num_cols) == 1:
            df[cat_cols[0]]=df[cat_cols[0]].astype(str)
            df[cat_cols[1]]=df[cat_cols[1]].astype(str)
            df[cat_cols[0]]=df[cat_cols[0]].fillna('NA')
            df[cat_cols[1]]=df[cat_cols[1]].fillna('NA')
            
            list_cat=df[cat_cols[0]].unique()
            st.write("\n\n")
            for i in list_cat:
                st.markdown(f"* {i} OVERVIEW *")
                df_fltr=df[df[cat_cols[0]]==i]
                df_fltr=df_fltr.drop(cat_cols[0],axis=1)
                num_categories=df_fltr[cat_cols[1]].nunique()
#                 num_categories2=df[cat_cols[1]].nunique()
                height = 600 #max(80,num_categories2*20)
                width=800
                df_fltr['Color'] = df_fltr[num_cols[0]].apply(lambda x: 'grey' if x < 0 else 'blue')
                bar=px.bar(df_fltr,x=num_cols[0],y=cat_cols[1],title=f"Distribution of '{num_cols[0]}' across '{cat_cols[1]}'",text=num_cols[0],color='Color')
                bar.update_traces(textposition='outside', textfont_size=12)
                bar.update_layout(width=width, height=height)
                bar.update_layout(showlegend=False)
                st.plotly_chart(bar)

#------------------------------------------------------------------------------------- Generic  ---------------------------------------------------------------------------------------#

def get_conversational_chain_generic():
    global model
    global history
    try:
        prompt_template = """
        You are an AI ChatBot where you will get the data of Copilot products and its competitors like 'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI' and you should generate the response based on that Dataset only for user question by following the below instructions.
        Context Interpretation:
        1. Recognize abbreviations for country names (e.g., 'DE' for Germany, 'USA' for the United States).
        2. Understand product family names even if written in reverse order or with missing words.
        3. If the user asks to summarize attributes of a product (e.g., "Summarize the attributes of Microsoft Copilot Pro in the USA"), relate the attributes to the corresponding dataset columns and provide the response accordingly.
            For example:
            Attributes: Aspect
            Product: Product_Family
            Location: Geography
            Based on these relations, generate the summary using the relevant data from the dataset.

        Data Utilization:
        IMPORTANT: 1. Use only the provided dataset for generating responses.
        IMPORTANT: 2. Do not use or rely on pre-trained information other than Given Copilot Product Data and its Competitor Data like 'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI' which is given in Dataset. Limit Yourself to data you are provided with.
        IMPORTANT: 3. Competitors of Copilot is 'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI'. If user asks about competitor you should able to give response for competitors mentioned before.
        IMPORTANT: 4. Competitors of various versions of Copilot are : 'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI', 'Gemini AI For Mobile', 'ChatGPT For Mobile', 'Perplexity AI For Mobile', 'Claude AI For Mobile'.
        Dataset Columns:
        Review: This column contains the opinions and experiences of users regarding different product families across geographies, providing insights into customer satisfaction or complaints and areas for improvement.
        Data_Source: This column indicates the platform from which the user reviews were collected, such as Reddit, Play Store, App Store, Tech Websites, or YouTube videos.
        Geography: This column lists the countries of the users who provided the reviews, allowing for an analysis of regional preferences and perceptions of the products.
        Product_Family: This column identifies the broader category of products to which the review pertains, enabling comparisons and trend analysis across different product families like 'Windows Copilot', 'Microsoft Copilot', 'Copilot for Security', 'Copilot Pro', 'Copilot for Microsoft 365', 'Edge Copilot', 'Github Copilot', 'Copilot for Mobile', 'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI', 'Gemini AI For Mobile', 'ChatGPT For Mobile', 'Perplexity AI For Mobile', 'Claude AI For Mobile'.
        Sentiment: This column reflects the overall tone of the review, whether positive, negative, or neutral, and is crucial for gauging customer sentiment.
        Aspect: This column highlights the particular features or attributes of the product that the review discusses, pinpointing areas of strength or concern.
        
        Tasks:
        1. Review Summarization: Summarize reviews by filtering relevant Aspect, Geography, Product_Family, Sentiment, or Data_Source. Provide insights based on available reviews and sentiments.
        2. Aspect Comparison: Summarize comparisons for overlapping features between product families or geographies. Highlight key differences with positive and negative sentiments.
        3. New Feature Suggestion/Recommendation: Generate feature suggestions or improvements based on the frequency and sentiment of reviews and mentioned aspects. Provide detailed responses by analyzing review sentiment and specific aspects.
        4. Hypothetical Reviews: Create hypothetical reviews for feature updates or new features, simulating user reactions. Include realistic reviews with all types of sentiments. Provide solutions for negative hypothetical reviews.
        5. Response Criteria: Minimum of 300 words. Provide as much detail as possible. Generate accurate responses without extra information.
        
        Understanding User Queries:
        1. Carefully read and understand the full user's question.
        2. If the question is outside the scope of the dataset, respond with: "Sorry! I do not have sufficient information. Can you provide more details?"
        3. Respond accurately based on the provided Copilot Products and its Competitors like 'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI'. Following is the previous conversation from User and Response, use it to get context only:""" + str(history) + """\n
                Use the above conversation chain to gain context if the current prompt requires context from previous conversation.\n
        Context:\n {context}?\n
        Question: \n{question}\n
 
        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for detailed review summarization: {e}"
        return err
        
        
        
def query_detailed_generic(user_question, vector_store_path="combine_indexes"):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="MV_Agusta")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_generic()
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        err = f"An error occurred while getting LLM response for detailed review summarization: {e}"
        return err

#----------------------------------------------------------Split Table--------------------------------------------------------------#

def split_table(data,device_a,device_b):
    # Initialize empty lists for each product
    copilot_index = data[data["ASPECT"] == str(device_b).upper()].index[0]
    if copilot_index != 0:
        device_a_table = data.iloc[:copilot_index]
        device_b_table = data.iloc[copilot_index:]
    else:
        copilot_index = data[data["ASPECT"] == str(device_a).upper()].index[0]
        device_a_table = data.iloc[:copilot_index]
        device_b_table = data.iloc[copilot_index:]

    return device_a_table, device_b_table
    
#-----------------------------------------------------------Miscellaneous----------------------------------------------------------------#

def make_desired_df(data):
    try:
        # Create DataFrame from the dictionary
        df1 = pd.DataFrame(data)
        df = pd.DataFrame(data)
        
        # Ensure the necessary columns are present
        if 'ASPECT_SENTIMENT' not in df.columns or 'REVIEW_COUNT' not in df.columns:
            raise ValueError("Input data must contain 'ASPECT_SENTIMENT' and 'REVIEW_COUNT' columns")
        
        df = df[df['ASPECT_SENTIMENT'] != 0]
        df = df[df['ASPECT'] != 'Generic']
        df = df[df['ASPECT'] != 'TOTAL']

        # Compute min and max values for normalization
        min_sentiment = df['ASPECT_SENTIMENT'].min(skipna=True)
        max_sentiment = df['ASPECT_SENTIMENT'].max(skipna=True)
        min_review_count = df['REVIEW_COUNT'].min(skipna=True)
        max_review_count = df['REVIEW_COUNT'].max(skipna=True)

        # Apply min-max normalization for ASPECT_SENTIMENT
        df['NORMALIZED_SENTIMENT'] = df.apply(
            lambda row: (row['ASPECT_SENTIMENT'] - min_sentiment) / (max_sentiment - min_sentiment)
            if pd.notnull(row['ASPECT_SENTIMENT'])
            else None,
            axis=1
        )

        # Apply min-max normalization for REVIEW_COUNT
        df['NORMALIZED_REVIEW_COUNT'] = df.apply(
            lambda row: (row['REVIEW_COUNT'] - min_review_count) / (max_review_count - min_review_count)
            if pd.notnull(row['REVIEW_COUNT'])
            else None,
            axis=1
        )

        # Calculate the aspect ranking based on normalized values
        weight_for_sentiment = 1
        weight_for_review_count = 3
        
        df['ASPECT_RANKING'] = df.apply(
            lambda row: (weight_for_sentiment * (1 - row['NORMALIZED_SENTIMENT']) + weight_for_review_count * row['NORMALIZED_REVIEW_COUNT'])
            if pd.notnull(row['NORMALIZED_SENTIMENT']) and pd.notnull(row['NORMALIZED_REVIEW_COUNT'])
            else None,
            axis=1
        )
        # Assign integer rankings based on the 'Aspect_Ranking' score
        df['ASPECT_RANKING'] = df['ASPECT_RANKING'].rank(method='max', ascending=False, na_option='bottom').astype('Int64')

        # Sort the DataFrame based on 'Aspect_Ranking' to get the final ranking
        df_sorted = df.sort_values(by='ASPECT_RANKING')
        
        # Extract and display the net sentiment and overall review count
        try:
            total_row = df1[df1['ASPECT'] == 'TOTAL'].iloc[0]
            net_sentiment = str(int(total_row["ASPECT_SENTIMENT"])) + '%'
            overall_review_count = int(total_row["REVIEW_COUNT"])
        except (ValueError, TypeError, IndexError):
            net_sentiment = total_row["ASPECT_SENTIMENT"]
            overall_review_count = total_row["REVIEW_COUNT"]

        st.write(f"Net Sentiment: {net_sentiment}")
        st.write(f"Overall Review Count: {overall_review_count}")
        df_sorted = df_sorted.drop(columns=["NORMALIZED_SENTIMENT", "NORMALIZED_REVIEW_COUNT", "ASPECT_RANKING"])
        return df_sorted
    except Exception as e:
        df = pd.DataFrame(data)
        return df


import numpy as np

def custom_color_gradient(val, vmin=-100, vmax=100):
    green_hex = '#347c47'
    middle_hex = '#dcdcdc'
    lower_hex = '#b0343c'
    
    # Adjust the normalization to set the middle value as 0
    try:
        # Normalize the value to be between -1 and 1 with 0 as the midpoint
        normalized_val = (val - vmin) / (vmax - vmin) * 2 - 1
    except ZeroDivisionError:
        normalized_val = 0
    
    if normalized_val <= 0:
        # Interpolate between lower_hex and middle_hex for values <= 0
        r = int(np.interp(normalized_val, [-1, 0], [int(lower_hex[1:3], 16), int(middle_hex[1:3], 16)]))
        g = int(np.interp(normalized_val, [-1, 0], [int(lower_hex[3:5], 16), int(middle_hex[3:5], 16)]))
        b = int(np.interp(normalized_val, [-1, 0], [int(lower_hex[5:7], 16), int(middle_hex[5:7], 16)]))
    else:
        # Interpolate between middle_hex and green_hex for values > 0
        r = int(np.interp(normalized_val, [0, 1], [int(middle_hex[1:3], 16), int(green_hex[1:3], 16)]))
        g = int(np.interp(normalized_val, [0, 1], [int(middle_hex[3:5], 16), int(green_hex[3:5], 16)]))
        b = int(np.interp(normalized_val, [0, 1], [int(middle_hex[5:7], 16), int(green_hex[5:7], 16)]))
    
    # Convert interpolated RGB values to hex format for CSS color styling
    hex_color = f'#{r:02x}{g:02x}{b:02x}'
    
    return f'background-color: {hex_color}; color: black;'


def custom_color_gradient_compare(val, vmin=-100, vmax=100):
    green_hex = '#347c47'
    middle_hex = '#dcdcdc'
    lower_hex = '#b0343c'
    
    # Adjust the normalization to set the middle value as 0
    try:
        # Normalize the value to be between -1 and 1 with 0 as the midpoint
        normalized_val = (int(val) - vmin) / (vmax - vmin) * 2 - 1
    except:
        normalized_val = 0
    
    if normalized_val <= 0:
        # Interpolate between lower_hex and middle_hex for values <= 0
        r = int(np.interp(normalized_val, [-1, 0], [int(lower_hex[1:3], 16), int(middle_hex[1:3], 16)]))
        g = int(np.interp(normalized_val, [-1, 0], [int(lower_hex[3:5], 16), int(middle_hex[3:5], 16)]))
        b = int(np.interp(normalized_val, [-1, 0], [int(lower_hex[5:7], 16), int(middle_hex[5:7], 16)]))
    else:
        # Interpolate between middle_hex and green_hex for values > 0
        r = int(np.interp(normalized_val, [0, 1], [int(middle_hex[1:3], 16), int(green_hex[1:3], 16)]))
        g = int(np.interp(normalized_val, [0, 1], [int(middle_hex[3:5], 16), int(green_hex[3:5], 16)]))
        b = int(np.interp(normalized_val, [0, 1], [int(middle_hex[5:7], 16), int(green_hex[5:7], 16)]))
    
    # Convert interpolated RGB values to hex format for CSS color styling
    hex_color = f'#{r:02x}{g:02x}{b:02x}'
    
    return f'background-color: {hex_color}; color: black;'


# In[11]:




def get_final_df(aspects_list,device):
    final_df = pd.DataFrame()
    device = device
    aspects_list = aspects_list

    # Iterate over each aspect and execute the query
    for aspect in aspects_list:
        # Construct the SQL query for the current aspect
        query = f"""
        SELECT Keywords,
               COUNT(CASE WHEN Sentiment = 'positive' THEN 1 END) AS Positive_Count,
               COUNT(CASE WHEN Sentiment = 'negative' THEN 1 END) AS Negative_Count,
               COUNT(CASE WHEN Sentiment = 'neutral' THEN 1 END) AS Neutral_Count,
               COUNT(*) as Total_Count
        FROM Copilot_Sentiment_Data
        WHERE Aspect = '{aspect}' AND Product_Family LIKE '%{device}%'
        GROUP BY Keywords
        ORDER BY Total_Count DESC;
        """

        # Execute the query and get the result in 'key_df'
        key_df = ps.sqldf(query, globals())

        # Calculate percentages and keyword contribution
        total_aspect_count = key_df['Total_Count'].sum()
        key_df['Positive_Percentage'] = (key_df['Positive_Count'] / total_aspect_count) * 100
        key_df['Negative_Percentage'] = (key_df['Negative_Count'] / total_aspect_count) * 100
        key_df['Neutral_Percentage'] = (key_df['Neutral_Count'] / total_aspect_count) * 100
        key_df['Keyword_Contribution'] = (key_df['Total_Count'] / total_aspect_count) * 100

        # Drop the count columns
        key_df = key_df.drop(['Positive_Count', 'Negative_Count', 'Neutral_Count', 'Total_Count'], axis=1)

        # Add the current aspect to the DataFrame
        key_df['Aspect'] = aspect

        # Sort by 'Keyword_Contribution' and select the top 2 for the current aspect
        key_df = key_df.sort_values(by='Keyword_Contribution', ascending=False).head(2)

        # Append the results to the final DataFrame
        final_df = pd.concat([final_df, key_df], ignore_index=True)
        
    return final_df
    
#-----------------------------------------------Classify Flow---------------------------------------------------#

def classify(user_question):
    global model
    try:
        prompt_template = """
        
            Given an input, classify it into one of two categories:
            
            ProductFamilies = 'Windows Copilot', 'Microsoft Copilot', 'Copilot for Security', 'Copilot Pro', 'Copilot for Microsoft 365', 'Edge Copilot', 'Github Copilot', 'Copilot for Mobile', 'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI', 'Gemini AI For Mobile', 'ChatGPT For Mobile', 'Perplexity AI For Mobile', 'Claude AI For Mobile'. 
            
            1stFlow: [Summarization of any Product Family or Product Family Name]. This flow is just for summarization of reviews of only one product/product Family name.
                    Choose 1st flow, if the user seeks for summarization of only one product choose this flow.
                    Eg: "Summarize reviews of copilot mobile in USA", "Summarize reviews of copilot mobile in USA"
                        
            
            
            2ndFlow: User is seeking any other information like geography wise performance or any quantitative numbers like what is net sentiment for different product families then categorize as 2ndFlow. It should even choose 2nd flow, if it asks for Aspect wise sentiment of one Product.
            
            Example - Geography wise how products are performing or seeking for information across different product families/products.
            What is net sentiment for any particular product/geography
            
        IMPORTANT : Only share the classified category name, no other extra words.
        IMPORTANT : Don't categorize into 1stFlow or 2ndFlow based on number of products, categorize based on the type of question the user is asking
        Input: User Question
        Output: Category (1stFlow or 2ndFlow)
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        response = chain({"input_documents": [], "question": user_question}, return_only_outputs=True)
        if "1stflow" in response["output_text"].lower():
            return "1"
        elif "2ndflow" in response["output_text"].lower():
            return "2"
        else:
            return "Others"+"\nPrompt Identified as:"+response["output_text"]+"\n"
    except Exception as e:
        err = f"An error occurred while generating conversation chain for identifying nature of prompt: {e}"
        return err
        

from openai import AzureOpenAI

# client = AzureOpenAI(
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
#     api_version="2024-02-01",
#     azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
#     )
    
deployment_name='Surface_Analytics'

context_Prompt = """


You are an AI Chatbot assistant. Carefully understand the user's question and follow these instructions to categorize their query into one of four features.

Whatever user is asking, it is purely based on consumer reviews.

So when user asks which is the best/poor, like/dislike everything is based on user reviews. For understanding the whole thing, we have 4 different features.

User can confuse a lot, but understand the question carefully and respond

Features:
    Summarization, Quantifiable and Visualization, Comparison, Generic

Instructions:


-There is only one table with table name Copilot_Sentiment_Data where each row is a user review, using that data we developed these functionalities. The table has 10 columns, they are:
                Review: Review of the Copilot Product
                Data_Source: From where is the review taken. It contains different retailers - It contains following values : [chinatechnews, DigitalTrends, Engadget, clubic, g2.com, gartner, JP-ASCII, Jp-Impresswatch, Itmedia, LaptopMag, NotebookCheck, PCMag, TechAdvisor, TechRadar, TomsHardware, TechCrunch, Verge, ZDNET, PlayStore, App Store, AppStore, Reddit, YouTube, Facebook, Instagram, X, VK, Forums, News, Print, Blogs/Websites, Reviews, Wordpress, Podcast, TV, Quora, LinkedIn, Videos]
                Geography: From which Country or Region the review was given. It contains different Geography.
                           list of Geographies in the table - Values in this column [China,France,Japan,US,Brazil,Canada,Germany,India,Mexico,UK,Australia,Unknown,Venezuela,Vietnam,Cuba,Colombia,Iran,Ukraine,Northern Mariana Islands,Uruguay,Taiwan,Spain,Russia,Bolivia,Argentina,Lebanon,Finland,Saudi Arabia,Oman,United Arab Emirates,Austria,Luxembourg,Macedonia,Puerto Rico,Bulgaria,Qatar,Belgium,Italy,Switzerland,Peru,Czech Republic,Thailand,Greece,Netherlands,Romania,Indonesia,Benin,Sweden,South Korea,Poland,Portugal,Tonga,Norway,Denmark,Samoa,Ireland,Turkey,Ecuador,Guernsey,Botswana,Kenya,Chad,Bangladesh,Nigeria,Singapore,Malaysia,Malawi,Georgia,Hong Kong,Philippines,South Africa,Jordan,New Zealand,Pakistan,Nepal,Jamaica,Egypt,Macao,Bahrain,Tanzania,Zimbabwe,Serbia,Estonia,Jersey,Afghanistan,Kuwait,Tunisia,Israel,Slovakia,Panama,British Indian Ocean Territory,Comoros,Kazakhstan,Maldives,Kosovo,Ghana,Costa Rica,Belarus,Sri Lanka,Cameroon,San Marino,Antigua and Barbuda]
                Title: What is the title of the review
                Review_Date: The date on which the review was posted
                Product: Corresponding product for the review. It contains following values: ['COPILOT','OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI']
                Product_Family: Which version or type of the corresponding Product was the review posted for. Different Device Names  - It contains following Values - ['Windows Copilot', 'Microsoft Copilot', 'Copilot for Security', 'Copilot Pro', 'Copilot for Microsoft 365', 'Edge Copilot', 'Github Copilot', 'Copilot for Mobile', 'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI', 'Gemini AI For Mobile', 'ChatGPT For Mobile', 'Perplexity AI For Mobile', 'Claude AI For Mobile']
                
                IMPORTANT : REMEMBER THAT ALL THE NAMES IN PRODUCT_FAMILIES ARE NOT DIFFERENT VERSION OF COPILOT, THERE ARE SOME COMPETITORS AS WELL.
                IMPORTANT : Different Product families/versions of copilot is 'Windows Copilot', 'Microsoft Copilot', 'Copilot for Security', 'Copilot Pro', 'Copilot for Microsoft 365', 'Github Copilot', 'Copilot for Mobile'.
                IMPORTANT : Competitors of various versions of Copilot are : 'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI', 'Gemini AI For Mobile', 'ChatGPT For Mobile', 'Perplexity AI For Mobile', 'Claude AI For Mobile'.
                IMPORTANT : Competitors of Copilot is 'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI'
            
            
                Sentiment: What is the sentiment of the review. It contains following values: 'positive', 'neutral', 'negative'.
                Aspect: The review is talking about which aspect or feature of the product. It contains following values: 'Interface', 'Connectivity', 'Security/Privacy','Compatibility', 'Generic', 'Innovation', 'Reliability','Productivity', 'Price', 'Text Summarization/Generation','Code Generation', 'Ease of Use', 'Performance','Personalization/Customization','Accessibility', 'Image Generation', 'End User Usecase'
                Keywords: What are the keywords mentioned in the product
                Review_Count - It will be 1 for each review or each row
                Sentiment_Score - It will be 1, 0 or -1 based on the Sentiment.

        IMPORTANT : User won't exactly mention the Geography Names, Product Names, Product Families, Data Source name, Aspect names. Please make sure to change/correct to the values that you know from the context and then provide SQL Query.
        Exmaple : User Question : "Summarize the reviews of Copilot from Chinanews"
                We know that Chinanews in not any of the DataSource, Geography and so on.
                So Change it to "Summarize the reviews of Copilot from Chinatechnews" as this is more relevant and faces no issues in understanding

        Exmaple : User Question : "Summarize the reviews of Copilot from USA"
                We know that USA in not any of the Geography, Data Source and so on.
                So Change it to "Summarize the reviews of Copilot from US" as this is more relevant and faces no issues in understanding
                
For which is the best and which is the worst product kind of Questions, Choose Comparison as it compares different Products.

Summarization:
        -Summarizes reviews for a specific Product Family.
        -Analyse reviews of any Product Family.
        -Overall summary of Product Family
        -Choose this if the user asks for a summary of aspects for one Product Family.
        -Do not choose this for general pros and cons or top verbatims.
        -If user mention 2 devices for summarization, Go with comparision
        -If user mention 3 devices for summarization, Go with Generic
        -Don't select this, if the question ask to summarize reviews for a feature/Aspect.
        
Quantifiable and Visualization:
       - Provides data retrieval and visualization for any Product Family.
       - Choose this for queries related to net sentiment, aspect sentiment, and review count.
       - Examples:
            "What is the net sentiment of [Product Family]?"
            "Give me the aspect sentiment of [Product Family]."
            "Which Product Family has the highest review count?" .
            "Net sentiment of [Product Family 1], [Product Family 2], [Product Family 3],..etc.?
        - Important - Compare aspect wise net sentiment of different Product Families or different copilot version - In this case, it should not choose this function.
       - Do not choose this for general questions.
       - Whenver user asks about top 20 aspects, keywords choose this function
       
Comparison:
        Basically compares the aspect wise sentiment/aspect wise net sentiment of 2 or more Product Families.
            Eg: Compare aspect wise net sentiment of different Product Families or different copilot version - In this case, it should choose this function.
        - Compares different Product Families/Product based on user reviews.
       Examples:
        - "Compare [Product Family 1] and [Product Family 2] on performance."
        - Compare [Product 1], [Product 2] - In this case, it should choose this function
        - IMPORTANT : Compare [Product 1], [Product 2], [Product 3], ...
        - IMPORTANT : Compare aspect wise net sentiment of different Product Families or different copilot version - In this case, it should choose this function.
        -Compare different versions of copilot - In this case, it should choose this function.
        - Compare Copilot with its competitors
        - Compare [Product Family] in different geographies
        - Compare different Product Families in different Geographies.
        - Which aspect is better in Product Family 1 than Product Family 2
        - Which Product is better for any feature.
        -What aspects are better in copilot than its competitors?
        -Which AI user feels is the best for [feature]?
        -Best/worst, good/bad, user like/user dislikes, praises/curses everything should be decided based on comparision.
        - Compare the net sentiment of keyword in end user usercase of different Product Families.

Generic:

        -For general questions about any Product Family.
        -Choose this for queries about pros and cons, common complaints, and top verbatims.
        -Also, choose this if the question involves more than two Product Families.
        -Choose this, if the question ask to summarize reviews for a Apect/feature of any Product or multiple products.
        -Examples:
            -What do people think about the pricing of [Product Family] in the US?
            - What do people talk about any feature/aspect of a product?
            - IMPORTANT : When user seeks for more deatils/ more information from previous response, choose Generic.
        -WHat are people talking about price of Github copilot.
        - "What are users view", "What are people talking" all these kind of questions falls under this Generic category.

IMPORTANT : User can confuse a lot, but understand the question carefully and respond:
            Example : I am a story writer , Can you tell me which AI is good for writing stories based on user reviews? -> In this case, user confuses by telling that he is a story teller and all but he just needs to know "What is the best AI for Text Generation" -> Which is again decided based on comparison.


Important Notes:

    -Generic should be chosen for any query not specific to net sentiment, aspect sentiment
    - Choose comparision if user asks about which product is liked the most/which is the best Product and those kind of Questions

Your response should be one of the following:

“Summarization”
“Quantifiable and Visualization”
“Comparison”
“Generic”

Following is the previous conversation from User and Response, use it to get context only:""" + str(history) + """\n
Use the above conversation chain to gain context if the current prompt requires context from previous conversation.\n. 


VERY IMPORTANT : When user asks uses references like "from previous response", ":from above response" or "from above", Please refer the previous conversation and respond accordingly.
"""

def classify_prompts(user_question):
    global context_Prompt
    # Append the new question to the context
    full_prompt = context_Prompt + "\nQuestion:\n" + user_question + "\nAnswer:"
    # Send the query to Azure OpenAI
    response = client.completions.create(
        model=deployment_name,
        prompt=full_prompt,
        max_tokens=500,
        temperature=0
    )
    
    # Extract the generated SQL query
    user_query = response.choices[0].text.strip()
    
    # Update context with the latest interaction
    context_Prompt += "\nQuestion:\n" + user_question + "\nAnswer:\n" + user_query
    
    return user_query   
        
        
#-----------------------------------------------------------------------------------------Comparision------------------------------------------------------------------------------------#


def get_conversational_chain_detailed_compare():
    global model
    global history
    try:
        prompt_template = """
        
            IMPORTANT: Use only the data provided to you and do not rely on pre-trained documents.
            
        Product = Microsoft Copilot, Copilot in Windows 11, Copilot, Github Copilot , Copilot for Security, Copilot Pro, Copilot for Microsoft 365, Copilot for Mobile (These are Microsoft Copilot Product) and their competitor includes : Competitors of Copilot is 'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI', 'Gemini AI For Mobile', 'ChatGPT For Mobile', 'Perplexity AI For Mobile', 'Claude AI For Mobile'.      
        1. Your Job is to Summarize the user reviews and sentiment data you get as an input for 2 or more Product that user mentioned.
        
        IMPORTANT : Do not just give summary of numbers, Give Pros and cons from reviews for wach aspect
        
        IMPORTANT : Mention their Positive and Negative of each Product for each aspects (What consumer feels) for each aspect.
        
        Example :
        
        Compare X,Y , Z and so on :
        
        
        VERY IMPORTANT : KEEP THE TEMPLATE SAME FOR ALL THE COMPARISION RELATED USER QUESTIONS.
            
            Summary of X:
            
            Positive:


            List down all the pros of X in pointers
            
            
            Negative:
            
            List down all the cons of X in pointers
            
            
            Summary of Y:

            Positive:

            List down all the Pros of Y in pointers

            
            Negative:
            
            List down all the cons of Y in pointers
            
            Summary of Z:

            Positive:

            List down all the Pros of Z in pointers

            
            Negative:
            
            List down all the cons of Z in pointers
            
            and so on for the rest of the Products.


            Overall, both X and Y have received positive feedback for their respective aspects However, there are areas for improvement such as (Whatever aspects that needs improvement) . Users appreciate (Aspects that have good net sentiment)
            
            
            If there are 3 or more Product, follow the same template and provide pros and cons for the others as well.
        
        Give a detailed summary for each aspects using the reviews. Use maximum use of the reviews. Do not use your pretrained data. Use the data provided to you. For each aspects. Summary should be 3 ro 4 lines
        
        
        VERY IMPORTANT : If user mentioned different Product Families/Copilot versions , Give Pros and cons that user mentioned for all the aspects for each Product Families.

                    
          Enhance the model’s comprehension to accurately interpret user queries by:
          Recognizing abbreviations for country names (e.g., ‘DE’ for Germany, ‘USA’or 'usa' or 'US' for the United States of America) and expanding them to their full names for clarity.
          Understanding product family names even when written in reverse order or missing connecting words (e.g., ‘copilot in windows 11’ as ‘copilot windows’ and ‘copilot for security’ as ‘copilot security’ etc.).
          Utilizing context and available data columns to infer the correct meaning and respond appropriately to user queries involving variations in product family names or geographical references
          Please provide a comprehensive Review summary, feature comparison, feature suggestions for specific product families and actionable insights that can help in product development and marketing strategies.
          Generate acurate response only, do not provide extra information.
            
            Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.
        Context:\n {context}?\n
        Question: \n{question}\n
 
        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for detailed review summarization: {e}"
        return err

# Function to handle user queries using the existing vector store
def query_detailed_compare(user_question, vector_store_path="combine_indexes"):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="MV_Agusta")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_detailed_compare()
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        err = f"An error occurred while getting LLM response for detailed review summarization: {e}"
        return err
        
        
def check_history_length(a,last_response):
    if len(a) > 3:
        a.pop(0)
    else:
        a.append(last_response)
    return a
        
#------------------------------------------------------Rephrase Prompts------------------------------------------------#

from openai import AzureOpenAI

# client = AzureOpenAI(
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
#     api_version="2024-02-01",
#     azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
#     )
    
deployment_name='Surface_Analytics'

rephrase_Prompt = """
 
 
You are an AI Chatbot assistant. Carefully understand the user's question and follow these instructions.
 
Your Job is to correct the user Question if user have misspelled any device names, Geographies, DataSource names... etc
Device Family Names, Geographies and DataSources names are clearly mentioned below
 
Below are the available column names and values from the Copilot sentiment data:
 
-There is only one table with table name Copilot_Sentiment_Data where each row is a user review, using that data we developed these functionalities. The table has 10 columns, they are:
                Review: Review of the Copilot Product
                Data_Source: From where is the review taken. It contains different retailers - It contains following values : [chinatechnews, DigitalTrends, Engadget, clubic, g2.com, gartner, JP-ASCII, Jp-Impresswatch, Itmedia, LaptopMag, NotebookCheck, PCMag, TechAdvisor, TechRadar, TomsHardware, TechCrunch, Verge, ZDNET, PlayStore, App Store, AppStore, Reddit, YouTube, Facebook, Instagram, X, VK, Forums, News, Print, Blogs/Websites, Reviews, Wordpress, Podcast, TV, Quora, LinkedIn, Videos]
                Geography: From which Country or Region the review was given. It contains different Geography.
                           list of Geographies in the table - Values in this column [China,France,Japan,US,Brazil,Canada,Germany,India,Mexico,UK,Australia,Unknown,Venezuela,Vietnam,Cuba,Colombia,Iran,Ukraine,Northern Mariana Islands,Uruguay,Taiwan,Spain,Russia,Bolivia,Argentina,Lebanon,Finland,Saudi Arabia,Oman,United Arab Emirates,Austria,Luxembourg,Macedonia,Puerto Rico,Bulgaria,Qatar,Belgium,Italy,Switzerland,Peru,Czech Republic,Thailand,Greece,Netherlands,Romania,Indonesia,Benin,Sweden,South Korea,Poland,Portugal,Tonga,Norway,Denmark,Samoa,Ireland,Turkey,Ecuador,Guernsey,Botswana,Kenya,Chad,Bangladesh,Nigeria,Singapore,Malaysia,Malawi,Georgia,Hong Kong,Philippines,South Africa,Jordan,New Zealand,Pakistan,Nepal,Jamaica,Egypt,Macao,Bahrain,Tanzania,Zimbabwe,Serbia,Estonia,Jersey,Afghanistan,Kuwait,Tunisia,Israel,Slovakia,Panama,British Indian Ocean Territory,Comoros,Kazakhstan,Maldives,Kosovo,Ghana,Costa Rica,Belarus,Sri Lanka,Cameroon,San Marino,Antigua and Barbuda]
                Title: What is the title of the review
                Review_Date: The date on which the review was posted
                Product: Corresponding product for the review. 
                    It contains following values: 'COPILOT', 'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI'
                Product_Family is one level deeper than Product. It will have different versions of Copilot and its competitors
                    Product_Family: Which version or type of the corresponding Product was the review posted for. 
                    Different product family Names  - It contains following Values - 'Windows Copilot', 'Microsoft Copilot', 'Copilot for Security', 'Copilot Pro', 'Copilot for Microsoft 365', 'Edge Copilot', 'Github Copilot', 'Copilot for Mobile', 'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI', 'Gemini AI For Mobile', 'ChatGPT For Mobile', 'Perplexity AI For Mobile', 'Claude AI For Mobile'.            
                IMPORTANT : Out of these Product Family Names, it can be segregated into 2 things : One is Different versions of Copilot like [Microsoft Copilot, Windows CoPilot, Copilot, Github Copilot , Copilot for Security, Copilot Pro, Copilot for Microsoft 365, Copilot for Mobile] and the other ones are Competitors of copilot like ['OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI', 'Gemini AI For Mobile', 'ChatGPT For Mobile', 'Perplexity AI For Mobile', 'Claude AI For Mobile']
                
                So, whenever user asks for compare different versions of copilot, the user meant compare Different versions of Copilot like [Microsoft Copilot, Windows CoPilot, Copilot, Github Copilot , Copilot for Security, Copilot Pro, Copilot for Microsoft 365, Copilot for Mobile]
                
                and whenever user asks for compare copilot and its competitors, the user meant compare 'Copilot','OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI' - Note : This is from Product column and not from Product_Family
                
                and whenever user asks for compare different versions of copilot, the user meant compare  'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI', 'Gemini AI For Mobile', 'ChatGPT For Mobile', 'Perplexity AI For Mobile', 'Claude AI For Mobile'. - Note : These are from Product_Family Column not from Product 
                
                Sentiment: What is the sentiment of the review. It contains following values: 'positive', 'neutral', 'negative'.
                Aspect: The review is talking about which aspect or feature of the product. It contains following values: 'Interface', 'Connectivity', 'Privacy','Compatibility', 'Generic', 'Innovation', 'Reliability','Productivity', 'Price', 'Text Summarization/Generation','Code Generation', 'Ease of Use', 'Performance','Personalization/Customization','Accessibility','End User Usecase'.
                Keywords: What are the keywords mentioned in the product
                Review_Count - It will be 1 for each review or each row
                Sentiment_Score - It will be 1, 0 or -1 based on the Sentiment.
 
        IMPORTANT : User won't exactly mention the Geography Names, Product Names, Product Families, Data Source name, Aspect names. Please make sure to change/correct to the values that you know from the context
        Exmaple : User Question : "Summarize the reviews of Copilot from Chinanews"
                We know that Chinanews in not any of the DataSource, Geography and so on.
                So Change it to "Summarize the reviews of Copilot from Chinatechnews" as this is more relevant and faces no issues in understanding
 
        Exmaple : User Question : "Summarize the reviews of Copilot from USA"
                We know that USA in not any of the Geography, Data Source and so on.
                So Change it to "Summarize the reviews of Copilot from US" as this is more relevant and faces no issues in understanding
                Exmaple : User Question : "Summarize the reviews of mobile copilot from USA"
                We know that USA in not any of the Geography, Data Source and so on. and mobile copilot is not in Product Family, Geography and so on.
                So Change it to "Summarize the reviews of Copilot for Mobile from US"
        IMPORTANT : if user Just mentioned Copilot -> It is 'Copilot' and its competitors, user seeks information related to Product column not from Product Family column 
        IMPORTANT : if user mentions, "Compare Copilot with Gemini" -> User meant "Compare Copilot with Gemini AI" and it is not "Compare Microsoft Copilot with Gemini AI"
        
        
        
        Note : These are the different version of [Microsoft Copilot, Windows CoPilot, Copilot, Github Copilot , Copilot for Security, Copilot Pro, Copilot for Microsoft 365, Copilot for Mobile]
        Note : These are the competitors of copilot ['OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI']"
        Note : Product column contains : ['COPILOT','OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI' ]
        IF user is asking about AI in whole, use all the Product Family name.
        
        IMPORTANT : These are the aspects we have : 'Interface', 'Connectivity', 'Privacy','Compatibility', 'Generic', 'Innovation', 'Reliability','Productivity', 'Price', 'Text Summarization/Generation','Code Generation', 'Ease of Use', 'Performance','Personalization/Customization','Accessibility','End User Usecase', 'Image Generation'.
        But user who writes the question might not know the exact name of the aspect we have, for example : User might write "Picture Generation" for 'Image Generarion' and "writing codes" for code generation. 
        You should be carefull while rephrasing it.
        
        IMPORTANT : User can confuse a lot, but understand the question carefully and respond:
        Example : I am a story writer , Can you tell me which AI is good for writing stories based on user reviews? -> In this case, user confuses by telling that he is a story teller and all but he just needs to know "What is the best AI for Text Generation" -> Which is again decided based on comparison.
        
        Mapping file for Products:

 
        Copilot in Windows 11 -> Windows Copilot
        Copilot for Security -> Copilot for Security
        Copilot Pro -> Copilot Pro
        Microsoft Copilot -> Microsoft Copilot
        Copilot for Microsoft 365 -> Copilot for Microsoft 365
        Github Copilot -> Github Copilot
        Copilot for Mobile -> Copilot for Mobile
        Windows Copilot -> Windows Copilot
        Copilot for Windows -> Windows Copilot
        Copilot Windows -> Windows Copilot
        Win Copilot -> Windows Copilot
        Security Copilot -> Copilot for Security
        Privacy Copilot -> Copilot for Security
        M365 -> Copilot for Microsoft 365
        Microsoft 365 -> Copilot for Microsoft 365
        Office copilot -> Copilot for Microsoft 365
        Github -> Github Copilot
        MS Office -> Copilot for Microsoft 365
        MSOffice -> Copilot for Microsoft 365
        Microsoft Office -> Copilot for Microsoft 365
        Office Product -> Copilot for Microsoft 365
        Copilot Mobile -> Copilot for Mobile
        ChatGPT Mobile -> ChatGPT For Mobile
        Perplexity Mobile -> Perplexity AI For Mobile
        Claude Mobile -> Claude AI For Mobile
        Google Mobile -> Gemini AI For Mobile
        Gemini Mobile -> Gemini AI For Mobile
        App -> Copilot for Mobile
        ios -> Copilot for Mobile
        apk -> Copilot for Mobile
        Copilot -> Microsoft Copilot
        OpenAI GPT -> OpenAI GPT
        chatgpt -> OpenAI GPT
        GPT -> OpenAI GPT
        OpenAI -> OpenAI GPT
        Gemini AI -> Gemini AI
        Gemini -> Gemini AI
        Claude AI -> Claude AI
        Claude -> Claude AI
        Bard -> Gemini AI
        Google AI -> Gemini AI
        Vertex AI -> Vertex AI
        Vertex -> Vertex AI
        Google Vertex -> Vertex AI
        Perplexity AI -> Perplexity AIcom
        Perplexity -> Perplexity AI
        AI -> Microsoft Copilot, Windows CoPilot, Copilot, Github Copilot , Copilot for Security, Copilot Pro, Copilot for Microsoft 365, Copilot for Mobile, 'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI', 'Gemini AI For Mobile', 'ChatGPT For Mobile', 'Perplexity AI For Mobile', 'Claude AI For Mobile'
 
IMPORTANT: If the input sentence mentions a device(Laptop or Desktop) instead of Copilot, keep the device name as it is.
 
"""

def rephrase_prompt(user_question):
    global rephrase_Prompt
    # Append the new question to the context
    full_prompt = rephrase_Prompt + "\nQuestion:\n" + user_question + "\nAnswer:"
    # Send the query to Azure OpenAI
    response = client.completions.create(
        model=deployment_name,
        prompt=full_prompt,
        max_tokens=500,
        temperature=0
    )
    
    # Extract the generated SQL query
    user_query = response.choices[0].text.strip()
    
    # Update context with the latest interaction
    rephrase_Prompt += "\nQuestion:\n" + user_question + "\nAnswer:\n" + user_query
    
    return user_query   
    
    
def identify_columns(data):
    a = data.columns
    if 'ASPECT' in a:
        return 'ASPECT'
    elif 'GEOGRAPHY' in a:
        return 'GEOGRAPHY'
    elif 'KEYWORDS' in a:
        return 'KEYWORDS'
    else:
        return None
    
#--------------------------------------------------------Main Function-------------------------------------------------#


#--------------------------------------------------------Main Function-------------------------------------------------#


def user_ques(user_question_1, user_question, classification, user_question_chart):
    global full_response
    global history
    if user_question_1:
        device_list = Copilot_Sentiment_Data['Product_Family'].to_list()
        sorted_device_list_desc = sorted(device_list, key=lambda x: len(x), reverse=True)

    # Convert user question and product family names to lowercase for case-insensitive comparison
        user_question_lower = user_question_1.lower()

        # Initialize variables for device names
        device_a = None
        device_b = None

        # Search for product family names in the user question
        for device in sorted_device_list_desc:
            if device.lower() in user_question_lower:
                if device_a is None and device != 'Copilot':
                    device_a = device
                else:
                    if device_a != device and device != 'Copilot':
                        device_b = device
                        break# Found both devices, exit the loop

        # st.write(device_a)
        # st.write(device_b)

        if classification == "Comparison":
            try:
                data = query_quant_classify2_compare(user_question_1)
                # st.dataframe(data)
                data_formatted = data
                print(data_formatted)
                column_found = identify_columns(data_formatted)
                print(column_found)
                dataframe_as_dict = data_formatted.to_dict(orient='records')
               
                try:
                    try:
                        data_formatted['REVIEW_COUNT'] = data_formatted['REVIEW_COUNT'].astype(str).str.replace(',', '').astype(float).astype(int)      
                        aspect_volume = data_formatted.groupby(column_found)['REVIEW_COUNT'].sum().reset_index()
                    except:
                        pass
                        
                    try:
                    # Merge the total volume back to the original dataframe
                        data_formatted = data_formatted.merge(aspect_volume, on=column_found, suffixes=('', '_Total'))

                        # Sort the dataframe based on the total volume
                        data_formatted = data_formatted.sort_values(by='REVIEW_COUNT_Total', ascending=False)
                    except:
                        pass

                    # Pivot the DataFrame
                    try:
                        pivot_df = data_formatted.pivot_table(index=column_found, columns='PRODUCT_FAMILY', values='NET_SENTIMENT_'+str(column_found))
                        pivot_df = pivot_df.reindex(data_formatted[column_found].drop_duplicates().values)
                        print("In 1st Case")
                    except Exception as e:
                        try:
                            pivot_df = data_formatted.pivot_table(index=column_found, columns='PRODUCT', values='NET_SENTIMENT_'+str(column_found))
                            pivot_df = pivot_df.reindex(data_formatted[column_found].drop_duplicates().values)
                            print("In 2nd Case")
                        except:
                            try:
                                pivot_df = data_formatted
                                pivot_df = pivot_df.drop(columns=['REVIEW_COUNT', 'REVIEW_COUNT_Total'])
                                print("In 3rd Case")
                            except:
                                try:
                                    pivot_df = data_formatted
                                    pivot_df = pivot_df.drop(columns=['REVIEW_COUNT'])
                                    print("In 4th Case")
                                except:
                                    pivot_df = data_formatted
                    
                    dataframe_as_dict = pivot_df.to_dict(orient='records')
                    try:
                        pivot_df = pivot_df.astype(int)
                    except:
                        pass
                    styled_df_comaparison = pivot_df.style.applymap(custom_color_gradient_compare)
                    styled_df_comaparison_new = styled_df_comaparison.set_properties(**{'text-align': 'center'})
                    st.dataframe(styled_df_comaparison_new)
                    styled_df_comaparison_html = styled_df_comaparison.to_html(index=False)
                    full_response += styled_df_comaparison_html
                except Exception as e:
                    print(e)
                    pass
                comparision_summary = query_detailed_compare(user_question + "Which have the following sentiment data" + str(dataframe_as_dict))
                st.write(comparision_summary)
                full_response += comparision_summary
                history = check_history_length(history,comparision_summary)
                save_list(history) 
            except:
                try:
                    comparision_summary = query_detailed_compare(user_question + "Which have the following sentiment data" + str(dataframe_as_dict))
                    st.write(comparision_summary)
                    full_response += comparision_summary
                    history = check_history_length(history,comparision_summary)
                    save_list(history)
                except:
                    error = "Unable to fetch relevant details based on the provided input. Kindly refine your search query and try again!"
                    st.write(error)   


        elif (device_a != None and device_b == None) | (device_a == None and device_b == None) | (device_a != None and device_b != None):
        
            try:
                try:
                    data = query_quant(user_question_1,[])
                    print(data)
                except:
                    pass
                try:
                    total_reviews = data.loc[data.iloc[:, 0] == 'TOTAL', 'REVIEW_COUNT'].iloc[0]
                except:
                    pass
                # total_reviews = data.loc[data['ASPECT'] == 'TOTAL', 'REVIEW_COUNT'].iloc[0]
                try:
                    data['REVIEW_PERCENTAGE'] = data['REVIEW_COUNT'] / total_reviews * 100
                except:
                    pass
                try:
                    dataframe_as_dict = data.to_dict(orient='records')
                except:
                    pass

                # classify_function = classify(user_question_1+str(dataframe_as_dict))
                if classification == "Summarization":
                    classify_function = "1"
                else:
                    classify_function = "2"

                if classify_function == "1":
                    device = device_a
                    print(device)
                    data_new = data
                    data_new = data_new.dropna(subset=['ASPECT_SENTIMENT'])
                    data_new = data_new[~data_new["ASPECT"].isin([ "Generic", "Account", "Customer-Service", "Browser"])]   
                    dataframe_as_dict = data_new.to_dict(orient='records')
                    data_new = make_desired_df(data_new)
                    styled_df = data_new.style.applymap(lambda x: custom_color_gradient(x, int(-100), int(100)), subset=['ASPECT_SENTIMENT'])
                    data_filtered = data_new[(data_new['ASPECT'] != 'TOTAL') & (data_new['ASPECT'] != 'Generic')]
                    top_four_aspects = data_filtered.head(4)
                    aspects_list = top_four_aspects['ASPECT'].to_list()
            #         formatted_aspects = ', '.join(f"'{aspect}'" for aspect in aspects_list)
                    key_df = get_final_df(aspects_list, device)
                    print(key_df)
                    b =  key_df.to_dict(orient='records')
                    summary_ans = query_aspect_wise_detailed_summary(user_question+"which have the following sentiment :" + str(dataframe_as_dict) + "and these are the imporatnt aspect based on aspect ranking : " + str(aspects_list) + "and their respective keywords" + str(b))
                    st.write(summary_ans)
                    full_response += summary_ans
                    st.dataframe(styled_df)
                    history = check_history_length(history,summary_ans)
                    save_list(history)
                    styled_df_html = styled_df.to_html(index=False)
                    full_response += styled_df_html  # Initialize full_response with the HTML table
                                
                elif classify_function == "2":
                    data= quantifiable_data(user_question_chart)
                    if 'NET_SENTIMENT' in data.columns:
                        overall_net_sentiment=data['NET_SENTIMENT'][0]
                        overall_net_sentiment = round(overall_net_sentiment, 1)
                        overall_review_count=data['REVIEW_COUNT'][0]
                        overall_review_count = round(overall_review_count)
                    #if 'visual' in user_question_chart.lower() or 'visualize' in user_question_chart.lower() or 'graph' in user_question_chart.lower() or 'chart' in user_question_chart.lower() or 'visualization' in user_question_chart.lower():
                        
                    words = user_question_chart.lower().split()
                    target_words = ['visual', 'visualize', 'graph', 'chart', 'visualization']
                    if any(word in words for word in target_words):
                        #st.write("for visual")
                        visual_data=data.copy()
                        numerical_cols = visual_data.select_dtypes(include='number').columns
                        visual_data[numerical_cols] = visual_data[numerical_cols].apply(lambda x: x.round(1) if x.dtype == 'float' else x)
                        generate_chart(visual_data)
                        
                        visual_data = visual_data[~visual_data.applymap(lambda x: x == 'TOTAL').any(axis=1)]
                        generate_chart_insight_llm(str(visual_data))
                        
                    elif len(data)>0:
                        
                        
                        
                        show_output=data.copy()
                        #show_output=data_show.drop(index=0)
                        numerical_cols = data.select_dtypes(include='number').columns
                        data[numerical_cols] = data[numerical_cols].apply(lambda x: x.round(1) if x.dtype == 'float' else x)
                        numerical_cols = show_output.select_dtypes(include='number').columns
                        show_output[numerical_cols] = show_output[numerical_cols].apply(lambda x: x.round(1) if x.dtype == 'float' else x)

                        #st.dataframe(show_output)

                        #data2=data.copy()
                        #show_output = show_output.replace('Unknown', pd.NA).dropna()
                        #data2['Impact']=np.where(data2['NET_SENTIMENT']<overall_net_sentiment,'LOW','HIGH')
                        if 'NET_SENTIMENT' in show_output.columns:
                            conditions = [
                              show_output['NET_SENTIMENT'] < overall_net_sentiment,
                              show_output['NET_SENTIMENT'] == overall_net_sentiment
                                         ]

                            choices = [
                                        'LOW',
                                        ' '
                                         ]

                            show_output['Impact'] = np.select(conditions, choices, default='HIGH')
                        #st.write(show_output)



                        #st.dataframe(data2)
                        dataframe_as_dict = show_output.to_dict(orient='records')

                        try:
                            user_question_chart = user_question_chart.replace("What is the", "Summarize reviews of")
                        except:
                            pass
                        if 'NET_SENTIMENT' in show_output.columns:
                            show_output = show_output.drop(index=0)
                            show_output2=show_output.copy()
                            show_output2.drop('Impact', axis=1, inplace=True)
                            show_output2 = show_output2.style.applymap(custom_color_gradient_compare,subset=['NET_SENTIMENT'])
                            show_output2 = show_output2.set_properties(**{'text-align': 'center'})
                            st.dataframe(show_output2)
                            #st.write(show_output2)
                            show_output2_html = show_output2.to_html(index=False)
                            full_response += show_output2_html
                            st.write(f" Overall Net Sentiment is {overall_net_sentiment} for {overall_review_count} reviews.")
                            qunat_summary = query_detailed_summary(str(show_output),user_question_chart + "Which have the following sentiment data : " + str(show_output),[])
                            full_response += qunat_summary
                            st.write(qunat_summary)
                        else:
                            show_output2=show_output.copy()
                            #show_output2.drop('Impact', axis=1, inplace=True)
                            st.write(show_output2)
                            show_output2_html = show_output2.to_html(index=False)
                            full_response += show_output2_html
                            qunat_summary = query_detailed_summary(str(show_output),user_question_chart + "Which have the following sentiment data : " + str(show_output),[])
                            st.write(qunat_summary)
                            full_response += qunat_summary


                        if(len(data))>1:

#                                 heat_map_visual = st.checkbox("Would you like to see visualization for this?")
#                                 if heat_map_visual:
                            generate_chart(data)
                    else:
                        st.write(data)
                        # st.write(f"Unable to fetch relevant details based on the provided input. Kindly refine your search query and try again!")
            except Exception as e:
                # st.write(data)
                error = "Unable to fetch relevant details based on the provided input. Kindly refine your search query and try again!"
        else:
            print('No Flow')



# Define the context globally
suggestions_context = """
Input:
You are an AI Assistant for an AI tool designed to provide insightful follow-up questions based on the user's initial query. Follow the instructions below strictly:

The AI tool has reviews data scraped from the web for different versions of Copilot:'Windows Copilot', 'Microsoft Copilot', 'Copilot for Security', 'Copilot Pro', 'Copilot for Microsoft 365', 'Edge Copilot', 'Github Copilot', 'Copilot for Mobile'.
The AI tool also has reviews data for competitors of different versions of Copilot: 'OpenAI GPT', 'Gemini AI', 'Claude AI', 'Vertex AI', 'Perplexity AI', 'Gemini AI For Mobile', 'ChatGPT For Mobile', 'Perplexity AI For Mobile', 'Claude AI For Mobile'.
The reviews are analyzed to extract aspects and sentiments related to the following aspects: 'Interface',,'Connectivity','Personalization/Customization','Privacy','Compatibility','Generic','Innovation','Reliability','Productivity','Price','Text Summarization/Generation','Code Generation','Ease of Use','Performance'.
Based on the user's previous response, which involved summarization, comparison, visualization, or generic queries about these reviews, suggest three follow-up prompts that the user can ask next to complete their story. Ensure the prompts cover a range of potential queries, including detailed summaries, aspect-wise comparisons, and more generic inquiries to provide a comprehensive understanding.

Your goal is to generate three prompts that are mutually exclusive and advance the user's exploration of the data. Consider the natural progression of inquiry, ensuring that the prompts align with a logical story flow such as:
    - Summarization
    - Visualization
    - Aspect-wise Net Sentiment
    - Comparison between competitors
    - Comparison within the same product family (e.g., different versions of Copilot)

Example Previous User Response: "Can you summarize the reviews for Copilot highlighting the different aspects?"

Model Task: Based on the provided previous user response, generate three related prompts that the user can ask next. These prompts should help the user delve deeper into the data to complete the story with sentiment data and should be related to the previous response.
IMPORTANT: Use simple English for questions. """

suggestions_interaction = """"""

def prompt_suggestion(user_question):
    global suggestions_context,suggestions_interaction
    full_prompt = suggestions_context + suggestions_interaction + "\nQuestion:\n" + user_question + "\nAnswer:"
    response = client.completions.create(
        model=deployment_name,
        prompt=full_prompt,
        max_tokens=500,
        temperature=0.3
    )
    # Extract the generated response
    user_query = response.choices[0].text.strip()
    # Update context with the latest interaction
    suggestions_interaction += "\nQuestion:\n" + user_question + "\nAnswer:\n" + user_query
    return user_query

            
            
def sugg_checkbox(user_question):
    if not st.session_state.prompt_sugg:
        questions = prompt_suggestion(user_question)
        print(f"Prompt Suggestions: {questions}")
        questions = questions.split('\n')
        questions_new = []
        for i in questions:
            if i[0].isdigit():
                x = i[3:]
                questions_new.append(x)
        st.session_state.prompt_sugg = questions_new
    checkbox_states = []
    checkbox_states = [st.checkbox(st.session_state.prompt_sugg[i],key = f"Checkbox{i}") for i in range(len(st.session_state.prompt_sugg))]
    for i, state in enumerate(checkbox_states):
        if state:
            st.session_state.selected_sugg = st.session_state.prompt_sugg[i]
            st.experimental_rerun()
            break
        st.session_state.selected_sugg = None
    return st.session_state.selected_sugg

              
            
global full_response
if __name__ == "__main__":
    global full_response
    if st.sidebar.subheader("Select an option"):
        options = ["Copilot", "Devices"]
        selected_options = st.sidebar.selectbox("Select product", options)
        if selected_options == "Copilot":
            # st.session_state['messages'] = []
            # st.session_state['chat_initiated'] = False
            st.session_state.devices_flag = False #Created this flag to reset the history for devices. Please do not delete this.
            st.header("Copilot Review Synthesis Tool")
            st.session_state.user_question = None #Resetting this variable for Devices, do not delete
            if "messages" not in st.session_state:
                st.session_state['messages'] = []
            if "chat_initiated" not in st.session_state:
                st.session_state['chat_initiated'] = False
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if message["role"] == "assistant" and "is_html" in message and message["is_html"]:
                        st.markdown(message["content"], unsafe_allow_html=True)
                    else:
                        st.markdown(message["content"])
            if user_inp := st.chat_input("Enter the Prompt: "):
                st.session_state.selected_questions = user_inp
                st.chat_message("user").markdown(st.session_state.selected_questions)
                st.session_state.messages.append({"role": "user", "content": st.session_state.selected_questions})
                
            if st.session_state.selected_sugg:
                st.session_state.selected_questions = st.session_state.selected_sugg
                st.chat_message("user").markdown(st.session_state.selected_questions)
                st.session_state.messages.append({"role": "user", "content": st.session_state.selected_questions})
                st.session_state.selected_sugg = None
                st.session_state.prompt_sugg = None
            if st.session_state.selected_questions:
                with st.chat_message("assistant"):
                    full_response = ""
                    if st.session_state.copilot_curr_ques != st.session_state.selected_questions:
                        try:
                            user_question = st.session_state.selected_questions.replace("Give me", "What is").replace("Give", "What is")
                            user_question_chart= user_question
                        except:
                            pass
                        classification = classify_prompts(user_question)
                        print(classification)
                        if classification != 'Generic':
                            user_question_1 = rephrase_prompt(str(user_question))
                            print(type(user_question_1))
                            print(user_question_1)
                            user_ques(str(user_question_1), user_question, classification, str(user_question_chart))
                        else:
                            user_question_1 = user_question
                            try: 
                                gen_ans = query_detailed_generic(user_question_1)
                                st.write(gen_ans)
                                Gen = query_quant_classify2_compare("Give me top 10 keywords with their review count  and % of positive, negative and neutral for the same keywords to answer : " +  user_question_1)
                                print(Gen)
                                st.dataframe(Gen)
                                dic = Gen.to_dict(orient = "dict")
                                Gen_Q = query_detailed_generic("Summarize reviews of keywords regarding " + user_question_1 + "Which have the following keyword data:" + str(dic))
                                st.write(Gen_Q)
                                full_response += Gen_Q
                                history = check_history_length(history,Gen_Q)
                                save_list(history)
                            except Exception as e:
                                print(e)
                                print("I couldn't get there")
                                gen_ans = query_detailed_generic(user_question_1)
                                st.write(gen_ans)
                                full_response += gen_ans
                                history = check_history_length(history,gen_ans)
                                save_list(history)

                        st.session_state.messages.append({"role": "assistant", "content": full_response, "is_html": True})
                    st.session_state.copilot_curr_ques = st.session_state.selected_questions
                    if selected_options == "Copilot" and 'full_response' in globals():
                        selected_questions = sugg_checkbox(full_response)
                        st.session_state['chat_initiated'] = True
            if st.session_state['chat_initiated'] and st.button("New Chat"):
                if os.path.exists(file_path):
                    os.remove(file_path)
                st.session_state['messages'] = []
                st.session_state['chat_initiated'] = False
                st.session_state.selected_sugg = None
                st.session_state.prompt_sugg = None
                st.session_state.selected_questions = ""
                st.session_state.copilot_curr_ques = None
                st.experimental_rerun()

        elif selected_options == "Devices":
            st.header("Devices Review Synthesis Tool")
            if st.sidebar.subheader("Select an approach for Topics"):
                options = ["Aspect-based", "Summary-based"]
                selected_options = st.sidebar.selectbox("Select approach", options)
                if not st.session_state.devices_flag:
                    st.session_state.display_history_devices = []
                    st.session_state.context_history_devices = []
                    st.session_state.curr_response = ""
                    st.session_state.user_question = None
                    st.session_state.devices_flag = True
                    st.session_state.selected_sugg = None
                    st.session_state.prompt_sugg = None
                    st.session_state.selected_questions = ""
                    st.session_state.copilot_curr_ques = None
                if "chat_initiated" not in st.session_state:
                    st.session_state['chat_initiated'] = False
                for message in st.session_state.display_history_devices:
                    with st.chat_message(message["role"]):
                        if message["role"] == "assistant" and "is_html" in message and message["is_html"]:
                            st.markdown(message["content"], unsafe_allow_html=True)
                        else:
                            st.markdown(message["content"])
                if user_inp := st.chat_input("Enter the Prompt: "):
                    st.chat_message("user").markdown(user_inp)
                    st.session_state.display_history_devices.append({"role": "user", "content": user_inp, "is_html": False})
                    st.session_state.user_question = user_inp
                if selected_options == "Aspect-based":
                    
######################################    APPROACH 1    #########################################################################
                    if st.session_state.devices_approach != "Aspect-based":
                        st.session_state.curr_response = ""
                        st.session_state.user_question = None
                        st.session_state.context_history_devices = []
                        st.session_state.devices_approach = "Aspect-based"
                        st.session_state.selected_sugg_devices = None
                        st.session_state.prompt_sugg_devices = None
                        
                    if st.session_state.selected_sugg_devices:
                        st.session_state.user_question = st.session_state.selected_sugg_devices
                        st.chat_message("user").markdown(st.session_state.user_question)
                        st.session_state.display_history_devices.append({"role": "user", "content": st.session_state.user_question, "is_html": False})
                        st.session_state.selected_sugg_devices = None
                        st.session_state.prompt_sugg_devices = None
                    
                    if st.session_state.user_question:
                        with st.chat_message("assistant"):
                            classification = identify_prompt(st.session_state.user_question)
                            print(f"\n\nPROMPT CLASSIFICATION FOR {st.session_state.user_question}: {classification}\n\n")
                            if classification == 'summarization':
                                device = device_summ(st.session_state.user_question.upper())
                                st.session_state.display_history_devices.append({"role": "assistant", "content": st.session_state.curr_response, "is_html": True})
                                st.session_state.curr_response = ""

                            elif classification == 'comparison':
                                devices = extract_comparison_devices(st.session_state.user_question)
                                if len(devices) == 2:
                                    dev_comp(devices[0],devices[1])
                                    st.session_state.display_history_devices.append({"role": "assistant", "content": st.session_state.curr_response, "is_html": True})
                                    st.session_state.curr_response = ""
                                elif len(devices) > 2:
                                    identified_devices = []
                                    for device in devices:
                                        identified_device = identify_devices(device.upper())
                                        if identified_device == "Device not available":
                                            st.write(f"Device {device} not present in the data.")
                                            continue
                                        identified_devices.append(identified_device)
                                    print(identified_devices)
                                    devices_data = query_quant_classify2_compare_devices("these are the exact names:" + str(identified_devices) + st.session_state.user_question)
                                    comparison_table = devices_data
                                    column_found_devices = (comparison_table)
                                    print(column_found_devices)
                                    dataframe_as_dict_devices = comparison_table.to_dict(orient = 'dict')

                                    try:
                                        # Pivot the table to get the desired format
                                        try:
                                            compared_pivot_df = comparison_table.pivot_table(index="ASPECT", columns='PRODUCT_FAMILY', values='NET_SENTIMENT_' + 'ASPECT', aggfunc='first')
                                            compared_pivot_df = compared_pivot_df.reindex(comparison_table["ASPECT"].drop_duplicates().values)
                                        except Exception as e:
                                            print(f"Error creating pivot table: {e}")

                                        # Process review counts
                                        try:
                                            comparison_table['REVIEW_COUNT'] = comparison_table['REVIEW_COUNT'].astype(str).str.replace(',', '').astype(float).astype(int)
                                            aspect_volume = comparison_table.groupby('ASPECT')['REVIEW_COUNT'].sum().reset_index()
                                        except Exception as e:
                                            print(f"Error processing review counts: {e}")

                                        # Merge and sort comparison table
                                        try:
                                            comparison_table = comparison_table.merge(aspect_volume, on='ASPECT', suffixes=('', '_Total'))
                                            comparison_table = comparison_table.sort_values(by='REVIEW_COUNT_Total', ascending=False)
                                        except Exception as e:
                                            print(f"Error merging and sorting comparison table: {e}")

                                        # Handle possible scenarios for creating pivot table
                                        try:
                                            device_pivot_df = comparison_table.pivot_table(index='ASPECT', columns='PRODUCT_FAMILY', values='NET_SENTIMENT_' + 'ASPECT')
                                            device_pivot_df = device_pivot_df.reindex(comparison_table['ASPECT'].drop_duplicates().values)
                                            print("In 1st Case")
                                        except Exception as e1:
                                            print(f"Error in 1st case: {e1}")
                                            try:
                                                device_pivot_df = comparison_table.pivot_table(index='ASPECT', columns='PRODUCT', values='NET_SENTIMENT_' + 'ASPECT')
                                                device_pivot_df = device_pivot_df.reindex(comparison_table['ASPECT'].drop_duplicates().values)
                                                print("In 2nd Case")
                                            except Exception as e2:
                                                print(f"Error in 2nd case: {e2}")
                                                try:
                                                    device_pivot_df = comparison_table.drop(columns=['REVIEW_COUNT', 'REVIEW_COUNT_Total'])
                                                    print("In 3rd Case")
                                                except Exception as e3:
                                                    print(f"Error in 3rd case: {e3}")
                                                    try:
                                                        device_pivot_df = comparison_table.drop(columns=['REVIEW_COUNT'])
                                                        print("In 4th Case")
                                                    except Exception as e4:
                                                        print(f"Error in 4th case: {e4}")
                                                        device_pivot_df = comparison_table
                                        try:
                                            additional_pivot_df = comparison_table.pivot_table(index='ASPECT', columns='PRODUCT_FAMILY', values='NET_SENTIMENT_ASPECT', aggfunc='first')
                                            additional_pivot_df = additional_pivot_df.reindex(comparison_table['ASPECT'].drop_duplicates().values)
                                            print("Pivoting in additional case")
                                        except Exception as e5:
                                            print(f"Error in additional case: {e5}")

                                        # Convert pivot table to dictionary and generate summary
                                        dataframe_as_dict_devices = device_pivot_df.to_dict(orient='dict')
                                        try:
                                            device_pivot_df = device_pivot_df.astype(int)
                                        except Exception as e:
                                            print(f"Error converting pivot table to int: {e}")
                                            pass
                                        st.dataframe(device_pivot_df)
                                        compared_pivot_html = device_pivot_df.to_html(index=False)
                                        save_history_devices(compared_pivot_html)
                                        compare_dict = device_pivot_df.to_dict(orient="dict")
                                        compare_summary = query_detailed_summary_devices("Based on this Data" + str(compare_dict) + "Give detailed answer for this question: " + st.session_state.user_question)
                                        st.write(compare_summary)
                                        save_history_devices(compare_summary)
                                        st.session_state.display_history_devices.append({"role": "assistant", "content": compare_summary, "is_html": True})
                                    except Exception as e:
                                        Gen_Ans = query_devices_detailed_generic(st.session_state.user_question)
                                        st.write(Gen_Ans)
                                        save_history_devices(Gen_Ans)
                                        st.session_state.display_history_devices.append({"role": "assistant", "content": Gen_Ans, "is_html": False})
                                else:
                                    Gen_Ans = query_devices_detailed_generic(st.session_state.user_question)
                                    st.write(Gen_Ans)
                                    save_history_devices(Gen_Ans)
                                    st.session_state.display_history_devices.append({"role": "assistant", "content": Gen_Ans, "is_html": False})  

                            elif classification == 'quant':
                                user_question_final=st.session_state.user_question.upper().replace("LAPTOP","")
                                user_question_final=user_question_final.replace("DEVICE","")
                                devices_quant_approach1(user_question_final)
                                

                            elif classification == 'sales':
                                user_question_final=st.session_state.user_question.upper().replace("LAPTOP","")
                                user_question_final=user_question_final.replace("DEVICE","")
                                sales_quant_approach1(user_question_final)

                            else:
                                Gen_Ans = query_devices_detailed_generic(st.session_state.user_question)
                                st.write(Gen_Ans)
                                save_history_devices(Gen_Ans)
                                st.session_state.display_history_devices.append({"role": "assistant", "content": Gen_Ans, "is_html": False})
                        st.session_state['chat_initiated'] = True
                        
                        print(f"Context History: {str(st.session_state.context_history_devices)}")
                        print(f"Classification: {classification}")
                        
                        if st.session_state.context_history_devices and classification != "summarization":
                            selected_questions = sugg_checkbox_devices(str(st.session_state.context_history_devices))
                        
                    if st.session_state['chat_initiated'] and st.button("New Chat"):
                        st.session_state['messages'] = []
                        st.session_state['chat_initiated'] = False
                        st.session_state.user_question = None
                        st.session_state.display_history_devices = []
                        st.session_state.context_history_devices = []
                        st.session_state.curr_response = ""
                        st.session_state.prompt_sugg_devices = None
                        st.session_state.selected_sugg_devices = None
                        st.experimental_rerun()
                        
                elif selected_options == "Summary-based":
                    if st.session_state.devices_approach != "Summary-based":
                        st.session_state.user_question = None
                        st.session_state.context_history_devices = []
                        st.session_state.curr_response = ""
                        st.session_state.devices_approach = "Summary-based"
                        st.session_state.selected_sugg_devices = None
                        st.session_state.prompt_sugg_devices = None
                        
                    if st.session_state.selected_sugg_devices:
                        st.session_state.user_question = st.session_state.selected_sugg_devices
                        st.chat_message("user").markdown(st.session_state.user_question)
                        st.session_state.display_history_devices.append({"role": "user", "content": st.session_state.user_question, "is_html": False})
                        st.session_state.selected_sugg_devices = None
                        st.session_state.prompt_sugg_devices = None
                        
                    if st.session_state.user_question:   
                    #if user_question := st.chat_input("Enter the Prompt: "):
                        with st.chat_message("assistant"):
                            classification = identify_prompt_new(st.session_state.user_question)
                            if classification == 'summarization':
                                device = device_summ_new(st.session_state.user_question.upper())
                                st.session_state.display_history_devices.append({"role": "assistant", "content": st.session_state.curr_response, "is_html": True})
                                st.session_state.curr_response = ""

                            elif classification == 'comparison':
                                devices = extract_comparison_devices_new(st.session_state.user_question)
                                if len(devices) == 2:
                                    dev_comp_new(devices[0],devices[1])
                                    st.session_state.display_history_devices.append({"role": "assistant", "content": st.session_state.curr_response, "is_html": True})
                                    st.session_state.curr_response = ""

                                elif len(devices) > 2:
                                    identified_devices = []
                                    for device in devices:
                                        identified_device = identify_devices_new(device.upper())
                                        if identified_device == "Device not available":
                                            st.write(f"Device {device} not present in the data.")
                                            continue
                                        identified_devices.append(identified_device)
                                    print(identified_devices)
                                    devices_data = query_quant_classify2_compare_devices_new("these are the exact names:" + str(identified_devices) + st.session_state.user_question)
                                    comparison_table = devices_data
                                    column_found_devices = (comparison_table)
                                    print(column_found_devices)
                                    dataframe_as_dict_devices = comparison_table.to_dict(orient = 'dict')

                                    try:
                                        # Pivot the table to get the desired format
                                        try:
                                            compared_pivot_df = comparison_table.pivot_table(index="ASPECT", columns='PRODUCT_FAMILY', values='NET_SENTIMENT_' + 'ASPECT', aggfunc='first')
                                            compared_pivot_df = compared_pivot_df.reindex(comparison_table["ASPECT"].drop_duplicates().values)
                                        except Exception as e:
                                            print(f"Error creating pivot table: {e}")

                                        # Process review counts
                                        try:
                                            comparison_table['REVIEW_COUNT'] = comparison_table['REVIEW_COUNT'].astype(str).str.replace(',', '').astype(float).astype(int)
                                            aspect_volume = comparison_table.groupby('ASPECT')['REVIEW_COUNT'].sum().reset_index()
                                        except Exception as e:
                                            print(f"Error processing review counts: {e}")

                                        # Merge and sort comparison table
                                        try:
                                            comparison_table = comparison_table.merge(aspect_volume, on='ASPECT', suffixes=('', '_Total'))
                                            comparison_table = comparison_table.sort_values(by='REVIEW_COUNT_Total', ascending=False)
                                        except Exception as e:
                                            print(f"Error merging and sorting comparison table: {e}")

                                        # Handle possible scenarios for creating pivot table
                                        try:
                                            device_pivot_df = comparison_table.pivot_table(index='ASPECT', columns='PRODUCT_FAMILY', values='NET_SENTIMENT_' + 'ASPECT')
                                            device_pivot_df = device_pivot_df.reindex(comparison_table['ASPECT'].drop_duplicates().values)
                                            print("In 1st Case")
                                        except Exception as e1:
                                            print(f"Error in 1st case: {e1}")
                                            try:
                                                device_pivot_df = comparison_table.pivot_table(index='ASPECT', columns='PRODUCT', values='NET_SENTIMENT_' + 'ASPECT')
                                                device_pivot_df = device_pivot_df.reindex(comparison_table['ASPECT'].drop_duplicates().values)
                                                print("In 2nd Case")
                                            except Exception as e2:
                                                print(f"Error in 2nd case: {e2}")
                                                try:
                                                    device_pivot_df = comparison_table.drop(columns=['REVIEW_COUNT', 'REVIEW_COUNT_Total'])
                                                    print("In 3rd Case")
                                                except Exception as e3:
                                                    print(f"Error in 3rd case: {e3}")
                                                    try:
                                                        device_pivot_df = comparison_table.drop(columns=['REVIEW_COUNT'])
                                                        print("In 4th Case")
                                                    except Exception as e4:
                                                        print(f"Error in 4th case: {e4}")
                                                        device_pivot_df = comparison_table
                                        try:
                                            additional_pivot_df = comparison_table.pivot_table(index='ASPECT', columns='PRODUCT_FAMILY', values='NET_SENTIMENT_ASPECT', aggfunc='first')
                                            additional_pivot_df = additional_pivot_df.reindex(comparison_table['ASPECT'].drop_duplicates().values)
                                            print("Pivoting in additional case")
                                        except Exception as e5:
                                            print(f"Error in additional case: {e5}")

                                        # Convert pivot table to dictionary and generate summary
                                        dataframe_as_dict_devices = device_pivot_df.to_dict(orient='dict')
                                        try:
                                            device_pivot_df = device_pivot_df.astype(int)
                                        except Exception as e:
                                            print(f"Error converting pivot table to int: {e}")
                                            pass
                                        st.dataframe(device_pivot_df)
                                        compared_pivot_html = device_pivot_df.to_html(index=False)
                                        save_history_devices_new(compared_pivot_html)
                                        compare_dict = device_pivot_df.to_dict(orient="dict")
                                        compare_summary = query_detailed_summary_devices_new("Based on this Data" + str(compare_dict) + "Give detailed answer for this question: " + st.session_state.user_question)
                                        st.write(compare_summary)
                                        save_history_devices_new(compare_summary)
                                        st.session_state.display_history_devices.append({"role": "assistant", "content": compare_summary, "is_html": True})
                                    except Exception as e:
                                        Gen_Ans = query_devices_detailed_generic_new(st.session_state.user_question)
                                        st.write(Gen_Ans)
                                        save_history_devices_new(Gen_Ans)
                                        st.session_state.display_history_devices.append({"role": "assistant", "content": Gen_Ans, "is_html": False})
                                else:
                                    Gen_Ans = query_devices_detailed_generic_new(st.session_state.user_question)
                                    st.write(Gen_Ans)
                                    save_history_devices_new(Gen_Ans)
                                    st.session_state.display_history_devices.append({"role": "assistant", "content": Gen_Ans, "is_html": False})  

                            elif classification == 'quant':
                                user_question_final=st.session_state.user_question.upper().replace("LAPTOP","")
                                user_question_final=user_question_final.replace("DEVICE","")
                                devices_quant_approach2(user_question_final)


                            elif classification == 'sales':
                                user_question_final=st.session_state.user_question.upper().replace("LAPTOP","")
                                user_question_final=user_question_final.replace("DEVICE","")
                                sales_quant_approach2(user_question_final)
                                
                            else:
                                Gen_Ans = query_devices_detailed_generic_new(st.session_state.user_question)
                                st.write(Gen_Ans)
                                save_history_devices_new(Gen_Ans)
                                st.session_state.display_history_devices.append({"role": "assistant", "content": Gen_Ans, "is_html": False})
                        st.session_state['chat_initiated'] = True
                        if st.session_state.context_history_devices and classification != "summarization":
                            selected_questions = sugg_checkbox_devices(str(st.session_state.context_history_devices))
                    
                    if st.session_state['chat_initiated'] and st.button("New Chat"):
                        st.session_state['messages'] = []
                        st.session_state['chat_initiated'] = False
                        st.session_state.user_question = None
                        st.session_state.display_history_devices = []
                        st.session_state.context_history_devices = []
                        st.session_state.curr_response = ""
                        st.session_state.prompt_sugg_devices = None
                        st.session_state.selected_sugg_devices = None
                        st.experimental_rerun()
