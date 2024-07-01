import streamlit as st
from azure.core.credentials import AzureKeyCredential
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
import plotly.express as px
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
import streamlit as st
from fuzzywuzzy import process
from rapidfuzz import process, fuzz


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")


client = AzureOpenAI(
     api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
     api_version="2024-02-01",
     azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
     )


deployment_name='Surface_Analytics'
azure_deployment_name = "Thruxton_R"
azure_embedding_name ="MV_Agusta"

RCR_Sales_Data = pd.read_csv('RCR Sales Data Sample V3.csv')
dev_mapping = pd.read_csv('SalesSentimentMapping.csv')
Devices_Sentiment_Data_New  = pd.read_csv("Windows_Data_116k_Summary_Final.csv")

#Filtering RCR Sales Data based on devices present in Sales-Sentiment Device Name Mapping File
mapping_sales_devices = list(dev_mapping['SalesDevice'].unique())
RCR_Sales_Data = RCR_Sales_Data[RCR_Sales_Data['Series'].isin(mapping_sales_devices)]

#Filtering Sentiment Data based on devices present in Sales-Sentiment Device Name Mapping File
mapping_sentiment_devices = list(dev_mapping['SentimentDevice'].unique())
Devices_Sentiment_Data_New = Devices_Sentiment_Data_New[Devices_Sentiment_Data_New['Product_Family'].isin(mapping_sentiment_devices)]

if not hasattr(st.session_state, 'selected_devices'):
    st.session_state.selected_devices = [None,None]
if not hasattr(st.session_state, 'past_inp2'):
    st.session_state.past_inp2 = None
if not hasattr(st.session_state, 'past_inp_comp_dev2'):
    st.session_state.past_inp_comp_dev2 = []
if not hasattr(st.session_state, 'display_history_devices'):
    st.session_state.display_history_devices = []
if not hasattr(st.session_state, 'context_history_devices'):
    st.session_state.context_history_devices = []
if not hasattr(st.session_state, 'curr_response'):
    st.session_state.curr_response = ""
if not hasattr(st.session_state, 'prompt_sugg_devices'):
    st.session_state.prompt_sugg_devices = ""
if not hasattr(st.session_state, 'selected_sugg_devices'):
    st.session_state.selected_sugg_devices = ""
    
def save_history_devices_new(summ):
    if not hasattr(st.session_state, 'context_history_devices'):
        st.session_state.context_history_devices = []
    if len(st.session_state.context_history_devices)>3:
        st.session_state.context_history_devices = st.session_state.context_history_devices[1:]
    st.session_state.context_history_devices.append(summ)


def Sentiment_Score_Derivation(value):
    try:
        if value == "Positive":
            return 1
        elif value == "Negative":
            return -1
        else:
            return 0
    except:
        err = f"An error occurred while deriving Sentiment Score."
        print(err)
        return err    

#Deriving Sentiment Score and Review Count columns into the dataset
Devices_Sentiment_Data_New["Sentiment_Score"] = Devices_Sentiment_Data_New["Sentiment"].apply(Sentiment_Score_Derivation)
Devices_Sentiment_Data_New["Review_Count"] = 1.0    
    

RCR_context = """
    1. Your Job is to convert the user question to SQL Query (Follow Microsoft SQL server SSMS syntax.). You have to give the query so that it can be used on Microsoft SQL server SSMS.You have to only return query as a result.
    2. There is only one table with table name RCR_Sales_Data where each row has. The table has 20 columns, they are:
        Month: Contains dates for the records
        Country: From where the sales has happened. It contains following values: 'Turkey','India','Brazil','Germany','Philippines','France','Netherlands','Spain','United Arab Emirates','Czech Republic','Norway','Belgium','Finland','Canada','Mexico','Russia','Austria','Poland','United States','Switzerland','Italy','Colombia','Japan','Chile','Sweden','Vietnam','Saudi Arabia','South Africa','Peru','Indonesia','Taiwan','Thailand','Ireland','Korea','Hong Kong SAR','Malaysia','Denmark','New Zealand','China' and 'Australia'.
        Geography: From which Country or Region the review was given. It contains following values: 'Unknown', 'Brazil', 'Australia', 'Canada', 'China', 'Germany','France'.
        OEMGROUP: OEM or Manufacturer of the Device. It contains following values: 'Lenovo','Acer','Asus','HP','All Other OEMs', 'Microsoft' and 'Samsung'
        SUBFORMFACTOR: Formfactor of the device. It contains following values: 'Ultraslim Notebook'.
        GAMINGPRODUCTS: Flag whether Device is a gaming device or not. It contains following values: 'GAMING', 'NO GAMING' and 'N.A.'.
        SCREEN_SIZE_INCHES: Screen Size of the Device.
        PRICE_BRAND_USD_3: Band of the price at which the device is selling. It contains following values: '0-300', '300-500', '500-800' and '800+.
        OS_VERSION: Operating System version intall on the device. It contains following values: 'Windows 11', 'Chrome', 'Mac OS'.
        Operating_System_Summary: Operating System installed on the device. This is at uber level. It contains following values: 'Windows', 'Google OS', 'Apple OS'.
        Sales_Units: Number of Devices sold for that device in a prticular month and country.
        Sales_Value: Revenue Generated by the devices sold.
        Series: Family of the device such as IdeaPad 1, HP Laptop 15 etc.
        Specs_Combination: Its contains the combination of Series, Processor, RAM , Storage and Screen Size. For Example: SURFACE LAPTOP GO | Ci5 | 8 GB | 256.0 SSD | 12" .
        Chassis Segment: It contains following values: 'SMB_Upper','Mainstream_Lower','SMB_Lower','Enterprise Fleet_Lower','Entry','Mainstream_Upper','Premium Mobility_Upper','Enterprise Fleet_Upper','Premium Mobility_Lower','Creation_Lower','UNDEFINED','Premium_Mobility_Upper','Enterprise Work Station','Unknown','Gaming_Musclebook','Entry_Gaming','Creation_Upper','Mainstrean_Lower'
        
    3.  When Asked for Price Range you have to use ASP Column to get minimum and Maxium value. Do not consider Negative Values. Also Consider Sales Units it shouldn't be 0.
        Exaple Query:
            SELECT MIN(ASP) AS Lowest_Value, MAX(ASP) AS Highest_Value
            FROM RCR_Sales_Data
            WHERE
            Series = 'Device Name'
            AND ASP >= 0
            AND Sales_Units <> 0;
    4. Total Sales_Units Should Always be in Thousands. 
        Example Query:
            SELECT (SUM(Sales_Units) / 1000) AS "TOTAL SALES UNITS"
            FROM RCR_Sales_Data
            WHERE
            SERIES LIKE '%SURFACE LAPTOP GO%';
    5. Average Selling Price (ASP): It is calculated by sum of SUM(Sales_Value)/SUM(Sales_Units)
    6. Total Sales Units across countries or across regions is sum of sales_units for those country. It should be in thousand of million hence add "K" or "M" after the number.
        Example to calculate sales units across country:
            SELECT Country, (SUM(Sales_Units) / 1000) AS "Sales_Units(In Thousands)"
            FROM RCR_Sales_Data
            GROUP BY Country
            ORDER BY Sales_Units DESC
    7. Total Sales Units across column "X" or across regions is sum of sales_units for those country. It should be in thousand of million hence add "K" or "M" after the number.
        Example to calculate sales units across country:
            SELECT "X", (SUM(Sales_Units) / 1000) AS "Sales_Units(In Thousands)"
            FROM RCR_Sales_Data
            GROUP BY "X"
            ORDER BY Sales_Units DESC
    8. If asked about the highest selling Specs Combination. 
        Example Query:
            SELECT Specs_Combination, (SUM(Sales_Units) / 1000) AS "TOTAL SALES UNITS"
            FROM RCR_Sales_Data
            WHERE SERIES LIKE '%Macbook AIR%'
            AND SALES_UNITS <> 0
            GROUP BY Specs_Combination
            ORDER BY "TOTAL SALES UNITS" DESC
            LIMIT 1;
    9. If asked about similar compete devices.
    Example Query:
            SQL = WITH DeviceNameASP AS (
                    SELECT
                        'Device Name' AS Series,
                        SUM(Sales_Value) / SUM(Sales_Units) AS ASP,
                        Chassis_Segment,
                        SUM(Sales_Units) AS Sales_Units
                    FROM
                        RCR_Sales_Data
                    WHERE
                        Series LIKE '%Device Name%'
                    GROUP BY
                        Chassis_Segment
                ),
                CompetitorASP AS (
                    SELECT
                        Series,
                        SUM(Sales_Value) / SUM(Sales_Units) AS ASP,
                        Chassis_Segment,
                        SUM(Sales_Units) AS Sales_Units
                    FROM
                        RCR_Sales_Data
                    WHERE
                        Operating_System_Summary IN ('Apple OS', 'Google OS','Windows OS')
                        AND SERIES NOT LIKE '%Device Name%'
                    GROUP BY
                        Series, Chassis_Segment
                ),
                RankedCompetitors AS (
                    SELECT
                        C.Series,
                        C.ASP,
                        C.Chassis_Segment,
                        C.Sales_Units,
                        ROW_NUMBER() OVER (PARTITION BY C.Chassis_Segment ORDER BY C.Sales_Units DESC) AS rank
                    FROM
                        CompetitorASP C
                    JOIN
                        DeviceNameASP S
                    ON
                        ABS(C.ASP - S.ASP) <= 400
                        AND C.Chassis_Segment = S.Chassis_Segment
                )
                SELECT
                    Series,
                    ASP AS CompetitorASP,
                    Sales_Units
                FROM
                    RankedCompetitors
                WHERE
                    rank <= 3;

    10. If asked about dates or year SUBSTR() function instead of Year() or Month()
    11. Convert numerical outputs to float upto 2 decimal point.
    12. Always include ORDER BY clause to sort the table based on the aggregate value calculated in the query.
    13. Always use 'LIKE' operator whenever they mention about any Country, Series. Use 'LIMIT' operator instead of TOP operator.Do not use TOP OPERATOR. Follow syntax that can be used with pandasql.
    14. If you are using any field in the aggregate function in select statement, make sure you add them in GROUP BY Clause.
    15. Make sure to Give the result as the query so that it can be used on Microsoft SQL server SSMS.
    16. Always use LIKE function instead of = Symbol while generating SQL Query
    17. Important: User can ask question about any categories including Country, OEMGROUP,OS_VERSION etc etc. Hence, include the in SQL Query if someone ask it.
    18. Important: Use the correct column names listed above. There should not be Case Sensitivity issue. 
    19. Important: The values in OPERATING_SYSTEM_SUMMARY are ('Apple OS', 'Google OS') not ('APPLE OS', 'GOOGLE OS'). So use exact values. Not everything should be capital letters.
    20. Important: You Response should directly starts from SQL query nothing else."""

interaction = ""

def generate_SQL_Query_new(user_question):
    global RCR_context, interaction
    try:
        # Append the new question to the context
        full_prompt = RCR_context + interaction + "\nQuestion:\n" + user_question + "\nAnswer:"

        # Send the query to Azure OpenAI
        response = client.completions.create(
            model=deployment_name,
            prompt=full_prompt,
            max_tokens=500,
            temperature=0
        )

        # Extract the generated SQL query
        sql_query = response.choices[0].text.strip()

        # Update context with the latest interaction
        interaction += "\nQuestion:\n" + user_question + "\nAnswer:\n" + sql_query
    except:
        print(f"An error occured in generate_SQL_Query_new() for {user_question}")
        sql_query = None
    
    return sql_query

#Converting Top Operator to Limit Operator as pandasql doesn't support Top
def convert_top_to_limit_new(sql):
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
    except:
        print(f"An error occured while converting Top to LIMIT in SQL Query: {sql}")
        return None


def process_tablename_new(sql, table_name):
    x = sql.upper()
    query = x.replace(table_name.upper(), table_name)
    return query


def process_tablename_devices_new(sql, table_name):
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
    except:
        err = f"An error occurred while processing table name in SQL query."
        return err
        




def get_sales_units_new(device_name):
    try:
        question = "Totals Sales Units for " + device_name
        a = generate_SQL_Query_new(question)
        SQL_Query = convert_top_to_limit_new(a)
        SQL_Query = process_tablename_new(SQL_Query,"RCR_Sales_Data")
        data = ps.sqldf(SQL_Query, globals())
        col_name = data.columns[0]
        total_sales = data[col_name][0]
        total_sales = str(round(total_sales)) + "K"
    except:
        print(f"Error in getting sales units for {device_name}")
        total_sales = "NA"
        
    return total_sales 
  
def get_star_rating_html_new(net_sentiment):
    try:
        # Normalize net sentiment from -100 to 100 to 0 to 10 for star ratings
        normalized_rating = (net_sentiment + 100) / 40

        # Determine the number of full and half stars
        full_stars = int(normalized_rating)
        half_star = 1 if normalized_rating - full_stars >= 0.5 else 0

        # CSS for the stars
        star_style = 'font-size: 16px; margin-right: 5px; color: gold;'  # Adjust font-size and margin-right as needed

        # Generate the HTML for the stars
        star_html = '<span>'
        star_html += f'<i class="fa fa-star" style="{star_style}"></i>' * full_stars
        if half_star:
            star_html += f'<i class="fa fa-star-half-o" style="{star_style}"></i>'  # Half-star icon from Font Awesome
        star_html += f'<i class="fa fa-star-o" style="{star_style}"></i>' * (5 - full_stars - half_star)
        star_html += '</span>'
        return star_html
    except:
        print(f"Error in getting star rating.")
        return "NA"

def correct_compete_sales_query_new(SQL_Query):
    strr = SQL_Query
    strr = strr.replace("ABS(C.ASP - S.ASP) < LIKE  400","ABS(C.ASP - S.ASP) <=  400")
    strr = strr.replace("RANK < LIKE  3","RANK <=  3")
    strr = strr.replace("AND C.CHASSIS_SEGMENT  LIKE  S.CHASSIS_SEGMENT", "AND C.CHASSIS_SEGMENT  =  S.CHASSIS_SEGMENT")
    return strr


def get_ASP_new(device_name):
    try:
        question = "What's ASP for " + device_name
        a = generate_SQL_Query_new(question)
        SQL_Query = convert_top_to_limit_new(a)
        SQL_Query = process_tablename_new(SQL_Query,"RCR_Sales_Data")
        data = ps.sqldf(SQL_Query, globals())
        col_name = data.columns[0]
        asp = data[col_name][0]
        asp = "$" + str(int(round(asp)))
    except:
        asp = "NA"
        print(f"Error in getting ASP for {device_name}")
    return asp


def get_date_range_new(device_name):
    try:
        device = get_sales_device_name(device_name)
        device_sales_data = RCR_Sales_Data.loc[RCR_Sales_Data["Series"] == device]
        device_sales_data['Month'] = pd.to_datetime(device_sales_data['Month'], format='%m/%d/%Y')
        min_date = device_sales_data["Month"].min().strftime('%Y-%m-%d')
        max_date = device_sales_data["Month"].max().strftime('%Y-%m-%d')
    except:
        min_date = "NA"
        max_date = "NA"
        print(f"Error occured in getting date range for sales data of {device_name}")
    return min_date, max_date



def get_highest_selling_specs_new(device_name):
    try:
        question = "What's highest selling Specs Combination for " + device_name
        a = generate_SQL_Query_new(question)
        SQL_Query = convert_top_to_limit_new(a)
        SQL_Query = process_tablename_new(SQL_Query,"RCR_Sales_Data")
        data = ps.sqldf(SQL_Query, globals())
        col_name1 = data.columns[0]
        col_name2 = data.columns[1]
        specs = data[col_name1][0]
        sales_unit = data[col_name2][0]
        sales_unit = str(round(sales_unit)) + "K"
    except:
        specs = "NA"
        sales_unit = "NA"
        print(f"Error in getting highest selling specs for {device_name}")
    return specs,sales_unit

def compete_device_new(device_name):
    try:
        question = "What are the compete device for " + device_name
        a = generate_SQL_Query_new(question)
        SQL_Query = convert_top_to_limit_new(a)
        SQL_Query = process_tablename_new(SQL_Query,"RCR_Sales_Data")
        SQL_Query = SQL_Query.replace('APPLE','Apple')
        SQL_Query = SQL_Query.replace('GOOGLE','Google')
        SQL_Query = SQL_Query.replace('WINDOWS','Windows')
#         st.write(f"SQL Query to get competitor device for {device_name}: \n{SQL_Query}")
        SQL_Query = correct_compete_sales_query_new(SQL_Query)
        data = ps.sqldf(SQL_Query, globals())
        
    except:
        data = None
        print(f"Error in getting competitor devices for {device_name}")
    return data
    
def get_sales_device_name_new(input_device):
    try:
        sales_device_name = dev_mapping[dev_mapping['SentimentDevice']==input_device]['SalesDevice']
        if len(sales_device_name) == 0:
            sales_device_name = dev_mapping[dev_mapping['SalesDevice']==input_device]['SalesDevice']
        if len(sales_device_name) == 0:
            sales_device_name = None
        else:
            sales_device_name = sales_device_name.to_list()[0]
    except:
        sales_device_name = None
        print(f"Error in getting sales device name for {input_device}")
    return sales_device_name

def get_sentiment_device_name_new(input_device):
    try:
        sentiment_device_name = dev_mapping[dev_mapping['SentimentDevice']==input_device]['SentimentDevice']
        if len(sentiment_device_name) == 0:
            sentiment_device_name = dev_mapping[dev_mapping['SalesDevice']==input_device]['SentimentDevice']
        if len(sentiment_device_name) == 0:
            sentiment_device_name = None
        else:
            sentiment_device_name = sentiment_device_name.to_list()[0]
    except:
        sentiment_device_name = None
        print(f"Error in getting sentiment device name for {input_device}")
    return sentiment_device_name


    
def get_device_image_new(user_input):
    dev = user_input
    try:
        # Assuming the images are in a folder named 'Device Images'
        img_folder = 'Device Images'
        img_path = os.path.join(img_folder, f"{dev}.JPG")
        if not os.path.exists(img_path):
            img_path = None
    except:
        img_path = None
        print(f"Error in getting device image for {user_input}")
    return (dev, img_path)
    
def get_net_sentiment_new(device_name):
    SQL_Query = f"""SELECT 'TOTAL' AS Aspect, 
                        ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                        SUM(Review_Count) AS Review_Count
                        FROM Devices_Sentiment_Data_New
                        WHERE Product_Family LIKE '{device_name}'

                        UNION

                        SELECT Aspect, 
                        ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                        SUM(Review_Count) AS Review_Count
                        FROM Devices_Sentiment_Data_New
                        WHERE Product_Family LIKE '%{device_name}%'
                        GROUP BY Aspect
                        ORDER BY Review_Count DESC"""
    SQL_Query = process_tablename_new(SQL_Query,"Devices_Sentiment_Data_New")
    a = ps.sqldf(SQL_Query, globals())
    try:
        Net_Sentiment = float(a[a['ASPECT']=='TOTAL']['ASPECT_SENTIMENT'].values[0])
        aspects = a["ASPECT"].unique()
        if "Performance" in aspects:
            Performance_Sentiment = float(a[a['ASPECT']=='Performance']['ASPECT_SENTIMENT'].values[0])
        else:
            Performance_Sentiment = 0
        
        if "Design" in aspects:
            Design_Sentiment = float(a[a['ASPECT']=='Design']['ASPECT_SENTIMENT'].values[0])
        else:
            Design_Sentiment = 0
        
        if "Display" in aspects:
            Display_Sentiment = float(a[a['ASPECT']=='Display']['ASPECT_SENTIMENT'].values[0])
        else:
            Display_Sentiment = 0
        
        if "Battery" in aspects:
            Battery_Sentiment = float(a[a['ASPECT']=='Battery']['ASPECT_SENTIMENT'].values[0])
        else:
            Battery_Sentiment = 0
        
        if "Price" in aspects:
            Price_Sentiment = float(a[a['ASPECT']=='Price']['ASPECT_SENTIMENT'].values[0])
        else:
            Price_Sentiment = 0
        
        if "Software" in aspects:
            Software_Sentiment = float(a[a['ASPECT']=='Software']['ASPECT_SENTIMENT'].values[0])
        else:
            Software_Sentiment = 0
            
            
        aspect_sentiment = list((Performance_Sentiment, Design_Sentiment, Display_Sentiment, Battery_Sentiment, Price_Sentiment, Software_Sentiment))
                                 
    except:
        Net_Sentiment = None
        aspect_sentiment = None   
        print(f"Error in getting net sentiment and aspect sentiment for {device_name}")
    return Net_Sentiment, aspect_sentiment

def get_comp_device_details_new(user_input, df1):
    df1['SERIES'] = df1['SERIES'].str.upper()
    sales_data = df1[df1['SERIES'] == user_input]
    dev = user_input
    try:
        sentiment_device_name = get_sentiment_device_name_new(user_input)
    except:
        sentiment_device_name = None
    try:
        # Assuming the images are in a folder named 'Device Images'
        img_folder = 'Device Images'
        img_path = os.path.join(img_folder, f"{sentiment_device_name}.JPG")
        if not os.path.exists(img_path):
            img_not_found = "IMAGE NOT FOUND"
            img_path = os.path.join(img_folder, f"{img_not_found}.JPG")
    except:
        img_path = None
        print(f"Error in getting competitor device image: {user_input}")
    if sales_data.empty:
        return user_input, img_path, None, None, None  # Return dev and link, but None for sales and ASP if no matching SERIES is found
    
    try:
        sales = str(int(round(float(sales_data['SALES_UNITS'].values[0]) / 1000))) + "K"
    except:
        print(f"Error in getting competitor sales data: {user_input}")
        sales = "NA"
    try:
        ASP = "$" + str(int(sales_data['COMPETITORASP'].values[0]))
    except:
        print(f"Error in getting competitor ASP: {user_input}")
        ASP = "NA"
    net_sentiment,aspect_sentiment = get_net_sentiment_new(sentiment_device_name)
    return dev, img_path, sales, ASP, net_sentiment,sentiment_device_name
    
    
def get_final_df_devices_new(aspects_list,device):
    try:
        final_df = pd.DataFrame()
        device = device
        aspects_list = aspects_list
        # Iterate over each aspect and execute the query
        for aspect in aspects_list:
            # Construct the SQL query for the current aspect
            query = f"""
            SELECT Aspect_Description,
                   COUNT(CASE WHEN Sentiment = 'Positive' THEN 1 END) AS Positive_Count,
                   COUNT(CASE WHEN Sentiment = 'Negative' THEN 1 END) AS Negative_Count,
                   COUNT(CASE WHEN Sentiment = 'Neutral' THEN 1 END) AS Neutral_Count,
                   COUNT(*) as Total_Count
            FROM Devices_Sentiment_Data_New
            WHERE Aspect = '{aspect}' AND Product_Family LIKE '%{device}%'
            GROUP BY Aspect_Description
            ORDER BY Total_Count DESC;
            """

            # Execute the query and get the result in 'key_df'
            key_df = ps.sqldf(query, globals())

            # Calculate percentages and keyword contribution
            total_aspect_count = key_df['Total_Count'].sum()
            key_df['Positive_Percentage'] = (key_df['Positive_Count'] / total_aspect_count) * 100
            key_df['Negative_Percentage'] = (key_df['Negative_Count'] / total_aspect_count) * 100
            key_df['Neutral_Percentage'] = (key_df['Neutral_Count'] / total_aspect_count) * 100
            key_df['Aspect_Description_Contribution'] = (key_df['Total_Count'] / total_aspect_count) * 100

            # Drop the count columns
            key_df = key_df.drop(['Positive_Count', 'Negative_Count', 'Neutral_Count', 'Total_Count'], axis=1)

            # Add the current aspect to the DataFrame
            key_df['Aspect'] = aspect

            # Sort by 'Keyword_Contribution' and select the top 2 for the current aspect
            key_df = key_df.sort_values(by='Aspect_Description_Contribution', ascending=False).head(2)

            # Append the results to the final DataFrame
            final_df = pd.concat([final_df, key_df], ignore_index=True)
    except:
        print(f"Error in get_final_df_devices_new() for Aspect list: {str(aspects_list)} and device: {device}")
        final_df = pd.DataFrame()
        
    return final_df

def get_conversational_chain_detailed_summary_devices_new():
    try:
        
        prompt_template = """
        1. Your Job is to analyse the Net Sentiment, Aspect wise sentiment and Key word regarding the different aspect and summarize the reviews that user asks for utilizing the reviews and numbers you get. Use maximum use of the numbers and Justify the numbers using the reviews.
        
        Your will receive Aspect wise net sentiment of the device. you have to concentrate on top 4 Aspects.
        For that top 4 Aspect you will get top 2 Aspect_Description (which you can consider as summary) for each aspect. You will receive each Aspect_Description's contribution and +ve mention % and negative mention %
        You will receive reviews of that devices focused on these aspects and Aspect_Descriptions.
        
        For Each Aspect
        
        Condition 1 : If the net sentiment is less than aspect sentiment, which means that particular aspect is driving the net sentiment Higher for that device. In this case provide why the aspect sentiment is lower than net sentiment.
        Condition 2 : If the net sentiment is high than aspect sentiment, which means that particular aspect is driving the net sentiment Lower for that device. In this case provide why the aspect sentiment is higher than net sentiment. 

            IMPORTANT: Use only the data provided to you and do not rely on pre-trained documents.

            Your summary should justify the above conditions and tie in with the net sentiment and aspect sentiment and Aspect_Description. Mention the difference between Net Sentiment and Aspect Sentiment (e.g., -2% or +2% higher than net sentiment) in your summary and provide justification.
            
            Your response should be : 
            
            For Each Aspect 
                    Net Sentiment of the device and aspect sentiment of that aspect of the device (Mention Performance, Aspect Sentiment) . 
                    Top Aspect_Description contribution and their positive and negative percentages and summarize Reviews what user have spoken regarding this Aspect_Description in 2 to 3 lines detailed
                    Top 2nd Aspect_Description contribution and their positive and negative percentages and summarize Reviews what user have spoken regarding this Aspect_Description in 2 to 3 lines detailed
                       Limit yourself to top 3 Aspect_Description and don't mention as top 1, top 2, top 3 and all. Mention them as pointers
                    Overall Summary
            
            IMPORTANT : Example Template :
            
            ALWAYS FOLLOW THIS TEMPLATE : Don't miss any of the below:
                                    
            Response : "BOLD ALL THE NUMBERS"
            
            IMPOPRTANT : Start with : "These are the 4 major aspects users commented about" and mention their review count contributions
               
                           These are the 4 major aspects users commented about:
                           
                        - Total Review for Vivobook Device is 1200
                        - Price: 13.82% of the reviews mentioned this aspect
                        - Performance: 11.08% of the reviews mentioned this aspect
                        - Software: 9.71% of the reviews mentioned this aspect
                        - Design: 7.37% of the reviews mentioned this aspect

                        Price:
                        - The aspect sentiment for price is 52.8%, which is higher than the net sentiment of 38.5%. This indicates that the aspect of price is driving the net sentiment higher for the Vivobook.
                        -  The top Aspect_Description for price is "Budget friendly" with a contribution of 28.07%.
                              - Users mentioned that the Vivobook offers good value for the price and is inexpensive.
                        - Another top Aspect_Description for price is "Affordable Price" with a contribution of 26.89%. 
                            - Users praised the affordable price of the Vivobook and mentioned that it is worth the money.

                        Performance:
                        - The aspect sentiment for performance is 36.5%, which is lower than the net sentiment of 38.5%. This indicates that the aspect of performance is driving the net sentiment lower for the Vivobook.
                        - The top Aspect_Description for performance is "Reliable Performance" with a contribution of 18.24%.
                            - Users mentioned that the Vivobook is fast and offers good speed.
                        - Another top Aspect_Description for performance is "Performance Issues" with a contribution of 12.06%.
                            - Users have reported certain performance issues for Vivobook.
                                            
                                            
                        lIKE THE ABOVE ONE EXPLAIN OTHER 2 ASPECTS

                        Overall Summary:
                        The net sentiment for the Vivobook is 38.5%, while the aspect sentiment for price is 52.8%, performance is 36.5%, software is 32.2%, and design is 61.9%. This indicates that the aspects of price and design are driving the net sentiment higher, while the aspects of performance and software are driving the net sentiment lower for the Vivobook. Users mentioned that the Vivobook offers good value for the price, is fast and efficient in performance, easy to set up and use in terms of software, and has a sleek and high-quality design.
  
                        Some Pros and Cons of the device, 
                        
                        
           IMPORTANT : Do not ever change the above template of Response. Give Spaces accordingly in the response to make it more readable.
           
           A Good Response should contains all the above mentioned poniters in the example. 
               1. Net Sentiment and The Aspect Sentiment
               2. Total % of mentions regarding the Aspect
               3. A Quick overview of whether the aspect is driving the sentiment high or low
               4. Top Aspect: Gaming (Contribution: 33.22%, Positive: 68.42%, Negative: 6.32%)
                    - Users have praised the gaming experience on the Lenovo Legion, with many mentioning the smooth gameplay and high FPS.
                    - Some users have reported experiencing lag while gaming, but overall, the gaming performance is highly rated.
                    
                Top 3 Aspect_Descriptions : Their Contribution, Postitive mention % and Negative mention % and one ot two positive mentions regarding this Aspect_Descriptions in each pointer
                
                5. IMPORTANT : Pros and Cons in pointers (overall, not related to any aspect)
                6. Overall Summary

                    
          Enhance the model’s comprehension to accurately interpret user queries by:
          Recognizing abbreviations for country names (e.g., ‘DE’ for Germany, ‘USA’or 'usa' or 'US' for the United States of America) and expanding them to their full names for clarity.
          Understanding product family names even when written in reverse order or missing connecting words (e.g., ‘copilot in windows 11’ as ‘copilot windows’ and ‘copilot for security’ as ‘copilot security’ etc.).
          Utilizing context and available data columns to infer the correct meaning and respond appropriately to user queries involving variations in product family names or geographical references
          Please provide a comprehensive Review summary, feature comparison, feature suggestions for specific product families and actionable insights that can help in product development and marketing strategies.
          Generate acurate response only, do not provide extra information.
          \nFollowing is the previous conversation from User and Response, use it to get context only:""" + str(st.session_state.context_history_devices) + """\n
                Use the above conversation chain to gain context if the current prompt requires context from previous conversation.\n. When user asks uses references like "from previous response", ":from above response" or "from above", Please refer the previous conversation and respond accordingly.\n
            
            Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(
            azure_deployment=azure_deployment_name,
            api_version='2023-12-01-preview',
            temperature = 0.0)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except:
        err = f"An error occurred while getting conversation chain for detailed review summarization."
        print(f"Error in get_conversational_chain_detailed_summary_devices_new()")
        return err

# Function to handle user queries using the existing vector store
def query_detailed_summary_devices_new(user_question, vector_store_path="faiss_index_windows_116K_Summary_Final"): #NEED TO CHANGE
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment=azure_embedding_name)
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_detailed_summary_devices_new()
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        err = f"An error occurred while getting LLM response for detailed review summarization."
        print(f"Error in query_detailed_summary_devices_new() for {user_question}")
        print(f"\n\n Error: {e}\n\n")
        return err

        
def get_detailed_summary_new(device_name):
    try:
        if device_name:
            data = query_quant_devices_new("Summarize the reviews of "+ device_name)
            total_reviews = data.loc[data['ASPECT'] == 'TOTAL', 'REVIEW_COUNT'].iloc[0]
            data['REVIEW_PERCENTAGE'] = data['REVIEW_COUNT'] / total_reviews * 100
            dataframe_as_dict = data.to_dict(orient='records')
            data_new = data
            data_new = data_new.dropna(subset=['ASPECT_SENTIMENT'])
            data_new = data_new[~data_new["ASPECT"].isin(["Generic", "Account", "Customer-Service", "Browser"])]
            vmin = data_new['ASPECT_SENTIMENT'].min()
            vmax = data_new['ASPECT_SENTIMENT'].max()
            styled_df = data_new.style.applymap(lambda x: custom_color_gradient(x, vmin, vmax), subset=['ASPECT_SENTIMENT'])
            data_filtered = data_new[data_new['ASPECT'] != 'TOTAL']
            data_sorted = data_filtered.sort_values(by='REVIEW_COUNT', ascending=False)
            top_four_aspects = data_sorted.head(4)
            aspects_list = top_four_aspects['ASPECT'].to_list()
            formatted_aspects = ', '.join(f"'{aspect}'" for aspect in aspects_list)
            key_df = get_final_df_devices_new(aspects_list, device_name)
            b =  key_df.to_dict(orient='records')
            su = query_detailed_summary_devices_new("Summarize reviews of " + device_name + " for " +  formatted_aspects +  " Aspects which have following Sentiment Scores: "+str(dataframe_as_dict)+ str(b))
    except:
        su = "I don't have sufficient data to provide a complete and accurate response at this time. Please provide more details or context."
        print(f"Error in get_detailed_summary_new() for {device_name}")
    return su

def get_conversational_chain_summary_new():
    try:
        prompt_template = """
        Your task is to analyze the reviews of Windows products and generate a summary of the pros and cons for each product based on the provided dataset.Provide an overall summary. focus only on listing the pros and cons. 
        Use the format below for your response:

        Pros and Cons of [Product Name]:

        Pros:

        [Aspect]: [Brief summary of positive feedback regarding this aspect. Include specific examples if available.]
        [Aspect]: [Brief summary of positive feedback regarding this aspect. Include specific examples if available.]
        [Aspect]: [Brief summary of positive feedback regarding this aspect. Include specific examples if available.]
        [Aspect]: [Brief summary of positive feedback regarding this aspect. Include specific examples if available.]
        [Aspect]: [Brief summary of positive feedback regarding this aspect. Include specific examples if available.]
        Cons:

        [Aspect]: [Brief summary of negative feedback regarding this aspect. Include specific examples if available.]
        [Aspect]: [Brief summary of negative feedback regarding this aspect. Include specific examples if available.]
        [Aspect]: [Brief summary of negative feedback regarding this aspect. Include specific examples if available.]
        [Aspect]: [Brief summary of negative feedback regarding this aspect. Include specific examples if available.]
        [Aspect]: [Brief summary of negative feedback regarding this aspect. Include specific examples if available.]

        [Overall Summary]: [Brief summary of overall feedback regarding all aspect.]
        The dataset includes the following columns:

        Review: Review of the Windows product.
        Data_Source: Source of the review, containing different retailers.
        Geography: Country or region of the review.
        Title: Title of the review.
        Review_Date: Date the review was posted.
        Product: Product the review corresponds to, with values: "Windows 11 (Preinstall)", "Windows 10".
        Product_Family: Version or type of the corresponding product.
        Sentiment: Sentiment of the review, with values: 'Positive', 'Neutral', 'Negative'.
        Aspect: Aspect or feature of the product discussed in the review, with values: "Audio-Microphone", "Software", "Performance", "Storage/Memory", "Keyboard", "Browser", "Connectivity", "Hardware", "Display", "Graphics", "Battery", "Gaming", "Design", "Ports", "Price", "Camera", "Customer-Service", "Touchpad", "Account", "Generic".
        Keywords: Keywords mentioned in the review.
        Review_Count: Will be 1 for each review or row.
        Sentiment_Score: Will be 1, 0, or -1 based on the sentiment.
        Aspect_Description: two to four word summary of the review.

        Please ensure that the response is based on the analysis of the provided dataset, summarizing both positive and negative aspects of each product.
        Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.\n


        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        model = AzureChatOpenAI(
        azure_deployment=azure_deployment_name,
        api_version='2023-12-01-preview',temperature = 0)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    except:
        print(f"Error in get_conversational_chain_summary_new()")
    return chain

def query_to_embedding_summarize_new(user_question, txt_file_path):
    try:
        text = get_txt_text_new(txt_file_path)
        chunks = get_text_chunks_new(text)
        get_vector_store_new(chunks)
        embeddings = AzureOpenAIEmbeddings(azure_deployment=azure_embedding_name)

        # Load the vector store with the embeddings model
        new_db = FAISS.load_local("faiss-index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain_summary_new()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        output = response['output_text']
    except:
        print(f"Error in query_to_embedding_summarize_new() for user question: {user_question} and text file path: {txt_file_path}")
        output = ""
    return output

#-----------------------------------------------------Quant and Visualization-------------------------------------------------------#


def get_conversational_chain_quant_classify2_devices_new():
    #global model
    try:
        prompt_template = """         
                    
                    
                    You are an AI Chatbot assistant. Understand the user question carefully and follow all the instructions mentioned below. The data contains different devices for Windows. 
                
                    1. Your Job is to convert the user question to SQL Query (Follow Microsoft SQL server SSMS syntax.). You have to give the query so that it can be used on Microsoft SQL server SSMS.You have to only return query as a result.
                    2. There is only one table with table name Devices_Sentiment_Data_New where each row is a user review. The table has 10 columns, they are-
                        Review: Review of the Copilot Product and Competitors of Copilot
                        Data_Source: From where is the review taken. It contains different retailers - It contains following values : [chinatechnews, DigitalTrends, Engadget, clubic, g2.com, gartner, JP-ASCII, Jp-Impresswatch, Itmedia, LaptopMag, NotebookCheck, PCMag, TechAdvisor, TechRadar, TomsHardware, TechCrunch, Verge, ZDNET, PlayStore, App Store, AppStore, Reddit, YouTube, Facebook, Instagram, X, VK, Forums, News, Print, Blogs/Websites, Reviews, Wordpress, Podcast, TV, Quora, LinkedIn, Videos]
                        Geography: From which Country or Region the review was given. It contains different Geography.
                                   list of Geographies in the table - Values in this column 
                                   [ 'China', 'Unknown', 'France', 'Japan', 'US', 'Australia', 'Brazil',
       'Canada', 'Germany', 'India', 'Mexico', 'UK' ]
                        Title: What is the title of the review
                        Review_Date: The date on which the review was posted
                        Product: Corresponding product for the review. It contains following values: "Windows 10" and "Windows 11".
                        
                        Product_Family: Which version or type of the corresponding Product was the review posted for. Different Product Names  - It contains following Values - ['ASUS ROG ZEPHYRUS G16 16', 'HP OMEN TRANSCEND 14',
       'LENOVO LEGION PRO 5I 16', 'ASUS ROG STRIX SCAR 18',
       'ASUS VIVOBOOK PRO 16', 'ASUS ROG ZEPHYRUS G14 14',
       'LENOVO SLIM 7I 16', 'ASUS ZENBOOK 14', 'DELL G16 16',
       'ASUS ROG STRIX 18', 'RAZER BLADE 16', 'MSI PRESTIGE 16',
       'HP LAPTOP 14', 'DELL G15 15', 'HP SPECTRE X360 16', 'HP OMEN 14',
       'RAZER BLADE 14', 'HP SPECTRE X360 14', 'HP PAVILION 14',
       'LENOVO V15 15', 'HP G10 15', 'LENOVO THINKPAD E16 16', 'HP G9 15',
       'LENOVO V14 14', 'LENOVO IDEAPAD 1 14', 'LENOVO THINKPAD P17 17',
       'HP LAPTOP 17', 'SAMSUNG GALAXY BOOK3 360 15',
       'LENOVO IDEAPAD SLIM 3 15', 'ACER ASPIRE 5 15',
       'DELL ALIENWARE X16 16', 'ASUS ROG STRIX G17 17',
       'ASUS TUF DASH 15', 'ACER ASPIRE VERO 14',
       'MICROSOFT SURFACE LAPTOP 4 15', 'LENOVO IDEAPAD 1 15',
       'HP LAPTOP 15', 'LENOVO IDEAPAD 3 15',
       'MICROSOFT SURFACE LAPTOP GO 3 12', 'ASUS ZENBOOK DUO 14',
       'HP STREAM 14', 'ASUS LAPTOP 15', 'HP PAVILION 15',
       'DELL INSPIRON 3000 15', 'LENOVO SLIM 7 PRO X 14', 'LG GRAM 17',
       'ACER SWIFT GO 14', 'MSI STEALTH 16', 'MSI TITAN 18',
       'ASUS TUF GAMING F15 15', 'DELL ALIENWARE M18 18',
       'ASUS VIVOBOOK 15', 'DELL PRECISION 3000 15', 'HP ENVY X360 15',
       'ASUS ZENBOOK 14X 14', 'HP OMEN TRANSCEND 16', 'HP ENVY 17T 17',
       'ASUS ROG STRIX G16 16', 'HP ENVY 16', 'HP VICTUS 15',
       'LENOVO IDEAPAD 14', 'MICROSOFT SURFACE LAPTOP 15',
       'LENOVO THINKPAD YOGA L13 13', 'ASUS ROG STRIX G15 15',
       'ASUS ZENBOOK PRO 14X 14', 'RAZER BLADE GAMING 15',
       'HP PAVILION 17', 'ASUS ROG STRIX G17  17',
       'ASUS TUF GAMING A15 15', 'MSI GAMING CYBORG 15',
       'ACER PREDATOR HELIOS 16', 'ASUS TUF GAMING A17 17',
       'ACER NITRO 5 15', 'ASUS VIVOBOOK FLIP 16',
       'LENOVO THINKPAD P14S 14', 'DELL XPS 13', 'ASUS ROG FLOW Z13 13',
       'MSI THIN GAMING GF63 15', 'DELL INSPIRON 15',
       'MICROSOFT SURFACE LAPTOP GO 12', 'LENOVO IDEAPAD 3I 15',
       'LENOVO LAPTOP 15', 'LENOVO LEGION PRO 7 16',
       'ASUS ROG ZEPHYRUS M16 16', 'LENOVO YOGA 7I 14',
       'SAMSUNG GALAXY BOOK4 PRO 14', 'HP ENVY X360 16',
       'ASUS ZENBOOK PRO 14', 'LENOVO THINKBOOK G6 16',
       'LENOVO THINKPAD X1 CARBON 14', 'SAMSUNG GALAXY BOOK2 PRO 360 13',
       'MICROSOFT SURFACE LAPTOP STUDIO 14', 'LENOVO THINKPAD YOGA X1 14',
       'SAMSUNG GALAXY BOOK4 PRO 16', 'MICROSOFT SURFACE LAPTOP GO 2 12',
       'DELL INSPIRON 14', 'HP ENVY X360 14', 'MICROSOFT SURFACE GO 13',
       'HP OMEN  15', 'DELL INSPIRON 3000  15', 'LENOVO IDEAPAD  15',
       'LENOVO SLIMPRO 9I 14', 'DELL INSPIRON 16',
       'HP OMEN GAMING LAPTOP', 'ASUS VIVOBOOK 16', 'HP ENVY 2',
       'HP ENVY  17', 'ASUS ROG FLOW 16', 'ASUS TUF GAMING A16 16',
       'ASUS ROG GAMING 14', 'LENOVO LOQ 15', 'LENOVO YOGA 7 16',
       'LENOVO SLIM PRO 14', 'LENOVO YOGA 6 13', 'LENOVO YOGA 7I 16',
       'LENOVO YOGA BOOK 9', 'LENOVO IDEAPAD 15',
       'SAMSUNG GALAXY BOOK 3 360 15', 'MICROSOFT SURFACE LAPTOP 5 13',
       'MICROSOFT SURFACE PRO 9 13', 'MICROSOFT SURFACE LAPTOP 5 15',
       'SAMSUNG GALAXY BOOK 2 PRO 360 15',
       'MICROSOFT SURFACE PRO 7 PLUS 12', 'MICROSOFT SURFACE LAPTOP 4 13',
       'LENOVO IDEAPAD 1  14', 'ALIENWARE M16 16', 'DELL XPS 15',
       'HP OMEN 16', 'ASUS VIVOBOOK 14', 'LENOVO YOGA 9I 14',
       'SAMSUNG GALAXY BOOK3 PRO 16', 'SAMSUNG GALAXY BOOK 3 PRO 360 16',
       'ACER ASPIRE 3 15', 'HP PAVILION X360 14',
       'MICROSOFT SURFACE STUDIO 2 14', 'HP STREAM 11',
       'ASUS VIVOBOOK S 15', 'LENOVO IDEAPAD 16', 'ASUS VIVOBOOK 17',
       'HP LAPTOP  17', 'ASUS ROG ZEPHYRUS 16',
       'SAMSUNG GALAXY BOOK3 ULTRA 16', 'DELL INSPIRON 3000 14',
       'ACER PREDATOR HELIOS 18', 'LENOVO THINKPAD X1 14',
       'ACER NITRO 17', 'MICROSOFT SURFACE GO LAPTOP 3 12',
       'MSI STEALTH 14', 'ASUS PROART STUDIOBOOK 16',
       'ASUS ROG ZEPHYRUS 14', 'ASUS ROG 16', 'LENOVO GAMING LAPTOP',
       'SAMSUNG GALAXY BOOK 2 PRO 13', 'ASUS VIVOBOOK  16',
       'HP NOTEBOOK 15', 'HP LAPTOP  15', 'DELL INSPIRON 5000  16',
       'MICROSOFT SURFACE GO 2 10', 'DELL INSPIRON 7000 14 PLUS',
       'ASUS TUF A16 16', 'LENOVO LEGION 5  14', 'ASUS LAPTOP 11',
       'LENOVO LEGION SLIM 5 16', 'MSI BRAVO 15',
       'ACER PREDATOR HELIOS NEO 16', 'LENOVO LEGION 16',
       'SAMSUNG GALAXY TAB S7 FE 12', 'ACER ASPIRE 1 15',
       'ASUS TUF GAMING A 17', 'DELL ALIENWARE  16',
       'SAMSUNG GALAXY BOOK 3 360 13', 'LENOVO IDEAPAD 3I 14',
       'ASUS BR1100 11', 'LENOVO IDEAPAD 3 14', 'ACER ASPIRE VERO 15',
       'LENOVO YOGA 9I 13', 'MICROSOFT SURFACE LAPTOP STUDIO2 13',
       'ASUS ZENBOOK PRO DUO 15', 'LENOVO WEI 9 PRO  16',
       'HP ENVY X360  15', 'ASUS ROG FLOW X13', 'LENOVO WEI 7 PRO  14',
       'ASUS TUF GAMING 15', 'HP SPECTRE X360 13',
       'MICROSOFT SURFACE GO 3 10', 'SAMSUNG GALAXY ULTRA 16',
       'ASUS VIVOBOOK PRO 15', 'ASUS VIVOBOOK PRO 16X  16',
       'ACER PREDATOR HELIOS 300 15', 'ASUS ROG STRIX G 17',
       'LENOVO IDEAPAD FLEX 5 14', 'ASUS L210 11',
       'LENOVO IDEAPAD GAMING 3 15', 'ASUS ROG STRIX G 16',
       'LENOVO IDEAPAD 1I 14', 'HP ENVY 17', 'LENOVO IDEAPAD 3 17',
       'ASUS ROG ZEPHYRUS G14  14', 'LENOVO FLEX 5 14',
       'MICROSOFT SURFACE PRO 8 13', 'ACER SWIFT EDGE 16',
       'ASUS ROG FLOW X13 13', 'ALIENWARE M18 18',
       'ASUS TUF GAMING A15  15', 'LENOVO LEGION PRO 7I 16',
       'ASUS VIVOBOOK GO 15', 'ACER SWIFT 3 14', 'ASUS ROG STRIX 17',
       'MSI CYBORG 15', 'HP STREAM  14', 'ASUS VIVOBOOK 11',
       'DELL G15 5000 15', 'SAMUSNG GALAXY BOOK 3 PRO 14',
       'DELL INSPIRON 7000 2-IN-1 16', 'MSI GAMING LAPTOP 16',
       'MSI STEALTH 17', 'MSI GAMING 15', 'DELL ALIENWARE M16 16',
       'LENOVO IDEAPAD FLEX 5  14', 'ASUS TUF GAMING F15',
       'LENOVO IDEAPAD FLEX 5 16', 'LENOVO LAPTOP 14',
       'MSI GAMINGTHIN 15', 'ASUS ROG ZEPHYRUS DUO 16', 'ACER SWIFT X 14',
       'MSI GF63 THIN 15', 'MICROSOFT SURFACE PRO 7 12', 'LENOVO FLEX 11',
       'LENOVO WEI 5 PRO  15', 'ASUS VIVOBOOK GO 14', 'ASUS VIVOBOOK  14',
       'MSI KATANA 15', 'HP LAPTOP 17T 17', 'ASUS VIVOBOOK GO 12',
       'ASUS L510 15', 'MICROSOFT SURFACE GO 10', 'DELL INSPRION 3000 11',
       'ASUS ZENBOOK PRO 15', 'DELL ALIENWARE M15 R7 15',
       'HP ENVY DESKTOP', 'ASUS ZENBOOK DUO 15', 'LG GRAM 16',
       'ASUS ROG STRIX SCAR 15', 'ASUS TUF GAMING F17 17', 'MSI SWORD 15',
       'ACER NITRO 5 17', 'DELL XPS 17', 'LENOVO LEGION 5 15',
       'RAZER GAMING LAPTOP 18', 'ASUS VIVOBOOK PRO  16',
       'DELL INSPIRON 5000 14', 'SAMSUNG GALAXY BOOK3 15',
       'HP ELITEBOOOK G7 14', 'LENOVO YOGA C740 15',
       'LENOVO YOGA SLIM 7 16', 'LENOVO SLIM 7 PRO 14', 'LG GRAM 14',
       'HP SPECTRE 17', 'ASUS ZENBOOK PRO DUO 14',
       'ASUS ROG ZEPHYRUS G16  16', 'HP VICTUS 16', 'MSI SUMMIT FLIP 14',
       'ASUS ZENBOOK PRO 17', 'ACER NITRO 16', 'LENOVO THINKPAD T16 16',
       'ASUS ZENBOOK S 13', 'MSI RAIDER 17', 'ASUS ROG STRIX SCAR 16',
       'MSI VECTOR 16', 'RAZER BLADE 15', 'DELL ALIENWARE M17 R5 17',
       'DELL XPS PLUS 13', 'LENOVO GAMING DESKTOP',
       'ASUS ROG FLOW X16 16', 'HP STREAM  17', 'ASUS ZENBOOK FLIP 14',
       'RAZER BLADE 18', 'MSI THIN GF63 15', 'ASUS VIVOBOOK PRO 16X 16',
       'ASUS VIVOBOOK FLIP 14', 'ASUS VIVOBOOK PRO X 16',
       'DELL INSPIRON 5620 DESKTOP', 'LENOVO 300W 3 11',
       'ASUS ZENBOOK 15', 'ACER ASPIRE 3 14',
       'SAMSUNG GALAXY BOOK2 PRO 15', 'ASUS VIVOBOOK PRO 14',
       'ASUS VIVOBOOK PRO 15X', 'SAMSUNG GALAXY BOOK 2 PRO 15',
       'MSI GV15 15', 'ASUS ROG STRIX G15  15', 'LENOVO IDEAPAD 3  15',
       'LG GRAM 15', 'SAMSUNG GALAXY BOOK3 16', 'LG GRAM 17Z95P',
       'LENOVO IDEAPAD FLEX 5 15', 'ASUS ROG STRIX G16   16',
       'LENOVO THINKPAD YOGA 11E', 'MSI CREATOR M 16',
       'LENOVO IDEAPAD 5I PRO 16', 'ASUS E410 14',
       'DELL INSPIRON 15 3000', 'SAMSUNG GALAXY BOOK3 PRO 15',
       'DELL INSPIRON 3501 15', 'ASUS VIVOBOOK 13',
       'MICROSOFT SURFACE LAPTOP 3 13', 'LENOVO IDEAPAD 5I 15',
       'DELL ALIENWARE X15 R2 15', 'ASUS VIVOBOOK PRO 15X 15',
       'SAMSUNG GALAXY BOOK GO 14']
                        Sentiment: What is the sentiment of the review. It contains following values: 'positive', 'neutral', 'negative'.
                        Aspect: The review is talking about which aspect or feature of the product. It contains following values: 'Interface', 'Connectivity', 'Privacy','Compatibility', 'Generic', 'Innovation', 'Reliability','Productivity', 'Price', 'Text Summarization/Generation','Code Generation', 'Ease of Use', 'Performance','Personalization/Customization','Accessibility'.
                        Keywords: What are the keywords mentioned in the product
                        Aspect_Description : It provides a summary of the particular review in two to three words
                        Review_Count - It will be 1 for each review or each row
                        Sentiment_Score - It will be 1, 0 or -1 based on the Sentiment.
                        
                IMPORTANT : User won't exactly mention the exact Geography Names, Product Names, Product Families, Data Source name, Aspect names. Please make sure to change/correct to the values that you know from the context and then provide SQL Query.

                           
                User Question  : best AI based on text Generation : By this user meant : What is the best Product Families for text Generation aspect based on net sentiment?

                IMPORTANT : Consider Product_Family Column when user is asking about any type of laptop or device.
                Eg - What is the net sentiment for Inspiron - You should consider Product_Family column for Inspiron.


                IMPORTANT : These are the aspects we have : ['Audio-Microphone', 'Software', 'Performance', 'Storage/Memory',
       'Keyboard', 'Browser', 'Connectivity', 'Hardware', 'Display',
       'Graphics', 'Battery', 'Gaming', 'Design', 'Ports', 'Price',
       'Camera', 'Customer-Service', 'Touchpad', 'Account', 'Generic']
                    But user who writes the question might not know the exact name of the aspect we have, for example : User might write "Picture Generation" for 'Image Generarion' and "writing codes" for code generation. 
                    You should be carefull while rephrasing it. 

                IMPORTANT : User can confuse a lot, but understand the question carefully and respond:
                Example : I am a story writer , Can you tell me which AI is good for writing stories based on user reviews? -> In this case, user confuses by telling that he is a story teller and all but he just needs to know "What is the best AI for Text Generation" -> Which is again decided based on comparison.
                Same goes for all the columns
                                
IMPORTANT : If a user is asking about which is best/poor, everything should be based on net sentiment and Review count. So Give it as Quantifiable and Visualization.

                User Question  : best AI based on text Generation : By this user meant : What is the best Product Families for text Generation aspect based on net sentiment?


                
                IMPORTANT: In the SQL Queries, always use ORDER BY REVIEW COUNT DESC in all cases
                
                IMPORTANT: If the user is asking "Give or calculate net sentiment of Inspiron, the user means the product family Inspiron.
            
                    1. If the user asks for count of column 'X', the query should be like this:
                            SELECT COUNT(DISTINCT ('X')) 
                            FROM Devices_Sentiment_Data_New
                    2. If the user asks for count of column 'X' for different values of column 'Y', the query should be like this:
                            SELECT 'Y', COUNT(DISTINCT('X')) AS Total_Count
                            FROM Devices_Sentiment_Data_New
                            GROUP BY 'Y'
                            ORDER BY TOTAL_COUNT DESC
                    3. If the user asks for Net overall sentiment the query should be like this:
                            SELECT ((SUM(Sentiment_Score))/(SUM(Review_Count))) * 100 AS Net_Sentiment,  SUM(Review_Count) AS Review_Count
                            FROM Devices_Sentiment_Data_New 
                            ORDER BY Review_Count DESC

                    4. If the user asks for Net Sentiment for column "X", the query should be exactly like this: 

                            SELECT X, ((SUM(Sentiment_Score)) / (SUM(Review_Count))) * 100 AS Net_Sentiment, SUM(Review_Count) AS Review_Count
                            FROM Devices_Sentiment_Data_New 
                            GROUP BY X
                            ORDER BY Review_Count DESC


                    5. If the user asks for overall review count, the query should be like this:
                            SELECT SUM(Review_Count) 
                            FROM Devices_Sentiment_Data_New 
                    6. If the user asks for review distribution across column 'X', the query should be like this:
                            SELECT 'X', SUM(Review_Count) * 100 / (SELECT SUM(Review_Count) FROM Sentiment_Data) AS Review_Distribution
                            FROM Devices_Sentiment_Data_New 
                            GROUP BY 'X'
                            ORDER BY Review_Distribution DESC
                    7. If the user asks for column 'X' Distribution across column 'Y', the query should be like this: 
                            SELECT 'Y', SUM('X') * 100 / (SELECT SUM('X') AS Reviews FROM Sentiment_Data) AS Distribution_PCT
                            FROM Devices_Sentiment_Data_New  
                            GROUP BY 'Y'
                            ORDER BY Distribution_PCT DESC
                            
                    8. If the user asks for aspect wise summaries across a particular product family, or aspect summaries for different aspects of a product_family, the query should be like this:
                            SELECT Aspect, Aspect_Description
                            FROM Devices_Sentiment_Data_New  
                            GROUP BY Aspect, Aspect_Description
                            ORDER BY Aspect
                            
                    9. If the user is asking to give aspect wise summary and review counts for a particular product, then your query should look like this - 
                            SELECT Aspect, Aspect_Description, SUM(Review_Count)
                            FROM Devices_Sentiment_Data_New 
                            WHERE Product_Family LIKE '%X%' 
                            GROUP BY Aspect, Aspect_Description 
                            ORDER BY Review_Count DESC
                            
                    9. If the user asks for the best and worst feature of all aspects or the most liked and most concerning prospect of all aspects under a product family 'X', the query should be like this-
                            WITH C1 AS (
                                        SELECT Aspect, Aspect_Description, Review_Count,"Most Preferred" AS Feature 
                                        FROM
                                        (SELECT *, RANK() OVER (PARTITION BY Aspect ORDER BY Review_Count desc) AS FEATURE_RANK 
                                        FROM 
                                        (SELECT Aspect, Aspect_Description, COUNT(Review) AS Review_Count
                                        FROM Devices_Sentiment_Data_New
                                        WHERE Device_Family LIKE '%X%' AND Sentiment='Positive'
                                        GROUP BY Aspect, Aspect_Description
                                        ORDER BY Aspect ASC, Review_Count DESC))
                                        WHERE FEATURE_RANK=1
                                        ),
                                C2 AS (
                                        SELECT Aspect, Aspect_Description, Review_Count,"Most Concerning" AS Feature 
                                        FROM
                                        (SELECT *, RANK() OVER (PARTITION BY Aspect ORDER BY Review_Count desc) AS FEATURE_RANK 
                                        FROM 
                                        (SELECT Aspect, Aspect_Description, COUNT(Review) AS Review_Count
                                        FROM Devices_Sentiment_Data_New
                                        WHERE Device_Family LIKE '%X%' AND Sentiment='Negative'
                                        GROUP BY Aspect, Aspect_Description
                                        ORDER BY Aspect ASC, Review_Count DESC))
                                        WHERE FEATURE_RANK=1
                                        )
                                SELECT * 
                                FROM  
                                (SELECT * FROM C1 
                                UNION
                                SELECT * FROM C2)
                                ORDER BY Feature desc, Review_Count desc;
                                
                    IMPORTANT: If the user is asking any questions related to summary, you should always interpret summary as 'Aspect_Description' column
        

                    Important: While generating SQL query to calculate net_sentiment across column 'X' and 'Y', if 'Y' has less distinct values, keep your response like this - SELECT 'Y','X', ((SUM(Sentiment_Score)) / (SUM(Review_Count))) * 100 AS Net_Sentiment, SUM(Review_Count) AS Review_Count FROM Devices_Sentiment_Data_New GROUP BY 'Y','X'
                    
                    IMPORTANT: Always replace '=' operator with LIKE keyword.
                    IMPORTANT IMPORTANT : Always add '%' before and after filter value in LIKE OPERATOR for single or multiple WHERE conditions in the generated SQL query . Example LIKE 'Performance' should be replaced by LIKE '%Performance%'
                    
                    IMPORTANT : For example, if the SQL query is like - 'SELECT * FROM Devices_Sentiment_Data_New WHERE PRODUCT='ABC' AND GEOGRAPHY='US' ORDER BY Review_Count' , you should modify the query and share the output like this - 'SELECT * FROM Devices_Sentiment_Data_New WHERE PRODUCT LIKE '%ABC%' AND GEOGRAPHY LIKE '%US%' ORDER BY Review_Count'

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

        
        model = AzureChatOpenAI(
                     azure_deployment=azure_deployment_name,
                     api_version='2023-12-01-preview',temperature = 0)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except:
        err = f"An error occurred while getting conversation chain for quantifiable review summarization."
        print(f"Error in get_conversational_chain_quant_classify2_devices_new()")
        return err

def query_quant_classify2_devices_new(user_question, vector_store_path="faiss_index_windows_116K_Summary_Final"):
    try:
        # Initialize the embeddings model
        #st.write("inside query_quant_classify2_devices")
        embeddings = AzureOpenAIEmbeddings(azure_deployment="MV_Agusta")
        
        # Load the vector store with the embeddings model
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        
        # Rest of the function remains unchanged
        chain = get_conversational_chain_quant_classify2_devices_new()
        #st.write(chain)
        docs = []
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        SQL_Query = response["output_text"]
        #st.write(SQL_Query)
        SQL_Query = convert_top_to_limit_new(SQL_Query)
        SQL_Query = process_tablename_devices_new(SQL_Query,"Devices_Sentiment_Data_New")
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
        
        data = ps.sqldf(SQL_Query, globals())
        #st.write(data)
        data_1 = data
        html_table = data.to_html(index=False)
    #     return html_table
        
        #Calculate overall net-sentiment and review_count based on filtered data
        try:
            data2=Devices_Sentiment_Data_New.copy()
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
    except:
        err = f"An error occurred while generating response for Quantify."
        print(f"Error in query_quant_classify2_devices_new() for {user_question}")
        return err
    
    
def get_conversational_chain_detailed_summary2_devices_new():
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
                Based on the provided sentiment data for Microsoft Surface Pro reviews from different geographies, here is a summary:


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
                  - Utilizing context and available data columns to infer the correct meaning and respond appropriately to user queries involving variations in product family names or geographical references]
                 Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs\n
          \nFollowing is the previous conversation from User and Response, use it to get context only:""" + str(st.session_state.context_history_devices) + """\n
                Use the above conversation chain to gain context if the current prompt requires context from previous conversation.\n. When user asks uses references like "from previous response", ":from above response" or "from above", Please refer the previous conversation and respond accordingly.\n
                
                  Context:\n {context}?\n
                  Question: \n{question}\n

          Answer:
          """
            
        model = AzureChatOpenAI(
                     azure_deployment=azure_deployment_name,
                     api_version='2023-12-01-preview',temperature = 0.2)
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except:
        err = f"An error occurred while getting conversation chain for detailed review summarization."
        print(f"Error in get_conversational_chain_detailed_summary2_devices_new().")
        return err
    
def query_detailed_summary2_devices_new(dataframe_as_dict,user_question, history, vector_store_path="faiss_index_windows_116K_Summary_Final"):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="MV_Agusta")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_detailed_summary2_devices_new()
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except:
        err = generate_chart_insight_llm_devices_new(dataframe_as_dict)
        print(f"Error in query_detailed_summary2_devices_new()")
        return err    
    
def generate_chart_insight_llm_devices_new(user_question):
    #global model
    try:
        prompt_template = """
        
        1.Ensure to include all possible insights and findings that can be extracted, which reveals vital trends and patterns in the data. 
        2.Share the findings or insights in a format which makes more sense to business oriented users, and can generate vital action items for them. 
        3.For values like Net_Sentiment score, positive values indicate positive overall sentiment, negative values indicate negative overall sentiment and 0 value indicate neutral overall sentiment. For generating insights around net_sentiment feature, consider this information.
        IMPORTANT: If the maximum numerical value is less than or equal to 100, then the numerical column is indicating percentage results - therefore while referring to numbers in your insights, add % at the end of the number.
        IMPORTANT : Use the data from the input only and do not give information from pre-trained data.
        IMPORTANT : Dont provide any prompt message written here in the response, this is for your understanding purpose
          \nFollowing is the previous conversation from User and Response, use it to get context only:""" + str(st.session_state.context_history_devices) + """\n
                Use the above conversation chain to gain context if the current prompt requires context from previous conversation.\n. When user asks uses references like "from previous response", ":from above response" or "from above", Please refer the previous conversation and respond accordingly.\n
           
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        
        model = AzureChatOpenAI(
                     azure_deployment=azure_deployment_name,
                     api_version='2023-12-01-preview',temperature = 0.4)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        response = chain({"input_documents": [], "question": user_question}, return_only_outputs=True)
        #st.write("\n\n",response["output_text"])
        return response["output_text"]
            
    except:
        err = "Apologies, unable to generate insights based on the provided input data. Kindly refine your search query and try again!"
        print(f"Error in generate_chart_insight_llm_devices_new() for {user_question}")
        return err
    

def quantifiable_data_devices_new(user_question):
    try:
        #st.write("correct_func")
        response = query_quant_classify2_devices_new(user_question)
        
        return response
    except:
        err = f"An error occurred while generating quantitative review summarization."
        print(f"Error in quantifiable_data_devices_new() for {user_question}")
        return err    
    
    

    
#--------------------------------------------------------SALES-----------------------------------------------------------------------#  
    
    
def get_conversational_chain_quant_classify2_sales_new():
    try:

#################################################################################################################################################################################################################################################
        prompt_template = """
    1. Your Job is to convert the user question to SQL Query (Follow Microsoft SQL server SSMS syntax.). You have to give the query so that it can be used on Microsoft SQL server SSMS.You have to only return query as a result.
    2. There is only one table with table name RCR_Sales_Data where each row has. The table has 20 columns, they are:
        Month: Contains dates for the records
        Country: From where the sales has happened. It contains following values: 'Turkey','India','Brazil','Germany','Philippines','France','Netherlands','Spain','United Arab Emirates','Czech Republic','Norway','Belgium','Finland','Canada','Mexico','Russia','Austria','Poland','United States','Switzerland','Italy','Colombia','Japan','Chile','Sweden','Vietnam','Saudi Arabia','South Africa','Peru','Indonesia','Taiwan','Thailand','Ireland','Korea','Hong Kong SAR','Malaysia','Denmark','New Zealand','China' and 'Australia'.
        Geography: From which Country or Region the review was given. It contains following values: 'Unknown', 'Brazil', 'Australia', 'Canada', 'China', 'Germany','France'.
        OEMGROUP: OEM or Manufacturer of the Device. It contains following values: 'Lenovo','Acer','Asus','HP','All Other OEMs', 'Microsoft' and 'Samsung'
        SUBFORMFACTOR: Formfactor of the device. It contains following values: 'Ultraslim Notebook'.
        GAMINGPRODUCTS: Flag whether Device is a gaming device or not. It contains following values: 'GAMING', 'NO GAMING' and 'N.A.'.
        SCREEN_SIZE_INCHES: Screen Size of the Device.
        PRICE_BRAND_USD_3: Band of the price at which the device is selling. It contains following values: '0-300', '300-500', '500-800' and '800+.
        OS_VERSION: Operating System version intall on the device. It contains following values: 'Windows 11', 'Chrome', 'Mac OS'.
        Operating_System_Summary: Operating System installed on the device. This is at uber level. It contains following values: 'Windows', 'Google OS', 'Apple OS'.
        Sales_Units: Number of Devices sold for that device in a prticular month and country.
        Sales_Value: Revenue Generated by the devices sold.
        Series: Family of the device such as IdeaPad 1, HP Laptop 15 etc.
        Specs_Combination: Its contains the combination of Series, Processor, RAM , Storage and Screen Size. For Example: SURFACE LAPTOP GO | Ci5 | 8 GB | 256.0 SSD | 12" .
        Chassis Segment: It contains following values: 'SMB_Upper','Mainstream_Lower','SMB_Lower','Enterprise Fleet_Lower','Entry','Mainstream_Upper','Premium Mobility_Upper','Enterprise Fleet_Upper','Premium Mobility_Lower','Creation_Lower','UNDEFINED','Premium_Mobility_Upper','Enterprise Work Station','Unknown','Gaming_Musclebook','Entry_Gaming','Creation_Upper','Mainstrean_Lower'
        
    3.  When Asked for Price Range you have to use ASP Column to get minimum and Maxium value. Do not consider Negative Values. Also Consider Sales Units it shouldn't be 0.
        Exaple Query:
            SELECT MIN(ASP) AS Lowest_Value, MAX(ASP) AS Highest_Value
            FROM RCR_Sales_Data
            WHERE
            Series = 'Device Name'
            AND ASP >= 0
            AND Sales_Units <> 0;
    4. Total Sales_Units Should Always be in Thousands. 
        Example Query:
            SELECT (SUM(Sales_Units) / 1000) AS "TOTAL SALES UNITS"
            FROM RCR_Sales_Data
            WHERE
            SERIES LIKE '%SURFACE LAPTOP GO%';
    5. Average Selling Price (ASP): It is calculated by sum of SUM(Sales_Value)/SUM(Sales_Units)
    6. Total Sales Units across countries or across regions is sum of sales_units for those country. It should be in thousand of million hence add "K" or "M" after the number.
        Example to calculate sales units across country:
            SELECT Country, (SUM(Sales_Units) / 1000) AS "Sales_Units(In Thousands)"
            FROM RCR_Sales_Data
            GROUP BY Country
            ORDER BY Sales_Units DESC
    7. Total Sales Units across column "X" or across regions is sum of sales_units for those country. It should be in thousand of million hence add "K" or "M" after the number.
        Example to calculate sales units across country:
            SELECT "X", (SUM(Sales_Units) / 1000) AS "Sales_Units(In Thousands)"
            FROM RCR_Sales_Data
            GROUP BY "X"
            ORDER BY Sales_Units DESC
    8. If asked about the highest selling Specs Combination. 
        Example Query:
            SELECT Specs_Combination, (SUM(Sales_Units) / 1000) AS "TOTAL SALES UNITS"
            FROM RCR_Sales_Data
            WHERE SERIES LIKE '%Macbook AIR%'
            AND SALES_UNITS <> 0
            GROUP BY Specs_Combination
            ORDER BY "TOTAL SALES UNITS" DESC
            LIMIT 1;
            
    9. If asked about monthly ASP or Average Selling Price, the query should be : 
           SELECT MONTH,AVG(ASP) AS Average Selling Price
           FROM RCR_Sales_Data
           WHERE SALES_UNITS <> 0
           GROUP BY MONTH
           
    10. If asked about similar compete devices.
    Example Query:
            SQL = WITH DeviceNameASP AS (
                    SELECT
                        'Device Name' AS Series,
                        SUM(Sales_Value) / SUM(Sales_Units) AS ASP,
                        Chassis_Segment,
                        SUM(Sales_Units) AS Sales_Units
                    FROM
                        RCR_Sales_Data
                    WHERE
                        Series LIKE '%Device Name%'
                    GROUP BY
                        Chassis_Segment
                ),
                CompetitorASP AS (
                    SELECT
                        Series,
                        SUM(Sales_Value) / SUM(Sales_Units) AS ASP,
                        Chassis_Segment,
                        SUM(Sales_Units) AS Sales_Units
                    FROM
                        RCR_Sales_Data
                    WHERE
                        Operating_System_Summary IN ('Apple OS', 'Google OS','Windows OS')
                        AND SERIES NOT LIKE '%Device Name%'
                    GROUP BY
                        Series, Chassis_Segment
                ),
                RankedCompetitors AS (
                    SELECT
                        C.Series,
                        C.ASP,
                        C.Chassis_Segment,
                        C.Sales_Units,
                        ROW_NUMBER() OVER (PARTITION BY C.Chassis_Segment ORDER BY C.Sales_Units DESC) AS rank
                    FROM
                        CompetitorASP C
                    JOIN
                        DeviceNameASP S
                    ON
                        ABS(C.ASP - S.ASP) <= 400
                        AND C.Chassis_Segment = S.Chassis_Segment
                )
                SELECT
                    Series,
                    ASP AS CompetitorASP,
                    Sales_Units
                FROM
                    RankedCompetitors
                WHERE
                    rank <= 3;

    11. If asked about dates or year SUBSTR() function instead of Year() or Month()
    12. Convert numerical outputs to float upto 2 decimal point.
    13. Always include ORDER BY clause to sort the table based on the aggregate value calculated in the query.
    14. Always use 'LIKE' operator whenever they mention about any Country, Series. Use 'LIMIT' operator instead of TOP operator.Do not use TOP OPERATOR. Follow syntax that can be used with pandasql.
    15. If you are using any field in the aggregate function in select statement, make sure you add them in GROUP BY Clause.
    16. Make sure to Give the result as the query so that it can be used on Microsoft SQL server SSMS.
    17. Always use LIKE function instead of = Symbol while generating SQL Query
    18. Important: User can ask question about any categories including Country, OEMGROUP,OS_VERSION etc etc. Hence, include the in SQL Queryif someone ask it.
    19. Important: Use the correct column names listed above. There should not be Case Sensitivity issue. 
    20. Important: The values in OPERATING_SYSTEM_SUMMARY are ('Apple OS', 'Google OS') not ('APPLE OS', 'GOOGLE OS'). So use exact values. Not everything should be capital letters.
    21. Important: You Response should directly starts from SQL query nothing else.
    22. IMPORTANT: When asked about Average Selling Price for every month, always use AVG(ASP) as aggregation
    
                Context:\n {context}?\n
                Question: \n{question}\n

                Answer:
    
               """
########################################################################################################################################
#########################################################################################
        

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(
            azure_deployment=azure_deployment_name,
            api_version='2024-03-01-preview',
            temperature = 0.1)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except:
        err = f"An error occurred while getting conversation chain for quantifiable review summarization."
        print(f"Error in get_conversational_chain_quant_classify2_sales_new() ")
        return err


def query_quant_classify2_sales_new(user_question, vector_store_path="faiss_index_windows_116K_Summary_Final"):
    try:
        # Initialize the embeddings model
        embeddings = AzureOpenAIEmbeddings(azure_deployment="MV_Agusta")
        
        # Load the vector store with the embeddings model
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        
        # Rest of the function remains unchanged
        chain = get_conversational_chain_quant_classify2_sales_new()
        docs = []
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        SQL_Query = response["output_text"]
        # st.write(SQL_Query)
        SQL_Query = convert_top_to_limit_new(SQL_Query)
        SQL_Query = process_tablename_devices_new(SQL_Query,"RCR_Sales_Data")
        #st.write(SQL_Query)
        data = ps.sqldf(SQL_Query, globals())
        data_1 = data
        html_table = data.to_html(index=False)
    #     return html_table
        return data_1
    except:
        err = f"An error occurred while generating response for quantitative review summarization."
        print(f"Error in query_quant_classify2_sales_new() for {user_question}")
        return err    
    
#------------------------------------------------------------------------------------------------------------------------------------#


def generate_device_details_new(device_input):
    global interaction
    try:
        device_name, img_link = get_device_image_new(device_input)
        net_Sentiment,aspect_sentiment = get_net_sentiment_new(device_name)
        sales_device_name = get_sales_device_name_new(device_name)
        total_sales = get_sales_units_new(sales_device_name)
        asp = get_ASP_new(sales_device_name)
        high_specs, sale = get_highest_selling_specs_new(sales_device_name)
        star_rating_html = get_star_rating_html_new(net_Sentiment)
    #     st.write(f"Sales Device Name: {sales_device_name}")
        comp_devices = compete_device_new(sales_device_name)
        interaction = ""
        return device_name, img_link, net_Sentiment, aspect_sentiment, total_sales, asp, high_specs, sale, star_rating_html, comp_devices
    except:
        print(f"Error in generate_device_details_new() for {device_input}")
        return device_input, None, None, None, None, None, "NA", None, "NA", None

def load_and_resize_image_new(url, new_height):
    try:
        img = Image.open(url)
        aspect_ratio = img.width / img.height
        new_width = int(aspect_ratio * new_height)
        resized_img = img.resize((new_width, new_height))
        return resized_img  # Return the resized PIL image object
    except:
        st.write("Image not available for this product.")
        return None
    
def get_txt_text_new(txt_file_path):
    with io.open(txt_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def get_text_chunks_new(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store_new(chunks):
    embeddings = AzureOpenAIEmbeddings(azure_deployment=azure_embedding_name)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss-index")
    
def device_details_new(device):
    try:
        device_name, img_link, net_Sentiment, aspect_sentiment, total_sales, asp, high_specs, sale, star_rating_html, comp_devices = generate_device_details_new(device)
        aspects = ['Performance', 'Design', 'Display', 'Battery', 'Price', 'Software']
        with st.container(border = True):
            if device_name:
                min_date, max_date = get_date_range_new(device_name)
                with st.container(border = False,height = 200):
                    col1, inter_col_space, col2 = st.columns((1, 4, 1))
                    with inter_col_space:
                        if img_link:
                            image1 = load_and_resize_image_new(img_link, 150)
                            st.image(image1)
                        else:
                            st.write("Image not available for this product.")
                with st.container(height=170, border = False):
                    st.header(device_name)
                with st.container(height=50, border = False):
                    st.markdown(star_rating_html, unsafe_allow_html=True)
                with st.container(height=225, border = False):
                    st.write(f"Total Devices Sold: {total_sales}")
                    st.write(f"Average Selling Price: {asp}")
                    st.write(f"Highest Selling Specs: {high_specs} - {sale}")
                    st.markdown(f"<p style='font-size:12px;'>*sales data is from {min_date} to {max_date}</p>", unsafe_allow_html=True)
                with st.container(height=300, border = False):
                    st.subheader('Aspect Ratings')
                    asp_rating = []
                    for i in aspect_sentiment:
                        asp_rating.append(get_star_rating_html_new(i))
                    for aspect, stars in zip(aspects, asp_rating):
                        st.markdown(f"{aspect}: {stars}",unsafe_allow_html=True)
                data_1 = Devices_Sentiment_Data_New.loc[Devices_Sentiment_Data_New["Product_Family"] == device]["Review"]
                a = device_name + "_Reviews.txt"
                data_1.to_csv(a, sep='\t')
                summary_1 = query_to_embedding_summarize_new("Give me the pros and cons of " + device_name, a)
    #             summary_1 = "Placeholder Summary"
                st.write(summary_1)
                save_history_devices_new(summary_1)

        if device_name:
            st.session_state.curr_response+=f"Device Name: {device_name}<br><br>"
            if summary_1:
                st.session_state.curr_response+=f"{summary_1}<br><br>"
    except:
        print(f"Error in device_details_new() for {device}")
        st.write(f"Unable to generate response for the input query. Please rephrase and try again, or contact the developer of the tool.")

def comparison_view_new(device1, device2):
    try:
        st.write(r"$\textsf{\Large Device Comparison}$")
        st.session_state.curr_response+=f"Device Comparison<br><br>"
        col1, col2 = st.columns(2)
        with col1:
            device_details_new(device1)
        with col2:
            device_details_new(device2)
    except:
        print(f"Error in comparison_view_new() for {device1} and {device2}")
        st.write(f"Unable to generate response for the input query. Please rephrase and try again, or contact the developer of the tool.")
        
        
def identify_devices_new(input_string):
    try:
        # First, check if any device in the Sales Data and Sentiment data is exactly in the input string
        input_string = input_string.upper()
        devices_list_sentiment = list(Devices_Sentiment_Data_New['Product_Family'].unique())
        for device in devices_list_sentiment:
            if device in input_string:
                return device

        # If no exact match is found, use fuzzy matching
        most_matching_device_sentiment = process.extractOne(input_string, devices_list_sentiment, scorer=fuzz.token_set_ratio)  

        # Check the matching score
        if most_matching_device_sentiment[1] >= 60:
            return most_matching_device_sentiment[0]

        devices_list_sales = list(dev_mapping['SalesDevice'].unique())
        for device in devices_list_sales:
            if device == "UNDEFINED":
                continue
            elif device in input_string:
                return device

        most_matching_device_sales = process.extractOne(input_string, devices_list_sales, scorer=fuzz.token_set_ratio)

        if most_matching_device_sales[1] >= 60:
            return most_matching_device_sales[0]
        else:
            return "Device not available"
    except:
        print(f"Error in identify_devices_new() for {input_string}")
        return "Device not available"
        
def device_summarization_new(user_input):
    try:
        if user_input == "Device not availabe":
            message = "I don't have sufficient data to provide a complete and accurate response at this time. Please provide more details or context."
            st.write(message)
            st.session_state.curr_response+=f"{message}<br>"
        else:
            inp = user_input
            new_inp_check = False
            if not hasattr(st.session_state, 'selected_devices'):
                st.session_state.selected_devices = [None,None]
            if not hasattr(st.session_state, 'past_inp2'):
                st.session_state.past_inp2 = None
            if not hasattr(st.session_state, 'past_inp_comp_dev2'):
                st.session_state.past_inp_comp_dev2 = []
            if not hasattr(st.session_state, 'display_history_devices'):
                st.session_state.display_history_devices = []
            if not hasattr(st.session_state, 'context_history_devices'):
                st.session_state.context_history_devices = []
            if not hasattr(st.session_state, 'curr_response'):
                st.session_state.curr_response = ""
            if (not st.session_state.past_inp2) or (st.session_state.past_inp2[0] != inp):
                new_inp_check = True
                st.session_state.past_inp_comp_dev2 = []
                device_name, img_link, net_Sentiment, aspect_sentiment, total_sales, asp, high_specs, sale, star_rating_html, comp_devices = generate_device_details_new(inp)
            else:
                new_inp_check = False
                old_inp, device_name, img_link, net_Sentiment, aspect_sentiment, total_sales, asp, high_specs, sale, star_rating_html, comp_devices, summ = st.session_state.past_inp2
            html_code = f"""
            <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); display: flex; align-items: center;">
                <div style="flex: 1; text-align: center;">
                    <img src="data:image/jpeg;base64,{base64.b64encode(open(img_link, "rb").read()).decode()}"  style="width: 150px; display: block; margin: 0 auto;">
                    <p style="color: black; font-size: 18px;">{device_name}</p>
                    <p>{star_rating_html}</p>
                </div>
                <div style="width: 2px; height: 150px; border-left: 2px dotted #ccc; margin: 0 20px;"></div>
                <div style="flex: 2; color: black; font-size: 18px;">
                    <p>Total Devices Sold: <strong>{total_sales}</strong></p>
                    <p>Average Selling Price: <strong>{asp}</strong></p>
                    <p>Highest Selling Specs: <strong>{high_specs}</strong> - <strong>{sale}</strong></p>
                </div>
            </div>
            """
            st.markdown(html_code, unsafe_allow_html=True)



            if new_inp_check:
                st.session_state.curr_response+=f"{html_code}<br>"
                summ = get_detailed_summary_new(inp)
                st.session_state.past_inp2 = (inp, device_name, img_link, net_Sentiment, aspect_sentiment, total_sales, asp, high_specs, sale, star_rating_html, comp_devices, summ)
                st.session_state.curr_response+=f"Detailed Summary"
                st.session_state.curr_response+=f"<br>{summ}<br>"

            st.write("")
            st.write(r"$\textsf{\Large Detailed Summary}$")
            st.write(summ)
            save_history_devices_new(summ)
            st.session_state.selected_devices[0] = device_name


            if len(comp_devices):
                st.write(r"$\textsf{\Large Compare with Similar Devices:}$")
                col_list = [None, None, None]
                checkbox_state = []
                comp_devices_list = comp_devices['SERIES'].tolist()
                for i in range(len(comp_devices_list)):
                    if i<3:
                        checkbox_state.append(False)
                col_list[0], col_list[1], col_list[2] = st.columns(3)
                com_sent_dev_list = [None,None,None]
                for i in range(len(comp_devices_list)):
                    if i < 3:
                        with col_list[i]:
                            if new_inp_check:
                                com_device_name, img_path, com_sales, ASP, net_sentiment,com_sent_dev_name = get_comp_device_details_new(comp_devices_list[i], comp_devices)
                                com_star_rating_html = get_star_rating_html_new(net_sentiment)
                                st.session_state.past_inp_comp_dev2.append((com_device_name, img_path, com_sales, ASP, net_sentiment,com_sent_dev_name,com_star_rating_html))
                            else:
                                com_device_name, img_path, com_sales, ASP, net_sentiment,com_sent_dev_name, com_star_rating_html = st.session_state.past_inp_comp_dev2[i]

                            com_sent_dev_list[i] = com_sent_dev_name
                            with st.container(border = True, height = 300):
                                with st.container(border = False, height = 220):
                                    html_content = f"""
                                    <div style="text-align: center; display: inline-block; ">
                                        <img src="data:image/jpeg;base64,{base64.b64encode(open(img_path, "rb").read()).decode()}" width = "80" style="margin-bottom: 10px;">
                                        <div style="font-size: 16px; color: #333;">{com_sent_dev_name}</div>
                                        <div style="font-size: 14px; color: #666;">Sales: {com_sales}</div>
                                        <div style="font-size: 14px; color: #666;">Average Selling Price: {ASP}</div>
                                        <p>{com_star_rating_html}</p>
                                    </div>
                                """
                                    st.markdown(html_content, unsafe_allow_html=True)

                                checkbox_state[i] = st.checkbox("Compare",key=f"comparison_checkbox_{i}")

            for i in range(len(checkbox_state)):
                if checkbox_state[i]:
                    st.session_state.selected_devices[1] = com_sent_dev_list[i]
                    break
                st.session_state.selected_devices[1] = None

            if st.session_state.selected_devices[1]:
                comparison_view_new(st.session_state.selected_devices[0],st.session_state.selected_devices[1])
                st.session_state.selected_devices = [None, None]
    except:
        print(f"Error in device_summarization_new() for {user_input}")
        st.write(f"Unable to generate response for the input query. Please rephrase and try again, or contact the developer of the tool.")

def extract_comparison_devices_new(user_question):
    try:
        # Define the prompt template
        prompt_template = """
        Given a user prompt to compare customer reviews for two products or devices, extract the two product names.
        
        For example: "How does the Surface Go 3 compare to the HP Zbook 9?"

        Extracted Products: Surface Go 3, HP Zbook 9
        
        Input: Compare [PRODUCT_A] to [PRODUCT_B].
        Output: PRODUCT_A, PRODUCT_B
        Context:
        {context}
        Question:
        {question}

        Answer:
        """


        # Initialize the model and prompt template
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(
            azure_deployment=azure_deployment_name,
            api_version='2023-12-01-preview',temperature = 0)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        # Get the response from the model
        response = chain({"input_documents": [], "question": user_question}, return_only_outputs=True)
        
        return response['output_text'].split(", ")
    
    except:
        print(f"Error in extract_comparison_devices_new() for {user_question}")
        return None

def identify_prompt_new(user_question):
    try:
        # Define the prompt template
        prompt_template = """
        You are an AI Chatbot assistant. Carefully understand the user's question and follow these instructions to categorize their query into one of 5 features.

Features:
    Summarization, Comparison, Quant, Sales, Other

Instructions:

The user query will be about laptops and devices or consumer reviews of laptops and devices. The user can ask query about any laptop or device or even multiple laptops or devices.


Summarization:
        -Choose this if the user is looking for a summary or analysis of the reviews for only one particular device.
        -IMPORTANT: Choose this if the user is looking for a summary of a particular device
        -Choose this if the user asks for a summary of a particular Product Family.
        -Do not choose this for general pros and cons or top verbatims.
        -IMPORTANT: If user mention 2 devices for summarization, Go with Comparision
        -IMPORTANT: If user mention 3 devices for summarization, Go with Others
        -if the user asks question to summarize reviews for a feature, Go with Others.

Comparison:
        -Choose this if user is looking for comparison of two different Product Families based on user reviews.
        -Choose this if user is looking for comparison of two different devices or laptops based on user reviews.
        -Choose this if user is looking for comparison of two different devices or laptops.
        -Choose this if exactly two Product Families or devices or laptops are mentioned.
        -IMPORTANT: Do not choose Comparison if more than two Product Families are mentioned.
        -IMPORTANT: Choose Others if user mentions more than 2 devices or laptops or Product Family.
        
        
Quant: 
        -Choose this if the prompt seeks a quantitative or numerical answer around net sentiment or review count for different product families or geographies
        Provides data retrieval and visualization for any Product Family around different features like aspects or different geographies
         IMPORTANT : If there are any type of questions involving any quantitative answer, you should select this category.
         
         IMPORTANT : If the user asks incomplete questions like aspect wise net sentiment of any laptop etc without explicitly mentioning "calculate or "give me" , then also you should select this category.
         
        (e.g., "Calculate the net sentiment for different product families",
                   "What is the net sentiment for different geographies of product_family "A"?
                   "What is the net sentiment across different aspects of device "X"? etc.)
                   
         IMPORTANT : Whenver user asks about any top or bottom aspects or other things like keywords choose this category
                   
        -Choose this category if the user asks for distinct aspect summaries across a particular aspect for a product_family
         Any type of quantitative or numerical questions involving summary or aspect descriptions or summaries across aspects or aspect wise summaries should fall in this category.  
        (e.g., "What are the different aspect wise summaries of Microsoft Surface Pro?",
        "Generate the aspect summaries for different aspects of Asus Vivobook",
        "Give me the summaries for different aspects of Lenovo Laptops", etc.)
        -Choose this if the user asks for best and worst feature of different aspects for a product_family
        (e.g., "What are the best and worst features of different aspects of Microsoft Surface Pro?",
        "Give me the most liked and most concerning prospect for different aspects of Lenovo Laptops", etc.)
        
Sales: Choose this if the prompt seeks a quantitative or numerical answer around net sales or net sales units or average sales price for different product families or geographies
       If user is asking any quantitative questions involving sales, you should select this category.
       
        (e.g., "Calculate the net sales for different product families",
                   "What is the net sales for different geographies of product_family "A"?
                   "What is the average sales price across different aspects of device "X"? etc.)
        

Other:

        -Choose this for general questions about any Product Family or Device or Laptop.
        - If the user asks about top or bottom aspects, don't select this category, select Quant category
        -Choose this for queries about pros and cons, common complaints, top features or verbatims.
        -Also, choose this if the question involves more than two Product Families or Laptops or Devices.
        -Choose this, if the question ask to summarize reviews for a particular feature and not a particular device or laptop or Product Family.
        

Important Notes:

    -Do not choose Comparison if more than two Product Families are mentioned.
    -Generic should be chosen for any query not specific to net sentiment, aspect sentiment, or comparing exactly two Product Families.

Your response should be one of the following:


"Summarization"
"Comparison"
"Quant"
"Sales"
"Other"

        Input: User prompt about customer reviews
        Output: Category (Summarization, Comparison, Quant, Sales or Other)
        Context:
        {context}
        Question:
        {question}

        Answer:
        """


        # Initialize the model and prompt template
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(
            azure_deployment=azure_deployment_name,
            api_version='2024-03-01-preview',temperature = 0)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        
        # Get the response from the model
        response = chain({"input_documents": [], "question": user_question}, return_only_outputs=True)

         # Determine the output category based on the response
        if "summarization" in response["output_text"].lower():
            return "summarization"
        elif "comparison" in response["output_text"].lower():
            return "comparison"
        elif "quant" in response["output_text"].lower():
            return "quant"
        elif "sales" in response["output_text"].lower():
            return "sales"
        else:
            return "other"
    except:
        print(f"Error in identify_prompt_new() for {user_question}")
        return None
        
def get_conversational_chain_devices_generic_new():
    try:
        prompt_template = """
        
            IMPORTANT: Use only the data provided to you and do not rely on pre-trained documents.

            INMPORTANT : Verbatims is nothing but Review. if user asks for top reviews. Give some important reviews user mentioned.
            
            Given a dataset with these columns: Review, Data_Source, Geography, Product_Family, Sentiment and Aspect (also called Features)
                      
                      Review: This column contains the opinions and experiences of users regarding different product families across geographies, providing insights into customer satisfaction or complaints and areas for improvement.
                      Data_Source: This column indicates the platform from which the user reviews were collected, such as Amazon, Flipkart, Bestbuy.
                      Geography: This column lists the countries of the users who provided the reviews, allowing for an analysis of regional preferences and perceptions of the products.
                      Product_Family: This column identifies the broader category of products to which the review pertains, enabling comparisons and trend analysis across different product families.
                      Sentiment: This column reflects the overall tone of the review, whether positive, negative, or neutral, and is crucial for gauging customer sentiment.
                      Aspect: This column highlights the particular features or attributes of the product that the review discusses, pinpointing areas of strength or concern.
                      
                      Perform the required task from the list below, as per user's query: 
                      1. Review Summarization - Summarize the reviews by filtering the relevant Aspect, Geography, Product_Family, Sentiment or Data_Source, only based on available reviews and their sentiments in the dataset.
                      2. Aspect Comparison - Provide a summarized comparison for each overlapping feature/aspect between the product families or geographies ,  only based on available user reviews and their sentiments in the dataset. Include pointers for each aspect highlighting the key differences between the product families or geographies, along with the positive and negative sentiments as per customer perception.
                      3. New Feature Suggestion/Recommendation - Generate feature suggestions or improvements or recommendations based on the frequency and sentiment of reviews and mentioned aspects and keywords. Show detailed responses to user queries by analyzing review sentiment, specific aspects, and keywords.
                      4. Hypothetical Reviews - Based on varying customer sentiments for the reviews in the existing dataset, generate hypothetical reviews for any existing feature updation or new feature addition in any device family across any geography, by simulating user reactions. Ensure to synthesize realistic reviews that capture all types of sentiments and opinions of users, by considering their hypothetical prior experience working with the new feature and generate output based on data present in dataset only. After these, provide solutions/remedies for negative hypothetical reviews. 
                      
                      IMPORTANT: Give as much as details as possible. Minimun number of Words should be 300 words atleat you can have more as well.
                      
                      Enhance the model’s comprehension to accurately interpret user queries by:
                      Recognizing abbreviations for country names (e.g., ‘DE’ for Germany, ‘USA’or 'usa' or 'US' for the United States of America) and expanding them to their full names for clarity.
                      Understanding product family names even when written in reverse order or missing connecting words such as HP Laptop 15, Lenovo Legion 5 15 etc
                      Utilizing context and available data columns to infer the correct meaning and respond appropriately to user queries involving variations in product family names or geographical references
                      Please provide a comprehensive Review summary, feature comparison, feature suggestions for specific product families and actionable insights that can help in product development and marketing strategies.
                      Generate acurate response only, do not provide extra information.
          \nFollowing is the previous conversation from User and Response, use it to get context only:""" + str(st.session_state.context_history_devices) + """\n
                Use the above conversation chain to gain context if the current prompt requires context from previous conversation.\n. When user asks uses references like "from previous response", ":from above response" or "from above", Please refer the previous conversation and respond accordingly.\n
            
            IMPORTANT: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.
            
            If the user question is not in the data provided. Just mention - "Sorry! I do not have sufficient reviews for mentioned product.". 
            But do not restrict yourself in responding to the user questions like 'hello', 'Hi' and basic chat question
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(
            azure_deployment=azure_deployment_name,
            api_version='2024-03-01-preview',temperature = 0.2)
        
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except:
        err = f"An error occurred while getting conversation chain for detailed review summarization."
        print(f"Error in get_conversational_chain_devices_generic_new()")
        return err
      
def query_devices_detailed_generic_new(user_question, vector_store_path="faiss_index_windows_116K_Summary_Final"):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment=azure_embedding_name)
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_devices_generic_new()
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except:
        err = f"An error occurred while getting LLM response for detailed review summarization."
        print(f"Error in query_devices_detailed_generic_new() for {user_question}")
        return err

def get_conversational_chain_quant_devices_new():
    try:
        prompt_template = """
        
        If an user is asking for Summarize reviews of any product. Note that user is not seeking for reviews, user is seeking for all the Quantitative things of the product(Net Sentiment & Review Count) and also (Aspect wise sentiment and Aspect wise review count)
        So choose to Provide Net Sentiment and Review Count and Aspect wise sentiment and their respective review count and Union them in single table
        
        Example : If the user Quesiton is "Summarize reviews of Microsoft Surface products"
        
        User seeks for net sentiment and aspect wise net sentiment of "Microsoft Surface" Product and their respective review count in a single table
        
        Your response should be : Overall Sentiment is nothing but the net sentiment and overall review count of the product
        
                        Aspect Aspect_SENTIMENT REVIEW_COUNT
                    0 TOTAL 40 15000.0
                    1 Performance 31.8 2302.0
                    2 Gaming 20.2 570.0
                    3 Display 58.9 397.0
                    4 Design -1.2 345.0
                    5 Touchpad 20.1 288.0
                    6 Storage/Memory -22.9 271.0
                    7 Audio-Microphone -43.7 247.0
                    8 Software -28.6 185.0
                    9 Hardware 52.9 170.0
                    10 Keyboard 19.1 157.0
                    11 Account -44.7 152.0
                    12 Price 29.5 95.0
                    13 Graphics 18.9 90.0 and so on
                    
                    The Query has to be like this 
                    
                SELECT 'TOTAL' AS Aspect, 
                ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                SUM(Review_Count) AS Review_Count
                FROM Devices_Sentiment_Data_New
                WHERE Product_Family LIKE '%Asus Rog Zephyrus%'

                UNION

                SELECT Aspect, 
                ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                SUM(Review_Count) AS Review_Count
                FROM Devices_Sentiment_Data_New
                WHERE Product_Family LIKE '%Asus Rog Zephyrus%'
                GROUP BY Aspect

                ORDER BY Review_Count DESC

                    
                    
                IMPORTANT : if any particular Aspect "Performance" in user prompt:
                    

                        SELECT 'TOTAL' AS Aspect, 
                        ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                        SUM(Review_Count) AS Review_Count
                        FROM Devices_Sentiment_Data_New
                        WHERE Product_Family LIKE '%Asus Rog Zephyrus%'

                        UNION

                        SELECT Aspect, 
                        ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                        SUM(Review_Count) AS Review_Count
                        FROM Devices_Sentiment_Data_New
                        WHERE Product_Family LIKE '%Asus Rog Zephyrus%'
                        GROUP BY Aspect
                        HAVING Aspect LIKE %'Performance'%

                        ORDER BY Review_Count DESC


        
        IMPORTANT : IT has to be Net sentiment and Aspect Sentiment. Create 2 SQL Query and UNION them
        
        1. Your Job is to convert the user question to SQL Query (Follow Microsoft SQL server SSMS syntax.). You have to give the query so that it can be used on Microsoft SQL server SSMS.You have to only return query as a result.
            2. There is only one table with table name Devices_Sentiment_Data_New where each row is a user review. The table has 13 columns, they are:
                Review: Review of the Windows Product
                Data_Source: From where is the review taken. It contains different retailers
                Geography: From which Country or Region the review was given. It contains different Grography.
                Title: What is the title of the review
                Review_Date: The date on which the review was posted
                Product: Corresponding product for the review. It contains following values: "Windows 11 (Preinstall)", "Windows 10"
                Product_Family: Which version or type of the corresponding Product was the review posted for. Different Device Names
                Sentiment: What is the sentiment of the review. It contains following values: 'Positive', 'Neutral', 'Negative'.
                Aspect: The review is talking about which aspect or feature of the product. It contains following values: "Audio-Microphone","Software","Performance","Storage/Memory","Keyboard","Browser","Connectivity","Hardware","Display","Graphics","Battery","Gaming","Design","Ports","Price","Camera","Customer-Service","Touchpad","Account","Generic"
                Keyword: What are the keywords mentioned in the product
                Review_Count - It will be 1 for each review or each row
                Sentiment_Score - It will be 1, 0 or -1 based on the Sentiment.
                Aspect_Description - two to four word summary of the review.
                
            3. Sentiment mark is calculated by sum of Sentiment_Score.
            4. Net sentiment is calculcated by sum of Sentiment_Score divided by sum of Review_Count. It should be in percentage. Example:
                    SELECT ((SUM(Sentiment_Score)*1.0)/(SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment 
                    FROM Devices_Sentiment_Data_New
                    ORDER BY Net_Sentiment DESC
            5. Net sentiment across country or across region is sentiment mark of a country divided by total reviews of that country. It should be in percentage.
                Example to calculate net sentiment across country:
                    SELECT Geography, ((SUM(Sentiment_Score)*1.0) / (SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment
                    FROM Devices_Sentiment_Data_New
                    GROUP BY Geography
                    ORDER BY Net_Sentiment DESC
            6. Net Sentiment across a column "X" is calculcated by Sentiment Mark for each "X" divided by Total Reviews for each "X".
                Example to calculate net sentiment across a column "X":
                    SELECT X, ((SUM(Sentiment_Score)*1.0) / (SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment
                    FROM Devices_Sentiment_Data_New
                    GROUP BY X
                    ORDER BY Net_Sentiment DESC
            7. Distribution of sentiment is calculated by sum of Review_Count for each Sentiment divided by overall sum of Review_Count
                Example: 
                    SELECT Sentiment, SUM(ReviewCount)*100/(SELECT SUM(Review_Count) AS Reviews FROM Devices_Sentiment_Data_New) AS Total_Reviews 
                    FROM Devices_Sentiment_Data_New
                    GROUP BY Sentiment
                    ORDER BY Total_Reviews DESC
            8. Convert numerical outputs to float upto 1 decimal point.
            9. Always include ORDER BY clause to sort the table based on the aggregate value calculated in the query.
            10. Top Country is based on Sentiment_Score i.e., the Country which have highest sum(Sentiment_Score)
            11. Always use 'LIKE' operator whenever they mention about any Country. Use 'LIMIT' operator instead of TOP operator.Do not use TOP OPERATOR. Follow syntax that can be used with pandasql.
            12. If you are using any field in the aggregate function in select statement, make sure you add them in GROUP BY Clause.
            13. Make sure to Give the result as the query so that it can be used on Microsoft SQL server SSMS.
            14. Important: Always show Net_Sentiment in Percentage upto 1 decimal point. Hence always make use of ROUND function while giving out Net Sentiment and Add % Symbol after it.
            15. Important: User can ask question about any categories including Aspects, Geography, Sentiment etc etc. Hence, include the in SQL Query if someone ask it.
            16. Important: Your Response should directly starts from SQL query nothing else.
            17. Important: Always use LIKE clause instead of = symbol while generating SQL query.
            18. Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.
            19. Sort all Quantifiable outcomes based on review count
          \nFollowing is the previous conversation from User and Response, use it to get context only:""" + str(st.session_state.context_history_devices) + """\n
                Use the above conversation chain to gain context if the current prompt requires context from previous conversation.\n. When user asks uses references like "from previous response", ":from above response" or "from above", Please refer the previous conversation and respond accordingly.\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(
            azure_deployment=azure_deployment_name,
            api_version='2023-12-01-preview',
            temperature = 0)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except:
        err = f"An error occurred while getting conversation chain for quantifiable review summarization."
        print(f"Error in get_conversational_chain_quant_devices_new().")
        return err

#Function to convert user prompt to quantitative outputs for Copilot Review Summarization
def query_quant_devices_new(user_question, vector_store_path="faiss_index_windows_116K_Summary_Final"):
    try:
        # Initialize the embeddings model
        embeddings = AzureOpenAIEmbeddings(azure_deployment=azure_embedding_name)
        # Load the vector store with the embeddings model
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        # Rest of the function remains unchanged
        chain = get_conversational_chain_quant_devices_new()
        docs = []
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        SQL_Query = response["output_text"]
        SQL_Query = convert_top_to_limit_new(SQL_Query)
        SQL_Query = process_tablename_new(SQL_Query,"Devices_Sentiment_Data_New")
        data = ps.sqldf(SQL_Query, globals())
        data_1 = data
        return data_1
    except:
        err = f"An error occurred while generating response for quantitative review summarization."
        print(f"Error in query_quant_devices_new() for {user_question}")
        return err
    
def get_conversational_chain_quant_classify2_compare_devices_new():
    global model
    try:
        prompt_template = """

                You are an AI Chatbot assistant. Understand the user question carefully and follow all the instructions mentioned below.
                1. Your job is to convert the user question to an SQL query (Follow Microsoft SQL Server SSMS syntax). You have to give the query so that it can be used on Microsoft SQL Server SSMS. You have to only return the query as a result.
                2. There is only one table with the table name `Devices_Sentiment_Data_New` where each row is a user review. The table has 10 columns:
                    - There is only one table with table name Devices_Sentiment_Data_New where each row has. The table has 10 columns, they are:
                    Review: This column contains the opinions and experiences of users regarding different product families across geographies, providing insights into customer satisfaction or complaints and areas for improvement.
                    Data_Source: This column indicates the platform from which the user reviews were collected, such as Amazon, Flipkart, Bestbuy.
                    Geography: This column lists the countries of the users who provided the reviews, allowing for an analysis of regional preferences and perceptions of the products.
                    Product_Family: This column identifies the broader category of products to which the review pertains, enabling comparisons and trend analysis across different product families.
                    Sentiment: This column reflects the overall tone of the review, whether positive, negative, or neutral, and is crucial for gauging customer sentiment.
                    Aspect: This column highlights the particular features or attributes of the product that the review discusses, pinpointing areas of strength or concern.These are the aspects for present in the table. Generic, Design, Software, Performance, Price, Hardware, Display, Hardware, Display, Battery, Keyboard, Storage/Memory, Connectivity, Gaming, Customer-Service, Browser, Audio-Microphone, Ports, Camera, Account, Graphics, Touchpad.
                    Aspect_Description: 2 to 3 word summary for the review
                
        IMPORTANT : You will get the user prompt and also you will get the exact device names also so rephrase that by yourself with the proper product family names.
                    Example : You will get like -Compare Battery aspect sentiment of Microsoft Surface pro, Microsoft Surface go and hp Pavillon. MICROSOFT SURFACE LAPTOP GO 3 12, MICROSOFT SURFACE PRO 9 13, HP PAVILON 15. 
                              you need to rephrase like - Comapre Battery aspect sentiment of MICROSOFT SURFACE LAPTOP GO 3 12, MICROSOFT SURFACE PRO 9 13 and HP PAVILON 15.
                              for the above user question you need to generate the sql Query.
                        
                    IMPORTANT : REMEMBER THAT ALL THE NAMES IN PRODUCT_FAMILIES HAS WINDOWS OS ONLY.                    
                    IMPORTANT : User can confuse a lot, but understand the question carefully and respond:
                    1. If the user asks for Net Sentiment for column "X", the query should be exactly like this: 

                            SELECT X, ((SUM(Sentiment_Score)) / (SUM(Review_Count))) * 100 AS Net_Sentiment, SUM(Review_Count) AS Review_Count
                            FROM Devices_Sentiment_Data_New
                            GROUP BY X
                            ORDER BY Review_Count DESC

                    2. If the user asks for overall review count, the query should be like this:
                            SELECT SUM(Review_Count) 
                            FROM Devices_Sentiment_Data_New
                    3. If the user asks for review distribution across column 'X', the query should be like this:
                            SELECT 'X', SUM(Review_Count) * 100 / (SELECT SUM(Review_Count) FROM Devices_Sentiment_Data_New) AS Review_Distribution
                            FROM Devices_Sentiment_Data_New 
                            GROUP BY 'X'
                            ORDER BY Review_Distribution DESC
                    4. If the user asks for column 'X' Distribution across column 'Y', the query should be like this: 
                            SELECT 'Y', SUM('X') * 100 / (SELECT SUM('X') AS Reviews FROM Devices_Sentiment_Data_New) AS Distribution_PCT
                            FROM Devices_Sentiment_Data_New 
                            GROUP BY 'Y'
                            ORDER BY Distribution_PCT DESC
                    5. If the user asks for net sentiment across any country: example : Net sentiment of Windows Copilot in US geography
                               SELECT ((SUM(Sentiment_Score)) / (SUM(Review_Count))) * 100 AS Net_Sentiment
                               FROM Devices_Sentiment_Data_New
                               WHERE Geography LIKE "%US%"
                               
                    6. IMPORTANT NOTE :
                   
                        THIS IS THE ONLY WAY TO CALCULATE NET SENTIMENT : ((SUM(Sentiment_Score)*1.0)/(SUM(Review_Count)*1.0)) * 100
                        
                    7. Review count mix/ Review count Percentage of a product by aspect:
                    
                    
                        SELECT ASPECT, (SUM(REVIEW_COUNT)*100/(SELECT SUM(REVIEW_COUNT) FROM Devices_Sentiment_Data_New WHERE PRODUCT_FAMILY LIKE '%GITHUB COPILOT%')) AS REVIEW_COUNT_PERCENTAGE
                        FROM Devices_Sentiment_Data_New
                        WHERE PRODUCT_FAMILY LIKE '%HP PAVILON 15%'
                        GROUP BY ASPECT
                        ORDER BY REVIEW_COUNT_PERCENTAGE DESC
                        
                    8. if user asks for Compare the battery aspect for these laptops microsoft surface pro, Microsoft Surface go anf acer nitro:
                        
                        SELECT PRODUCT_FAMILY, 'BATTERY' AS ASPECT, ((SUM(SENTIMENT_SCORE)) / (SUM(REVIEW_COUNT))) * 100 AS NET_SENTIMENT
                        FROM Devices_Sentiment_Data_New
                        WHERE PRODUCT_FAMILY LIKE '%MICROSOFT SURFACE PRO 9 13%' 
                           OR PRODUCT_FAMILY LIKE '%MICROSOFT SURFACE LAPTOP GO 3 12%' 
                           OR PRODUCT_FAMILY LIKE '%ACER NITRO 5 15%'
                        GROUP BY PRODUCT_FAMILY
                        ORDER BY NET_SENTIMENT DESC;
                        
                    9. If user asks for more than two or more aspects use the below format
                        SELECT PRODUCT_FAMILY, 'BATTERY' AS ASPECT, 
                               ((SUM(CASE WHEN ASPECT = 'BATTERY' THEN SENTIMENT_SCORE ELSE 0 END)) / 
                                (SUM(CASE WHEN ASPECT = 'BATTERY' THEN REVIEW_COUNT ELSE 0 END))) * 100 AS NET_SENTIMENT
                        FROM Devices_Sentiment_Data_New
                        WHERE PRODUCT_FAMILY LIKE '%MICROSOFT SURFACE PRO 9 13%' 
                           OR PRODUCT_FAMILY LIKE '%MICROSOFT SURFACE LAPTOP GO 3 12%' 
                           OR PRODUCT_FAMILY LIKE '%ACER NITRO 5 15%'
                        GROUP BY PRODUCT_FAMILY

                        UNION ALL

                        SELECT PRODUCT_FAMILY, 'SOFTWARE' AS ASPECT, 
                               ((SUM(CASE WHEN ASPECT = 'SOFTWARE' THEN SENTIMENT_SCORE ELSE 0 END)) / 
                                (SUM(CASE WHEN ASPECT = 'SOFTWARE' THEN REVIEW_COUNT ELSE 0 END))) * 100 AS NET_SENTIMENT
                        FROM Devices_Sentiment_Data_New
                        WHERE PRODUCT_FAMILY LIKE '%MICROSOFT SURFACE PRO 9 13%' 
                           OR PRODUCT_FAMILY LIKE '%MICROSOFT SURFACE LAPTOP GO 3 12%' 
                           OR PRODUCT_FAMILY LIKE '%ACER NITRO 5 15%'
                        GROUP BY PRODUCT_FAMILY

                        ORDER BY PRODUCT_FAMILY, ASPECT;
                    
                    For all the comparison related user query, the format should remain the same. i.e., the column names that we are giving as alias should remain the same. Do not change the schema. 

                    IMPORTANT -> Comparison SQL Queries
                    
                    IMPORTANT : User can confuse a lot, but understand the question carefully and respond:                            
                                Follow this same template whenever user asks for compare different  product families across Geography, make sure to change all the aspects to Geography
                                  WITH NetSentiment AS (
                                                           SELECT 
                                                                PRODUCT_FAMILY, 
                                                                ASPECT, 
                                                                ((SUM(SENTIMENT_SCORE) * 1.0) / (SUM(REVIEW_COUNT) * 1.0)) * 100 AS NET_SENTIMENT, 
                                                                SUM(REVIEW_COUNT) AS Review_Count
                                                            FROM 
                                                                Devices_Sentiment_Data_New
                                                            GROUP BY 
                                                                PRODUCT_FAMILY, ASPECT 
                                                        ),
                                                        OrderedNetSentiment AS (
                                                            SELECT 
                                                                PRODUCT_FAMILY, 
                                                                ((SUM(SENTIMENT_SCORE) * 1.0) / (SUM(REVIEW_COUNT) * 1.0)) * 100 AS NET_SENTIMENT
                                                            FROM 
                                                                Devices_Sentiment_Data_New
                                                            GROUP BY 
                                                                PRODUCT_FAMILY
                                                            ORDER BY 
                                                                NET_SENTIMENT DESC
                                                        )

                                                        -- Combine the aspect net sentiments with the overall net sentiment
                                                        SELECT 
                                                            ns.PRODUCT_FAMILY, 
                                                            ns.ASPECT,
                                                            ns.NET_SENTIMENT AS NET_SENTIMENT_ASPECT, comment : (for geography name this as NET_SENTIMENT_GEOGRAPHY)
                                                            ons.NET_SENTIMENT AS NET_SENTIMENT_OVERALL,
                                                            ns.Review_Count
                                                        FROM 
                                                            NetSentiment ns
                                                        JOIN 
                                                            OrderedNetSentiment ons 
                                                        ON 
                                                            ns.PRODUCT_FAMILY = ons.PRODUCT_FAMILY

                                                        UNION ALL

                                                        -- Add the overall net sentiment as a row with 'Overall' as the ASPECT
                                                        SELECT 
                                                            ons.PRODUCT_FAMILY,
                                                            'Overall' AS ASPECT,
                                                            ons.NET_SENTIMENT AS NET_SENTIMENT_ASPECT, comment : (for geography name this as NET_SENTIMENT_GEOGRAPHY)
                                                            ons.NET_SENTIMENT AS NET_SENTIMENT_OVERALL,
                                                            SUM(ns.Review_Count) AS Review_Count
                                                        FROM 
                                                            OrderedNetSentiment ons
                                                        JOIN 
                                                            NetSentiment ns
                                                        ON 
                                                            ons.PRODUCT_FAMILY = ns.PRODUCT_FAMILY
                                                        GROUP BY 
                                                            ons.PRODUCT_FAMILY, ons.NET_SENTIMENT

                                                        ORDER BY 
                                                            Review_Count DESC;

                        VERY IMPORTANT : For Comparision, Always give the Product_Family/Product column along with aspect or geography in the SQL Query                   
                        
                    Same goes for all comparisions.

                    Important: While generating SQL query to calculate net_sentiment across column 'X' and 'Y', if 'Y' has less distinct values, keep your response like this - SELECT 'Y','X', ((SUM(Sentiment_Score)) / (SUM(Review_Count))) * 100 AS Net_Sentiment, SUM(Review_Count) AS Review_Count FROM Devices_Sentiment_Data_New GROUP BY 'Y','X'
                    
                    Important: Always replace '=' operator with LIKE keyword and add '%' before and after filter value for single or multiple WHERE conditions in the generated SQL query . For example, if the query is like - 'SELCT * FROM Devices_Sentiment_Data_New WHERE PRODUCT='ABC' AND GEOGRAPHY='US' ORDER BY Review_Count' , you should modify the query and share the output like this - 'SELCT * FROM Devices_Sentiment_Data_New WHERE PRODUCT LIKE '%ABC%' AND GEOGRAPHY LIKE '%US%' ORDER BY Review_Count'

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
        model = AzureChatOpenAI(
            azure_deployment=azure_deployment_name,
            api_version='2023-12-01-preview',
            temperature = 0.0)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except:
        err = f"An error occurred while getting conversation chain for quantifiable review summarization."
        print(f"Error in get_conversational_chain_quant_classify2_compare_devices_new().")
        return err

def query_quant_classify2_compare_devices_new(user_question, vector_store_path="faiss_index_windows_116K_Summary_Final"):
    global history
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="MV_Agusta")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_quant_classify2_compare_devices_new()
        docs = []
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        SQL_Query = response["output_text"]
        # st.write(SQL_Query)
        SQL_Query = convert_top_to_limit_new(SQL_Query)
        SQL_Query = process_tablename_new(SQL_Query,"Devices_Sentiment_Data_New")
        # st.write(SQL_Query)
        data = ps.sqldf(SQL_Query, globals())
        data_1 = data
        html_table = data.to_html(index=False)
    #     return html_table
        return data_1
    except:
        err = f"An error occurred while generating response for quantitative review summarization."
        print(f"Error in query_quant_classify2_compare_devices_new() for {user_question}.")
        return err
    
def custom_color_gradient_compare_devices(val, vmin=-100, vmax=100):
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

    
    
    
def devices_quant_approach2(user_question_final):
    device = identify_devices_new(user_question_final)
                                
#################################################################################################################################
    if device == "Device not available":
        try:
            data= quantifiable_data_devices_new(user_question_final)
            if 'NET_SENTIMENT' in data.columns:
                    overall_net_sentiment=data['NET_SENTIMENT'][0]
                    overall_net_sentiment = round(overall_net_sentiment, 1)
                    overall_review_count=data['REVIEW_COUNT'][0]
                    overall_review_count = round(overall_review_count)
        except:
            print(f"Cannot generate quantitative output for {user_question_final}")
            
        words = user_question_final.lower().split()
        target_words = ['visual', 'visualize', 'graph', 'chart', 'visualization']
        
        try:
            if any(word in words for word in target_words):
                #st.write("for visual")
                visual_data=data.copy()
                numerical_cols = visual_data.select_dtypes(include='number').columns
                visual_data[numerical_cols] = visual_data[numerical_cols].apply(lambda x: x.round(1) if x.dtype == 'float' else x)
                generate_chart(visual_data)

                visual_data = visual_data[~visual_data.applymap(lambda x: x == 'TOTAL').any(axis=1)]
                generate_chart_insight_llm_devices_new(str(visual_data))


            elif len(data)>0:

                show_output=data.copy()
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
    #                             st.write("after impact calculation")    
    #                             st.write(show_output)



                #st.dataframe(data2)
                dataframe_as_dict = show_output.to_dict(orient='records')

                try:
                    user_question_final = user_question_final.replace("What is the", "Summarize reviews of")
                except:
                    pass
                if 'NET_SENTIMENT' in show_output.columns:
                    show_output = show_output.drop(index=0)
                    show_output2=show_output.copy()
                    show_output2.drop('Impact', axis=1, inplace=True)
                    show_output2 = show_output2.style.applymap(custom_color_gradient_compare_devices,subset=['NET_SENTIMENT'])
                    show_output2 = show_output2.set_properties(**{'text-align': 'center'})
                  
                    st.dataframe(show_output2)
                    #st.write("final dataframe")
#                     st.write(show_output2)
                    show_output2_html = show_output2.to_html(index=False)
                    st.session_state.display_history_devices.append({"role": "assistant", "content": show_output2_html, "is_html": True})
                    #st.write(f" Overall Net Sentiment is {overall_net_sentiment} for {overall_review_count} reviews.")
                    #st.session_state.display_history_devices.append({"role": "assistant", "content": f" Overall Net Sentiment is {overall_net_sentiment} for {overall_review_count} reviews.", "is_html": False})
                    qunat_summary = query_detailed_summary2_devices_new(str(show_output),user_question_final + "Which have the following sentiment data : " + str(show_output),[])
                    st.write(qunat_summary)
                    save_history_devices_new(qunat_summary)
                    st.session_state.display_history_devices.append({"role": "assistant", "content": qunat_summary, "is_html": False})
                else:
                    show_output2=show_output.copy()
                    #show_output2.drop('Impact', axis=1, inplace=True)
                    #st.write("final dataframe")
                    st.write(show_output2)
                    show_output2_html = show_output2.to_html(index=False)
                    st.session_state.display_history_devices.append({"role": "assistant", "content": show_output2_html, "is_html": True})
                    qunat_summary = query_detailed_summary2_devices_new(str(show_output),user_question_final + "Which have the following sentiment data : " + str(show_output),[])
                    st.write(qunat_summary)
                    save_history_devices_new(qunat_summary)
                    st.session_state.display_history_devices.append({"role": "assistant", "content": qunat_summary, "is_html": False})


                if(len(data))>1:
                    generate_chart(data)
            else:
                st.write(data)
        except:
            st.write(f"Cannot generate quantitative output for the given prompt. Please rephrase and try again!")

##################################################################################################################################
    else:
        try:
            device1 = get_sentiment_device_name_new(device)
            if device1:
                user_question_final=user_question_final.replace(device,device1)
                data= quantifiable_data_devices_new(user_question_final)

                if len(data)==0:
                    Gen_Ans = query_devices_detailed_generic_new(st.session_state.user_question_final)
                    st.write(Gen_Ans)
                    save_history_devices_new(Gen_Ans)
                    st.session_state.display_history_devices.append({"role": "assistant", "content": Gen_Ans, "is_html": False})

                else:    
                    if 'NET_SENTIMENT' in data.columns:
                        overall_net_sentiment=data['NET_SENTIMENT'][0]
                        overall_net_sentiment = round(overall_net_sentiment, 1)
                        overall_review_count=data['REVIEW_COUNT'][0]
                        overall_review_count = round(overall_review_count)


                    words = user_question_final.lower().split()
                    target_words = ['visual', 'visualize', 'graph', 'chart', 'visualization']
                    if any(word in words for word in target_words):
                        #st.write("for visual")
                        visual_data=data.copy()
                        numerical_cols = visual_data.select_dtypes(include='number').columns
                        visual_data[numerical_cols] = visual_data[numerical_cols].apply(lambda x: x.round(1) if x.dtype == 'float' else x)
                        generate_chart(visual_data)

                        visual_data = visual_data[~visual_data.applymap(lambda x: x == 'TOTAL').any(axis=1)]
                        generate_chart_insight_llm_devices_new(str(visual_data))

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
    #                                     st.write("after impact calculation")    
    #                                     st.write(show_output)



                        #st.dataframe(data2)
                        dataframe_as_dict = show_output.to_dict(orient='records')

                        try:
                            user_question_final = user_question_final.replace("What is the", "Summarize reviews of")
                        except:
                            pass
                        if 'NET_SENTIMENT' in show_output.columns:
                            show_output = show_output.drop(index=0)
                            show_output2=show_output.copy()
                            show_output2.drop('Impact', axis=1, inplace=True)
                            show_output2 = show_output2.style.applymap(custom_color_gradient_compare_devices,subset=['NET_SENTIMENT'])
                            show_output2 = show_output2.set_properties(**{'text-align': 'center'})
                            
                            st.dataframe(show_output2)
                            #st.write("final dataframe")
#                             st.write(show_output2)
                            show_output2_html = show_output2.to_html(index=False)
                            st.session_state.display_history_devices.append({"role": "assistant", "content": show_output2_html, "is_html": True})
                            #st.write(f" Overall Net Sentiment is {overall_net_sentiment} for {overall_review_count} reviews.")
                            #st.session_state.display_history_devices.append({"role": "assistant", "content": f" Overall Net Sentiment is {overall_net_sentiment} for {overall_review_count} reviews.", "is_html": False})
                            qunat_summary = query_detailed_summary2_devices_new(str(show_output),user_question_final + "Which have the following sentiment data : " + str(show_output),[])
                            st.write(qunat_summary)
                            save_history_devices_new(qunat_summary)
                            st.session_state.display_history_devices.append({"role": "assistant", "content": qunat_summary, "is_html": False})
                        else:
                            show_output2=show_output.copy()
                            #show_output2.drop('Impact', axis=1, inplace=True)
                            #st.write("final dataframe")
                            st.write(show_output2)
                            show_output2_html = show_output2.to_html(index=False)
                            st.session_state.display_history_devices.append({"role": "assistant", "content": show_output2_html, "is_html": True})
                            qunat_summary = query_detailed_summary2_devices_new(str(show_output),user_question_final + "Which have the following sentiment data : " + str(show_output),[])
                            st.write(qunat_summary)
                            save_history_devices_new(qunat_summary)
                            st.session_state.display_history_devices.append({"role": "assistant", "content": qunat_summary, "is_html": False})


                        if(len(data))>1:
                            generate_chart(data)
                    else:
                        st.write(data)      

            else:                                
                Gen_Ans = query_devices_detailed_generic_new(st.session_state.user_question)
                st.write(Gen_Ans)
                save_history_devices_new(Gen_Ans)
                st.session_state.display_history_devices.append({"role": "assistant", "content": Gen_Ans, "is_html": False})  
        except:
            st.write(f"Cannot generate quantitative output for the given prompt. Please rephrase and try again!") 
            
            
            
            
def sales_quant_approach2(user_question_final):
    try:
        
    
        device = identify_devices_new(user_question_final)
                                    #st.write(device)
        if device == "Device not available":

    ##################################################################################################################################
            #brand_list=list(RCR_Sales_Data['OEMGROUP'].unique())
            #RCR_Sales_Data['Series'] = RCR_Sales_Data['Series']+RCR_Sales_Data['OEMGROUP']
            response=query_quant_classify2_sales_new(user_question_final)
            if isinstance(response, pd.DataFrame):
                if 'Month' in response.columns:
                    response['Month']=pd.to_datetime(response['Month'])
                    response=response.sort_values('Month')
                st.dataframe(response)
                show_output2_html = response.to_html(index=False)
                st.session_state.display_history_devices.append({"role": "assistant", "content": show_output2_html, "is_html": True})
                insight_sales=generate_chart_insight_llm_devices_new(response)
                st.write(insight_sales)
                generate_chart(response)
            else:
                st.write(response)
                st.session_state.display_history_devices.append({"role": "assistant", "content": response, "is_html": False})

    ###################################################################################################################################

        else:

            device1 = get_sales_device_name_new(device)
            #st.write(device1)
            if device1:
                user_question_final=user_question_final.replace(device,device1)

                #brand_list=list(RCR_Sales_Data['OEMGROUP'].unique())
    #                                 for i in range(len(RCR_Sales_Data)):
    #                                     RCR_Sales_Data['Series'][i]=RCR_Sales_Data['OEMGROUP'][i]+" "+RCR_Sales_Data['Series'][i]

                #RCR_Sales_Data['Series'] = RCR_Sales_Data['Series']+RCR_Sales_Data['OEMGROUP']
                #st.write(RCR_Sales_Data['Series'].unique)
    #                                 for i in brand_list:
    #                                     if i.upper() in user_question_final:
    #                                         user_question_final=user_question_final.replace(i.upper(),'')

                #st.write(user_question_final)
                response=query_quant_classify2_sales_new(user_question_final)
                if isinstance(response, pd.DataFrame):
                    if 'Month' in response.columns:
                        response['Month']=pd.to_datetime(response['Month'])
                        response=response.sort_values('Month')
                    st.dataframe(response)
                    show_output2_html = response.to_html(index=False)
                    st.session_state.display_history_devices.append({"role": "assistant", "content": show_output2_html, "is_html": True})
                    insight_sales=generate_chart_insight_llm_devices_new(response)
                    st.write(insight_sales)
                    generate_chart(response)
                else:
                    st.write(response)
                    st.session_state.display_history_devices.append({"role": "assistant", "content": response, "is_html": False})
    except:
        st.write(f"Cannot generate quantitative output for the given prompt. Please rephrase and try again!")
                
                
                
def generate_chart(df):
    try:
        
        global full_response
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
    except:
        pass
    
    



##################### New Functions for multiple devices ######################################  
    
def extract_devices_new(user_question):
    try:
        # Define the prompt template
        prompt_template = """
        -You are an AI tool to help identify Devices and Laptops mentioned in the user query.
        -You will be given a user input and your job is to identify all the devices and laptops mentioned in it.
        -Here devices means laptops from various brands.
        -Your response should be the list of all the devices or laptops mentioned in the user input, separated by a comma.
        -Make sure to identify all possible devices mentioned in the user prompt.
        -The user input can have the name of any laptop, be it Microsoft laptops, Google Laptops, HP Laptops, Acer Laptops, Apple Laptops, Dell Laptops, Asus Laptops, or any other laptops.
        -The user input can have mentions of multiple devices. Make sure you identify all of them and include then in your response.
        -The user will either mention a very specific laptop like "Asus ROG Zephyrus G16 16", or a series of laptop like "Asus ROG" laptops. You should be able to identify the device in either case. 
            *Just for example*: if "Asus ROG Zephyrus G16 16" is mentioned then you should reply with "Asus ROG Zephyrus G16 16". If "Asus ROG" laptops are mentioned then you should reply with "Asus ROG".
        -Only focus on the device names or laptop names mentioned in the user input. Ignore everything else.
        -If there are no devices mentioned in the user input, the output should be "NO DEVICES FOUND".
        -The user input can mention laptops or devices after mentioning a laptop name, make sure you identify the laptop name only.
        -The laptop name can also sometimes contain "Laptop" word in it, like Surface Laptop. In this case you should keep Surface Laptop as the laptop name.

        **Note: Chromebook is a type of laptop.**

        **IMPORTANT**: YOU ARE STRICTLY PROHIBITED FROM ANSWERING THE QUERY ABOUT WHAT USER IS ASKING. DO NOT GIVE ANY REPLY OTHER THAN LIST OF DEVICES MENTIONED IN THE USER PROMPT.

        Examples:

        Example Input: "What are users talking about Inspiron laptops?"
        Example Output: "Inspiron"

        Example Input: "Summarize consumer reviews of Surface Pro laptops"
        Example Output: "Surface Pro"

        Example Input: "How are the reviews of Acer Aspire 13 devices?"
        Example Output: "Acer Aspire 13"

        Example Input: "Give a summary of Apple Macbook Pro 15."
        Example Output: "Apple Macbook Pro 15"

        Example Input: "Compare Microsoft Surface Pro with Surface Laptop Go and Vivobook laptops."
        Example Output: "Microsoft Surface Pro, Surface Laptop Go, Vivobook"

        Example Input: "Do you think the HP SPECTRE X360 14 is a good choice for students?"
        Example Output: "HP SPECTRE X360 14"

        Example Input: "Do you think Chromebook is good for playing games?"
        Example Output: "Chromebook"

        **IMPORTANT**: DO NOT ANSWER THE QUESTION USER IS ASKING IN THE INPUT. REPLY WITH THE DEVICE NAMES ONLY. IF THERE IS NO DEVICE IN THE INPUT, THEN REPLY WITH "NO DEVICES FOUND". DO NOT GENERATE ANY ANSWER TO THE QUESTION USER IS ASKING FOR. ONLY REPLY WITH THE DEVICE NAME. YOU ARE STRICTLY PROHIBITED FROM ANSWERING ANY QUESTION THAT USER IS ASKING. YOU ARE ONLY ALLOWED TO REPLY WITH THE DEVICE NAMES OR "NO DEVICES FOUND".

        Context:
        {context}
        Question:
        {question}

        Answer:
        """


        # Initialize the model and prompt template
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(
            azure_deployment=azure_deployment_name,
            api_version='2023-12-01-preview',temperature = 0.2)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        # Get the response from the model
        response = chain({"input_documents": [], "question": user_question}, return_only_outputs=True)
        output = response['output_text']
        if "NO DEVICES FOUND" in output:
            return None
        else:
            return output.split(", ")
    except:
        print(f"Error occured in extract_devices_new() for User Query: {user_question}")
        return None
    
def match_device_new(input_str):
    try:
        input_str = input_str.upper()
        devices_list = list(Devices_Sentiment_Data_New['Product_Family'].unique())
        device_matching = []
        if devices_list:
            for i in devices_list:
                score = fuzz.token_set_ratio(input_str, i)
                device_matching.append((i,score))
            sorted_list = sorted(device_matching, key=lambda x: x[1],reverse = True)
    #         sorted_list = [(x,y) for x,y in sorted_list if y>90]
        else:
            sorted_list = None
        return sorted_list
    except:
        print(f"Error occured in match_device_new() for {input_str}")
        return []


def identify_all_devices_new(user_input):
    try:
        dev_ext = extract_devices_new(user_input)
        print(f"\n\nDevice Extracted from Input: {dev_ext}")
        identified_devices = []
        if dev_ext:
            for i in dev_ext:
                out = match_device_new(i)
                out = [x for x,y in out if y>95]
                if out:
                    identified_devices.append((user_input,i,out))
        else:
            out = match_device_new(user_input)
            out = [x for x,y in out if y>50]
            if out:
                identified_devices.append((user_input,None,out))
        return identified_devices
    except:
        print(f"Error occured in identify_all_devices_new() for {user_input}")
        return []

def get_net_sent_new(device_names):
    try:
        data = Devices_Sentiment_Data_New.loc[Devices_Sentiment_Data_New["Product_Family"].isin(device_names)]
        net_sentiment = (data["Sentiment_Score"].sum())/(data["Review_Count"].sum())
    except:
        net_sentiment = None
    try:
        aspect_sentiment = data.groupby(["Aspect"])["Sentiment_Score"].sum()/data.groupby(["Aspect"])["Review_Count"].sum()
    except:
        aspect_sentiment = None
    return net_sentiment, aspect_sentiment

def get_sales_info_new(sales_device_name):
    try:
        sales_data = RCR_Sales_Data.loc[RCR_Sales_Data["Series"].isin(sales_device_name)]
        asp = sales_data["Sales_Value"].sum()/sales_data["Sales_Units"].sum()
        asp_str = "$"+str(round(asp))
        total_sales = sales_data["Sales_Units"].sum()/1000
        total_sales_str = str(round(total_sales))+"K"
        hs = sales_data.groupby(["Specs_Combination"])["Sales_Units"].sum().reset_index().sort_values(by="Sales_Units",ascending = False).head(1)
        high_specs = hs.values[0][0]
        high_specs_sales = hs.values[0][1]/1000
        high_specs_sales_str = str(round(high_specs_sales))+"K"
        chassis_segment = list(sales_data["Chassis_Segment"].unique())
        comp_data = RCR_Sales_Data.loc[(~RCR_Sales_Data["Series"].isin(sales_device_name)) & (RCR_Sales_Data["Chassis_Segment"].isin(chassis_segment))]
        comp_data = comp_data.groupby(["Series"])["Sales_Value","Sales_Units"].sum().reset_index()
        comp_data["Asp"] = comp_data["Sales_Value"]/comp_data["Sales_Units"]
        comp_data = comp_data.sort_values(by = "Asp",key = lambda col: abs(col-asp)).head(3)
        comp_data = comp_data[["Series","Asp","Sales_Units"]]
        comp_data.columns = ["Series","CompetitorASP","Sales_Units"]
        comp_data["Sales_Units"] = comp_data["Sales_Units"]/1000
        
    except:
        print(f"Error occured in get_sales_info_new() for {sales_device_name}")
        total_sales_str = ""
        asp_str = ""
        high_specs = ""
        high_specs_sales_str = ""
        comp_data = pd.DataFrame()
    return total_sales_str,asp_str,high_specs,high_specs_sales_str,comp_data
        

def get_date_range_all_new(sales_device_names_list):
    try:
        device_sales_data = RCR_Sales_Data.loc[RCR_Sales_Data["Series"].isin(sales_device_names_list)]
        device_sales_data['Month'] = pd.to_datetime(device_sales_data['Month'], format='%m/%d/%Y')
        min_date = device_sales_data["Month"].min().strftime('%Y-%m-%d')
        max_date = device_sales_data["Month"].max().strftime('%Y-%m-%d')
    except:
        min_date = "NA"
        max_date = "NA"
        print(f"Error occured in getting date range for sales data of {sales_device_names_list}")
    return min_date, max_date

def get_all_sent_dev_name_new(device_name):
    try:
        sentiment_device_name = dev_mapping[dev_mapping['SentimentDevice']==device_name]['SentimentDevice']
        if len(sentiment_device_name) == 0:
            sentiment_device_name = dev_mapping[dev_mapping['SalesDevice']==device_name]['SentimentDevice']
        if len(sentiment_device_name) == 0:
            sentiment_device_name = None
        sentiment_device_name = sentiment_device_name.to_list()
    except:
        print(f"Error occured in get_all_sent_dev_name_new() for {device_name}")
        sentiment_device_name = []
    return sentiment_device_name

def get_comp_dev_info_new(inp,inp_df):
    try:
        inp_df = inp_df.loc[inp_df["Series"]==inp]
        oem = RCR_Sales_Data.loc[RCR_Sales_Data["Series"]==inp]["OEMGROUP"].unique()[0].upper()
        device_name = oem+" "+inp
        sales_device_name = inp
        sentiment_device_names = get_all_sent_dev_name_new(sales_device_name)
        temp, img_link = get_device_image_new(sentiment_device_names[0])
        sales_units = str(round(inp_df["Sales_Units"].iloc[0]))+"K"
        asp = "$"+str(round(inp_df["CompetitorASP"].iloc[0]))
        net_sentiment, aspect_sentiment = get_net_sent_new(sentiment_device_names)
        star_rating_html = get_star_rating_html_new(net_sentiment)
        min_date, max_date = get_date_range_all_new([sales_device_name])
    except:
        print(f"Error occured in get_comp_dev_info_new() for {inp}")
        device_name = "NA"
        sales_device_name = "NA"
        sentiment_device_names = []
        img_link = "NA"
        sales_units = "NA"
        asp = "NA"
        aspect_sentiment = None
        star_rating_html = "NA"
        min_date = None
        max_date = None
    return device_name, sales_device_name, sentiment_device_names, img_link, sales_units, asp, aspect_sentiment, star_rating_html, min_date, max_date

def get_dev_summ_details_new(inp):
    try:
        devices = identify_all_devices_new(inp)
        if not devices or len(devices) != 1:
            #generic function
            return None,None,None,None,None,None,None,None,None,None,None
        dev = devices[0][1]
        sales_device_names = []
        sentiment_device_names = devices[0][2]
        for i in sentiment_device_names:
            sales_device_names.append(get_sales_device_name_new(i))
        sales_device_names = list(set(sales_device_names))
        if dev:
            device_name = dev
        else:
            device_name = sales_device_names[0]
        dev, img_link = get_device_image_new(sentiment_device_names[0])
        net_sentiment, aspect_sentiment = get_net_sent_new(sentiment_device_names)
        total_sales, asp, high_specs, sale, comp_devices = get_sales_info_new(sales_device_names)
        star_rating_html = get_star_rating_html_new(net_sentiment)
        min_date, max_date = get_date_range_all_new(sales_device_names)
    except:
        print(f"Error occured in get_dev_summ_details_new() for {inp}")
        device_name = "NA"
        sales_device_names = []
        sentiment_device_names = []
        img_link = None
        total_sales = "NA"
        asp =  "NA"
        high_specs =  "NA"
        sale =  "NA"
        comp_devices = pd.DataFrame()
        star_rating_html = "NA"
        min_date = "NA"
        max_date = "NA"
    return device_name,sales_device_names,sentiment_device_names, img_link,total_sales, asp, high_specs, sale, comp_devices,star_rating_html,min_date, max_date

def get_device_summary_new(user_input,device_name):
    try:
        data = query_quant_devices_new(user_input)
        total_reviews = data.loc[data['ASPECT'] == 'TOTAL', 'REVIEW_COUNT'].iloc[0]
        data['REVIEW_PERCENTAGE'] = data['REVIEW_COUNT'] / total_reviews * 100
        dataframe_as_dict = data.to_dict(orient='records')
        data_new = data
        data_new = data_new.dropna(subset=['ASPECT_SENTIMENT'])
        data_new = data_new[~data_new["ASPECT"].isin(["Generic", "Account", "Customer-Service", "Browser"])]
        vmin = data_new['ASPECT_SENTIMENT'].min()
        vmax = data_new['ASPECT_SENTIMENT'].max()
        styled_df = data_new.style.applymap(lambda x: custom_color_gradient(x, vmin, vmax), subset=['ASPECT_SENTIMENT'])
        data_filtered = data_new[data_new['ASPECT'] != 'TOTAL']
        data_sorted = data_filtered.sort_values(by='REVIEW_COUNT', ascending=False)
        top_four_aspects = data_sorted.head(4)
        aspects_list = top_four_aspects['ASPECT'].to_list()
        formatted_aspects = ', '.join(f"'{aspect}'" for aspect in aspects_list)
    except:
        dataframe_as_dict = {}
        formatted_aspects = ""
        aspects_list = ["Price","Software","Performance","Design"]
        print(f"Error in get_detailed_summary_new(). Unable to generate quant data and aspects for the user input.")
    try:
        key_df = get_final_df_devices_new(aspects_list, device_name)
        b =  key_df.to_dict(orient='records')
    except:
        b = {}
        print(f"Error in get_detailed_summary(). Unable to generate keywords for certain aspects")
    try:
        su = query_detailed_summary_devices_new(user_input + ".Do this for " +  formatted_aspects +  " Aspects which have following Sentiment Scores: "+str(dataframe_as_dict)+ str(b))
    except Exception as e:
        su = "I don't have sufficient data to provide a complete and accurate response at this time. Please provide more details or context."
        print(f"Error in get_detailed_summary(). Cannot generate summary for the given input.\n\nError: {e}\n\n")
    return su

def comparison_device_details_new(sent_device_list):
    try:
        if sent_device_list:
            sales_devices = []
            for i in sent_device_list:
                sales_devices.append(get_sales_device_name_new(i))
            sales_devices = list(set(sales_devices))
            oem = RCR_Sales_Data.loc[RCR_Sales_Data["Series"]==sales_devices[0]]["OEMGROUP"].unique()[0].upper()
            device_name = oem+" "+sales_devices[0]
            dev, img_link = get_device_image_new(sent_device_list[0])
            net_sentiment, aspect_sentiment = get_net_sent_new(sent_device_list)
            total_sales, asp, high_specs, sale, comp_devices = get_sales_info_new(sales_devices)
            star_rating_html = get_star_rating_html_new(net_sentiment)
            min_date, max_date = get_date_range_all_new(sales_devices)
            aspects = ['Performance', 'Design', 'Display', 'Battery', 'Price', 'Software']
            asp_rating = []
            for i in aspects:
                asp_rating.append(get_star_rating_html_new(aspect_sentiment[i]))
            with st.container(border = True):
                with st.container(border = False,height = 200):
                    col1, inter_col_space, col2 = st.columns((1, 4, 1))
                    with inter_col_space:
                        if img_link:
                            image1 = load_and_resize_image_new(img_link, 150)
                            st.image(image1)
                        else:
                            st.write("Image not available for this product.")
                with st.container(height=170, border = False):
                    st.header(device_name)
                with st.container(height=50, border = False):
                    st.markdown(star_rating_html, unsafe_allow_html=True)
                with st.container(height=225, border = False):
                    st.write(f"Total Devices Sold: {total_sales}")
                    st.write(f"Average Selling Price: {asp}")
                    st.write(f"Highest Selling Specs: {high_specs} - {sale}")
                    st.markdown(f"<p style='font-size:12px;'>*sales data is from {min_date} to {max_date}</p>", unsafe_allow_html=True)
                with st.container(height=300, border = False):
                    st.subheader('Aspect Ratings')
                    for aspect, stars in zip(aspects, asp_rating):
                        st.markdown(f"{aspect}: {stars}",unsafe_allow_html=True)
                data_1 = Devices_Sentiment_Data_New.loc[Devices_Sentiment_Data_New["Product_Family"].isin(sent_device_list)]["Review"]
                a = device_name + "_Summary_Reviews.txt"
                data_1.to_csv(a, sep='\t')
                summary_1 = query_to_embedding_summarize_new("Give me the pros and cons of " + device_name, a)
                st.write(summary_1)
                save_history_devices_new(summary_1)
            st.session_state.curr_response+=f"Device Name: {device_name}<br><br>"
            if summary_1:
                st.session_state.curr_response+=f"{summary_1}<br><br>"
    except:
        print(f"Error occured in comparison_device_details_new()")
        st.write(f"Unable to generate response for the input query. Please rephrase and try again, or contact the developer of the tool.")

def comparison_view_sent_devices_new(device1_list, device2_list):
    try:
        st.write(r"$\textsf{\Large Device Comparison}$")
        st.session_state.curr_response+=f"Device Comparison<br><br>"
        col1, col2 = st.columns(2)
        with col1:
            comparison_device_details_new(device1_list)
        with col2:
            comparison_device_details_new(device2_list)
    except:
        print(f"Error in comparison_view_sent_devices_new() for {device1_list} and {device2_list}")
        st.write(f"Unable to generate response for the input query. Please rephrase and try again, or contact the developer of the tool.")

def dev_comp_new(device1, device2):
    try:
        sent_devices1 = []
        sent_devices2 = []
        out = match_device_new(device1)
        out = [x for x,y in out if y>95]
        if out:
            sent_devices1 = out
        out = match_device_new(device2)
        out = [x for x,y in out if y>95]
        if out:
            sent_devices2 = out
        if sent_devices1 and sent_devices2:
            comparison_view_sent_devices_new(sent_devices1,sent_devices2)
        elif sent_devices1:
            st.write(f"""Cannot identify "{device2}", please rephrase device name and try again.""")
        elif sent_devices2:
            st.write(f"""Cannot identify "{device1}", please rephrase device name and try again.""")
        else:
            st.write(f"""Cannot identify "{device1}" and "{device2}", please rephrase device names and try again.""")
    except:
        print(f"Error in dev_comp_new() for {device1} and {device2}")
        st.write(f"Unable to generate response for the input query. Please rephrase and try again, or contact the developer of the tool.")
    

def device_summ_new(user_input):
    try:
        inp = user_input
        new_inp_check = False
        if not hasattr(st.session_state, 'selected_devices'):
            st.session_state.selected_devices = [None,None]
        if not hasattr(st.session_state, 'past_inp2'):
            st.session_state.past_inp2 = None
        if not hasattr(st.session_state, 'past_inp_comp_dev2'):
            st.session_state.past_inp_comp_dev2 = []
        if not hasattr(st.session_state, 'display_history_devices'):
            st.session_state.display_history_devices = []
        if not hasattr(st.session_state, 'context_history_devices'):
            st.session_state.context_history_devices = []
        if not hasattr(st.session_state, 'curr_response'):
            st.session_state.curr_response = ""
        if (not st.session_state.past_inp2) or (st.session_state.past_inp2[0] != inp):
            new_inp_check = True
            st.session_state.past_inp_comp_dev2 = []
            device_name,sales_device_names,sentiment_device_name,img_link,total_sales, asp, high_specs, sale, comp_devices,star_rating_html,min_date, max_date = get_dev_summ_details_new(inp)
            if not device_name:
                return None
        else:
            new_inp_check = False
            old_inp, device_name,sales_device_names,sentiment_device_name,img_link,total_sales, asp, high_specs, sale, comp_devices,star_rating_html,min_date, max_date, summ = st.session_state.past_inp2

        html_code = f"""
            <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); display: flex; align-items: center;">
                <div style="flex: 1; text-align: center;">
                    <img src="data:image/jpeg;base64,{base64.b64encode(open(img_link, "rb").read()).decode()}"  style="width: 150px; display: block; margin: 0 auto;">
                    <p style="color: black; font-size: 18px;">{device_name.upper()}</p>
                    <p>{star_rating_html}</p>
                </div>
                <div style="width: 2px; height: 150px; border-left: 2px dotted #ccc; margin: 0 20px;"></div>
                <div style="flex: 2; color: black; font-size: 18px;">
                    <p>Total Devices Sold: <strong>{total_sales}</strong></p>
                    <p>Average Selling Price: <strong>{asp}</strong></p>
                    <p>Highest Selling Specs: <strong>{high_specs}</strong> - <strong>{sale}</strong></p><br>
                    <p style='font-size:12px;'>*sales data is from {min_date} to {max_date}</p>
                </div>
            </div>
            """
        st.markdown(html_code, unsafe_allow_html=True)
        if new_inp_check:
            st.session_state.curr_response+=f"{html_code}<br>"
            summ = get_device_summary_new(inp,sales_device_names[0])
    #         summ = "PLACEHOLDER SUMMARY"
            st.session_state.past_inp2 = (inp, device_name,sales_device_names,sentiment_device_name,img_link,total_sales, asp, high_specs, sale, comp_devices,star_rating_html,min_date, max_date, summ)
            st.session_state.curr_response+=f"Detailed Summary"
            st.session_state.curr_response+=f"<br>{summ}<br>"

        st.write("")
        st.write(r"$\textsf{\Large Detailed Summary}$")
        st.write(summ)
        save_history_devices_new(summ)
        st.session_state.selected_devices[0] = sentiment_device_name

        if len(comp_devices):
            st.write(r"$\textsf{\Large Compare with Similar Devices:}$")
            col_list = [None, None, None]
            checkbox_state = []
            comp_devices_list = comp_devices['Series'].tolist()
            for i in range(len(comp_devices_list)):
                if i<3:
                    checkbox_state.append(False)
            col_list[0], col_list[1], col_list[2] = st.columns(3)
            com_sent_dev_list = [None,None,None]
            for i in range(len(comp_devices_list)):
                if i < 3:
                    with col_list[i]:
                        if new_inp_check:
                            comp_device_name, comp_sales_device_name, comp_sent_device_name,comp_img_link, comp_sales_units, comp_asp, comp_aspect_sentiment, comp_star_rating_html, comp_min_date, comp_max_date = get_comp_dev_info_new(comp_devices_list[i], comp_devices)
                            st.session_state.past_inp_comp_dev2.append((comp_device_name, comp_sales_device_name, comp_sent_device_name, comp_img_link, comp_sales_units, comp_asp, comp_aspect_sentiment, comp_star_rating_html, comp_min_date, comp_max_date))
                        else:
                            comp_device_name, comp_sales_device_name, comp_sent_device_name, comp_img_link, comp_sales_units, comp_asp, comp_aspect_sentiment, comp_star_rating_html, comp_min_date, comp_max_date = st.session_state.past_inp_comp_dev2[i]

                        com_sent_dev_list[i] = comp_sent_device_name
                        with st.container(border = True, height = 320):
                            with st.container(border = False, height = 240):
                                html_content = f"""
                                <div style="text-align: center; display: inline-block; ">
                                    <img src="data:image/jpeg;base64,{base64.b64encode(open(comp_img_link, "rb").read()).decode()}" width = "80" style="margin-bottom: 10px;">
                                    <div style="font-size: 16px; color: #333;">{comp_device_name}</div>
                                    <div style="font-size: 14px; color: #666;">Sales: {comp_sales_units}</div>
                                    <div style="font-size: 14px; color: #666;">Average Selling Price: {comp_asp}</div>
                                    <p>{comp_star_rating_html}</p>
                                    <p style='font-size:10px;'>*sales data is from {comp_min_date} to {comp_max_date}</p>
                                </div>
                            """
                                st.markdown(html_content, unsafe_allow_html=True)

                            checkbox_state[i] = st.checkbox("Compare",key=f"comparison_checkbox_{i}")

        for i in range(len(checkbox_state)):
            if checkbox_state[i]:
                st.session_state.selected_devices[1] = com_sent_dev_list[i]
                break
            st.session_state.selected_devices[1] = None

        if st.session_state.selected_devices[1]:
            comparison_view_sent_devices_new(st.session_state.selected_devices[0],st.session_state.selected_devices[1])
            st.session_state.selected_devices = [None, None]
    except:
        print(f"Error in device_summ_new() for User Query: {user_input}")
        st.write(f"Unable to generate response for the input query. Please rephrase and try again, or contact the developer of the tool.")
