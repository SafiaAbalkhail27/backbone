from django.shortcuts import render
import openai
import pytrends
from pytrends.request import TrendReq
import pandas as pd
import json
from ast import literal_eval
from plotly.offline import plot
import plotly.graph_objs as go
import country_converter as coco
import plotly.express as px
import numpy as np
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, QuestionAnswerPrompt, BeautifulSoupWebReader, SimpleWebPageReader, GPTListIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import os

openai.api_key = "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"
os.environ["OPENAI_API_KEY"] = "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"
model_engine = "gpt-3.5-turbo" 
pytrend = TrendReq()
index = GPTVectorStoreIndex([])

# Create your views here.
def index(request):
    return render(request, 'index.html')

def analayse_trends(request):
    #request will be something like (VR in education), (Less employment in HR) etc
    tech = "VR"
    keywords = ["VR education"]
    

    #2 5 years trend chart
    pytrend.build_payload(keywords, timeframe='today 5-y')
    interest_over_time_df = pytrend.interest_over_time().drop(columns='isPartial')
    pd.options.plotting.backend = "plotly"
    fig = interest_over_time_df[keywords].plot()
    fig.update_layout(
        title_text='Search volume over time',
        legend_title_text='Search terms'
    )
    
    trends_5y = plot({'data': fig}, output_type='div')
    
    #3 last year trend bar-chart
    start_time = '2022-01-01'
    end_time = '2022-12-01'
    df = interest_over_time_df.loc[
        (interest_over_time_df.index >= start_time) & (interest_over_time_df.index < end_time)
    ]
    fig2 = go.Figure()
    for kw in keywords:
        fig2.add_trace(go.Bar(
            x=df.index.astype(str),
            y=df[kw],
            name=kw
        ))
    fig2.update_layout(
        barmode='group',
        xaxis_tickangle=-45,
        title_text=f'Search volume between {start_time} and {end_time}',
        legend_title_text='Search terms'
    )

    trends_last_y = plot({'data': fig2}, output_type='div')

    #4 trend by country
    interest_by_region_df = pytrend.interest_by_region(resolution='COUNTRY', inc_low_vol=True)
    interest_by_region_df.reset_index(inplace=True)
    df = interest_by_region_df.sort_values(keywords[0], ascending=False).head(20)
    fig3 = px.bar(df, x='geoName', y=keywords)
    fig3.update_layout(
        title_text=f'Search volumes by country',
        legend_title_text='Search terms'
    )
    fig3.update_yaxes(title_text='Volume')
    fig3.update_xaxes(title_text='Country')

    
    country_trend = plot({'data': fig3}, output_type='div')

    context = {'trends_5y': trends_5y, 'trends_last_y': trends_last_y, 'country_trend': country_trend}


    return render(request, 'results.html', context)


def construct_index_from_data(self):
    directory_path = 'static/context'
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(
    temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))
    prompt_helper = PromptHelper(
    max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # create documents
    documents = SimpleDirectoryReader(directory_path).load_data()
    # create Index
    for document in documents:
        index.insert(document)

    return index

def add_linked_in(request):
    return index

def get_results():
    query_str = "What did the author do growing up?"
    QA_PROMPT_TMPL = (
        
         ''' you are a consultant to an education institution who wants to do digital transformation using VR and your job is to analyse the institution data and the information given to help you provide a digital transformation strategy. this data consists of:
        - data about the company's culture and situation and needs which was taken from a survey  
        - a sample of the employees who works at the company and their skills  
        - info and articles about digital transformation in VR in education which you can use as a source to help you with the analyzation 
        - Facebook posts related to VR in education to gain more information
        - possible vendors providing the VR technology

         
        
        your strategy should follow these points:
        - why do the company need digital transformation to be applied depending on data analysis and the given data from the survey
        - what are the challenges, problems, and barriers that prevents you from using digital transformation on your case depending on the data analysis and the given data from the form
        - you should explain how to work around these challenges and barriers
        - you should provide similar cases that have the same purpose that the company want to achieve and it's the most similar to the company's goals and culture
        - you should be able to estimate the needed budget by comparing the budgets used in similar cases and the prices vendors provide
        - you should be able to give the company possible vendors suitable for their case and budget
        - you should implement a road map of the strategy that shows step by step what to do
        
        

        
        

        '''
        "{context_str}"
        "\n---------------------\n"
        "your answer should just be in json following this format: {query_str}\n"
    )
    QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

    query_engine = index.as_query_engine(
    text_qa_template=QA_PROMPT
    )
    response = query_engine.query(query_str)

    return response