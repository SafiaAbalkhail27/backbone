from django.shortcuts import render
import openai
import pytrends
from pytrends.request import TrendReq
import pandas as pd
from ast import literal_eval
from plotly.offline import plot
import plotly.graph_objs as go
import country_converter as coco
import plotly.express as px
import numpy as np
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, QuestionAnswerPrompt, BeautifulSoupWebReader, SimpleWebPageReader, GPTListIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import os
from django.shortcuts import render
from .forms import UploadFileForm
import csv 
import json 
import codecs


openai.api_key = "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"
os.environ["OPENAI_API_KEY"] = "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"
model_engine = "gpt-3.5-turbo" 
pytrend = TrendReq()
myIndex = GPTListIndex([])

def construct_index_from_data():
    directory_path = 'static/context'
    # create documents
    documents = SimpleDirectoryReader(directory_path).load_data()
    # create Index
    for document in documents:
        myIndex.insert(document)
construct_index_from_data()
# Create your views here.


def get_form(request):
    if request.method == 'POST':
        # Retrieve form data
        company_name = request.POST.get('CName')
        linkedin_link = request.POST.get('link')
        company_description = request.POST.get('companyDescription')
        company_size = request.POST.get('size')
        num_of_employees = request.POST.get('empNum')
        num_of_departments = request.POST.get('depNum')
        departments = request.POST.getlist('depar')
        other_department = request.POST.get('otherText')
        business_goals = request.POST.get('goals')
        issue = request.POST.get('issue')
        budget = request.POST.get('budget')
        budget_options = request.POST.get('budget-options')
        communication = request.POST.get('communication')
        training_and_support = request.POST.get('training')
        measure_of_success = request.POST.get('measure')
        feedback_mechanism = request.POST.get('feedback')
        primary_customers = request.POST.get('user')
        target_audience_interaction = request.POST.get('pain')
        technologies = request.POST.get('tech')
        digital_strategy = request.POST.get('strat')
        transformation_barriers = request.POST.get('barriers')
        success_metrics = request.POST.get('success')
        dedicatedIT_team = request.POST.get('team')
        transformation_targets = request.POST.get('what')
        employee_background = request.POST.get('background')
        available_resources = request.POST.get('resources')
        potential_risks_and_opportunities = request.POST.get('solution')
        potential_outcomes_and_impacts = request.POST.get('impact')
        options_tried = request.POST.get('previous')
        preferences_and_priorities = request.POST.get('communication')
        employee_openness = request.POST.get('open')
        
        # Create a dictionary with the form data
        form_data = {
            'company_name': company_name,
            'linkedin_link': linkedin_link,
            'company_description': company_description,
            'company_size': company_size,
            'num_of_employees': num_of_employees,
            'num_of_departments': num_of_departments,
            'departments': departments,
            'other_department': other_department,
            'business_goals': business_goals,
            'issue': issue,
            'budget': budget,
            'budget_options': budget_options,
            'communication': communication,
            'training_and_support': training_and_support,
            'measure_of_success': measure_of_success,
            'feedback_mechanism': feedback_mechanism,
            'primary_customers': primary_customers,
            'target_audience_interaction': target_audience_interaction,
            'technologies': technologies,
            'digital_strategy': digital_strategy,
            'transformation_barriers': transformation_barriers,
            'success_metrics': success_metrics,
            'dedicatedIT_team': dedicatedIT_team,
            'transformation_targets': transformation_targets,
            'employee_background': employee_background,
            'available_resources': available_resources,
            'potential_risks_and_opportunities': potential_risks_and_opportunities,
            'potential_outcomes_and_impacts': potential_outcomes_and_impacts,
            'options_tried': options_tried,
            'preferences_and_priorities': preferences_and_priorities,
            'employee_openness': employee_openness,
        }
        
        documents = SimpleDirectoryReader(form_data).load_data()
        
        myIndex.insert(documents)

    return get_results(request)

def get_results(request):
    query_str = "can you help with my digital strategy?"
    QA_PROMPT_TMPL = (
        
         ''' 
         {query_str}
         you are a consultant to an education institution who wants to do digital transformation and your job is to analyse the institution data and the information given to help you provide a digital transformation strategy
    
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
        "your answer should just be in json format\n"
    )
    QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

    query_engine = myIndex.as_query_engine(text_qa_template=QA_PROMPT)
    response = query_engine.query(query_str)
    print(response.response)

    return view_resutls(request, response)


def home(request):
    return render(request, 'index.html')

def analayse_trends():
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

    return context

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        
        jsonArray = []
        if form.is_valid():
            # Process the uploaded file
            uploaded_file = form.cleaned_data['file']
            if(uploaded_file.content_type == 'text/csv'):
                csvfile = csv.DictReader(codecs.iterdecode(uploaded_file, 'utf-8'))
                #convert each csv row into python dict
                for row in csvfile: 
                #add this python dict to json array
                    jsonArray.append(row)
        
        data = {}

        
        return render(request, 'success.html', {'file': uploaded_file, 'type':uploaded_file.content_type, 'data': data})
    else:
        form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})

        
def view_resutls(request, response):
    #do all the stpes
    
    #2 get graphs
    graphs = analayse_trends()
    print(response)
    #3 render them as context to results.html
    return render(request, 'results.html')

def page1(request):
    return render(request, 'page1.html')

def page2(request):
    return render(request, 'page2.html')

def page3(request):
    return render(request, 'page3.html')