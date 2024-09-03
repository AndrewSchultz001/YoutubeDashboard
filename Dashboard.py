# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:03:24 2024

@author: spark
"""

# libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime

# define functions

def style_negative(v, props=''):
    try:
        return props if v < 0 else None
    except: 
        pass
    
def style_positive(v, props=''):
    try:
        return props if v > 0 else None
    except:
        pass
    
def audience_simple(country):
    if country == 'US':
        return 'USA'
    elif country == 'IN':
        return 'India'
    else: 
        return 'Other'

# load data
@st.cache_data
def load_Data():
    df_agg = pd.read_csv('Aggregated_Metrics_By_Video.csv')
    df_agg = df_agg.iloc[1:,:]
    
    # Editing Columns for df_agg
    df_agg.columns = ['Video', 'Video Title', 'Video Publish Time', 'Comments Added', 'Shares', 'Dislikes', 'Likes', 
                      'Subscribers Lost', 'Subscribers Gained', 'RPM(USD)', 'CPM(USD)', 'Average % Viewed', 
                      'Average View Duration', 'Views', 'Watch Time (Hours)', 'Subscribers', 
                      'Estimated Revenue (USD)', 'Impressions', 'Impressions CTR(%)']
    df_agg['Video Publish Time'] = pd.to_datetime(df_agg['Video Publish Time'], format="%b %d, %Y")
    df_agg['Average View Duration'] = df_agg['Average View Duration'].apply(lambda x: datetime.strptime(x, '%H:%M:%S'))
    df_agg['Average_Duration_Sec'] = df_agg['Average View Duration'].apply(lambda x: x.second + x.minute * 60 + x.hour * 3600)
    df_agg['Engagement_Ratio'] = (df_agg['Comments Added'] + df_agg['Shares'] + df_agg['Dislikes'] + df_agg['Likes']) / df_agg.Views
    df_agg['Views / Sub Gained'] = df_agg['Views'] / df_agg['Subscribers Gained']
    df_agg.sort_values(by = 'Video Publish Time', ascending = False, inplace = True)
    
    df_agg_sub = pd.read_csv('Aggregated_Metrics_By_Country_And_Subscriber_Status.csv')
    df_time = pd.read_csv('Video_Performance_Over_Time.csv')
    df_time['Date'] = df_time['Date'].str.replace('Sept', 'Sep')
    df_time['Date'] = pd.to_datetime(df_time['Date'], format="%d %b %Y")
    
    return df_agg, df_agg_sub, df_time

# Create dataframes from load_Data()
df_agg, df_agg_sub, df_time = load_Data()

# engineer data
df_agg_diff = df_agg.copy()
metric_date_12mo = df_agg_diff['Video Publish Time'].max() - pd.DateOffset(months = 12)
median_agg = df_agg_diff[df_agg_diff['Video Publish Time'] >= metric_date_12mo]
numeric_cols = median_agg.select_dtypes(include=[np.number])
median_agg = numeric_cols.median()
numeric_cols = np.array((df_agg_diff.dtypes == 'float64') | (df_agg_diff.dtypes == 'int64'))
df_agg_diff.iloc[:,numeric_cols] = (df_agg_diff.iloc[:,numeric_cols] - median_agg).div(median_agg)

#merge daily data with publish data to get delta 
df_time_diff = pd.merge(df_time, df_agg.loc[:,['Video', 'Video Publish Time']], left_on = 'External Video ID', right_on = 'Video')
df_time_diff['Days_Published'] = (df_time_diff['Date'] - df_time_diff['Video Publish Time']).dt.days

# get last 12 months of data rather than all data 
date_12mo = df_agg['Video Publish Time'].max() - pd.DateOffset(months = 12)
df_time_diff_yr = df_time_diff[df_time_diff['Video Publish Time'] >= date_12mo]

# Group by 'days_published' and calculate mean, median, 80th and 20th percentiles
views_days = (df_time_diff_yr.groupby('Days_Published')['Views'].agg(
    mean_views = np.mean, median_views = np.median, pct_80_views = lambda x: np.percentile(x, 80), 
    pct_20_views = lambda x: np.percentile(x, 20)).reset_index())

views_days = views_days[views_days['Days_Published'].between(0, 30)]
views_cumulative = views_days[['Days_Published', 'median_views', 'pct_80_views', 'pct_20_views']].copy()
views_cumulative[['median_views', 'pct_80_views', 'pct_20_views']] = views_cumulative[['median_views', 'pct_80_views', 'pct_20_views']].cumsum()


# What metrics are relevant?
# Difference from baseline
# Percent change by video

# build dashboard
add_sidebar = st.sidebar.selectbox('Aggregate or Individual Video', ('Aggregate Metrics', 'Individual Video Analysis'))

# total picture
if add_sidebar == 'Aggregate Metrics':
    df_agg_metrics = df_agg[['Video Publish Time', 'Views', 'Likes', 'Subscribers', 'Shares', 'Comments Added', 
                             'RPM(USD)', 'Average % Viewed', 'Average_Duration_Sec', 'Engagement_Ratio', 
                             'Views / Sub Gained']]
    
    metric_date_6mo = df_agg_diff['Video Publish Time'].max() - pd.DateOffset(months = 6)
    metric_date_12mo = df_agg_diff['Video Publish Time'].max() - pd.DateOffset(months = 12)
    metric_medians6mo = df_agg_metrics[df_agg_metrics['Video Publish Time'] >= metric_date_6mo]
    metric_medians12mo = df_agg_metrics[df_agg_metrics['Video Publish Time'] >= metric_date_12mo]
    numeric_cols_6mo = metric_medians6mo.select_dtypes(include=[np.number])
    numeric_cols_12mo = metric_medians12mo.select_dtypes(include=[np.number])
    metric_medians6mo = numeric_cols_6mo.median()
    metric_medians12mo = numeric_cols_12mo.median()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    columns = [col1, col2, col3, col4, col5]
    
    common_metrics = metric_medians6mo.index.intersection(metric_medians12mo.index)
    
    count = 0
    for i in metric_medians6mo.index:
        with columns[count]:
            # Correct delta calculation
            delta = (metric_medians6mo[i] - metric_medians12mo[i]) / metric_medians12mo[i]
            st.metric(label=i, value=round(metric_medians6mo[i], 1), delta="{:.2%}".format(delta))
            count += 1
            if count >= 5:
                count = 0
                
    df_agg_diff['Publish Date'] = df_agg_diff['Video Publish Time'].apply(lambda x: x.date())
    df_agg_diff_final = df_agg_diff.loc[:,['Video Title', 'Publish Date', 'Views', 'Likes', 'Subscribers', 'Shares',
                                           'Comments Added', 'RPM(USD)', 'Average % Viewed', 'Average_Duration_Sec',
                                           'Engagement_Ratio', 'Views / Sub Gained']]
    

    df_agg_numeric = df_agg_diff_final.select_dtypes(include=[np.number])
    df_agg_numeric_lst = df_agg_numeric.median().index.tolist()
    df_to_pct = {}
    for i in df_agg_numeric_lst:
        df_to_pct[i] = '{:.1%}'.format
    
    st.dataframe(df_agg_diff_final.style.hide().applymap(style_negative, props='color:red;').applymap(style_positive, props='color:green;').format(df_to_pct))

    
if add_sidebar == 'Individual Video Analysis':
    videos = tuple(df_agg['Video Title'])
    video_select = st.selectbox('Pick a Video:', videos)
    
    agg_filtered = df_agg[df_agg['Video Title'] == video_select]
    agg_sub_filtered = df_agg_sub[df_agg_sub['Video Title'] == video_select]
    agg_sub_filtered['Country'] = agg_sub_filtered['Country Code'].apply(audience_simple)
    agg_sub_filtered.sort_values(by = 'Is Subscribed', inplace = True)
    
    fig = px.bar(agg_sub_filtered, x = 'Views', y = 'Is Subscribed', color = 'Country', orientation = 'h')
    st.plotly_chart(fig)
    
    agg_time_filtered = df_time_diff[df_time_diff['Video Title'] == video_select]
    first_30 = agg_time_filtered[agg_time_filtered['Days_Published'].between(0, 30)]
    first_30 = first_30.sort_values('Days_Published')
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(x = views_cumulative['Days_Published'], y = views_cumulative['pct_20_views'],
                              mode = 'lines',
                              name = '20th percentile', line = dict(color = 'purple', dash = 'dash')))
    fig2.add_trace(go.Scatter(x = views_cumulative['Days_Published'], y = views_cumulative['median_views'],
                              mode = 'lines',
                              name = '50th percentile', line = dict(color = 'black', dash = 'dash')))
    fig2.add_trace(go.Scatter(x = views_cumulative['Days_Published'], y = views_cumulative['pct_80_views'],
                              mode = 'lines',
                              name = '80th percentile', line = dict(color = 'royalblue', dash = 'dash')))
    fig2.add_trace(go.Scatter(x = first_30['Days_Published'], y = first_30['Views'].cumsum(),
                              mode = 'lines',
                              name = 'Current Video', line = dict(color = 'firebrick', width = 8)))
    
    fig2.update_layout(title = 'View Comparison First 30 Days',
                       xaxis_title = 'Days Since Published',
                       yaxis_title = 'Cumulative Views')
    
    st.plotly_chart(fig2)
    
    
# individual video

# improvements