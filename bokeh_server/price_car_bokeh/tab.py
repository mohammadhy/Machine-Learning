"""
Dataset: car_price
Title: bokeh_server
@author: HY
Ex: Tab_Table
"""
import pandas as pd
import numpy as np
from bokeh.models import ColumnDataSource, Panel
from bokeh.models.widgets import TableColumn, DataTable

def table_tab(data):
    data_group =  data.groupby('Name')['Kilometers_Driven'].describe()
    data_group['mean'] = data_group['mean'].round(1)
    data_group['std'] = data_group['std'].round(1)
    data_group['min'] = data_group['min'].round(1)
    data_group['25%'] = data_group['25%'].round(1)
    data_group['50%'] = data_group['50%'].round(1)
    data_group['75%'] = data_group['75%'].round(1)
    data_group['max'] = data_group['max'].round(1)
    data_group = data_group.reset_index()
    column_s  = ColumnDataSource(data_group)
    
    company_ride = DataTable(
        source = column_s,
        columns=[
            TableColumn(field='Name', title='برند خودرو سازی'),
            TableColumn(field='count', title='تعداد مدل مختلف بر اساس برند'),
            TableColumn(field='mean', title='میانگین مسافت'),
            #TableColumn(field='std', title='انحراف استاندارد'),
            #TableColumn(field='min', title='کمترین مسافت طی شده'),
            #TableColumn(field='25%', title='چارک اول'),
            #TableColumn(field='50%', title='چارک دوم'),
            #TableColumn(field='75%', title='چارک سوم'),
            TableColumn(field='max', title='بیشتزین مسافت'),
            ], width = 1000)
    tab = Panel(child = company_ride, title = 'مسافت طی شده بر اساس برند خودرو سازی')
    return tab