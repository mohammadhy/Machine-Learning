"""
Title: bokeh_server
@author: HY
Dataset: car_price
"""
import pandas as pd
from bokeh.models.widgets import Tabs
from bokeh.io import curdoc
from tab import table_tab


data = pd.read_csv('test-data.csv', index_col = 0).dropna()

data['Name'] = data['Name'].transform(lambda x: x.split(' ')[0])

tab_table = table_tab(data)
tabs = Tabs(tabs=[tab_table])

curdoc().add_root(tabs)
