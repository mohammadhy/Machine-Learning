"""
Created on Sun Mar 14 18:39:22 2021

@author: HY
Ex: Basic Bokeh Server
"""

from random import random 
from bokeh.layouts import Column
from bokeh.models import Button
from bokeh.plotting import figure, curdoc
from bokeh.palettes import Spectral3

p = figure(x_range=(0, 100), y_range=(0, 100))
res = p.text(x = [], y = [], text = [], text_color = [])
datasource = res.data_source


i = 0
def show_random_num():
    global i 
    new = dict()
    new['x'] = datasource.data['x'] + [random() * 70 + 15]
    new['y'] = datasource.data['y'] + [random() * 70 + 15]
    new['text'] = datasource.data['text'] + [str(random())]
    new['text_color'] = datasource.data['text_color'] + [Spectral3[i%3]]
    datasource.data = new 
    i+=1
    
b = Button(label='Show')
b.on_click(show_random_num)
curdoc().add_root(Column(b ,p))




