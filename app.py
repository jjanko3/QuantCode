#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:17:49 2018

@author: jj
"""

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()

app.layout = html.Div(children = [html.H1(children = 'Dash Tutorials'), id='example', \
                                  figure={ \
            'data': [ \
                {'x': [1, 2, 3, 4, 5], 'y': [9, 6, 2, 1, 5], 'type': 'line', 'name': 'Boats'},\
                {'x': [1, 2, 3, 4, 5], 'y': [8, 7, 2, 7, 3], 'type': 'bar', 'name': 'Cars'}, \
            ],\
            'layout': {\
                'title': 'Basic Dash Example'
            }
        }
    )
                                  
                                  
                                  ])


if __name__ == '__main__':
    app.run_server(debug = True)