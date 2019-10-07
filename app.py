import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import pandas as pd
import datetime as dt
import datetime
import base64
import os
import time
import json
from dash.dependencies import Output
from dash.dependencies import Input
import plotly.graph_objs as go 
import matplotlib
import numpy as np
import psycopg2
import pandas.io.sql as sqlio

# dependencies
dbname='videodetect_db01'
user='videodetect01'
password='VideoDetect01!'
host='34.244.229.213'
port=8050
good_photos_folder_name = 'photos_good'
schedule_json_name = 'schedule.json'

with open(schedule_json_name, 'r') as f:
    schedule_dict = json.load(f)

schedule_dict = {pers: dt.datetime.strptime(x, '%H:%M').time() for pers,x in schedule_dict.items()}

def get_week_label(series):
    
    start_day = series.min().day
    end_date = series.min() + dt.timedelta(days=6)
    end_day = end_date.day
    month = end_date.month
    year = end_date.year

    return '{}-{}.{}.{}'.format(start_day, end_day, month, year)

# connect to DB and read a DataFrame from DB
conn = psycopg2.connect(dbname=dbname, 
						user=user, 
                        password=password, 
                        host=host)
cursor = conn.cursor()

sql = "select * from users_aggregate;"
df = sqlio.read_sql_query(sql, conn)
conn = None

# construct DataFrame's final look
df['Start'] = df['time_start'].dt.time
df['End'] = df['time_finish'].dt.time
df['Date'] = df['time_start'].dt.date
df['Day_of_week'] = df['time_start'].dt.day_name()
df['Week_number'] = df['time_start'].dt.week

df['Lower_line'] = df['person_id'].map(schedule_dict)
df['Higher_line'] = df['Lower_line'].apply(lambda x: (dt.datetime.combine(dt.date(1, 1, 1), x) + dt.timedelta(minutes=30)).time())

series_week_label = df.groupby(['Week_number'])['time_start'].apply(lambda x: get_week_label(x))
series_week_label.name = 'Week_label'
df = df.merge(series_week_label, how='left', left_on='Week_number', right_index=True)

df['Time_dif'] = ((df.time_finish - df.time_start)/np.timedelta64(1, 'h')).round(2)

for col in ['Start', 'End', 'Lower_line', 'Higher_line']:
    df[col] = pd.to_datetime(df[col].astype(str))

# create a list of matplotlib colors
hex_colors_dic = {}
rgb_colors_dic = {}
hex_colors_only = []
for name, hex in matplotlib.colors.cnames.items():
    hex_colors_only.append(hex)
    hex_colors_dic[name] = hex
    rgb_colors_dic[name] = matplotlib.colors.to_rgb(hex)

# colors that look ok on the dashboard
norm_colors = ['lightslategrey',                
               'mediumaquamarine', 
               'lightseagreen', 
               'mediumpurple', 
               'cornflowerblue', 
               'lightpink',
               'steelblue', 
               'palevioletred',
               'lightskyblue', 
               'rgb(216, 191, 216)',
               'rgb(108, 79, 126)',
               'rgb(3, 156, 148)', 
               'darkseagreen', 
               'lightcoral',
               'mediumvioletred']

# append colors from hex_colors_only if there ate not enough colors in norm_colors
for i in range(len(schedule_dict)-len(norm_colors)):
    norm_colors.append(hex_colors_only[i])

# create dictionary where each person is associated with a color
color_dict = dict(zip(schedule_dict.keys(), norm_colors))

# dictionary where each emotion is associated with color
emo_dict = {'neutral':'rgb(186, 201, 219)',
            'sad':'rgb(90,129,158)', 
            'fear':'rgb(125,122,162)',
            'angry':'rgb(246,126,125)',
            'happy':'rgb(255,193,167)',
            'surprise':'rgb(108,194,189)',
            'disgust':'rgb(143,169,106)'}

files = os.listdir(good_photos_folder_name)

# create dictionary where each person is associated with a photo
encoded_image = {}
for file in files:
    encoded_image[file.split('.')[0]] = base64.b64encode(open(good_photos_folder_name+'/'+file, 'rb').read())

person_id_list = list(schedule_dict.keys())
person_id_list.append('All')

body = dbc.Container([dbc.Row(html.Div([
                                        html.P(children='Week statistics', 
                                               style = {'textAlign': 'center', 
                                                        'font-size': '40px',
                                                        'font-family': "Courier New, monospace", 
                                                        'color': "#2B1232", 
                                                        'font-style': 'bold'})
                                        ],
                                        style= {'width':'100vw', 'height':'13vh', 'marginTop':'3vh'})),
                      dbc.Row([dbc.Col([
                      					html.Div([
                                                  html.P(children = 'Employee: ',  
                                                         style = {'textAlign': 'right', 
                                                                  'marginTop': '1%',
                                                                  'font-size': '20px',
                                                                  'font-family': "Courier New, monospace",
                                                                  'color': "#595B68"}),
                                                  html.Div([
                                                            html.P(children = 'Week: ',  
	                                                               style = {'textAlign': 'right', 
	                                                                        'marginTop': '1%', 
	                                                                        'font-size': '20px',
	                                                                        'font-family': "Courier New, monospace", 
	                                                                        'color': "#595B68"})])
                                                  ], style={'width':'85%'})
                      					],
	                                    style={'marginTop': '2vh'}),

                                dbc.Col([
                                    	dcc.Dropdown(id = 'person',
                                    				 options=[{'label': Person_id, 'value': Person_id} 
                                    				 				for Person_id in person_id_list],
	                                        		 value=df.person_id.unique()[0], 
	                                        		 style = {'textAlign': 'left', 
	                                                 		  'marginLeft': '-8%',
	                                                 		  'font-size': '20px',
	                                                 		  'font-family': "Courier New, monospace", 
	                                                 		  'color': "#595B68"}),
                                   		dcc.Dropdown(id = 'week',
                                        			 options=[{'label': week_label, 'value': week_label}
                                        			 				 for week_label in df.Week_label.unique()],
                                        			 value=df.Week_label.unique()[0], 
                                        			 style = {'textAlign': 'left', 
                                                 			  'marginLeft': '-8%', 
                                                		      'marginTop': '3%', 
                                                 			  'font-size': '20px', 
                                                 			  'font-family': "Courier New, monospace", 
                                                 			  'color': "#595B68"})
                                   		],
                                   		style = {'textAlign': 'left', 
                                             	 'width':'100%',
                                             	 'marginTop': '2vh'}),

                                dbc.Col([html.Div([
				                                    html.Img(id = 'person-img',
				                                             style={'height': '50%', 
			                                             			'width': '50%', 
			                                             			'marginTop': '-4vh',
			                                             			'marginLeft': '-10%'})
				                                    ], 
				                                    style={'textAlign': 'center'}),

				                                    html.P(id = 'total-hours', 
				                                           style =  {'textAlign': 'center', 
				                                                     'marginLeft': '-10%', 
				                                                 	 'marginTop': '2%',
				                                                     'font-size': '24px', 
				                                                     'font-family': "Courier New, monospace", 
				                                                     'font-style': 'bold',
				                                                     'color': "#595B68"})
			                                        ])
			                                ]),
                        dbc.Row([dbc.Col(html.Div([
                                                    dcc.Graph(id='num-of-hours',
                                                    	      style = {'height': '52vh'})
                                                    ]),
                        				style = {'marginLeft': '-5vw'}),
                                 dbc.Col(html.Div([
                                                    dcc.Graph(id='time-of-start',style = {'height': '52vh'})]), 
                                        style = {'marginRight': '-5vw'}),

                                 html.Div([dcc.Interval(id='interval-component',
                                             			interval=1*1000, # in milliseconds
                                             			n_intervals=0)])
                                ])
                     ], style = {'height': '100vh'} )

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([body], style={'backgroundImage': 'url(\'https://img3.akspic.ru/image/28815-angle-kvadrat-uzor-geometriya-simmetriya-1920x1080.jpg\')'})


@app.callback(dash.dependencies.Output('week', 'options'),
             [dash.dependencies.Input('interval-component', 'n_intervals'),
              dash.dependencies.Input('week', 'value')])
def update_dropdown(n_intervals, month):
    
    conn = psycopg2.connect(dbname=dbname, 
                            user=user, 
                            password=password, 
                            host=host)
    cursor = conn.cursor()
    sql = "select * from users_aggregate;"
    df = sqlio.read_sql_query(sql, conn)
    conn = None
    
    df['Start'] = df['time_start'].dt.time
    df['End'] = df['time_finish'].dt.time
    df['Date'] = df['time_start'].dt.date
    df['Day_of_week'] = df['time_start'].dt.day_name()
    df['Week_number'] = df['time_start'].dt.week
    series_week_label = df.groupby(['Week_number'])['time_start'].apply(lambda x: get_week_label(x))
    series_week_label.name = 'Week_label'
    df = df.merge(series_week_label, how='left', left_on='Week_number', right_index=True)
    df.sort_values(by='Week_label', inplace=True)
    
    return [{'label': week, 'value': week}
                                        for week in df['Week_label'].unique()]

@app.callback(Output('num-of-hours', 'figure'),
             [Input('interval-component', 'n_intervals'),
              Input('person', 'value'),
              Input('week', 'value')])

def update_num_of_hours(n_intervals, person, week):
    conn = psycopg2.connect(dbname=dbname, user=user, 
                        password=password, host=host)
    cursor = conn.cursor()

    sql = "select * from users_aggregate;"
    df = sqlio.read_sql_query(sql, conn)
    conn = None


    df['Start'] = df['time_start'].dt.time
    df['End'] = df['time_finish'].dt.time
    df['Date'] = df['time_start'].dt.date
    df['Day_of_week'] = df['time_start'].dt.day_name()
    df['Week_number'] = df['time_start'].dt.week
    series_week_label = df.groupby(['Week_number'])['time_start'].apply(lambda x: get_week_label(x))
    series_week_label.name = 'Week_label'
    df = df.merge(series_week_label, how='left', left_on='Week_number', right_index=True)

    df['Time_dif'] = ((df.time_finish - df.time_start)/np.timedelta64(1, 'h')).round(2)

    for col in ['Start', 'End']:
        df[col] = pd.to_datetime(df[col].astype(str))
    
    
    data = []
    
    layout = {'title': 'Number of hours worked during week <br>for {}'.format(person), 
              'plot_bgcolor': 'rgba(255, 255, 255, 0)', 
              'paper_bgcolor': 'rgba(255, 255, 255, 0.4)',
              'font': {'color': '#595B68'}, 
              'hovermode':'closest',
              'xaxis':{'tickangle':-30}}
    
    if person=='All':
        
        df_filt = df[df['Week_label']==week]
        df_gr = df_filt.groupby(['Day_of_week', 'emotion_start']).size()

        for emo in emo_dict.keys():
            df_emo = df_gr[df_gr.index.get_level_values(1)==emo]
            df_emo = pd.DataFrame(df_emo, columns=['num_people'])
            df_emo.reset_index(level=1, inplace=True)
            df_plot = pd.DataFrame()
            df_plot['Day_of_week'] = ['Monday','Tuesday','Wednesday','Thursday','Friday', 'Saturday', 'Sunday']
            df_plot = df_plot.merge(df_emo, left_on='Day_of_week', right_index=True, how='left')

            y = df_plot.apply(lambda x: int(x.num_people) if x.emotion_start==emo else 0, axis=1).values
            x = df_plot.Day_of_week.values

            data.append(go.Bar(x=x,
                               y=y,
                               marker=dict(color=emo_dict[emo], opacity=0.9),
                               name=emo))
            layout['title'] = 'Emotions'
            layout['yaxis'] = {'title':'number of people'}
            layout['barmode'] = 'stack'
            layout['hovermode'] = 'x'
    
    else:   
        df_pers = df[(df['person_id'] == person)&(df['Week_label']==week)]
        
        df_plot = pd.DataFrame()
        df_plot['Day_of_week'] = ['Monday','Tuesday','Wednesday','Thursday','Friday', 'Saturday', 'Sunday']
        df_plot = df_plot.merge(df_pers, on='Day_of_week', how='left')
        
        layout['hovermode'] = 'x'
        layout['hoverinfo'] = 'y+text'

        data.append({'x': df_plot.Day_of_week, 'y': df_plot.Time_dif, 'type': 'bar', 
                        'text':'hours', 'marker': {'color': 'rgb(51, 69, 109)', 'opacity': 0.9}})
    
    return {'data': data,
            'layout': layout}

@app.callback(Output('time-of-start', 'figure'),
             [Input('interval-component', 'n_intervals'),
              Input('person', 'value'), 
              Input('week', 'value')])

def update_time_of_start(n_intervals, person, week):
    
    conn = psycopg2.connect(dbname=dbname, user=user, 
                        password=password, host=host)
    cursor = conn.cursor()

    sql = "select * from users_aggregate;"
    df = sqlio.read_sql_query(sql, conn)
    conn = None

    df['Start'] = df['time_start'].dt.time
    df['End'] = df['time_finish'].dt.time
    df['Date'] = df['time_start'].dt.date
    df['Day_of_week'] = df['time_start'].dt.day_name()
    df['Week_number'] = df['time_start'].dt.week

    df['Lower_line'] = df['person_id'].map(schedule_dict)
    df['Higher_line'] = df['Lower_line'].apply(lambda x: (dt.datetime.combine(dt.date(1, 1, 1), x) + dt.timedelta(minutes=30)).time())

    series_week_label = df.groupby(['Week_number'])['time_start'].apply(lambda x: get_week_label(x))
    series_week_label.name = 'Week_label'
    df = df.merge(series_week_label, how='left', left_on='Week_number', right_index=True)

    for col in ['Start', 'End', 'Lower_line', 'Higher_line']:
        df[col] = pd.to_datetime(df[col].astype(str))
    
    data = []
        
    if person=='All':
        
        df_week = df[df['Week_label']==week]
        
        df_filt = pd.DataFrame()
        df_filt['Day_of_week'] = ['Monday','Tuesday','Wednesday','Thursday','Friday', 'Saturday', 'Sunday']
        df_filt = df_filt.merge(df_week, on='Day_of_week', how='left')

        for pers in df_filt['person_id'].unique():
            if type(pers)!=str:
                df_pers = df_filt[df_filt['person_id']==pers]
                data.append(go.Scatter(
                                    x=df_pers.Day_of_week,
                                    y=df_pers.Start,
                                    mode='markers',
                                    marker=dict(size=0,
                                                color='rgb(0,0,0)',
                                                opacity=0),
                                    name=pers))
            else:
                df_pers = df_filt[df_filt['person_id']==pers]
                data.append(go.Scatter(
                                        x=df_pers.Day_of_week,
                                        y=df_pers.Start,
                                        mode='markers',
                                        marker=dict(size=9,
                                                    color=color_dict[pers],
                                                    opacity=0.9),
                                        name=pers))

        layout = {'title': 'Time of coming to work',
                  'yaxis': {'type':'date', 
                  			'tickformat': '%H:%M'},
                  'xaxis':{'tickangle':-30,
                           'autorange': False},
                  'plot_bgcolor': 'rgba(255, 255, 255, 0)', 
                  'paper_bgcolor': 'rgba(255, 255, 255, 0.4)', 
                  'hovermode':'x'}
        
    else: 
        
        df_pers = df[(df['person_id'] == person)&(df['Week_label']==week)]
        
        df_plot = pd.DataFrame()
        df_plot['Day_of_week'] = ['Monday','Tuesday','Wednesday','Thursday','Friday', 'Saturday', 'Sunday']
        df_plot = df_plot.merge(df_pers, on='Day_of_week', how='left')
                
        data.append({'x': df_plot.Day_of_week, 'y': df_plot.Lower_line, 'mode': 'lines', 
                    'line': {'dash': 'dash', 'width': 1.5, 'color': '#802D4B'}, 
                    'name': 'Schedule'})
        data.append({'x': df_plot.Day_of_week, 'y': df_plot.Higher_line, 'mode': 'lines',
                    'line': {'dash': 'dash', 'width': 1.5, 'color': '#802D4B'}, 
                    'fillcolor': 'rgba(209,180,182,0.2)', 'fill' : 'tonexty', 
                    'name': 'Schedule', 'showlegend': False})
        data.append({'x': df_plot.Day_of_week, 'y': df_plot.Start, 'type': 'line', 
                    'line': {'color': 'rgb(51, 69, 109)', 'opacity':0.9}, 'name': 'Work start'})
        
        layout = {'title': 'Time of coming to work  <br>for {}'.format(person),
                  'yaxis': {'type':'date', 'tickformat': '%H:%M'},
                  'xaxis':{'tickangle':-30},
                  'plot_bgcolor': 'rgba(255, 255, 255, 0)', 'paper_bgcolor': 'rgba(255, 255, 255, 0.4)'}
        
    return {'data': data, 'layout': layout}

@app.callback(Output('total-hours', 'children'),
             [Input('interval-component', 'n_intervals'),
              Input('person', 'value'), 
              Input('week', 'value')])

def update_total_hours(n_intervals, person, week):
    
    conn = psycopg2.connect(dbname=dbname, user=user, 
                        password=password, host=host)
    cursor = conn.cursor()

    sql = "select * from users_aggregate;"
    df = sqlio.read_sql_query(sql, conn)
    conn = None

    df['Week_number'] = df['time_start'].dt.week

    series_week_label = df.groupby(['Week_number'])['time_start'].apply(lambda x: get_week_label(x))
    series_week_label.name = 'Week_label'
    df = df.merge(series_week_label, how='left', left_on='Week_number', right_index=True)

    df['Time_dif'] = ((df.time_finish - df.time_start)/np.timedelta64(1, 'h')).round(2)

    if person=='All': 
        df_plot = df[df['Week_label']==week]
        df_total_hours_person = df_plot.groupby(['person_id'])['Time_dif'].sum()
        mean_hours_for_nonzero_people = df_total_hours_person.iloc[df_total_hours_person.to_numpy().nonzero()].mean()
        val = 'Mean total hours: '+str(round(mean_hours_for_nonzero_people, 1)) 
    
    else:
        df_plot = df[(df['person_id'] == person)&(df['Week_label']==week)]
        val = 'Total hours: '+str(round(df_plot.Time_dif.sum(), 1)) 
    
    return val

@app.callback(dash.dependencies.Output('person-img', 'src'),
             [dash.dependencies.Input('person', 'value')])

def update_img(person):
    
    return 'data:image/png;base64,{}'.format(encoded_image[person].decode()) 

if __name__ == "__main__":
    app.run_server(debug = False, port=port)
