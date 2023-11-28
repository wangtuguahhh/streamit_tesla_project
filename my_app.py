import streamlit as st
import pandas as pd
import plotly.express as px

# Set Page Layout
st.set_page_config(layout='wide')

# load data
data = pd.read_csv('FinBERT_Final_Results_Tesla_News_Only.csv', index_col = 0)
data['date'] = pd.to_datetime(data['date'])

# write a page title
st.title('Sentiment Analysis for Investment Strategies on Tesla Stock')

# subheader
st.subheader('Analysis on Tesla Stock and News from 2023-9-12 to 2023-10-11')

# separator
st.markdown('---')

# add a date range slider

# Convert dates to strings for the slider
data['date_int'] = (data['date']- data['date'].min()).dt.days
min_date_int = data['date_int'].min()
max_date_int = data['date_int'].max()

selected_date_range_int = st.slider(
    'Select day range (number of days since the start date)',
    min_value=min_date_int,
    max_value=max_date_int,
    value=(min_date_int, max_date_int),
    step = 1
)

# Convert selected date range back to Timestamps
selected_date_range = (
    data['date'].min() + pd.to_timedelta(selected_date_range_int[0], unit='D'),
    data['date'].min() + pd.to_timedelta(selected_date_range_int[1], unit='D')
)

filtered_data = data[(data['date'] >= selected_date_range[0]) & (data['date'] <= selected_date_range[1])]

# add graphics
col1, col2 = st.columns(2)

# column 1 - line chart for stock price
with col1:
    g1 = px.line(filtered_data[~filtered_data['open'].isnull()],
                 x = 'date',
                 y = ['open', 'close'],
                 color_discrete_map = {
                     'open': 'royalblue',
                     'close': 'pink'
                 },
                 title = '| Tesla Stock Price Movements'
    )
    st.plotly_chart(g1, use_container_width=True)

# column 2 - bar plot for one-day stock close price change vs news sentiments (positive - negative)
with col2:
    g2 = px.bar(filtered_data.melt(id_vars = ['date'], value_vars = ['yesterday_move', 'pos_neg_diff_shift_1'],
                          var_name = 'series', value_name = 'Value'),
                x = 'date',
                y = 'Value',
                color = 'series',
                barmode = 'overlay',
                title = '| Close Price Change from Yesterday VS Positive News - Negative News from Yesterday'      
    )
    st.plotly_chart(g2, use_container_width=True)
