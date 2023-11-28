import streamlit as st
import pandas as pd
import plotly.express as px

# Set Page Layout
st.set_page_config(layout='wide')

# load data 1
data = pd.read_csv('FinBERT_Final_Results_Tesla_News_Only.csv', index_col = 0)
data['date'] = pd.to_datetime(data['date'])

# load data 2
news = pd.read_csv('News_Sentiments_FinBERT.csv', index_col=0)
news['date'] = pd.to_datetime(news['date'])

# write a page title
st.title('Sentiment Analysis for Investment Strategies on Tesla Stock')

# subheader
st.subheader('Analysis on Tesla Stock and Tesla News from `2023-9-12` to `2023-10-11`')

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
    filtered_data.rename(columns = {'yesterday_move':'close_price_change_from_yesterday', 'pos_neg_diff_shift_1':'pos_news_-_neg_news_from_yesterday'}, inplace = True)
    ind2 = filtered_data.melt(id_vars = ['date'], value_vars = ['close_price_change_from_yesterday', 'pos_news_-_neg_news_from_yesterday'],
                          var_name = 'series', value_name = 'Value')
    g2 = px.bar(ind2,
                x = 'date',
                y = 'Value',
                color = 'series',
                color_discrete_map = {
                     'close_price_change_from_yesterday': 'black',
                     'pos_news_-_neg_news_from_yesterday': 'green'
                 },
                barmode = 'overlay',
                title = '| Close Price Change from Yesterday VS [Positive News - Negative News] from Yesterday'      
    )
    st.plotly_chart(g2, use_container_width=True)
    st.text('For some days, Tesla stock price change correlated with the news sentiments,')
    st.text('when the black bar and the green bar had the same polarity.')

# separator
st.markdown('---')

# add subheader
st.subheader('Tesla News Dashboard')
"""
Tesla news from multiple sources were extracted with key word of "Tesla" or "Elon Musk". Some news are not directly related to Tesla and a classification label is available in the data.

"""

# add a date range slider

# Convert dates to strings for the slider
news['date_int'] = (news['date']- news['date'].min()).dt.days
min_daten_int = news['date_int'].min()
max_daten_int = news['date_int'].max()

selected_daten_range_int = st.slider(
    'Select day range for news (number of days since the start date)',
    min_value=min_daten_int,
    max_value=max_daten_int,
    value=(min_daten_int, max_daten_int),
    step = 1
)

# Convert selected date range back to Timestamps
selected_daten_range = (
    news['date'].min() + pd.to_timedelta(selected_daten_range_int[0], unit='D'),
    news['date'].min() + pd.to_timedelta(selected_daten_range_int[1], unit='D')
)

filtered_news = news[(news['date'] >= selected_daten_range[0]) & (news['date'] <= selected_daten_range[1])]

# add select boxes
col6, col7 = st.columns(2)

with col6:
    filter_tesla = st.selectbox('Filter by whether directly related to Tesla or not',
             options = ('all', 'yes', 'no'))

with col7:
    countries = ['US','UK','Israel','India','Australia','Ireland','South Africa','Qatar','Canada']
    filter_country = st.multiselect('Filter by source country', options = countries, default=countries)

if filter_tesla == 'all':
    pass
else:
    filtered_news = filtered_news.query('one_shot_class == @filter_tesla')

filtered_news = filtered_news.query('source_country in @filter_country')

# add graphics
col3, col4, col5 = st.columns(3)

# column 1 - pie chart for news sentiments
with col3:
    ind3 = pd.DataFrame(filtered_news.groupby('description_label').description_label.count()).rename(columns={'description_label':'count'}).reset_index()
    g3 = px.pie(ind3,
                 values = 'count',
                 names = 'description_label',
                 color = 'description_label',
                 color_discrete_map = {
                     'positive': 'mediumseagreen',
                     'negative': 'tomato',
                     'neutral': 'antiquewhite'
                 },
                 title = '| Sentiment Distribution of Tesla News'
    )
    st.plotly_chart(g3, use_container_width=True)

# column 2 - bar plot for total news from each source
with col4:
    ind4 = pd.DataFrame(filtered_news.groupby('source_country').source_country.count()).rename(columns={'source_country':'count'}).reset_index()
    ind4.sort_values(by = ['count'], ascending = False, inplace = True)
    g4 = px.bar(ind4,
                y = 'source_country',
                x = 'count',
                color = 'source_country',
                barmode = 'group',
                title = '| News Distribution based on Source Country'      
    )
    st.plotly_chart(g4, use_container_width=True)


# column 3 - bar plot for news sentiment polarity
with col5:
    count1 = pd.DataFrame(filtered_news.groupby('source_name')['description_label'].apply(lambda x: x[x =='negative'].count() / x.count()).reset_index())
    count2 = pd.DataFrame(filtered_news.groupby('source_name')['description_label'].apply(lambda x: x[x =='positive'].count() / x.count()).reset_index())
    ind5 = pd.merge(count1, count2, on = 'source_name', how = 'left')
    ind5['polarity'] = ind5['description_label_y'] - ind5['description_label_x']
    ind5 = ind5.sort_values(by = ['polarity'], ascending = True)
    ind5['color'] = ind5['polarity'].apply(lambda x: 'more positive than negative' if x > 0 else 'more negative than positive')
    g5 = px.bar(ind5,
                y = 'source_name',
                x = 'polarity',
                color = 'color',
                barmode = 'group',
                color_discrete_map={'more negative than positive': 'tomato', 'more positive than negative': 'mediumseagreen'},
                title = '| News Sentiment Polarity based on Source (NEUTRAL IGNORED)'      
    )
    st.plotly_chart(g5, use_container_width=True)
