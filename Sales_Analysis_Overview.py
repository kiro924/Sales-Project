
import plotly.express as px
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sales and Returns Performance KPI", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")
pages=st.sidebar.selectbox('Select Page', ['Home Page' , "📊Sales Analysis Page", "🔁Return Analysis Page"])

st.markdown("""
    <style>
        .title {
            background-color: #ffffff;
            color: #616f89;
            padding: 10px;
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            border: 4px solid #000083;
            border-radius: 10px;
            box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.2);
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: #ffffff;
            border: 2px solid #000083;
            border-radius: 15px;
            padding: 15px;
            box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin-bottom: 15px;
            transition: transform 0.2s ease-in-out;
        }
        .metric-card:hover {
            box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.3);
            transform: scale(1.02);
        }
        .metric-value {
            color: #000083;
            font-size: 26px;
            font-weight: 600;
            font-style: italic;
            text-shadow: 1px 1px 2px #000083;
            margin: 10px 0;
        }
        .metric-label {
            margin-bottom: 5px;
            font-size: 20px;
            font-weight: 500;
            color: #999999;
        }
        .expander-header {
            font-size: 24px !important;
            font-weight: bold !important;
            color: #000083 !important;
        }
    </style>
""", unsafe_allow_html=True)

if pages=='Home Page':

    st.markdown('<div class="title">Sales & Returns Analysis Overview</div>', unsafe_allow_html=True)
    st.image("dataset-cover.jpg")
    
    def load_data():
         return pd.read_csv('New Data.csv')
    
    df=load_data()
    
    st.subheader("🔁 Sales KPIs")
    with st.sidebar:
        st.header("🔍Sales Filter Options")
        Seasons=df['Order Season'].unique()
        Continent=df['Continent'].unique()
        Country=df['Country'].unique()
    
        season=st.multiselect('🛒Order Season', Seasons, default=Seasons)
        continent=st.multiselect('🌍Continent', Continent, default=Continent)
        country=st.multiselect('🌐Country', Country, default=Country)
    
        filtered_df=df[
        (df['Order Season'].isin(season)&
        (df['Continent'].isin(continent))&
        (df['Country'].isin(country))
        )]
    
    Total_Orders=filtered_df['OrderQuantity'].sum()
    Total_Revenue=filtered_df['Revenue'].sum()
    Total_Profit=filtered_df['Profit'].sum()
    Avg_Order_Value=Total_Revenue / Total_Orders
    Profit_Margin=Total_Profit / Total_Revenue *100
    
    # 🏆 Top Insights
    Top_Selling_Product=filtered_df.groupby('ProductName')[['Revenue']].sum().idxmax().values[0]
    Top_Region_by_Revenue=filtered_df.groupby('Region')[['Revenue']].sum().idxmax().values[0]
    Top_Country_by_Profit=filtered_df.groupby('Country')[['Revenue']].sum().idxmax().values[0]
    
    # 📂 Category & Subcategory
    Best_Category=filtered_df.groupby('CategoryName')[['Revenue']].sum().idxmax().values[0]
    Best_Subcategory=filtered_df.groupby('SubcategoryName')[['Revenue']].sum().idxmax().values[0]
    Best_Model=filtered_df.groupby('ModelName')[['Revenue']].sum().idxmax().values[0]
    
    with st.expander('📈 Sales Performance'):
        col1,col2,col3=st.columns([2,2,2])
        
        col1.metric('💰 Total Revenue', f'${Total_Revenue:,.2f}')
        col2.metric('📈 Total Profit', f'${Total_Profit:,.2f}')
        col3.metric('📊 Profit Margin', f'${Profit_Margin:,.2f}')
        col1.metric('🛒 Total Orders', f'{Total_Orders:,}')
        col2.metric('🧾 Avg. Order Value', f'${Avg_Order_Value:,.2f}')
        
    with st.expander('🏆 Top Insights'):
        col1,col2,col3=st.columns(3)
    
        col1.metric('🥇 Top Product', f'{Top_Selling_Product}')
        col2.metric('🌍 Top Region', f'{Top_Region_by_Revenue}')
        col3.metric('🌐 Top Country', f'{Top_Country_by_Profit:}')
    
    with st.expander('📂 Products Details'):
        col1,col2,col3=st.columns(3)
    
        col1.metric('📌 Best Category', f'{Best_Category}')
        col2.metric('🔍 Best Subcategory', f'{Best_Subcategory}')
        col3.metric('🥇 Best Model', f'{Best_Model}')
    
    def load_return_data():
         return pd.read_csv('Returns.csv')
    
    returns=load_return_data()


    with st.sidebar:
        st.header("🔍 Return Filter Options")
        Seasons = returns['Season'].unique()
        Continent = returns['Continent'].unique()
        Country = returns['Country'].unique()
    
        season = st.multiselect('🛒 Return Season', Seasons, default=Seasons)
        continent = st.multiselect('🌍 Continent', Continent, default=Continent)
        country = st.multiselect('🌐 Country', Country, default=Country)
    
        filtered_returns = returns[
            (returns['Season'].isin(season)) &
            (returns['Continent'].isin(continent)) &
            (returns['Country'].isin(country))
        ]
    # Volume Metrics
    Total_Returns = filtered_returns['ReturnQuantity'].sum()
    Top_Category = filtered_returns.groupby('CategoryName')['ReturnQuantity'].sum().idxmax()
    Top_Model = filtered_returns.groupby('ModelName')['ReturnQuantity'].sum().idxmax()
    
    # Financial Impact
    Revenue_Loss = filtered_returns['Returned Revenue Loss'].sum()
    Internal_Return_Cost = filtered_returns['Internal Cost of Returns'].sum()
    Profit_Lost_on_Returns = filtered_returns['Profit Lost on Returns'].sum()
    Avg_Loss_per_Return = Profit_Lost_on_Returns / Total_Returns if Total_Returns else 0
    
    # Replace the `with col2:` block with this
    st.subheader("🔁 Returns KPIs")
    
    # Financial + Volume KPIs (use container outside col2 context)
    with st.expander('💸 Return Financial Impact'):
        r1, r2 = st.columns(2)
        r1.metric('💰 Revenue Loss', f'${Revenue_Loss:,.2f}')
        r2.metric('🏭 Internal Return Cost', f'${Internal_Return_Cost:,.2f}')
        r1.metric('📉 Profit Lost on Returns', f"${Profit_Lost_on_Returns:,.2f}")
        r2.metric('📊 Avg. Loss per Return', f"${Avg_Loss_per_Return:,.2f}")
    

    with st.expander('📦 Return Volume Impact'):
        v1, v2, v3 = st.columns(3)
        v1.metric('🔁 Total Returns', f'{Total_Returns:,}')
        v2.metric('🥇 Top Category', Top_Category)
        v3.metric('📦 Top Model', Top_Model)
        
###____________________________________________________sales page_______________________________________________________
elif pages=='📊Sales Analysis Page':

    st.markdown('<div class="title">Sales Performance Dashboard</div>', unsafe_allow_html=True)
    
    def load_data():
         return pd.read_csv('New Data.csv')
    
    df=load_data()

    st.sidebar.header('🔍Sales Filter Options')
    Seasons=df['Order Season'].unique()
    Continent=df['Continent'].unique()
    Country=df['Country'].unique()

    season=st.sidebar.multiselect('🛒Order Season', Seasons, default=Seasons)
    continent=st.sidebar.multiselect('🌍Continent', Continent, default=Continent)
    country=st.sidebar.multiselect('🌐Country', Country, default=Country)

    filtered_df=df[
    (df['Order Season'].isin(season)&
    (df['Continent'].isin(continent))&
    (df['Country'].isin(country))
    )]
    
    st.subheader("📈 Univariate Analysis")
    select_col=st.selectbox("Select a column for univariate analysis:", filtered_df.columns)

    if pd.api.types.is_numeric_dtype(filtered_df[select_col]):
        col1,col2=st.columns(2)
        col1.plotly_chart(px.histogram(filtered_df,x=select_col,nbins=20,barmode='group',text_auto=True, 
        color_discrete_sequence=['#1f77b4'], 
        ).update_layout(xaxis_title=select_col, yaxis_title='Frequency', plot_bgcolor='rgba(0,0,0,0)',  
        ).update_traces(
        hovertemplate=f"<b>{select_col}</b>: %{{x}}<br>Count: %{{y}}<extra></extra>"
        ))
        
        fig = px.box(filtered_df,x=select_col,title=f'Box Plot of {select_col}'.title(),
                     color_discrete_sequence=['#1f77b4']
                    )
    
        fig.update_layout(
            xaxis_title=select_col,
            yaxis_title="Value",
            plot_bgcolor='rgba(0,0,0,0)'
        )
    
        col2.plotly_chart(fig)

        with st.expander(f"📊 Detailed Statistics for {select_col.title()}"):
            col1,col2,col3=st.columns(3)
            col1.write(filtered_df[select_col].describe())
            col2.write("🔼 Highest 5 Values:")
            col2.dataframe(filtered_df.nlargest(5, select_col)[[select_col]])
            col3.write("🔽 Lowest 5 Values:")
            col3.dataframe(filtered_df.nsmallest(5, select_col)[[select_col]])
            q1 = filtered_df[select_col].quantile(0.25)
            q3 = filtered_df[select_col].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            lower_bound = q1 - 1.5 * iqr
            outliers = filtered_df[(filtered_df[select_col] > upper_bound) | (filtered_df[select_col] < lower_bound)]
            st.write(f"🚨 Outliers count: {outliers.shape[0]}")

    else:
        col1,col2=st.columns(2)
        cat_df=filtered_df[select_col].value_counts().reset_index().head(10)
        if len(cat_df) <6:
            cat_df.columns=[select_col, 'count']
            
            col1.plotly_chart(px.bar(cat_df, x=select_col, y='count', text_auto=True, title=f'Count of each {select_col}'.title(),
                                     labels={select_col: select_col.title(), 'count': 'Count'},
                                    color='count'))

            col2.plotly_chart(px.pie(cat_df, names=select_col, values='count', title=f'percentage of each {select_col}'.title(),
                                    colors='count'))
            with st.expander(f"📊 Frequency Distribution for {select_col.title()}"):
                col1,col2=st.columns(2)
                col1.write("🔢 Absolute Counts:")
                col1.write(filtered_df[select_col].value_counts())
            
                col2.write("📊 Percentage Distribution (%):")
                col2.write((filtered_df[select_col].value_counts(normalize=True) * 100).round(2))

        else:
            col1.plotly_chart(px.bar(cat_df, x=select_col, y='count', text_auto=True, title=f'Count of each {select_col}'.title(),
                                    color='count', labels=True))

            col2.plotly_chart(px.treemap(cat_df, path=[select_col], values='count', 
                 title=f'{select_col} Distribution Treemap'))

            with st.expander(f"📊 Frequency Distribution for {select_col.title()}"):
                col1,col2=st.columns(2)
                col1.write("🔢 Absolute Counts:")
                col1.write(filtered_df[select_col].value_counts())
            
                col2.write("📊 Percentage Distribution (%):")
                col2.write((filtered_df[select_col].value_counts(normalize=True) * 100).round(2))
        
    st.title("Choose Categorical Column")
    st.subheader("📈 Bivariate Analysis")

    time_based_cols = ['Month Name', 'Day', 'Order Season', 'Week']

    if select_col in ['ProductName', 'ModelName', 'Region', 'Country', 'Continent',
                  'SubcategoryName', 'CategoryName', 'Month Name', 'Day', 'Order Season', 'Week']:
    
        col1, col2 = st.columns(2)
    
        # PROFIT
        group_profit = filtered_df.groupby(select_col)['Profit'].sum().reset_index().sort_values(ascending=False, by='Profit').head(10)
        col1.plotly_chart(
            px.bar(group_profit, x=select_col, y='Profit', color="Profit",
                   text_auto=True, title=f'📦Profit for each {select_col}'.title(),
                   color_discrete_sequence=px.colors.qualitative.Bold,
                   labels={select_col: select_col, 'Profit': 'Profit ($)'}),
            use_container_width=True
        )

        col1.plotly_chart(px.box(filtered_df, x=select_col, y='Profit',
                                title=f'📦 Profit box plot by {select_col}'.title(),
                                ))
    
        # Optionally add line chart for time-like column
        if select_col in time_based_cols:
            group_profit = filtered_df.groupby(select_col)['Profit'].sum().reset_index()
            fig = px.line(group_profit, x=select_col, y='Profit', markers=True,
                          title=f'📦Trend of Profit over {select_col}'.title(),
                          labels={select_col: select_col, 'Profit': 'Profit ($)'})
            col1.plotly_chart(fig, use_container_width=True)
    
        # REVENUE
        group_revenue = filtered_df.groupby(select_col)['Revenue'].sum().reset_index().sort_values(ascending=False, by='Revenue').head(10)
        col2.plotly_chart(
            px.bar(group_revenue, x=select_col, y='Revenue', color="Revenue",
                   text_auto=True, title=f'📦Revenue for each {select_col}'.title(),
                   color_discrete_sequence=px.colors.qualitative.Bold,
                   labels={select_col: select_col, 'Revenue': 'Revenue ($)'}),
            use_container_width=True
        )

        col2.plotly_chart(px.box(filtered_df, x=select_col, y='Revenue',
                                title=f'📦 Revenue box plot by {select_col}'.title(),
                                ))
    
        # Optionally add line chart for time-like column
        if select_col in time_based_cols:
            group_revenue = filtered_df.groupby(select_col)['Revenue'].sum().reset_index().head(10)
            fig = px.line(group_revenue, x=select_col, y='Revenue', markers=True,
                          title=f'📦Trend of Revenue over {select_col}'.title(),
                          labels={select_col: select_col, 'Revenue': 'Revenue ($)'})
            col2.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Multivariate Analysis")
    col1, col2 = st.columns(2)
    time_based_cols = ['Month Name', 'Day', 'Week']
    
    if select_col in ['ProductName', 'ModelName', 'Region', 'Country', 'Continent',
                      'SubcategoryName', 'CategoryName', 'Month Name', 'Day', 'Week']:
    
        # ------------------ TOP 10 BY PROFIT ------------------
        top_10_profit_items = (
            filtered_df.groupby(select_col)['Profit'].sum().sort_values(ascending=False).head(10).index)
        group_profit = (
            filtered_df[filtered_df[select_col].isin(top_10_profit_items)]
            .groupby([select_col, 'Order Season'])['Profit']
            .sum()
            .reset_index()
        )
        ordered_items = group_profit.groupby(select_col)['Profit'].sum().sort_values(ascending=False).index

        col1.plotly_chart(
            px.bar(group_profit, x=select_col, y='Profit', color='Order Season',
                   text_auto=True, title=f'📦Profit in each season for Top 10 {select_col}'.title(),
                   labels={select_col: select_col, 'Profit': 'Profit ($)'},
                   category_orders={select_col: list(ordered_items)}),
            use_container_width=True
        )
    
        # 📈 Line chart for time-based column
        if select_col in time_based_cols:
            fig = px.line(group_profit, x=select_col, y='Profit', markers=True, color="Order Season",
                          title=f'📦Trend of Profit in each season over {select_col}'.title(),
                          labels={select_col: select_col, 'Profit': 'Profit ($)'})
            col1.plotly_chart(fig, use_container_width=True)
    
        # ------------------ TOP 10 BY REVENUE ------------------
        top_10_revenue_items = (filtered_df.groupby(select_col)['Revenue'].sum().sort_values(ascending=False).head(10).index)
        group_revenue = (
            filtered_df[filtered_df[select_col].isin(top_10_revenue_items)]
            .groupby([select_col, 'Order Season'])['Revenue']
            .sum()
            .reset_index()
        )

        ordered_items = group_revenue.groupby(select_col)['Revenue'].sum().sort_values(ascending=False).index
        col2.plotly_chart(
            px.bar(group_revenue, x=select_col, y='Revenue', color='Order Season',
                   text_auto=True, title=f'📦Revenue in each season for Top 10 {select_col}'.title(),
                   labels={select_col: select_col, 'Revenue': 'Revenue ($)'},
                   category_orders={select_col: list(ordered_items)}),
            use_container_width=True
        )
    
        # 📈 Line chart for time-based column
        if select_col in time_based_cols:
            fig = px.line(group_revenue, x=select_col, y='Revenue', markers=True, 
                          title=f'📦Trend of Revenue in each season over {select_col}'.title(),
                          labels={select_col: select_col, 'Revenue': 'Revenue ($)'})
            col2.plotly_chart(fig, use_container_width=True)

    if st.checkbox("📊 Display Pivot Table"):
        if select_col in ['ProductName', 'ModelName', 'Region', 'Country', 'Continent',
                          'SubcategoryName', 'CategoryName', 'Month Name', 'Day', 'Week']:
    
            values_col = st.selectbox("💰 Values", filtered_df.select_dtypes(include='number').drop('Week', axis=1).columns)
            aggfunc = st.selectbox("📐 Aggregation Function", ['sum', 'mean', 'count', 'max', 'min'])
    
            if aggfunc == 'sum':
                pivot_df = pd.pivot_table(filtered_df, index=select_col, columns='Order Season', values=values_col, aggfunc='sum', fill_value=0)
            elif aggfunc == 'mean':
                pivot_df = pd.pivot_table(filtered_df, index=select_col, columns='Order Season', values=values_col, aggfunc='mean', fill_value=0)
            elif aggfunc == 'count':
                pivot_df = pd.pivot_table(filtered_df, index=select_col, columns='Order Season', values=values_col, aggfunc='count', fill_value=0)
            elif aggfunc == 'max':
                pivot_df = pd.pivot_table(filtered_df, index=select_col, columns='Order Season', values=values_col, aggfunc='max', fill_value=0)
            else:
                pivot_df = pd.pivot_table(filtered_df, index=select_col, columns='Order Season', values=values_col, aggfunc='min', fill_value=0)
    
            # Create a copy to format HTML manually
            html_df = pivot_df.copy()
    
            for idx in html_df.index:
                row = html_df.loc[idx]
                max_val = row.max()
                min_val = row.min()
                html_df.loc[idx] = [
                    f'<span style="background-color:green; color:white; padding:2px 4px; border-radius:4px;">{v}</span>' if v == max_val else
                    f'<span style="background-color:red; color:white; padding:2px 4px; border-radius:4px;">{v}</span>' if v == min_val else
                    f'{v}'
                    for v in row
                ]
    
            styled_html = html_df.to_html(escape=False, classes='styled-table')
    
            # Dark theme CSS
            custom_css = """
                <style>
                .styled-table {
                    font-size: 15px;
                    border-collapse: collapse;
                    width: 100%;
                    background-color: #1e1e2f;
                    color: #f4f4f4;
                }
                .styled-table th, .styled-table td {
                    border: 2px solid #444;
                    padding: 10px;
                    text-align: center;
                }
                .styled-table th {
                    background-color: #2c3e50;
                    color: #ffffff;
                }
                .styled-table td {
                    color: #E0E0E0;
                }
                </style>
            """
    
            st.subheader(f"📊 Pivot Table for {values_col} in each season for each {select_col}".title())
            st.markdown(custom_css, unsafe_allow_html=True)
            st.markdown(styled_html, unsafe_allow_html=True)


##_________________________________________________________Return Page_________________________________________________    
else:

    def load_data():
     return pd.read_csv('Returns.csv')

    df=load_data()

    st.sidebar.header('🔍Returns Filter Options')
    Seasons=df['Season'].unique()
    Continent=df['Continent'].unique()
    Country=df['Country'].unique()
    
    season=st.sidebar.multiselect('🔁Season', Seasons, default=Seasons)
    continent=st.sidebar.multiselect('🌍Continent', Continent, default=Continent)
    country=st.sidebar.multiselect('🌐Country', Country, default=Country)
    
    filtered_df=df[
    (df['Season'].isin(season)&
    (df['Continent'].isin(continent))&
    (df['Country'].isin(country))
    )]
    
    st.subheader("📈 Univariate Analysis")
    select_col=st.selectbox("Select a column for univariate analysis:", filtered_df.columns)
    if pd.api.types.is_numeric_dtype(filtered_df[select_col]):
        col1,col2=st.columns(2)
        col1.plotly_chart(px.histogram(filtered_df,x=select_col,nbins=20,barmode='group',text_auto=True, 
        color_discrete_sequence=['#1f77b4'], 
        ).update_layout(xaxis_title=select_col, yaxis_title='Frequency', plot_bgcolor='rgba(0,0,0,0)',  
        ).update_traces(
        hovertemplate=f"<b>{select_col}</b>: %{{x}}<br>Count: %{{y}}<extra></extra>"
        ))
        
        fig = px.box(filtered_df,x=select_col,title=f'Box Plot of {select_col}'.title(),
                     color_discrete_sequence=['#1f77b4']
                    )
    
        fig.update_layout(
            xaxis_title=select_col,
            yaxis_title="Value",
            plot_bgcolor='rgba(0,0,0,0)'
        )
    
        col2.plotly_chart(fig)

        with st.expander(f"📊 Detailed Statistics for {select_col.title()}"):
            col1,col2,col3=st.columns(3)
            col1.write(filtered_df[select_col].describe())
            col2.write("🔼 Highest 5 Values:")
            col2.dataframe(filtered_df.nlargest(5, select_col)[[select_col]])
            col3.write("🔽 Lowest 5 Values:")
            col3.dataframe(filtered_df.nsmallest(5, select_col)[[select_col]])
            q1 = filtered_df[select_col].quantile(0.25)
            q3 = filtered_df[select_col].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            lower_bound = q1 - 1.5 * iqr
            outliers = filtered_df[(filtered_df[select_col] > upper_bound) | (filtered_df[select_col] < lower_bound)]
            st.write(f"🚨 Outliers count: {outliers.shape[0]}")

    else:
        col1,col2=st.columns(2)
        cat_df=filtered_df[select_col].value_counts().reset_index().head(10)
        if len(cat_df) <6:
            cat_df.columns=[select_col, 'count']
            col1.plotly_chart(px.bar(cat_df, x=select_col, y='count', text_auto=True, title=f'Count of each {select_col}'.title(),
                                    color='count', labels=True))
            col2.plotly_chart(px.pie(cat_df, names=select_col, values='count', title=f'percentage of each {select_col}'.title(),
                                    color_discrete_sequence=px.colors.qualitative.Bold))
            with st.expander(f"📊 Frequency Distribution for {select_col.title()}"):
                col1,col2=st.columns(2)
                col1.write("🔢 Absolute Counts:")
                col1.write(filtered_df[select_col].value_counts())
            
                col2.write("📊 Percentage Distribution (%):")
                col2.write((filtered_df[select_col].value_counts(normalize=True) * 100).round(2))

        else:
            col1.plotly_chart(px.bar(cat_df, x=select_col, y='count', text_auto=True, title=f'Count of each {select_col}'.title(),
                                     labels=True, color='count'))

            col2.plotly_chart(px.treemap(cat_df, path=[select_col], values='count', 
                 title=f'{select_col} Distribution Treemap'))

            with st.expander(f"📊 Frequency Distribution for {select_col.title()}"):
                col1,col2=st.columns(2)
                col1.write("🔢 Absolute Counts:")
                col1.write(filtered_df[select_col].value_counts())
            
                col2.write("📊 Percentage Distribution (%):")
                col2.write((filtered_df[select_col].value_counts(normalize=True) * 100).round(2))

    st.subheader("📈 Bivariate Analysis")
    st.write("Choose Categorical Column")

    time_based_cols = ['Month', 'Day', 'Order', 'Week']

    if select_col in ['ProductName', 'ModelName', 'Region', 'Country', 'Continent',
                  'SubcategoryName', 'CategoryName', 'Month', 'Day', 'Season', 'Week']:

        col1, col2 = st.columns(2)
    
        # PROFIT
        group_profit = filtered_df.groupby(select_col)['Profit Lost on Returns'].sum().reset_index().sort_values(ascending=False, by='Profit Lost on Returns').head(10)
        col1.plotly_chart(
            px.bar(group_profit, x=select_col, y='Profit Lost on Returns', color="Profit Lost on Returns",
                   text_auto=True, title=f'🧭Profit Lost on Returns for each {select_col}'.title(),
                   color_discrete_sequence=px.colors.qualitative.Bold,
                   labels={select_col: select_col, 'Profit Lost on Returns': 'Profit Lost on Returns ($)'}),
            use_container_width=True
        )

        col1.plotly_chart(px.box(filtered_df, x=select_col, y='Profit Lost on Returns',
                                title=f'🧭 Profit Lost on Returns box plot by {select_col}'.title()))
    
         # Optionally add line chart for time-like column
        if select_col in time_based_cols:
            group_profit = filtered_df.groupby(select_col)['Profit Lost on Returns'].sum().reset_index()
            fig = px.line(group_profit, x=select_col, y='Profit Lost on Returns', markers=True,
                          title=f'🧭Trend of Profit Lost on Returns over {select_col}'.title(),
                          labels={select_col: select_col, 'Profit Lost on Returns': 'Profit Lost on Returns ($)'})
            col1.plotly_chart(fig, use_container_width=True)

        # REVENUE
        group_revenue = filtered_df.groupby(select_col)['Returned Revenue Loss'].sum().reset_index().sort_values(ascending=False, by='Returned Revenue Loss').head(10)
        col2.plotly_chart(
            px.bar(group_revenue, x=select_col, y='Returned Revenue Loss', color="Returned Revenue Loss",
                   text_auto=True, title=f'🧭Returned Revenue Loss for each {select_col}'.title(),
                   color_discrete_sequence=px.colors.qualitative.Bold,
                   labels={select_col: select_col, 'Returned Revenue Loss': 'Returned Revenue Loss ($)'}),
            use_container_width=True
        )

        col2.plotly_chart(px.box(filtered_df, x=select_col, y='Returned Revenue Loss',
                                title=f'🧭 Returned Revenue Loss box plot by {select_col}'.title(),
                                ))
    
        # Optionally add line chart for time-like column
        if select_col in time_based_cols:
            group_revenue = filtered_df.groupby(select_col)['Returned Revenue Loss'].sum().reset_index()
            fig = px.line(group_revenue, x=select_col, y='Returned Revenue Loss', markers=True,
                          title=f'🧭Trend of Returned Revenue Loss over {select_col}'.title(),
                          labels={select_col: select_col, 'Returned Revenue Loss': 'Returned Revenue Loss ($)'})
            col2.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Multivariate Analysis")
    col1, col2 = st.columns(2)
    time_based_cols = ['Month', 'Day', 'Week']
    if select_col in ['ProductName', 'ModelName', 'Region', 'Country', 'Continent',
                  'SubcategoryName', 'CategoryName', 'Month', 'Day', 'Week']:
    
        # PROFIT
        top10_profit = (filtered_df.groupby(select_col)['Profit Lost on Returns'].sum().sort_values(ascending=False).head(10).index)
        group_profit=(filtered_df[filtered_df[select_col].isin(top10_profit)]
                     .groupby([select_col,'Season'])['Profit Lost on Returns'].sum().reset_index())

        ordered_items = group_profit.groupby(select_col)['Profit Lost on Returns'].sum().sort_values(ascending=False).index

        col1.plotly_chart(
            px.bar(group_profit, x=select_col, y='Profit Lost on Returns', color="Season",
                   text_auto=True, title=f'📦Profit Lost on Returns in each season for each  {select_col}'.title(),
                   category_orders={select_col: list(ordered_items)},
                   labels={select_col: select_col, 'Profit Lost on Returns': 'Profit Lost on Returns ($)'}),
            use_container_width=True
        )


         # Optionally add line chart for time-like column
        if select_col in time_based_cols:
            group_profit = filtered_df.groupby([select_col,'Season'])['Profit Lost on Returns'].sum().reset_index()
            fig = px.line(group_profit, x=select_col, y='Profit Lost on Returns', markers=True, color="Season",
                          title=f'📦Trend of Profit Lost on Returns in each season over {select_col}'.title(),
                          labels={select_col: select_col, 'Profit Lost on Returns': 'Profit Lost on Returns ($)'})
            col1.plotly_chart(fig, use_container_width=True)

        top10_revenue = (filtered_df.groupby(select_col)['Returned Revenue Loss'].sum().sort_values(ascending=False).head(10).index)
        group_revenue=(filtered_df[filtered_df[select_col].isin(top10_revenue)]
                       .groupby([select_col,'Season'])['Returned Revenue Loss'].sum().reset_index())

        ordered_items = group_revenue.groupby(select_col)['Returned Revenue Loss'].sum().sort_values(ascending=False).index
        col2.plotly_chart(
            px.bar(group_revenue, x=select_col, y='Returned Revenue Loss', color="Season",
                   text_auto=True, title=f'📦Returned Revenue Loss in each season for each {select_col}'.title(),
                   category_orders={select_col: list(ordered_items)},
                   labels={select_col: select_col, 'Returned Revenue Loss': 'Returned Revenue Loss ($)'}),
            use_container_width=True
        )

        
        # Optionally add line chart for time-like column
        if select_col in time_based_cols:
            group_revenue = filtered_df.groupby([select_col,'Season'])['Returned Revenue Loss'].sum().reset_index()
            fig = px.line(group_revenue, x=select_col, y='Returned Revenue Loss', markers=True,
                          title=f'📦Trend of Returned Revenue Loss in each season over {select_col}'.title(),
                          labels={select_col: select_col, 'Returned Revenue Loss': 'Returned Revenue Loss ($)'})
            col2.plotly_chart(fig, use_container_width=True)

        

    if st.checkbox("📊Display Pivot Table"):
        if select_col in ['ProductName', 'ModelName', 'Region', 'Country', 'Continent',
                          'SubcategoryName', 'CategoryName', 'Month', 'Day', 'Week']:
    
            values_col = st.selectbox(
                "💰 Values",
                filtered_df.select_dtypes(include='number').drop('Week', axis=1).columns
            )
            aggfunc = st.selectbox(
                "📐 Aggregation Function",
                ['sum', 'mean', 'count', 'max', 'min']
            )
    
            if aggfunc == 'sum':
                pivot_df = pd.pivot_table(filtered_df, index=select_col, columns='Season',
                                          values=values_col, aggfunc='sum', fill_value=0)
            elif aggfunc == 'mean':
                pivot_df = pd.pivot_table(filtered_df, index=select_col, columns='Season',
                                          values=values_col, aggfunc='mean', fill_value=0)
            elif aggfunc == 'count':
                pivot_df = pd.pivot_table(filtered_df, index=select_col, columns='Season',
                                          values=values_col, aggfunc='count', fill_value=0)
            elif aggfunc == 'max':
                pivot_df = pd.pivot_table(filtered_df, index=select_col, columns='Season',
                                          values=values_col, aggfunc='max', fill_value=0)
            else:
                pivot_df = pd.pivot_table(filtered_df, index=select_col, columns='Season',
                                          values=values_col, aggfunc='min', fill_value=0)
    
            # Highlight max & min per row
            def highlight_extremes(s):
                styles = []
                max_val = s.max()
                min_val = s.min()
                for v in s:
                    if v == max_val:
                        styles.append('background-color: green; color: black; font-weight: bold;')
                    elif v == min_val:
                        styles.append('background-color: #ff6666; color: white; font-weight: bold;')
                    else:
                        styles.append('background-color: #1e1e2f; color: #E0E0E0;')  # default dark mode
                return styles
    
            styled_pivot = (
                pivot_df
                .style
                .apply(highlight_extremes, axis=1)
                .set_table_styles([
                    {'selector': 'table', 'props': [
                        ('border-collapse', 'collapse'),
                        ('font-size', '15px'),
                        ('width', '100%')
                    ]},
                    {'selector': 'th', 'props': [
                        ('background-color', '#2c3e50'),
                        ('color', '#ffffff'),
                        ('border', '2px solid #444'),
                        ('padding', '10px'),
                        ('text-align', 'center')
                    ]},
                    {'selector': 'td', 'props': [
                        ('border', '2px solid #444'),
                        ('padding', '10px'),
                        ('text-align', 'center')
                    ]}
                ], overwrite=False)  # <- VERY important
            )
    
            st.subheader(f"📊 Pivot Table for {values_col} in each season for each {select_col}".title())
            st.markdown(styled_pivot.to_html(), unsafe_allow_html=True)


    
   
                
    
        
        


