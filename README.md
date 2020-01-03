
# Best Bets in Oregon Real Estate Investment

Oregon offers many exciting opportunities for investment in real estate, with [a market that has recovered strongly since the lows of the Great Recession](https://listwithclever.com/real-estate-blog/top-5-best-real-estate-investment-markets-in-oregon/). Just like anywhere else, there are risks to investing in real estate, and in Oregon there is a risk that housing market growth is slowing or plateauing, and that property values cannot increase much more without outstripping wage growth. In [an article for Forbes in April 2019](https://www.forbes.com/sites/ingowinzer/2019/04/25/how-best-to-invest-in-real-estate-in-the-northwest/#47f25e466150), Ingo Winzer recommended that investors focus on apartments, since the rental market will continue to grow as Oregon's cities expand. 

This project uses data from Zillow to determine the top 5 zipcodes for real estate investment in Oregon and forecasts their growth over the next five years. The dataset contains the median home prices per zipcode recorded monthly over the period from April 1996 to April 2018. 

To select the top 5 zipcodes, I looked at return on investment over 5-, 10-, and 22-year periods (as far back as the data goes) and selected zipcodes that were in the top 10 for growth in one or more of these periods. I calculated how much median price dropped during the recession and ranked the zipcodes according to which suffered the least loss. I also collected data on how many properties are currently listed for sale in each zipcode.

After selecting the top 5 zipcodes, I used time series modeling to predict how the median home price will grow in each over the next 5 years.

## Summary of findings

(For more details, see **Interpretations and recommendations** below.)

* In Portland, 97227 and 97217 are safe bets, although they can be competitive markets for investors. I predict ROIs of 48% and 53%, respectively, over the period 2018-2023. 

* For a riskier undertaking, 97739 in La Pine (Bend) offers a moderate number of low-priced properties with the potential for 45% ROI over the period 2018-2023. 

* For those who want to invest in Portland but can't handle the competition (and sticker prices) in 97227 or 97217, 97266 and 97203 are good alternatives, with a projected 55% ROI over the period 2018-2023. 

## About this repo

This repository contains the following files:

* **best-bets-or-real-estate.ipynb**: Jupyter Notebook containing project code

* **zillow_data.csv**: CSV file containing main dataset from Zillow

* **zillow_listings_counts.csv**: CSV file containing additional data from Zillow

* **presentation.pdf**: PDF file containing slides from a non-technical presentation of project results

For more technical details, see the Appendix below.

# Get started

## Import packages and load data

First I will import packages and the dataset. Note that I will be using Facebook Prophet for modeling; I will explain this choice in more detail in the **Modeling** section below.


```python
# Import needed packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style('white')
sns.set_context('talk')
sns.set_palette('colorblind')

import warnings
warnings.filterwarnings('ignore')

from fbprophet import Prophet as proph
```


```python
# Read in the data
df = pd.read_csv('zillow_data.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RegionID</th>
      <th>RegionName</th>
      <th>City</th>
      <th>State</th>
      <th>Metro</th>
      <th>CountyName</th>
      <th>SizeRank</th>
      <th>1996-04</th>
      <th>1996-05</th>
      <th>1996-06</th>
      <th>...</th>
      <th>2017-07</th>
      <th>2017-08</th>
      <th>2017-09</th>
      <th>2017-10</th>
      <th>2017-11</th>
      <th>2017-12</th>
      <th>2018-01</th>
      <th>2018-02</th>
      <th>2018-03</th>
      <th>2018-04</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>84654</td>
      <td>60657</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>Chicago</td>
      <td>Cook</td>
      <td>1</td>
      <td>334200.0</td>
      <td>335400.0</td>
      <td>336500.0</td>
      <td>...</td>
      <td>1005500</td>
      <td>1007500</td>
      <td>1007800</td>
      <td>1009600</td>
      <td>1013300</td>
      <td>1018700</td>
      <td>1024400</td>
      <td>1030700</td>
      <td>1033800</td>
      <td>1030600</td>
    </tr>
    <tr>
      <td>1</td>
      <td>90668</td>
      <td>75070</td>
      <td>McKinney</td>
      <td>TX</td>
      <td>Dallas-Fort Worth</td>
      <td>Collin</td>
      <td>2</td>
      <td>235700.0</td>
      <td>236900.0</td>
      <td>236700.0</td>
      <td>...</td>
      <td>308000</td>
      <td>310000</td>
      <td>312500</td>
      <td>314100</td>
      <td>315000</td>
      <td>316600</td>
      <td>318100</td>
      <td>319600</td>
      <td>321100</td>
      <td>321800</td>
    </tr>
    <tr>
      <td>2</td>
      <td>91982</td>
      <td>77494</td>
      <td>Katy</td>
      <td>TX</td>
      <td>Houston</td>
      <td>Harris</td>
      <td>3</td>
      <td>210400.0</td>
      <td>212200.0</td>
      <td>212200.0</td>
      <td>...</td>
      <td>321000</td>
      <td>320600</td>
      <td>320200</td>
      <td>320400</td>
      <td>320800</td>
      <td>321200</td>
      <td>321200</td>
      <td>323000</td>
      <td>326900</td>
      <td>329900</td>
    </tr>
    <tr>
      <td>3</td>
      <td>84616</td>
      <td>60614</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>Chicago</td>
      <td>Cook</td>
      <td>4</td>
      <td>498100.0</td>
      <td>500900.0</td>
      <td>503100.0</td>
      <td>...</td>
      <td>1289800</td>
      <td>1287700</td>
      <td>1287400</td>
      <td>1291500</td>
      <td>1296600</td>
      <td>1299000</td>
      <td>1302700</td>
      <td>1306400</td>
      <td>1308500</td>
      <td>1307000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>93144</td>
      <td>79936</td>
      <td>El Paso</td>
      <td>TX</td>
      <td>El Paso</td>
      <td>El Paso</td>
      <td>5</td>
      <td>77300.0</td>
      <td>77300.0</td>
      <td>77300.0</td>
      <td>...</td>
      <td>119100</td>
      <td>119400</td>
      <td>120000</td>
      <td>120300</td>
      <td>120300</td>
      <td>120300</td>
      <td>120300</td>
      <td>120500</td>
      <td>121000</td>
      <td>121500</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 272 columns</p>
</div>



Note that each row in this DataFrame represents a zipcode. The first seven columns contain identifying information for the zipcode, and the subsequent columns contain the median home price in that zipcode at monthly intervals from April 1996 to April 2018.

My analysis is limited to zipcodes in Oregon, so I will subset the data to just that state. A couple of these columns are not useful for my analysis, so I'm going to drop them right away to avoid confusion.


```python
# Select Oregon data and drop columns that definitely aren't needed
oregon = df[df.State == 'OR']
oregon.drop(['RegionID', 'SizeRank'], axis=1, inplace=True)
```


```python
# Check number of OR zipcodes
oregon.info()
```

    <class 'pandas.core.frame.DataFrame'>  
    Int64Index: 224 entries, 135 to 14716  
    Columns: 270 entries, RegionName to 2018-04  
    dtypes: float64(219), int64(47), object(4)  
    memory usage: 474.2+ KB


This leaves me with 224 zipcodes to analyze. Are there any missing values?


```python
# Allow display of a greater number of rows
pd.set_option('display.max_rows', 500)

# Check for missing values
oregon.isna().sum()
```




    RegionName     0  
    City           0  
    State          0  
    Metro          8  
    CountyName     0  
    1996-04       14  
    1996-05       14  
    1996-06       14  
    1996-07       14...
 
 
 _(See project notebook for the rest of this long printout.)_


Yes, there appear to be some missing values in the dataset, mostly for the earlier dates. I will backfill missing values row-wise at this point and check that they have been resolved. Note that backfilling is not actually a good strategy in the `Metro` column, but I will be dropping this column anyway.


```python
# Back-fill any missing values
oregon = oregon.fillna(method='bfill', axis=0)

# Check that NaNs are all resolved (should print nothing)
for col in oregon.columns:
    if oregon.isna().sum()[col] > 0:
        print(col)
```

I now have a clean dataset for the state of Oregon.

## Create custom functions

I need a few custom functions to avoid repeating code, so I will define those here.


```python
def melt_it(df):
    '''Melts data from wide to long format. Returns time series in a 
       format suited for use with Facebook Prophet.
       
       Dependencies: pandas.'''
    
    melted = pd.melt(df, var_name='ds')
    melted['ds'] = pd.to_datetime(melted['ds'], infer_datetime_format=True)
    melted.columns = ['ds', 'y']
    
    return melted
```


```python
def melt_plot(df, title):
    '''Melts time series data from wide to long format and produces a quick 
       line plot.'''
    
    melted = melt_it(df)
    
    plt.figure(figsize=(16,4))
    fig = plt.plot(melted['ds'], melted['y']);
    plt.title('Median home price in {}'.format(title))
    plt.ylabel('Median home price ($)')
    
    return melted, fig
```


```python
def prep_and_plot(data, title):
    '''Performs 90:10 train-test split on time series data of n=264. Plots
       the train and test series.
       
       Dependencies: matplotlib.'''
    
    # Split the data
    train = data[:236]
    test = data[236:]
    
    # Reset index of test set
    test.reset_index(inplace=True)
    test.drop('index', axis=1, inplace=True)
    
    # Plot the data
    plt.plot(train['ds'], train['y'], label='train');
    plt.plot(test['ds'], test['y'], label='test')
    plt.legend()
    plt.title('Training and validation data for {}'.format(title))
    plt.show();
    
    return train, test
```


```python
def proph_it(train, test, whole, interval=0.95, forecast_periods1=28, 
             forecast_periods2=60):
    '''Uses Facebook Prophet to fit model to train set, evaluate performance
       with test set, and forecast with whole dataset. The model has a 95%
       confidence interval by default.
       
       Remember: datasets need to have two columns, `ds` and `y`.
       Dependencies: fbprophet
       Parameters:
          train: training data
          test: testing/validation data
          whole: all available data for forecasting
          interval: confidence interval (percent)
          forecast_periods1: number of months for forecast on training data
          forecast_periods2: number of months for forecast on whole dataset'''
    
    # Fit model to training data and forecast
    model = proph(interval_width=interval)
    model.fit(train)
    future = model.make_future_dataframe(periods=forecast_periods1, freq='MS')
    forecast = model.predict(future)
    
    # Plot the model and forecast
    model.plot(forecast, uncertainty=True)
    plt.title('Training data with forecast')
    plt.legend();
    
    # Make predictions and compare to test data
    y_pred = model.predict(test)
    
    # Plot the model, forecast, and actual (test) data
    model.plot(y_pred, uncertainty=True)
    plt.plot(test['ds'], test['y'], color='r', label='actual')
    plt.title('Validation data v. forecast')
    plt.legend();
    
    # Fit a new model to the whole dataset and forecast
    model2 = proph(interval_width=interval)
    model2.fit(whole)
    future2 = model2.make_future_dataframe(periods=forecast_periods2, 
                                          freq='MS')
    forecast2 = model2.predict(future2)
    
    # Plot the model and forecast
    model2.plot(forecast2, uncertainty=True)
    plt.title('{}-month forecast'.format(forecast_periods2))
    plt.legend();
    
    # Plot the model components
    model2.plot_components(forecast);
    
    return y_pred, forecast2
```


```python
def last_forecast(df):
    '''Returns last predicted value, upper and lower bounds from a DataFrame 
       of predicted values returned by Facebook Prophet. 
       
       In other words, returns the latest prediction.'''
    
    date = str(df[-1:]['ds'])
    value = round(float(df[-1:]['yhat']),2)
    upper = round(float(df[-1:]['yhat_upper']),2)
    lower = round(float(df[-1:]['yhat_lower']),2)
    ci = (lower, upper)
    
    print('Prediction for last date of period:')
    print('Median home price: ${}'.format(value))
    print('95% CI: {}'.format(ci))

    return value, ci
```

# Identify top 5 zipcodes for investment in Oregon

Because the dataset contains only median home prices for each zipcode, I will need to use those to determine which are the best 5 zipcodes for investment. (Typically, an analysis like this would make use of other data as well, like features of particular houses, vacancy rates, etc.)

The dataset covers a 22-year period, so I will look for zipcodes that have shown the return on investment over various time intervals. I calculate this by finding the difference between median home price at two dates and dividing that by the price at the earlier date to get a percent change in price over that time interval.

Note that since I backfilled missing values above, zipcodes that had missing values for the earlier dates will probably show less growth than they might have if the data had not been missing. I don't expect this to have a huge impact on the analysis, and there's not much I could do about it anyway.

## Greatest ROI 1996-2018

Which zipcodes saw the greatest ROI over the period from 1996 to 2018?


```python
# Calculate growth over 22-year period and select top 10 by growth
oregon_copy = oregon.copy()
oregon_copy['growth'] = oregon_copy['2018-04'] - oregon_copy['1996-04']
oregon_copy['roi'] = oregon_copy['growth']/oregon_copy['1996-04']*100
top_10_growth_all = oregon_copy.sort_values('roi', ascending=False)[:10]
top_10_growth_all
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RegionName</th>
      <th>City</th>
      <th>State</th>
      <th>Metro</th>
      <th>CountyName</th>
      <th>1996-04</th>
      <th>1996-05</th>
      <th>1996-06</th>
      <th>1996-07</th>
      <th>1996-08</th>
      <th>...</th>
      <th>2017-09</th>
      <th>2017-10</th>
      <th>2017-11</th>
      <th>2017-12</th>
      <th>2018-01</th>
      <th>2018-02</th>
      <th>2018-03</th>
      <th>2018-04</th>
      <th>growth</th>
      <th>roi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>10068</td>
      <td>97227</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>84500.0</td>
      <td>85100.0</td>
      <td>85800.0</td>
      <td>86400.0</td>
      <td>87000.0</td>
      <td>...</td>
      <td>527300</td>
      <td>531100</td>
      <td>534600</td>
      <td>537100</td>
      <td>540300</td>
      <td>543600</td>
      <td>543300</td>
      <td>540300</td>
      <td>455800.0</td>
      <td>539.408284</td>
    </tr>
    <tr>
      <td>2520</td>
      <td>97211</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>88000.0</td>
      <td>88700.0</td>
      <td>89400.0</td>
      <td>90200.0</td>
      <td>91000.0</td>
      <td>...</td>
      <td>457500</td>
      <td>456400</td>
      <td>456100</td>
      <td>458800</td>
      <td>463200</td>
      <td>468000</td>
      <td>472300</td>
      <td>474300</td>
      <td>386300.0</td>
      <td>438.977273</td>
    </tr>
    <tr>
      <td>949</td>
      <td>97330</td>
      <td>Corvallis</td>
      <td>OR</td>
      <td>Corvallis</td>
      <td>Benton</td>
      <td>72000.0</td>
      <td>71900.0</td>
      <td>71800.0</td>
      <td>71500.0</td>
      <td>71200.0</td>
      <td>...</td>
      <td>357600</td>
      <td>361000</td>
      <td>363700</td>
      <td>367100</td>
      <td>370400</td>
      <td>369500</td>
      <td>365300</td>
      <td>362500</td>
      <td>290500.0</td>
      <td>403.472222</td>
    </tr>
    <tr>
      <td>1769</td>
      <td>97217</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>90600.0</td>
      <td>91500.0</td>
      <td>92300.0</td>
      <td>93200.0</td>
      <td>94000.0</td>
      <td>...</td>
      <td>434000</td>
      <td>433100</td>
      <td>433600</td>
      <td>436100</td>
      <td>439100</td>
      <td>442400</td>
      <td>445100</td>
      <td>445000</td>
      <td>354400.0</td>
      <td>391.169978</td>
    </tr>
    <tr>
      <td>4423</td>
      <td>97333</td>
      <td>Corvallis</td>
      <td>OR</td>
      <td>Corvallis</td>
      <td>Benton</td>
      <td>67300.0</td>
      <td>67500.0</td>
      <td>67600.0</td>
      <td>67600.0</td>
      <td>67600.0</td>
      <td>...</td>
      <td>323500</td>
      <td>326400</td>
      <td>329400</td>
      <td>332300</td>
      <td>334400</td>
      <td>333200</td>
      <td>329600</td>
      <td>326700</td>
      <td>259400.0</td>
      <td>385.438336</td>
    </tr>
    <tr>
      <td>2557</td>
      <td>97214</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>136700.0</td>
      <td>137700.0</td>
      <td>138800.0</td>
      <td>139900.0</td>
      <td>141000.0</td>
      <td>...</td>
      <td>590000</td>
      <td>587100</td>
      <td>586200</td>
      <td>589400</td>
      <td>593800</td>
      <td>597200</td>
      <td>598000</td>
      <td>596300</td>
      <td>459600.0</td>
      <td>336.210680</td>
    </tr>
    <tr>
      <td>3130</td>
      <td>97203</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>88000.0</td>
      <td>88700.0</td>
      <td>89500.0</td>
      <td>90300.0</td>
      <td>91200.0</td>
      <td>...</td>
      <td>373600</td>
      <td>373400</td>
      <td>373500</td>
      <td>375000</td>
      <td>377200</td>
      <td>379300</td>
      <td>379800</td>
      <td>378600</td>
      <td>290600.0</td>
      <td>330.227273</td>
    </tr>
    <tr>
      <td>8743</td>
      <td>97370</td>
      <td>Philomath</td>
      <td>OR</td>
      <td>Corvallis</td>
      <td>Benton</td>
      <td>69300.0</td>
      <td>69700.0</td>
      <td>70000.0</td>
      <td>70300.0</td>
      <td>70600.0</td>
      <td>...</td>
      <td>281300</td>
      <td>284400</td>
      <td>287100</td>
      <td>290100</td>
      <td>292900</td>
      <td>293000</td>
      <td>291800</td>
      <td>291500</td>
      <td>222200.0</td>
      <td>320.634921</td>
    </tr>
    <tr>
      <td>3153</td>
      <td>97321</td>
      <td>Albany</td>
      <td>OR</td>
      <td>Albany</td>
      <td>Linn</td>
      <td>71600.0</td>
      <td>71900.0</td>
      <td>72200.0</td>
      <td>72400.0</td>
      <td>72500.0</td>
      <td>...</td>
      <td>288600</td>
      <td>289500</td>
      <td>289700</td>
      <td>290700</td>
      <td>292100</td>
      <td>293200</td>
      <td>295200</td>
      <td>297700</td>
      <td>226100.0</td>
      <td>315.782123</td>
    </tr>
    <tr>
      <td>999</td>
      <td>97202</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>132600.0</td>
      <td>133400.0</td>
      <td>134200.0</td>
      <td>135000.0</td>
      <td>135800.0</td>
      <td>...</td>
      <td>522700</td>
      <td>521700</td>
      <td>522200</td>
      <td>525700</td>
      <td>530100</td>
      <td>534100</td>
      <td>535900</td>
      <td>535000</td>
      <td>402400.0</td>
      <td>303.469080</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 272 columns</p>
</div>



The zipcodes in the table above are the top 10 by ROI over the period from 1996-2018. Note that they are all mostly in the areas of Portland and Corvallis, with one in Albany. The huge increases&mdash;up to 539\% ROI&mdash;reflect rapid growth in these areas over this period. 


```python
# Plot the top 10 zipcodes by growth, 1996-2018
xticks = ['1996-04', '2000-04', '2004-04', '2008-04', '2012-04', '2016-04']
xlabels = ['1996', '2000', '2004', '2008', '2012', '2016']
yticks = [100000, 200000, 300000, 400000, 500000, 600000]
ylabels = ['100k', '200k', '300k', '400k', '500k', '600k']
plt.figure(figsize=(16,8))
for n, index in enumerate(top_10_growth_all.index):
    sample = top_10_growth_all.loc[index,'1996-04':'2018-04']
    zipcode = top_10_growth_all.loc[index]['RegionName']
    plt.plot(sample, label=top_10_growth_all.loc[index]['RegionName'])
plt.xticks(xticks, labels=xlabels)
plt.yticks(yticks, labels=ylabels)
plt.legend()
plt.title('Top 10 Zipcodes by Growth over Period 1996-2018')
plt.ylabel('Median home price ($)')
plt.xlabel('Year')
sns.despine()
plt.show();
```


![](./images/output_33_0.png)


## Greatest ROI 2008-2018

Next I will identify the top 10 zipcodes by ROI over the last decade. In 2008, most zipcodes were hitting their pre-crash peak for house prices, so this ten-year interval encompasses both the crash and subsequent recovery.


```python
# Calculate growth over 10 year period and identify top 10 zipcodes by growth
oregon_copy = oregon.copy()
oregon_copy['growth'] = oregon_copy['2018-04'] - oregon_copy['2008-04']
oregon_copy['roi'] = oregon_copy['growth']/oregon_copy['2008-04']*100
top_10_growth_10 = oregon_copy.sort_values('roi', ascending=False)[:10]
top_10_growth_10
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RegionName</th>
      <th>City</th>
      <th>State</th>
      <th>Metro</th>
      <th>CountyName</th>
      <th>1996-04</th>
      <th>1996-05</th>
      <th>1996-06</th>
      <th>1996-07</th>
      <th>1996-08</th>
      <th>...</th>
      <th>2017-09</th>
      <th>2017-10</th>
      <th>2017-11</th>
      <th>2017-12</th>
      <th>2018-01</th>
      <th>2018-02</th>
      <th>2018-03</th>
      <th>2018-04</th>
      <th>growth</th>
      <th>roi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1769</td>
      <td>97217</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>90600.0</td>
      <td>91500.0</td>
      <td>92300.0</td>
      <td>93200.0</td>
      <td>94000.0</td>
      <td>...</td>
      <td>434000</td>
      <td>433100</td>
      <td>433600</td>
      <td>436100</td>
      <td>439100</td>
      <td>442400</td>
      <td>445100</td>
      <td>445000</td>
      <td>173200.0</td>
      <td>63.723326</td>
    </tr>
    <tr>
      <td>10068</td>
      <td>97227</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>84500.0</td>
      <td>85100.0</td>
      <td>85800.0</td>
      <td>86400.0</td>
      <td>87000.0</td>
      <td>...</td>
      <td>527300</td>
      <td>531100</td>
      <td>534600</td>
      <td>537100</td>
      <td>540300</td>
      <td>543600</td>
      <td>543300</td>
      <td>540300</td>
      <td>209700.0</td>
      <td>63.430127</td>
    </tr>
    <tr>
      <td>644</td>
      <td>97206</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>96600.0</td>
      <td>97600.0</td>
      <td>98600.0</td>
      <td>99600.0</td>
      <td>100700.0</td>
      <td>...</td>
      <td>371900</td>
      <td>372000</td>
      <td>373000</td>
      <td>374900</td>
      <td>377100</td>
      <td>379600</td>
      <td>381800</td>
      <td>382200</td>
      <td>147000.0</td>
      <td>62.500000</td>
    </tr>
    <tr>
      <td>3130</td>
      <td>97203</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>88000.0</td>
      <td>88700.0</td>
      <td>89500.0</td>
      <td>90300.0</td>
      <td>91200.0</td>
      <td>...</td>
      <td>373600</td>
      <td>373400</td>
      <td>373500</td>
      <td>375000</td>
      <td>377200</td>
      <td>379300</td>
      <td>379800</td>
      <td>378600</td>
      <td>145500.0</td>
      <td>62.419562</td>
    </tr>
    <tr>
      <td>7028</td>
      <td>97218</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>100800.0</td>
      <td>101800.0</td>
      <td>102900.0</td>
      <td>103900.0</td>
      <td>104800.0</td>
      <td>...</td>
      <td>381400</td>
      <td>381300</td>
      <td>381700</td>
      <td>382500</td>
      <td>383400</td>
      <td>384900</td>
      <td>387200</td>
      <td>388400</td>
      <td>146500.0</td>
      <td>60.562216</td>
    </tr>
    <tr>
      <td>2520</td>
      <td>97211</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>88000.0</td>
      <td>88700.0</td>
      <td>89400.0</td>
      <td>90200.0</td>
      <td>91000.0</td>
      <td>...</td>
      <td>457500</td>
      <td>456400</td>
      <td>456100</td>
      <td>458800</td>
      <td>463200</td>
      <td>468000</td>
      <td>472300</td>
      <td>474300</td>
      <td>178400.0</td>
      <td>60.290639</td>
    </tr>
    <tr>
      <td>5968</td>
      <td>97215</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>134400.0</td>
      <td>135400.0</td>
      <td>136400.0</td>
      <td>137400.0</td>
      <td>138400.0</td>
      <td>...</td>
      <td>528800</td>
      <td>529600</td>
      <td>531000</td>
      <td>533800</td>
      <td>536900</td>
      <td>540400</td>
      <td>542300</td>
      <td>541400</td>
      <td>192400.0</td>
      <td>55.128940</td>
    </tr>
    <tr>
      <td>6568</td>
      <td>97216</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>103700.0</td>
      <td>104500.0</td>
      <td>105300.0</td>
      <td>106100.0</td>
      <td>107000.0</td>
      <td>...</td>
      <td>319500</td>
      <td>319900</td>
      <td>320900</td>
      <td>323200</td>
      <td>325900</td>
      <td>328900</td>
      <td>331900</td>
      <td>333600</td>
      <td>117700.0</td>
      <td>54.515980</td>
    </tr>
    <tr>
      <td>999</td>
      <td>97202</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>132600.0</td>
      <td>133400.0</td>
      <td>134200.0</td>
      <td>135000.0</td>
      <td>135800.0</td>
      <td>...</td>
      <td>522700</td>
      <td>521700</td>
      <td>522200</td>
      <td>525700</td>
      <td>530100</td>
      <td>534100</td>
      <td>535900</td>
      <td>535000</td>
      <td>186000.0</td>
      <td>53.295129</td>
    </tr>
    <tr>
      <td>5713</td>
      <td>97232</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>174100.0</td>
      <td>175300.0</td>
      <td>176500.0</td>
      <td>177700.0</td>
      <td>178900.0</td>
      <td>...</td>
      <td>678600</td>
      <td>680300</td>
      <td>682300</td>
      <td>686600</td>
      <td>692200</td>
      <td>696500</td>
      <td>695500</td>
      <td>690600</td>
      <td>238200.0</td>
      <td>52.652520</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 272 columns</p>
</div>



As we might have expected, Portland's rapid expansion in the last decade is reflected here. Zipcodes 97217 and 97227 are the top performers, with 63.4\% and 63.7\% ROI, respectively.


```python
# Plot the top 10 zipcodes by growth in the period 2008-2018
xticks = ['2008-04', '2010-04', '2012-04', '2014-04', '2016-04', '2018-04']
xlabels = ['2008', '2010', '2012', '2014', '2016', '2018']
yticks = [200000, 300000, 400000, 500000, 600000, 700000, 800000]
ylabels = ['200k', '300k', '400k', '500k', '600k', '700k', '800k']

plt.figure(figsize=(16,8))
for index in top_10_growth_10.index:
    sample = top_10_growth_10.loc[index,'2008-04':'2018-04']
    zipcode = top_10_growth_10.loc[index]['RegionName']
    plt.plot(sample, label=top_10_growth_10.loc[index]['RegionName'])
plt.xticks(xticks, xlabels)
plt.yticks(yticks, ylabels)
plt.legend(loc='upper left', ncol=2)
plt.title('Top 10 Zipcodes by Growth over Period 2008-2018')
plt.ylabel('Median home price ($)')
plt.xlabel('Year')
sns.despine()
plt.show();
```


![](./images/output_38_0.png)


## Greatest ROI 2013-2018

Finally, I will identify the top performers over the last five years. Any zipcode that has been a top performer at this time interval seems likely to continue performing well in the next five years.


```python
# Calculate growth over the past 5 years and find the top 10 performers
oregon_copy = oregon.copy()
oregon_copy['growth'] = oregon_copy['2018-04'] - oregon_copy['2013-04']
oregon_copy['roi'] = oregon_copy['growth']/oregon_copy['2013-04']*100
top_10_growth_5 = oregon_copy.sort_values('roi', ascending=False)[:10]
top_10_growth_5
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RegionName</th>
      <th>City</th>
      <th>State</th>
      <th>Metro</th>
      <th>CountyName</th>
      <th>1996-04</th>
      <th>1996-05</th>
      <th>1996-06</th>
      <th>1996-07</th>
      <th>1996-08</th>
      <th>...</th>
      <th>2017-09</th>
      <th>2017-10</th>
      <th>2017-11</th>
      <th>2017-12</th>
      <th>2018-01</th>
      <th>2018-02</th>
      <th>2018-03</th>
      <th>2018-04</th>
      <th>growth</th>
      <th>roi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3081</td>
      <td>97266</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>98900.0</td>
      <td>99700.0</td>
      <td>100400.0</td>
      <td>101000.0</td>
      <td>101700.0</td>
      <td>...</td>
      <td>286800</td>
      <td>288000</td>
      <td>290100</td>
      <td>293400</td>
      <td>297200</td>
      <td>300300</td>
      <td>302900</td>
      <td>303900</td>
      <td>152300.0</td>
      <td>100.461741</td>
    </tr>
    <tr>
      <td>6854</td>
      <td>97739</td>
      <td>La Pine</td>
      <td>OR</td>
      <td>Bend</td>
      <td>Deschutes</td>
      <td>71500.0</td>
      <td>72300.0</td>
      <td>73200.0</td>
      <td>74100.0</td>
      <td>74900.0</td>
      <td>...</td>
      <td>217100</td>
      <td>217400</td>
      <td>218200</td>
      <td>220700</td>
      <td>221900</td>
      <td>224300</td>
      <td>230300</td>
      <td>235800</td>
      <td>117700.0</td>
      <td>99.661304</td>
    </tr>
    <tr>
      <td>12763</td>
      <td>97026</td>
      <td>Gervais</td>
      <td>OR</td>
      <td>Salem</td>
      <td>Marion</td>
      <td>85100.0</td>
      <td>85300.0</td>
      <td>85600.0</td>
      <td>85800.0</td>
      <td>86200.0</td>
      <td>...</td>
      <td>214500</td>
      <td>217800</td>
      <td>219800</td>
      <td>221500</td>
      <td>225300</td>
      <td>228900</td>
      <td>233000</td>
      <td>236500</td>
      <td>114800.0</td>
      <td>94.330320</td>
    </tr>
    <tr>
      <td>6568</td>
      <td>97216</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>103700.0</td>
      <td>104500.0</td>
      <td>105300.0</td>
      <td>106100.0</td>
      <td>107000.0</td>
      <td>...</td>
      <td>319500</td>
      <td>319900</td>
      <td>320900</td>
      <td>323200</td>
      <td>325900</td>
      <td>328900</td>
      <td>331900</td>
      <td>333600</td>
      <td>160200.0</td>
      <td>92.387543</td>
    </tr>
    <tr>
      <td>4401</td>
      <td>97071</td>
      <td>Woodburn</td>
      <td>OR</td>
      <td>Salem</td>
      <td>Marion</td>
      <td>97700.0</td>
      <td>97600.0</td>
      <td>97600.0</td>
      <td>97700.0</td>
      <td>98000.0</td>
      <td>...</td>
      <td>241100</td>
      <td>244900</td>
      <td>248700</td>
      <td>251100</td>
      <td>252600</td>
      <td>253900</td>
      <td>256000</td>
      <td>257600</td>
      <td>123500.0</td>
      <td>92.095451</td>
    </tr>
    <tr>
      <td>8953</td>
      <td>97760</td>
      <td>Terrebonne</td>
      <td>OR</td>
      <td>Roseburg</td>
      <td>Jefferson</td>
      <td>93400.0</td>
      <td>93400.0</td>
      <td>93400.0</td>
      <td>93400.0</td>
      <td>93300.0</td>
      <td>...</td>
      <td>337200</td>
      <td>341000</td>
      <td>343300</td>
      <td>344800</td>
      <td>346100</td>
      <td>346700</td>
      <td>348700</td>
      <td>351900</td>
      <td>166700.0</td>
      <td>90.010799</td>
    </tr>
    <tr>
      <td>2419</td>
      <td>97233</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>110800.0</td>
      <td>111500.0</td>
      <td>112300.0</td>
      <td>113000.0</td>
      <td>113700.0</td>
      <td>...</td>
      <td>276000</td>
      <td>277100</td>
      <td>278600</td>
      <td>280900</td>
      <td>283600</td>
      <td>286300</td>
      <td>288400</td>
      <td>288800</td>
      <td>136100.0</td>
      <td>89.129011</td>
    </tr>
    <tr>
      <td>3471</td>
      <td>97220</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>103000.0</td>
      <td>104000.0</td>
      <td>105000.0</td>
      <td>105900.0</td>
      <td>106800.0</td>
      <td>...</td>
      <td>324900</td>
      <td>325300</td>
      <td>325800</td>
      <td>328000</td>
      <td>331100</td>
      <td>333600</td>
      <td>335100</td>
      <td>335300</td>
      <td>157400.0</td>
      <td>88.476672</td>
    </tr>
    <tr>
      <td>3130</td>
      <td>97203</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>88000.0</td>
      <td>88700.0</td>
      <td>89500.0</td>
      <td>90300.0</td>
      <td>91200.0</td>
      <td>...</td>
      <td>373600</td>
      <td>373400</td>
      <td>373500</td>
      <td>375000</td>
      <td>377200</td>
      <td>379300</td>
      <td>379800</td>
      <td>378600</td>
      <td>177600.0</td>
      <td>88.358209</td>
    </tr>
    <tr>
      <td>644</td>
      <td>97206</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>96600.0</td>
      <td>97600.0</td>
      <td>98600.0</td>
      <td>99600.0</td>
      <td>100700.0</td>
      <td>...</td>
      <td>371900</td>
      <td>372000</td>
      <td>373000</td>
      <td>374900</td>
      <td>377100</td>
      <td>379600</td>
      <td>381800</td>
      <td>382200</td>
      <td>174900.0</td>
      <td>84.370478</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 272 columns</p>
</div>



In the last five years, 97266 in Portland has seen a 100\% ROI, with 97739 in Bend coming in a close second with 99.6\% ROI. Portland's growth over this period may be expected, and Bend has become a hotspot for tourism and craft brewing in recent years.

Let's take a look at their performance over the five-year period.


```python
# Plot the top 10 zipcodes for growth over the last 5 years
xticks = ['2013-04', '2014-04', '2015-04', '2016-04', '2017-04', '2018-04']
xlabels = ['2013', '2014', '2015', '2016', '2017', '2018']
yticks = [150000, 200000, 250000, 300000, 350000] 
ylabels = ['150k', '200k', '250k', '300k', '350k']

plt.figure(figsize=(16,8))
for index in top_10_growth_5.index:
    sample = top_10_growth_5.loc[index,'2013-04':'2018-04']
    zipcode = top_10_growth_5.loc[index]['RegionName']
    plt.plot(sample, label=top_10_growth_5.loc[index]['RegionName'])
plt.xticks(xticks, xlabels)
plt.yticks(yticks, ylabels)
plt.legend(loc='upper left', ncol=2)
plt.title('Top 10 Zipcodes by Growth over Period 2013-2018')
plt.ylabel('Median home price ($)')
plt.xlabel('Year')
sns.despine()
plt.show();
```


![](./images/output_43_0.png)


## The top 5 zips

Based on the analyses above, I select the following zipcodes as the top 5 for investment in Oregon (in no particular order):

* 97266 (Portland): ranked 1st for ROI over the last 5 years

* 97739 (La Pine/Bend): ranked 2nd for ROI over the last 5 years

* 97217 (Portland): ranked 1st for ROI over the last 10 years

* 97227 (Portland): ranked 2nd for ROI over the last 10 years

* 97203 (Portland): in the top 10 for ROI at 5-, 10-, and 22-year intervals

To strike a balance between strong performers in the short and long terms, I selected the top two zipcodes for ROI at the 5- and 10-year intervals. I also included 97203, the only zipcode to make the top 10 for ROI at all three time intervals. 

Let's examine the time series for the top 5 zipcodes:


```python
top5_zips = ['97266', '97739', '97217', '97227', '97203']

top_data = oregon[oregon.RegionName.isin(top5_zips)]
top_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RegionName</th>
      <th>City</th>
      <th>State</th>
      <th>Metro</th>
      <th>CountyName</th>
      <th>1996-04</th>
      <th>1996-05</th>
      <th>1996-06</th>
      <th>1996-07</th>
      <th>1996-08</th>
      <th>...</th>
      <th>2017-07</th>
      <th>2017-08</th>
      <th>2017-09</th>
      <th>2017-10</th>
      <th>2017-11</th>
      <th>2017-12</th>
      <th>2018-01</th>
      <th>2018-02</th>
      <th>2018-03</th>
      <th>2018-04</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1769</td>
      <td>97217</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>90600.0</td>
      <td>91500.0</td>
      <td>92300.0</td>
      <td>93200.0</td>
      <td>94000.0</td>
      <td>...</td>
      <td>433700</td>
      <td>434100</td>
      <td>434000</td>
      <td>433100</td>
      <td>433600</td>
      <td>436100</td>
      <td>439100</td>
      <td>442400</td>
      <td>445100</td>
      <td>445000</td>
    </tr>
    <tr>
      <td>3081</td>
      <td>97266</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>98900.0</td>
      <td>99700.0</td>
      <td>100400.0</td>
      <td>101000.0</td>
      <td>101700.0</td>
      <td>...</td>
      <td>283100</td>
      <td>285100</td>
      <td>286800</td>
      <td>288000</td>
      <td>290100</td>
      <td>293400</td>
      <td>297200</td>
      <td>300300</td>
      <td>302900</td>
      <td>303900</td>
    </tr>
    <tr>
      <td>3130</td>
      <td>97203</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>88000.0</td>
      <td>88700.0</td>
      <td>89500.0</td>
      <td>90300.0</td>
      <td>91200.0</td>
      <td>...</td>
      <td>372300</td>
      <td>373200</td>
      <td>373600</td>
      <td>373400</td>
      <td>373500</td>
      <td>375000</td>
      <td>377200</td>
      <td>379300</td>
      <td>379800</td>
      <td>378600</td>
    </tr>
    <tr>
      <td>6854</td>
      <td>97739</td>
      <td>La Pine</td>
      <td>OR</td>
      <td>Bend</td>
      <td>Deschutes</td>
      <td>71500.0</td>
      <td>72300.0</td>
      <td>73200.0</td>
      <td>74100.0</td>
      <td>74900.0</td>
      <td>...</td>
      <td>210300</td>
      <td>214600</td>
      <td>217100</td>
      <td>217400</td>
      <td>218200</td>
      <td>220700</td>
      <td>221900</td>
      <td>224300</td>
      <td>230300</td>
      <td>235800</td>
    </tr>
    <tr>
      <td>10068</td>
      <td>97227</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>84500.0</td>
      <td>85100.0</td>
      <td>85800.0</td>
      <td>86400.0</td>
      <td>87000.0</td>
      <td>...</td>
      <td>525300</td>
      <td>524900</td>
      <td>527300</td>
      <td>531100</td>
      <td>534600</td>
      <td>537100</td>
      <td>540300</td>
      <td>543600</td>
      <td>543300</td>
      <td>540300</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 270 columns</p>
</div>




```python
# Plot the top 5 zipcodes
xticks = ['1996-04', '1998-04', '2000-04', '2002-04', '2004-04', '2006-04', 
          '2008-04', '2010-04', '2012-04', '2014-04', '2016-04', '2018-04']
xlabels = ['1996', '1998', '2000', '2002', '2004', '2006', '2008', '2010',
           '2012', '2014', '2016', '2018']
yticks = [100000, 200000, 300000, 400000, 500000, 600000]
ylabels = ['100k', '200k', '300k', '400k', '500k', '600k']

plt.figure(figsize=(16,10))
for index in top_data.index:
    sample = top_data.loc[index,'1996-04':'2018-04']
    zipcode = top_data.loc[index]['RegionName']
    plt.plot(sample, label=top_data.loc[index]['RegionName'])
plt.xticks(xticks, xlabels)
plt.yticks(yticks, ylabels)
plt.legend()
plt.title('Top 5 Zipcodes for Real Estate Investment in Oregon')
plt.ylabel('Median home price ($)')
plt.xlabel('Year')
sns.despine()
plt.show();
```


![](./images/output_47_0.png)


I like the fact that, in addition to offering good ROI, these five zipcodes also represent a range of price points at present. For the cost of one median-priced home in Portland 97227, one could buy two in La Pine 97739!

## Least loss during recession (Jan. 2007 v. Jun. 2011)

Although the top 5 zipcodes showed strong ROI over various time intervals, it's important to consider how much value they lost during the recession. This may be an indicator of how they would fare in a future downturn of the housing market.

To calculate loss due to the recession, I will compare median home prices in January 2007, before the recession began, to prices in June 2011, which is at or near the recession low-point for each of these zipcodes. I will use the difference in these values (expressed as a percent of the pre-crash value) as a rough proxy for the risk of investing in each zipcode. To put each zipcode in a statewide perspective, I also assign a rank based on how much median home price changed. The zipcode whose median home price declined the least (or even grew) during the recession would rank at number 1, while the zipcode hit hardest by the recession would rank 224th.


```python
# Calculate loss in median home value during recession period
# Rank zipcodes by amount lost (rank 1 = least loss)
oregon_copy = oregon.copy()
oregon_copy['rec_loss'] = np.abs(oregon_copy['2011-06'] - oregon_copy['2007-01'])
oregon_copy['rec_loss_pct'] = oregon_copy['rec_loss']/oregon_copy['2007-01']
least_rec_loss = oregon_copy.sort_values('rec_loss_pct', ascending=False)
least_rec_loss['rank'] = least_rec_loss['rec_loss_pct'].rank()
least_rec_loss.tail(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RegionName</th>
      <th>City</th>
      <th>State</th>
      <th>Metro</th>
      <th>CountyName</th>
      <th>1996-04</th>
      <th>1996-05</th>
      <th>1996-06</th>
      <th>1996-07</th>
      <th>1996-08</th>
      <th>...</th>
      <th>2017-10</th>
      <th>2017-11</th>
      <th>2017-12</th>
      <th>2018-01</th>
      <th>2018-02</th>
      <th>2018-03</th>
      <th>2018-04</th>
      <th>rec_loss</th>
      <th>rec_loss_pct</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>10317</td>
      <td>97352</td>
      <td>Jefferson</td>
      <td>OR</td>
      <td>Salem</td>
      <td>Marion</td>
      <td>84300.0</td>
      <td>84600.0</td>
      <td>84900.0</td>
      <td>85300.0</td>
      <td>85800.0</td>
      <td>...</td>
      <td>213300</td>
      <td>215800</td>
      <td>218300</td>
      <td>221000</td>
      <td>223500</td>
      <td>226500</td>
      <td>228600</td>
      <td>15200.0</td>
      <td>0.097561</td>
      <td>10.0</td>
    </tr>
    <tr>
      <td>4423</td>
      <td>97333</td>
      <td>Corvallis</td>
      <td>OR</td>
      <td>Corvallis</td>
      <td>Benton</td>
      <td>67300.0</td>
      <td>67500.0</td>
      <td>67600.0</td>
      <td>67600.0</td>
      <td>67600.0</td>
      <td>...</td>
      <td>326400</td>
      <td>329400</td>
      <td>332300</td>
      <td>334400</td>
      <td>333200</td>
      <td>329600</td>
      <td>326700</td>
      <td>22600.0</td>
      <td>0.090509</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>5779</td>
      <td>97850</td>
      <td>La Grande</td>
      <td>OR</td>
      <td>La Grande</td>
      <td>Union</td>
      <td>89800.0</td>
      <td>90400.0</td>
      <td>91000.0</td>
      <td>91600.0</td>
      <td>92200.0</td>
      <td>...</td>
      <td>169800</td>
      <td>170800</td>
      <td>171900</td>
      <td>172600</td>
      <td>173100</td>
      <td>173700</td>
      <td>174400</td>
      <td>9200.0</td>
      <td>0.067251</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>4270</td>
      <td>97838</td>
      <td>Hermiston</td>
      <td>OR</td>
      <td>Hermiston-Pendleton</td>
      <td>Umatilla</td>
      <td>97700.0</td>
      <td>97600.0</td>
      <td>97600.0</td>
      <td>97700.0</td>
      <td>98000.0</td>
      <td>...</td>
      <td>179900</td>
      <td>179600</td>
      <td>179800</td>
      <td>180900</td>
      <td>182200</td>
      <td>184200</td>
      <td>185800</td>
      <td>6100.0</td>
      <td>0.045085</td>
      <td>7.0</td>
    </tr>
    <tr>
      <td>12987</td>
      <td>97883</td>
      <td>Union</td>
      <td>OR</td>
      <td>La Grande</td>
      <td>Union</td>
      <td>78400.0</td>
      <td>78900.0</td>
      <td>79400.0</td>
      <td>79800.0</td>
      <td>80200.0</td>
      <td>...</td>
      <td>145200</td>
      <td>147000</td>
      <td>148800</td>
      <td>149900</td>
      <td>150500</td>
      <td>151900</td>
      <td>153400</td>
      <td>3800.0</td>
      <td>0.035023</td>
      <td>6.0</td>
    </tr>
    <tr>
      <td>5082</td>
      <td>97801</td>
      <td>Pendleton</td>
      <td>OR</td>
      <td>Hermiston-Pendleton</td>
      <td>Umatilla</td>
      <td>252000.0</td>
      <td>253000.0</td>
      <td>254100.0</td>
      <td>255200.0</td>
      <td>256600.0</td>
      <td>...</td>
      <td>168700</td>
      <td>168700</td>
      <td>168900</td>
      <td>169900</td>
      <td>171100</td>
      <td>172900</td>
      <td>174400</td>
      <td>4300.0</td>
      <td>0.032502</td>
      <td>5.0</td>
    </tr>
    <tr>
      <td>14523</td>
      <td>97886</td>
      <td>Weston</td>
      <td>OR</td>
      <td>Hermiston-Pendleton</td>
      <td>Umatilla</td>
      <td>107000.0</td>
      <td>107600.0</td>
      <td>108300.0</td>
      <td>108900.0</td>
      <td>109600.0</td>
      <td>...</td>
      <td>138700</td>
      <td>138700</td>
      <td>139600</td>
      <td>140500</td>
      <td>141200</td>
      <td>142700</td>
      <td>143800</td>
      <td>2400.0</td>
      <td>0.021563</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>8743</td>
      <td>97370</td>
      <td>Philomath</td>
      <td>OR</td>
      <td>Corvallis</td>
      <td>Benton</td>
      <td>69300.0</td>
      <td>69700.0</td>
      <td>70000.0</td>
      <td>70300.0</td>
      <td>70600.0</td>
      <td>...</td>
      <td>284400</td>
      <td>287100</td>
      <td>290100</td>
      <td>292900</td>
      <td>293000</td>
      <td>291800</td>
      <td>291500</td>
      <td>3300.0</td>
      <td>0.015981</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>13765</td>
      <td>97868</td>
      <td>Pilot Rock</td>
      <td>OR</td>
      <td>Hermiston-Pendleton</td>
      <td>Umatilla</td>
      <td>175300.0</td>
      <td>176400.0</td>
      <td>177500.0</td>
      <td>178500.0</td>
      <td>179500.0</td>
      <td>...</td>
      <td>124400</td>
      <td>124900</td>
      <td>125500</td>
      <td>126300</td>
      <td>128000</td>
      <td>130800</td>
      <td>132600</td>
      <td>1200.0</td>
      <td>0.011952</td>
      <td>2.0</td>
    </tr>
    <tr>
      <td>10113</td>
      <td>97882</td>
      <td>Umatilla</td>
      <td>OR</td>
      <td>Hermiston-Pendleton</td>
      <td>Umatilla</td>
      <td>84300.0</td>
      <td>84600.0</td>
      <td>84900.0</td>
      <td>85300.0</td>
      <td>85800.0</td>
      <td>...</td>
      <td>133200</td>
      <td>133100</td>
      <td>133500</td>
      <td>134400</td>
      <td>135600</td>
      <td>137400</td>
      <td>138800</td>
      <td>900.0</td>
      <td>0.009000</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 273 columns</p>
</div>




```python
# View the losses and ranks of the top 5 zipcodes
top_zips_least_loss = least_rec_loss[least_rec_loss.RegionName.isin(top5_zips)]
top_zips_least_loss[['RegionName','rec_loss', 'rank']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RegionName</th>
      <th>rec_loss</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>6854</td>
      <td>97739</td>
      <td>84000.0</td>
      <td>216.0</td>
    </tr>
    <tr>
      <td>3081</td>
      <td>97266</td>
      <td>67900.0</td>
      <td>190.0</td>
    </tr>
    <tr>
      <td>3130</td>
      <td>97203</td>
      <td>42000.0</td>
      <td>48.0</td>
    </tr>
    <tr>
      <td>1769</td>
      <td>97217</td>
      <td>36600.0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <td>10068</td>
      <td>97227</td>
      <td>40400.0</td>
      <td>17.0</td>
    </tr>
  </tbody>
</table>
</div>



Of the top 5 zipcodes, 97227 suffered the least during the recession, with a drop in median home price of only \$40,400 and an overall rank of 17th in Oregon. 97217 and 97203 also fared relatively well, but 97266 and 97739 were among the hardest hit.

Let's inspect the recession and recovery period for the top 5 zipcodes:


```python
# Plot the top 5 zipcodes over the period Jan. 2007-Jun. 2011
xticks = ['2007-01', '2008-01', '2009-01', '2010-01', '2011-01', '2012-01', 
          '2013-01']
xlabels = ['2007', '2008', '2009', '2010', '2011', '2012', '2013']
yticks = [100000, 150000, 200000, 250000, 300000]
ylabels = ['100k', '150k', '200k', '250k', '300k']

plt.figure(figsize=(16,8))
for n,index in enumerate(top_zips_least_loss.index):
    sample = top_zips_least_loss.loc[index,'2007-01':'2013-01']
    zipcode = top_zips_least_loss.loc[index]['RegionName']
    plt.plot(sample, label=top_zips_least_loss.loc[index]['RegionName'])
plt.xticks(xticks, xlabels)
plt.yticks(yticks, ylabels)
plt.legend(loc='lower left', ncol=2)
plt.title('Top zipcodes during the recession')
plt.ylabel('Median home price ($)')
plt.xlabel('Year')
sns.despine()
plt.show();
```


![](./images/output_54_0.png)


If the decline in median home price during the recession can be used as a proxy for future risk, then 97227 is the least risky of these zipcodes, and 97739 the riskiest. By 2013, when 97227 was making a strong recovery, 97739 was barely turning up again. Keep in mind that 97739 also saw the most growth in median home price over the period 2013-2018, so it eventually made a strong recovery, but 97227 has been a more dependable performer over time.

## Properties currently available

One last feature I want to consider for the top 5 zipcodes is the number of properties currently listed for sale on Zillow. The number of current listings represents how many opportunities there are to invest, and it could give investors an indication of how much choice or competition they might face in that market. 

Zillow collects data on numbers of listings monthly for some zipcodes, but not all. I found monthly listing counts for just 97227, 97266, and 97739, which I plotted below. I collected current listing counts for the other zipcodes manually by searching Zillow for current listings. 


```python
# Load listings data
listings = pd.read_csv('zillow_listings_counts.csv')
listings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SizeRank</th>
      <th>RegionID</th>
      <th>RegionName</th>
      <th>RegionType</th>
      <th>StateName</th>
      <th>2013-01</th>
      <th>2013-02</th>
      <th>2013-03</th>
      <th>2013-04</th>
      <th>2013-05</th>
      <th>...</th>
      <th>2018-12</th>
      <th>2019-01</th>
      <th>2019-02</th>
      <th>2019-03</th>
      <th>2019-04</th>
      <th>2019-05</th>
      <th>2019-06</th>
      <th>2019-07</th>
      <th>2019-08</th>
      <th>2019-09</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.0</td>
      <td>61639</td>
      <td>10025</td>
      <td>Zip</td>
      <td>NY</td>
      <td>338.0</td>
      <td>332.0</td>
      <td>325.0</td>
      <td>316.0</td>
      <td>308.0</td>
      <td>...</td>
      <td>259.0</td>
      <td>258.0</td>
      <td>257.0</td>
      <td>269.0</td>
      <td>286.0</td>
      <td>293.0</td>
      <td>286.0</td>
      <td>278.0</td>
      <td>282.0</td>
      <td>287</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2.0</td>
      <td>84654</td>
      <td>60657</td>
      <td>Zip</td>
      <td>IL</td>
      <td>510.0</td>
      <td>503.0</td>
      <td>484.0</td>
      <td>483.0</td>
      <td>483.0</td>
      <td>...</td>
      <td>491.0</td>
      <td>510.0</td>
      <td>504.0</td>
      <td>495.0</td>
      <td>518.0</td>
      <td>545.0</td>
      <td>564.0</td>
      <td>580.0</td>
      <td>579.0</td>
      <td>574</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3.0</td>
      <td>61637</td>
      <td>10023</td>
      <td>Zip</td>
      <td>NY</td>
      <td>449.0</td>
      <td>435.0</td>
      <td>436.0</td>
      <td>436.0</td>
      <td>437.0</td>
      <td>...</td>
      <td>523.0</td>
      <td>508.0</td>
      <td>489.0</td>
      <td>483.0</td>
      <td>492.0</td>
      <td>511.0</td>
      <td>521.0</td>
      <td>520.0</td>
      <td>525.0</td>
      <td>537</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4.0</td>
      <td>91982</td>
      <td>77494</td>
      <td>Zip</td>
      <td>TX</td>
      <td>948.0</td>
      <td>932.0</td>
      <td>889.0</td>
      <td>860.0</td>
      <td>835.0</td>
      <td>...</td>
      <td>941.0</td>
      <td>962.0</td>
      <td>947.0</td>
      <td>934.0</td>
      <td>958.0</td>
      <td>972.0</td>
      <td>962.0</td>
      <td>948.0</td>
      <td>908.0</td>
      <td>875</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5.0</td>
      <td>84616</td>
      <td>60614</td>
      <td>Zip</td>
      <td>IL</td>
      <td>584.0</td>
      <td>566.0</td>
      <td>525.0</td>
      <td>500.0</td>
      <td>499.0</td>
      <td>...</td>
      <td>629.0</td>
      <td>644.0</td>
      <td>665.0</td>
      <td>699.0</td>
      <td>739.0</td>
      <td>743.0</td>
      <td>724.0</td>
      <td>714.0</td>
      <td>698.0</td>
      <td>690</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 86 columns</p>
</div>




```python
# Plot listings history for 97210 and 97034, current listings for other zips
top_listings = listings[listings.RegionName.isin(top5_zips)]
xticks = ['2013-01', '2014-01', '2015-01', '2016-01', '2017-01', '2018-01', 
          '2019-01']
xlabels = ['2013', '2014', '2015', '2016', '2017', '2018', '2019']


plt.figure(figsize=(16,10))
for n,index in enumerate(top_listings.index):
    sample = top_listings.loc[index,'2013-01':'2019-09']
    zipcode = top_listings.loc[index]['RegionName']
    plt.plot(sample, label=zipcode)
plt.scatter('2019-10', 100, marker='.', label='97203', color='fuchsia')
plt.scatter('2019-10', 11, marker='.', label='97227', color='brown')
plt.xticks(xticks, xlabels)
plt.legend(loc='lower left')
plt.title('Number of properties for sale in top zipcodes')
plt.ylabel('Number of properties')
plt.xlabel('Year')
sns.despine()
plt.show();
```


![](./images/output_59_0.png)


The plot above illustrates the number of properties available in each zipcode (either over time, if that data was available, or just currently). 

Note that 97217 is a much bigger market, with about 160 properties currently listed for sale, while 97227 has only 11 properties listed this month.

## Summary (non-technical) visualizations

The top five zipcodes for investment in Oregon are 97217, 97266, 97739, 97203, and 97227. The ranking of these five depends on the investor's priorities and risk tolerance. 

Remember that 97266 and 97739 were strong performers for ROI in the short term, 97217 and 97227 in the mid-term, and and 97203 over the whole 22-year dataset.

The charts below summarize the current median home price, current number of properties for sale, and recession-era loss for each of the top 5 zipcodes.


```python
# Collate data for summary barplots
top_5 = ['97266', '97739', '97203', '97217', '97227']
avails = [116, 297, 100, 181, 11]
rec_ranks = [190, 216, 48, 22, 17]
top_5_data = oregon[oregon.RegionName.isin(top_5)]
top_5_data['rec_loss'] = top_5_data['2011-06'] - top_5_data['2007-01']
top_5_data['rec_loss_pct'] = top_5_data['rec_loss']/top_5_data['2007-01']*100
top_5_data['rec_rank'] = [x for x in rec_ranks]
top_5_data['avails'] = [x for x in avails]
top_5_data.sort_values(by='RegionName', axis=0, inplace=True)
top_5_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RegionName</th>
      <th>City</th>
      <th>State</th>
      <th>Metro</th>
      <th>CountyName</th>
      <th>1996-04</th>
      <th>1996-05</th>
      <th>1996-06</th>
      <th>1996-07</th>
      <th>1996-08</th>
      <th>...</th>
      <th>2017-11</th>
      <th>2017-12</th>
      <th>2018-01</th>
      <th>2018-02</th>
      <th>2018-03</th>
      <th>2018-04</th>
      <th>rec_loss</th>
      <th>rec_loss_pct</th>
      <th>rec_rank</th>
      <th>avails</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3130</td>
      <td>97203</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>88000.0</td>
      <td>88700.0</td>
      <td>89500.0</td>
      <td>90300.0</td>
      <td>91200.0</td>
      <td>...</td>
      <td>373500</td>
      <td>375000</td>
      <td>377200</td>
      <td>379300</td>
      <td>379800</td>
      <td>378600</td>
      <td>-42000.0</td>
      <td>-18.284719</td>
      <td>48</td>
      <td>100</td>
    </tr>
    <tr>
      <td>1769</td>
      <td>97217</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>90600.0</td>
      <td>91500.0</td>
      <td>92300.0</td>
      <td>93200.0</td>
      <td>94000.0</td>
      <td>...</td>
      <td>433600</td>
      <td>436100</td>
      <td>439100</td>
      <td>442400</td>
      <td>445100</td>
      <td>445000</td>
      <td>-36600.0</td>
      <td>-13.879408</td>
      <td>190</td>
      <td>116</td>
    </tr>
    <tr>
      <td>10068</td>
      <td>97227</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>84500.0</td>
      <td>85100.0</td>
      <td>85800.0</td>
      <td>86400.0</td>
      <td>87000.0</td>
      <td>...</td>
      <td>534600</td>
      <td>537100</td>
      <td>540300</td>
      <td>543600</td>
      <td>543300</td>
      <td>540300</td>
      <td>-40400.0</td>
      <td>-12.708399</td>
      <td>17</td>
      <td>11</td>
    </tr>
    <tr>
      <td>3081</td>
      <td>97266</td>
      <td>Portland</td>
      <td>OR</td>
      <td>Portland</td>
      <td>Multnomah</td>
      <td>98900.0</td>
      <td>99700.0</td>
      <td>100400.0</td>
      <td>101000.0</td>
      <td>101700.0</td>
      <td>...</td>
      <td>290100</td>
      <td>293400</td>
      <td>297200</td>
      <td>300300</td>
      <td>302900</td>
      <td>303900</td>
      <td>-67900.0</td>
      <td>-32.612872</td>
      <td>216</td>
      <td>297</td>
    </tr>
    <tr>
      <td>6854</td>
      <td>97739</td>
      <td>La Pine</td>
      <td>OR</td>
      <td>Bend</td>
      <td>Deschutes</td>
      <td>71500.0</td>
      <td>72300.0</td>
      <td>73200.0</td>
      <td>74100.0</td>
      <td>74900.0</td>
      <td>...</td>
      <td>218200</td>
      <td>220700</td>
      <td>221900</td>
      <td>224300</td>
      <td>230300</td>
      <td>235800</td>
      <td>-84000.0</td>
      <td>-43.455768</td>
      <td>22</td>
      <td>181</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 274 columns</p>
</div>




```python
# Plot current number of properties for sale
xticks = ['Portland 97203', 'Portland 97217', 'Portland 97227',
          'Portland 97266', 'La Pine 97739']
plt.figure(figsize=(12,6))
plt.bar(xticks, top_5_data.avails, color='steelblue')
plt.xticks(xticks)
plt.ylabel('Properties')
plt.title('Number of properties for sale on October 30, 2019')
sns.despine()
plt.show();
```


![](./images/output_64_0.png)



```python
# Plot recession loss percents
plt.figure(figsize=(12,6))
plt.bar(xticks, top_5_data.rec_loss_pct, color='indianred')
plt.xticks(xticks)
plt.ylabel('Percent difference')
plt.title('Difference in median home price, June 2011 v. Jan. 2007', y=1.05)
sns.despine()
plt.show();
```


![](./images/output_65_0.png)



```python
# Plot most recent median home prices
yticks = [100000, 200000, 300000, 400000, 500000]
ylabels = ['100k', '200k', '300k', '400k', '500k']

plt.figure(figsize=(12,6))
plt.bar(xticks, top_5_data['2018-04'], color='steelblue')
plt.xticks(xticks)
plt.yticks(yticks, ylabels)
plt.ylabel('Median home price ($)')
plt.title('Most recent median price (April 2018)', y=1.05)
sns.despine()
plt.show();
```


![](./images/output_66_0.png)


# Extract time series for top 5 zipcodes

Now that I have identified the top 5 zipcodes, I will create a subset of the data for each.


```python
# Drop unwanted columns
or_trimmed = oregon.drop(['City','Metro','State','CountyName'], axis=1)

# Create zipcode subsets
or_97203 = or_trimmed[or_trimmed.RegionName==97203].drop('RegionName', axis=1)
or_97217 = or_trimmed[or_trimmed.RegionName==97217].drop('RegionName', axis=1)
or_97227 = or_trimmed[or_trimmed.RegionName==97227].drop('RegionName', axis=1)
or_97266 = or_trimmed[or_trimmed.RegionName==97266].drop('RegionName', axis=1)
or_97739 = or_trimmed[or_trimmed.RegionName==97739].drop('RegionName', axis=1)

# Preview one zipcode's subset
or_97203.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1996-04</th>
      <th>1996-05</th>
      <th>1996-06</th>
      <th>1996-07</th>
      <th>1996-08</th>
      <th>1996-09</th>
      <th>1996-10</th>
      <th>1996-11</th>
      <th>1996-12</th>
      <th>1997-01</th>
      <th>...</th>
      <th>2017-07</th>
      <th>2017-08</th>
      <th>2017-09</th>
      <th>2017-10</th>
      <th>2017-11</th>
      <th>2017-12</th>
      <th>2018-01</th>
      <th>2018-02</th>
      <th>2018-03</th>
      <th>2018-04</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3130</td>
      <td>88000.0</td>
      <td>88700.0</td>
      <td>89500.0</td>
      <td>90300.0</td>
      <td>91200.0</td>
      <td>92000.0</td>
      <td>93000.0</td>
      <td>94000.0</td>
      <td>95100.0</td>
      <td>96200.0</td>
      <td>...</td>
      <td>372300</td>
      <td>373200</td>
      <td>373600</td>
      <td>373400</td>
      <td>373500</td>
      <td>375000</td>
      <td>377200</td>
      <td>379300</td>
      <td>379800</td>
      <td>378600</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 265 columns</p>
</div>



Although I now have the data I want, note that it is not in a useful format for modeling. I will reshape and visualize each zipcode's time series below.

# EDA and visualization

In this section I use a custom function (defined above) to transform each time series into an appropriate format for modeling (specifically, the required format for use with Facebook Prophet). My custom function also returns a small plot of the series, just to give us an idea of its shape.

## 97203


```python
# Transform and plot data from 97203
or_97203, fig = melt_plot(or_97203, '97203')

# Preview the data and view the plot
display(or_97203.head())
fig;
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1996-04-01</td>
      <td>88000.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1996-05-01</td>
      <td>88700.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1996-06-01</td>
      <td>89500.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1996-07-01</td>
      <td>90300.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1996-08-01</td>
      <td>91200.0</td>
    </tr>
  </tbody>
</table>
</div>



![](./images/output_74_1.png)


It looks like prices in this zipcode experienced a big growth spurt leading up to the crisis in 2008, followed by a drop that reached its minimum sometime around 2012. It seems to have taken until around 2014 to reach pre-crash levels.

## 97217


```python
# Transform and plot data from 97217
or_97217, fig = melt_plot(or_97217, '97217')
fig;
```


![](./images/output_77_0.png)


In this zipcode, the pre-crash growth spurt was not as dramatic, and the subsequent decline was also not as bad. The median home price seems to have reached its pre-crash level around 2013, a bit quicker than in 97203.

## 97227


```python
# Transform and plot data from 97227
or_97227, fig = melt_plot(or_97227, '97227')
fig;
```


![](./images/output_80_0.png)


Of the 5 zipcodes, this one was the least impacted by the crash. 

## 97266


```python
# Transform and plot data from 97266
or_97266, fig = melt_plot(or_97266, '97266')
fig;
```


![](./images/output_83_0.png)


Pre-crash growth and post-crash decline both seem pretty dramatic here. Pre-crash level was reached again around 2015.

## 97739


```python
# Transform and plot data from 97739
or_97739, fig = melt_plot(or_97739, '97739')
fig;
```


![](./images/output_86_0.png)


La Pine has only been a top performer for ROI in the last 5 years, and it was hit hardest by the recession. It looks like it took until 2016 to reach the pre-crash median price again.

## All together

Finally, let's compare the top 5 zipcodes to the statewide average over time.


```python
# Prep data for OR mean
or_all = or_trimmed.drop('RegionName', axis=1)
or_all_melt = melt_it(or_all)
or_all_melt = or_all_melt.groupby('ds').mean()
or_all_melt.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
    </tr>
    <tr>
      <th>ds</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1996-04-01</td>
      <td>124696.428571</td>
    </tr>
    <tr>
      <td>1996-05-01</td>
      <td>125219.642857</td>
    </tr>
    <tr>
      <td>1996-06-01</td>
      <td>125730.803571</td>
    </tr>
    <tr>
      <td>1996-07-01</td>
      <td>126237.946429</td>
    </tr>
    <tr>
      <td>1996-08-01</td>
      <td>126762.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot top 5 zipcodes and OR mean
yticks = [100000, 200000, 300000, 400000, 500000, 600000]
ylabels = ['100k', '200k', '300k', '400k', '500k', '600k']

plt.figure(figsize=(16,8))
plt.plot(or_97203['ds'], or_97203['y'], label='97203', lw=4)
plt.plot(or_97217['ds'], or_97217['y'], label='97217', lw=4)
plt.plot(or_97227['ds'], or_97227['y'], label='97227', lw=4)
plt.plot(or_97266['ds'], or_97266['y'], label='97266', lw=4)
plt.plot(or_97739['ds'], or_97739['y'], label='97739', lw=4)
plt.plot(or_all_melt.index, or_all_melt['y'], label='OR Mean',
        linestyle=':', color='k', lw=4)
plt.legend()
plt.title('The top 5 Oregon zipcodes over time', y=1.05)
plt.yticks(yticks,ylabels)
plt.xlabel('Year')
plt.ylabel('Median home price ($)')
sns.despine()
plt.show();
```


![](./images/output_91_0.png)


I'm glad to see that the five zipcodes represent a range of price points, some above and some below the statewide mean.

# Modeling

Now that I have selected the top 5 zipcodes for investment, I want to build models to predict how they might perform in the future.

I have decided to use [Facebook Prophet](https://facebook.github.io/prophet/) rather than the [Statsmodels SARIMAX](https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html) package because Prophet handles the selection of ARIMA parameters internally. Selecting ARIMA parameters myself using a grid search could be very time-consuming; I could only consider a small range of possibilities before the operation became very slow and resource-intensive. Prophet has the advantages of being easy to use, concise, and fast, and it produces attractive plots of the model and predictions. Prophet can also output separate plots of the trend and seasonality in the data on multiple scales (daily, weekly, monthly). I appreciate being able to access a convenient DataFrame of the predictions with the upper and lower bounds of their confidence intervals.

My method is as follows:

1. Split each zipcode's data into "training" and "validation" sets, reserving the last 10% of each time series for validation.

2. View a quick plot of the data to confirm that the split worked and to get a sense of whether the validation series differs dramatically from the training series.

3. Fit a model to the training series.

4. Predict values for the period covered by the validation series.

5. Compare the predictions to the actual values of the validation series. 

6. Fit a new model to the entire series (training and validation combined).

7. Using the second model, forecast median home prices over the next 60 months.

Steps 1 and 2 are wrapped in a custom function that returns the training and validation series and prints the plot. Steps 3 through 7 use Prophet and are wrapped in a custom function that prints plots of the various models and returns two DataFrames: one containing the predicted values to be compared to the validation series, and the other containing the 60-month forecast values.


```python
# Reset context to 'paper' since the following plots are for analysis only
sns.set_context('paper')
```

## 97203

### Train-test split

Because all the zipcodes took a hard dip during the recession, I want to make sure that my training set incorporates some of the upswing as the housing market exited the recession. For this reason, I will reserve the last 10% (2 years and 4 months) of the data for testing.


```python
# Split and plot data for 97203
or_97203_train, or_97203_test = prep_and_plot(or_97203, '97203')
```


![](./images/output_99_0.png)


### Modeling and forecasting


```python
# Get predictions and plot models
ypred_97203, forecast_97203 = proph_it(or_97203_train, or_97203_test, or_97203)
```

    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.



![](./images/output_101_1.png)



![](./images/output_101_2.png)



![](./images/output_101_3.png)



![](./images/output_101_4.png)


To get a sense for how the model performed, we can compare the predicted value to the actual value at the latest date involved in the prediction (April 2018).


```python
# Compare predicted v. actual values for 2018-04
pred_diff_97203 = round((ypred_97203.loc[28,'yhat'] - 
                         or_97203_test.loc[28,'y'])/or_97203_test.loc[28,'y']*100, 2)
pred_diff_97203
```




    -10.68



This calculation tells us that the predicted value was 11% lower than the actual value by the end of the prediction period.

What median home price can we expect for this zipcode five years in the future?


```python
# Print last predicted value and confidence interval
last_pred_97203, ci_97203 = last_forecast(forecast_97203)
```

    Prediction for last date of period:  
    Median home price: $597322.65  
    95% CI: (498692.27, 693441.91)


## 97217

### Train-test split


```python
# Split and plot data for 97217
or_97217_train, or_97217_test = prep_and_plot(or_97217, '97217')
```


![](./images/output_108_0.png)


### Modeling and forecasting


```python
# Get predictions and plot models
ypred_97217, forecast_97217 = proph_it(or_97217_train, or_97217_test, or_97217)
```


![](./images/output_110_1.png)



![](./images/output_110_2.png)



![](./images/output_110_3.png)



![](./images/output_110_4.png)



```python
# Compare predicted v. actual values for 2018-04
pred_diff_97217 = round((ypred_97217.loc[28,'yhat'] - or_97217_test.loc[28,'y'])/or_97217_test.loc[28,'y']*100, 2)
pred_diff_97217
```




    -5.41



For zipcode 97217, the model was off by only 5\% by April 2018.

What median home price does the model predict for April 2023?


```python
# Print last predicted value and confidence interval
last_pred_97217, ci_97217 = last_forecast(forecast_97217)
```

    Prediction for last date of period:  
    Median home price: $684008.46  
    95% CI: (583885.98, 792972.58)


## 97227

### Train-test split


```python
# Split and plot data for 97227
or_97227_train, or_97227_test = prep_and_plot(or_97227, '97227')
```


![](./images/output_116_0.png)


### Modeling and forecasting


```python
# Get predictions and plot models
ypred_97227, forecast_97227 = proph_it(or_97227_train, or_97227_test, or_97227)
```


![](./images/output_118_1.png)



![](./images/output_118_2.png)



![](./images/output_118_3.png)



![](./images/output_118_4.png)



```python
# Compare predicted v. actual values for 2018-04
pred_diff_97227 = round((ypred_97227.loc[28,'yhat'] - or_97227_test.loc[28,'y'])/or_97227_test.loc[28,'y']*100, 2)
pred_diff_97227
```




    -5.28



Here again the model was off by only 5%.

What's the prediction for April 2023?


```python
# Print last predicted value and confidence interval
last_pred_97227, ci_97227 = last_forecast(forecast_97227)
```

    Prediction for last date of period:  
    Median home price: $801983.84  
    95% CI: (682344.08, 922097.81)


## 97266

### Train-test split


```python
# Split and plot data for 97266
or_97266_train, or_97266_test = prep_and_plot(or_97266, '97266')
```


![](./images/output_124_0.png)


### Modeling and forecasting


```python
# Get predictions and plot models
ypred_97266, forecast_97266 = proph_it(or_97266_train, or_97266_test, or_97266)
```

![](./images/output_126_1.png)



![](./images/output_126_2.png)



![](./images/output_126_3.png)



![](./images/output_126_4.png)



```python
# Compare predicted v. actual values for 2018-04
pred_diff_97266 = round((ypred_97266.loc[28,'yhat'] - or_97266_test.loc[28,'y'])/or_97266_test.loc[28,'y']*100, 2)
pred_diff_97266
```




    -15.33



For zipcode 97266, the model did not perform as well; it was off by 15% by April 2018.

What might the median home price be in 2023, assuming the model performs well?


```python
# Print last predicted value and confidence interval
last_pred_97266, ci_97266 = last_forecast(forecast_97266)
```

    Prediction for last date of period:  
    Median home price: $470504.66  
    95% CI: (364630.13, 560598.02)


## 97739

### Train-test split


```python
# Split and plot data for 97739
or_97739_train, or_97739_test = prep_and_plot(or_97739, '97739')
```


![](./images/output_132_0.png)


### Modeling and forecasting


```python
# Get predictions and plot models
ypred_97739, forecast_97739 = proph_it(or_97739_train, or_97739_test, 
                                       or_97739)
```


![](./images/output_134_1.png)



![](./images/output_134_2.png)



![](./images/output_134_3.png)



![](./images/output_134_4.png)



```python
# Compare predicted v. actual values for 2018-04
pred_diff_97739 = round((ypred_97739.loc[28,'yhat'] - or_97739_test.loc[28,'y'])/or_97739_test.loc[28,'y']*100, 2)
pred_diff_97739
```




    -46.36



I'm not at all surprised to see that the model has a hard time predicting values for 97739. This zipcode experienced a boom just before the recession, a big drop during the recession, and then a dramatic recovery in the last five years. This combination of events makes it hard for the model to find a consistent pattern over the last 22 years on which to base its predictions.

Let's take a look at the prediction for April 2023 anyway:


```python
# Print last predicted value and confidence interval
last_pred_97739, ci_97739 = last_forecast(forecast_97739)
```

    Prediction for last date of period:  
    Median home price: $341169.86  
    95% CI: (213369.31, 471691.54)


#### Sidetrack: modeling and predicting for 97739 using only the last 5 years' data

The model for 97739 is performing poorly, and it's no wonder: the zipcode has seen some wild fluctuation in median home price in the last 22 years. I'm going to try the model again, this time using only the last 5 years' worth of data, to see how that changes things. 


```python
# Slice the 97739 data from index 204 onward to get last 5 years' data
or_97739_5 = or_97739[204:]
or_97739_5_train = or_97739[204:259]
or_97739_5_test = or_97739[259:]

or_97739_5_test.reset_index(inplace=True)

# Get predictions and plot models
ypred_97739_5, forecast_97739_5 = proph_it(or_97739_5_train, or_97739_5_test, 
                                           or_97739_5,
                                           forecast_periods1=6, 
                                           forecast_periods2=60)
```


![](./images/output_140_1.png)



![](./images/output_140_2.png)



![](./images/output_140_3.png)



![](./images/output_140_4.png)



```python
# Compare predicted v. actual values for 2018-04
pred_diff_97739_5 = round((ypred_97739_5.loc[5,'yhat'] - or_97739_5_test.loc[5,'y'])/or_97739_5_test.loc[5,'y']*100, 2)
pred_diff_97739_5
```




    -2.66



That's the smallest percent difference between predicted and actual values I've seen so far. This model, however, only had to predict over the last 10% (6 months) of a much smaller series; there was less opportunity for it to stray.

Although this 5-year model _seemed_ to perform much better than the 22-year one, keep in mind that it is only looking at part of the history of 97739. If the model fits just this little part of the data very well (i.e., if it is overfitting in the grand scheme of things), then its predictions will probably only continue to be this good for a short time into the future.

Let's examine its prediction for April 2023:


```python
# Print last predicted value and confidence interval
last_pred_97739_5, ci_97739_5 = last_forecast(forecast_97739_5)
```

    Prediction for last date of period:  
    Median home price: $395151.2  
    95% CI: (232633.12, 556155.9)


The values predicted by the two models differ by about \$54k, and note that the second model has an even wider 95\% confidence interval. 

Ultimately, I wouldn't trust either of these models to predict what might happen in 2023, but I would feel fairly confident in either of them up to 2019 or maybe 2020. Since it's currently October 2019, this doesn't do anybody much good. If I had data up to the present, I would feel confident in predicting prices a year or so into the future, which would be useful for flippers, but not so much for long-term investors.

# Interpretations and recommendations

Now that the modeling is complete, I can offer some interpretations of the results and model-based recommendations for investors.

## Comparing model performance

Recall that I decided to validate each model by using it to predict values for the period from 2015-2018, a period for which I actually had data set aside. To assess how each model performed, I compare the model's prediction for the last date of the validation period&mdash;April 2018&mdash; with the actual value recorded for that date. I then express the difference between the predicted and actual values as a percentage of the actual value. 

If that percentage is small, it means that the model was not very far off from the actual values by the end of the validation period, and I can feel more confident in the model's ability to predict with reasonable accuracy (although with decreasing precision) into the future.

If the percentage is large, it means that the model was not as close to the actual values by the end of the validation period, and presumably it would continue to stray farther as it forecasts into the future. I would have less confidence in a model like this.

Let's take a look:


```python
# Plot percent differences of last predictions and actual values
labels = ['97203', '97217', '97227', '97266', '97739']
values = [pred_diff_97203, pred_diff_97217, pred_diff_97227, pred_diff_97266, 
          pred_diff_97739]

sns.set_context('talk')
plt.figure(figsize=(12,8))
plt.bar(labels, values, color='indianred')
plt.title('Differences between model predictions and actual values for April 2018', 
          y=1.05)
plt.ylabel('Percent difference')
sns.despine()
plt.show();
```


![](./images/output_149_0.png)


To recap: for each zipcode, I trained a model on the first 90% of the time series and then asked the model to predict how median home price would rise in the last 10% (from 2015 to April 2018).

For each of these zipcodes, the models underestimated the actual median home price in April 2018. For 97217 and 97227, the models were only off by about 5%, while for 97739 the model underestimated the price by over 40%. I made a new model for 97739 using just the last 5 years' worth of data, and this model underestimated the price by 2.66% (not depicted in the plot above).

Over the five year period after April 2018, I would place more confidence in the models for 97217 and 97227 than for the others. Neither of the models for 97739 is very reliable, but the 5-year model may be good at predicting home price for the first year or so.

Here is a summary of where the various models predict that median home prices will end up by April 2023:


```python
# Plot predicted median home prices in 2023
labels = ['Portland 97203', 'Portland 97217', 'Portland 97227', 
          'Portland 97266', 'La Pine 97739']
preds = [last_pred_97203, last_pred_97217, last_pred_97227, last_pred_97266, 
         last_pred_97739]
yticks = [200000, 400000, 600000, 800000]
ylabels = ['200k', '400k', '600k', '800k']

plt.figure(figsize=(12,8))
plt.bar(labels, preds, color='lightsteelblue', hatch='/', 
        label='Potential ROI by 2023')
plt.bar(labels, top_5_data['2018-04'], color='steelblue', 
        label='Price in 2018')
plt.yticks(yticks,ylabels)
plt.title('Median home prices, 2018 v. 2023', y=1.05)
plt.xlabel('Zipcode')
plt.ylabel('Median home price ($)')
plt.legend()
sns.despine()
plt.show();
```


![](./images/output_151_0.png)



```python
# Calculate ROI over period April 2018-April 2023
pred_roi_97203 = round((last_pred_97203 - or_97203_test.loc[28,'y'])/or_97203_test.loc[28,'y']*100,2)
pred_roi_97217 = round((last_pred_97217 - or_97217_test.loc[28,'y'])/or_97217_test.loc[28,'y']*100,2)
pred_roi_97227 = round((last_pred_97227 - or_97227_test.loc[28,'y'])/or_97227_test.loc[28,'y']*100,2)
pred_roi_97266 = round((last_pred_97266 - or_97266_test.loc[28,'y'])/or_97266_test.loc[28,'y']*100,2)
pred_roi_97739a = round((last_pred_97739 - or_97739_test.loc[28,'y'])/or_97739_test.loc[28,'y']*100,2)

preds = [pred_roi_97203, pred_roi_97217, pred_roi_97227, pred_roi_97266, 
         pred_roi_97739a]

# Plot predicted ROI
plt.figure(figsize=(12,8))
plt.bar(labels, preds, color='lightsteelblue', hatch='/')
plt.title('Predicted ROI over 5 Years (April 2018-April 2023)', y=1.05)
plt.ylabel('Percent')
plt.xlabel('Zipcode')
sns.despine()
plt.show();
```


![](./images/output_152_0.png)


## Recommendations

The five zipcodes I selected offer a range of choices and opportunities for investors, depending on their particular priorities and tolerance for risk. 

In Portland, 97227 and 97217 are safe bets, although they can be competitive markets for investors. Both models performed fairly well, and they predict ROIs of 48% and 53%, respectively, over the period 2018-2023. These are also the most expensive zipcodes, and they have the least number of properties currently listed, so the opportunity to invest in them is limited. During the recession, however, these two zipcodes fared the best, and this suggests that they may be able to weather future market downturns just as well.

For a riskier undertaking, 97739 in La Pine (Bend) offers a moderate number of low-priced properties with the potential for 45% ROI (using the 22-year model) or 68% ROI (using the 5-year model). This particular market has seen some volatility over the last two decades, and of the five considered here, this zipcode suffered the worst during the recession. If the risk is tolerable, an investor could buy two homes in 97739 for the price of one in 97227, and at the moment, there are plenty of properties in 97739 to choose from.

For those who want to invest in Portland but can't handle the competition (and sticker prices) in 97227 or 97217, 97266 and 97203 are good alternatives. Prices are lower, and there are plenty of properties currently listed for sale. Although I have less confidence in these models than in those for 97227 and 97217, I think a 55+% ROI by 2023 is not a bad estimate. Keep in mind that these zipcodes took bigger hits than 97217 and 97227 during the recession, but they have more than recovered since then.

## Future work

Since the dataset only goes up to April 2018, it would be great to check the model predictions against actual values up to the present. Ideally, one would build the models into a pipeline so that they could be continuously updated as new data comes in. 

I would also like to incorporate additional data about these zipcodes to better understand whether median home price can really keep growing as predicted. For instance, I could use information about wage growth in Portland to determine whether prices in the Portland zipcodes are already close to the maximum the market will allow. 

It would also be helpful to do more research on the history of La Pine over the last 22 years. What happened there to prompt the rapid growth in home prices in the mid-2000s? Is the current rapid growth sustainable? The population of La Pine in 2010 was only 1,653; could my calculations for La Pine have been affected by a small sample size?

# Appendix

## Project description

This repository ([github.com/jrkreiger/or-real-estate](https://github.com/jrkreiger/or-real-estate)) contains files from a project assigned to me as part of Flatiron School's Online Data Science Bootcamp. The purpose of the project is to demonstrate skills in time series modeling.

Here's what you'll find inside:

* **best-bets-or-real-estate.ipynb**: Jupyter Notebook containing project code

* **zillow_data.csv**: CSV file containing main dataset from Zillow

* **zillow_listings_counts.csv**: CSV file containing additional data from Zillow

* **presentation.pdf**: PDF file containing slides from a non-technical presentation of project results

## Technical specifications

I created the Jupyter Notebook in a virtual environment provided to me by Flatiron School. Things may work a little differently for you if you're running different versions of Python or the packages I used.

If you download the notebook to view locally, note that I formatted it using Jupyter Notebook extensions (https://github.com/ipython-contrib/jupyter_contrib_nbextensions). The extensions let me easily create a table of contents and collapse groups of cells to make the notebook easier to read and navigate. If you don't have the extensions installed locally, the notebook will look a little different for you.

In addition to Numpy, Pandas, Matplotlib, and Seaborn, you will also need Facebook Prophet to run the notebook on your local machine.

## Related content

You can read my blog post about the project at this link: http://jrkreiger.net/uncategorized/time-series-modeling-with-facebook-prophet/.

View a video of my non-technical presentation here: https://youtu.be/nd-XS--k5DY.

## Authorship

Flatiron School provided me the main dataset and a starter notebook with two helper functions that I changed to suit my purposes. I wrote the rest of the code in the Jupyter Notebook, the text of this README, my blog post linked above, and the content of my non-technical presentation.
