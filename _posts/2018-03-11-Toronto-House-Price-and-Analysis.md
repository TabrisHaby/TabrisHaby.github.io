---
layout:     post
title:      Toronto House Price Analysis
subtitle:   Visualization and Trend
date:       2018-03-011
author:     Haby
header-img: img/post-bg-2015.jpg
catalog: true
tags:
    - Tableau
    - My Work
    - Visualization
    - Machine Learning
---

> Data From :
> <p>House Price Index (https://housepriceindex.ca/)</p>
> <p>Historical Mortgage Rate (https://www.ratehub.ca/mortgage-rate-history-canada)</p>
> <p>Graphs from : TREB(http://www.trebhome.com/)</p>
> <p>HPI Data For Different House Type (https://www.crea.ca)
> <p>All data are able to use under acamedic and/or personal purpose only.</p>


1.The reason of this report

I am considering buying a condo/house recently since after working for years, I have saved some money. And I know, my money will defintely lose its value one day if I just save it in bank and do nothing. So I started to look at some condo information online, and to my surprise, Toronto HPI has increased dramatically recently, especially in last year(2017). Then I digged some data online and did this visualization report to show what is going on in Toronto House Market. For most of data I digged out couldn't be allowed to published online, so all visualization here is based on open sourse data.


2.What I want to get from this report.

The most important part defintely is the house/condo price trend in next 3-12 months. And also I am interested in the key features for house price, like locations, house size, mortgage rate, insurance and so on. Next thing I am interested in is the relationship of 'listing price', 'selling price' and 'listing days'. The relationship of this 3 variables can tell you how much money can you save when buying a house/condo.


3.Working Environment

Languages : Python 3.6.1<br>
GUI : Spyder in Anaconda 3.2.8<br>
OS : Ubuntu 16.04.4 LTS<br>
Visualization : Tableau Desktop 10.5<br>

4.Report

HPI : A house price index (HPI) measures the price changes of residential housing. In Canada, the New Housing Price Index is calculated monthly by Statistics Canada. Additionally, a resale house price index is also maintained by the Canadian Real Estate Association, based on reported sale prices submitted by real estate agents, and averaged by region. In December 2008, the private National Bank and the information technology firm Teranet began a separate monthly house price index based on resale prices of individual single-family houses in selected metropolitan areas, using a methodology similar to the Case-Shiller index and based on actual sale prices taken from government land registry databases. This allows Teranet and the National Bank to track prices without allowing periods of high sales in one city to push up the national average. The National Bank also operates a forward market on Canadian housing prices.(https://en.wikipedia.org/wiki/House_price_index)

Graph1 : 3 Main Cities Yearly HPI Comparison (Toronto, Quebec City, Ottawa)

  ![png](/img/thp1.png)

The trend of HPI for all cities seem to increase as time goes by, however, Toronto has a dramatically increase in 2017, which is more than 20% than last year, while other cities was in 2009-2011. Quebec City and Ottawa increased slowly in last 10 years compared to Toronto. Another Interest story we can find is that before 2011, HPI of Quebec City and Ottawa are higher than Toronto, while Toronto outstripped others since 2016 and in 2017, this number is over 40%. In the first 3 months in 2018,however, things changed. HPI value of Toronto and Quebec City dropped by around 1-3%. This is not a notable value but it maybe means some kind of trend that house price in this two cities will drop in the future because of some political or financial methods. We will check next graph.


Graph2 : Toronto HPI from 2011-2018 compared with Mortgage Interest Rate

  ![png](/img/thp2.png)

From first plot, we can easily find out that 2017.8 was the peak of all HPI graph, with the value of 240.7. And from the second plot, we know the reason of HPI value dropped after that. In July 2017, the mortgage Interest rate was 3.140, which is same with value of 2013, however, in that 4 years, HPI of Toronto increased more than 40%. In another word, that means Canada government didn't do something in mortgage rate to limit the house price in that 4 years and to some extent, this leads the out-of-control of Toronto house now. In Oct-2017, mortgage rate increased, which made HPI value drop even considering seasonal reasons in Time Series. And Jan-2018 HPI increased around 8%,compared with Jan-2017, which comes back to original level(Graph3) and monthly increasing rate was dropped dramatically(Graph3). Next, I want to analyze


Graph3 : Increasing Rate For each Jan and Monthly HPI increasing rate

  ![png](/img/thp3.png)

Graph4 : Price of Different Types Houses vs Mortgage Rate

  ![png](/img/thp4.png)

From graph4, we see that peak of price of all types of houses was around May-2017, and then price seemed to drop while apartment price continuous increased although not so dramatically as before. This is not a good news for the ones who want to buy apartment recently since the financial methods didn't work on apartment. For other types of houses, price started to increase again since Jan-2018.

I also label 3 time points,Feb-2017,May-2017,Oct-2017. On Feb-2017, CBC and other medias started to report that centre bank had plan to increase the mortgage rate and immediately, then house price increased like insane in next 3 months. After that, price dropped down since more and more people refused to buy and/or can't afford houses. On Oct-2017, since the mortgage finially increased, house price dropped as before, but the trend seemed slower, which is confused me because financial methods didn't work this time and gradually, price stopped dropped on Dec-2017 and increased again.


Graph5 : Estimated HPI and Price

  ![png](/img/thp5.png)

Based on time series, we know that HPI will increase continuously in 2018 and peak at July and August. So if you do want to buy a house recently, I highly recommend to finish everything before April. While in other hand, apartment price will increase gradually and slightly and the increasing of mortgage will still not work on it. So if someone want to buy apartment, the earlier the better. Notice that this is just basic regular prediction without any influence of others, such as limit the number of houses each people can buy, the amount of mortgage and so on. But no doubt that, the price of Toronto house will still increase untill more financial and political mehods will be applied.


Graph6 : Some details numbers in 2018 coming from TREB(http://www.trebhome.com/market_news/release_market_updates/news2018/nr_market_watch_0218.htm)

  ![png](/img/thp6.png)

  ![png](/img/thp7.png)
