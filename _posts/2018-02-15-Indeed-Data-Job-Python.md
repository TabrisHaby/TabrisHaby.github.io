---
layout:     post
title:      Indeed Web Crawler by Python
subtitle:   For Data-Related in Toronto ON
date:       2018-02-15
author:     Haby
header-img: img/post-bg-2015.jpg
catalog: true
tags:
    - Python
    - My Work
    - Web Crawler
---

```python
        # -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 10:32:35 2018

original code : https://medium.com/@msalmon00/web-scraping-job-postings-from-indeed-96bd588dcb4b

Find out the data-related job in Toronto, ON from Indeed.com

Do some simple analysis and visualization for job title name and key features for employees

@modify: Haby
        """
```

```python
# import package

import requests
import bs4
from bs4 import BeautifulSoup
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
# set up functions for each location on pages

```
>Function code from Medium
 >https://medium.com/@msalmon00/web-scraping-job-postings-from-indeed-96bd588dcb4b

```python
# job title is under div/a node with data-tn-element/title label
def job_title(soup):
  jobs = []
  for div in soup.find_all(name='div', attrs={'class':'row'}):
    for a in div.find_all(name='a', attrs={'data-tn-element':'jobTitle'}):
        jobs.append(a['title'])
  return(jobs)

```


```python
# company title : <span class="company"> in div node
def company(soup):
    companies = []
    for div in soup.find_all(name='div', attrs={'class':'row'}):
        company = div.find_all(name='span', attrs={'class':'company'})
        # if can be select by company
        if len(company) > 0:
            for b in company:
                companies.append(b.text.strip())
        # else pick up by result-link-source
        else:
            sec_try = div.find_all(name='span', attrs={'class':'result-link-source'})
            for span in sec_try:
                companies.append(span.text.strip())
    return(companies)
```


```python
# location
def location(soup):
  locations = []
  spans = soup.find_all('span', attrs={'class':'location'})
  for span in spans:
      locations.append(span.text)
  return(locations)
```


```python
# Salary : most of salary doens't exist, fill NA
def salary(soup):
  salaries = []
  for div in soup.find_all(name='div', attrs={'class':'row'}):
    try:
      salaries.append(div.find('nobr').text)
    except:
      try:
        div_two = div.find(name='div', attrs={'class':'sjcl'})
        div_three = div_two.find('div')
        salaries.append(div_three.text.strip())
      except:
        salaries.append('Nothing_found')
  return(salaries)
```


```python
# Job Summary
def job_summary(soup):
  summaries = []
  spans = soup.findAll('span', attrs={'class': 'summary'})
  for span in spans:
    summaries.append(span.text.strip())
  return(summaries)
```


```python
# Crawler Part

# iterate over pages : there are 10 job list in each page and the new page starts at 10
url_o = 'https://ca.indeed.com/jobs?q=Data&l=Toronto,+ON&start='
page = [i*10 for i in range(0,21)]

# new dataframe for data
df = pd.DataFrame()#columns = columns)

# loop 20 pages
for i in page :
    url =  url_o + str(i)  

    # get the url
    page = requests.get(url)

    # decode the pages with BeautifulSoup
    soup = BeautifulSoup(page.text, 'html.parser')

    # concat all as dataframe
    temp = pd.DataFrame([job_title(soup),company(soup),location(soup),job_summary(soup),salary(soup)])
    df = df.append(temp.T,ignore_index = True)

# set columns name
df.columns = ['job_title', 'company_name', 'location', 'summary', 'salary']
print(df.head())
```

                                              job_title    company_name  \
    0  AI / Machine Learning Scientist – Toronto AI Lab  LG Electronics   
    1                                        IT Analyst    Canada Goose   
    2                           Data Conversion Analyst       FieldEdge   
    3                                    Data Scientist      LoyaltyOne   
    4         Data Analyst, Visa Consulting & Analytics            Visa   

          location                                            summary  \
    0  Toronto, ON  Spark, Hadoop), large scale data analysis, opt...   
    1  Toronto, ON  Gather required data from Systems, Vendors, an...   
    2  Toronto, ON  The Data Analyst will help new clients convert...   
    3  Toronto, ON  Expert knowledge of data modeling and understa...   
    4  Toronto, ON  Extensive experience with SQL for extracting a...   

              salary  
    0  Nothing_found  
    1  Nothing_found  
    2  Nothing_found  
    3  Nothing_found  
    4  Nothing_found  



```python
# Visualization
# some simple analysis for job_title

# job title : split first title by '-' and ','
df['First_title'] = df['job_title'].map(lambda x : x.split('–')[0].split(',')[0].split('(')[0].split('$')[0].split('-')[0].strip())

# top 10 job titles
df['First_title'].value_counts()[:20].plot.barh(rot = 30,title = 'Counts for Different Titles',legend = True)
plt.show()
```


![png](img/indeed1.png)


#### From graph above, we find that for data-related job in Toronto, Data Scientist is the most needed job. Although 'AI/ Machine Learning Scientist' ranks top 2, they were post by same company.

#### Most of Top ranking jobs are analyst jobs,  but in different areas, such as IT, Business, and so on. It means that Data Analyst is needed by many areas, not only in Math / Stats and computer area.

#### Next, I will pick up some keywords in job describtion and summary to see what are the most important features for data-related workers for each company


```python
# find keywords in job summary

words = []
for row in df['summary'] :
    for word in row.replace(',',' ').replace('.',' ').split() :
        word.replace('(',' ').replace(')',' ').strip()
        word = word.lower()
        words.append(word)
```


```python
# I notice that there are lots of useless words in this list, such as 'I', 'for' and so on. I will delete them out of the words list

# delete useless elements
dropped = ['and','to','the','of','in','with','a','for','will','is','our','an','data','large','skills','requierd','work',
           'as','from','including','we','are','by','be','for','us','it','.','-','onto','but','not',
           'on','new']
key_words = []
for word in words :
    if word not in dropped :
        key_words.append(word)
```


```python
# list the top 20 important words in data related job posts
pd.Series(key_words).value_counts()[:20].plot.barh(rot = 30,
  title = 'Top 20 Keywords for data-related job',legend = True)
plt.show()
```


![png](img/indeed2.png)



```python
# list the top 20-40 important words in data related job posts
pd.Series(key_words).value_counts()[20:40].plot.barh(rot = 30,
  title = 'Top 40 Keywords for data-related job',legend = True)
plt.show()
```


![png](img/indeed3.png)



```python
# list the top 40-60 important words in data related job posts
pd.Series(key_words).value_counts()[40:60].plot.barh(rot = 30,
  title = 'Top 60 Keywords for data-related job',legend = True)
plt.show()
```


![png](img/indeed4.png)


### Top 20 keywords seem to be more general skills that companies need, while top 20-40 seem to be professional skills

#### Top 1 Analyst : Main skill for data-worker, find out the useful infomation from messes.
#### Top 2 Experience : It seems new graduators are not welcomed.
#### Top 5 Team : Team working skills are more important, suck as commucation skills and so on.
#### Top18 Resposible : HRs will like you if you are responsible to your job.
#### Top 19 Learning : Graduation does't mean finish studying.

#### Top 20-40 : Optimization / functional / hadoop : professional skills for big data mining
#### Multi-disciplinary : word I never notice before. Ability to combine different parts together.
