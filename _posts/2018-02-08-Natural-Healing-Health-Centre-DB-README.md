---
layout:     post
title:      Natural Healing Health Centre Customer DB
subtitle:   ReadMe File
date:       2018-02-08
author:     Haby
header-img: img/post-bg-2015.jpg
catalog: true
tags:
    - MySQL
    - My Work
---

> This EER Diagram is allowed to publish here by the owner of Natural Healing Health Center.

> This diagram is licensed under a Creative Commons Attribution 4.0 International license.


<font size = '6'> Natural Healing Health Centre Customer DB </font>
<font size = '5'> ReadMe File </font>


1.version

    Version 1 : 2016-02-08
    Version 2 : 2018-01-16

    1.Add new table Duration : record each treatment durtion for different customers

    2.delete table estimated_income : estimate value based on internet data is not accurate

    3.connect Insurance Table to consumption_habit table directly

    4.correct Sunlife company contact number in Insurance table


2.Working Environment

    1.OS : Windows 10 v1702 64 bit + Ubuntu 16.04.3 LTS

    2.SQL : MySQL 5.7.20

    3.SQL GUI : MySQL WorkBench 6.3

    4.Analysis Software : Python 3.6.3

    5.Analysis Software GUI : Spyder from Anaconda 5.0.1


3.Database Details<br>

About data

    1.Data from two ways :

      # Client health history and conscent form

      # Paper and Online Questionares

    2.data volumes :

      # Around 5000 rows and 18 variables in total


There are 11 tables in total, which are :

    # Customer : Customer general information, like name,gender,gender,phone,email,etc

    # Address : Address location information, like city, province, post code, etc

    # Address Area : Divide the area into different geographical districts

    # Customer Service Duration : connection of Service, Duration and Customer table. This is a mant-to-many connection

    # Service : Service treatment Information.

    # Duration : Duration for each treatment

    # Consumption Habit : many-to-many connection of Customer table and Treatment Frequency, Insurance table,Payment Method table,How_to_know table

    # Treatment Frequency : Treatment Frequency for each customers, customers Consumption Habit

    # Insurance : Insurance company for each customer, customers Consumption Habit

    # Payment Method : Payment method each customer prefer, customers Consumption Habit

    # How_to_know : from questionare, check the methods that customers know us, find the best way for advertising
