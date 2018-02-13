<font size = '5'>Version</font><br>
<br>

  <font size = '3'>Version 1 : 2017-02-08</font><br>


  <font size = '3'>Version 2 : 2018-01-16</font>

    1.Add new table Duration : record each treatment durtion for different customers

    2.delete table estimated_income : estimate value based on internet data is not accurate

    3.connect Insurance Table to consumption_habit table directly

    4.correct Sunlife company contact number in Insurance table


<font size = '5'>Working Environment</font><br>

    1.OS : Windows 10 v1702 64 bit + Ubuntu 16.04.3 LTS

    2.SQL : MySQL 5.7.20

    3.SQL GUI : MySQL WorkBench 6.3

    4.Analysis Software : Python 3.6.3

    5.Analysis Software GUI : Spyder from Anaconda 5.0.1


<font size = '5'>Database Details</font><br>
<font size = '4'>About data</font><br>

  Data from two ways :

      # Client health history and conscent form

      # Paper and Online Questionares

  data volumes :

      # Around 5000 rows and 18 variables in total


<font size = '4'>There are 11 tables in total, which are :</font><br>

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
