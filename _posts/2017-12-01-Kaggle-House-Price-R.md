---
layout:     post
title:      House Price Analysis in R with XGBoosting
subtitle:   Demo in Zillow House Price Competition in Kaggle
date:       2017-12-01
author:     Haby
header-img: img/post-bg-2015.jpg
catalog: true
tags:
    - R
    - My Work
    - Machine Learning
    - XGBoosting
    - Algorithms
---
> Zillow House Data or any results are not allowed to publish outside of Kaggle.com, so I post scripts only here.

Input Packages

    library(ggplot2) # Data visualization
    library(caret) # CSV file I/O, e.g. the read_csv function
    library(dplyr)
    library(mice)
    library(Hmisc)
    library(lubridate)
    library(data.table)
    library(xgboost)
    library(gbm)

Data Import and Transformation

    # import data
    properties <- fread('../input/properties_2016.csv', stringsAsFactors = FALSE)
    train <- fread('../input/train_2016_v2.csv',stringsAsFactors = FALSE)

    # check NAs
    colSums(is.na(properties)) %>% sort(decreasing = T) %>% barchart(pt.cex = 2.5, main = 'Amount of Missing Value ')

    # basic data transformation
    properties$hashottuborspa <- ifelse(properties$hashottuborspa == 'true', 1, 0)
    properties$fireplaceflag <- ifelse(properties$fireplaceflag == 'true', 1, 0)
    properties$taxdelinquencyflag <- ifelse(properties$taxdelinquencyflag == 'Y', 1, 0)
    properties$propertycountylandusecode <- as.numeric(as.factor(properties$propertycountylandusecode))
    properties$propertyzoningdesc <- as.numeric(as.factor(properties$propertyzoningdesc))

    # basic transformation : numeric/ integer to factor
    properties$buildingqualitytypeid <- as.factor(properties$buildingqualitytypeid)
    properties$fips <- as.factor(properties$fips)
    properties$heatingorsystemtypeid <- as.factor(properties$heatingorsystemtypeid)
    properties$propertylandusetypeid <- as.factor(properties$propertylandusetypeid)
    properties$regionidcity <- as.factor(properties$regionidcity)
    properties$regionidcounty <- as.factor(properties$regionidcounty)
    properties$regionidcity <- as.factor(properties$regionidcity)
    properties$unitcnt <- as.factor(properties$unitcnt)



imputation           


    # training set
    df <- left_join(train,properties,on = 'parcelid')

    # outliers
    df <- df %>% filter(logerror <= 0.4 & logerror >= -0.39)

    # add new variables
    df$age <- as.numeric(2016 - df$yearbuilt)
    df$transactiondate <- as.Date(df$transactiondate)
    df$month <- as.factor(month(df$transactiondate))
    df$weekdays <- as.factor(weekdays(df$transactiondate))
    df$transactiondate <- as.numeric(df$transactiondate)


Modeling

XGBoost Linear

    ## transformation all to integer/numeric
    df1 <- df %>% mutate_if(is.factor,as.integer)

    ## user-defined mae function
    maeSummary <- function (data,
                            lev = NULL,
                            model = NULL) {
      out <- ModelMetrics::mae(data$obs, data$pred)  
      names(out) <- "MAE"
      out
    }

    ## parameter
    xgb_grid <- expand.grid(nrounds = c(160),
                            eta = c(0.1),
                            max_depth = c(5),
                            gamma = c(1),
                            colsample_bytree = c(0.8),
                            min_child_weight = c(0.8),
                            subsample = c(0.9))
    fitControl <- trainControl(method = "repeatedcv",
                               number = 10,
                               repeats = 3,
                               summaryFunction = maeSummary)

    # train data set
    train_x <- as.matrix(df1[,-c(1,2,3)])
    train_y <- df1[,"logerror"]

    # Model
    set.seed(1)
    xgbtree_mod <- train(x = train_x,
                         y = train_y,
                         method = 'xgbTree',
                         trControl = fitControl,
                         tuneGrid = xgb_grid,
                         metric = 'MAE')
    print(xgbtree_mod)   # MAE 0.05256457


predict
    properties1 <-data.matrix(properties)
    submission <- properties %>%
      mutate("201610"=predict(object=xgbtree_mod, newdata=properties1),
             month=factor("11", levels = levels(train$month)),
             "201611"=predict(object=xgbtree_mod, newdata=properties1),
             month=factor("12", levels = levels(train$month)),
             "201612"=predict(object=xgbtree_mod, newdata=properties1),
             #month=factor("10", levels = levels(train$month)),
             "201710"=0,
             #month=factor("11", levels = levels(train$month)),
             "201711"=0,
             #month=factor("12", levels = levels(train$month)),
             "201712"=0) %>%
      select(parcelid, `201610`, `201611`, `201612`, `201710`, `201711`, `201712`)


    options(scipen = 999) ## DO not use scientific notation
    write.csv(submission, "submission_test.csv", row.names = FALSE)
