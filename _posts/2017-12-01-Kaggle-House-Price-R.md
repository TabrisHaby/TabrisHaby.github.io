---
layout:     post
title:      kaggle Competition : House Price in Zillow Code in R
subtitle:   R Code Version with Random Forest
date:       2017-12-01
author:     Haby
header-img: img/post-bg-2015.jpg
catalog: true
tags:
    - R
    - My Work
    - Machine Learning
---

### import library and packages
    library(ggplot2)
    library(corrplot)
    library(randomForest)

    train <- read.csv(choose.files(),header = T)
    test <- read.csv(choose.files(),header = T)

#### saleprice plot

    ggplot(train,aes(SalePrice))+
        geom_histogram(bins = 50,col = "white",alpha = 0.7,fill = "blue")+
        geom_density()

    ggplot(train,aes(log(SalePrice)))+
        geom_histogram(bins = 50,col = "white",alpha = 0.7,fill = "blue")

### EDA

    # all vars vs log(SalePrice)
    train$LogSalePrice <- log(train$SalePrice)

    # set up train_numeric
    train_numeric <-train[,sapply(train,is.numeric)]

    # find out NA obs
    apply(train_numeric,2,sum)

    # replace NAs with 0
    train_numeric[is.na(train_numeric)] <- 0

    # set up corrplot
    corrplot(cor(train_numeric),method = "circle")

    # based on corrplot, we find out the most corrated vars to saleprice and logsaleprice
    # OverQual YearBuilt YearRemodAdd MasVnrArea TotalBsmtSF X1stFlrSF GrlivArea FullBath
    # TotRmsAbvGrd GarafeCars GarageArea 11 in total

#### Set Linear modeling for relationship

    # set up linear model with numeric vars
    lm1 <- lm(LogSalePrice ~ OverallQual +YearBuilt +YearRemodAdd +MasVnrArea +TotalBsmtSF +
           X1stFlrSF +GrLivArea +FullBath+ TotRmsAbvGrd +GarageCars+ GarageArea,
       data = train_numeric)

    summary(lm1)
    # 81.92 corR

    # adjust once
    lm2 <- lm(LogSalePrice ~ OverallQual +YearBuilt +YearRemodAdd +TotalBsmtSF +
                  X1stFlrSF +GrLivArea +GarageCars,
              data = train_numeric)
    summary(lm2)
    # 81.9 corR
#### Imputation and Groupize
    # set up train_char
    train_char <- train[,sapply(train,is.factor)]

    # check data
    str(train_char)

    # four vars have many NAs, which is Alley PoolQC Fence and MiscFeature

    # plot all char vars 1-6
    par(mfrow = c(2,2))
    plot_char <- function(x)
    {
        i = 1
        for (i in 1:x)
        plot(train_char[,i],train$LogSalePrice)
    }
    plot_char(44)

    # we pick up MSSubClass LotShape Neighbourhood PavedDrive HeatingQC

    str(train_char$LotShape)

    # setup new variable lotshape_new and switch all IR to 0 and REG to 1
    train$LotShape_new[train_char$LotShape == "IR1" ] <- "0"
    train$LotShape_new[train_char$LotShape == "IR2" ] <- "0"
    train$LotShape_new[train_char$LotShape == "IR3" ] <- "0"
    train$LotShape_new[train_char$LotShape == "Reg" ] <- "1"
    str(train$LotShape_new)

    plot(train_char$LotShape_new,train$LogSalePrice)
    plot(train_char$PavedDrive)
    plot(train_char$HeatingQC)

### Modeling
    # set up random forest
    library(randomForest)
    set.seed(111)

    random_forest_1 <- randomForest(LogSalePrice ~ OverallQual +YearBuilt +
                                        YearRemodAdd +TotalBsmtSF +X1stFlrSF +
                                        GrLivArea +GarageCars+LotShape_new +
                                        PavedDrive + HeatingQC + Neighborhood,
                                    data = train,importance = T, ntree = 2000)
    varImpPlot(random_forest_1)
    print(random_forest_1)
    # % Var explained: 86.6
    random_forest_2 <- randomForest(LogSalePrice ~ OverallQual +YearBuilt +
                                        YearRemodAdd +TotalBsmtSF +X1stFlrSF +
                                        GrLivArea +GarageCars+LotShape_new +
                                         HeatingQC + Neighborhood,
                                    data = train,importance = T, ntree = 2000)
    varImpPlot(random_forest_2)
    print(random_forest_2)
    # % Var explained: 86.64

    test$LotShape_new[test$LotShape == "IR1" ] <- "0"
    test$LotShape_new[test$LotShape == "IR2" ] <- "0"
    test$LotShape_new[test$LotShape == "IR3" ] <- "0"
    test$LotShape_new[test$LotShape == "Reg" ] <- "1"
    test$GarageCars[is.na(test$GarageCars)] <- 0
    test$TotalBsmtSF[is.na(test$TotalBsmtSF)] <- 0

    prediction1 <- predict(random_forest_3, test)
    submit1 <- data.frame(Id = test$Id, SalePrice = exp(prediction1))
    write.csv(submit1, file = "Kaggle_House_Price_20161125_1.csv",row.names = F)


    # try more variables redo the numeric analysis, we pick up MasVnrArea Fireplaces TotalBsmtSF
    # TotRmsAbvGrd GarageArea
    lm2 <- lm(LogSalePrice ~ OverallQual +YearBuilt +YearRemodAdd +MasVnrArea +TotalBsmtSF +
                  X1stFlrSF +GrLivArea +FullBath+ TotRmsAbvGrd +GarageCars+ GarageArea+
                  MasVnrArea +Fireplaces +TotalBsmtSF +TotRmsAbvGrd +GarageArea,
              data = train_numeric)
    summary(lm2)

    lm3 <- lm(LogSalePrice ~ OverallQual +YearBuilt +YearRemodAdd+TotalBsmtSF +
                  GrLivArea + TotRmsAbvGrd +GarageCars+ GarageArea+
                  Fireplaces +TotalBsmtSF +TotRmsAbvGrd +GarageArea,
              data = train_numeric)
    summary(lm3)
    # R-sq 0.8308

    set.seed(111)

    random_forest_4 <- randomForest(LogSalePrice ~ OverallQual +YearBuilt +YearRemodAdd+TotalBsmtSF +
                                        GrLivArea + TotRmsAbvGrd +GarageCars+ GarageArea+
                                        Fireplaces +TotalBsmtSF +TotRmsAbvGrd +GarageArea+LotShape_new +
                                        PavedDrive + HeatingQC + Neighborhood,
                                    data = train,importance = T, ntree = 2000)
    varImpPlot(random_forest_4)
    print(random_forest_4)
    # % Var explained: 87.22

    random_forest_5 <- randomForest(LogSalePrice ~ OverallQual +YearBuilt +YearRemodAdd+TotalBsmtSF +
                                        GrLivArea + TotRmsAbvGrd +GarageCars+ GarageArea+
                                        Fireplaces +TotalBsmtSF +TotRmsAbvGrd +GarageArea+
                                         HeatingQC + Neighborhood +PavedDrive,
                                    data = train,importance = T, ntree = 2000)
    varImpPlot(random_forest_5)
    print(random_forest_5)
    #   % Var explained: 87.15

    test$GarageArea[is.na(test$GarageArea)] <- 0

    prediction2 <- predict(random_forest_5, test)
    submit2 <- data.frame(Id = test$Id, SalePrice = exp(prediction2))
    write.csv(submit2, file = "Kaggle_House_Price_20161125_2.csv",row.names = F)
    # score 0.15431

    random_forest_6 <- randomForest(LogSalePrice ~ OverallQual + YearBuilt + YearRemodAdd +
                                            TotalBsmtSF + GrLivArea + TotRmsAbvGrd + GarageCars +
                                            GarageArea + Fireplaces + TotalBsmtSF + TotRmsAbvGrd +
                                            GarageArea + HeatingQC + Neighborhood + MSSubClass +
                                            Foundation + MSZoning,
                                        data = train,importance = T, ntree = 2000)

    varImpPlot(random_forest_6)
    print(random_forest_6)
