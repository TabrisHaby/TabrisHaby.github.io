---
layout:     post
title:      A little about Algorithms
subtitle:   Boosting, Bagging and Bootstrap
date:       2018-03-01
author:     Haby
header-img: img/post-bg-2015.jpg
catalog: true
tags:
    - Python
    - My Work
    - Algorithms
    - Machine Learning
---

### 1.Ensemble Learning Method

Ensemble Learning Method is a way to aggregrate some of learners together to make a better result. The principle is from PCA(Probably Approximately Correct). If the accuracy of some guess is just a little better than random guess(0.5), we name them as weak learning and for the good guess, we name them as strong learner. In most of data, it is hard to find out the strong learners, and it is easier to search the weak learners. Ensemble Learning Method will aggregrate these weak learners and make a strong learner for better prediction.

### 2.Bagging, Boosting and Booststrap

Bagging is also called bootstrap aggregating. It is a kind of Booststrap method, so I will focus on bagging and boosting.

  #### 2.1 The main difference of bagging and boosting

  In sample selection, bagging method will select sample and put it back in next selection. For example, there are 5 balls and number them 1-5.When The first time I select No.2 and I will put No.2 back before next selection. So I also have the chance to select No.2 in next selection. The weight of different sample is same. For boosting , the train set is the same one for different selection while the weight of different sample is changed based on result of last round.

  Since for boosting, every model needs the result of last round to adjust the weight of sample in this round, it has to be the linear workflow, which means this needs more time to get result. For bagging however, all models can be generated parallelly.

  #### 2.2 Bagging method vs Boosting Method

  Random Forest is the most popular bagging method recently, it is combination of  bagging method and Decision Tree. As a bagging method, RF is very sensitive to outliers/meanless variables and that is the reason for most of Machine Learing data with RF algorithm, we need to drop outliers and useless variables to reduce variance and overfitting problems.

  AdaBoost is the iteration algorithm in Boosting using exponential loss function as weight. It is short for adaptive boosting and  found by Yoav Freund and Robert Schapire. It combines many weak learners with different weights by voting method. When we combined AdaBoost algorithm with Decision Tree, we can get a new method claaed Boosting Tree.
