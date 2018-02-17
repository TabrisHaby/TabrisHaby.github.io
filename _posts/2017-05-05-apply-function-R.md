---
layout:     post
title:      Apply Function Group in R
subtitle:   Demo code with Iris data
date:       2017-05-04
author:     Haby
header-img: img/post-bg-2015.jpg
catalog: true
tags:
    - R
    - My Work
---

>https://nsaunders.wordpress.com/2010/08/20/a-brief-introduction-to-apply-in-r

>A brief introduction to apply§ in R

CREATE A MATRIX OF 10 * 2

    m <-matrix(c(1:10,11:20),nrow = 10)

Simple: either the rows (1), the columns (2) or both (1:2). By both, we mean. Apply the function to each individual value.

    apply(m,1,mean)
    apply(m,2,mean)
    apply(m,1:2,function(x) x/2)

    attach(iris)
    head(iris)

 get the mean of the first 4 variables
    apply(iris[,1:4],2,mean)

The by function is a little more complex than that. Read a little further and the documentation tells you that a data frame is split by row into data frames subsetted by the values of one or more factors, and function FUN is applied to each subset in turn.So, we use this one where factors are involved. get the mean of the first 4 variables, by species

    by(iris[,1:4],Species,colMeans)


 eapply
 Description: eapply applies FUN to the named values from an environment and returns the results as a list.This one is a little trickier, since you need to know something about environments in R. An environment, as the name suggests, is a self-contained object with its own variables and functions.

     a new environment
        e <-new.env()
     two environment variables
        e$a <- 1:10
        e$b <- 11:20
     mean of the environment
        eapply(e,mean)


 lapply
 Description: lapply returns a list of the same length as X, each element of which is the result of applying FUN to the corresponding element of X.create a list with 2 elements

    l <-list(a = 1:10,b = 11:20)
    lapply(l,mean)
    lapply(l,sum)
    class(lapply(l,mean))
    class(sapply(l,mean))


 sapply
 Description: sapply is a user-friendly version of lapply by default returning a vector or matrix if appropriate create a list with 2 elements

    l <- list(a = 1:10, b = 11:20)
    l.mean <-sapply(l,mean)
    class(l.mean)


 vapply
 A third argument is supplied to vapply, which you can think of as a kind of template for the output.five number of values using vapply

    l.fivenum <- vapply(l,fivenum,c(Min. = 0, "1st Qu." = 0, Median = 0, "3rd Qu." = 0, Max. = 0))
    class(l.fivenum)
    l.fivenum


 replicate

    replicate(10,rnorm(10))

 mapply

    l1 <- list(a = c(1:10),b = c(11:20))
    l2 <- list(c = c(21:30),d = c(31:40))
    mapply(sum,l1$a,l1$b,l2$c,l2$d)

 rapply
 Description: rapply is a recursive version of lapply. I think recursive§ is a little misleading. What rapply does is apply functions to lists in different ways, depending on the arguments supplied. Best illustrated by examples:

    rapply(l,log2)
    rapply(l,log2,how = "list")

 tapply
 Description: Apply a function to each cell of a ragged array, that is to each (non-empty) group of values given by a unique combination of the levels of certain factors.Woah there. That sounds complicated. Dont panic though, it becomes clearer when the required arguments are described. Usage is tapply(X, INDEX, FUN = NULL,  simplify = TRUE), where X is an atomic object, typically a vector and INDEX is a list of factors, each of same length as X.So, to go back to the famous iris data, Species§ might be a factor and iris$Petal.Width would give us a vector of values. We could then run something like:

    tapply(iris$Petal.Length,Species,mean)
    by(iris$Petal.Length,Species,mean)
