# Optimizing Algorithmic Options Trading using Machine Learning

## Introduction

Algorithmic trading involves the creation of rule-based strategies that can be executed automatically and programmatically in the stock market. When this is done with option trading, there are numerous standard strategies that can be customized for execution in the market, without having to create more complex strategies. Like any form of trading, algorithmic options trading goes through profits and losses. The purpose of this project is to explore different areas of Machine Learning that could potentially help in getting more consistent positive returns.

The options strategy that we intend to use for this project is a simple intraday short straddle that runs from the morning 10AM till 3PM. We plan to customize this strategy by using the concept of stoploss and trailing stoploss for each of the legs of the straddle based on different conditions. The intuition behind a short straddle is that the probability that the underlying instrument swings by a large range within a single day is low and that can be used to our advantage. Despite this, there are instances where there could be a highly directional movement on a given day leading to a stoploss trigger for one of the legs of the straddle. In some cases, mid-way through the trading session, the market sentiment changes, and the underlying instrument starts moving in the opposite. This turns out to be detrimental to the strategy and what was once a good profit, turns out to be a loss.

The main task of the project is to use Machine Learning to pinpoint this market sentiment through indicators, other underlying instruments, and sentiment analysis. Using these features, we want to create a multinomial classification model that can determine a range of directional movement for a given timeframe. The predicted directional movement range can be used with the strategy logic to ascertain if the market outcome works in the favor of our strategy or against it. To put it in simple terms, we are looking to automate the process of taking discretionary decisions in the market using Machine Learning.

## Data for the Project

We plan to use the S&P500 index as the underlying instrument for our short straddle. The minutewise underlying instrument data and options data will be downloaded by creating a script that can use REST APIs provided by polygon.io, and then saved into a data store either in the form of parquet/CSV files on Amazon S3 which can be queried for future processing. We plan to go with the polygon.io Basic Pricing Plan [1], which gives us 2 year historical Stocks, Options, Indices, and Currency data for free, with the limitation of 5 API calls per minute. We plan to create multiple accounts and run the API script on multiple serverless AWS Lambda function setup with AWS EventBridge CRON jobs which reduces a significant amount of manual effort. As the Lambda function fetches the data, it would store it on Amazon S3 directly, from where the data can be queried.

Once we retrieve the underlying instrument and options data, the next step would be to generate the data that is important for us i.e., the profit and loss data for our short straddle. We will have to simulate the short straddle strategy historically and collect the hourly profit and loss data for the strategy. This hourly data will be used as the principal data for the prediction, along with indicators. The general idea is that if there is a significant change in the profit or loss in a one-hour timeframe, can it be predicted using the market sentiment.

## The Model

We plan on using classification models to predict a range of potential values for the underlying instrument and how that would affect our strategy based on the payoff curve. The other important factor is to determine which are the features that can be used to accurately predict this direction. SUI Xue-shen, et al. [2] used different technical indicators with Support Vector Machines for feature selection with respect to the Shanghai Stock Market and were able to use fitness measures to evaluate the best features for the classification problem. Similarly, we will use Support Vector Machines to select the best features in the form of technical indicators, fundamental data through sentiment analysis, and OHLCV data of indices and VIX, to ascertain which are the best features to detect the direction of market movement accurately.

Given the random walk nature of the stock market [3], Lohrmann and Luukka use fuzzy entropy methods for feature selection and use these features with an ensemble Random Forest classifier to predict the open-close return of the S&P500 index. We can use a similar concept and predict the open-close return of the index for a shorter time interval which will in-turn give us the direction of the market.

Another path that we can explore is presented by Brunhuemer et. al. [4] in the use of machine learning for short-option strategies. The idea presented is that static back-tested strategies had lower performance as compared to dynamic decision-finding in choosing variants of a strategy. The goal of their model was to use machine learning in the form of gradient tree boosting to check if entering a strategy given the market conditions would be effective or not. In a similar space, we will use gradient tree boosting to check if exiting a strategy early would be beneficial given the market conditions.

## Potential Challenges
The first challenge would be retrieving options data from polygon.io. Since the free API has a limitation of only 5 API calls per minute, it would require a significant amount of time to retrieve the data as there is a massive amount of options data available for each expiry. We would have to streamline this process by setting up CRON jobs or Python scripts on either a scheduled microservice architecture like AWS Lambda or on an EC2 Instance.

The other challenge that could potentially hamper our results is the use of classification models. Classification models generally have the tendency to overfit the data and since we are using this model on a short straddle, it could work well for the short straddle but would fail for other option strategies. The model should work for most standard strategies for it
to be called robust.
