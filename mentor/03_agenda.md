# Mentor Session 03 Agenda

#### April 27, 7:00am (45 Minutes)


[Data Science Conversation](#ds_converstation)

[Status Update](#status_update)

[Technical Questions](#technical_questions)

[Notes](#notes)


---
### <a name="ds_conversation"></a> Data Science Conversation (10 minutes)
- [Simple Time Series Forecasting Models to Test So That You Don't Fool Yourself](http://machinelearningmastery.com/simple-time-series-forecasting-models/)
    - [Notebook Calculations](../notebooks/Time_Series_Forcasting.ipynb)
    - Persistence Model ("Naive" Forecast)
        - Additional Article On Persistence Model
            - [How to Make Baseline Predictions for Time Series Forecasting with Python](http://machinelearningmastery.com/persistence-time-series-forecasting-with-python/)
        - **GOALS**
            - Establish baseline performance on forecast method as quickly 
            as possible.
            - Gain understanding of the dataset to develop a more advanced 
            model.
        - Test Harness
            - **Dataset**: use to train and evaluate models
            - **Resampling Technique**: test / train split
                - Simple
                - Fast
                - Repeatable
            - **Performance Measure**: evaluation method
        - Explicit time solution algorithm
        - Implementation
            1. Transform the univariate dataset into a supervised learning 
            problem.
            1. Establish the train and test datasets for the test harness.
            1. Define the persistence model.
            1. Make a forecast and establish a baseline performance.
            1. Review the complete example and plot the output.

[Table of Contents](#toc)


---
### <a name="status_update"></a> Status Update
- Created Time Series Forecasting notebook 
- Created GitHub pages website
    - https://timothyhelton.github.io/
    - This site will serve as a portfolio of my projects.
    
#### Unit 5: Introduction to Data Science
- Reading "The Art of Data Science"
    - Chapter 3 Complete
    
#### Unit 6: Getting and Cleaning Data
- NYC MTA Turnstile Dataset
    - Completed all exercises
    - Need to review the solution and submit

[Table of Contents](#toc)


---
### <a name="technical_questions"></a> Technical Questions 
- Project Ideas:
    - [DATA.GOV](https://www.data.gov/)
        - Looks like this site has datasets, so web scrapping would not be 
        required.  Think it would be better to mine our own data, but a 
        topic from this site may be considered as a fall back option.
        - [U.S. Chronic Disease Indicators (CDI)](https://catalog.data.gov/dataset/u-s-chronic-disease-indicators-cdi-e50c9)
        

[Table of Contents](#toc)


---
### <a name="notes"></a> Notes
- Next Week's Data Science Conversation:
    - [Simple Time Series Forecasting Models to Test So That You Don't Fool Yourself](http://machinelearningmastery.com/simple-time-series-forecasting-models/)
        - Expanding Window Model
            - Discuss in mentor session 04
        - Rolling Window Forecast Model
            - Discuss in mentor session 05
            
- Look into this packages for time series analysis:
    - [Prophet](http://blog.revolutionanalytics.com/2017/02/facebook-prophet.html)
    - [Smart Drill](http://smartdrill.com/time-series.html)
    
- Update the About section of the webpage:
    - Add thesis information
    - Talk about work
    - Make it more like a blog
    - I'm the guy to bring big data to engineering designs

- Pull Historical Data to create a unique dataset
    - Track medical device companies web pages from history and look at what
     they promised over time.
        - Identify medical device companies
        - Mask IP to avail getting blacklisted
        - Try to get into thousands of companies
        - Look at crunch base 
    - [Way Back Machine](https://archive.org/help/wayback_api.php)
    - [Big Query](https://cloud.google.com/bigquery/what-is-bigquery)
    - [Facebook](https://developers.facebook.com/docs/graph-api)

[Table of Contents](#toc)

