# PredictNBARookiesToHallOfFame

This project was largely done in a class at Rose-Hulman Institute of Technology called Introduction to Machine Learning
I did this project with my fellow Rose-Hulman peers Michael Domke, and Albert Udom

We used a single file R-script to do our data analysis which can be found in [ActualScriptWithAllAlgorithms.R](https://github.com/joshpal97/PredictNBARookiesToHallOfFame/blob/master/ActualScriptWithAllAlgorithms.R)

The data that we used was found on Kaggle.com and other various sites that collected a massive amount of NBA data. You can 
see the data we used in any csv in [data_spreadsheets](https://github.com/joshpal97/PredictNBARookiesToHallOfFame/tree/master/data_spreadsheets)

Any analysis of data using the algorithms can be found in the [final report](https://github.com/joshpal97/PredictNBARookiesToHallOfFame/blob/master/Final_Report.docx) which you can view raw as a word document

Most of the pictures didn't upload correctly to git, so the final report document has most of the results of our algorthm use

## Algorithms in Project
We used 7 different algorithms:
- Lazy learning
- Naïve Bayes 
- Decision Trees and Rules
- Regression 
- Neural Nets and Support Vector Machines 
- Market Basket Analysis
- Clustering with K-means


## Summary of Project

This process has been very fun and informative for our project. Due to the numeric nature of our data, we found we could only really use algorithms that worked well with numeric data. Due to this, we were unable to use the Naive Bayes nor the Market Basket Analysis methods to analyze our data. As far as the other algorithms we learned go, we feel that regression trees were most suited to our data. The other reason that we thought this was that regression trees seemed to make more intuitive sense. Others that worked well was clustering with k-means, simple decision trees, and neural nets, for the most part.

One big drawback of our dataset is that there are not many hall of famers that have been drafted since 1980 (30 or so out of 1500). This data set also includes current players that will more than likely be in the hall of fame, but are not in yet. Another thing we noted time and time again is that the nature of basketball has changed multiple times in the time-frame of our data. For example, the 80s was more or less dominated by bigger players while modern day NBA has seen a change to smaller players and guards being the focal point of the team.

The hall of fame is also not a very set award, meaning that there is a committee that votes players and coaches into the hall of fame. There may be some correlation between things such as average points, player efficiency, etc and their status as hall of fame or not, but it does not necessarily mean that they will be or wont be. Some players have been voted in as players, others as analysts, coaches, General Managers (GMs), or even their impact on their communities with regards to basketball. Every year, the committee votes on players, coaches, GMs, owners, and others making it just that much harder to predict hall of fame inductees from the NBA.  
