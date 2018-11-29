# Dude, Where's My Bus?

#### Using Machine Learning to solve the San Francisco commute dilemma

*Jonathan Bate*

:trolleybus: :train: :bus: :light_rail: :bus: :train: :trolleybus:

##### Overview

Although the residents of San Francisco are blessed with a frequent transit network, the total journey time of trips on Muni is notoriously highly variable, due to both unexpectedly long wait times and unexpected delays during the travel itself.

*Dude, Where’s My Bus?* is a Data Science project to create a Machine Learning model that will generate a probabilistic prediction of the total journey time (wait time plus travel time), from every bus, train, and cable car stop in the city, to every other stop in the city.

By making predictions in a probabilistic manner, it is possible to predict with a specified confidence threshold (e.g. 95%) when a customer needs to be at their home stop in order to get to work on time.

##### Methodology
* The GPS locations for every bus, train, and cable car in the city are collected from the 511.org API every minute throughout each day.
* This raw location data is then parsed into stop events, which occur when it is detected that a vehicle has passed a stop on it’s route.
* From these stop events, the travel time between each stop pair for each journey is calculated.
* From these travel times, a distribution of total journey times (including wait times) is calculated for every stop pair in the city and every hour of every day.
* Exploratory Data Analysis indicates that a gamma function is the most appropriate representation of the total journey time distributions. A gamma distribution is fitted to each of the distributions, and the shape, scale, and mean parameters are stored.
* Two Random Forest Regression models are trained on the calculated shape, scale, and mean parameters, and used to predict them for trips between any given pair of stops at any date and time in future. The models continue to learn as new data is collected and processed.
* Finally, a web application allows users to obtain travel time distributions from the regression models for any stop pair in the system.
##### Technology
* Data Science - Anaconda, Scikit-Learn, Numpy, Pandas, Matplotlib
* Web Development - Flask, JQuery, CSS
* Amazon Web Services - Redshift, EC2, S3, Elastic Beanstalk, Route 53
##### Project Links
http://www.dudewheresmybus.net/
https://github.com/jonobate/muni
https://github.com/jonobate/muni-web
