# Youtube Comment Analyzer

Hey this is Aryan,

   This is a sentiment analysis web application that allows users to enter a YouTube video URL and analyze the sentiment of the comments associated with the video. The application is built using Flask, a Python web framework. The machine learning model used for sentiment analysis is a logistic regression model trained using natural language processing techniques on the sentiment140 (1.6M) tweet dataset. The application uses the Google API to extract comments associated with the video and classifies the comments as positive or negative using logistic regression. In addition to the above technologies, this web application also uses a MySQL database to store the sentiment analysis results and searched comments for each video that is analyzed. The application is configured to use an Amazon RDS instance for MySQL. By using MySQL and MySQL RDS on AWS, you can store the sentiment analysis results and searched comments in a persistent database and retrieve them later for analysis and other stuff. I have host this web application on Steamlit and AWS.

## Skills used:

Python  <br>
Streamlit  <br>
Flask <br>
SQL <br>
MySQL <br>
AWS <br>
NLP <br>
Machine Learning <br>
Google API<br>


## Deployment:

The web application is hosted on an streamlit

### streamlit application - https://aryann29-youtube-comment-analyzer-stapp-fswa2l.streamlit.app/

web application is also hosted on an AWS server( not working rn)
aws =   http://ec2-13-48-49-211.eu-north-1.compute.amazonaws.com:8080/



## Database

Here's how MySQL database looks like  <br>

![alt text](https://user-images.githubusercontent.com/63531062/229336573-4e33c64a-6b58-4e1f-b754-c0aba5f34a04.png)
