import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

class HeadlineAnalyzer:
    def publication_freq_plot(self,df):
        # Extract day of the week, hour, and month
        df['day_of_week'] = df['date_only'].dt.day_name()
        df['hour'] = df['time_only'].apply(lambda x: x.hour if pd.notnull(x) else None)
        df['month'] = df['date_only'].dt.month_name()

        # Count articles by day of the week
        day_of_week_counts = df['day_of_week'].value_counts()

        # Count articles by hour
        hour_counts = df['hour'].value_counts().sort_index()

        # Count articles by month
        month_counts = df['month'].value_counts().reindex([
            'January', 'February', 'March', 'April', 'May', 'June', 
            'July', 'August', 'September', 'October', 'November', 'December'
        ], fill_value=0)

        # Plotting publication frequency
        plt.figure(figsize=(16, 15))

        # Plot day of the week
        plt.subplot(3, 1, 1)
        day_of_week_counts.plot(kind='bar', color='lightgreen')
        plt.title('Articles Published by Day of the Week')
        plt.xlabel('Day of the Week')
        plt.ylabel('Number of Articles')

        # Plot hour of the day
        plt.subplot(3, 1, 2)
        hour_counts.plot(kind='bar', color='skyblue')
        plt.title('Articles Published by Hour of the Day')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Number of Articles')

        # Plot month of the year
        plt.subplot(3, 1, 3)
        month_counts.plot(kind='bar', color='salmon')
        plt.title('Articles Published by Month')
        plt.xlabel('Month')
        plt.ylabel('Number of Articles')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()
    
    def sentiment_analysis(self,data):
        sentiment_data=data.copy()
        sia = SentimentIntensityAnalyzer()
        # Calculate the sentiment of the headlines
        data.loc[:,'compound']=data['headline'].apply(lambda x: sia.polarity_scores(text=x)['compound'] )
        
        def categorize_sentiment(score):
            if score >= 0.6:
                return 'Very Positive'
            elif score >= 0.2:
                return 'Positive'
            elif score > -0.2:
                return 'Neutral'
            elif score > -0.6:
                return 'Negative'
            else:
                return 'Very Negative'

        # Apply categorization
        data.loc[:,'sentiment'] = data['compound'].apply(categorize_sentiment)
        
        # Count of each sentiment
        sentiment_counts = data['sentiment'].value_counts()
        print(sentiment_counts)
        
        # Plotting sentiment distribution
        plt.figure(figsize=(8, 6))
        sentiment_counts.plot(kind='bar', color=['lightgreen', 'lightcoral', 'lightgrey'])
        plt.title('Sentiment Distribution of Headlines')
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Headlines')
        plt.show()