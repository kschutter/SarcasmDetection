import numpy as np
import pandas as pd
from pandasql import sqldf
import os
import matplotlib.pyplot as plt
from datetime import datetime
plt.style.use('ggplot')

# Plot comments and votes over time, showcasing reddits rise in
# popularity
def plot_activity_over_time(df):

    # SQL requests to create tables with the data we need
    votes_over_time = """
                    Select
                        date,
                        SUM(abs(score)) AS total_votes
                    FROM df
                    GROUP BY date
                    ORDER BY date
                    """
    comments_over_time = """
                    Select
                        date,
                        COUNT(*) AS comment_count
                    FROM df
                    GROUP BY date
                    ORDER BY date
                    """
    sarcasm_over_time = """
                    Select
                        date,
                        COUNT(*) AS sarcasm_count
                    FROM df
                    WHERE label=1
                    GROUP BY date
                    ORDER BY date
                    """

    # Create the tables with the request strings
    df_votetime = sqldf(votes_over_time, locals())
    df_counttime = sqldf(comments_over_time, locals())
    df_sarcasmtime = sqldf(sarcasm_over_time, locals())

    # Plot our findings
    fix, ax = plt.subplots(1,1)
    ax.plot(df_votetime['date'], df_votetime['total_votes'])
    ax.plot(df_counttime['comment_count'])
    ax.legend(loc='upper left')
    xrange = np.array([0,24,48,72, 95])
    ax.set_xticks(xrange)

def plot_activity_over_week(df):

    #Convert the strings in the date column to datetime objects
    week_df = df[['label', 'created_utc']]
    def to_weekday(string):
        return datetime.strptime(string, '%Y-%m-%d %H:%M:%S').weekday()
    week_df['weekday'] = week_df['created_utc'].apply(lambda x: to_weekday(x))

    # SQL requests
    sarcasm_over_week = """
                    Select
                        weekday,
                        COUNT(*) AS sarcasm_count
                    FROM week_df
                    WHERE label=1
                    GROUP BY weekday
                    ORDER BY weekday
                    """
    comments_over_week = """
                    Select
                        weekday,
                        COUNT(*) AS comment_count
                    FROM week_df
                    GROUP BY weekday
                    ORDER BY weekday
                    """
    df_sarcasm_week = sqldf(sarcasm_over_week, locals())
    df_comment_week = sqldf(comments_over_week, locals())

    # Plot our findings
    fig, axs = plt.subplots(1,2, figsize=(10,3))
    axs[0].bar(df_sarcasm_week['weekday'], df_sarcasm_week['sarcasm_count'])
    axs[1].bar(df_comment_week['weekday'], df_comment_week['comment_count']);

if __name__=='__main__':
    df = pd.read_csv('../data/train-balanced-sarcasm.csv')

    plot_activity_over_time(df)
    plot_activity_over_week(df)
