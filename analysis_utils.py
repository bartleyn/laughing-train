import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
import scipy
from numpy.random import default_rng
rng = default_rng()


data_dir = ''

def gen_session_times_user(activity_df, num_per_day):
   activity_df.loc[:,'date_created'] = activity_df['date_created'].apply(lambda x: datetime.datetime.strptime(datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'), '%Y-%m-%d') )
   session_out = []
   days_covered = list(set([   datetime.datetime.strptime(datetime.datetime.strftime(x, '%Y-%m-%d'), '%Y-%m-%d') for x in pd.to_datetime(activity_df['date_created'].values) \
       if x >= datetime.datetime(2014,3,1) and x <= datetime.datetime(2014,9,1)]))
   return days_covered


def job(user):
    collated_tweet_df = pd.read_csv('{}/{}.csv'.format(data_dir, user), index_col=0).sample(frac=1).reset_index(drop=True) 
    activity_df = pd.read_csv('{}/{}_activity.csv'.format(data_dir, user), index_col=0).sample(frac=1).reset_index(drop=True) 

    collated_tweet_df['date_created'] = pd.to_datetime(collated_tweet_df['date_created'])
    tweet_dates = np.array([datetime.strftime(x, '%Y-%m-%d') for x in pd.to_datetime(collated_tweet_df['date_created'].values)])
    all_session_times = gen_session_times_user(activity_df, 5)
    activity_df['date_created'].fillna(datetime.datetime(1970,1,1), inplace=True)
    
    num_sessions = len(all_session_times)
    session_lengths = []
    tweets_seen = []
    sesh_list = []
    stats_list = []
    tot_tweets =0
    for session in sorted(all_session_times):

        
        mask = tweet_dates == (str(session)[:10])#(collated_tweet_df['date_created'].apply(lambda x: x  - session).astype(int) <= 0 )
        #locator = (collated_tweet_df[mask]['date_created'].apply(lambda x: x - session).astype(int)).abs().argsort()[:session_length]
        #sesh_df = collated_tweet_df.iloc[locator]
        sesh_df = collated_tweet_df[mask]
        num_sesh_tweets = len(sesh_df.index)
        session_lengths.append(num_sesh_tweets)
        tot_tweets += len(collated_tweet_df[mask].index)
    
        #print(session) 
        #print(list(zip(collated_tweet_df['date_created'].values, collated_tweet_df['date_created'].apply(lambda x: x - session).astype(int)))) 
    
        sesh_df.loc[:,'seeduser'] = [user] * num_sesh_tweets
        sesh_df.loc[:,'session'] = [session] * num_sesh_tweets
        sesh_df.loc[:,'ix'] = [x for x in range(1, num_sesh_tweets+1)]
        #print(sesh_df['date_created'])
        #sesh_df['time_exposed'] = sesh_df['date_created']#.fillna(datetime(1970,1,1)).apply(lambda x: x  + timedelta(minutes=15))
        sesh_list.append(sesh_df)
        #print(sesh_df)
        new_tweets_seen = len([x for x in sesh_df['tweet_id'].value_counts().index if x not in tweets_seen])
        stats_dict = {'new_tweets_seen': new_tweets_seen, \
                      'num_friends_seen':len([x for x in sesh_df['user_id'].value_counts().index ]), \
                      'total_unique_tweets_seen_until_now': len(set(tweets_seen)) + new_tweets_seen,
                      'total_unique_tweets_until_now': tot_tweets  } 

        tweets_seen.extend(sesh_df['tweet_id'].value_counts().index.tolist())
        
        #tweets_seen += [tweet_id for tweet_id in sesh_df['tweet_id'].value_counts().index if tweet_id not in tweets_seen]
        stats_list.append(stats_dict)
    return (user, stats_list, sesh_list)
    #print((user, sesh_list, stats_list))


def compute_gini(array, ):
    
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    try:
        if np.amin(array) < 0:
            # Values cannot be negative:
            array -= np.amin(array)
    except ValueError:
        pass
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    if n == 0:
        return np.nan
    else:
        return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))


def get_correlation_graph(corr_dicts, users, start, end, kind, corrs):
 

    fig, axarr = plt.subplots(1,1, figsize=(8,10))
    # for each length then make a boxplot for each user at that session length
    posn = 0
    colors = ['r', 'g', 'b', 'c']
    cix = 0
    boxes = []
    lngths = [10, 50, 100] + [2000000000]
    offset = 0.00
    markers = list(matplotlib.markers.MarkerStyle.markers.keys())
    ctr = 0
    cmap = matplotlib.colors.Colormap('tab10')
    
    
    # 1 for users 3 for tweets, 4 gini, 5 local bias
    val = 5
    if val == 5:
        title = 'Local Bias'
        ylabel = 'Bias'
    elif val == 4:
        title = 'Gini Coefficient'
        ylabel = 'Coefficient'
    elif val == 3:
        title = 'Frac. (+) Tweets Per Day'
        ylabel = 'Frac. (+) Tweets Per Day'
    elif val == 1:
        title = 'Frac. (+) Users Per Day'
        ylabel = 'Frac. (+) Users Per Day'
    for length in lngths:
        print(length)
        cur_corr_vals = []
        for corr in corrs:#, 0.75, 1.0]:
            print("corr {}".format(corr))
            frac_tweets_dict = corr_dicts[corr]
            
            if len(frac_tweets_dict) < 10:
                frac_tweets_dict = frac_tweets_dict[val -1]
            else:
                frac_tweets_dict = [x[val] for x in frac_tweets_dict if len(x) > 4]
                temp = {}
                for x in frac_tweets_dict:
                    temp.update(x)
                frac_tweets_dict = temp
                #frac_tweets_dict = [y for x in frac_tweets_dict for y in x]

            nums = []
            for user in users:
                if user not in frac_tweets_dict:
                    continue
                vals_to_analyze = frac_tweets_dict[user]
                #print(vals_to_analyze)
                if val < 4:
                    vals_to_plot = [x for x in vals_to_analyze[str(length)][0]]
                else:
                    vals_to_plot = [x for x in vals_to_analyze[str(length)]]

                if len(vals_to_plot) == 0:
                    continue
                nums.extend( [x for x in vals_to_plot if not np.isnan(x) and x >= 0.0] )

            cur_corr_vals.append((np.mean(nums), scipy.stats.sem(nums)))
            print(scipy.stats.sem(nums))
        axarr.fill_between(x=[corr for corr in corrs], \
                        y1=[tup[0]-tup[1] for tup in cur_corr_vals], \
                        y2=[tup[0]+tup[1] for tup in cur_corr_vals])
        axarr.plot([corr for corr in corrs], [tup[0] for tup in cur_corr_vals], marker=markers[ctr], \
                label='length {}'.format(length if length < 1000000 else 'entire', cmap=cmap))
        ctr += 1
        
    axarr.set_xlabel('Correlation', fontsize=20)
    axarr.set_ylabel('{}'.format(ylabel), fontsize=20)
    if val == 5:
        axarr.set_ylim([0,3.5])
    axarr.tick_params(axis='both', which='major', labelsize=12)
    plt.legend() 
    return fig, axarr



def get_correlation_graph_bars(corr_dicts, users, start, end, kind, corrs, num_samples):
 

    fig, axarr = plt.subplots(1,1, figsize=(8,10))
    # for each length then make a boxplot for each user at that session length
    posn = 0
    colors = ['r', 'g', 'b', 'c']
    cix = 0
    boxes = []
    lngths = [10, 50, 100] + [2000000000]
    offset = 0.00
    markers = list(matplotlib.markers.MarkerStyle.markers.keys())
    ctr = 0
    cmap = matplotlib.colors.Colormap('tab10')
    
    
    # 1 for users 3 for tweets, 4 gini, 5 local bias
    val = 5
    if val == 5:
        title = 'Local Bias'
        ylabel = 'Bias (a.u.)'
    elif val == 4:
        title = 'Gini Coefficient'
        ylabel = 'Coefficient'
    elif val == 3:
        title = 'Frac. (+) Tweets Per Day'
        ylabel = 'Frac. (+) Tweets Per Day'
    elif val == 1:
        title = 'Frac. (+) Users Per Day'
        ylabel = 'Frac. (+) Users Per Day'
    for length in lngths:
        print(length)
        cur_corr_vals = {}
        for corr in corrs:#, 0.75, 1.0]:
            print("corr {}".format(corr))
            frac_tweets_dict = corr_dicts[corr]
            cur_corr_vals[corr] = []
            if len(frac_tweets_dict) < 10:
                frac_tweets_dict = frac_tweets_dict[val -1]
            else:
                frac_tweets_dict = [x[val] for x in frac_tweets_dict if len(x) > 4]
                temp = {}
                for x in frac_tweets_dict:
                    temp.update(x)
                frac_tweets_dict = temp

            for sample in range(num_samples):
                nums = []
                user_sample = rng.choice(users, size=500)
                for user in user_sample:
                    if user not in frac_tweets_dict:
                        continue
                    
                    vals_to_analyze = frac_tweets_dict[user]
                    if val < 4:
                        vals_to_plot = [x for x in vals_to_analyze[str(length)][0]]
                    else:
                        vals_to_plot = [x for x in vals_to_analyze[str(length)]]

                    if len(vals_to_plot) == 0:
                        continue
                    nums.extend( [x for x in vals_to_plot if not np.isnan(x) and x >= 0.0] )

                cur_corr_vals[corr].append((np.mean(nums), scipy.stats.sem(nums)))
            print(scipy.stats.sem(nums))
        axarr.fill_between(x=[corr for corr in corrs], \
                        y1=[np.mean([x[0] for x in cur_corr_vals[corr]])-scipy.stats.sem([x[0] for x in cur_corr_vals[corr]]) for corr in cur_corr_vals], \
                        y2=[np.mean([x[0] for x in cur_corr_vals[corr]])+scipy.stats.sem([x[0] for x in cur_corr_vals[corr]]) for corr in cur_corr_vals])
        axarr.plot([corr for corr in corrs], [np.mean([x[0] for x in cur_corr_vals[corr]]) for corr in cur_corr_vals], marker=markers[ctr], \
                label='length {}'.format(length if length < 1000000 else 'entire', cmap=cmap))
        ctr += 1

    axarr.set_title('{}: {} vs Correlation - Resampled'.format(kind, title), fontsize=24)
    #axarr.set_xticklabels([str(x) for x in [10, 50, 100, 'all']], fontsize=14)
    axarr.set_xlabel('Correlation', fontsize=20)
    axarr.set_ylabel('{}'.format(ylabel), fontsize=20)
    axarr.tick_params(axis='both', which='major', labelsize=12)
    plt.legend() 
    return fig, axarr

    