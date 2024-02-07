import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from joblib import Parallel, delayed
import seaborn as sns
import analysis_utils as au
import datetime
import scipy
from scipy.sparse import lil_matrix, coo_matrix
from sklearn.decomposition import NMF

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.regularizers import l2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder





data_dir = '/nas/home/nbartley/data/Twitter Bias/Network Data/2014-2015-twitter-seed/indiv_users'
seeduser = pd.read_csv('{}/../seeduser.csv'.format(data_dir), header=0)
seeduser['screen_name'] = seeduser['screen_name'].apply(lambda x: x[1:-1])

session_length = 30

#condition = 'random'
condition = 'popularity'


import pickle as pkl

#with open('{}/intermediate_results_1_5k_rand.pkl'.format(data_dir), 'rb') as fp:
#    results = pkl.load(fp)

who_follows_who = pd.read_csv('{}/../usergraph.csv'.format(data_dir), header=0, lineterminator='\n')


map_friends = who_follows_who[['id', 'tid']].groupby(['id', 'tid']).count().index.tolist()
idx = pd.IndexSlice
#map_friends = {tup[0]:[x[1] for x in map_friends if x[0] == tup[0]] for tup in map_friends}

map_friends_2 = {}
for tup in map_friends:
    try:
        map_friends_2[tup[0]].append(tup[1])
    except:
        map_friends_2[tup[0]] = [tup[1]]

map_friends = map_friends_2

degrees_in = who_follows_who['id'].value_counts().to_dict()
degrees_out = who_follows_who['tid'].value_counts().to_dict()

#who_follows_who[who_follows_who['id'] == user]['tid'].value_counts().index.tolist()

if condition == 'correlated':
    probability_based_on_follows = who_follows_who['tid'].value_counts() / who_follows_who['tid'].value_counts().max()

wfw_friends = who_follows_who['tid'].value_counts().index.tolist()
wfw_followers = who_follows_who['id'].value_counts().index.tolist()
#map_friends_ix = {x: ix for ix, x in enumerate(list(set(wfw_friends)))}
#map_friends_ix[-1] = -1

users_with_most_sessions = [x for x in seeduser['user_id'].astype(int).values]
#ranked_order_df  = pd.read_csv('{}/../3900_seedusers_ranked_tweets_predictions.csv'.format(data_dir))

print("loading retweet counts")
if condition == 'popularity':
    ranked_by_retweets = pd.read_csv('{}/../retweet_counts.csv'.format(data_dir), names=['tweet_id', 'values'])#, dtype={'tweet_id': np.int64})
    ranked_by_retweets['tweet_id'].fillna(-1, inplace=True)
    ranked_by_retweets['tweet_id'] = pd.to_numeric(ranked_by_retweets['tweet_id'], errors='coerce')
    ranked_by_retweets['values'] = pd.to_numeric(ranked_by_retweets['values'], errors='coerce')
    #ranked_by_retweets['tweet_id'] = ranked_by_retweets.index


if condition == 'nnmf' or condition == 'neural' or condition == 'logit':
    print("entering models")
    nmf_model = NMF(n_components=5, init='random', random_state=0)
    rt_tracker = lil_matrix((5599, len(who_follows_who['tid'].value_counts().index)), dtype=np.int16)

    retweets = pd.read_csv('{}/../retweets_random_full.csv'.format(data_dir), header=0, on_bad_lines='skip', lineterminator='\n', nrows=3000000)
    retweets.loc[:,'user_id'] = pd.to_numeric(retweets['user_id'], errors='coerce')
    retweets.loc[:,'original_user_id'] = pd.to_numeric(retweets['original_user_id'], errors='coerce')

    print("loaded retweets")
    rts = retweets[['user_id', 'original_user_id', 're_tweet_id', 'retweet_time', 'original_time', 'original_tweet_id']]
    users_ix_dict = {usr: ix for ix, usr in enumerate(users_with_most_sessions)}
    friends_ix_dict = {fr: ix for ix, fr in enumerate(rts['original_user_id'].value_counts().index)}
    rts.loc[:,'user_id'] = rts['user_id'].apply(lambda x: users_ix_dict.get(x, -1) )
    rts.loc[:,'original_user_id'] = rts['original_user_id'].apply(lambda x: friends_ix_dict.get(x, -1))
    rts.loc[:,'original_tweet_id'] = rts['original_tweet_id'].apply(lambda x: x.strip("')(")).astype(int)
    rts.loc[:,'re_tweet_id'] = rts['re_tweet_id'].apply(lambda x: x.strip("'")).astype(int)
    print("past user id")
    #rts['retweet_time'] = pd.to_datetime(rts['retweet_time'], errors='coerce')
    #rts['original_time'] = pd.to_datetime(rts['original_time'], errors='coerce')

    print("past times")

    #for usr in rts['user_id'].value_counts().index:
    #    rt_tracker[usr, rts[rts['user_id'] == usr]['original_user_id'].values] += 1


    
    usrs = rts['user_id'].value_counts().index.tolist()

    unique_user_rts = [rts[rts['user_id']  == usr]['original_user_id'].value_counts().index.tolist() for usr in usrs]
    temp_df = pd.DataFrame({"features":  list(unique_user_rts)})
    matrix = temp_df['features'].apply(pd.value_counts).fillna(0).astype(int)
    #ids = list(usrs)
    #matrix.index = ids
    #ids = list(unique_user_rts)
    #matrix.index = ids
    #matrix = matrix.reindex(sorted(matrix.columns), axis=1)


    rt_tracker = lil_matrix(coo_matrix(matrix))
    

    #usrs = rts['user_id'].value_counts().index
    #nums = [rts[rts['user_id'] == usr]['original_user_id'].value_counts() for usr in usrs]
    #rt_tracker[usrs, [rts[rts['user_id'] == usr]['original_user_id'].values.tolist() for usr in usrs]] += 1
    print("usrs ", usrs)
    print("matrix ", matrix)
    print("rt tracker shape, ", rt_tracker.shape)
    #print(get logistic regression in )
    if condition == 'nnmf':
        nmf_W = nmf_model.fit_transform(rt_tracker)
        nmf_H = nmf_model.components_
        nmf_rec = nmf_W.dot(nmf_H)
        print(nmf_rec.shape)
    elif condition == 'neural':
        # Assume you have a DataFrame 'df' with columns 'user_id', 'item_id', 'interaction' (1 if interaction, 0 otherwise),
        # and additional features such as 'time_since_post', 'user_followers', 'item_popularity', etc.

        # Map user and item IDs to contiguous integer values
        user_mapping = {user_id: idx for idx, user_id in enumerate(retweets['user_id'].unique())}
        item_mapping = {item_id: idx for idx, item_id in enumerate(retweets['original_user_id'].unique())}

        retweets['user_idx'] = retweets['user_id'].map(user_mapping)
        retweets['item_idx'] = retweets['original_user_id'].map(item_mapping)

        # Split the data into training and testing sets
        train_data, test_data = train_test_split(retweets[['user_idx', 'item_idx', 'interaction', 'time_since_post', 'user_followers', 'item_popularity']], test_size=0.2, random_state=42)

        # Number of unique users and items
        num_users = len(user_mapping)
        num_items = len(item_mapping)

        # Embedding size for user and item vectors
        embedding_size = 32

        # Additional features
        num_features = len(train_data.columns) - 3  # Exclude 'user_idx', 'item_idx', and 'interaction'
        feature_input = Input(shape=(num_features,), name='feature_input')

        # User embedding layer
        user_input = Input(shape=(1,), name='user_input')
        user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size)(user_input)
        user_embedding = Flatten()(user_embedding)

        # Item embedding layer
        item_input = Input(shape=(1,), name='item_input')
        item_embedding = Embedding(input_dim=num_items, output_dim=embedding_size)(item_input)
        item_embedding = Flatten()(item_embedding)

        # Concatenate user and item embeddings with additional features
        concatenated = Concatenate()([user_embedding, item_embedding, feature_input])

        # Neural Collaborative Filtering model with additional features
        hidden_layer = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(concatenated)
        dropout_layer = Dropout(0.2)(hidden_layer)
        output_layer = Dense(1, activation='sigmoid')(dropout_layer)

        # Compile the model
        model = Model(inputs=[user_input, item_input, feature_input], outputs=output_layer)
        model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=[AUC()])

        # Train the model
        model.fit([train_data['user_idx'], train_data['item_idx'], train_data.iloc[:, 3:]], train_data['interaction'], epochs=5, batch_size=64, validation_split=0.2)

        # Evaluate the model on the test set
        test_loss, test_auc = model.evaluate([test_data['user_idx'], test_data['item_idx'], test_data.iloc[:, 3:]], test_data['interaction'])
        print(f'Test Loss: {test_loss}, Test AUC: {test_auc}')


'''


'''


'''
    nmf_model = NMF(n_components=4, init='random', random_state=0)
    self.W = nmf_model.fit_transform(self.likes_tracker)
    self.H = nmf_model.components_
    self.rec = self.W.dot(self.H)
    likes_tracker == 5599 x |G| mat
'''
global model_bk
model_bk = None
def job(user):
    if condition == 'random':
        collated_tweet_df = pd.read_csv('{}/{}.csv'.format(data_dir, user), index_col=0, dtype={'RT': str}).sample(frac=1).reset_index(drop=True)
        activity_df = pd.read_csv('{}/{}_activity.csv'.format(data_dir, user), index_col=0, dtype={'RT': str}).sample(frac=1).reset_index(drop=True) # personal activity
    if condition == 'correlated':

        collated_tweet_df = pd.read_csv('{}/{}.csv'.format(data_dir, user), index_col=0, dtype={'RT': str})
        try:
            collated_tweet_df = collated_tweet_df.sample(frac=1, weights=[probability_based_on_follows[x] if x in probability_based_on_follows else 0.00001 for x in collated_tweet_df['user_id'].values]).reset_index(drop=True) 
        except ValueError as e:
            print("collated_df ", e, user, len(collated_tweet_df))
        activity_df = pd.read_csv('{}/{}_activity.csv'.format(data_dir, user), index_col=0, dtype={'RT': str})
        try:
            activity_df = activity_df.sample(frac=1, weights=[probability_based_on_follows[x] if x in probability_based_on_follows else 0.00001 for x in activity_df['user_id'].values]).reset_index(drop=True)
        except ValueError as e:
            print("act_df ", e, user, len(activity_df))
            return (user, [], [])
    else:
        collated_tweet_df = pd.read_csv('{}/{}.csv'.format(data_dir, user), index_col=0, dtype={'RT': str})
        activity_df = pd.read_csv('{}/{}_activity.csv'.format(data_dir, user), index_col=0, dtype={'RT': str})
    #collated_tweet_df['tweet_id'] = pd.to_numeric(collated_tweet_df['tweet_id'], errors='coerce')
    
    collated_tweet_df['date_created'] = pd.to_datetime(collated_tweet_df['date_created'])
    collated_tweet_df['date_created'] = collated_tweet_df['date_created'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d'))

    if condition == 'random_once':
        # For ranking randomly
        # for every day
        # for ever friend on that day
        # take one random tweet from that friend on that day

        #print(collated_tweet_df['date_created'])
        collated_tweet_days = collated_tweet_df['date_created'].values
        tweet_dates = np.unique(np.array([datetime.datetime.strftime(x, '%Y-%m-%d')for x in pd.to_datetime(collated_tweet_days)]))
        daily_tweet_df_list = []
        #print(tweet_dates)
        for day in tweet_dates:
            tweets_on_day = collated_tweet_df[collated_tweet_df['date_created'] == day]
            friends_on_day = tweets_on_day['user_id'].value_counts().index.tolist()
            one_tweet_per_friend = [tweets_on_day[tweets_on_day['user_id'] == friend].sample(n=1) for friend in friends_on_day]
            daily_tweet_df_list.extend(one_tweet_per_friend)
        try:

            collated_tweet_df = pd.concat(daily_tweet_df_list)
        except ValueError as e:
            print(e, user)
            return (user, [], [])
    #print(user, len(collated_tweet_df.index))
    if condition == 'revchron':
        collated_tweet_df = collated_tweet_df.iloc[::-1]



    if condition == 'popularity':
        ## for ranking by popularity
        joined_df = collated_tweet_df.merge(ranked_by_retweets, how='left', on='tweet_id')
        joined_df['values'].fillna(0, inplace=True)
        joined_df.sort_values(by=['values'], ascending=False, inplace=True)

        collated_tweet_df = joined_df
    
    if condition == 'nnmf':
        try:
            rec_comp = nmf_rec[usrs.index(user), :]
        except ValueError as e:
            rec_comp = None
        if rec_comp is not None:
            print('rc ', rec_comp.shape)
            #collated_tweet_df sort based on user id in the rec comp
            collated_tweet_df_users = collated_tweet_df['user_id']
            sorted_ix = np.argsort([rec_comp[friends_ix_dict.get(usr,-1)] for usr in collated_tweet_df_users])[::-1]
            collated_tweet_df = collated_tweet_df.iloc[sorted_ix]
            #tweets_seen = sorted(tweets_seen, key = lambda x: rec_comp[model.map_user_id[x[0]]], reverse=True)
        

    elif condition == 'logit':
        global model_bk
        

        #I have collated tweet data and the actual retweets the users did 
        # current_time - original_time, original_user_id, if hashtag, retweet_yn
        # Sample DataFrame
        all_user_retweets = activity_df[activity_df['RT'] == 'RT']
        found_rts_users = rts[rts['re_tweet_id'].isin(all_user_retweets['tweet_id'])][['original_user_id', 'retweet_time', 'original_time', 'original_tweet_id']]

        found_rts_users['friend_deg'] = found_rts_users['original_user_id'].apply(lambda x: degrees_out[x] if x in degrees_out else -1)
        found_rts_users['friend_bin'] = found_rts_users['original_user_id'].apply(lambda x: 1 if user in map_friends and x in map_friends[user] else 0)
        #found_rts_users['retweet_time'] = pd.to_datetime(found_rts_users['retweet_time'])
        #found_rts_users['original_time'] = pd.to_datetime(found_rts_users['original_time'])
        found_rts_users['RT'] = [1 for x in found_rts_users.index]
        #found_rts_users['timedelta'] = (found_rts_users['retweet_time'] - found_rts_users['original_time']).dt.total_seconds()

        all_friend_tweets = collated_tweet_df[['user_id', 'date_created', 'tweet_id']]
        all_friend_tweets['tweet_id'] = all_friend_tweets['tweet_id'].astype(int)
        #all_friend_tweets['original_time'] = pd.to_datetime(all_friend_tweets['date_created'])
        
        frs_with_rts = all_friend_tweets[all_friend_tweets['tweet_id'].isin(found_rts_users['original_tweet_id'])].index
        all_friend_tweets['friend_deg'] = all_friend_tweets['user_id'].apply(lambda x: degrees_out[x] if x in degrees_out else -1)
        all_friend_tweets['friend_bin'] = all_friend_tweets['user_id'].apply(lambda x: 1 if x in map_friends[user] else 0)
    
        all_friend_tweets['RT'] = [0 for x in all_friend_tweets.index]
        all_friend_tweets.loc[frs_with_rts, 'RT'] = 1
        #print("frs with rts ", frs_with_rts, " and all frs ", all_friend_tweets['RT'].describe())
        
        df = pd.concat([found_rts_users, all_friend_tweets])[['friend_deg', 'friend_bin', 'RT']]


        #print("full df rt vc ", df['RT'].value_counts())
        features = ['friend_deg', 'friend_bin']  # Add other relevant features
        X = df[features]
        y = df['RT']

        # Splitting data into training and testing sets
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Training the logistic regression model
            model = LogisticRegression()

            model.fit(X_train, y_train)
        except ValueError as e:
            print("user {} had error {}".format(user, e))
            model = model_bk
        
        if model_bk is None:
            model_bk = model

        # Predicting and evaluating the model
        #y_pred = model.predict(X_test)
        #try:
        #    accuracy = accuracy_score(y_test, y_pred)
        #    roc_auc = roc_auc_score(y_test, y_pred)

        #    print(f"Accuracy: {accuracy}")
        #    print(f"ROC-AUC: {roc_auc}")

        # Assuming 'new_tweets' has the same structure as your training data


        # Make predictions
        try:
            predictions = model.predict_proba(all_friend_tweets[['friend_deg', 'friend_bin']])[:,1]
        except ValueError as e:
            predictions = [1.0 for x in collated_tweet_df.index]
        #print(predictions)
        # The 'predictions' variable now contains the predicted probabilities of engagement for each tweet
        # You can add these probabilities to your DataFrame for further analysis or ranking
        collated_tweet_df['predicted_prob_engagement'] = predictions

        # Now you can rank tweets based on the predicted probabilities
        collated_tweet_df = collated_tweet_df.iloc[np.argsort(predictions)[::-1]]

    collated_tweet_df2 = collated_tweet_df
    #collated_tweet_df2['date_created'] = pd.to_datetime(collated_tweet_df2['date_created'])
    tweet_dates = collated_tweet_df2['date_created'].values

        


        


    #ranked_tweets_user = pd.read_csv('{}/{}_predictions.csv'.format(data_dir, user), index_col=0, dtype={'RT':str}).sort_values(by=['RT', 'Tweet'], ascending=False)

    #collated_tweet_df['tweet_id'] = collated_tweet_df['tweet_id'].apply(lambda x: x.strip("'\n"))
    #df.index.intersection(row_indices)
    '''
    print(len([x for x in ranked_tweets_user['Tweet'].apply(lambda x: int(x.strip("'\n"))) if x in collated_tweet_df['tweet_id']]))
    collated_tweet_df = collated_tweet_df.set_index('tweet_id')
    try:
        collated_tweet_df2 = collated_tweet_df.loc[[x for x in ranked_tweets_user['Tweet'].apply(lambda x: int(x.strip("'\n"))).values if x in collated_tweet_df.index]]#collated_tweet_df.loc[ranked_tweets_user['Tweet'].apply(lambda x: int(x.strip("'\n")))]
        collated_tweet_df2.reset_index(inplace=True)
    except pd.errors.InvalidIndexError:
        print("user {} failed".format(user))
        return (user, None, None)

    print(len([x for x in ranked_tweets_user['Tweet'] if x in collated_tweet_df.index]))
    print(collated_tweet_df2['tweet_id'].value_counts())
    #.apply(lambda x: int(x.strip("'")))
    #collated_tweet_df = collated_tweet_df[collated_tweet_df['tweet_id'].isin(ranked_tweets_user)]
    '''

    #collated_tweet_df['date_created'].fillna(datetime(1970,1,1), inplace=True)
    #print(collated_tweet_df)
    
    all_session_times = au.gen_session_times_user(activity_df, 5)
    #activity_df['date_created'] = activity_df['date_created'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S') )
    activity_df['date_created'].fillna(datetime.datetime(1970,1,1), inplace=True)
    
    num_sessions = len(all_session_times)
    session_lengths = []
    tweets_seen = []
    sesh_list = []
    stats_list = []
    tot_tweets =0
    #print(user, " ALL SESSION TIMES ", len(all_session_times))
    for session in sorted(all_session_times):

        if session < datetime.datetime(2014,3,1) or session > datetime.datetime(2014,7,9):
            continue
        #print(collated_tweet_df['date_created'], type(session))
        #raise Exception
  
        #print(tweet_dates == str(session)[:10])
        #raise Exception
        

        #mask = (tweet_dates <= (str(session)[:10])) & (tweet_dates >= (str(session - timedelta(days=7))[:10]))# in [str(x)[:10] for x in [session - timedelta(days=num) for num in range(1,5)]]
        mask = tweet_dates == (str(session)[:10])#(collated_tweet_df['date_created'].apply(lambda x: x  - session).astype(int) <= 0 )
        #locator = (collated_tweet_df[mask]['date_created'].apply(lambda x: x - session).astype(int)).abs().argsort()[:session_length]
        #sesh_df = collated_tweet_df.iloc[locator]
        if (isinstance(mask, bool)): 
            continue
        sesh_df = collated_tweet_df2[mask].copy()
       
        num_sesh_tweets = len(sesh_df.index)
        session_lengths.append(num_sesh_tweets)
        tot_tweets += len(collated_tweet_df2[mask].index)
    
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

print(users_with_most_sessions[0])
print(job(users_with_most_sessions[0])[2][:5])
#raise Exception




import joblib

with joblib.parallel_backend(backend="loky"):
    parallel = Parallel(verbose=100, n_jobs=20,  temp_folder="/nas/home/nbartley/data/" )
    #for ct_df, ct_df2, user in results:
    #    ct_df.to_csv('{}/{}.csv'.format(data_dir, user))
    #    ct_df2.to_csv('{}/{}_activity.csv'.format(data_dir, user))


    user_data = parallel(delayed(job)(user) for user in users_with_most_sessions)#, stats, seshs, lens in users_with_most_sessions)
    
    print("processed all user sessions")

    with open('{}/random_sesh_lists.pkl'.format(data_dir), 'wb') as fp:
        pkl.dump(user_data, fp)



    print("loaded follow graph")
    #print(user_data[0])
    #raise Exception
    def job(user_info, friends):
        gini_dict = {}
        session_lengths = list(np.arange(10,150,10))

        user = user_info[0]
        stats_list = user_info[1]
        sesh_list = user_info[2]
        if stats_list is None or sesh_list is None:
            return (None, None)
        #friends = who_follows_who[who_follows_who['id'] == user]['tid'].value_counts().index.tolist()
        num_friends = len(friends)
        days = len([x for x in sesh_list if len(x) > 0])
        print("num days: {}".format(days))
        days_list = [str(x['date_created'].values[0])[:10] for x in sesh_list if len(x) > 0]
        #days_list = [datetime.strptime(datetime.strftime(pd.to_datetime(x['date_created'].values[0]), '%Y-%m-%d'), '%Y-%m-%d') for x in sesh_list if len(x) > 0]

        if user not in gini_dict:
            gini_dict[user] = {}
        act_dfs = []
        #list(np.arange(10,150,10))+ [2000000000]
        for length in session_lengths + [2000000000]:
            if length in gini_dict[user]:
                continue
            gini_dict[user][length] = {}
            zeros = scipy.sparse.lil_matrix((num_friends, days))#scipy.sparse.eye(len(friends), days) - scipy.sparse.eye(len(friends), days)
            #activity_dict[user] = zeros
            act_df = zeros
            #print(user)

            for sessionfull, day in zip(sesh_list, days_list):
                    
                    cur_day = days_list.index(day)
                    #print(day)
                    #act_df = activity_dict[user]   
                    
                    session = sessionfull[sessionfull['ix'] <= length]

                    num_obs_tweets = session[session['user_id'].isin(friends)]['user_id'].value_counts()#session[['user_id']].groupby(['user_id']).count()
                    mask = num_obs_tweets.index.map(lambda x: friends.index(x))
                
                    
    
                    #mask = [x for x in mask if x is not None]
                    try:
                        act_df[:, cur_day] += np.array([num_obs_tweets[x] if x in num_obs_tweets.index  else 0 for x in friends ]).reshape(len(friends), 1 ) #num_obs_tweets[mask].values
                    except KeyError as e:
                        print(e, mask)
                        pass
                    except ValueError as e:
                        print(e, "act_df shape {} act_df size {} mask, cur_day size {} num_obs_tweets {} ".format(\
                            act_df.shape, act_df.size, act_df[mask, cur_day].size,  num_obs_tweets.size))
                        pass
                    #raise Exception
                    '''
                    for user_obs in friends:#act_df['user_id'].value_counts().index:
                        #num_obs_tweets = leah_df[leah_df['user_seen'] == user_obs]['obs_likes'].sum()
                        #num_obs_tweets = len(leah_df[leah_df['user_seen'] == user_obs].index)

                        #every day/session has only  unique tweet ids since they're divded by day
                        num_obs_tweets = len(session[session['user_id'] == user_obs].index)
                        
                        
                        try:
                            act_df[friends.index(user_obs), cur_day] += num_obs_tweets
                        except KeyError:
                            print(user_obs, " leah1")
                            pass
                    '''

                    try:
                        gini = au.compute_gini(np.array([x for x in act_df[:,cur_day].sum(axis=1) if x > 0. ]))
                    except ZeroDivisionError:
                        #print("divide by 0")
                        gini = np.nan
                    gini_dict[user][length][day] = gini
            act_dfs.append(act_df)
        return (gini_dict, act_dfs)
        

    users_to_process = user_data
    user_friends = [who_follows_who[who_follows_who['id'] == user[0]]['tid'].value_counts().index.tolist() for user in users_to_process]

    job(users_to_process[0], user_friends[0])
    #raise Exception


    import joblib
    import scipy

    #with joblib.parallel_backend(backend="loky"):
    #parallel = Parallel(verbose=100, n_jobs=8, require='sharedmem', temp_folder="/nas/home/nbartley/data/")#,require='sharedmem')
    #users_to_process = user_data[:750]#[user for user in seeduser['user_id'].values][:1000]#if int(user) >= 14078815]
    #qrtt = job(users_to_process[0])
    results = parallel(delayed(job)(user, frns) for user, frns in zip(users_to_process, user_friends))
    #for ct_df, ct_df2, user in results:
    #    ct_df.to_csv('{}/{}.csv'.format(data_dir, user))
    #    ct_df2.to_csv('{}/{}_activity.csv'.format(data_dir, user))


gini_dict = {}
act_dfs_dict = {}
for result in results:
    if result[0] is None:
        continue
    username = list(result[0].keys())[0]
    gini_dict[username] = result[0][username]
    act_dfs = result[1]
    act_dfs_dict[username] = act_dfs

with open('{}/gini_tuple_all_new_{}_full_july_actual.pkl'.format(data_dir, condition), 'wb') as fp:
    pkl.dump((gini_dict, act_dfs_dict), fp)



# %%
