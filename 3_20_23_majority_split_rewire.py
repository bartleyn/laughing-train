import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from joblib import Parallel, delayed
import seaborn as sns
import analysis_utils as au
import datetime
import pickle as pkl

import joblib 
import os
import sys

from joblib import dump, load

import time
import scipy.sparse

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


data_dir = ''
seeduser = pd.read_csv('{}/../seeduser.csv'.format(data_dir), header=0)
seeduser['screen_name'] = seeduser['screen_name'].apply(lambda x: x[1:-1])

session_length = 30

lngths = list(np.arange(10,150,10)) + [2000000000]
split_done = False
#all_new_popularity_3, all_new_random, all_new_revchron_2
#random_sample = np.random.choice([user for user in seeduser['user_id'].values], size=500, replace=False)
random_sample = [user for user in seeduser['user_id'].values]

kind = 'revchron'

if not split_done:
    with open('{}/gini_tuple_all_new_revchron_full_july_actual.pkl'.format(data_dir, kind), 'rb') as fp:##with open('{}/gini_tuple_all_new_popularity_3.pkl'.format(data_dir), 'rb') as fp:
        gini_tup = pkl.load(fp)
    #random_sample = list(gini_tup[1].keys())#np.random.choice(list(gini_tup[1].keys()), replace=False, size=1000))

if kind == 'popularity' or kind == 'random' or kind == 'revchron' or kind == 'logit' or kind == 'nnmf':
    attn_func = lambda edge: (degrees_out[edge[1]] / max(1, map_deg_friends[edge[0]])) if edge[1] in degrees_out and edge[0] in map_deg_friends else 0.0
elif kind == 'random' or kind == 'revchron':# or kind=='popularity':
    attn_func = lambda edge: 1.0/degrees_in[edge[0]]


try:
    with open('{}/days_to_keep_dict.pkl'.format(data_dir), 'rb') as fp:
        days_to_keep_dict = pkl.load(fp)
        loaded_days = True
    with open('{}/frac_friend_dicts.pkl'.format(data_dir), 'rb') as fp:
        loaded_pkl = pkl.load(fp)
        frac_dict = loaded_pkl[1]
        friends_dict = loaded_pkl[3]
        loaded_dicts = False


except Exception as e:
    pass


who_follows_who = pd.read_csv('{}/../usergraph.csv'.format(data_dir), header=0, lineterminator='\n')

map_friends = who_follows_who[['id', 'tid']].groupby(['id', 'tid']).count().index.tolist()
idx = pd.IndexSlice

map_friends_2 = {}
for tup in map_friends:
    try:
        map_friends_2[tup[0]].append(tup[1])
    except:
        map_friends_2[tup[0]] = [tup[1]]

map_friends = map_friends_2

del map_friends_2

from numpy.random import default_rng
rng = default_rng(seed=42)

wfw_id = who_follows_who['id'].value_counts().index.tolist()
wfw_tid = who_follows_who['tid'].value_counts().index.tolist()
total_users = list(set(wfw_id + wfw_tid))


act_dfs_dict = gini_tup[1]
try:
    with open('{}/edge_dict_{}.pkl'.format(data_dir, kind), 'rb') as fp:
        edge_dict = pkl.load(fp)
except Exception as e:
    edge_dict = {length: [] for length in lngths}
    for user in [x for x in act_dfs_dict if x in map_friends]:
        
        for act_df, length in zip(act_dfs_dict[user], lngths):
            friends_seen = np.array([x > 0 for x in act_df.sum(axis=1)])
            edge_dict[length].extend([( user, fr) for fr, seen in zip(map_friends[user], friends_seen) if seen == True])
    with open('{}/edge_dict_{}.pkl'.format(data_dir, kind), 'wb') as fp:
        pkl.dump(edge_dict, fp)
        


# RANDOM ASSIGNMENT

true_prevalence = 0.05
vals = rng.binomial(n=1, p=true_prevalence, size=len(total_users))
user_map = {user:val for user, val in zip(total_users, vals)}


map_friends_scores = {x:[user_map[y] for y in map_friends[x]] for x in map_friends}



#gini_dict = gini_tup[0]


def job(act_dfs, ginis,  user):
    days_to_keep_dict = {user:{y:[] for y in lngths} }
    
    #act_dfs = gini_tup[1][user]
    #ginis = gini_tup[0][user]
    
    for ln, act_mat in zip(lngths, act_dfs):
        days = sorted(ginis[ln].keys())
        days_to_keep = []
        for day in range(act_mat.get_shape()[1]):
            if act_mat[:,day].nnz >= 2:
                days_to_keep.append(days[day])
        days_to_keep_dict[user][ln] = days_to_keep
    return days_to_keep_dict

def corr(x,y, um):
    avg_pos_degree = np.mean([degrees_out[x] if x in degrees_out else 0.0 for x in um ])
    #avg_degree = np.mean(x)
    #std_deg = np.std(x)
    std_vals = np.std(y)
    p_pos = np.sum(y) / float(len(y))
    print((p_pos / (std_deg * std_vals), np.abs(avg_pos_degree - avg_degree))) 
    return (p_pos / (std_deg * std_vals)) * np.abs(avg_pos_degree - avg_degree)

degrees_in = who_follows_who['id'].value_counts().to_dict()
degrees_out = who_follows_who['tid'].value_counts().to_dict()
G_in = degrees_in
G_out = degrees_out
degree_dist = [degrees_out[x] if x in degrees_out else 0 for x in user_map]
assignments = [user_map[x] for x in user_map]

avg_degree = np.mean(degree_dist)
std_deg = np.std(degree_dist)



with joblib.parallel_backend(backend="loky"):
    if not loaded_days:

        parallel = Parallel(verbose=100, n_jobs=14, temp_folder="/nas/home/nbartley/data/" )

        folder = '{}/joblib_memmap'.format(data_dir)
        try:
            os.mkdir(folder)
        except FileExistsError:
            pass

        print("entering job")

        days_to_keep_dicts = parallel(delayed(job)(gini_tup[1][user], gini_tup[0][user], user) for user in random_sample)


if not loaded_days:
    days_to_keep_dict = {}
    for x in days_to_keep_dicts:
        days_to_keep_dict.update(x)
    
def local_bias(edges, user_map, true_prev, day, G_in, G_out, attn_func = None):
    in_deg = G_in
    out_deg = G_out
    if attn_func is None:
        attn_func = lambda edge: (1.0/in_deg(edge[1])) if in_deg(edge[1]) > 0 else 0.0
    num_edges = len(edges)
    exp_val = 0.0
    vals = map(lambda x: user_map[x[1]] * attn_func(x) / num_edges, edges)
    exp_val = np.nansum(list(vals))

    return np.mean(list(dict(in_deg).values())) * exp_val - true_prev



print("done with gini loading etc")



## some over the days
## any friends who are positive are links 
## make a dict that are the links 



np_nonzero = np.nonzero
np_sum = np.sum
np_unique = np.unique
# stack each users act_dfs then get the size of each users and then stack all them 


from multiprocessing import current_process
import time
print(time.time())
#print({x:user_map[x] for x in map_friends.loc[idx[12, :]].index.tolist()})
def job(user_chunk):#(days_list, friends, act_dfs, map_friends, user_map, user):

    frac_dict = {}
    friends_dict = {}
    frac_tweets_dict = {}
    local_bias_vals = {}
    gini_dict = {}
    bias_dict = {}
    idx = pd.IndexSlice
    for user in user_chunk:    
        t0 = time.time()
        try:
            friends = {x:[] for x in map_friends[user]}#wfw_dict[user]
            friends_scores = map_friends_scores[user]
        except Exception as e:
            print("no friends for user ", user)
            continue
            return user, {}, {}, {}#frac_dict, friends_dict, frac_tweets_dict
        #act_dfs =  gini_tup[1][user]
        with open('{}/activity_dfs_{}_{}.pkl'.format(data_dir, kind, user), 'rb') as fp:
            act_dfs = pkl.load(fp)


        friends_list = np.array(list(friends.keys()))
        

        



        
        for friend in friends:
            try:
                friend_neighbors = map_friends[friend]#map_friends_xs(int(friend), level=0)#map_friends.loc[idx[int(friend),:]].index.tolist()#who_follows_who_mm[who_follows_who_mm['id'] == friend]['tid'].value_counts().index.tolist()
                friend_neighbors_scores = map_friends_scores[friend]
            except Exception as e:
                #print(e)
                continue
            #neighbor_pol_scores = [user_map[neighbor] for neighbor in friend_neighbors]
            len_neighbs = float(len(friend_neighbors))
            friends_dict[friend] = sum(friend_neighbors_scores) / len_neighbs if len_neighbs > 0 else -1
        

        t1 = time.time()
        #logging.debug("Child {process.name} t1 after friends {}".format(t1 - t0))
        t0 = t1

        total_friend_pos_frac = sum(friends_scores) / float(len(friends)) if len(friends) > 0 else -1 
        frac_dict[user] = {'whole': total_friend_pos_frac}
        frac_tweets_dict[user] = {}
        gini_dict[user] = {}
        bias_dict[user] = {}

        t1 = time.time()
        #logging.debug("Child {process.name} t1 before lengths {}".format(t1 - t0))
        t0 = t1
        
    
        #local_bias(edges, user_map, true_prev, day, G_in, G_out, attn_func = None)
        
        for length, act_df, act_df_bin in zip(list(np.arange(10,150,10))+[2000000000], act_dfs, [act_df.copy() for act_df in act_dfs]):
            
            local_bias_vals = local_bias(edge_dict[length], user_map, true_prevalence, None, G_in, G_out, attn_func)

            days_list = [x for x in range(act_df.shape[1])]#x for x in gini_dict[user][length]]# if len(x) >0]
            num_days = len(days_list)
            str_len = str(length)
            frac_dict[user][str_len] = []
            frac_tweets_dict[user][str_len] = []
            gini_dict[user][str_len] = []
            bias_dict[user][str_len] = []

            gini_vals = np.array([au.compute_gini(act_df[:,day].astype('float64').todense()) for day in range(act_df.shape[1])])
            bias_vals = np.array([local_bias_vals for day in range(act_df.shape[1])])
            
            index = np.arange(np.shape(act_df)[0])
            rng.shuffle(index)
            act_df = act_df[index,:]
            act_df_bin = act_df_bin[index,:]
            
            act_df_bin[act_df_bin != 0] = 1

            friends_seen_total = np.asarray(np_nonzero(act_df_bin))

            days_gte_10 = (np.sum(act_df, axis=0) >= 10.0).tolist()[0]
            #print(len(days_gte_10), days_gte_10)

            #friends_seen_subset = [x for x in friends_seen_total[0,:] if x in friends]#f
            friends_seen_subset = np.unique(friends_seen_total[0,:])#[x in friends for x in friends_seen_total[0,:]]]
            friends_seen_sb_list = friends_list[friends_seen_subset]
            fs_tr = [user_map[friend] for friend in friends_seen_sb_list]
            where_fs_eq_1 = np_unique([friend for friend, tr in zip(friends_seen_subset, fs_tr) if tr == 1])
            
            #print(user, len(where_fs_eq_1), act_df_bin.shape)
            #print(user, (np.sum(act_df_bin[where_fs_eq_1,:], axis=0) / (np.sum(act_df_bin, axis=0) +1.0)).flatten().tolist(), np.sum(act_df[where_fs_eq_1,:], axis=0) / (np.sum(act_df, axis=0) + 1.0).flatten().tolist() )
            if len(where_fs_eq_1) == 0:
                sesh_friends_pos_frac = [[0.0] * num_days]
                sesh_friend_tweets_pos_frac = [[0.0] * num_days]
                frac_dict[user][str_len].extend(sesh_friends_pos_frac)
                frac_tweets_dict[user][str_len].extend(sesh_friend_tweets_pos_frac)
                gini_dict[user][str_len].extend(gini_vals.flatten().tolist())
                bias_dict[user][str_len].extend(bias_vals.flatten().tolist())
                continue

            if len(days_gte_10) == 1:
                sesh_friends_pos_frac = np_sum(act_df_bin[where_fs_eq_1,:], axis=0) / (np_sum(act_df_bin, axis=0))
                sesh_friend_tweets_pos_frac = np_sum(act_df[where_fs_eq_1,:], axis=0) / (np_sum(act_df, axis=0) )
                frac_dict[user][str_len].extend(sesh_friends_pos_frac.flatten().tolist())
                frac_tweets_dict[user][str_len].extend(sesh_friend_tweets_pos_frac.flatten().tolist())
                gini_dict[user][str_len].extend(gini_vals.flatten().tolist())
                bias_dict[user][str_len].extend(bias_vals.flatten().tolist())
                continue



            try:
                sesh_friends_pos_frac = np_sum(act_df_bin[where_fs_eq_1,:][:,days_gte_10], axis=0) / (np_sum(act_df_bin[:,days_gte_10], axis=0) )   # array of length | number_days |
            except ValueError as e:
                print("hi ", act_df_bin[where_fs_eq_1,days_gte_10].shape,act_df_bin[:,days_gte_10].shape)
                raise ValueError
            except IndexError as e:
                #print("hi ", days_gte_10)
                raise IndexError("hi {}  act_df_bin.size {} len where_fs {} act_bin shape {}".format(days_gte_10, act_df_bin.size, len(where_fs_eq_1), act_df_bin.shape))
            sesh_friend_tweets_pos_frac = np_sum(act_df[where_fs_eq_1,:][:,days_gte_10], axis=0) / (np_sum(act_df[:,days_gte_10], axis=0) )
            
            
            #print(  act_df.shape, len(days_list), friends_seen_total.shape, fs_subset_tr.shape, sesh_friends_pos_frac.shape, np.mean(sesh_friends_pos_frac), )
            frac_dict[user][str_len].extend(sesh_friends_pos_frac.flatten().tolist())
            frac_tweets_dict[user][str_len].extend(sesh_friend_tweets_pos_frac.flatten().tolist())
            gini_dict[user][str_len].extend(gini_vals.flatten().tolist())
            bias_dict[user][str_len].extend(bias_vals.flatten().tolist())
            
            #frac_dict[user][str(length)] /= float(len(days_list)) if len(days_list) > 0 else 1
    return user_chunk, frac_dict, friends_dict, frac_tweets_dict, gini_dict, bias_dict






#print(wfw_dict[8943])
#wfw_num_ones = {user:who_follows_who[who_follows_who['id'] == user]['tid'].apply(lambda x: user_map[x]).value_counts() for user in random_sample}

def rewire(goal_corr):
    cur_assignments = assignments
    cur_user_map = user_map
    list_user_map = list(user_map.keys())
    
    delta = 100000

    rev_user_map = {0:{x:0 for x in cur_user_map if cur_user_map[x] == 0}, 1:{x:0 for x in cur_user_map if cur_user_map[x] == 1}}
    pos_neg_bf = np.array([cur_user_map[x] for x in cur_user_map])
    posn_mapping = {user:ix for ix, user in enumerate(list_user_map)}
    rev_mapping = {ix: user for ix, user in enumerate(list_user_map)}

    positive_nodes = [rev_mapping[x] for x in np.nonzero(pos_neg_bf)[0]]#{x:0 for x in rev_user_map[1] if x in degrees}#set([node for node,assignment in zip(list_user_map, cur_assignments) if assignment == 1])
    negative_nodes = [rev_mapping[x] for x in np.nonzero(pos_neg_bf == 0)[0]]#{x:0 for x in rev_user_map[0] if x in degrees}#set([node for node,assignment in zip(list_user_map, cur_assignments) if assignment == 0])
    #cur_corr = corr(degree_dist, [*cur_user_map.values()], [*positive_nodes])
    cur_corr = corr(degree_dist, pos_neg_bf, positive_nodes)

    print("done with ", goal_corr, " with corr ", cur_corr)
    iters = 0

    np_rand_choice = np.random.choice
    list_user_map_index = list_user_map.index
    while (cur_corr < goal_corr) and (iters < 5000):

        #es. For example, to increase ρkx, we randomly
        #choose nodes v1 with x = 1 and v0 with x = 0 and swap their attributes if the degree of v0 is
        #greater than the degree of v1. W
        
        #rand_pos = np_rand_choice(list([x for x in positive_nodes]), size=1)[0]
        rand_pos = np_rand_choice([rev_mapping[x] for x in np.nonzero(pos_neg_bf)[0]], size=100000, replace=False)
        #print("selected pos node")
        #rand_neg = np_rand_choice(list([x for x in negative_nodes]), size=1)[0]
        rand_neg = np_rand_choice([rev_mapping[x] for x in np.nonzero(pos_neg_bf == 0)[0]], size=100000, replace=False)
        
        degrees_pos = [degree_dist[posn_mapping[x]] for x in rand_pos]
        degrees_neg = [degree_dist[posn_mapping[x]] for x in rand_neg]
        neg_gt_pos = [neg >= pos for neg, pos in zip(degrees_neg, degrees_pos)]
        if any(neg_gt_pos):
            true_tups = [ix for ix, val in enumerate(neg_gt_pos) if val]
            for tup_ix in true_tups:
                pos = rand_pos[tup_ix]
                neg = rand_neg[tup_ix]
                cur_user_map[pos] = 0
                cur_user_map[neg] = 1
                pos_neg_bf[posn_mapping[pos]] = 0
                pos_neg_bf[posn_mapping[neg]] = 1

                #positive_nodes.pop(pos)
                #positive_nodes[neg] = 0
                #negative_nodes[pos] = 0
                #negative_nodes.pop(neg)
            new_corr = corr(degree_dist, pos_neg_bf, [rev_mapping[x] for x in np.nonzero(pos_neg_bf)[0]])
            delta = new_corr - cur_corr
            cur_corr = new_corr
        

        iters += 1
        
        print(cur_corr, delta, iters)
    #return cur_assignments, cur_user_map, cur_corr
    return cur_user_map.values(), cur_user_map, cur_corr




del wfw_id
del wfw_tid
del who_follows_who
del seeduser
del gini_tup


import joblib 

start = float(sys.argv[1])
end = float(sys.argv[2])


for goal_corr in np.arange(start, end, 0.15):#[0, 0.25, 0.50, 0.75, 1.0]:
    print("entering loop for correlation_kx ", goal_corr)
    assignments, user_map, cor = rewire(goal_corr)
    map_friends_scores = {x:[user_map[y] for y in map_friends[x]] for x in map_friends}

    import joblib 

    #if kind == 'popularity':
    map_deg_friends = {user: sum([degrees_out[fr] if fr in degrees_out else 0.0 for fr in map_friends[user]]) for user in map_friends}
    folder = '{}/joblib_memmap'.format(data_dir)
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

    

    
    with joblib.parallel_backend(backend="loky"):
        if not loaded_dicts:
            parallel = joblib.Parallel(verbose=100, n_jobs=14, temp_folder="/nas/home/nbartley/data/")#,require='sharedmem')
  
            
            user_map_list = np.array_split(np.array(list(random_sample)), 25)
            #results = Parallel(n_jobs=4)(delayed(job)(user_chunk) for user_chunk in user_map_list)
            
            #frac_friend_dicts = parallel(delayed(job2)(random_sample[slicer:slicer+500]) for slicer in np.arange(0,1500,500))
            frac_dict = {}
            friends_dict = {}
            frac_tweets_dict = {}
            gini_dict = {}
            bias_dict = {}
            results = parallel(delayed(job)(user_chunk) for user_chunk in user_map_list)
            try:
                for user_chnk, frac, frs, frac_tweets, gini, bias in results:
                    frac_dict.update(frac)
                    friends_dict.update(frs)
                    frac_tweets_dict.update(frac_tweets)
                    gini_dict.update(gini)
                    bias_dict.update(bias)
                with open('{}/frac_friend_dicts_{}_t5_{}_corrX_allusrs_gini_bias_weightedA_fixed_gini_actual.pkl'.format(data_dir,kind,  goal_corr), 'wb') as fp:
                    pkl.dump((frac_dict, friends_dict, frac_tweets_dict, gini_dict, bias_dict), fp)  
            except ValueError as e:
                print("screwed up unpacking ", e)
                with open('{}/frac_friend_dicts_{}_t5_{}_corrX_allusrs_gini_bias_weightedA_fixed_gini_actual.pkl'.format(data_dir,kind,  goal_corr), 'wb') as fp:
                    pkl.dump(results, fp)
              
