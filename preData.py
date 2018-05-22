import numpy as np
import pandas as pd
####17535655*3条数据
data = pd.read_table("../lastfm_data/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv",
                         usecols=[0, 2, 3],
                         names=['user', 'artist', 'plays'],na_filter=False)
# data = pd.read_table("./u.data",
#                          usecols=[0, 1, 2],
#                          names=['user', 'artist', 'plays'],na_filter=False)
# data['user'] = data['user'].astype("category")
# data['artist'] = data['artist'].astype("category")

popular_user = data[['user','plays']].groupby('user').sum().reset_index()
popular_user = popular_user.sort_values('plays',ascending=False).reset_index(drop=True)

print("total user",len(popular_user))   ###100000万条数据，943个用户
total_play_count = sum(popular_user.plays)
print((float)(popular_user.head(n=10000).plays.sum())/total_play_count*100)


popular_artist = data[['artist','plays']].groupby('artist').sum().reset_index()
popular_artist = popular_artist.sort_values('plays',ascending = False).reset_index(drop=True)
print(len(popular_artist))   ####1682个artist
total_play_count = sum(popular_artist.plays)
print((float)(popular_artist.head(n=1000).plays.sum())/total_play_count*100)


popular_user_10K = popular_user.head(n=10000)
popular_artist_1K = popular_artist.head(n=1000)
popular_user_list = list(popular_user_10K.user)
popular_artist_list = list(popular_artist_1K.artist)
small_data_user = data[data.user.isin(popular_user_list)]
del(data)
small_data = small_data_user[small_data_user.artist.isin(popular_artist_list)]
del(small_data_user)
small_data.shape  ###(263042*3)
print(len(set(small_data.user)))  ###所以取了10000个人，1000首歌
print(len(set(small_data.artist)))
small_data.to_csv(path_or_buf='./small_data.csv',index = False,encoding ="UTF-8")