import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.impute import SimpleImputer
import pickle
import random
from thefuzz import fuzz

links = pd.read_csv('./ml-latest-small/links.csv')
tags = pd.read_csv('./ml-latest-small/tags.csv')
movies = pd.read_csv('./ml-latest-small/movies.csv')
ratings = pd.read_csv('./ml-latest-small/ratings.csv')

user = pd.merge(ratings.reset_index(),tags.reset_index(),how="left",on = ['userId','movieId','timestamp'])
movie_ratings = pd.merge(user,movies, on =['movieId'], how ='outer')
movie_ratings = pd.merge(movie_ratings,links, on =['movieId'], how ='outer')
movie_ratings['userId'].fillna(0,inplace = True)
movie_ratings['userId']= movie_ratings['userId'].astype('int64')
movie_ratings['rating'].fillna(0,inplace = True)

user_rating= movie_ratings.groupby(['userId','genres'])['rating'].mean()
user_genres= movie_ratings.groupby(['userId'])['genres'].value_counts()
# rank movie for recommending
movie_ranking= movie_ratings.groupby(['genres','title'])['rating'].mean().sort_values(ascending=False)

R_users = movie_ratings[['userId','rating','title']]
R_users_dropna= R_users[R_users['userId'] > 0]
R_users_pivot = R_users_dropna.pivot_table(columns = 'title', index='userId', values = 'rating')
R_users_pivot.fillna(0,inplace = True)
films = R_users_pivot.columns
comps = [f'genres_{i}' for i in range(200)]


nmf =  pickle.load(open('./nmf.sav','rb'))  
imputer =  pickle.load(open('./imputer.sav','rb')) 

Q = pd.DataFrame(nmf.components_, columns=films, index=comps)

def dataframe():
    return movie_ranking

def random_movies(n):
    '''
    return n random movies
    '''
    return list(random.sample(list(films), n))


def base_recommender(userid):
    '''
    recommending films with user genres based on rating and films count
    retrun df, with films in the index
    '''
    user_gen = user_rating[userid][:5].index.to_list() # top 5 rated genres
    #print(user_gen)
    user_genres2 = user_genres[userid][:3].index.to_list() # top 3 wachted genres
    #print(user_genres2)
    
    gen_user = user_gen
    
    for gen in user_genres2:
        if gen not in user_gen:
            gen_user.append(gen)
         
    df = pd.DataFrame(columns=['rating'])
    for gen in gen_user:
        df = pd.concat([df,pd.DataFrame(movie_ranking[gen][:30],columns=['rating'])])
    df = df.loc[df['rating']> 4.0]
    df = df[df.index.isin(movie_ratings[movie_ratings['userId'] == userid]['title'].tolist())==False]
    
    dic = {}
    for film in df.head(10).index:
        dic[film] = str(movie_ratings[movie_ratings['title'] == film]['imdbId'].unique().item()).zfill(7)
    
    return df.head(10), dic


def NMF_recommender(X,nmf=nmf, imputer = imputer):
    '''recommending films with NMF
    film is index in the return df
    '''
    new_dict = {}
    print(X)
    for film in films:
        for f in X:
            if fuzz.token_set_ratio(film, f) == 100:
                new_dict[film] = X[f]
    X = new_dict
    
    new_dict = {}
    for film in films:
        if film in X:
            new_dict[film] = X[film]
        else:
            new_dict[film] = np.nan
    
    comps = [f'genres_{i}' for i in range(200)]
    new_user_df = pd.DataFrame(new_dict, index=[0])
    new_user_df.index = ["new_user"]
    
    new_user_df_imp = pd.DataFrame(imputer.transform(new_user_df), columns=new_user_df.columns, index=new_user_df.index)
    P_new_user = pd.DataFrame(nmf.transform(new_user_df_imp), index=new_user_df_imp.index, columns=comps)
    Q = pd.DataFrame(nmf.components_, columns=films, index=comps)
    R_estimate = P_new_user.dot(Q)
    
    # drop used rated films
    #print(R_estimate)
    R_estimate = R_estimate.drop(X.keys(), axis=1)
    
    R_estimate_sorted = R_estimate.T.sort_values("new_user", ascending=False)

    dic = {}
    for film in R_estimate_sorted.head(10).index:
        dic[film] = str(movie_ratings[movie_ratings['title'] == film]['imdbId'].unique().item()).zfill(7)
    
    return R_estimate_sorted.head(10), dic

### cosine similarity
def cosim(x, y):
    num = np.dot(x, y)
    den = np.sqrt(np.sum(x**2)) * np.sqrt(np.sum(y**2))
    return num/den
films = R_users_pivot.columns
def cos_similarity(X):
    '''
    recommend movies based on cosine similarity
    X is a dictionary
    '''
    
    new_dict = {}
    for film in films:
        for f in X:
            if fuzz.token_set_ratio(film, f) == 100:
                new_dict[film] = X[f]
    X = new_dict
    
    #list unseen movies
    unseen_movies = unseen(X)
    
    #prepare new user data
    new_dict = {}
    for film in films:
        if film in X:
            new_dict[film] = X[film]
        else:
            new_dict[film] = np.nan
            
    new_user_df = pd.DataFrame(new_dict, index=[0])
    new_user_df.index = ["new_user"]
    
    imputer = SimpleImputer(strategy = 'constant', fill_value = 0)
    new_user_df_imp = pd.DataFrame(imputer.fit_transform(new_user_df), columns=new_user_df.columns, index=new_user_df.index)
   
    weighted_ratings = NBCF(unseen_movies,new_user_df_imp)

    predicted_rating_df = pd.DataFrame(weighted_ratings, columns=["movie", "rating"])
    predicted_rating_df = predicted_rating_df.sort_values("rating", ascending=False).head(10).set_index('movie')
    dic = {}
    for film in predicted_rating_df.index:
        dic[film] = str(movie_ratings[movie_ratings['title'] == film]['imdbId'].unique().item()).zfill(7)
    
    return predicted_rating_df, dic


def unseen(X):
    '''
    return unseen movies 
    X is the na filled dataframe
    '''
    unseen_list = []
    for film in films:
        if film not in X:
            film_rating = movie_ratings[movie_ratings['title'] == film]['rating'].mean()
            if film_rating > 4.95:
                unseen_list.append(film)
    return unseen_list

def users(movie):
    '''return other users who has watched the movie'''
    return movie_ratings[(movie_ratings['title'] == movie) & (movie_ratings['rating'] > 0)]['userId'].tolist()

def NBCF(unseen,X):
    '''return list of movies with weighted rating
    input unseen movies list and X : new user dataframe'''
    predicted_ratings = []
    for movie in unseen:
        #other userd who has watched(rated) the movie
        other_users = users(movie)                        
        
        # calculate the average weighted sum of other uses
        num = 0
        den = 0
        for user in other_users:
            # capture rating for this `user'
            user_rating = R_users_pivot.loc[user][movie]
            similarity = cosim(R_users_pivot.loc[user], X.loc['new_user'])
            num += user_rating * similarity
            den += similarity
        if den != 0:
            predicted_rating = num/den
            predicted_ratings.append((movie, predicted_rating))
        else:
            pass
        
    return predicted_ratings
        
        
    
    
if __name__ == '__main__':
    base_recommender(2)