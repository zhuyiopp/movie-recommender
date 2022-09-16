from flask import Flask, render_template, request
from recommender import base_recommender, NMF_recommender, random_movies, cos_similarity

app = Flask(__name__)

#http://localhost:5000 to see the page
# start the first page
@app.route("/")
def homepage():
    films = random_movies(6)
    films_dic = dict([(a, a.replace(' ', '_')) for a in films ]) 
    return render_template('homepage.html', title = 'Don\'t know which movie to watch? You must try our movie recommender!',movies = films_dic )

# the 2nd page
@app.route("/rec1")
def recommendation():
    user = request.args
    films, dic = base_recommender(int(user['user_id'])) 

    links = {}
    for item in dic.keys():
        links[item] = "https://www.imdb.com/title/tt"+dic[item]

    return render_template('base_recommender.html', title = 'Welcome back!',user = int(user['user_id']), movies = films, imdb = links)
@app.route("/rec2")
def recommendation2():
    films_input = request.args
    films_input=dict([a.replace('_', ' '), int(x)] for a, x in films_input.items())
    films, dic = NMF_recommender(films_input)

    links = {}
    for item in dic.keys():
        links[item] = "https://www.imdb.com/title/tt"+dic[item]

    return render_template('nmf_recommender.html', title = 'Welcome to the best movie recommender', movies = films, imdb = links)

@app.route("/rec3")
def recommendation3():
    films_input = request.args
    films_input=dict([a.replace('_', ' '), int(x)] for a, x in films_input.items())
    films, dic = cos_similarity(films_input)

    links = {}
    for item in dic.keys():
        links[item] = "https://www.imdb.com/title/tt"+dic[item]
        
    return render_template('cos_recommender.html', title = 'Welcome to the best movie recommender', movies = films, imdb = links)




if __name__ == "__main__":
    app.run(debug = True, port=5000)
