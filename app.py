# Import library
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import streamlit as st

import base64
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu


# Set page layout and title
st.set_page_config(page_title="movies recommendation", page_icon="data/img/logo.png")

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }

    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background('./data/img/background.png')

def sidebar(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
          background-size: contain;
          background
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
sidebar('./data/img/sidebar.png')
st.markdown(f"""
      <style>
      .css-1629p8f h1, .css-1629p8f h2, .css-1629p8f h3, .css-1629p8f h4, .css-1629p8f h5, .css-1629p8f h6, .css-1629p8f span {{
          background-color: white;
          color: black;
          border-radius: 20px;
          text-align: center;
          
      }}
      .css-184tjsw p , .css-1offfwp p{{
          background-color: white;
          color: black;
          border-radius: 10px;
          padding:8px;
      }}
      .css-18ni7ap{{
          background: rgb(144,146,255);
          background: linear-gradient(90deg, rgba(144,146,255,0.7259278711484594) 0%, rgba(223,142,145,0.8183648459383753) 22%, rgba(226,135,135,0.8519782913165266) 82%, rgba(105,117,226,0.7539390756302521) 100%);        }}
      </style>
      """,
      unsafe_allow_html=True,)

# Baca dataset MovieLens (Anda dapat mengganti path dengan lokasi dataset Anda)
ratings_data = pd.read_csv('./data/dataset/ratings.csv')
movies_data = pd.read_csv('./data/dataset/movies.csv')

# Gabungkan dataset ratings dan movies berdasarkan kolom 'movieId'
data = pd.merge(ratings_data, movies_data, on='movieId')

# Inisialisasi pembaca (Reader) untuk dataset Surprise
reader = Reader(rating_scale=(1, 5))

# Muat data ke dalam format Surprise
dataset = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)

# Bagi data menjadi train dan test set
trainset, testset = train_test_split(dataset, test_size=0.2)

# Inisialisasi model KNN dengan item-based collaborative filtering
sim_options = {
    'name': 'cosine',  # Pengukuran kesamaan (bisa juga 'pearson')
    'user_based': False  # Gunakan item-based collaborative filtering
}

model = KNNBasic(sim_options=sim_options)

def load_movie_data(movies,ratings):
    
    # Merge average ratings with movie data
    movies = pd.merge(movies, ratings, on='movieId', how='left')
    
    # Split genres into list
    movies['genres'] = movies['genres'].str.split('|')
    
    return movies

def prepare_data(movies):
    # Create a new DataFrame with exploded genres
    genres_expanded = movies.explode('genres')
    # Pivot table to create binary indicator for each genre
    genres_pivot = pd.pivot_table(genres_expanded, index='movieId', columns='genres', aggfunc=lambda x: 1, fill_value=0)
    return genres_pivot

def get_movie_recommendations(movie_title, movies, genres_pivot):
    # Find movieId for given movie title
    movie_id = movies[movies['title'] == movie_title]['movieId'].values[0]
    # Get genres for the movie
    movie_genres = genres_pivot.loc[movie_id].values.reshape(1, -1)
    movie_genre_list = movies[movies['title'] == movie_title]['genres'].values[0]

    # Calculate cosine similarity between the selected movie and all other movies
    similarities = cosine_similarity(genres_pivot, movie_genres)
    # Create DataFrame to store movieId and similarity scores
    sim_scores_df = pd.DataFrame(similarities, index=genres_pivot.index, columns=['similarity'])
    
    # Use KNN to find nearest neighbors
    nn = NearestNeighbors(n_neighbors=50, metric='cosine')
    nn.fit(genres_pivot)
    distances, indices = nn.kneighbors(movie_genres)
    
    # Get top 10 most similar movies
    top_movies = sim_scores_df.iloc[indices[0][1:]]
    
    # Merge with movie data to get movie titles, genres, and ratings
    top_movies = top_movies.merge(movies, left_index=True, right_on='movieId')
    

    result_df = top_movies.sort_values(by=['similarity', 'rating'], ascending=False)
    # Remove input movie from recommendations
    result_df = result_df[result_df['title'] != movie_title]
    # Remove duplicate movies
    result_df = result_df.drop_duplicates(subset=['title'],ignore_index=True)
    return result_df[['title', 'similarity', 'genres', 'rating']], movie_genre_list

# Fungsi untuk membuat grafik MAE
def plot_mae(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data['K'], data['MAE'], marker='o', linestyle='-', color='b')
    plt.title('Grafik MAE terhadap K')
    plt.xlabel('Nilai K')
    plt.ylabel('MAE')
    plt.grid(True)
    st.pyplot(plt)

# Streamlit App
def main():
    with st.sidebar :
        selected_menu = option_menu('SYSTEM RECOMMENDATION',["HOME", "DATA PREPROCESSING", "REKOMENDATION", "EVALUASI"])

    if selected_menu == "HOME":
        st.header("Sistem Rekomendasi ")
        st.write("Sistem rekomendasi ialah suatu mekanisme yang dapat memberikan informasi kepada user suatu rekomendasi atau saran tentang item yang akan digunakan konsumen atau pelangggan,Dengan banyaknya jumlah film yang telah dibuat, diperlukan sistem rekomendasi yang mempertimbangkan rating yang diberikan oleh user lain sebagai acuan untuk mendapatkan rekomendasi film yang sesuai.")
        st.subheader("item-based collaborative filtering")
        st.image('./data/img/item-CF.png',use_column_width=True,caption='Konsep item-based filtering (Marafi,2014)')
        st.write('Item-based CF berasumsi bahwa jika mayoritas pengguna memberi penilaian beberapa item secara serupa, pengguna aktif yang ditargetkan juka akan memberikan penilaian terhadap item-item tersebut secara serupa pula (Pantreath,2015). Metode item-based CF akan merekomendasikan item kepada user dengan cara item – item yang di rating oleh pengguna maka item tersebut memeliki kemiripan yang besar, tidak perduli tentang konten yang dimiliki item tersebut.')

    if selected_menu == "DATA PREPROCESSING":
        selected_menu = st.selectbox("Select Recommendation By", ['statis','dinamis'])
        if selected_menu == "dinamis":
            data_movie = st.file_uploader("masukan dataset movie", key="movie", type='csv')
            data_ratings = st.file_uploader("masukan dataset ratings", key="ratings", type='csv')
            if data_movie and data_ratings is not None :
                movie = pd.read_csv(data_movie)
                rating = pd.read_csv(data_ratings)
                st.write('berikut merupakan tapilan untuk data movie')
                st.dataframe(movie)
                st.write('berikut merupakan tapilan untuk data rating')
                st.dataframe(rating)
                if st.button('start prepro'):
                    #menggabungkan data movies dan data ratings sesuai nilai ISBN
                    data = pd.merge(rating, movie, on='movieId')
                    df=data.copy()
                    #menghapus data kosong
                    df.dropna(inplace=True)
                    df.reset_index(drop=True,inplace=True)
                    #menghapus kolom yang tidak di gunakan
                    df.drop(columns=["timestamp"],axis=1,inplace=True)
                    #membuat tabel matrix item_userr
                    item_userr=df.pivot_table(index=["title"],columns=["userId"],values="rating")
                    item_userr.fillna(0,inplace=True)
                    model.fit(trainset)
                    # Dapatkan matriks similarity antar item dari model
                    item_similarity_matrix = model.sim

                    # Konversi matriks similarity ke array NumPy
                    item_similarity_matrix_array = np.array(item_similarity_matrix)

                    datamatriks=pd.DataFrame(item_similarity_matrix_array)

                    st.write('berikut merupakan tampilan data gabungan')
                    st.dataframe(data)
                    st.write('berikut merupakan tampilan penghapusan kolom timestamp dan nilai NAN')
                    st.dataframe(df)
                    st.write('berikut merupakan tampilan matrix item_userr')
                    st.dataframe(item_userr)
                    st.write('berikut merupakan tampilan matrix cosine_similarity')
                    st.dataframe(datamatriks)
        if selected_menu == "statis":
            data_merge= pd.read_csv('./data/prepro/merge_data.csv')
            data_pivot=pd.read_csv('./data/prepro/item-userr.csv')
            matrix_cosine=pd.read_csv('./data/prepro/matrix_cosine_.csv')
            st.header("Data preprocessing")
            st.write("Data yang dikumpulkan bersumber dari data sekunder yang di ambil dari website Movielens  dengan jumlah data buku sebanyak 9000 data film dan terdapat 600 data user dengan 100,000 data rating dengan format Comma Separated Values (csv).")
            st.subheader("dataset movies")
            st.dataframe(movies_data.head(1000),use_container_width=True)
            st.subheader("dataset ratings")
            st.dataframe(ratings_data.head(1000),use_container_width=True)
            st.write("Tabel diatas merupakan contoh dari dataset yang akan digunakan dalam penelitian ini. Pada tabel pertama merupakan dataset movies dengan parameter movieId, title, dan genre sengangkan pada tabel ke dua merupakan dataset rating dengan parameter userId, movieId, rating,dan timestamp.")
            st.subheader("dataset merge")
            st.dataframe(data_merge.head(1000),use_container_width=True)
            st.write('tabel diatas merupakan contoh dari gabungan antara movies dataset dan ratings.dimana dataset sudah mengalami tahapan data preprocessing yaitu penggabungan dataset movies dan ratings serta pengecekan nilai NAN penghapusan kolom yang tidak digunakan untuk pembuatan model. Adapun kolom yang dihapus yaitu timestamp. lalu melakukan modifikasi pada table agar index dan kolom sesuai dan disimpan ke dalam dataframe baru sehingga didapatkan data yang siap digunakan (final dataset),')
            st.subheader("tabel item-user")
            st.dataframe(data_pivot,use_container_width=True)
            st.write("tabel diatas merupakan matriks item-user menggunakan item sebagai indeks dan user sebagai kolom-kolom yang diisi dengan rating user terhadap item")
            st.subheader("matrix cosine similarity")
            st.dataframe(matrix_cosine,use_container_width=True)
            st.write('table diatas merupakan tabel cosine simalarity menunjukan nilai similaritas antar item. Nilai similaritas 1 pada matriks tersebut menunjukan bahwa kedua item dikatakan mirip dan jika nilai similaritas 0 maka dikatakan tidak mirip')

    if selected_menu == "REKOMENDATION":
        st.header("movie recomendation with item-based collaborative filtering")        
        movie_title = st.selectbox("Enter movie title:",movies_data["title"].unique())            
        number_of_movie = st.number_input(label="Enter the number of movie you want to recommend",value=10,step=5,max_value=50,min_value=1)
        if st.button('recomendation') :
            # Load movie data
            movies = load_movie_data(movies_data,ratings_data)
            # Prepare data
            genres_pivot = prepare_data(movies)
            recommended_movies,movie_genre_list= get_movie_recommendations(movie_title, movies, genres_pivot)
            st.write(f"recommend movie for movie title {movie_title} dengan genre {movie_genre_list} ")
            st.dataframe(recommended_movies.head(number_of_movie),use_container_width=True)

    if selected_menu == "EVALUASI":
        data_eval=pd.read_csv('./data/evaluasi/eval_k.csv')
        st.header("Evaluation")
        st.subheader("evaluasi nilai K")
        st.dataframe(data_eval,use_container_width=True)
        st.subheader("grafik pengujian nilai K")
        plot_mae(data_eval)
        st.write("pada grafik diatas merupakan mengujian nilai K dengan rentang nilai K dari 5,10,20,30,40,50 kita dapat melihat jika nilai MAE semakin kecil seiring bertambahnya nilai K, semakin kecil nilai MAE semakin baik pula performa dari model yang digunakan, sehingga didapatkan hasil akhir dimana nilai MAE yang paling kecil terdapat pada pengujian nilai K = 50 dengan nilai MAE sebesar 0.7594, dapat disimpulkan dengan nilai MAE yang semakin kecil maka model yang di gunakan memiliki performa yang baik")

if __name__ == "__main__":
    main()
