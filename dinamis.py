# Import library
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import streamlit as st

import base64
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import time

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

def get_popular_movies(data, jumlah):
    """
    Mengembalikan daftar film yang paling populer berdasarkan rata-rata rating.

    Parameters:
    - data: DataFrame yang berisi data rating film.
    - n: Jumlah film yang ingin ditampilkan.

    Returns:
    DataFrame: DataFrame berisi informasi film-film yang paling populer.
    """
    popular_movies = data.groupby('movieId')['rating'].mean().sort_values(ascending=False).index
    popular_movies = popular_movies[:jumlah]

    # Dapatkan informasi tambahan tentang film dari data movies
    popular_movies_info = pd.merge(pd.DataFrame({'movieId': popular_movies}), movies_data, on='movieId')

    # Tambahkan kolom similarity dan atur nilainya menjadi 1
    popular_movies_info['similarity'] = 1

    # Tambahkan kolom pred_rating dan atur nilainya menjadi nilai rating
    popular_movies_info = pd.merge(popular_movies_info, data[['movieId', 'rating']], on='movieId')
    popular_movies_info.rename(columns={'rating': 'pred_rating'}, inplace=True)

    return popular_movies_info[['movieId', 'similarity', 'title', 'pred_rating']]



# Fungsi untuk merekomendasikan film berdasarkan judul
def get_top_n_recommendations_by_title(title, n_jumlah):
    # Latih model pada data pelatihan
    model.fit(trainset)
    try:
        movie_id = data[data['title'] == title]['movieId'].iloc[0]

        # Check if the movie is in the training set
        if movie_id not in model.trainset.all_items():
            print("error ")
            popular_movie=get_popular_movies(data, jumlah=n_jumlah)
            return popular_movie
        #Dapatkan ID film dari judul
        movie_inner_id = model.trainset.to_inner_iid(movie_id)
        # Dapatkan rekomendasi berdasarkan ID film
        raw_recs = model.get_neighbors(movie_inner_id, k=20)

        # Konversi ID film ke judul dan tambahkan pred_rating dan similarity
        movie_recs = [
            (
                model.trainset.to_raw_iid(inner_id),
                model.sim[movie_inner_id, inner_id],
                model.predict(model.trainset.to_raw_uid(movie_inner_id), model.trainset.to_raw_iid(inner_id)).est
            )
            for inner_id in raw_recs
        ]
        movie_recs_df = pd.DataFrame(movie_recs, columns=['movieId', 'similarity', 'pred_rating'])
        # Gabungkan dengan data movie untuk mendapatkan judul
        recommendations_title = pd.merge(movie_recs_df, movies_data[['movieId', 'title']], on='movieId')
        print ("recommendations_title before: " ,len(recommendations_title))
        # Lengkapi rekomendasi dengan film populer jika kurang dari n
        if len(recommendations_title) < n_jumlah:
            num_additional_recommendations = n_jumlah - len(recommendations_title)
            popular_movies = get_popular_movies(data, jumlah=num_additional_recommendations)
            recommendations_title = pd.concat([recommendations_title, popular_movies], ignore_index=True)

        return recommendations_title
    except ValueError:
        # Default recommendation: Show popular movies
        popular_movies = get_popular_movies(data, jumlah=n_jumlah)
        return popular_movies

# Fungsi untuk merekomendasikan film untuk pengguna tertentu
def get_top_n_recommendations_by_user(user_id, n):
    # Latih model pada data pelatihan
    model.fit(trainset)
    
    # Dapatkan similarity matrix setelah melatih model
    similarity_matrix = model.sim
    predictions = model.test(testset)
    top_n = {}

    # Ambil daftar film yang ada dalam data pelatihan
    train_movie_ids = set(model.trainset.to_raw_iid(iid) for iid in model.trainset.all_items())
    
    for uid, iid, true_r, est, _ in predictions:
        # Pastikan item (film) ada dalam data pelatihan
        if iid not in train_movie_ids:
            continue
        
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))

    # Urutkan rekomendasi berdasarkan estimasi rating tertinggi
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    # Buat DataFrame rekomendasi dengan kolom 'movieId', 'pred_rating'
    recommendations_df = pd.DataFrame(top_n[user_id], columns=['movieId', 'pred_rating'])

    # Ambil similarity hanya untuk film yang ada dalam rekomendasi
    similarities = [similarity_matrix[model.trainset.to_inner_iid(movie_id), model.trainset.to_inner_iid(movie_id)]
                    for movie_id in recommendations_df['movieId']]
    recommendations_df['similarity'] = similarities

    # Gabungkan dengan informasi film untuk mendapatkan kolom 'title'
    recommendations_user = pd.merge(recommendations_df, movies_data[['movieId', 'title']], on='movieId')
    # Jika rekomendasi dari model berbasis pengguna kurang dari n, lengkapi dengan film populer
    if len(recommendations_user) < n:
        remaining_count = n - len(recommendations_user)
        popular_movies = get_popular_movies(data, remaining_count)
        recommendations_user = pd.concat([recommendations_user, popular_movies], ignore_index=True)

    return recommendations_user[['movieId', 'similarity', 'title', 'pred_rating']]
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



    if selected_menu == "REKOMENDATION":
        st.header("movie recomendation with item-based collaborative filtering")        
        selected_rect = st.selectbox("Select Recommendation By", ['movie_title','userID'])
        if selected_rect == "userID":
            user_id = st.number_input(label="Enter the number of user ID",value=10,step=5,max_value=610,min_value=1)
            number_of_movie = st.number_input(label="Enter the number of movie you want to recommend",value=10,step=5)
            if st.button('recomendation') :
                st.write(f"recommend movie for user id {user_id}")
                rekomendation=get_top_n_recommendations_by_user(user_id, number_of_movie)
                st.dataframe(rekomendation,use_container_width=True)
        if selected_rect == "movie_title":
            movie_title = st.selectbox("Enter movie title:",data["title"].unique())            
            number_of_movie = st.number_input(label="Enter the number of movie you want to recommend",value=10,step=5)
            if st.button('recomendation') :
                st.write(f"recommend movie for movie title {movie_title}")
                rekomendation=get_top_n_recommendations_by_title( movie_title, number_of_movie)
                st.dataframe(rekomendation,use_container_width=True)

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
