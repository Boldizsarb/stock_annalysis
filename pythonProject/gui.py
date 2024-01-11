import streamlit as st
from streamlit_lottie import st_lottie

from nasdaq_100 import func_calling
import json
from companies.ASML import asml
from companies.META import meta
from companies.BKNG import bkng
from companies.NVDA import nvda
import requests



def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()



# calculate clusters and save them to a JSON file
@st.cache
def calculate_and_save_clusters():
    clusters = func_calling(False)
    with open("clusters.json", "w") as file:
        json.dump(clusters, file)


#  load clusters from a JSON file
def load_clusters():
    try:
        with open("clusters.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return None

## Animation
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def gui(): ## the whole gui!
    st.set_page_config(page_title="Top 100 Stock / Trading", page_icon="🏢", layout="wide") ## can be called once only, at the begining of the render
    #lottie_stocks = load_lottiefile("pythonProject/stock.json")
    lottie_stocks = load_lottiefile("stock.json")


    with st.container():
        st.subheader("Solent Financial Technology")
        left_column, right_column = st.columns(2)
        with left_column:
            st.title("SOLFINTECH")
            st.write(
                "The most accurate and up-to-date financial data is made available through Yahoo APIs on the website. This ensures that users are provided with the most reliable information at the present moment.")
            st.write("[Learn more >](https://finance.yahoo.com/)")
        with right_column:

            st_lottie(
                lottie_stocks,
                speed=0.5,
                reverse=False,
                loop=True,
                quality="low",
                height=200,
                width=None,
                key=None
            )


    st.write("---")
    st.write(
        "The Nasdaq 100 identified the 100 top-performing companies, which were subsequently organized into clusters using the clustering technique. By employing the K-means algorithm, four distinct groups were formed, and one representative from each group was selected for in-depth analysis.")

#### BUTTON FOR CLUSTERs
    if st.button('Calculate Clusters'):
        with st.spinner('Calculating clusters...'):
            clusters = load_clusters()
            if clusters is None:
                calculate_and_save_clusters()
                clusters = load_clusters()
        st.write('The Clustering calculated real time on a year data based on the Closing price of the companies! *(In the bacground using multithreading)*')
        st.write('Here are the 4 clusters: ')
        cluster1, cluster2, cluster3, cluster4 = clusters
        st.write(f' **Cluster 1** : {cluster1}')
        st.write(f' **Cluster 2** : {cluster2}')
        st.write(f' **Cluster 3** : {cluster3}')
        st.write(f' **Cluster 4** : {cluster4}')
    st.write("---")
    st.subheader("Chose one:")
    option = st.selectbox( ## selection between companies (the 4)
        ">", ('ASML ## Advanced Semiconductor Materials Lithography',
                            'META ## Facebook', 'BKNG ## Booking Holdings Inc',
                            'NVDA ## NVIDIA')
    )
    if option == 'ASML ## Advanced Semiconductor Materials Lithography':
        asml()
    elif option == 'META ## Facebook':
        meta()
    elif option == 'BKNG ## Booking Holdings Inc':
        bkng()
    elif option == 'NVDA ## NVIDIA':
        nvda()





