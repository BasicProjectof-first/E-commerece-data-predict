import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
from apyori import apriori
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder 
st.set_page_config (page_title="E-Commerece",
                   page_icon="basketanalysis")
hide_st_style="""
             <style>
             #MainMenu {visibility:hidden;}
             footer {visibility:hidden;}
             footer {visibility:hidden;}
             </style>
             """
st.markdown(hide_st_style,unsafe_allow_html=True)
dataset=st.container()
graphs=st.container()
frequentpattern=st.container()
file_uploader=st.container()
with dataset:
    dataset=pd.read_csv("flipkart.csv")
    st.header("Market Data Analysis")
    st.write(dataset)
#datacleaning to remove the unwanted columns from dataset
    st.write(dataset.describe())
    col=['uniq_id','crawl_timestamp','product_url','product_category_tree','pid','image','is_FK_Advantage_product','description','overall_rating','product_specifications']
    data=dataset.drop(columns=col,axis=1)
    #st.write(data)
#nullvalues in the updated dataset
    dar=data.isnull().sum()
    st.subheader("Total Null values of dataset")
    st.write(dar)
#data['brand'].fillna('Ax',inplace=True)
    data['product_rating']=data['product_rating'].mask(data['brand']=='Alisha',2)
    data['product_rating']=data['product_rating'].mask(data['brand']=='Ax',3.5)
#data['brand']=data['brand'].mask(data['product_rating']==2,'Alisha')
    dt=data.isnull().sum()
    st.info("Market Basket Analysis Correlation")
    st.write(data.corr())
with graphs:
    cola=['product_name','discounted_price','product_rating']
    ds=data.drop(columns=cola,axis=1)
    #st.write(ds)
    st.info("Bar chart between brand and retail price")
    fig=px.bar (
        ds,x="brand",
        y="retail_price",
        hover_name="brand",
        color_continuous_scale="blue",
        color_discrete_sequence=['#0083B8']*len(ds)
        )
    
    st.plotly_chart(fig,theme="streamlit")
    st.info("Bar Chart between products and retail price")
    fig1=px.bar(data,
              x="product_name",
              y="retail_price",
              hover_name="product_name",
             color_discrete_sequence=['#0083B8']*len(data) )
    st.plotly_chart(fig1)
    #fig3=plt.figure()
    #sns.histplot(data,
                 #x="retail_price",
                 #hue='discounted_price')
    #st.pyplot(fig3)
    
    cod=['product_name','brand','product_rating']
    #to drop the string attributes
    dt=data.drop(columns=cod,axis=1)
    #st.write(dt)
    #heatmap between retailprice and discounted price
    #dt.groupby('retail_price').discounted_price.value_counts()
    origin=(dt)
    fig4=plt.figure()
    sns.heatmap(origin,annot=True,linewidths=0.1,vmin=0,vmax=3)
    st.pyplot(fig4)
#apriori algorithm delete unwanted columns of retail_price,discounted price and brand
with frequentpattern:
    st.header("Frequent Pattern Growth For products")
    flip=pd.read_csv("flipkart.csv")
    st.write(flip)
    flip.fillna(value='ama',inplace=True)
    dt=flip['product_name']
    #st.write(dt)
    te=TransactionEncoder()
    te_ary=te.fit(dt).transform(dt)
   #st.write(te_ary)
    dt=pd.DataFrame(te_ary)
    st.write(dt)
    frequent_itemsets=fpgrowth(dt,min_support=0.2)
    st.write(frequent_itemsets)
#st.write(frequent_itemsets[2])
    st.header("Apriori and Association rules for Products")
    frequent_itemsets=apriori(dt,min_support=0.6,use_colnames=True)
    st.write(frequent_itemsets)
    rules=association_rules(frequent_itemsets,metric="confidence",min_threshold=0.6)
    st.write(rules)
    rules1=rules[['antecedents','consequents','support','confidence','lift']]
    rules2=rules1[rules1['confidence']>=1]
    st.write(rules2)
#calculate to brand of comapines
    st.header("Frequent Pattern Growth for brand")
    flip.fillna(value='ama',inplace=True)
    dt=flip['brand']
    #st.write(dt)
    te=TransactionEncoder()
    te_ary=te.fit(dt).transform(dt)
   #st.write(te_ary)
    dt=pd.DataFrame(te_ary)
    #st.write(dt)
    frequent_itemsets=fpgrowth(dt,min_support=0.2)
    st.write(frequent_itemsets)
#st.write(frequent_itemsets[2])
    st.header("Apriori and Association rules for brand")
    frequent_itemsets=apriori(dt,min_support=0.2,use_colnames=True)
    st.write(frequent_itemsets)
    rules=association_rules(frequent_itemsets,metric="confidence",min_threshold=0.6)
    st.write(rules)

with file_uploader:
    st.info("If Any file is uploaded")
    upload_file=st.file_uploader('Upload a file conataining a market Basket Analysis data')
    if upload_file is not None:
        dat=pd.read_csv(upload_file)
        te=TransactionEncoder()
        te_ary=te.fit(dat).transform(dat)
   #st.write(te_ary)
        dat=pd.DataFrame(te_ary)
    #st.write(dt)
        frequent_itemsets=fpgrowth(dat,min_support=0.03)
        st.write(frequent_itemsets)
        


      