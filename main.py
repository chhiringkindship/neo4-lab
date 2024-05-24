import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from graphdatascience import GraphDataScience
from neo4j_tools import gds_db_load, gds_utils
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.graphs import Neo4jGraph
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda
import gradio as gr

load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
AURA_DS = os.getenv('AURA_DS') == 'False'

print(AURA_DS)

gds = GraphDataScience(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    aura_ds=AURA_DS)

gds.set_database("neo4j")

gds.debug.sysInfo()
#embedding 
embedding_model = OpenAIEmbeddings()
embedding_dimension = 1536

department_df = pd.read_csv('https://storage.googleapis.com/neo4j-workshop-data/genai-hm/department.csv')
product_df = pd.read_csv('https://storage.googleapis.com/neo4j-workshop-data/genai-hm/product.csv')
article_df = pd.read_csv('https://storage.googleapis.com/neo4j-workshop-data/genai-hm/article.csv')
customer_df = pd.read_csv('https://storage.googleapis.com/neo4j-workshop-data/genai-hm/customer.csv')
transaction_df = pd.read_csv('https://storage.googleapis.com/neo4j-workshop-data/genai-hm/transaction.csv')

# create constraints - one uniqueness constraint for each node label
gds.run_cypher('CREATE CONSTRAINT unique_department_no IF NOT EXISTS FOR (n:Department) REQUIRE n.departmentNo IS UNIQUE')
gds.run_cypher('CREATE CONSTRAINT unique_product_code IF NOT EXISTS FOR (n:Product) REQUIRE n.productCode IS UNIQUE')
gds.run_cypher('CREATE CONSTRAINT unique_article_id IF NOT EXISTS FOR (n:Article) REQUIRE n.articleId IS UNIQUE')
gds.run_cypher('CREATE CONSTRAINT unique_customer_id IF NOT EXISTS FOR (n:Customer) REQUIRE n.customerId IS UNIQUE')

# load nodes
gds_db_load.load_nodes(gds, department_df, 'departmentNo', 'Department')
gds_db_load.load_nodes(gds, article_df.drop(columns=['productCode', 'departmentNo']), 'articleId', 'Article')
gds_db_load.load_nodes(gds, product_df, 'productCode', 'Product')
gds_db_load.load_nodes(gds, customer_df, 'customerId', 'Customer')

# load relationships
gds_db_load.load_rels(gds, article_df[['articleId', 'departmentNo']], source_target_labels=('Article', 'Department'),
                      source_node_key='articleId', target_node_key='departmentNo',
                      rel_type='FROM_DEPARTMENT')
gds_db_load.load_rels(gds, article_df[['articleId', 'productCode']], source_target_labels=('Article', 'Product'),
                      source_node_key='articleId',target_node_key='productCode',
                      rel_type='VARIANT_OF')
gds_db_load.load_rels(gds, transaction_df, source_target_labels=('Customer', 'Article'),
                      source_node_key='customerId', target_node_key='articleId', rel_key='txId',
                      rel_type='PURCHASED')

# convert transaction dates
gds.run_cypher('''
MATCH (:Customer)-[r:PURCHASED]->()
SET r.tDat = date(r.tDat)
''')

# create combined text property. This will help simplify later with semantic search and RAG
gds.run_cypher("""
    MATCH(p:Product)
    SET p.text = '##Product\n' +
        'Name: ' + p.prodName + '\n' +
        'Type: ' + p.productTypeName + '\n' +
        'Group: ' + p.productGroupName + '\n' +
        'Garment Type: ' + p.garmentGroupName + '\n' +
        'Description: ' + p.detailDesc
    RETURN count(p) AS propertySetCount
    """)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.width', 0)

product_emb_df = product_df[['productCode', 'prodName', 'productTypeName', 'productGroupName', 'garmentGroupName', 'detailDesc']]
product_emb_df = product_emb_df[product_emb_df.detailDesc.notnull()]

def create_doc(row):
    return f'''
##Product
Name: {row.prodName}
Type: {row.productTypeName}
Group: {row.productGroupName}
Garment Type: {row.garmentGroupName}
Description: {row.detailDesc}
'''

product_emb_df['text'] = product_emb_df.apply(create_doc, axis=1)
product_emb_df = product_emb_df.drop(columns=['prodName', 'productTypeName', 'productGroupName', 'garmentGroupName', 'detailDesc'])
product_emb_df

count = 0
embeddings = []
for docs in gds_db_load.chunks(product_emb_df.text, n=500):
    count += len(docs)
    print(f'Embedded {count} of {product_emb_df.shape[0]}')
    embeddings.extend(embedding_model.embed_documents(docs))

# Set as column of dataframe to prepare for loading
product_emb_df['textEmbedding'] = embeddings
product_emb_df

count = 0
embeddings = []
for docs in gds_db_load.chunks(product_emb_df.text, n=500):
    count += len(docs)
    print(f'Embedded {count} of {product_emb_df.shape[0]}')
    embeddings.extend(embedding_model.embed_documents(docs))

# Set as column of dataframe to prepare for loading
product_emb_df['textEmbedding'] = embeddings
product_emb_df
print(product_emb_df)

# load vector properties
records = product_emb_df[['productCode', 'textEmbedding']].to_dict('records')
print(f'======  loading Product text embeddings ======')
total = len(records)
print(f'staging {total:,} records')
cumulative_count = 0
for recs in gds_db_load.chunks(records, n=100):
    res = gds.run_cypher('''
    UNWIND $recs AS rec
    MATCH(n:Product {productCode: rec.productCode})
    CALL db.create.setNodeVectorProperty(n, "textEmbedding", rec.textEmbedding)
    RETURN count(n) AS propertySetCount
    ''', params={'recs': recs})
    cumulative_count += res.iloc[0, 0]
    print(f'Set {cumulative_count:,} of {total:,} text embeddings')

#create index
gds.run_cypher('''
CREATE VECTOR INDEX product_text_embeddings IF NOT EXISTS FOR (n:Product) ON (n.textEmbedding)
OPTIONS {indexConfig: {
 `vector.dimensions`: toInteger($dim),
 `vector.similarity_function`: 'cosine'
}}''', params={'dim': embedding_dimension})

gds.run_cypher('CALL db.awaitIndex("product_text_embeddings", 300)')
