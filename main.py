import json
from typing import List, Union, Any
from sentence_transformers import util
from pydantic import BaseModel
import pickle

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder

app = FastAPI()

origins = [
    "http://localhost:4290",
    "http://localhost:4291",
    "http://localhost:4292"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Read the saved model using pickle
with open('models/model_pickle.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/sentence_embeddings.pkl', 'rb') as f:
    sentence_embeddings = pickle.load(f)

with open('models/subset.pkl', 'rb') as f:
    subset = pickle.load(f)

with open('models/dataframe.pkl', 'rb') as f:
    dataframe = pickle.load(f)

print("Subset: \n", subset.head())
print("\n===========================================================================")
print("==================================== BML ===================================")
print("===========================================================================\n")
print("Dataframe: \n", dataframe.head())


class Product(BaseModel):
    ProductID: int
    ProductName: str
    ProductBrand: str
    Gender: str
    Price: int
    NumImages: int
    Description: Union[str, None] = None
    PrimaryColor: Union[str, None] = None


@app.get("/", tags=["home"])
@app.get("/api/", tags=["api"])
async def hello_world():
    return {"Welcome to BML Product Recommendation API"}


@app.get("/api/products", response_model=List[Product], response_model_exclude_unset=True)
async def products():
    data = dataframe.head(150)
    result = data.to_json(orient='records')
    parsed_products = json.loads(result)
    dumped_products = json.dumps(parsed_products, indent=4)

    # with open('datasets/product_data_cleaned.json', 'w') as file:
    #     file.write(dumped_products)

    # print(json_list)
    print("===================================== **** =======================================")
    print(dumped_products)

    return parsed_products


@app.get("/api/products/{product_id}", response_model=Product)
async def product(product_id):
    product_id = int(product_id)
    data = dataframe[dataframe['ProductID'] == product_id]
    result = data.to_json(orient='records', lines=True)
    parsed_product = json.loads(result)
    dumped_product = json.dumps(parsed_product, indent=4)

    # print(json_list)
    print("===================================== *** ", product_id, " *** =======================================")
    print(dumped_product)

    return {result}


@app.get("/api/products/similar-product/{product_id}", response_model=List[Product], response_model_exclude_unset=True)
async def similar_products(product_id):
    try:
        product_id = int(product_id)
        cosine_scores = util.cos_sim(model.encode(subset[subset["ProductID"] == product_id]["Feature_Set"].values[0]),
                                     sentence_embeddings)
        score = cosine_scores[0].tolist()
        recommended_products = []
        for i in range(0, 31):
            max_score = score.index(max(score))
            recommended_products.append(subset['ProductID'][max_score])
            score[max_score] = -1

        recommended_products.remove(product_id)
        products_df = dataframe[dataframe["ProductID"].isin(recommended_products)]
        result = products_df.to_json(orient='records')
        parsed_products = json.loads(result)
        dumped_products = json.dumps(parsed_products, indent=4)

        print("==================================================== "
              "Recommendation Based on Product with ID: ", product_id,
              " ====================================================")
        print("Product ID: ", product_id)
        print("Similar Products ID: ", recommended_products)
        print("Similar Products: \n", dumped_products)
        print("========================================================================== "
              "The END "
              "==========================================================================")

        return parsed_products

    except IndexError:
        print("Product Index Error")
    except:
        print("Product not found for the given id")
