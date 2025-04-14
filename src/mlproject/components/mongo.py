from pymongo import MongoClient
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

class MongoDBClient:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGO_DB_URI"))
        self.db = self.client[os.getenv("MONGO_DB_NAME")]
        print(f"Connected to MongoDB at {os.getenv('MONGO_DB_URI')}, using database: {self.db.name}")

    def export_collection_to_dataframe(self, collection_name='students'):
        try:
            if collection_name not in self.db.list_collection_names():
                available = self.db.list_collection_names()
                raise Exception(
                    f"Collection '{collection_name}' does not exist in DB '{self.db.name}'. "
                    f"Available collections: {available}"
                )

            collection = self.db[collection_name]
            data = list(collection.find())
            if not data:
                raise Exception(f"No data found in collection '{collection_name}'.")

            return pd.DataFrame(data)

        except Exception as e:
            raise Exception(f"Error exporting collection to dataframe: {e}")
