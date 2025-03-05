import json
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

def create_vector_database(data_path, output_path):
    """Create and save a FAISS vector database with Vietnamese product data."""
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    
    for category in data["categories"]:
        category_text = f"""
        Danh mục: {category['name']}
        Mô tả: {category['description']}
        Sản phẩm trong danh mục này: {', '.join([p['title'] for p in category['products']])}
        """
        
        category_metadata = {
            "id": category['id'],
            "type": "category",
            "name": category['name'],
            "image_path": category['image_path'] if 'image_path' in category else None
        }
        
        category_doc = Document(page_content=category_text, metadata=category_metadata)
        documents.append(category_doc)
        
        for product in category['products']:
            product_text = f"""
            Sản phẩm: {product['title']}
            Danh mục: {category['name']}
            Mô tả: {product['description']}
            Giá: {product['price']} ₫
            Đặc điểm: {', '.join(product['features'])}
            """
            
            product_metadata = {
                "id": product['id'],
                "type": "product",
                "title": product['title'],
                "category_id": category['id'],
                "category_name": category['name'],
                "price": product['price'],
                "image_path": product['image_path']
            }
            
            product_doc = Document(page_content=product_text, metadata=product_metadata)
            documents.append(product_doc)
    
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    db = FAISS.from_documents(documents, embeddings)
    

    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        os.makedirs(output_path, exist_ok=True)

    db.save_local(output_path)
    
    return db

if __name__ == "__main__":
    data_path = "data/products_data.json"
    output_path = "vector_db"
    
create_vector_database(data_path, output_path)