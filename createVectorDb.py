import json
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

def create_vector_database(data_path, output_path):
    """Create and save a FAISS vector database with Vietnamese product data."""
    print(f"Loading product data from {data_path}...")
    
    # Load JSON data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    
    # Create documents for categories and products
    for category in data["categories"]:
        # Create category document
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
        
        # Create documents for each product in this category
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
    
    print(f"Created {len(documents)} documents for categories and products.")
    
    # Create embeddings using Ollama instead of HuggingFace
    print("Creating embeddings with Ollama...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Create vector store
    print("Building vector database...")
    db = FAISS.from_documents(documents, embeddings)
    
    # Save vector store to disk
    print(f"Saving vector database to {output_path}...")
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # With:
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        os.makedirs(output_path, exist_ok=True)

    db.save_local(output_path)
    
    print(f"Vector database created and saved successfully to {output_path}")
    return db

if __name__ == "__main__":
    data_path = "data/products_data.json"
    output_path = "vector_db"
    
create_vector_database(data_path, output_path)