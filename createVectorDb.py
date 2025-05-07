import json
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain.docstore.document import Document
import numpy as np
import faiss
from langchain.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

def create_vector_database(data_path, output_path):
    """Create and save a FAISS vector database with Vietnamese product data, strictly using GPU."""
    
    if not (faiss.get_num_gpus() > 0 and hasattr(faiss, "GpuIndexFlatL2") and hasattr(faiss, "StandardGpuResources")) :
        raise RuntimeError(
            "FAISS-GPU is required. No GPU detected by FAISS or faiss-gpu components (GpuIndexFlatL2, StandardGpuResources) are missing. "
            "Please ensure you have a compatible GPU, CUDA installed, and faiss-gpu installed correctly via Conda."
        )
    print(f"GPU detected. Number of GPUs available: {faiss.get_num_gpus()}. Proceeding with GPU-only FAISS index creation.")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f) # Tải toàn bộ cấu trúc JSON
    
    documents = []
    
    # Lặp qua từng danh mục trong trường "categories" của dữ liệu JSON
    for category in data.get("categories", []):
        # 1. Tạo Document cho chính danh mục đó
        # Lấy danh sách tên sản phẩm trong danh mục hiện tại
        category_product_titles = [p.get('title', 'Sản phẩm không tên') for p in category.get('products', [])]
        
        # Tạo nội dung text cho Document của danh mục
        category_text_content = f"""
        Danh mục: {category.get('name', 'Danh mục không tên')}
        Mô tả: {category.get('description', 'Không có mô tả cho danh mục này.')}
        Sản phẩm trong danh mục này: {', '.join(category_product_titles)}
        """
        
        # Tạo metadata cho Document của danh mục
        category_meta = {
            "id": str(category.get('id', f"cat_auto_id_{category.get('name', 'unknown_category')}")), # Tạo ID tự động nếu không có
            "type": "category", # Đánh dấu đây là tài liệu loại danh mục
            "name": category.get('name', 'Danh mục không tên'),
            "image_path": category.get('image_path') # Lấy đường dẫn ảnh của danh mục
        }
        # Lọc bỏ các giá trị None khỏi metadata để giữ cho nó sạch sẽ
        category_meta = {k: v for k, v in category_meta.items() if v is not None}
        
        category_doc = Document(page_content=category_text_content.strip(), metadata=category_meta)
        documents.append(category_doc) # Thêm Document danh mục vào danh sách

        # 2. Lặp qua từng sản phẩm trong danh mục hiện tại
        for product_item in category.get("products", []):
            # Xây dựng nội dung text cho Document của sản phẩm
            text_parts = []
            product_title = product_item.get('title', 'Sản phẩm không tên') # 'title' từ products_data.json
            text_parts.append(f"Sản phẩm: {product_title}")

            # Thêm tên danh mục vào mô tả sản phẩm để có thêm ngữ cảnh
            text_parts.append(f"Thuộc danh mục: {category.get('name', 'Không rõ danh mục')}")
            
            product_description = product_item.get('description')
            if product_description:
                text_parts.append(f"Mô tả chi tiết: {product_description}")

            product_price = product_item.get('price')
            if product_price is not None: 
                text_parts.append(f"Giá bán: {product_price} ₫")
            
            product_features = product_item.get('features')
            if product_features and isinstance(product_features, list):
                text_parts.append(f"Đặc điểm nổi bật: {', '.join(product_features)}")
            elif product_features and isinstance(product_features, str): # Trường hợp features là một chuỗi
                 text_parts.append(f"Đặc điểm nổi bật: {product_features}")

            product_text_content = "\\n".join(text_parts) # Nối các phần text lại
            
            # Xây dựng metadata cho Document của sản phẩm
            current_product_metadata = {
                "id": str(product_item.get('id', f"prod_auto_id_{product_title}")), # Tạo ID tự động nếu không có
                "type": "product", # Đánh dấu đây là tài liệu loại sản phẩm
                "title": product_title,
                "category_id": str(category.get('id')), # Lưu ID của danh mục cha
                "category_name": category.get('name', 'Không rõ danh mục'), # Lưu tên của danh mục cha
                "price": product_price,
                "image_path": product_item.get('image_path') # Lấy đường dẫn ảnh của sản phẩm
            }
            # Lọc bỏ các giá trị None
            current_product_metadata = {k: v for k, v in current_product_metadata.items() if v is not None}
            
            product_doc = Document(page_content=product_text_content.strip(), metadata=current_product_metadata)
            documents.append(product_doc) # Thêm Document sản phẩm vào danh sách

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    if not documents:
        print("Không có tài liệu nào để xử lý. Tạo một chỉ mục FAISS GPU trống.")
        # Create a dummy GPU index and docstore for an empty FAISS object
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, 1)  # Dummy dimension for GPU index
        docstore = InMemoryDocstore({})
        index_to_docstore_id = {}
    else:
        texts = [doc.page_content for doc in documents]
        text_embeddings_list = embedding_model.embed_documents(texts)
        text_embeddings_np = np.array(text_embeddings_list, dtype=np.float32)

        if text_embeddings_np.ndim == 1 and text_embeddings_np.size > 0:
            text_embeddings_np = np.expand_dims(text_embeddings_np, axis=0)

        if text_embeddings_np.size == 0:
            print("Không có embeddings nào được tạo từ tài liệu. Tạo một chỉ mục FAISS GPU trống.")
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(res, 1) # Dummy dimension for GPU index
            docstore = InMemoryDocstore({})
            index_to_docstore_id = {}
        else:
            dimension = text_embeddings_np.shape[1]
            
            print("Đang sử dụng GPU để tạo chỉ mục FAISS.")
            res = faiss.StandardGpuResources()
            gpu_index = faiss.GpuIndexFlatL2(res, dimension) # Sử dụng tên biến khác để rõ ràng
            
            gpu_index.add(text_embeddings_np)

            print("Đang chuyển đổi chỉ mục GPU về chỉ mục CPU để lưu trữ.")
            index = faiss.index_gpu_to_cpu(gpu_index) # Chuyển đổi trở lại chỉ mục CPU
            
            doc_ids = [str(i) for i in range(len(documents))]
            docstore = InMemoryDocstore(
                {doc_ids[i]: documents[i] for i in range(len(documents))}
            )
            index_to_docstore_id = {i: doc_ids[i] for i in range(len(documents))}

    db = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        normalize_L2=False 
    )
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not output_dir and not os.path.exists(output_path):
         os.makedirs(output_path)

    db.save_local(output_path)
    print(f"Chỉ mục FAISS GPU đã được tạo và lưu tại: {output_path}")
    return db

if __name__ == "__main__":
    data_path = "data\\products_data.json"
    output_path = "vector_db"
    
    print("Bắt đầu tạo cơ sở dữ liệu vector (YÊU CẦU GPU)...")
    try:
        created_db = create_vector_database(data_path, output_path)
        if created_db:
            print(f"Cơ sở dữ liệu vector GPU đã được tạo thành công và lưu tại: {output_path}")
            if hasattr(created_db.index, 'ntotal'):
                 print(f"Tổng số vector trong chỉ mục: {created_db.index.ntotal}")
            # Check if it's a GpuIndex (it should be)
            if hasattr(faiss, "GpuIndex") and isinstance(created_db.index, faiss.GpuIndex):
                print("Xác nhận: Chỉ mục đang sử dụng GPU.")
            else:
                # This case should ideally not be reached if the logic is correct
                print("Cảnh báo: Chỉ mục không phải là GpuIndex như mong đợi.")
        else:
            print("Không thể tạo cơ sở dữ liệu vector GPU.")
    except RuntimeError as e:
        print(f"LỖI: {e}")
    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn trong quá trình tạo cơ sở dữ liệu vector: {e}")