import os
import re
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import faiss

def get_category_prompt_template():
    """Prompt template for category browsing in Vietnamese."""
    template = """
    Bạn là trợ lý thương mại điện tử hữu ích giúp khách hàng Việt Nam tìm sản phẩm.
    Sử dụng thông tin sau về danh mục sản phẩm:

    {context}

    Nếu người dùng không chắc họ muốn danh mục nào, hãy yêu cầu làm rõ và đề xuất các danh mục.
    Nếu người dùng không đề cập là họ muốn danh mục nào và không hỏi thẳng vào tên của một sản phẩm, hãy yêu cầu làm rõ và đề xuất các danh mục.
    Người dùng đang hỏi về các danh mục sản phẩm có sẵn. Hãy trả lời bằng tiếng Việt và cung cấp danh sách các danh mục với mô tả ngắn gọn cho mỗi danh mục. Định dạng câu trả lời thành danh sách đánh số.

    Nếu yêu cầu của người dùng dường như liên quan đến một sản phẩm cụ thể hơn là danh mục, gợi ý họ duyệt danh mục liên quan trước hoặc hỏi về sản phẩm cụ thể.

    Câu hỏi của người dùng: {question}
    """
    return PromptTemplate(template=template, input_variables=["context", "question"])

def get_product_prompt_template():
    """Prompt template for specific product recommendations in Vietnamese."""
    template = """
    Bạn là trợ lý thương mại điện tử hữu ích đề xuất sản phẩm dựa trên yêu cầu của người dùng Việt Nam.
    Bạn sẽ hỗ trợ người dùng tìm sản phẩm phù hợp với nhu cầu của họ hoàn toàn bằng tiếng việt.
    Sử dụng thông tin sau về sản phẩm:

    {context}

    Người dùng đang hỏi về sản phẩm trong một danh mục cụ thể. Hãy trả lời bằng tiếng Việt và gợi ý các sản phẩm phù hợp. Đối với mỗi sản phẩm, hãy bao gồm:
    1. Ảnh sản phẩm
    2. Tên sản phẩm
    3. Giá bán
    4. Tính năng hoặc lợi ích chính
    5. Giải thích ngắn gọn tại sao sản phẩm này phù hợp với nhu cầu của người dùng

    Định dạng câu trả lời thành danh sách đánh số các đề xuất.
    Chỉ đề xuất sản phẩm phù hợp với danh mục mà người dùng đang hỏi.
    Nếu người dùng không chắc họ muốn danh mục nào, hãy yêu cầu làm rõ và đề xuất các danh mục.

    Câu hỏi của người dùng: {question}
    """
    return PromptTemplate(template=template, input_variables=["context", "question"])

def load_vector_db(vector_db_path):
    """Loads a FAISS vector database and strictly moves it to GPU."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    print(f"Đang tải cơ sở dữ liệu vector từ: {vector_db_path}")
    db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    print("Cơ sở dữ liệu vector đã được tải vào CPU. Chuẩn bị chuyển sang GPU.")

    if not (faiss.get_num_gpus() > 0 and hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu")):
        raise RuntimeError(
            "FAISS-GPU is required for loading. No GPU detected by FAISS or faiss-gpu components "
            "(StandardGpuResources, index_cpu_to_gpu) are missing. "
            "Please ensure you have a compatible GPU, CUDA installed, and faiss-gpu installed correctly via Conda."
        )
    
    current_index = db.index
    is_cpu_index = "Gpu" not in type(current_index).__name__ # A simple check

    if is_cpu_index:
        print(f"Số GPU có sẵn: {faiss.get_num_gpus()}. Đang chuyển chỉ mục FAISS đã tải sang GPU 0...")
        try:
            res = faiss.StandardGpuResources()  # Initialize GPU resources
            # Assuming we always want to use the first GPU (device 0)
            db.index = faiss.index_cpu_to_gpu(res, 0, current_index) 
            print("Chỉ mục FAISS đã được chuyển thành công sang GPU.")
        except Exception as e:
            # This error will now stop the application if GPU transfer fails.
            raise RuntimeError(f"Không thể chuyển chỉ mục FAISS từ CPU sang GPU: {e}")
    else:
        # This case implies the index loaded was already a GpuIndex, which is unusual for load_local.
        # Or, the type check was not accurate.
        print("Chỉ mục FAISS đã ở trên GPU hoặc không được nhận dạng là chỉ mục CPU có thể chuyển đổi.")
        # We can add an explicit check here if we want to be more robust
        if not (hasattr(faiss, "GpuIndex") and isinstance(db.index, faiss.GpuIndex)):
             print("Cảnh báo: Chỉ mục không phải là GpuIndex như mong đợi sau khi tải.")

    return db

def is_category_query(query):
    """Determine if the Vietnamese query is about browsing categories or specific products."""
    category_keywords = [
        "danh mục", "loại", "các loại", "bạn có gì", "có những gì", "cho tôi xem", 
        "liệt kê", "duyệt", "lựa chọn", "sản phẩm nào", "có bán gì", "bán những gì"
    ]
    
    query_lower = query.lower()
    for keyword in category_keywords:
        if keyword in query_lower:
            return True
    
    if len(query.split()) < 4 and not any(cat.lower() in query_lower for cat in ["điện thoại", "máy tính", "tai nghe"]):
        return True
    
    return False

def setup_rag_chain(vector_db, query):
    llm = Ollama(model="gemma3:4b")
    
    if is_category_query(query):  
        retriever = vector_db.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": {"type": "category"}
            }
        )
        prompt = get_category_prompt_template()
    else:
        retriever = vector_db.as_retriever(
            search_kwargs={"k": 10}
        )
        prompt = get_product_prompt_template()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain, is_category_query(query)

def process_images(response, is_category):
    image_paths = []
    image_captions = []
    unique_items = {}
    
    for doc in response["source_documents"]:
        if "id" in doc.metadata and doc.metadata["id"] not in unique_items:
            unique_items[doc.metadata["id"]] = doc.metadata
    
    item_count = 0
    max_items = 3 if is_category else 5
    if len(unique_items) > 0:
        cols = st.columns(min(max_items, len(unique_items)))
        
        for item_id, metadata in unique_items.items():
            if item_count >= max_items:
                break
                
            if "image_path" in metadata and metadata["image_path"] is not None and os.path.exists(metadata["image_path"]):
                image_paths.append(metadata["image_path"])
                
                if is_category:
                    caption = metadata['name'] if 'name' in metadata else metadata.get('title', 'Danh mục')
                else:
                    caption = f"{metadata.get('title', 'Sản phẩm')} - {metadata.get('price', 'N/A')}₫"
                    
                image_captions.append(caption)
                
                with cols[item_count]:
                    st.image(metadata["image_path"], caption=caption)
                
                item_count += 1
    
    return image_paths, image_captions

def run_app():
    st.title("Trợ lý Mua sắm Thông minh (GPU Required)")
    
    vector_db_path = "vector_db"
    if not os.path.exists(vector_db_path):
        st.error("Không tìm thấy thư mục cơ sở dữ liệu vector. Vui lòng chạy createVectorDb.py (yêu cầu GPU) trước.")
        return
    
    try:
        with st.spinner("Đang tải hệ thống gợi ý (Yêu cầu GPU)..."):
            vector_db = load_vector_db(vector_db_path)
        st.success("Hệ thống gợi ý đã sẵn sàng trên GPU!")
    except RuntimeError as e:
        st.error(f"LỖI KHỞI TẠO HỆ THỐNG: {e}")
        st.error("Vui lòng đảm bảo bạn có GPU tương thích, trình điều khiển CUDA và faiss-gpu được cài đặt đúng cách từ Conda.")
        return
    except Exception as e:
        st.error(f"Đã xảy ra lỗi không mong muốn khi tải cơ sở dữ liệu vector: {e}")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "images" in message and len(message["images"]) > 0:
                cols = st.columns(len(message["images"]))
                for i, img_path in enumerate(message["images"]):
                    try:
                        with cols[i]:
                            st.image(img_path, caption=message["image_captions"][i])
                    except Exception as e:
                        st.error(f"Lỗi hiển thị hình ảnh: {e}")

    if not st.session_state.messages:
        st.chat_message("assistant").markdown(
            "👋 Xin chào! Tôi là trợ lý mua sắm của bạn. Bạn có thể hỏi tôi về các sản phẩm"
        )

    if prompt := st.chat_input("Bạn muốn tìm sản phẩm gì hôm nay?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Đang tìm sản phẩm phù hợp cho bạn..."):
                qa_chain, is_category = setup_rag_chain(vector_db, prompt)
                response = qa_chain({"query": prompt})
                result_text = response["result"]
                final_answer = re.sub(r"<think>.*?</think>", "", result_text, flags=re.DOTALL).strip()

                st.markdown(final_answer)
                
                image_paths, image_captions = process_images(response, is_category)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": final_answer,
            "images": image_paths,
            "image_captions": image_captions
        })

if __name__ == "__main__":
    run_app()