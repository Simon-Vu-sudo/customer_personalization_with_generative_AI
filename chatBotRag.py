import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings

# Define prompt templates for Vietnamese queries
def get_category_prompt_template():
    """Prompt template for category browsing in Vietnamese."""
    template = """
    Bạn là trợ lý thương mại điện tử hữu ích giúp khách hàng tìm sản phẩm.
    Sử dụng thông tin sau về danh mục sản phẩm:

    {context}

    Người dùng đang hỏi về các danh mục sản phẩm có sẵn. Hãy trả lời bằng tiếng Việt và cung cấp danh sách các danh mục với mô tả ngắn gọn cho mỗi danh mục. Định dạng câu trả lời thành danh sách đánh số.

    Nếu yêu cầu của người dùng dường như liên quan đến một sản phẩm cụ thể hơn là danh mục, gợi ý họ duyệt danh mục liên quan trước hoặc hỏi về sản phẩm cụ thể.

    Câu hỏi của người dùng: {question}
    """
    return PromptTemplate(template=template, input_variables=["context", "question"])

def get_product_prompt_template():
    """Prompt template for specific product recommendations in Vietnamese."""
    template = """
    Bạn là trợ lý thương mại điện tử hữu ích đề xuất sản phẩm dựa trên yêu cầu của người dùng.
    Sử dụng thông tin sau về sản phẩm:

    {context}

    Người dùng đang hỏi về sản phẩm trong một danh mục cụ thể. Hãy trả lời bằng tiếng Việt và gợi ý các sản phẩm phù hợp. Đối với mỗi sản phẩm, hãy bao gồm:
    1. Tên sản phẩm
    2. Giá bán
    3. Tính năng hoặc lợi ích chính
    4. Giải thích ngắn gọn tại sao sản phẩm này phù hợp với nhu cầu của người dùng

    Định dạng câu trả lời thành danh sách đánh số các đề xuất.
    Chỉ đề xuất sản phẩm phù hợp với danh mục mà người dùng đang hỏi.
    Nếu bạn không chắc họ muốn danh mục nào, hãy yêu cầu làm rõ.

    Câu hỏi của người dùng: {question}
    """
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Load the vector database
def load_vector_db(vector_db_path):
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    return db

# Determine if query is about categories or specific products
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
    
    # If the query is short and general, assume it's a category query
    if len(query.split()) < 4 and not any(cat.lower() in query_lower for cat in ["điện thoại", "máy tính", "tai nghe"]):
        return True
    
    return False

# Create the RAG chain based on query type
def setup_rag_chain(vector_db, query):
    # Initialize the LLM
    llm = Ollama(model="llama3.2:1B")
    
    # Determine if query is about categories or products
    if is_category_query(query):
        # Category query - filter for category documents and use category prompt
        retriever = vector_db.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": {"type": "category"}
            }
        )
        prompt = get_category_prompt_template()
    else:
        # Product query - use product prompt
        retriever = vector_db.as_retriever(
            search_kwargs={"k": 10}
        )
        prompt = get_product_prompt_template()
    
    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain, is_category_query(query)

# Process and display images based on query type
def process_images(response, is_category):
    image_paths = []
    image_captions = []
    unique_items = {}
    
    # Get unique items (categories or products) from source documents
    for doc in response["source_documents"]:
        if "id" in doc.metadata and doc.metadata["id"] not in unique_items:
            unique_items[doc.metadata["id"]] = doc.metadata
    
    # Get item images
    item_count = 0
    max_items = 3 if is_category else 5
    if len(unique_items) > 0:
        cols = st.columns(min(max_items, len(unique_items)))
        
        for item_id, metadata in unique_items.items():
            if item_count >= max_items:
                break
                
            if "image_path" in metadata and os.path.exists(metadata["image_path"]):
                image_paths.append(metadata["image_path"])
                
                if is_category:
                    caption = metadata['name'] if 'name' in metadata else metadata.get('title', 'Danh mục')
                else:
                    caption = f"{metadata.get('title', 'Sản phẩm')} - {metadata.get('price', 'N/A')}₫"
                    
                image_captions.append(caption)
                
                # Display image
                with cols[item_count]:
                    st.image(metadata["image_path"], caption=caption)
                
                item_count += 1
    
    return image_paths, image_captions

# Streamlit app
def run_app():
    st.title("Trợ lý Mua sắm Thông minh")
    
    # Check if vector database exists
    vector_db_path = "vector_db"
    if not os.path.exists(vector_db_path):
        st.error("Không tìm thấy cơ sở dữ liệu vector. Vui lòng chạy create_vector_db.py trước.")
        return
    
    # Load vector database
    with st.spinner("Đang tải hệ thống gợi ý..."):
        vector_db = load_vector_db(vector_db_path)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display images if this is a response with product recommendations
            if message["role"] == "assistant" and "images" in message and len(message["images"]) > 0:
                cols = st.columns(len(message["images"]))
                for i, img_path in enumerate(message["images"]):
                    try:
                        with cols[i]:
                            st.image(img_path, caption=message["image_captions"][i])
                    except Exception as e:
                        st.error(f"Lỗi hiển thị hình ảnh: {e}")

    # Display a welcoming message for new users
    if not st.session_state.messages:
        st.chat_message("assistant").markdown(
            "👋 Xin chào! Tôi là trợ lý mua sắm của bạn. Bạn có thể hỏi tôi về các danh mục sản phẩm"
            "hoặc sản phẩm cụ thể. Thử hỏi 'Cửa hàng có những loại sản phẩm nào?' hoặc 'Cho tôi xem điện thoại thông minh.'"
        )

    # Chat input
    if prompt := st.chat_input("Bạn muốn tìm sản phẩm gì hôm nay?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Đang tìm sản phẩm phù hợp cho bạn..."):
                # Get response from RAG chain
                qa_chain, is_category = setup_rag_chain(vector_db, prompt)
                response = qa_chain({"query": prompt})
                result_text = response["result"]
                
                # Display text response
                st.markdown(result_text)
                
                # Process and display images
                image_paths, image_captions = process_images(response, is_category)
        
        # Add assistant response to chat history with image data
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response["result"],
            "images": image_paths,
            "image_captions": image_captions
        })

if __name__ == "__main__":
    run_app()