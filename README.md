# Multimodal RAG Assistant with NVIDIA NeMo
Adapted example built on top of and using NVIDIA AI Foundation Models.

### Example Use Case
Obtaining a Sample HTML Template for a Product Page Tailored to Your Company's Specific Products.

### Brief Overview
This tool simplifies the process of refreshing product sections on websites and modifying marketing content. 
Just upload a product area image you wish to replicate and choose your company from the external documents. 
The AI Assistant will then produce HTML code that replicates the look of the image, tailored to showcase 
your specific products.

# Implemented Features
- [RAG in 5 minutes Chatbot Video](https://youtu.be/N_OOfkEWcOk) Setup with NVIDIA AI Playground components
- Source references with options to download the source document
- Analytics through Streamlit at ```/?analytics=on```
- Multimodal parsing of documents - images, text through multimodal LLM APIs
- Uses fuyu_8b to get image description
- Uses llama2_code_34b to develop code based on image description
- External document containing sample company product information in the output.txt file
