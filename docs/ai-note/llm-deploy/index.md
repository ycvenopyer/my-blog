# LLM Deploy

Deploying LLMs at scale is an engineering feat that can require multiple clusters of GPUs. In other scenarios, demos and local apps can be achieved with much lower complexity.

Local deployment: Privacy is an important advantage that open-source LLMs have over private ones. Local LLM servers (LM Studio, Ollama, oobabooga, kobold.cpp, etc.) capitalize on this advantage to power local apps.

Demo deployment: Frameworks like Gradio and Streamlit are helpful to prototype applications and share demos. You can also easily host them online, for example, using Hugging Face Spaces.

Server deployment: Deploying LLMs at scale requires cloud (see also SkyPilot) or on-prem infrastructure and often leverages optimized text generation frameworks like TGI, vLLM, etc.

Edge deployment: In constrained environments, high-performance frameworks like MLC LLM and mnn-llm can deploy LLM in web browsers, Android, and iOS.

📚 References:

Streamlit - Build a basic LLM app: Tutorial to make a basic ChatGPT-like app using Streamlit.

HF LLM Inference Container: Deploy LLMs on Amazon SageMaker using Hugging Face's inference container.

Philschmid blog by Philipp Schmid: Collection of high-quality articles about LLM deployment using Amazon SageMaker.

Optimizing latence by Hamel Husain: Comparison of TGI, vLLM, CTranslate2, and mlc in terms of throughput and latency.
