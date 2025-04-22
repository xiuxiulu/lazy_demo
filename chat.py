# -*- coding: utf-8 -*-

import os
import requests
import lazyllm

query = "你好，你是谁"

# 从环境变量获取API密钥，如果没有则使用默认值
api_key_glm = os.environ.get("LAZYLLM_GLM_API_KEY")
print(api_key_glm)
try:
    print("Initializing chat module...")
    chat = lazyllm.OnlineChatModule(
        source="glm",
    )
    print("Chat module initialized successfully")
    
    print("Sending request to API...")
    try:
        response = chat.forward(query)
        print(f"Answer: {response}")
    except requests.RequestException as req_err:
        print(f"API request failed: {req_err}")
        print("Please check your API key and internet connection")
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()
