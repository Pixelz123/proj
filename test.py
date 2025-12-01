import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
# AIzaSyBKhcEojaHi3OwOYZjxbfS3oxmZzLo5p68
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found in environment.")
    print("Please set it in your .env file or paste it below.")
    api_key = input("Enter your Google API Key: ").strip()

genai.configure(api_key=api_key)

print("\nFetching available models for API version 'v1beta'...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
            
    print("\nUse one of the names above in your 'src/rag_engine.py' file.")
except Exception as e:
    print(f"Error fetching models: {e}")