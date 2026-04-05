import google.generativeai as genai

genai.configure(api_key="AIzaSyB9JsNkezqM8RY-hoUrrsajS2oELQCMmJI")

for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)