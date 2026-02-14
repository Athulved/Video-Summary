import requests

# Your Hugging Face API Key
API_KEY = ""

# Model Endpoint
url = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B"

# Headers
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def query_model(prompt: str):
    data = {"inputs": prompt}
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"Error: {response.status_code}, {response.text}"

# Example usage
if __name__ == "__main__":
    test_prompt = "Explain the concept of deep reinforcement learning."
    response = query_model(test_prompt)
    print("Generated Response:", response)
