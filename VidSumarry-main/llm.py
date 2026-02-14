from transformers import pipeline

# Load the model locally (ensure you have the required model downloaded)
pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B")

def query_model(prompt: str) -> str:
    result = pipe(prompt, max_length=300, do_sample=True)
    return result[0]["generated_text"]

# Example usage
if __name__ == "__main__":
    try:
        test_prompt = "Explain the concept of deep reinforcement learning."
        response = query_model(test_prompt)
        print("Generated Response:", response)
    except Exception as e:
        print("Error:", e)
