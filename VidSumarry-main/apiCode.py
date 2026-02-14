from groq import Groq
import os


client = Groq(api_key="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")  # Pass API key explicitly

completion = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=[
        {
            "role": "user",
            "content": "What is deep learning? Give plain text."
        }
    ],
    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    stream=True,
    stop=None,
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
