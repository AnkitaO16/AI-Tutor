import ollama

# Test basic chat functionality
try:
    response = ollama.chat(model='llama3.1:8b', messages=[
        {
            'role': 'user',
            'content': 'Explain photosynthesis in one sentence.',
        },
    ])
    print("Response:", response['message']['content'])
    print("Ollama is working correctly!")
except Exception as e:
    print(f"Error: {e}")
    print("Make sure Ollama is running with 'ollama serve'")