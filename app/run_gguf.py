from llama_cpp import Llama
from load_dotenv import load_dotenv


# 1. Configuration (Change this)
# IMPORTANT: Replace '/path/to/your/model.gguf' with the actual file path
GGUF_MODEL_PATH = "/path/to/your/model.gguf" 

# Set parameters for efficient inference
N_CONTEXT = 4096  # Context window size (adjust based on model/RAM)
N_THREADS = 8     # Number of CPU threads to use

# 2. Instantiate the Llama Model
try:
    print(f"Loading model from: {GGUF_MODEL_PATH}")
    llm = Llama(
        model_path=GGUF_MODEL_PATH,
        n_ctx=N_CONTEXT,
        n_threads=N_THREADS,
        verbose=False # Set to True for detailed logging
    )
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the GGUF_MODEL_PATH is correct and the file exists.")
    exit()

# 3. Define the Prompt and Parameters
PROMPT = "Write a short, three-sentence story about a robot who discovers rain for the first time."

# Parameters for the generation
MAX_TOKENS = 150
TEMPERATURE = 0.8  # Higher means more creative
STOP_SEQUENCES = ["\n", "###"] # Stop generation when these strings are encountered

# 4. Run Inference
print("-" * 40)
print(f"Prompt: {PROMPT}")
print("-" * 40)

# The __call__ method generates the response
output = llm(
    prompt=PROMPT,
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
    stop=STOP_SEQUENCES,
    echo=False # Don't repeat the prompt in the output text
)

# 5. Extract and Print the Result
# The output is a dictionary, and the text is in the 'choices' list
generated_text = output["choices"][0]["text"].strip()

print(f"Generated Text:\n{generated_text}")
print("-" * 40)
print(f"Generation Stats: {output['usage']}")