# Offline GenAI Chatbot

A lightweight **offline Generative AI chatbot** built with Python, PyTorch, and Hugging Face Transformers. The project uses the **SmolLM2-360M-Instruct** language model to generate responses to user questions and maintain short conversational context—all from a locally saved model running on the CPU.

> **Project status:** Notebook-based prototype. The notebook downloads and saves the model, reloads it in local-only mode, and provides an interactive command-line chat loop. It does not currently include a web UI, API, database, or formal automated tests.

## Features

- **Local model inference:** Loads the saved model with `local_files_only=True` after the initial download.
- **CPU execution:** Runs the model on the CPU using PyTorch; no CUDA GPU is required.
- **Instruction-following chat model:** Uses `HuggingFaceTB/SmolLM2-360M-Instruct`.
- **Simple teaching-assistant behavior:** System instructions ask the model to explain concepts in simple English, provide examples, and acknowledge uncertainty.
- **Conversation context:** Includes recent user/assistant messages to support follow-up questions.
- **Sampling controls:** Uses `temperature=0.7`, `top_p=0.9`, and `repetition_penalty=1.1` during text generation.
- **Terminal controls:** `/reset` clears the current conversation; `/exit` ends the session.
- **Bounded history:** Retains recent messages to limit how much conversation is passed back to the model.

## How it works

1. **Install dependencies** — install PyTorch's CPU build and Hugging Face Transformers.
2. **Download the model** — download the tokenizer and model weights from Hugging Face, then save them into `smollm2_model/`.
3. **Load locally** — reload the tokenizer and model from the local folder with network access disabled for this loading step.
4. **Prepare messages** — combine the system prompt, recent conversation history, and the latest user question.
5. **Format the prompt** — use the tokenizer's chat template to create model input.
6. **Generate a response** — call `model.generate()` under `torch.inference_mode()`.
7. **Decode the output** — return only newly generated tokens, rather than repeating the input prompt.
8. **Continue chatting** — store the latest user/assistant messages and allow the user to reset or exit.

## Technology stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| PyTorch (`torch`) | Loads and runs the language model |
| Hugging Face Transformers | Tokenizer, pretrained model, chat template, and text generation |
| SmolLM2-360M-Instruct | Small instruction-tuned language model |
| Jupyter Notebook | Development and execution environment |

## Project structure

After running the notebook's download cell, the project will look similar to this:

```text
offline-genai-chatbot/
├── AI_Project.ipynb       # Main notebook: setup, model loading, generation, and chat loop
├── README.md              # Project documentation
└── smollm2_model/          # Created by the notebook; contains locally saved model/tokenizer files
    ├── config.json
    ├── generation_config.json   # May be present depending on Transformers/model version
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── ...                      # Other files saved by Hugging Face
```

The exact files inside `smollm2_model/` can vary by model and library version. Do not commit large model-weight files to GitHub unless you intentionally want to distribute them and have checked repository size limits and model licensing.

## Requirements

- Python 3.10 or a compatible Python version supported by the installed PyTorch and Transformers releases
- `pip`
- Jupyter Notebook or JupyterLab
- Internet access for the **initial model download**
- Sufficient RAM and disk space for the model and Python libraries

The notebook explicitly installs the CPU version of PyTorch. CPU inference is convenient for a simple local setup, but response speed depends on your computer and the length of the prompt and answer.

## Setup and installation

### 1. Clone or download the project

If you have uploaded the notebook to your own GitHub repository, clone it:

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
```

Or open `AI_Project.ipynb` directly in Jupyter.

### 2. Install dependencies

Run these commands in a notebook cell (the notebook uses `%pip`, which is a Jupyter/IPython command):

```python
%pip install --upgrade pip
%pip install torch --index-url https://download.pytorch.org/whl/cpu
%pip install --upgrade transformers
```

If you run the project outside Jupyter, use your environment's package installer instead. The `%pip` syntax will not work in a standard Python script or ordinary terminal.

### 3. Download and save the model

Run the notebook cell that defines:

```python
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
MODEL_DIR = Path("smollm2_model")
```

That cell downloads the model and tokenizer and saves them to `smollm2_model/`. This step requires internet access and can take some time depending on your connection.

### 4. Run local inference

Run the cell that loads the tokenizer and model with `local_files_only=True`. It loads the saved files from disk and configures the model for CPU inference.

### 5. Start chatting

Run the notebook cells in order, including the cell that defines `generate_response()` and the final cell that calls `start_chat()`.

Example interaction:

```text
==================================================
       OFFLINE GENAI CHATBOT
==================================================
Ask any question.
Type /reset to clear history.
Type /exit to stop.
==================================================

You: Explain Generative AI in simple terms.
Bot: ...
```

The model's exact response will vary from run to run.

## Using the chatbot

| Input | Result |
|---|---|
| Ask a question | Generates a response |
| Ask a follow-up question | Uses recent conversation history when available |
| `/reset` | Clears conversation history for the current session |
| `/exit` | Ends the chat loop |

The notebook includes a `generate_response(user_message, history=None)` function that can also be called from other Python code after the model and tokenizer have been loaded.

## Configuration

The notebook currently defines these generation settings:

| Setting | Current value | Purpose |
|---|---:|---|
| `MAX_NEW_TOKENS` | `120` | Maximum number of new tokens generated per answer |
| `MAX_HISTORY_MESSAGES` | `6` | Number of recent history entries selected by `generate_response()` |
| `temperature` | `0.7` | Controls randomness when sampling is enabled |
| `top_p` | `0.9` | Nucleus-sampling threshold |
| `repetition_penalty` | `1.1` | Discourages repetitive text |
| Device | CPU | Runs inference without requiring CUDA |

**History note:** The interactive loop also limits its stored history to 12 message entries. The generation function selects the last six entries from that history list. These are message entries (user or assistant messages), not six complete question-and-answer pairs.

## Limitations

- **Model quality:** SmolLM2-360M-Instruct is a compact model. It can produce incorrect, incomplete, or inconsistent answers and may struggle with complex reasoning.
- **No live web access:** After downloading the model, the chatbot generates answers from the model's learned parameters and provided conversation; it does not search the internet.
- **No persistent memory:** Chat history exists only while the current chat loop is running. `/reset` clears it, and exiting the program ends the session.
- **Initial internet required:** The first model download requires a connection. The inference-loading cell uses local files, but this does not by itself guarantee that every dependency or notebook operation is fully offline.
- **CPU performance:** Generation may be slower on lower-powered computers.
- **No user interface beyond the terminal:** The current notebook does not implement a web app, desktop UI, or REST API.
- **No formal evaluation yet:** The notebook does not include a benchmark, automated tests, or a systematic quality evaluation.
- **No safety or factuality guarantee:** Generated answers should be checked, especially for important decisions.

## Troubleshooting

### Model files not found

Run the download-and-save cell first and confirm that `smollm2_model/` exists relative to the notebook's current working directory.

### Hugging Face model download fails

Check your internet connection, retry the download, and confirm that the model repository is accessible. Ensure you have enough free disk space.

### Package or version conflicts

Use a dedicated virtual environment and install compatible versions of PyTorch and Transformers. Restart the notebook kernel after changing core dependencies.

### Slow responses

The project runs on CPU and generates up to 120 new tokens per response. Reduce `MAX_NEW_TOKENS` for shorter responses, or consider suitable hardware and model optimizations if needed.

### Chat history seems short

The function passes at most six recent history entries to the model, while the terminal loop stores at most twelve entries. Older context is intentionally discarded.

## Future improvements

- Add a Streamlit or Gradio user interface.
- Move reusable logic into `src/chatbot.py` and add a separate application entry point.
- Add a `requirements.txt` with tested dependency versions.
- Add structured logging and clearer exception messages.
- Add unit tests for message preparation, history trimming, and generation behavior.
- Evaluate response quality on a small set of documented questions.
- Add configurable system prompts and generation parameters.
- Explore quantization or other CPU optimizations after measuring memory use and response speed.
- Add a privacy note explaining that prompts are processed locally during inference, while the initial model download contacts Hugging Face.

## Responsible use and privacy

The model runs locally after its files have been downloaded, and the notebook does not send each chat question to a hosted inference API. However, the initial download contacts Hugging Face, and this project does not implement a privacy audit or guarantee that other software on the computer cannot access local data. Avoid entering passwords, confidential information, or sensitive personal data.

## Author

**Karan Ambhore**

## License

No project license is specified in the notebook. Add a `LICENSE` file before publicly distributing the project, and review the model's license and usage terms on its Hugging Face model page before redistributing model files.

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [SmolLM2-360M-Instruct model page](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
- [PyTorch](https://pytorch.org/)
