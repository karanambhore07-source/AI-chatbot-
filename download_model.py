from transformers import AutoModelForCausalLM, AutoTokenizer

from chatbot import MODEL_DIR, MODEL_ID


def main() -> None:
    print(f"Downloading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR, safe_serialization=True)
    print(f"Model downloaded to {MODEL_DIR}")


if __name__ == "__main__":
    main()
