from pathlib import Path
from threading import Lock
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = Path(__file__).resolve().parent / "smollm2_model"
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
MAX_NEW_TOKENS = 120
MAX_HISTORY_MESSAGES = 6

SYSTEM_PROMPT = (
    "You are Karan Ambhore's personal AI assistant. Help Karan prepare clear, "
    "well-organized reports; summarize information he provides; draft writing; "
    "and organize ideas, priorities, and plans. When asked for a report, use a "
    "suitable title and concise sections. Treat his notes as the only source: "
    "include only facts explicitly provided, keep planned work in the future, "
    "and do not add generic claims, outcomes, recommendations, or details. "
    "If the notes are brief, keep the report brief. Explain unfamiliar topics "
    "in simple English. Never invent facts, sources, completed work, or personal "
    "details; clearly say when information is missing or uncertain. Ask a "
    "focused question if essential details are missing. You only know what "
    "Karan shares in this "
    "conversation: do not claim access to his files, email, calendar, accounts, "
    "or ability to take actions outside this chat. Protect personal information."
)

_MODEL_LOCK = Lock()
_tokenizer: Any = None
_model: Any = None


def format_report(user_message: str) -> str:
    notes = user_message.strip()[len("/report") :].strip()
    if not notes:
        return (
            "Add your source notes after `/report`. For example:\n"
            "/report\nTitle: Project update\n- Website redesign finished\n"
            "- Seven tests passed\n- Next step: mobile testing"
        )

    lines = [line.strip() for line in notes.splitlines() if line.strip()]
    title = "Report"
    if lines and lines[0].lower().startswith("title:"):
        provided_title = lines.pop(0).partition(":")[2].strip()
        if provided_title:
            title = provided_title

    facts = [line.lstrip("-*• ").strip() for line in lines]
    facts = [line for line in facts if line]
    if not facts:
        return (
            "Add at least one source fact after `/report`. "
            "Only the facts you provide will be included."
        )

    return "\n".join(
        [
            title,
            "=" * len(title),
            "",
            "Provided notes:",
            *(f"- {fact}" for fact in facts),
            "",
            "Formatted only from the notes you provided.",
        ]
    )


def _load_model() -> tuple[Any, Any]:
    global _model, _tokenizer

    if _model is None or _tokenizer is None:
        if not MODEL_DIR.is_dir():
            raise FileNotFoundError(
                f"Model files were not found in {MODEL_DIR}. Run `python download_model.py` first."
            )

        torch.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )
        model.to("cpu")
        model.eval()
        _tokenizer = tokenizer
        _model = model

    return _tokenizer, _model


def generate_response(user_message: str, history: list[dict[str, str]] | None = None) -> str:
    recent_history = (history or [])[-MAX_HISTORY_MESSAGES:]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *recent_history,
        {"role": "user", "content": user_message},
    ]

    with _MODEL_LOCK:
        tokenizer, model = _load_model()
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        answer = tokenizer.decode(
            output_ids[0][input_length:],
            skip_special_tokens=True,
        ).strip()

    return answer or "I could not generate a response. Please try again."
