import requests
from config import HF_API_KEY

MODEL_ID = "facebook/bart-large-mnli"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

CATEGORIES = ["Spam", "Safe"]


def ask_hf(message: str):
    payload = {"inputs": message, "parameters": {"candidate_labels": CATEGORIES}}
    r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
    if not r.ok:
        raise RuntimeError(f"HF Error: {r.status_code} {r.text}")
    data = r.json()

    if isinstance(data, dict) and "labels" in data and "scores" in data:
        return [
            {"label": label, "score": score}
            for label, score in zip(data["labels"], data["scores"])
        ]
    return data


def best_category(preds: list):
    best = max(preds, key=lambda x: x["score"])
    return best["label"], best["score"]


def bar(score: float) -> str:
    pct = score * 100
    blocks = int(pct // 10)
    return "█" * blocks + "░" * (10 - blocks)


def show(message: str, preds: list):
    top_label, top_score = best_category(preds)
    print("\n" + "=" * 60)
    print("Spam vs Safe Message Classifier")
    print("=" * 60)
    print("Message:", message)
    print(f"Classification: {top_label}")
    print(f"Confidence: {round(top_score * 100, 1)}% [{bar(top_score)}]")
    print("\nAll Predictions:")
    sorted_preds = sorted(preds, key=lambda x: x["score"], reverse=True)
    for i, p in enumerate(sorted_preds, start=1):
        print(f"{i}. {p['label']:<11} {round(p['score'] * 100, 1)}% [{bar(p['score'])}]")
    print("=" * 60)


def main():
    print("Welcome! Type a message and I will classify it as Spam or Safe.")
    print("Categories:", ", ".join(CATEGORIES))
    print("\nType 'exit' to quit the program\n")

    while True:
        message = input("Message: ").strip()
        if message.lower() == "exit":
            print("Bye! Keep coding!")
            break
        if not message:
            print("Please type a message (not empty).\n")
            continue

        try:
            preds = ask_hf(message)
            if isinstance(preds, list) and preds and "label" in preds[0]:
                show(message, preds)
            else:
                print("Unexpected response from Hugging Face API:", preds)
        except Exception as e:
            print("Oops! Something went wrong. Please try again later.")
            print("Reason:", str(e))
            print("Tips: Check HF_API_KEY + internet.\n")


if __name__ == "__main__":
    main()