# utils/collections.py

def build_collection_name(subject: str, topic: str) -> str:
    """
    Standardizes collection names for ChromaDB.
    Converts subject+topic into lowercase, underscores instead of spaces.
    """
    return f"{subject.strip().lower()}_{topic.strip().lower()}".replace(" ", "_")
