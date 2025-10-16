from sentence_transformers import SentenceTransformer

def main():
    # Modell laden (aus Cache, wenn schon vorhanden)
    model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
    print(f"Model loaded: distiluse-base-multilingual-cased-v2")
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Testeingaben
    sentences = [
        "Wie groß ist die Datei?",
        "The quick brown fox jumps over the lazy dog.",
        "DataBotti is my favorite project."
    ]

    embeddings = model.encode(sentences)

    for sent, emb in zip(sentences, embeddings):
        print(f"\nSentence: {sent}")
        print(f"Vector length: {len(emb)}")
        print(f"First 5 dims: {emb[:5]}")  # zur Übersicht nur die ersten 5 Werte

if __name__ == "__main__":
    main()