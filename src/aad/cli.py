"""CLI entry point for the AAD agent. Run with: python -m aad.cli

Interactive loop: type queries, get answers. Ctrl+C or /quit to exit.
"""

import sys

from aad.config import Settings
from aad.store import AADStore
from aad.vector_index import VectorIndex
from aad.embedder import Embedder
from aad.agent import AADAgent
from aad.seed import seed_store


def main() -> None:
    """Entry point: load/init store, build index, seed if empty, start agent loop."""
    settings = Settings()

    # Validate API key
    if not settings.deepseek_api_key:
        print("ERROR: DEEPSEEK_API_KEY not set. Create a .env file or export it.")
        print("  cp .env.example .env  # then edit with your key")
        sys.exit(1)

    # Initialize components
    store = AADStore(settings.store_path)
    embedder = Embedder(dim=settings.embedding_dim)
    index = VectorIndex(dim=settings.embedding_dim)

    # Seed if empty
    if len(store) == 0:
        print("Store is empty. Generating seed data...")
        count = seed_store(store, embedder)
        print(f"Seeded {count} nodes.")

    # Rebuild FAISS index from store
    index.rebuild(store._nodes)

    # Initialize agent
    agent = AADAgent(
        store=store,
        index=index,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_chat_model,
    )

    print(f"AAD Agent ready. {len(store)} nodes loaded. Chat model: {settings.deepseek_chat_model}")
    print("Type your questions. /quit to exit, /nodes to list all nodes.\n")

    # Interactive loop
    try:
        while True:
            user_input = input("> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("/quit", "/exit", "/q"):
                print("Goodbye.")
                break
            if user_input.lower() == "/nodes":
                names = store.list_names()
                print(f"Nodes ({len(names)}): {', '.join(sorted(names))}")
                continue

            print()
            try:
                answer = agent.run(user_input)
                print(answer)
            except Exception as exc:
                print(f"Agent error: {exc}")
            print()

    except KeyboardInterrupt:
        print("\nGoodbye.")


if __name__ == "__main__":
    main()
