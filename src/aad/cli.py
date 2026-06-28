"""CLI entry point for the AAD agent. Run with: python -m aad.cli

Interactive loop with persistent long-term store and session-scoped short-term memory.
--reseed to force regeneration of seed data.
"""

import sys

from aad.config import Settings
from aad.store import AADStore
from aad.vector_index import VectorIndex
from aad.embedder import Embedder
from aad.session import SessionMemory
from aad.agent import AADAgent
from aad.seed import seed_store


def main() -> None:
    settings = Settings()
    reseed = "--reseed" in sys.argv

    if not settings.deepseek_api_key:
        print("ERROR: DEEPSEEK_API_KEY not set.")
        print("  cp .env.example .env  # then edit with your key")
        sys.exit(1)

    # ── Long-term store ──────────────────────────────────────────
    store = AADStore(settings.store_path)
    embedder = Embedder(dim=settings.embedding_dim)
    index = VectorIndex(dim=settings.embedding_dim)

    if reseed or len(store) == 0:
        if reseed:
            store.clear()
            print("Reseed requested. Regenerating seed data...")
        else:
            print("Store is empty. Generating seed data...")
        count = seed_store(store, embedder)
        print(f"Seeded {count} nodes.")

    index.rebuild(store._nodes)

    # ── Short-term session memory ────────────────────────────────
    session = SessionMemory(embedder, settings.embedding_dim)

    # ── Agent ────────────────────────────────────────────────────
    agent = AADAgent(
        store=store, index=index,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_chat_model,
    )

    print(f"AAD Agent ready. {len(store)} long-term nodes, "
          f"model: {settings.deepseek_chat_model}")
    print("/quit /q 退出  /nodes 长期节点  /session 短期状态  /reseed 重生数据\n")

    # ── Interactive loop ─────────────────────────────────────────
    try:
        while True:
            user_input = input("> ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("/quit", "/exit", "/q"):
                session.clear()
                print(f"会话结束。短期记忆已清除（{session.node_count} 节点）。Goodbye.")
                break

            if user_input.lower() == "/nodes":
                names = store.list_names()
                print(f"长期节点 ({len(names)}): {', '.join(sorted(names))}")
                continue

            if user_input.lower() == "/session":
                print(session.summary())
                print(f"  镜像节点: {session.mirrored_count}")
                continue

            print()
            try:
                answer = agent.run(user_input, session=session)
                print(answer)
            except Exception as exc:
                print(f"Agent error: {exc}")
            print()

    except KeyboardInterrupt:
        session.clear()
        print(f"\n会话结束。短期记忆已清除。Goodbye.")


if __name__ == "__main__":
    main()
