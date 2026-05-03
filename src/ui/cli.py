"""Minimal CLI chat interface.

Commands:
    /show       toggle showing retrieved context for the next answer
    /context    show retrieved context for the most recent answer
    /clear      reset conversation history
    /reset-cache  wipe the response cache
    /model NAME switch the LLM (e.g. /model phi3:mini)
    /stream on|off  toggle streaming mode
    /stats      print store + cache stats
    /help       show this help
    /exit       quit
"""
from __future__ import annotations

import sys
import time

from src.cache.response_cache import ResponseCache
from src.config import PRIMARY_LLM
from src.generate.llm import list_models
from src.generate.pipeline import RAGPipeline
from src.retrieve.retriever import Retriever


HELP = """\
Commands:
  /show          toggle showing retrieved context with each answer
  /context       show retrieved context for the most recent answer
  /clear         clear conversation history
  /reset-cache   wipe the response cache
  /model NAME    switch the LLM (default llama3.2:3b)
  /stream on|off toggle streaming mode (default on)
  /stats         print store + cache stats
  /help          this help
  /exit          quit
"""


def main() -> None:
    retriever = Retriever()
    cache = ResponseCache()
    pipeline = RAGPipeline(retriever=retriever, cache=cache, model=PRIMARY_LLM)

    history: list[dict] = []
    show_context = False
    streaming = True
    last_chunks = []

    print("Local Wikipedia RAG Assistant — type /help for commands, /exit to quit.\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.startswith("/"):
            cmd, *rest = q.split(maxsplit=1)
            arg = rest[0] if rest else ""
            if cmd in ("/exit", "/quit"):
                break
            if cmd == "/help":
                print(HELP)
                continue
            if cmd == "/show":
                show_context = not show_context
                print(f"context display: {'on' if show_context else 'off'}")
                continue
            if cmd == "/context":
                if not last_chunks:
                    print("(no context yet)")
                else:
                    for i, c in enumerate(last_chunks, 1):
                        print(f"\n[{i}] {c.title} — {c.section} ({c.type}) [{','.join(c.sources)} score={c.score:.3f}]")
                        print(f"    {c.url}")
                        print(f"    {c.text[:300]}{'…' if len(c.text) > 300 else ''}")
                continue
            if cmd == "/clear":
                history.clear()
                last_chunks = []
                print("conversation cleared.")
                continue
            if cmd == "/reset-cache":
                cache.clear()
                print("response cache cleared.")
                continue
            if cmd == "/model":
                if not arg:
                    print(f"current model: {pipeline.model}")
                    print(f"installed: {', '.join(list_models()) or '(ollama unreachable)'}")
                else:
                    pipeline.model = arg.strip()
                    print(f"model -> {pipeline.model}")
                continue
            if cmd == "/stream":
                streaming = arg.strip().lower() != "off"
                print(f"streaming: {'on' if streaming else 'off'}")
                continue
            if cmd == "/stats":
                stats = retriever.store.stats()
                print(f"store: {stats['chunks']} chunks across {stats['people']} people + {stats['places']} places")
                print(f"cache entries: {cache.size()}")
                continue
            print(f"unknown command: {cmd} (try /help)")
            continue

        t0 = time.perf_counter()
        if streaming:
            stub, stream = pipeline.answer_stream(q, history=history)
            last_chunks = stub.chunks
            print(f"[route: {stub.routing.target} — {stub.routing.rationale}]")
            print(f"[retrieved {len(stub.chunks)} chunks in {stub.timings_ms.get('retrieve_ms', 0):.0f}ms]")
            print("bot> ", end="", flush=True)
            collected = []
            t_gen = time.perf_counter()
            for piece in stream:
                sys.stdout.write(piece)
                sys.stdout.flush()
                collected.append(piece)
            print()
            answer = "".join(collected)
            stub.answer = answer
            stub.timings_ms["generate_ms"] = (time.perf_counter() - t_gen) * 1000
        else:
            ans = pipeline.answer(q, history=history)
            last_chunks = ans.chunks
            print(f"[route: {ans.routing.target} — {ans.routing.rationale}]")
            print(f"[retrieved {len(ans.chunks)} chunks in {ans.timings_ms.get('retrieve_ms', 0):.0f}ms; "
                  f"gen {ans.timings_ms.get('generate_ms', 0):.0f}ms"
                  f"{'; cached' if ans.cached else ''}]")
            print(f"bot> {ans.answer}")
            answer = ans.answer

        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})
        if show_context and last_chunks:
            print("\n--- context ---")
            for i, c in enumerate(last_chunks, 1):
                print(f"[{i}] {c.title} — {c.section} ({c.type}) {c.url}")
        print(f"[total {(time.perf_counter() - t0) * 1000:.0f}ms]\n")


if __name__ == "__main__":
    main()
