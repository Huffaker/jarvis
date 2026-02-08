from agent_core import answer

print("Agent ready. Type 'exit' or 'quit' to quit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in {"exit", "quit"}:
        break

    response, sources = answer(user_input)
    print("\n🤖", response, "\n")
    if sources:
        print("Web sources:")
        for url in sources:
            print(f" - {url}")
        print()
