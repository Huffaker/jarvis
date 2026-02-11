from diffusion import fast_generate

print("Agent ready. Type 'exit' or 'quit' to quit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in {"exit", "quit"}:
        break

    fast_generate(user_input, "out.png")
    print("\n🤖 [Image Generated] \n")
