#from comfyui import generate_image
from image_gen.diffusion import fast_generate

print("Agent ready. Type 'exit' or 'quit' to quit.\n")

try:
    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break

        if user_input.lower() in {"exit", "quit"}:
            break

        result = fast_generate(user_input, "out.png")
        print("\n🤖 [Image Generated]\n")
except KeyboardInterrupt:
    print("\nBye.")
