"""Example: Generate RAG framework adapter code."""

from quantvec.adapters import (
    generate_langchain_adapter,
    generate_llamaindex_adapter,
    generate_haystack_adapter,
)


def main():
    print("=" * 60)
    print("QuantVec-CLI RAG Adapter Generator Example")
    print("=" * 60)

    # LangChain adapter
    print("\n1. LangChain Adapter (TurboQuant, 4-bit, 768D)")
    print("-" * 60)
    code = generate_langchain_adapter("turboquant", 4, 768)
    print(code[:500] + "...")

    # LlamaIndex adapter
    print("\n2. LlamaIndex Adapter (Scalar, 2-bit, 512D)")
    print("-" * 60)
    code = generate_llamaindex_adapter("scalar", 2, 512)
    print(code[:500] + "...")

    # Haystack adapter
    print("\n3. Haystack Adapter (Product, 8-bit, 1024D)")
    print("-" * 60)
    code = generate_haystack_adapter("product", 8, 1024)
    print(code[:500] + "...")

    print("\n" + "=" * 60)
    print("Use `quantvec adapter --framework <name>` to generate full code!")
    print("=" * 60)


if __name__ == "__main__":
    main()
