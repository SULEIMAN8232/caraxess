from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


def main():
    model = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
    )

    response = model.invoke(
        [
            HumanMessage(
                content="Explain embeddings in two simple sentences."
            )
        ]
    )

    print("\nCareerPilot AI — LLM Test")
    print("-" * 35)
    print(response.content)


if __name__ == "__main__":
    main()