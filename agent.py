from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from tools import (
    retrieve_course_material_tool,
    search_jobs_tool,
)


def run_agent(question: str) -> str:
    """Route student questions to course retrieval or career search tools."""

    if not question.strip():
        return "Please enter a question."

    model = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
    )

    tools = [
        retrieve_course_material_tool,
        search_jobs_tool,
    ]

    model_with_tools = model.bind_tools(tools)

    messages = [
        SystemMessage(
            content=(
                "You are CareerPilot, an AI learning and career assistant.\n\n"

                "COURSE QUESTIONS:\n"
                "- Use retrieve_course_material_tool for questions about course notes.\n"
                "- Base your answer on the retrieved passages.\n"
                "- Explain difficult concepts in simple language.\n\n"

                "JOB AND INTERNSHIP SEARCH:\n"
                "- Use search_jobs_tool for job or internship searches.\n"
                "- Extract the job topic or skill from the user's request.\n"
                "- Extract the requested city or location. "
                "If none is given, use India.\n"
                "- Extract minimum and maximum salary when provided.\n"
                "- If salary filters are not provided, use 0 for both.\n"
                "- Pass the extracted values to search_jobs_tool using its "
                "topic, location, minimum_salary, and maximum_salary arguments.\n"
                "- Show the job title, company, location, salary range, "
                "description, and application link from the tool results.\n"
                "- Never invent listings, salaries, or companies.\n"
                "- If salary is not disclosed, say so clearly.\n\n"

                "TOOL USAGE:\n"
                "- Call the appropriate tool when needed.\n"
                "- After receiving a tool result, answer the user directly.\n"
                "- Do not repeatedly call the same tool for the same request.\n"
                "- If the user asks about both course concepts and careers, "
                "use both tools when appropriate."
            )
        ),
        HumanMessage(content=question),
    ]

    # Allow multiple tool calls, with a limit to prevent endless loops.
    for _ in range(4):
        response = model_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return str(response.content)

        for tool_call in response.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]

            print(f"[Tool called: {name}]")

            if name == retrieve_course_material_tool.name:
                result = retrieve_course_material_tool.invoke(args)

            elif name == search_jobs_tool.name:
                result = search_jobs_tool.invoke(args)

            else:
                result = f"Unknown tool: {name}"

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )

        # If the tool was called, summarize its result directly.
        tool_results = [
            message.content
            for message in messages
            if isinstance(message, ToolMessage)
        ]

        if tool_results:
            summary_messages = [
                SystemMessage(
                    content=(
                        "You are CareerPilot. Answer the user's question using "
                        "the tool results below. For job searches, include job "
                        "title, company, location, salary range, description, "
                        "and application link when available. Do not invent "
                        "missing information."
                    )
                ),
                HumanMessage(
                    content=(
                        f"User question: {question}\n\n"
                        f"Tool results:\n{tool_results[-1]}"
                    )
                ),
            ]

            final_response = model.invoke(summary_messages)
            return str(final_response.content)

    return "I couldn't complete the request. Please try again."


if __name__ == "__main__":
    print("CareerPilot AI")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            answer = run_agent(question)
            print(f"\nCareerPilot: {answer}\n")
        except Exception as error:
            print(f"\nError: {error}\n")