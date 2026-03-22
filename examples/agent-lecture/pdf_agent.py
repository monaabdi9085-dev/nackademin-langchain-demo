from langchain.agents import create_agent

from util.models import get_model
from util.pretty_print import get_user_input


def run():
    model = get_model(
        temperature=0.5,
        top_p=0.5
    )

    agent = create_agent(
    model=model,
    system_prompt="""
Du är en PDF-agent som specialiserar sig på att strukturera och sammanfatta långa PDF-dokument.

Din uppgift är att:
1. Identifiera det viktigaste innehållet
2. Sammanfatta tydligt
3. Strukturera med rubriker och punktlistor

Regler:
- Hitta inte på fakta
- Ta bort onödig information
- Utgå endast från texten

Svara alltid på svenska.
"""
)  

    conversation = []

    while True:
        user_input = get_user_input("Klistra in text från PDF som du vill sammanfatta")

        if not user_input or user_input.lower() == "exit":
            print("PDF agenten avslutas ")
            break

        conversation.append({"role": "user", "content": user_input})

        response = agent.invoke({"messages": conversation})

        assistant_message = response["messages"][-1].content
        print(f"\nAgent: {assistant_message}\n")

        conversation.append({"role": "assistant", "content": assistant_message})


if __name__ == "__main__":
    run()