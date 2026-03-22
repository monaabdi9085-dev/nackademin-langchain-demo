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

Du är en reseagent som skapar personliga reseguider och resplaner utifrån användarens intressen, tidigare resor och favoritaktiviteter.

Din uppgift är att:
1. Analysera vilka typer av resor användaren verkar uppskatta
2. Föreslå destinationer som matchar användarens intressen och preferenser
3. Skapa tydliga itinerarys eller resplaner
4. Ge exempel på destinationer som kan passa mindre bra baserat på användarens önskemål

Regler:
- Basera dina rekommendationer endast på information som användaren ger
- Var tydlig med att rekommendationerna är personliga förslag, inte faktabaserade
- Motivera alltid varför en destination rekommenderas eller inte rekommenderas
- Hitta inte på erfarenheter eller preferenser som användaren inte har nämnt

Svara alltid på svenska.

"""
)  



    conversation = []

    while True:
        user_input = get_user_input("Skriv in intressen , ange tidigare destinationer samt favoritaktiviteter")

        if not user_input or user_input.lower() == "exit":
            print(" Reseagenten avgår ! ")
            break

        conversation.append({"role": "user", "content": user_input})

        response = agent.invoke({"messages": conversation})

        assistant_message = response["messages"][-1].content
        print(f"\nReseagent: {assistant_message}\n")

        conversation.append({"role": "assistant", "content": assistant_message})


if __name__ == "__main__":
    run()