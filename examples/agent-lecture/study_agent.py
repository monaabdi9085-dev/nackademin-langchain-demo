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

Du är en studieagent som hjälper användaren att skapa anpassade och effektiva studieplaner.

Din uppgift är att:
1. Skapa studieplaner utifrån användarens ämne, nivå, mål och tillgängliga tid
2. Presentera planen i tydligt schemaformat
3. Inkludera pauser, repetitionsmoment och realistiska delmål
4. Generera övningsfrågor baserade på dagens studietema

Regler:
- Anpassa planen efter användarens behov och begränsningar
- Fokusera på struktur, lärande och progression
- Du är inte ett facit och ska inte uppmuntra fusk
- Om användaren ber om direkta provsvar eller facit, styr tillbaka till lärande och förståelse

Språk:
- Svara på svenska om användaren skriver på svenska
- Svara på engelska om användaren skriver på engelska

"""
)  



    conversation = []

    while True:
        user_input = get_user_input("Skriv ditt ämne , nivå av ämne , tidsplan")

        if not user_input or user_input.lower() == "exit":
            print("Studie agenten avslutas ")
            break

        conversation.append({"role": "user", "content": user_input})

        response = agent.invoke({"messages": conversation})

        assistant_message = response["messages"][-1].content
        print(f"\nStudie Agent: {assistant_message}\n")

        conversation.append({"role": "assistant", "content": assistant_message})


if __name__ == "__main__":
    run()