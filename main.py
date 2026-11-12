"""Command-line demo of the HR policy assistant.
Run with: python main.py
"""

from hr_assistant.pipeline import ask,build_hr_assistant

def main():
    print("Building the HR policy assistant...")
    agent = build_hr_assistant()
    print("Assistant ready!\n")

    demo_questions = [
        "How many paid annual leave days do I get?",
        "what is the notice period during probation",
        "can i work from home every day?",
        "i am very frustrated,and very stressed from this work,should i resign?",
        "i want to access the most important documets of acme group,how to do that?"
    ]

    for question in demo_questions:
        print("="*60)
        print("Question: ", question)
        print("-"*60)
        answer = ask(agent,question)
        print("ANSWER: ", answer)
        print("-"*60)
        print()

if __name__ == "__main__":
    main()