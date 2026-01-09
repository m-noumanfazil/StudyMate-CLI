from src.classes import vectordb, RAGAssistant
from src.functions import create_session, add_documents, list_sessions, start_chat, delete_session, clear_screen

db = vectordb()
assistant = RAGAssistant(db)

def print_menu():
    print("""
StudyMate CLI Menu
-----------------
1) Create new session
2) Add documents to a session
3) List all sessions
4) Select active session (chat)
5) Delete a session
6) Exit
""")
    

COMMANDS = {
    "1": create_session,
    "2": add_documents,
    "3": list_sessions,
    "4": start_chat,
    "5": delete_session
}

def main():
    print("Welcome to StudyMate")
    print("--------------------")
    while True:
        print_menu()
        choice = input("Enter your choice: ").strip()

        if choice == "6":
            print("Exiting StudyMate. Goodbye!")
            break

        action = COMMANDS.get(choice)
        if action:
            action(db, assistant) 
        else:
            print("Invalid option. Please try again.")
            clear_screen() # Call the function associated with the choice


if __name__ == "__main__":
    main()