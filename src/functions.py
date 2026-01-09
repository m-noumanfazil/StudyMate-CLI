import os
import time
from classes import vectordb, RAGAssistant

def clear_screen(delay: float = 2.0):
    """
    Clears the terminal screen.
    Optional delay lets user read the last message before screen resets.
    """
    if delay > 0:
        time.sleep(delay)

    os.system("cls" if os.name == "nt" else "clear")

def create_session(db, assistant):
    """
    Loops until user enters a unique session name.
    Creates session when valid.
    Clears screen after creation.
    """
    while True:
        session_name = input("Enter the session name (no space allowed. Only small letters): ").strip()
        if not session_name:
            print("Session name cannot be empty. Try again.\n")
            continue

        success = db.create_session(session_name)
        if success:
            print(f"Session '{session_name}' created successfully!")
            break
        else:
            print(f"Session '{session_name}' already exists. Try a different name.\n")
    clear_screen()


def list_sessions(db, assistant):
    db.list_session()    

def add_documents(db, assistant):
    list_sessions(db, assistant)
    documents_list = []
    
    while True:
        session_name = str(input("Write the session name to add documents (pdfs) in it: "))
        if db.get_session(session_name) is None:
            print("Session name not exist.")
            continue
        break

    while True:
        doc_name = str(input("Write the session name to add documents in it or write done to stop adding documents: "))
        if str.lower(doc_name) == "done":
           break
        documents_list.append(doc_name)
        print("Document path added to bucket.")
    print("Adding Documents in the session.")
    db.add_file(documents_list, session_name)
    print(f"All documents added to the {session_name} session.")
    clear_screen()
 
def start_chat(db, assistant):
    list_sessions(db, assistant)
    
    while True:
        session_name = str(input("Write the session name to start the chat: "))
        if db.get_session(session_name) is None:
            print("Session name not exist.")
            continue
        break
    
    while True:
        question = str(input("Ask (or type quit to stop): "))
        if str.lower(question) == "quit":      
           print("Going back to main menu.")
           break
        response = assistant.query(session_name, question)
        print(f"Answer: {response}")
    clear_screen() # Call the function associated with the choice

def delete_session(db, assistant):
    list_sessions(db, assistant)
    while True:
        session_name = str(input("Write the session name to delete that session: "))
        if db.get_session(session_name) is None:
            print("Session name not exist.")
            continue
        break
    db.delete_session(session_name)
    clear_screen() # Call the function associated with the choice

    