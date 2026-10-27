"""Set or change the dashboard chatbot's trading PIN (4-12 digits). Run it yourself:  python chatbot/set_pin.py
Only a salted hash is saved (.chat_pin.json, gitignored). Changing it needs the old PIN; if you forgot it, delete
.chat_pin.json and run this again."""
import getpass
import sys

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chatbot import pin as chat_pin

def main():
    if chat_pin.is_set():
        ok, msg = chat_pin.verify(getpass.getpass("Current PIN: "))
        if not ok:
            sys.exit(msg)
    pin = getpass.getpass("New PIN (4-12 digits): ")
    if pin != getpass.getpass("Repeat new PIN: "):
        sys.exit("The two PINs differ; nothing changed.")
    try:
        chat_pin.set_pin(pin)
    except ValueError as e:
        sys.exit(str(e))
    print(f"PIN saved (hashed) in {chat_pin.PIN_FILE}.")

if __name__ == "__main__":
    main()
