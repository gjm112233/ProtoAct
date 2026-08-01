import os, json
from typing import Sequence
from langchain_core.messages import message_to_dict, messages_from_dict, BaseMessage
# message_to_dict: single message object (BaseMessage class instance) -> dictionary
# messages_from_dict: [dictionary, dictionary, ...] -> [message, message]
from langchain_core.chat_history import BaseChatMessageHistory


class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        self.session_id = session_id  # Session id, used as the file name
        self.storage_path = storage_path  # Folder path for storage files with different session ids
        # Build the complete file path
        self.file_path = os.path.join(self.storage_path, self.session_id)
        # Ensure the folder exists
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    # Add messages
    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        # Sequence is similar to list or tuple.

        # self.messages comes from the parent class property and records existing messages.
        all_messages = list(self.messages)
        # Add new messages to the message list.
        all_messages.extend(messages)

        # Synchronize data to the local file.
        # Class objects are written to files in binary form.
        # For readability, convert BaseMessage messages to dictionaries.
        new_messages = []
        for message in all_messages:
            d = message_to_dict(message)  # Single message object (BaseMessage class instance) -> dictionary
            new_messages.append(d)  # Add the converted dictionary to the list.

        # Write data to the file.
        with open(self.file_path, 'w', encoding="utf-8") as f:
            json.dump(new_messages, f)  # Convert all list content to JSON and write it to the file.

    # Get messages
    @property  # Use the decorator to turn the messages method into an instance property.
    def messages(self) -> list[BaseMessage]:
        try:
            with open(self.file_path, 'r', encoding="utf-8") as f:
                # In the current file: list[dictionary]. Dictionaries need to be converted to BaseMessage objects.
                messages_data = json.load(f)
                # This function converts [dictionary, dictionary, ...] -> [message, message].
                return messages_from_dict(messages_data)

        except FileNotFoundError:
            return []

    def clear(self) -> None:
        with open(self.file_path, 'w', encoding="utf-8") as f:
            # Clear the file.
            json.dump([], f)


# Get a FileChatMessageHistory instance so messages can be inserted, retrieved, or cleared.
def get_history(session_id):
    return FileChatMessageHistory(session_id, "./chat_history")
