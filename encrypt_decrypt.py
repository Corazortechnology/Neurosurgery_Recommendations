from cryptography.fernet import Fernet 
key = Fernet.generate_key()
fernet = Fernet(key)
class EncryptDecrypt:
    def __init__(self,text):
        self.text = text
    def encrypt(self):
        encoded_text = fernet.encrypt(self.text.encode())
        return encoded_text

    def decrypt(self,encoded_text):
        decoded_text = fernet.decrypt(encoded_text).decode()
        return decoded_text

# encoder_decoder = EncryptDecrypt("hello")
# encrypt_text = encoder_decoder.encrypt()
# print(encrypt_text)
# decrypt_text = encoder_decoder.decrypt(encrypt_text)
# print(decrypt_text)

