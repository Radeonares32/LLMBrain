class AuthService:
    def login(self, username, password):
        if username == "admin" and password == "secret":
            return {"token": "sample_token"}
        return None
