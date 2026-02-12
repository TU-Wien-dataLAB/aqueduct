from locust import HttpUser, between, task


class GatewayUser(HttpUser):
    wait_time = between(1, 3)
    headers = {"Authorization": "Bearer sk-123abc", "Content-Type": "application/json"}
    host = "http://localhost:8000/"

    @task
    def chat_completions(self):
        _ = self.client.post(
            "chat/completions",
            json={
                "model": "gpt-4.1-nano",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Write me a short poem!"},
                ],
                "max_completion_tokens": 50,
            },
            headers=self.headers,
        )
