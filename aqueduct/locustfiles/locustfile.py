from locust import HttpUser, between, task


class GatewayUser(HttpUser):
    wait_time = between(1, 3)
    headers = {
        "Authorization": "Bearer sk-abc123",  # TODO: test that the token works
        "Content-Type": "application/json",
    }
    host = "http://localhost:8000/"

    # def on_start(self) -> None:
    #     # TODO: this should be loaded in the Django app - maybe in the docker-compose setup?
    #     #  It wouldn't work here, because locust runs outside of the Django context.
    #     # management.call_command("loaddata", "locust_fixture.json")
    #     ...

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
