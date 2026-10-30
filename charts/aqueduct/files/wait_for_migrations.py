"""Block application startup until this Helm revision's migration Job completes."""

import json
import os
import ssl
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def wait_for_migrations(job: str) -> None:
    account = Path("/var/run/secrets/kubernetes.io/serviceaccount")
    namespace = (account / "namespace").read_text().strip()
    context = ssl.create_default_context(cafile=str(account / "ca.crt"))
    host = os.environ["KUBERNETES_SERVICE_HOST"]
    port = os.environ["KUBERNETES_SERVICE_PORT_HTTPS"]
    url = f"https://{host}:{port}/apis/batch/v1/namespaces/{namespace}/jobs/{job}"
    while True:
        request = Request(
            url, headers={"Authorization": f"Bearer {(account / 'token').read_text().strip()}"}
        )
        try:
            with urlopen(request, context=context, timeout=10) as response:
                status = json.load(response).get("status", {})
        except HTTPError as error:
            if error.code != 404:
                raise
        except (URLError, TimeoutError) as error:
            print(f"Waiting for Kubernetes API: {error}", flush=True)
        else:
            conditions = status.get("conditions", [])
            if any(c["type"] == "Failed" and c["status"] == "True" for c in conditions):
                raise RuntimeError(f"Migration Job {job} failed; inspect its logs before retrying")
            if any(c["type"] == "Complete" and c["status"] == "True" for c in conditions):
                print(f"Migration Job {job} completed", flush=True)
                return
        print(f"Waiting for migration Job {job}", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    wait_for_migrations(sys.argv[1])
