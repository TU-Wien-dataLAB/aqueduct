"""Chart contract checks: uv run python -m unittest discover -s charts/tests."""

import subprocess
import unittest
from pathlib import Path

import yaml

CHART = Path(__file__).resolve().parents[1] / "aqueduct"


def render(values=None, *, upgrade=False, release="test"):
    command = ["helm", "template", release, str(CHART)]
    if upgrade:
        command.append("--is-upgrade")
    result = subprocess.run(
        [*command, "-f", "-"], input=yaml.safe_dump(values or {}),
        text=True, capture_output=True, check=True,
    )
    return {
        (resource["kind"], resource["metadata"]["name"]): resource
        for resource in yaml.safe_load_all(result.stdout) if resource
    }


class WebScalingTests(unittest.TestCase):
    def test_long_release_names_have_distinct_migration_jobs(self):
        names = []
        for suffix in ("one", "two"):
            resources = render(release=f"{'a' * 45}-{suffix}")
            jobs = [name for kind, name in resources if kind == "Job"]
            self.assertEqual(len(jobs), 1)
            self.assertLessEqual(len(jobs[0]), 63)
            names.extend(jobs)
        self.assertNotEqual(names[0], names[1])

    def test_web_replica_default_and_override(self):
        for values, expected in [({}, 1), ({"web": {"replicas": 2}}, 2)]:
            with self.subTest(replicas=expected):
                resources = render(values)
                self.assertEqual(resources["Deployment", "test"]["spec"]["replicas"], expected)
                service = resources["Service", "test-svc"]["spec"]
                self.assertEqual(service["selector"], {"app": "test", "component": "web"})
                self.assertNotIn("sessionAffinity", service)

    def test_beat_is_a_non_overlapping_singleton_independent_of_web(self):
        resources = render({"web": {"replicas": 2}})
        beat = resources["Deployment", "test-celery-beat"]["spec"]
        self.assertEqual(beat["replicas"], 1)
        self.assertEqual(beat["strategy"], {"type": "Recreate"})
        web = resources["Deployment", "test"]["spec"]["template"]["spec"]
        self.assertEqual(
            [item["name"] for item in web["initContainers"]],
            ["wait-for-migrations", "collectstatic"],
        )
        self.assertEqual([item["name"] for item in web["containers"]], ["django"])
        self.assertIn("beat", beat["template"]["spec"]["containers"][0]["command"])

    def test_release_migrations_gate_all_application_processes(self):
        for upgrade in (False, True):
            with self.subTest(upgrade=upgrade):
                resources = render(upgrade=upgrade)
                job = resources["Job", "test-migrate-1"]
                self.assertNotIn("helm.sh/hook", job["metadata"].get("annotations", {}))
                self.assertEqual(job["metadata"]["annotations"]["helm.sh/resource-policy"], "keep")
                self.assertNotIn("ttlSecondsAfterFinished", job["spec"])
                self.assertEqual(job["spec"]["backoffLimit"], 0)
                self.assertEqual(job["spec"]["activeDeadlineSeconds"], 600)
                job_pod = job["spec"]["template"]["spec"]
                self.assertEqual(job_pod["initContainers"][0]["name"], "wait-for-database")
                migration = job_pod["containers"][0]
                self.assertEqual(migration["command"], ["python", "manage.py", "migrate", "--noinput"])
                role = resources["Role", "test-migrate-1"]
                self.assertEqual(role["metadata"]["annotations"]["helm.sh/resource-policy"], "keep")
                binding = resources["RoleBinding", "test-migrate-1"]
                self.assertEqual(binding["metadata"]["annotations"]["helm.sh/resource-policy"], "keep")
                self.assertEqual(binding["roleRef"]["name"], "test-migrate-1")
                self.assertEqual(role["rules"], [{
                    "apiGroups": ["batch"], "resources": ["jobs"],
                    "resourceNames": ["test-migrate-1"], "verbs": ["get"],
                }])
                for name in ("test", "test-celery-worker", "test-celery-beat"):
                    pod = resources["Deployment", name]["spec"]["template"]["spec"]
                    self.assertEqual(pod["serviceAccountName"], "test-migration-gate")
                    gate = pod["initContainers"][0]
                    self.assertEqual(gate["name"], "wait-for-migrations")
                    self.assertEqual(gate["command"][-1], "test-migrate-1")
                    self.assertNotIn("migrate", [item["name"] for item in pod["initContainers"]])

    def test_database_budget_is_per_role_and_worker_rollouts_are_bounded(self):
        resources = render()
        for kind, name, maximum in [
            ("Deployment", "test", "10"),
            ("Deployment", "test-celery-worker", "2"),
            ("Deployment", "test-celery-beat", "2"),
            ("Job", "test-migrate-1", "2"),
        ]:
            pod = resources[kind, name]["spec"]["template"]["spec"]
            env = {entry["name"]: entry.get("value") for entry in pod["containers"][0]["env"]}
            self.assertEqual(env["POSTGRESQL_POOL_MIN_SIZE"], "0")
            self.assertEqual(env["POSTGRESQL_POOL_MAX_SIZE"], maximum)
            self.assertIn("requests", pod["containers"][0]["resources"])
        for name in ("test", "test-celery-worker"):
            strategy = resources["Deployment", name]["spec"]["strategy"]
            self.assertEqual(strategy["rollingUpdate"], {"maxSurge": 1, "maxUnavailable": 0})
        resources = render({
            "web": {"databasePool": {"maxSize": 12}},
            "celery": {"workerDeployment": {"databasePool": {"maxSize": 3}}},
        })
        web = resources["Deployment", "test"]["spec"]["template"]["spec"]["containers"][0]
        worker = resources["Deployment", "test-celery-worker"]["spec"]["template"]["spec"]["containers"][0]
        self.assertIn({"name": "POSTGRESQL_POOL_MAX_SIZE", "value": "12"}, web["env"])
        self.assertIn({"name": "POSTGRESQL_POOL_MAX_SIZE", "value": "3"}, worker["env"])

    def test_scheduling_and_resources_can_be_set_for_each_role(self):
        scheduling = {
            "nodeSelector": {"pool": "gateway"},
            "tolerations": [{"key": "gateway", "operator": "Exists"}],
            "affinity": {"nodeAffinity": {"preferredDuringSchedulingIgnoredDuringExecution": []}},
            "topologySpreadConstraints": [{
                "maxSkew": 1, "topologyKey": "kubernetes.io/hostname",
                "whenUnsatisfiable": "ScheduleAnyway",
                "labelSelector": {"matchLabels": {"app": "test", "component": "web"}},
            }],
        }
        limits = {"requests": {"cpu": "500m", "memory": "4Gi"}, "limits": {"memory": "5Gi"}}
        resources = render({
            "web": {**scheduling, "resources": limits},
            "celery": {
                "workerDeployment": {**scheduling, "resources": limits},
                "beatScheduling": scheduling, "beatResources": limits,
            },
            "migrate": {"scheduling": scheduling, "resources": limits},
        })
        for kind, name in [
            ("Deployment", "test"), ("Deployment", "test-celery-worker"),
            ("Deployment", "test-celery-beat"), ("Job", "test-migrate-1"),
        ]:
            with self.subTest(name=name):
                pod = resources[kind, name]["spec"]["template"]["spec"]
                for key, value in scheduling.items():
                    self.assertEqual(pod[key], value)
                self.assertEqual(pod["containers"][0]["resources"], limits)
                default_pod = render()[kind, name]["spec"]["template"]["spec"]
                self.assertNotIn("affinity", default_pod)

    def test_public_model_limits_use_application_env_without_router_redis(self):
        limits = {"name": "AQUEDUCT_MODEL_RPM_LIMITS", "value": '{"public":10}'}
        resources = render({"web": {"replicas": 2}, "env": [limits]})
        for kind, name in [
            ("Deployment", "test"), ("Deployment", "test-celery-worker"),
            ("Deployment", "test-celery-beat"), ("Job", "test-migrate-1"),
        ]:
            container = resources[kind, name]["spec"]["template"]["spec"]["containers"][0]
            self.assertIn(limits, container["env"])
            self.assertNotIn("LITELLM_REDIS_URL", {item["name"] for item in container["env"]})


if __name__ == "__main__":
    unittest.main()
