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
            ["migrate", "collectstatic"],
        )
        self.assertEqual([item["name"] for item in web["containers"]], ["django"])
        self.assertIn("beat", beat["template"]["spec"]["containers"][0]["command"])

    def test_release_migrations_gate_all_application_processes(self):
        for upgrade in (False, True):
            with self.subTest(upgrade=upgrade):
                resources = render(upgrade=upgrade)
                self.assertFalse(any(kind in ("Job", "Role", "RoleBinding") for kind, _ in resources))
                self.assertIn(("ConfigMap", "test-migration-init"), resources)
                for name in ("test", "test-celery-worker", "test-celery-beat"):
                    pod = resources["Deployment", name]["spec"]["template"]["spec"]
                    # Explicitly clear both persisted service-account aliases on upgrades.
                    self.assertEqual(pod["serviceAccountName"], "default")
                    init = pod["initContainers"][0]
                    self.assertEqual(init["name"], "migrate")
                    self.assertEqual(init["image"], pod["containers"][0]["image"])
                    self.assertEqual(init["command"], ["python", "/etc/aqueduct/migrate.py"])
                    self.assertEqual({m["name"] for m in init["volumeMounts"]},
                                     {"settings-py", "router-config", "migration-init"})
                    self.assertEqual(init["env"], pod["containers"][0]["env"])
                    self.assertNotIn("sqlite-data", {v["name"] for v in pod["volumes"]})

    def test_role_overrides_match_application_environment_in_order(self):
        overrides = [
            [{"name": "POSTGRES_HOST", "value": "web-db"}, {"name": "SILKY_ENABLED", "value": "False"}],
            [{"name": "POSTGRES_HOST", "valueFrom": {"secretKeyRef": {"name": "worker", "key": "host"}}},
             {"name": "DATABASE_ENGINE", "value": "django.db.backends.postgresql"}],
            [{"name": "POSTGRES_DB", "value": "beat-db"}, {"name": "SILKY_ENABLED", "value": "True"}],
        ]
        resources = render({"web": {"extraEnv": overrides[0], "databasePool": {"maxSize": 12}},
                            "celery": {"workerExtraEnv": overrides[1], "beatExtraEnv": overrides[2]}})
        for name, extra in zip(("test", "test-celery-worker", "test-celery-beat"), overrides):
            pod = resources["Deployment", name]["spec"]["template"]["spec"]
            env = pod["containers"][0]["env"]
            self.assertEqual(env[-len(extra):], extra)
            self.assertEqual(pod["initContainers"][0]["env"], env)

    def test_rejects_migration_only_overrides(self):
        for name in ("SILKY_ENABLED", "POSTGRES_HOST"):
            with self.subTest(name=name), self.assertRaises(subprocess.CalledProcessError):
                render({"migrate": {"extraEnv": [{"name": name, "value": "other"}]}})

    def test_removed_migration_job_options_are_rejected(self):
        for key, value in (("databasePool", {}), ("scheduling", {}), ("activeDeadlineSeconds", 60)):
            with self.subTest(key=key):
                with self.assertRaises(subprocess.CalledProcessError) as error:
                    render({"migrate": {key: value}})
                self.assertIn(f"migrate.{key}", error.exception.stderr)

    def test_long_valid_release_names_have_distinct_migration_configs(self):
        names = []
        for suffix in ("a", "b"):
            resources = render(release="a" * 52 + suffix)
            # Locate the migration ConfigMap by its script rather than its truncated suffix.
            configs = [name for kind, name in resources if kind == "ConfigMap" and
                       "migrate.py" in resources[kind, name].get("data", {})]
            self.assertEqual(len(configs), 1)
            self.assertLessEqual(len(configs[0]), 63)
            names.extend(configs)
        self.assertNotEqual(*names)

    def test_database_budget_is_per_role_and_worker_rollouts_are_bounded(self):
        resources = render()
        for kind, name, maximum in [
            ("Deployment", "test", "10"),
            ("Deployment", "test-celery-worker", "2"),
            ("Deployment", "test-celery-beat", "2"),
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

    def test_legacy_pool_env_overrides_are_rejected(self):
        for values, path in [
            ({"env": [{"name": "POSTGRESQL_POOL_MIN_SIZE", "value": "1"}]}, "env"),
            ({"web": {"extraEnv": [{"name": "POSTGRESQL_POOL_MAX_SIZE", "value": "1"}]}}, "web.extraEnv"),
            ({"celery": {"workerExtraEnv": [{"name": "POSTGRESQL_POOL_MAX_SIZE", "value": "1"}]}}, "celery.workerExtraEnv"),
            ({"celery": {"beatExtraEnv": [{"name": "POSTGRESQL_POOL_MIN_SIZE", "value": "1"}]}}, "celery.beatExtraEnv"),
            ({"migrate": {"extraEnv": [{"name": "POSTGRESQL_POOL_MAX_SIZE", "value": "1"}]}}, "migrate.extraEnv"),
        ]:
            with self.subTest(path=path), self.assertRaises(subprocess.CalledProcessError) as error:
                render(values)
            self.assertIn(path, error.exception.stderr)
            self.assertIn("databasePool", error.exception.stderr)

        render({"web": {"extraEnv": [{"name": "POSTGRESQL_POOL_TIMEOUT", "value": "30"}]}})

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
            "migrate": {"resources": limits},
        })
        for kind, name in [
            ("Deployment", "test"), ("Deployment", "test-celery-worker"),
            ("Deployment", "test-celery-beat"),
        ]:
            with self.subTest(name=name):
                pod = resources[kind, name]["spec"]["template"]["spec"]
                for key, value in scheduling.items():
                    self.assertEqual(pod[key], value)
                self.assertEqual(pod["containers"][0]["resources"], limits)
                self.assertEqual(pod["initContainers"][0]["resources"], limits)
                default_pod = render()[kind, name]["spec"]["template"]["spec"]
                self.assertNotIn("affinity", default_pod)

    def test_public_model_limits_use_application_env_without_router_redis(self):
        limits = {"name": "AQUEDUCT_MODEL_RPM_LIMITS", "value": '{"public":10}'}
        resources = render({"web": {"replicas": 2}, "env": [
            {"name": "DATABASE_ENGINE", "value": "django.db.backends.postgresql"}, limits,
        ]})
        for kind, name in [
            ("Deployment", "test"), ("Deployment", "test-celery-worker"),
            ("Deployment", "test-celery-beat"),
        ]:
            container = resources[kind, name]["spec"]["template"]["spec"]["containers"][0]
            self.assertIn(limits, container["env"])
            self.assertNotIn("LITELLM_REDIS_URL", {item["name"] for item in container["env"]})


if __name__ == "__main__":
    unittest.main()
