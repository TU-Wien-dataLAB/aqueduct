---
title: Deployment
nav_order: 4
---

# Deployment

> **TODO:**
> This guide will cover deploying Aqueduct in production using Helm on Kubernetes.
>
> **Planned topics:**
> - Writing and using a Helm chart for Aqueduct
> - Required configuration (secrets, settings.py for orgs etc.)
> - Scaling
> - Authentication (OIDC/Dex) configuration
> - Reverse proxy and HTTPS setup

## Access key expiration

Set `MAX_ACCESS_KEY_EXPIRATION_DAYS` to a positive integer to limit the expiration
of user and service-account keys. The default, `-1`, imposes no maximum. New or
edited keys with no expiration or a later expiration are capped at the current
time plus the configured number of days; earlier expirations are preserved.

Run `python manage.py migrate` with the new setting whenever enabling or reducing
the limit. This shortens existing keys with no expiration or an expiration beyond
the new deadline; it does not extend earlier expirations. The Helm chart and
Docker Compose run migrations during deployment. All application and migration
processes must use the same setting. In Helm, set it under the shared `env` list,
not `web.extraEnv`, so the migration container receives it too. A restart without
running migrations does not update existing keys. Increasing or disabling the limit
does not restore previously shortened expirations; already-expired keys are never
reactivated by reconciliation.

## Database

The chart ships one PostgreSQL backend, the **CloudNativePG cluster**
(`cnpg.enabled`, default `true`). It requires the
[CNPG operator](https://cloudnative-pg.io/) to be installed in the cluster.

### Where the app connects

The app's PostgreSQL connection is fully configurable under `database:` in
`values.yaml` (`host`, `port`, `name`, `username`, `password` /
`existingSecret`). When these are left blank, the chart auto-derives the CNPG
connection: the `<release>-postgres-rw` service, database/user `aqueduct`, and
the password from the operator-created `<cluster>-app` Secret. Set
`database.host` (and credentials) explicitly to point the app at an external
PostgreSQL instance instead. If CNPG is disabled, also set `database.name`,
`database.username`, and either `database.password` or `database.existingSecret`.

### Migrating from an external PostgreSQL instance

CNPG can clone an existing PostgreSQL instance into a new cluster at creation
time via `bootstrap.pg_basebackup`, so no separate dump/restore job is needed.
Enable it with `cnpg.bootstrapFromExternal.enabled: true` and point
`cnpg.bootstrapFromExternal.connection` at the source (`host` is required):

```yaml
cnpg:
  enabled: true
  bootstrapFromExternal:
    enabled: true
    connection:
      host: "<postgres-host>"    # required
      # The pg_basebackup source user needs REPLICATION privilege — use the
      # `postgres` superuser and its password.
      username: postgres
      password: "<postgres-password>"   # or existingSecret + secretKeys.password
```

> **Prerequisites on the source instance:** `wal_level >= replica`
> (PostgreSQL default), a `pg_hba.conf` `replication` rule allowing connections
> from the CNPG pod, and a user with `REPLICATION` privilege (the `postgres`
> superuser works). `pg_basebackup` only runs on first creation — the CNPG
> `Cluster` resource must not already exist.

Runbook:

1. Stop writes to the source before cloning and keep them stopped until cutover;
   `pg_basebackup` does not continuously replicate subsequent changes. Deploy
   with `bootstrapFromExternal.enabled: true` and `database.host` pointing at
   the source.
2. Wait for the CNPG cluster to become ready:
   `kubectl get cluster -n <ns>` (`READY 1`, `STATUS: Cluster in healthy state`).
3. Cut over by pointing the app at CNPG (a `helm upgrade` is enough; no DB
   change needed because the clone has the same users/databases):
   ```yaml
   database:
     host: "<release>-postgres-rw"      # CNPG read-write service
   ```
4. Verify the app works, then disable the clone job:
   ```yaml
   cnpg:
     bootstrapFromExternal:
       enabled: false    # no longer needed after first creation
   ```

### Fresh CNPG deployment

```yaml
cnpg:
  enabled: true
  # bootstrapFromExternal stays disabled (initdb creates a fresh cluster)
```

The app auto-derives the CNPG connection (no `database.*` overrides needed).

## Message broker (Valkey)

Celery uses a Redis-compatible broker. The chart ships one broker, the
**Valkey release** (`valkey.enabled`, default `true`) from the
[Valkey Helm chart](https://github.com/valkey-io/valkey-helm), a drop-in Redis
fork.

### Where the app connects

The Celery broker URL is set by `celery.brokerUrl` in `values.yaml`. When left
blank, the chart auto-derives `redis://valkey:6379/0` (the Valkey release
service). Set `celery.brokerUrl` explicitly to point at an external broker.

The broker holds only transient Celery task messages (`valkey.dataStorage` is
disabled by default), so there is no persistent data.

## Upgrading

### Before upgrading from the Bitnami-backed chart

> **Back up your database before upgrading.** This chart removes the Bitnami
> PostgreSQL and Redis workloads. Upgrading before migrating can interrupt
> service and connect Aqueduct to a new, empty database. The old broker's
> pending tasks can also be lost.

1. Check whether the current release still uses Bitnami PostgreSQL or Redis.
   If it does, stay on the previous chart until migration is complete. Install
   the CNPG operator and follow the [previous chart's migration guide](https://github.com/TU-Wien-dataLAB/aqueduct/blob/f6e8598ba5dc2c048ae216ae5d9bbbe8192ad50b/docs/deployment.md#database)
   to bring up CNPG and Valkey alongside the old services. Stop writes before
   cloning, keep them stopped until cutover, and verify the copied data and
   application connection. Drain pending Celery tasks before switching brokers.
2. Verify the application uses CNPG and Valkey (or explicitly configured
   external replacements) and that the new services are healthy. Confirm the
   database backup can be restored.
3. Update your values for the new chart. Remove `postgresql`, `redis`, and
   `global.postgresql` settings; explicitly enable `cnpg` and `valkey` if using
   them. If using external services, configure `database` and `celery.brokerUrl`
   instead. Review retained settings when using `helm upgrade --reuse-values`:
   old `cnpg.enabled: false` or `valkey.enabled: false` values can prevent the
   replacement services from starting.
4. Upgrade the chart and verify the application and Celery worker. For upgrades
   from before chart 0.2.0, also follow the Deployment selector steps below.

### To chart 0.2.0

The web Deployment's selector now includes `component: web`. Before, it
selected only `app: <release>`, which the Celery worker pods carry too, so
tools that resolve `deployment/<release>` to a pod (`kubectl exec`,
`kubectl logs`, `kubectl port-forward`) could pick a worker pod.

A Deployment's selector is immutable, so `helm upgrade` fails with
`field is immutable` unless the old Deployment is removed first. Delete it
without its pods, then upgrade:

```bash
kubectl delete deployment <release> --cascade=orphan -n <ns>
helm upgrade <release> ... -n <ns>
```

The pods keep running. The new Deployment adopts the existing ReplicaSet, so
nothing restarts unless the upgrade changes the pod template anyway. With Argo
CD, delete the Deployment the same way, then sync.

## Notes

- User and admin management will be covered in the User Guide.
