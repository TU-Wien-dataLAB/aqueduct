---
title: Deployment
nav_order: 4
has_children: true
---

# Deployment

> **TODO:**
> This guide will cover deploying Aqueduct in production using Helm on Kubernetes.
>
> **Planned topics:**
> - Writing and using a Helm chart for Aqueduct
> - Required configuration (secrets, settings.py for orgs etc.)
> - Authentication (OIDC/Dex) configuration
> - Reverse proxy and HTTPS setup

## Horizontal scaling

- [Roll out horizontal scaling](horizontal-scaling-rollout.md): legacy Beat transition, staging
  validation, load measurements, promotion and recovery.

Production web replicas remain **1** until two-replica staging validation passes. Staging evidence
and measurements are currently **pending / not run**; this is not an end-to-end HA or slowdown-fix claim.

## Access key expiration

Set `MAX_ACCESS_KEY_EXPIRATION_DAYS` to a positive integer to limit the expiration
of user and service-account keys. The default, `-1`, imposes no maximum. The key
forms show the maximum expiration and reject an explicit date later than the
current time plus the configured number of days. Leaving the expiration empty uses
the maximum. Programmatic saves with no expiration or a later expiration are still
capped at the maximum; earlier expirations are preserved.

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

## Concurrency lease renewal

Token and model concurrency slots are Redis leases with a fixed lifetime,
configured by `AQUEDUCT_TOKEN_CONCURRENT_REQUESTS_LIMIT_TTL_SECONDS` (default
`600` seconds). The lease is reserved while the request view runs and is not
renewed on a timer.

For streaming responses, each non-empty chunk renews the lease only if the same
lease still exists and has not expired. Normal completion, errors, cancellation,
and closing the response release the reservation. Empty chunks do not renew it.

If a request produces no stream progress, its lease expires after the TTL and
frees accounting capacity. This applies to stalled views, stalled streams,
never-consumed streams, and crashed workers. Expiry does not cancel or close the
local view or upstream request: Aqueduct does not enforce a maximum request
lifetime, and silent or backpressured upstream work may continue after its
capacity has been released. If that work later produces a chunk, the expired
lease cannot be renewed; the gateway stops forwarding that stream.

A Redis refresh error also stops forwarding. Release is best effort, as with
other cleanup paths. A reservation released after its lease expired cannot
remove a newer reservation.

## Metrics

Every web pod serves Prometheus metrics at `/metrics`. The endpoint is off
unless `AQUEDUCT_METRICS_TOKEN` is set, and then answers only requests that send
`Authorization: Bearer <token>`:

| Request | Response |
|---|---|
| `AQUEDUCT_METRICS_TOKEN` unset or empty | `404`, as for any path without a route |
| Token missing or wrong | `401` |
| Correct token | `200`, Prometheus text format |

### Enabling with Helm

Add the token to an existing Secret, then set:

```yaml
metrics:
  enabled: true
  existingSecret: aqueduct-secrets   # required when enabled
  existingSecretKey: METRICS_TOKEN   # default
  podMonitor:
    enabled: true                    # needs the Prometheus Operator CRDs
    interval: 30s                    # default
```

The web pods read `AQUEDUCT_METRICS_TOKEN` from the Secret. The PodMonitor
selects only the web pods, scrapes their `http` port at `/metrics`, and sends the
token from the same Secret. It picks up every replica, including the extra pod of
a rolling update. Rendering fails if `metrics.enabled` is set without
`metrics.existingSecret`.

### Reported metrics

All values are read when Prometheus scrapes; nothing is counted in the process.
*Per pod* metrics describe the pod that answered. *Gateway-wide* metrics read
shared state, so every pod reports the same value: aggregate them with
`max by (model)` (or `max by (model, status)`), not `sum`.

| Metric | Type | Labels | Scope | Meaning |
|---|---|---|---|---|
| `aqueduct_build_info` | gauge | `version` | per pod | Always 1; the running Aqueduct version |
| `aqueduct_db_up` | gauge | | per pod | 1 if the database answers a query |
| `aqueduct_redis_up` | gauge | | per pod | 1 if the Redis used for concurrency limiting answers `PING` |
| `aqueduct_db_pool_size` | gauge | | per pod | Connections open in the pool |
| `aqueduct_db_pool_available` | gauge | | per pod | Idle connections in the pool; in use = size − available |
| `aqueduct_db_pool_max` | gauge | | per pod | Maximum pool size |
| `aqueduct_db_pool_requests_waiting` | gauge | | per pod | Requests waiting for a pool connection |
| `aqueduct_db_pool_requests_total` | counter | | per pod | Connections requested from the pool |
| `aqueduct_db_pool_requests_errors_total` | counter | | per pod | Pool requests that failed, for example by timing out |
| `aqueduct_db_pool_connections_lost_total` | counter | | per pod | Pool connections found broken |
| `aqueduct_db_pool_wait_seconds_total` | counter | | per pod | Time spent waiting for pool connections |
| `aqueduct_inflight_requests` | gauge | `model` | gateway-wide | Requests holding a concurrency slot of the model |
| `aqueduct_model_concurrency_limit` | gauge | `model` | gateway-wide | Concurrent requests allowed for the model |
| `aqueduct_requests_last_minute` | gauge | `model`, `status` | gateway-wide | Requests in the last 60 seconds by HTTP status code, `pending` while unfinished |
| `aqueduct_scrape_duration_seconds` | gauge | | per pod | Time spent collecting the Aqueduct metrics |

The standard `process_*` and `python_*` metrics of the Prometheus client are
included too.

`model` is the model's name in the router config, so requests for an alias count
under the model it points to. Only models in the router config are reported.
There are no labels for users, tokens, organizations or paths.
`aqueduct_requests_last_minute` counts recorded requests, so it leaves out
requests rejected before a request record exists. The pool counters count since
the pod's pool started, and the pool metrics are left out when no connection
pool is configured (SQLite).

A scrape always answers `200`. If the database does not answer,
`aqueduct_db_up` is 0 and `aqueduct_requests_last_minute` is left out; if Redis
does not answer, `aqueduct_redis_up` is 0 and `aqueduct_inflight_requests` is
left out. A scrape takes at most one pool connection, waits for it at most
`HEALTH_DB_TIMEOUT` seconds and gives it back before answering. PostgreSQL cancels
its queries after about 1 second, and its Redis calls time out after about 1 second.

## Upgrading

### Migration init containers

Every web, worker and Beat pod validates the router configuration and runs Django migrations
before starting. Pods explicitly use the namespace's `default` ServiceAccount so upgrades clear
both the old `serviceAccountName` and Kubernetes' deprecated `serviceAccount` alias; they no longer
need migration-specific Kubernetes API permissions. Verify both live fields before removing the
old gate ServiceAccount. PostgreSQL migrations use a session advisory lock shared by all Aqueduct pods
on the same database; each later pod runs Django's normal no-op migration check. PostgreSQL
limits lock acquisition to 10 minutes, and connection attempts use a 5-second libpq timeout.
The lock timeout is reset before migration; a network stall is not subject to a hard deadline.
A failed init container is retried by Kubernetes and the application container does not start.
Inspect `kubectl logs <pod> -c migrate` and resolve the failure before proceeding. A failed
nontransactional migration may have left partial changes; assess them before retrying.

Use a direct PostgreSQL connection or a session-pooling proxy, not transaction pooling: the lock
must remain on the migration session; migrations must not close or reopen that connection.
Budget for one migration connection per starting pod in
addition to running application pools and surge/terminating pods. Init containers use the pod's
role scheduling and `migrate.resources` and `migrate.containerSecurityContext`. Each init receives
exactly its application role's environment, including role-specific `extraEnv`; configure database
and app settings there or in global `env`. `migrate.extraEnv` must be empty to prevent configuration
mismatch. The migration connection is deliberately unpooled; `migrate.databasePool`,
`migrate.scheduling` and `migrate.activeDeadlineSeconds` have been removed; rendering fails if
these obsolete keys are still configured, so remove them from existing values files. Pin the same
immutable image identity for init and application containers.

The Helm chart requires PostgreSQL. The init fails if effective Django settings resolve to
SQLite or another backend, including settings supplied by secrets or `settingsPy.extra`.
SQLite remains supported for local Django development outside this chart.

Migrations can run while old pods serve. Use expand/contract schema changes compatible with both
images, or schedule maintenance and stop traffic, producers, workers and Beat for incompatible or
uncertain changes. A lock serializes migrations, not application access. See the
[rollout guide](horizontal-scaling-rollout.md) for the transition from Job-based releases, cleanup,
and rollback precautions.

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
