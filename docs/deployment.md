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

## Database

The chart ships with two optional PostgreSQL backends, controlled independently:

- **Bitnami `postgresql` subchart** (`postgresql.enabled`, default `true`) — the
  legacy default. The app connects to it automatically.
- **CloudNativePG cluster** (`cnpg.enabled`, default `false`) — requires the
  [CNPG operator](https://cloudnative-pg.io/) to be installed in the cluster.

Only one backend needs to run in steady state, but both can be enabled at the
same time to migrate data with zero downtime.

### Where the app connects

The app's PostgreSQL connection is fully configurable under `database:` in
`values.yaml` (`host`, `port`, `name`, `username`, `password` /
`existingSecret`). When these are left blank, the chart resolves them as
follows:

1. If `postgresql.enabled` is `true` → the Bitnami subchart connection (the
   current default, and the safe choice while a CNPG cluster bootstraps).
2. Else if `cnpg.enabled` is `true` → the CNPG cluster (`<release>-postgres-rw`
   service, password from the operator-created `<cluster>-app` Secret).
3. Else → the Bitnami `global.postgresql.*` values.

Because Bitnami wins when both are enabled, **enabling CNPG never moves the app
on its own**. You cut over by setting `database.host` (and credentials)
explicitly once the CNPG cluster is ready.

### Migrating from Bitnami to CNPG

CNPG can clone an existing PostgreSQL instance into a new cluster at creation
time via `bootstrap.pg_basebackup`, so no separate dump/restore job is needed.
Enable it with `cnpg.bootstrapFromExternal.enabled: true`; the source defaults
to the Bitnami subchart connection and is overridable under
`cnpg.bootstrapFromExternal.connection`.

```yaml
postgresql:
  enabled: true              # keep the source running during migration
cnpg:
  enabled: true
  bootstrapFromExternal:
    enabled: true
    connection:
      # The pg_basebackup source user needs REPLICATION privilege. For the
      # Bitnami subchart, use the `postgres` superuser and its password.
      username: postgres
      password: "<postgres-password>"   # or existingSecret + secretKeys.password
```

> **Prerequisites on the source (Bitnami) instance:** `wal_level >= replica`
> (PostgreSQL default), a `pg_hba.conf` `replication` rule allowing connections
> from the CNPG pod, and a user with `REPLICATION` privilege (the `postgres`
> superuser works). `pg_basebackup` only runs on first creation — the CNPG
> `Cluster` resource must not already exist.

Runbook:

1. Deploy with both backends enabled and `bootstrapFromExternal.enabled: true`.
   The app keeps running against Bitnami; CNPG clones the data automatically.
2. Wait for the CNPG cluster to become ready:
   `kubectl get cluster -n <ns>` (`READY 1`, `STATUS: Cluster in healthy state`).
3. Cut over by pointing the app at CNPG (a `helm upgrade` is enough; no DB
   change needed because the clone has the same users/databases):
   ```yaml
   database:
     host: "<release>-postgres-rw"      # CNPG read-write service
     # credentials: same as Bitnami (cloned user), or the <cluster>-app Secret
   ```
4. Verify the app works, then decommission Bitnami:
   ```yaml
   postgresql:
     enabled: false
   cnpg:
     bootstrapFromExternal:
       enabled: false    # no longer needed after first creation
   ```

### Fresh CNPG deployment (no Bitnami)

```yaml
postgresql:
  enabled: false
cnpg:
  enabled: true
  # bootstrapFromExternal stays disabled (initdb creates a fresh cluster)
```

The app auto-derives the CNPG connection (no `database.*` overrides needed).

## Message broker (Redis / Valkey)

Celery uses a Redis-compatible broker. The chart ships with two optional
backends, controlled independently:

- **Bitnami `redis` subchart** (`redis.enabled`, default `true`) — the legacy
  default. The app connects to it automatically.
- **Valkey release** (`valkey.enabled`, default `false`) — the
  [Valkey](https://github.com/valkey-io/valkey-helm) chart, a drop-in Redis fork.

Only one backend needs to run in steady state, but both can be enabled at the
same time to migrate with no app downtime.

### Where the app connects

The Celery broker URL is set by `celery.brokerUrl` in `values.yaml`. When left
blank, the chart resolves it as follows:

1. If `redis.enabled` is `true` → `redis://redis-master:6379/0` (the Bitnami
   subchart service).
2. Else if `valkey.enabled` is `true` → `redis://valkey:6379/0` (the Valkey
   release service).

Because Bitnami wins when both are enabled, **enabling Valkey never moves the
app on its own**. You cut over by setting `redis.enabled: false` (or by setting
`celery.brokerUrl` explicitly).

The broker holds only transient Celery task messages (the Bitnami subchart runs
with `persistence.enabled: false`), so there is no data to migrate — drain
pending tasks before cutting over.

### Migrating from Bitnami Redis to Valkey

1. Deploy with both backends enabled. The app keeps running against Bitnami;
   Valkey starts up empty alongside it.
   ```yaml
   redis:
     enabled: true              # keep the source running during migration
   valkey:
     enabled: true
   ```
2. Wait for the Valkey pod to become ready:
   `kubectl get pod -l app.kubernetes.io/name=valkey`.
3. Cut over by disabling Bitnami (a `helm upgrade` is enough — the app then
   auto-derives `redis://valkey:6379/0`):
   ```yaml
   redis:
     enabled: false
   valkey:
     enabled: true
   ```
4. Verify the app works (enqueue a request, watch the Celery worker pick it up),
   then remove the `redis:` block entirely if desired.

### Fresh Valkey deployment (no Bitnami)

```yaml
redis:
  enabled: false
valkey:
  enabled: true
```

The app auto-derives the Valkey connection (no `celery.brokerUrl` override
needed).

## Notes

- User and admin management will be covered in the User Guide.
