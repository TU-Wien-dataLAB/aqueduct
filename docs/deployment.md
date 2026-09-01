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
PostgreSQL instance instead.

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

1. Deploy with `bootstrapFromExternal.enabled: true`. The app keeps running
   against the source (point `database.host` at it) while CNPG clones the data.
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

## Notes

- User and admin management will be covered in the User Guide.
