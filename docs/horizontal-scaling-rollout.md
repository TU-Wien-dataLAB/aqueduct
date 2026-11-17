---
title: Roll out horizontal scaling
parent: Deployment
nav_order: 2
---

# Roll out horizontal scaling

Use this guide to validate two web replicas in staging, then promote only the measured configuration.
Consult `charts/aqueduct/values.yaml` for configuration defaults.
**No staging checks or load measurements below have been run; no live cluster changes were made
while writing this guide. Keep initial production web replicas at 1 until validation passes.**

Treat automated Helm, configuration and gateway endpoint tests as regression checks only.
They do not verify shared Redis state or cross-replica behavior; complete the staging checks below
before promotion.

## 1. Prepare the release

- Confirm PR #194 is integrated. Configure [public-model RPM limits](deployment.md#public-model-rpm-limits)
  identically across all gateway processes.
- Use an isolated staging namespace, disposable test data, controlled upstreams and production-like
  PostgreSQL/shared Valkey. Disable sticky sessions at both Service and ingress/load-balancer layers.
  Keep `DJANGO_DEBUG=False`; verify effective Django sessions use the database and cache uses shared
  Redis/Valkey, not a process-local backend. Keep database, cache, secrets and OIDC configuration
  identical across replicas; do not combine scaling with a database/broker cutover.
- Take a recoverable backup and agree on abort thresholds for errors, latency, CPU/memory and
  database connections. Check schema compatibility with both old and new images (expand-contract).
- Move any `POSTGRESQL_POOL_MIN_SIZE` and `POSTGRESQL_POOL_MAX_SIZE` entries from `env`,
  `web.extraEnv`, `celery.workerExtraEnv`, `celery.beatExtraEnv`, or `migrate.extraEnv` into
  the respective `web.databasePool`, `celery.workerDeployment.databasePool`,
  and `celery.beatDatabasePool` `minSize`/`maxSize` values. Apply formerly global values to
  each role that needs them. The chart rejects the old entries; other pool environment settings
  remain available. Remove all `migrate.extraEnv` entries and put required settings in `env`
  or the corresponding role's `extraEnv`; migration inherits that role's environment. Remove
  the obsolete `migrate.databasePool` setting: migration connections are unpooled.
- Review effective pool overrides, node capacity for requests plus surge, and the calculated connection budget
  against actual PostgreSQL limits. Include terminating pods and non-Aqueduct clients.

Run chart commands from the **repository root**. Set explicit staging context and operator-owned
values paths; do not copy these placeholders into production:

```bash
export KUBE_CONTEXT='<staging-context>' NS='<staging-namespace>' RELEASE='<release>'
export VALUES='/absolute/path/to/staging-values.yaml' IMAGE_TAG='<desired-image-tag>'
export WEB="$RELEASE" WORKER="$RELEASE-celery-worker" BEAT="$RELEASE-celery-beat"
helm template "$RELEASE" charts/aqueduct -n "$NS" -f "$VALUES" \
  --set image.tag="$IMAGE_TAG" --set web.replicas=1 > /tmp/aqueduct-one.yaml
helm template "$RELEASE" charts/aqueduct -n "$NS" -f "$VALUES" \
  --set image.tag="$IMAGE_TAG" --set web.replicas=2 > /tmp/aqueduct-two.yaml
```

Inspect both renders: singleton `Recreate` Beat, web/worker fixed surge/unavailability, all role pools,
resource requests and scheduling. Each pod must have a `migrate` init container using the same
image as its application, with settings and router configuration mounted. No migration Job,
Kubernetes API gate or migration RBAC is required. Pin an immutable image identity. Migration
inits use pod scheduling, not the removed `migrate.scheduling` setting. Budget one unpooled
migration connection per starting pod, including rollout surge. Confirm actual Deployment names
and selectors before using the commands below. Treat rendered files and values as sensitive.

### Transition from the Job-based chart

Schedule maintenance for the first transition: stop traffic and producers, drain tasks, stop all
old web, worker and Beat pods, and wait for any old migration Job to finish or be explicitly
investigated and stopped before upgrading. Old Jobs do not take the new advisory lock, so they
must never overlap new migration inits. Before upgrading, deliberately retire all old pods,
ReplicaSets and rollback targets that depend on the old gate; confirm none can start replacement
pods. The upgrade removes the old gate ServiceAccount and ConfigMap. Do not rely on transparent
rollback to those templates. Confirm the target schema is compatible with the chosen image; for
incompatible changes keep maintenance in place through migration and verification. Have a tested
backup and restore plan with a matching image for recovery.

Old revision Jobs, Roles and RoleBindings were retained by Helm. Review them with
`kubectl --context "$KUBE_CONTEXT" -n "$NS" get jobs,roles,rolebindings -l "app=$RELEASE,component=migrate"`.
Only after deliberately retiring all old workloads and rollback targets that need them, delete
reviewed, named obsolete resources. They may survive uninstall. If recovery requires an old
release, use the approved backup/restore procedure and matching image under maintenance; recreate
and validate its required gate resources as part of that procedure. Never allow old migration Jobs
to overlap the new inits.

## 2. Remove legacy Beat sidecars before the first upgrade

**For the first transition from web-pod Beat sidecars, schedule maintenance.** Stop incoming traffic,
scale the old web Deployment to zero, and wait for every old web pod (including terminating pods)
to disappear **before running Helm upgrade**. Otherwise the new singleton can overlap legacy Beat.

```bash
kubectl --context "$KUBE_CONTEXT" -n "$NS" scale deployment/"$WEB" --replicas=0
kubectl --context "$KUBE_CONTEXT" -n "$NS" wait --for=delete pod \
  -l "app=$RELEASE,component=web" --timeout=300s
```

Inspect pods/owners to confirm the selector covered all legacy web pods; do not proceed on timeout.
For incompatible migrations, also stop producers, drain queued/active tasks and stop old workers;
wait until their pods are gone before migration. If compatibility is uncertain, use maintenance
rather than relying on the startup gates to protect old running processes.

## 3. Deploy one replica and check migration ordering

For a fresh install, skip the legacy shutdown. For an existing release, preserve all required
operator configuration in `$VALUES`; do not rely on defaults or reuse stale pool/affinity overrides.

```bash
helm upgrade --install "$RELEASE" charts/aqueduct --kube-context "$KUBE_CONTEXT" \
  -n "$NS" -f "$VALUES" --set image.tag="$IMAGE_TAG" --set web.replicas=1 \
  --wait --timeout 30m
helm history "$RELEASE" --kube-context "$KUBE_CONTEXT" -n "$NS"
```

Allow for a PostgreSQL lock wait of up to 10 minutes, migration and startup when choosing the
Helm timeout. Connection attempts use a 5-second timeout; network stalls have no hard deadline.
Kubernetes retries failed init containers. Avoid `--atomic`: application rollback is not schema recovery.
Inspect each pod's init logs and status:

```bash
kubectl --context "$KUBE_CONTEXT" -n "$NS" get pods -l "app=$RELEASE" -o wide
kubectl --context "$KUBE_CONTEXT" -n "$NS" logs <pod> -c migrate
kubectl --context "$KUBE_CONTEXT" -n "$NS" describe pod <pod>
```

Verify each new app container starts only after its migration init succeeds. Test fresh database
bootstrap and upgrades. Kubernetes retries failing init containers; investigate failures and partial
nontransactional changes before allowing retries to proceed. The advisory lock requires a direct
or session-pooled PostgreSQL connection, not transaction pooling.

## 4. Establish a baseline, then test two replicas

Measure the one-replica release before changing capacity. Reuse
[`aqueduct/locustfiles/locustfile.py`](https://github.com/TU-Wien-dataLAB/aqueduct/blob/main/aqueduct/locustfiles/locustfile.py), but prepare its fixture,
`main`/`embedding`/`image`/`tts`/`transcribe` model routes, Files API and controlled upstreams in
**isolated staging only**. The fixture includes a known bootstrap token; `LOAD_TESTING=True` exposes
test-user creation/cleanup endpoints. Never enable these or load the fixture in production.
The local Compose stack is useful for rehearsal, not proof of Kubernetes multi-replica behavior.

If provisioning an isolated database with the existing fixture, run the management command from
`aqueduct/`, with that staging database's connection environment already selected:

```bash
# From repository root; NEVER target a production database.
(cd aqueduct && uv run python manage.py loaddata locustfiles/locust_fixture.json)
```

Run the load generator from the **repository root**, targeting the normal staging load balancer,
not localhost or a pod port-forward:

```bash
export STAGING_URL='https://<staging-host>/'
uv run --group dev-locust locust -f aqueduct/locustfiles/locustfile.py \
  --host "$STAGING_URL" --headless --users 10 --spawn-rate 1 --run-time 10m \
  --csv /tmp/aqueduct-one --html /tmp/aqueduct-one.html
```

Treat 10 users/10 minutes as a starting workload, not an acceptance threshold. Record warm-up,
request mix, upstream delay/error configuration, test duration and generator capacity. Use the
same workload and upstream behavior for both runs; add representative concurrency levels.
The existing Locust mix is not a substitute for the login, shared-token limiter or stream checks below.

Upgrade with the command in step 3 changed to `--set web.replicas=2`, then rerun Locust with
`--csv /tmp/aqueduct-two --html /tmp/aqueduct-two.html`. Record the image and configuration.
Replacement and scaled pods each run the serialized migration/no-op check before starting.

Collect per-pod request evidence and resource/connection samples before, during and after load
and rollouts (repeat sampling; one snapshot can miss peaks):

```bash
kubectl --context "$KUBE_CONTEXT" -n "$NS" logs \
  -l "app=$RELEASE,component=web" -c django --prefix --since=10m --max-log-requests=10
kubectl --context "$KUBE_CONTEXT" -n "$NS" top pods --containers
kubectl --context "$KUBE_CONTEXT" -n "$NS" get pods -l "app=$RELEASE" -o wide
# Use a read-only monitoring connection; do not put passwords in shell history.
psql "$MONITORING_DATABASE_URL" -c 'SHOW max_connections;'
psql "$MONITORING_DATABASE_URL" -c \
  "SELECT name, setting FROM pg_settings WHERE name IN ('superuser_reserved_connections', 'reserved_connections');"
psql "$MONITORING_DATABASE_URL" -c \
  'SELECT datname, usename, application_name, state, count(*) FROM pg_stat_activity GROUP BY 1,2,3,4 ORDER BY 1,2,3,4;'
```

Correlate identifiable test requests/timestamps with pod-prefixed application logs (or existing
LB access logs identifying backend pods). If current logging lacks that visibility, enable suitable
staging access logging before testing. Pod readiness or aggregate throughput alone does not prove
both pods received traffic. Capture terminating pod counts, queue depth and task logs alongside samples.

## 5. Complete the staging safety gates

Run every check through the normal non-sticky load balancer unless testing Kubernetes lifecycle
ordering. Store timestamps, release/image/config identity, logs and client results in the evidence
column. **Every row is currently pending / not run.**

| Check | Required evidence / acceptance | Status / evidence |
|---|---|---|
| LB distribution | Correlated successful requests reaching each of two web pods; no affinity cookies or client-IP stickiness | Pending / not run |
| Login | Sign in once; reuse the session cookie across requests observed on both pods without logout | Pending / not run |
| Sequential Aqueduct limiter | One token, known allowance/window, requests on both pods; no separate allowance per pod | Pending / not run |
| Concurrent Aqueduct limiter | Same token across concurrent requests on both pods; record counts/429s and accepted contention-related best-effort overshoot, not strict enforcement | Pending / not run |
| Public-model limits | Shared canonical-model RPM across both pods, aliases and different tokens; concurrent admission, UTC minute rollover, 429/Retry-After and cache-error rejection; separately verify provider-wide quotas upstream | Pending / not run in staging |
| MCP JSON and POST SSE | Post-#194 self-contained exchanges on both pods; correct JSON and streamed events through LB; use [MCP reference](api/mcp.md) | Pending / not run |
| Chat streams | `stream: true` chat completions through LB on both pods; valid events and termination | Pending / not run |
| Worker dispatch | Enqueue representative background work, observe worker execution/results independently of web count | Pending / not run |
| Beat rollout | Observe old Beat stop before new Beat starts; periodic dispatch continues after pause, no scheduler-overlap duplicates during web scaling or Beat replacement | Pending / not run |
| Rolling web update/replacement | Roll a compatible image under load and delete one web pod; remaining/replacement pods serve, tasks continue, connections fit budget; log interrupted requests/streams (allowed) | Pending / not run |
| Migration install/upgrade ordering | Fresh DB bootstrap and forward upgrade complete; each web/worker/Beat init succeeds before its app starts; old compatible pods serve during later upgrades | Pending / not run |
| Migration failure | Induce a controlled failing migration on a disposable staging DB; affected app containers stay blocked; inspect retries and partial changes, retain logs | Pending / not run |
| Contention/scaling | Start multiple pods simultaneously; verify serialized migrations and later no-op checks; replace/scale pods and verify init succeeds | Pending / not run |
| Budget and resources | Repeated `psql`/`kubectl top` samples including surge/terminating pods; no exhaustion, observed CPU/memory/queue behavior supports chosen requests/limits | Pending / not run |

Do not repeat rollouts arbitrarily to create load: terminating generations can exceed a one-extra-pod
reserve. Wait for termination and database connections to settle before another rollout. Interruption
of in-flight requests/streams is allowed; require continued service, not zero errors or automatic replay.

Record comparable results; use `not measured`, never estimates presented as observations:

| Run | Workload / config / duration | Latency p50/p95/p99 | Throughput / error rate | Per-role CPU / memory peaks | PostgreSQL peak / budget; terminating pods | Evidence |
|---|---|---|---|---|---|---|
| 1 web replica | Pending | Not measured | Not measured | Not measured | Not measured | Not run |
| 2 web replicas | Pending; same workload | Not measured | Not measured | Not measured | Not measured | Not run |

## 6. Promote or recover

Promote only after all gates pass and a reviewed connection/resource budget supports the measured
configuration. Record the accepted workload envelope and thresholds; do not assume linear scaling
or claim the historical slowdown is fixed. Keep production at **1** while evidence is pending, then
explicitly select **2** only if validated. Apply the same non-sticky LB/shared-state checks after promotion.
Remove staging test users/data and disable `LOAD_TESTING` after the exercise; verify test-auth endpoints
are unavailable. Review CPU/memory limits from measurements rather than adding arbitrary defaults.

If errors, resource pressure or connections breach agreed thresholds, stop the load and further
rollouts, capture init/pod logs and database state, then return to one web replica via reviewed Helm
values. Do not force old serving pods away to bypass a failed migration.

For application rollback, first establish that the current schema is compatible with the target image.
Prefer a new `helm upgrade` using this chart and reviewed values with the older image. Every new pod
runs its migration init; neither this nor `helm rollback` undoes schema changes. Do not assume
transparent rollback to the retired Job-based chart; use the maintenance and backup/restore plan
above. If incompatible, stop traffic, stop/drain workers
and Beat, and perform approved database recovery/backup restore with a matching image before
reopening service. Prefer a reviewed forward fix when safe.

If returning to a legacy chart with Beat sidecars, use maintenance: stop the standalone Beat and
wait until it is fully gone before starting legacy web pods. Never allow both scheduler layouts to
run together. Record recovery results and re-run the failed gates before another promotion attempt.
