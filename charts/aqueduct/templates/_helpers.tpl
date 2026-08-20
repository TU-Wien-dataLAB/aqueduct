{{/*
Generate environment variables for the Django container.
Usage:
  {{- include "aqueduct.env" . | nindent 12 }}
*/}}
{{- define "aqueduct.env" }}
{{ toYaml .Values.env | nindent 0 }}
- name: DJANGO_DEBUG
  value: "False"
{{- if .Values.ingress.enabled }}
- name: ALLOWED_HOSTS
  value: {{ .Values.ingress.host | quote }}
- name: MCP_ALLOWED_HOSTS
  value: {{ .Values.ingress.host | quote }}
{{- end }}
- name: SECRET_KEY
  {{- if .Values.djangoSecretKey.value }}
  value: {{ .Values.djangoSecretKey.value }}
  {{- else }}
  valueFrom:
  {{ toYaml .Values.djangoSecretKey.valueFrom | nindent 4 }}
  {{- end }}
# Postgres configuration
# The connection target is resolved by the `aqueduct.db.*` helpers below:
# explicit `database.*` values win; otherwise the app stays on the Bitnami
# `postgresql` subchart when it is enabled (current default, and the safe choice
# while a CNPG cluster bootstraps side-by-side), and only auto-derives the CNPG
# connection when Bitnami is disabled. Set `database.host` explicitly to cut over
# to the CNPG cluster once its data is ready.
- name: POSTGRES_DB
  value: {{ include "aqueduct.db.name" . | quote }}
- name: POSTGRES_USER
  value: {{ include "aqueduct.db.username" . | quote }}
- name: POSTGRES_PASSWORD
  {{- include "aqueduct.db.password" . | nindent 2 }}
- name: POSTGRES_HOST
  value: {{ include "aqueduct.db.host" . | quote }}
- name: POSTGRES_PORT
  value: {{ include "aqueduct.db.port" . | quote }}
# OIDC configuration
- name: OIDC_RP_SIGN_ALGO
  value: {{ .Values.oidc.signAlgo | quote }}
- name: OIDC_OP_JWKS_ENDPOINT
  value: {{ .Values.oidc.jwksEndpoint | quote }}
- name: OIDC_OP_AUTHORIZATION_ENDPOINT
  value: {{ .Values.oidc.authorizationEndpoint | quote }}
- name: OIDC_OP_TOKEN_ENDPOINT
  value: {{ .Values.oidc.tokenEndpoint | quote }}
- name: OIDC_OP_USER_ENDPOINT
  value: {{ .Values.oidc.userEndpoint | quote }}
{{- if .Values.oidc.existingSecret }}
- name: OIDC_RP_CLIENT_ID
  valueFrom:
    secretKeyRef:
      name: {{ .Values.oidc.existingSecret }}
      key: {{ .Values.oidc.secretKeys.clientIdKey | default "clientId" }}
- name: OIDC_RP_CLIENT_SECRET
  valueFrom:
    secretKeyRef:
      name: {{ .Values.oidc.existingSecret }}
      key: {{ .Values.oidc.secretKeys.clientSecretKey | default "clientSecret" }}
 {{- else }}
- name: OIDC_RP_CLIENT_ID
  value: {{ .Values.oidc.clientId | quote }}
- name: OIDC_RP_CLIENT_SECRET
  value: {{ .Values.oidc.clientSecret | quote }}
{{- end }}
- name: CELERY_BROKER_URL
  value: {{ include "aqueduct.celery.brokerUrl" . | quote }}
- name: CELERY_WORKER_CONCURRENCY
  value: {{ .Values.celery.workerConcurrency | quote }}
- name: REQUEST_RETENTION_DAYS
  value: {{ .Values.celery.requestRetentionDays | quote }}
- name: REQUEST_RETENTION_SCHEDULE
  value: {{ .Values.celery.schedule | quote }}
- name: SILKY_ENABLED
{{- if .Values.silk.enabled }}
  value: "True"
{{- else }}
  value: "False"
{{- end }}
- name: AQUEDUCT_FILES_API_URL
  value: {{ .Values.filesApi.url | quote }}
{{- if .Values.filesApi.apiKey }}
- name: AQUEDUCT_FILES_API_KEY
  value: {{ .Values.filesApi.apiKey | quote }}
{{- else if .Values.filesApi.existingSecret }}
- name: AQUEDUCT_FILES_API_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Values.filesApi.existingSecret }}
      key: {{ .Values.filesApi.secretKeys.apiKey }}
{{- end }}
{{- if .Values.tika.enabled }}
- name: TIKA_SERVER_URL
  value: http://{{ .Release.Name }}-tika:9998
{{- end }}
- name: MCP_CONFIG_FILE_PATH
  value: "/etc/aqueduct/mcp.json"
- name: TOS_ENABLED
{{- if .Values.tos.enabled }}
  value: "True"
{{- else }}
  value: "False"
{{- end }}
- name: TOS_GATEWAY_VALIDATION
{{- if .Values.tos.gatewayValidation }}
  value: "True"
{{- else }}
  value: "False"
{{- end }}
- name: DJANGO_LOG_LEVEL
  value: {{ .Values.logLevel }}
- name: CORS_ALLOW_ALL_ORIGINS
{{- if .Values.cors.allowAllOrigins }}
  value: "True"
{{- else }}
  value: "False"
{{- end }}
- name: CORS_URLS_REGEX
  value: {{ .Values.cors.urlsRegex | quote }}
{{- end }}

{{/*
PostgreSQL connection resolution helpers.

These decouple where Aqueduct connects from which backend deploys the database,
so the Bitnami subchart and a CNPG cluster can coexist during migration.

Resolution order for each field:
  1. explicit `database.*` value (use this to pin the app to a specific backend)
  2. the Bitnami `postgresql` subchart, when `postgresql.enabled` (current
     default; also the safe choice while a CNPG cluster bootstraps side-by-side)
  3. the CNPG cluster, when `cnpg.enabled` and Bitnami is disabled
  4. the Bitnami `global.postgresql.*` values (legacy default)

Because Bitnami wins when both are enabled, enabling CNPG never moves the app
on its own: set `database.host` (and credentials) explicitly to cut over.
*/}}
{{- define "aqueduct.cnpg.clusterName" -}}
{{- .Values.cnpg.cluster.nameOverride | default (printf "%s-postgres" .Release.Name) -}}
{{- end -}}

{{/* CNPG read-write Service DNS (short name, same namespace as the app) */}}
{{- define "aqueduct.cnpg.host" -}}
{{- printf "%s-rw" (include "aqueduct.cnpg.clusterName" .) -}}
{{- end -}}

{{- define "aqueduct.db.host" -}}
{{- .Values.database.host | default (ternary (include "aqueduct.cnpg.host" .) .Values.global.postgresql.fullnameOverride (and .Values.cnpg.enabled (not .Values.postgresql.enabled))) -}}
{{- end -}}

{{- define "aqueduct.db.port" -}}
{{- .Values.database.port | default (ternary "5432" .Values.global.postgresql.service.ports.postgresql (and .Values.cnpg.enabled (not .Values.postgresql.enabled))) | default "5432" -}}
{{- end -}}

{{- define "aqueduct.db.name" -}}
{{- .Values.database.name | default (ternary (.Values.cnpg.cluster.database | default "aqueduct") (.Values.global.postgresql.auth.database | default "aqueduct") (and .Values.cnpg.enabled (not .Values.postgresql.enabled))) | default "aqueduct" -}}
{{- end -}}

{{- define "aqueduct.db.username" -}}
{{- .Values.database.username | default (ternary (.Values.cnpg.cluster.owner | default "aqueduct") .Values.global.postgresql.auth.username (and .Values.cnpg.enabled (not .Values.postgresql.enabled))) | default "aqueduct" -}}
{{- end -}}

{{/*
Renders either `value: ...` or a `valueFrom: secretKeyRef: ...` block.
Order: database.password -> database.existingSecret -> (CNPG <cluster>-app
secret when CNPG-only, else Bitnami global.postgresql.auth).
*/}}
{{/*
Name of the Secret holding the source (external cluster) password, referenced
by `externalClusters[].password`. Uses `bootstrapFromExternal.connection.existingSecret`
when provided, otherwise an auto-created Secret `<cluster>-source-password` filled
from the plaintext `bootstrapFromExternal.connection.password`.
*/}}
{{- define "aqueduct.cnpg.sourcePasswordSecret" -}}
{{- $mig := .Values.cnpg.bootstrapFromExternal -}}
{{- $mig.connection.existingSecret | default (printf "%s-source-password" (include "aqueduct.cnpg.clusterName" .)) -}}
{{- end -}}

{{- define "aqueduct.db.password" -}}
{{- $db := .Values.database -}}
{{- if $db.password }}
value: {{ $db.password | quote }}
{{- else if $db.existingSecret }}
valueFrom:
  secretKeyRef:
    name: {{ $db.existingSecret }}
    key: {{ $db.secretKeys.password | default "password" }}
{{- else if and .Values.cnpg.enabled (not .Values.postgresql.enabled) }}
valueFrom:
  secretKeyRef:
    name: {{ include "aqueduct.cnpg.clusterName" . }}-app
    key: password
{{- else if .Values.global.postgresql.auth.existingSecret }}
valueFrom:
  secretKeyRef:
    name: {{ .Values.global.postgresql.auth.existingSecret }}
    key: {{ .Values.global.postgresql.auth.secretKeys.userPasswordKey | default "password" }}
{{- else }}
value: {{ .Values.global.postgresql.auth.password | quote }}
{{- end }}
{{- end -}}

{{/*
Celery broker (Redis/Valkey) connection resolution.

Mirrors the database helpers: decouples where the app connects from which
backend deploys the broker, so the Bitnami `redis` subchart and a Valkey
release can coexist during migration.

Resolution order for the broker URL:
  1. explicit `celery.brokerUrl` (use this to pin the app to a specific backend)
  2. the Bitnami `redis` subchart service (`redis-master`), when `redis.enabled`
     (current default; also the safe choice while a Valkey release stands up)
  3. the Valkey release service (`valkey`), when `valkey.enabled` and Bitnami is
     disabled

Because Bitnami wins when both are enabled, enabling Valkey never moves the
app on its own: set `celery.brokerUrl` explicitly (or disable `redis`) to cut
over. Celery uses the broker only for transient task messages, so no data
needs to be migrated.
*/}}
{{- define "aqueduct.celery.brokerUrl" -}}
{{- .Values.celery.brokerUrl | default (printf "redis://%s:6379/0" (ternary "valkey" "redis-master" (and .Values.valkey.enabled (not .Values.redis.enabled)))) -}}
{{- end -}}
