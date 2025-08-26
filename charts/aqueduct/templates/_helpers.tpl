{{/*
Generate environment variables for the Django container.
Usage:
  {{- include "aqueduct.env" . | nindent 12 }}
*/}}
{{- define "aqueduct.env" }}
{{ toYaml .Values.env | nindent 0 }}
- name: DJANGO_DEBUG
  value: "False"
- name: SECRET_KEY
  {{- if .Values.djangoSecretKey.value }}
  value: {{ .Values.djangoSecretKey.value }}
  {{- else }}
  valueFrom:
  {{ toYaml .Values.djangoSecretKey.valueFrom | nindent 4 }}
  {{- end }}
# Postgres configuration
- name: POSTGRES_DB
  value: {{ .Values.global.postgresql.auth.database }}
- name: POSTGRES_USER
  value: {{ .Values.global.postgresql.auth.username }}
{{- if .Values.global.postgresql.auth.existingSecret }}
- name: POSTGRES_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ .Values.global.postgresql.auth.existingSecret }}
      key: {{ .Values.global.postgresql.auth.secretKeys.userPasswordKey | default "password" }}
{{- else }}
- name: POSTGRES_PASSWORD
  value: {{ .Values.global.postgresql.auth.password | quote }}
{{- end }}
- name: POSTGRES_HOST
  value: {{ .Values.global.postgresql.fullnameOverride }}
- name: POSTGRES_PORT
  value: "{{ .Values.global.postgresql.service.ports.postgresql }}"
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
{{- if .Values.oidc.existingSecret }}
- name: CELERY_BROKER_URL
  value: {{ .Values.celery.brokerUrl | quote }}
- name: CELERY_WORKER_CONCURRENCY
  value: {{ .Values.celery.workerConcurrency | quote }}
- name: REQUEST_RETENTION_DAYS
  value: {{ .Values.celery.requestRetentionDays | quote }}
- name: REQUEST_RETENTION_SCHEDULE
  value: {{ .Values.celery.schedule | quote }}
{{- end }}
- name: SILKY_ENABLED
{{- if .Values.silk.enabled }}
  value: "True"
{{- else }}
  value: "False"
{{- end }}
{{- if .Values.persistence.files.enabled }}
- name: AQUEDUCT_FILES_API_ROOT
  value: {{ .Values.persistence.files.mountPath | quote }}
{{- end }}
{{- if .Values.tika.enabled }}
- name: TIKA_SERVER_URL
  value: http://{{ .Release.Name }}-tika:9998
{{- end }}
{{- end }}
