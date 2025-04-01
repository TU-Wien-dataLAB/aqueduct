#!/bin/sh
set -e

exec directus-template-cli apply -p \
    --userEmail "$ADMIN_EMAIL" \
    --userPassword "$ADMIN_PASSWORD" \
    --directusUrl "$URL" \
    --templateLocation="./template" \
    --templateType="local" \
    --partial \
    --no-content \
    --no-users
