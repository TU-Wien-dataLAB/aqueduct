# Use a specific Node.js LTS version on Alpine for a small image size
FROM node:20-alpine

# Install necessary tools: Tini and directus-template-cli
RUN apk add --no-cache tini && npm install -g directus-template-cli

# Set the working directory inside the container
WORKDIR /app

# Copy your local template directory into the container image
COPY ./template ./template

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Declare environment variables for Directus instance URL and admin credentials
ENV URL=""
ENV ADMIN_EMAIL=""
ENV ADMIN_PASSWORD=""

# Use Tini as the entrypoint
ENTRYPOINT ["/sbin/tini", "--", "/entrypoint.sh"]
