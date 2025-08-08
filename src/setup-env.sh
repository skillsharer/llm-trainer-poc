#!/bin/bash

EXAMPLE_FILE="src/.env.example"
ENV_FILE="src/.env"

if [ -f "$ENV_FILE" ]; then
  echo "$ENV_FILE already exists. Not overwriting."
else
  cp "$EXAMPLE_FILE" "$ENV_FILE"
  echo "Created $ENV_FILE from $EXAMPLE_FILE."
fi