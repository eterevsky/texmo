#!/bin/sh
# Shortcut for the rig head script with its dependency group.
exec uv run --group rig rig.py "$@"
