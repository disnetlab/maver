"""Configuration: nested settings groups, model profiles, and arm files.

Environment variables win over YAML so that a ConfigMap remains the deployment
control surface. Secrets are typed so they never reach a log or a config dump.
"""
