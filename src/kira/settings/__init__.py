"""Configuration: nested settings groups, model profiles, and arm files.

Environment variables win over file-based sources so that a ConfigMap remains the
deployment control surface. Secrets are typed so they never reach a log or a config
dump. Every group's fields also answer to the flat names the research prototype used,
so existing evaluation run scripts keep working unchanged.
"""

from kira.settings.base import ENV_PREFIX, SettingsGroup, group_config, setting
from kira.settings.groups import (
    AuthoritySettings,
    IntakeSettings,
    KafkaSettings,
    MemorySettings,
    ModelSettings,
    ObservabilitySettings,
    PipelineSettings,
    RuntimeSettings,
    SamplingOverrides,
    SecondPassSettings,
    ServerSettings,
    ToolSettings,
    VerifySettings,
)
from kira.settings.root import (
    Settings,
    get_settings,
    legacy_names_in_use,
    load_settings,
    reset_settings,
)

__all__ = [
    "ENV_PREFIX",
    "AuthoritySettings",
    "IntakeSettings",
    "KafkaSettings",
    "MemorySettings",
    "ModelSettings",
    "ObservabilitySettings",
    "PipelineSettings",
    "RuntimeSettings",
    "SamplingOverrides",
    "SecondPassSettings",
    "ServerSettings",
    "Settings",
    "SettingsGroup",
    "ToolSettings",
    "VerifySettings",
    "get_settings",
    "group_config",
    "legacy_names_in_use",
    "load_settings",
    "reset_settings",
    "setting",
]
