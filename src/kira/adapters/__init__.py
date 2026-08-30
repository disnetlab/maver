"""Environment adapters.

An adapter binds Kira to a benchmark or a production alerting environment: it
constructs alerts, routes submissions, and supplies a transport. Adapters are
discovered through the ``kira.adapters`` entry-point group; core modules must never
import one directly.
"""
