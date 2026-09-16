"""Environment adapters.

An adapter binds MAVER to a benchmark or a production alerting environment: it
constructs alerts, routes submissions, and supplies a transport. Adapters are
discovered through the ``maver.adapters`` entry-point group; core modules must never
import one directly.
"""
