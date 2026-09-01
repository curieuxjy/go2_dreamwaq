"""DreamWaQ DirectRLEnv implementation for IsaacLab.

A 1:1 port of the original IsaacGym DreamWaQ onto the ``DirectRLEnv`` API, kept alongside the
manager-based stack ``dreamwaq_manager`` as a cross-check of the same algorithm.

Termination matches the original IsaacGym DreamWaQ exactly — trunk contact force > 1 N
(``terminate_after_contacts_on = ["base"]``) — and is identical to ``dreamwaq_manager``.
See ``KNOWN_ISSUES.md`` — in particular the GPU collision-filtering fix that this stack needs.
"""

import os

# Convenience path for this extension
DREAMWAQ_DIRECT_EXT_DIR = os.path.dirname(os.path.abspath(__file__))
