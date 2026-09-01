"""DreamWaQ task environments."""

import importlib

# Import sub-packages to trigger gymnasium registration
importlib.import_module(".locomotion", __name__)
