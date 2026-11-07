"""
Streamlit Cloud entry-point compatibility wrapper.

The current Streamlit deployment is configured to launch validate.py.
Keep the interactive application in app.py and import it here so the
existing deployment URL opens the actual simulator.

For command-line engine checks, run:
    python validate_engine.py
"""

# Importing app executes its Streamlit page definitions.
import app  # noqa: F401
