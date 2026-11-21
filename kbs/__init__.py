"""IEFD-KBS – Industrial Equipment Fault Diagnosis Knowledge-Based System.

A knowledge-based system that combines
  * a frame-based equipment ontology (classes with inheritance, components, assets),
  * production rules with MYCIN-style certainty factors,
  * forward chaining (data-driven diagnosis) and backward chaining
    (goal-driven questions),
  * an explanation facility (HOW / WHY) and
  * a safety-first maintenance recommender for different stakeholders.

Only the Python standard library is required.
"""

__version__ = "2.0.0"

from .kb import KnowledgeBase, load_default_kb  # noqa: F401
from .system import DiagnosisSystem  # noqa: F401
