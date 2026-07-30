"""spaCy NER + PhraseMatcher-based resume information extraction.

spaCy's small English model (`en_core_web_sm`) has no native "SKILL" entity
label, so we use a `PhraseMatcher` against a curated skill vocabulary to
detect skills. We additionally use the model's built-in NER labels for
organizations (`ORG`) and locations (`GPE`), and a regex pass for degree
names.

Key responsibilities
--------------------
* Load the spaCy model lazily and surface a clear `RuntimeError` with the
  exact download command if `en_core_web_sm` is missing.
* Build a `PhraseMatcher` once from a curated skill list.
* `extract_entities(text)` returns a dict of:
    - skills        : list[str]
    - organizations : list[str]
    - degrees       : list[str]
    - locations     : list[str]
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import spacy
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc

# ---------------------------------------------------------------------------
# Curated skill vocabulary
# ---------------------------------------------------------------------------

# A curated, deliberately non-exhaustive list of common technical/professional
# skills. The PhraseMatcher matches these case-insensitively (via the model's
# lowercasing rules) as whole token spans. Extend this list to improve recall.
SKILL_VOCABULARY: List[str] = [
    # Programming languages
    "Python", "Java", "JavaScript", "TypeScript", "C", "C++", "C#", "Go",
    "Rust", "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R", "MATLAB",
    "Perl", "Shell", "Bash", "SQL", "PL/SQL",
    # Web / frontend
    "HTML", "CSS", "React", "Angular", "Vue", "Node.js", "Express",
    "Django", "Flask", "FastAPI", "Spring", "Bootstrap", "Tailwind",
    "jQuery", "Redux", "Next.js", "Svelte",
    # Data / ML
    "Machine Learning", "Deep Learning", "Natural Language Processing",
    "NLP", "Computer Vision", "Data Science", "Data Analysis",
    "Data Engineering", "Pandas", "NumPy", "SciPy", "scikit-learn",
    "TensorFlow", "Keras", "PyTorch", "Hugging Face", "Spark", "Hadoop",
    "ETL", "Power BI", "Tableau", "Excel", "Airflow", "Kafka",
    # Cloud / DevOps
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Ansible",
    "Jenkins", "CI/CD", "Git", "GitHub", "GitLab", "Linux", "Unix",
    "Nginx", "Apache",
    # Databases
    "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "Oracle",
    "SQLite", "Cassandra", "DynamoDB", "Snowflake",
    # Tools / concepts
    "REST", "GraphQL", "gRPC", "Microservices", "Agile", "Scrum", "Kanban",
    "Jira", "Confluence", "Selenium", "Cypress", "JUnit", "pytest",
    "Object-Oriented Programming", "OOP", "Data Structures", "Algorithms",
    "Statistics", "Probability", "A/B Testing", "Project Management",
    "Communication", "Leadership", "Teamwork", "Problem Solving",
]

# spaCy PhraseMatcher label for skills. We use the convention "SKILL" so it's
# easy to filter matched spans.
_SKILL_LABEL = "SKILL"

# ---------------------------------------------------------------------------
# Degree regex
# ---------------------------------------------------------------------------

# Matches common degree patterns: B.Tech, B.E, B.E.(Computer), Bachelor of
# Technology, M.S., M.Tech, Ph.D, MBA, etc. Case-insensitive, allows dots and
# optional specialization in parentheses.
_DEGREE_REGEX = re.compile(
    r"\b("
    r"B\.?\s?Tech\.?|B\.?\s?E\.?|B\.?\s?Sc\.?|Bachelor(?:\s+of\s+(?:Science|Engineering|Technology|Arts|Commerce|Business))?|"
    r"M\.?\s?Tech\.?|M\.?\s?E\.?|M\.?\s?Sc\.?|M\.?\s?A\.?|M\.?\s?Com\.?|"
    r"Master(?:\s+of\s+(?:Science|Engineering|Technology|Arts|Commerce|Business|Computer\s+Applications))?|"
    r"MBA|M\.?\s?B\.?\s?A\.?|PGDM|Post\s+Graduate\s+Diploma|"
    r"Ph\.?\s?D\.?|Doctorate|Doctor\s+of\s+Philosophy|"
    r"Diploma|Associate\s+Degree|High\s+School\s+Diploma"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Model loading with a clear error message
# ---------------------------------------------------------------------------

def _load_spacy_model() -> Language:
    """Load `en_core_web_sm`, raising a clear RuntimeError if missing.

    spaCy raises a generic error when a model isn't installed. We catch that
    and re-raise with the exact download command so the user knows exactly
    what to run instead of seeing a cryptic traceback.
    """
    model_name = "en_core_web_sm"
    try:
        # `spacy.load` raises if the model package isn't installed.
        nlp = spacy.load(model_name)
    except OSError as exc:
        # This is the exact error spaCy raises for a missing model.
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. "
            f"Please install it by running:\n\n"
            f"    python -m spacy download {model_name}\n\n"
            f"Then restart the application."
        ) from exc
    return nlp


# ---------------------------------------------------------------------------
# NER engine
# ---------------------------------------------------------------------------

class NEREngine:
    """Wraps a spaCy model + PhraseMatcher for resume entity extraction.

    The model and matcher are built lazily on first use so importing this
    module is cheap and side-effect free.
    """

    def __init__(self, skill_vocab: List[str] = SKILL_VOCABULARY) -> None:
        self.skill_vocab = skill_vocab
        self._nlp: Optional[Language] = None
        self._phrase_matcher: Optional[PhraseMatcher] = None

    # ------------------------------------------------------------------
    # Lazy builders
    # ------------------------------------------------------------------

    @property
    def nlp(self) -> Language:
        if self._nlp is None:
            self._nlp = _load_spacy_model()
        return self._nlp

    @property
    def phrase_matcher(self) -> PhraseMatcher:
        if self._phrase_matcher is None:
            # `attr="LOWER"` makes the matcher case-insensitive by comparing
            # the lowercased token text. This lets "python" match "Python".
            matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            # Build Doc patterns once; this is the recommended efficient way.
            patterns = [self.nlp.make_doc(skill) for skill in self.skill_vocab]
            matcher.add(_SKILL_LABEL, patterns)
            self._phrase_matcher = matcher
        return self._phrase_matcher

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract skills, organizations, degrees and locations from text.

        Returns a dict with four keys, each a list of unique strings (order
        preserved by first occurrence). Empty lists when nothing is found.
        """
        if not text or not text.strip():
            return {
                "skills": [],
                "organizations": [],
                "degrees": [],
                "locations": [],
            }

        doc: Doc = self.nlp(text)

        # --- Skills via PhraseMatcher -------------------------------------
        skills: List[str] = []
        seen_skills = set()
        matches = self.phrase_matcher(doc)
        for _, start, end in matches:
            span = doc[start:end]
            skill_text = span.text.strip()
            # Dedupe case-insensitively while preserving the original casing
            # of the first occurrence.
            key = skill_text.lower()
            if key and key not in seen_skills:
                seen_skills.add(key)
                skills.append(skill_text)

        # --- Organizations & locations via built-in NER -------------------
        organizations: List[str] = []
        locations: List[str] = []
        seen_orgs = set()
        seen_locs = set()
        for ent in doc.ents:
            if ent.label_ == "ORG":
                name = ent.text.strip()
                key = name.lower()
                if name and key not in seen_orgs:
                    seen_orgs.add(key)
                    organizations.append(name)
            elif ent.label_ == "GPE":  # Geo-Political Entity = location
                name = ent.text.strip()
                key = name.lower()
                if name and key not in seen_locs:
                    seen_locs.add(key)
                    locations.append(name)

        # --- Degrees via regex --------------------------------------------
        degrees: List[str] = []
        seen_degrees = set()
        for m in _DEGREE_REGEX.finditer(text):
            degree = m.group(0).strip()
            # Normalize internal whitespace (e.g. "B. Tech" vs "B.Tech").
            key = re.sub(r"\s+", " ", degree.lower())
            if degree and key not in seen_degrees:
                seen_degrees.add(key)
                degrees.append(degree)

        return {
            "skills": skills,
            "organizations": organizations,
            "degrees": degrees,
            "locations": locations,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

# Shared instance; the spaCy model is expensive to load so we reuse it.
ner_engine = NEREngine()