from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import pandas as pd

try:
    import spacy
    from spacy.matcher import PhraseMatcher
except Exception:  # pragma: no cover
    spacy = None
    PhraseMatcher = None

try:
    from keybert import KeyBERT
except Exception:  # pragma: no cover
    KeyBERT = None


DEFAULT_ALIASES = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "dsa": "data structures and algorithms",
    "dbms": "database management systems",
    "os": "operating systems",
    "js": "javascript",
    "ts": "typescript",
    "rdbms": "relational database management systems",
    "ds": "data science",
    "ui": "user interface",
    "ux": "user experience",
    "api": "rest api",
    "rest": "rest api",
    "c": "c programming",
    "r": "r programming",
}


def normalize_skill_token(skill: str) -> str:
    if not isinstance(skill, str):
        return ""
    text = skill.strip().lower()
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = re.sub(r"[^a-z0-9+\s.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_skill_tokens(value: str | None) -> list[str]:
    if value is None:
        return []
    text = str(value).replace(";", ",").replace("|", ",")
    return [token.strip() for token in re.split(r"[,/]+", text) if token.strip()]


def build_taxonomy(job_df: pd.DataFrame) -> tuple[set[str], dict[str, str]]:
    taxonomy: set[str] = set()
    aliases = DEFAULT_ALIASES.copy()
    for column in ["required_skills", "good_to_have_skills"]:
        if column not in job_df.columns:
            continue
        for raw in job_df[column].fillna("").tolist():
            for token in split_skill_tokens(raw):
                cleaned = normalize_skill_token(token)
                if not cleaned:
                    continue
                taxonomy.add(cleaned)
                aliases.setdefault(cleaned, cleaned)
                if cleaned in aliases and aliases[cleaned] != cleaned:
                    taxonomy.add(aliases[cleaned])
    for skill in list(taxonomy):
        alias_key = skill.replace(" ", "")
        aliases.setdefault(alias_key, skill)
    return taxonomy, aliases


def skill_alias_expand(skill: str, alias_map: dict[str, str]) -> str:
    normalized = normalize_skill_token(skill)
    if not normalized:
        return ""
    return alias_map.get(normalized, normalized)


def extract_skills_from_text(text: str, taxonomy: set[str], alias_map: dict[str, str], nlp=None) -> list[str]:
    if not isinstance(text, str):
        return []
    text = text.lower()
    candidates: list[str] = []
    for chunk in re.split(r"[;,.\n]+", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        normalized = skill_alias_expand(chunk, alias_map)
        candidates.append(normalized)
    filtered = []
    for candidate in candidates:
        if not candidate:
            continue
        if len(candidate.split()) <= 2 and candidate not in taxonomy and candidate not in alias_map.values():
            continue
        filtered.append(candidate)
    unique = []
    seen = set()
    for item in filtered:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def _fallback_keyphrases(text: str, limit: int = 5) -> list[str]:
    if not text:
        return []
    tokens = re.findall(r"[a-z0-9][a-z0-9+.-]{1,}", text.lower())
    counts = Counter(tokens)
    phrases = []
    for phrase, _ in counts.most_common(limit):
        if len(phrase) > 2 and phrase not in {"have", "with", "using", "based", "project", "student"}:
            phrases.append(phrase)
    return phrases


def extract_student_skills(student_row: pd.Series, taxonomy: set[str], alias_map: dict[str, str], kw_model=None) -> tuple[list[str], list[str]]:
    values = []
    for field in ["skills_listed", "skills_text", "project", "internship_detail", "lagging", "want_to_learn"]:
        if field in student_row.index:
            values.append(str(student_row[field]))
    raw_skills: list[str] = []
    for value in values:
        raw_skills.extend(split_skill_tokens(value))
    found: list[str] = []
    seen = set()
    for skill in raw_skills:
        normalized = normalize_skill_token(skill)
        expanded = skill_alias_expand(normalized, alias_map)
        if not expanded:
            continue
        if len(expanded) <= 2 and expanded not in taxonomy and " " not in expanded:
            if "skills_listed" in student_row.index and normalized in str(student_row["skills_listed"]).lower():
                pass
            else:
                continue
        if expanded not in seen:
            seen.add(expanded)
            found.append(expanded)

    extra_keyphrases: list[str] = []
    if kw_model is not None:
        text_blob = " ".join([str(student_row.get(field, "")) for field in ["skills_text", "project", "internship_detail"] if field in student_row.index])
        try:
            keywords = kw_model.extract_keywords(text_blob, keyphrase_ngram_range=(1, 2), stop_words="english", use_mmr=True, diversity=0.5)
            extra_keyphrases = [normalize_skill_token(k[0]) for k in keywords if normalize_skill_token(k[0])]
        except Exception:
            extra_keyphrases = _fallback_keyphrases(text_blob)
    else:
        extra_keyphrases = _fallback_keyphrases(" ".join([str(student_row.get(field, "")) for field in ["skills_text", "project", "internship_detail"] if field in student_row.index]))

    for term in extra_keyphrases:
        term = normalize_skill_token(term)
        if term and term not in found:
            found.append(term)

    return found, extra_keyphrases


def run_skill_extraction(students: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    taxonomy, alias_map = build_taxonomy(jobs)
    kw_model = KeyBERT(model="all-MiniLM-L6-v2") if KeyBERT is not None else None

    records = []
    for idx, row in students.iterrows():
        skills, extra = extract_student_skills(row, taxonomy, alias_map, kw_model)
        records.append({
            "student_id": row.get("student_id"),
            "skills": skills,
            "n_skills": len(skills),
            "skills_str": ";".join(skills),
            "extra_keyphrases": extra,
        })
    output = pd.DataFrame(records)
    output = output.set_index("student_id")
    students = students.copy()
    for column in ["skills", "n_skills", "skills_str", "extra_keyphrases"]:
        students[column] = output[column].reindex(students["student_id"]).to_numpy()
    return students
