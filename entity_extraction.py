"""
Extract medical entities using SciSpaCy (NER + UMLS Linking)
"""

import scispacy
import spacy


def load_nlp():
    print("🔬 Loading SciSpaCy model...")
    nlp = spacy.load("en_core_sci_md")
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True})
    return nlp


def extract_medical_entities(text, nlp=None):
    """
    Extract entities + UMLS CUIs + canonical names
    """
    if nlp is None:
        nlp = load_nlp()

    linker = nlp.get_pipe("scispacy_linker")
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        if len(ent._.kb_ents) == 0:
            continue

        cui, score = ent._.kb_ents[0]
        umls_ent = linker.kb.cui_to_entity[cui]

        entities.append(
            {
                "text": ent.text,
                "cui": cui,
                "canonical_name": umls_ent.canonical_name,
                "score": score,
            }
        )

    return entities


if __name__ == "__main__":
    sample = """
    Patient has chronic liver cirrhosis, worsening ascites,
    HIV on ART, COPD, and was given furosemide and spironolactone.
    """

    print("\n🔍 Extracting entities...\n")
    ents = extract_medical_entities(sample)

    for e in ents:
        print(f"{e['text']}  →  {e['canonical_name']} (CUI={e['cui']})")
