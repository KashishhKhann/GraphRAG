"""
Extract medical entities using SciSpaCy (NER + UMLS Linking)
"""

import scispacy
import spacy
from scispacy.linking import EntityLinker

print("🔬 Loading SciSpaCy model...")
nlp = spacy.load("en_core_sci_md")

# Add UMLS linker
linker = EntityLinker(resolve_abbreviations=True, name="umls")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True})

def extract_medical_entities(text):
    """
    Extract entities + UMLS CUIs + canonical names
    """
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        if len(ent._.kb_ents) == 0:
            continue

        cui, score = ent._.kb_ents[0]
        umls_ent = linker.kb.cui_to_entity[cui]

        entities.append({
            "text": ent.text,
            "cui": cui,
            "canonical_name": umls_ent.canonical_name,
            "score": score
        })

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
