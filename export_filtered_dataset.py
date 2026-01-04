from pymongo import MongoClient
import random

MONGO = "mongodb://localhost:27017/"
DB = "MIMIC"
RAW_COLLECTION = "notes"
NEW_COLLECTION = "filtered_notes"

client = MongoClient(MONGO)
db = client[DB]
notes_col = db[RAW_COLLECTION]
filtered_col = db[NEW_COLLECTION]

###########################################################
# OPTION 1 → randomly sample X patients
###########################################################
PATIENT_COUNT = 50                # ← change this number anytime

# Get unique subject_ids present in database
all_patients = notes_col.distinct("subject_id")
random.shuffle(all_patients)         # random sampling
selected_patients = all_patients[:PATIENT_COUNT]

print(f"📌 Selected {len(selected_patients)} random patients")
###########################################################


###########################################################
# OPTION 2 → manually insert list if you want specific IDs
###########################################################
# selected_patients = [10000032, 10000145, 10000354, ...]   # <- your own list
###########################################################


###########################################################
# Export notes for selected patients into new collection
###########################################################
print("\n🚀 Exporting records for selected patients...\n")

batch_size = 2000
count = 0

for pid in selected_patients:
    docs = list(notes_col.find({"subject_id": pid}))
    if docs:
        filtered_col.insert_many(docs)
        count += len(docs)

print(f"\n====================================================")
print(f"🎉 Export COMPLETE")
print(f"👤 Patients exported : {len(selected_patients)}")
print(f"📄 Notes copied     : {count}")
print(f"📂 New collection   : {NEW_COLLECTION}")
print(f"====================================================")
