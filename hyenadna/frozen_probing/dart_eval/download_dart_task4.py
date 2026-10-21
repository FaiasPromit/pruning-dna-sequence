"""
Download only the DART-Eval Task 4 data that probing needs, from Synapse.

The Task 4 folder on Synapse (syn60581041) also holds DART-Eval's own embeddings,
probes and outputs for the models of the DART-Eval paper, which are large (about
168 GB) and not needed here. This script downloads only:
    refs/                                      the genome FASTA (syn60581044)
    task_4_chromatin_activity/processed_data/  peak and non-peak BED files and the
                                               DNase bigwig files (the regression labels)

It needs a Synapse personal access token in SYNAPSE_AUTH_TOKEN and the DART-Eval work
folder in DART_WORK_DIR (dart_download_task4.sbatch sets both). Folders are downloaded
with synapseutils.syncFromSynapse, which keeps the subfolder structure.
"""

import os
import sys
import synapseclient
import synapseutils

TASK4_PARENT = "syn60581041"   # DART-Eval Task 4 folder
REFS_ID      = "syn60581044"   # genome reference

WORK = os.environ.get("DART_WORK_DIR")
if not WORK:
    sys.exit("ERROR: DART_WORK_DIR not set")

token = os.environ.get("SYNAPSE_AUTH_TOKEN")
if not token:
    sys.exit("ERROR: SYNAPSE_AUTH_TOKEN not set")

syn = synapseclient.Synapse()
syn.login(authToken=token, silent=True)
print("Synapse login OK\n", flush=True)

# 1. List the Task 4 folder and find processed_data
print(f"Contents of {TASK4_PARENT}:", flush=True)
processed_id = None
for c in syn.getChildren(TASK4_PARENT, includeTypes=["folder", "file"]):
    marker = ""
    if c["name"] == "processed_data":
        processed_id = c["id"]
        marker = "   <-- the only one downloaded"
    print(f"  {c['id']:<14} {c['type'].split('.')[-1]:<12} {c['name']}{marker}", flush=True)

if processed_id is None:
    sys.exit(f"\nERROR: no 'processed_data' folder found under {TASK4_PARENT}")

# 2. Genome reference
refs_dir = os.path.join(WORK, "refs")
os.makedirs(refs_dir, exist_ok=True)
if os.listdir(refs_dir):
    print(f"\nrefs/ already has files, skipping {REFS_ID}", flush=True)
else:
    print(f"\n=== Downloading genome reference {REFS_ID} -> {refs_dir} ===", flush=True)
    synapseutils.syncFromSynapse(syn, REFS_ID, path=refs_dir)

# 3. processed_data only
proc_dir = os.path.join(WORK, "task_4_chromatin_activity", "processed_data")
os.makedirs(proc_dir, exist_ok=True)
print(f"\n=== Downloading processed_data {processed_id} -> {proc_dir} ===", flush=True)
synapseutils.syncFromSynapse(syn, processed_id, path=proc_dir)

print("\nDownload complete.", flush=True)
