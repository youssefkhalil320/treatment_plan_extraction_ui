import os
import json
import uuid
import tempfile
import gradio as gr
from datasets import load_dataset

# =========================
# Config
# =========================
DATASET_NAME = "youssefkhalil320/MedSynth-1perICD10_new"
SPLIT = "train"
NUM_ANNOTATORS = 10  # annotator IDs: 0..9


def out_path_for(annotator_id: int) -> str:
    return f"extractions_annotator_{int(annotator_id)}.json"


# =========================
# Load dataset
# =========================
ds = load_dataset(DATASET_NAME, split=SPLIT)
COLS = list(ds.column_names)


def find_col(preferred: str):
    low = {c.lower(): c for c in COLS}
    if preferred.lower() in low:
        return low[preferred.lower()]
    for c in COLS:
        if preferred.lower() in c.lower():
            return c
    return None


DIALOGUE_COL = find_col("dialogue")
NOTE_COL = find_col("note")
ICD10_COL = find_col("icd10")
ICD10_DESC_COL = find_col("icd10_desc")

print("Detected columns:")
print("  DIALOGUE_COL =", DIALOGUE_COL)
print("  NOTE_COL =", NOTE_COL)
print("  ICD10_COL =", ICD10_COL)
print("  ICD10_DESC =", ICD10_DESC_COL)
print("All columns:", COLS)

# =========================
# Schema columns (DataFrames)
# =========================
LAB_COLS = ["lab_investigation_name", "sample_source", "order_timing"]
# For imaging: contrast will use radio buttons separately
IMAGING_COLS = ["imaging_modality", "site", "view", "order_timing"]
NCS_COLS = ["study_type", "target_region", "order_timing"]
# Tissue sampling - only procedure_type in dataframe, tissue_sampling_ordered is separate radio
TISSUE_COLS = ["procedure_type"]
# Monitoring (per schema)
MONITOR_COLS = ["monitoring_parameter", "frequency_interval", "duration"]
# Treatment
PREVENTION_COLS = ["active_prevention"]
# Conservative (per schema wording)
CONSERVATIVE_METHOD_COLS = ["conservative_method"]
LIFESTYLE_COLS = ["lifestyle_habit_modifications"]
# Medical (12 fields per schema)
MED_COLS = [
    "name",
    "treatment_status",
    "frequency",
    "dose",
    "duration",
    "discontinuation_criteria",
    "route",
    "dosage_form",
    "timing_instructions",
    "side_effects_contraindications",
    "drug_class",
    "condition_treated",
]
# Surgical
SURG_COLS = ["procedure_name", "site"]
# Follow-up (unlimited) - follow_up_ordered will use radio separately
FOLLOWUP_COLS = ["scheduled_follow_up_time", "aim_of_follow_up"]
# Referral (unlimited) - referral_ordered will use radio separately
REFERRAL_COLS = ["specialty_or_doctor", "aim_of_referral"]

ISSUE_TYPE_CHOICES = [
    "Dialogue unclear / incomplete",
    "Contradiction in case",
    "Missing key clinical info",
    "Ambiguous timeline",
    "Non-medical / garbage text",
    "Unsafe / harmful recommendation",
    "Other",
]


# =========================
# Helpers
# =========================
def norm_scalar(x):
    if x is None:
        return None
    s = str(x).strip()
    return None if s == "" else s


def safe_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def empty_rows(cols, n=1):
    return [["" for _ in cols] for _ in range(max(1, int(n)))]


def rows_to_items(rows, cols):
    """ DataFrame rows -> list[dict] or None (drops fully-empty rows) """
    if not rows:
        return None
    items = []
    for r in rows:
        r = (list(r) + [""] * len(cols))[: len(cols)]
        item = {c: norm_scalar(v) for c, v in zip(cols, r)}
        if any(v is not None for v in item.values()):
            items.append(item)
    return items if items else None


def dict_items_to_rows(items, cols):
    """ list[dict] -> DataFrame rows, at least 1 row """
    if not items or not isinstance(items, list):
        return empty_rows(cols, 1)
    rows = []
    for it in items:
        if not isinstance(it, dict):
            continue
        rows.append([(it.get(c) or "") for c in cols])
    return rows if rows else empty_rows(cols, 1)


def df_add_row(rows, cols):
    rows = rows or []
    rows = [list(r) for r in rows] if isinstance(rows, list) else []
    rows.append(["" for _ in cols])
    return rows


def df_remove_row(rows, cols):
    rows = rows or []
    rows = [list(r) for r in rows] if isinstance(rows, list) else []
    if len(rows) <= 1:
        return empty_rows(cols, 1)
    return rows[:-1]


def df_clear_rows(cols):
    return empty_rows(cols, 1)


def assigned_indices(annotator_id: int):
    a = int(annotator_id)
    return list(range(a, len(ds), NUM_ANNOTATORS))


def clamp_pos(pos: int, n: int):
    if n <= 0:
        return 0
    return max(0, min(int(pos), n - 1))


def load_existing_json_list(path: str):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def atomic_save_json(path: str, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

def error_report_dir(annotator_id: int) -> str:
    return f"ann_{int(annotator_id)}_errors_reports"


def get_error_report_content(annotator_id: int, global_row_idx: int) -> str:
    """Read the error report txt file for this annotator + case, if it exists."""
    dir_path = error_report_dir(int(annotator_id))
    txt_path = os.path.join(dir_path, f"{int(global_row_idx)}.txt")
    if not os.path.exists(txt_path):
        return ""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def render_dialogue_html(text):
    import html
    escaped = html.escape(text or "")
    return (
        f'<div style="height:600px;overflow-y:auto;padding:10px;'
        f'background:#1f2937;color:#f3f4f6;font-size:13px;'
        f'line-height:1.6;white-space:pre-wrap;border-radius:6px;'
        f'border:1px solid #374151;">{escaped}</div>'
    )

    
# =========================
# Saved-record cache + lookup
# =========================
_SAVED_CACHE = {}


def _build_saved_index(annotator_id: int):
    path = out_path_for(annotator_id)
    if not os.path.exists(path):
        return None, {}
    mtime = os.path.getmtime(path)
    data = load_existing_json_list(path)
    idx = {}
    for x in data:
        if not isinstance(x, dict):
            continue
        rid = x.get("id", x.get("row_index"))
        if isinstance(rid, int):
            idx[rid] = x
        elif isinstance(rid, str) and rid.isdigit():
            idx[int(rid)] = x
    return mtime, idx


def get_saved_index(annotator_id: int) -> dict:
    annotator_id = int(annotator_id)
    path = out_path_for(annotator_id)
    mtime = os.path.getmtime(path) if os.path.exists(path) else None
    cached = _SAVED_CACHE.get(annotator_id)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    new_mtime, idx = _build_saved_index(annotator_id)
    _SAVED_CACHE[annotator_id] = (new_mtime, idx)
    return idx


def get_saved_record(annotator_id: int, global_row_idx: int):
    idx = get_saved_index(int(annotator_id))
    return idx.get(int(global_row_idx))


# =========================
# Progress helpers
# =========================
def completed_global_rows_for(annotator_id: int) -> set[int]:
    return set(get_saved_index(int(annotator_id)).keys())


def infer_resume_shard_pos(annotator_id: int) -> int:
    shard = assigned_indices(int(annotator_id))
    if not shard:
        return 0
    done = completed_global_rows_for(int(annotator_id))
    for pos, global_i in enumerate(shard):
        if global_i not in done:
            return pos
    return len(shard) - 1


def infer_next_unfinished_after(annotator_id: int, current_pos: int) -> int:
    annotator_id = int(annotator_id)
    shard = assigned_indices(annotator_id)
    n = len(shard)
    if n == 0:
        return 0
    done = completed_global_rows_for(annotator_id)
    current_pos = clamp_pos(current_pos, n)
    for pos in range(current_pos + 1, n):
        if shard[pos] not in done:
            return pos
    for pos in range(0, current_pos + 1):
        if shard[pos] not in done:
            return pos
    return n - 1


def is_current_saved(annotator_id: int, shard_pos: int) -> bool:
    shard = assigned_indices(int(annotator_id))
    if not shard:
        return True
    shard_pos = clamp_pos(int(shard_pos), len(shard))
    global_i = shard[shard_pos]
    return global_i in completed_global_rows_for(int(annotator_id))


def global_row_to_shard_pos(annotator_id: int, global_row_idx: int) -> int:
    """Convert global row index to shard position for this annotator"""
    annotator_id = int(annotator_id)
    global_row_idx = int(global_row_idx)
    shard = assigned_indices(annotator_id)
    
    if global_row_idx not in shard:
        return -1  # Not in this annotator's shard
    
    return shard.index(global_row_idx)


def jump_to_case_guarded(annotator_id, jump_to_case):
    """Jump to a specific case by global row index"""
    if annotator_id is None or str(annotator_id).strip() == "":
        out = list(empty_ui())
        out[-2] = "Please select an annotator first."  # jump_status
        return tuple(out)
    
    if jump_to_case is None or jump_to_case < 0:
        out = list(load_case(int(annotator_id), 0))
        out[-2] = "Please enter a valid case ID."  # jump_status
        return tuple(out)
    
    annotator_id = int(annotator_id)
    global_row_idx = int(jump_to_case)
    
    # Check if this case belongs to this annotator
    shard_pos = global_row_to_shard_pos(annotator_id, global_row_idx)
    
    if shard_pos == -1:
        # Case not in this annotator's shard
        shard = assigned_indices(annotator_id)
        out = list(load_case(annotator_id, 0))
        out[-2] = f"❌ Case {global_row_idx} is not assigned to Annotator {annotator_id}. Your cases: {min(shard)}-{max(shard)} (every {NUM_ANNOTATORS}th row)"
        return tuple(out)
    
    # Load the case
    out = list(load_case(annotator_id, shard_pos))
    out[-2] = f"✅ Jumped to Case {global_row_idx}"
    return tuple(out)

# =========================
# Medication management functions
# =========================
def med_list_to_state(items):
    """Convert list of med dicts to State format"""
    if not items or not isinstance(items, list):
        return []
    return [dict(m) for m in items if isinstance(m, dict)]


def state_to_med_items(state):
    """Convert State format back to list of dicts"""
    if not state:
        return None
    items = []
    for m in state:
        if not isinstance(m, dict):
            continue
        item = {c: norm_scalar(m.get(c)) for c in MED_COLS}
        if any(v is not None for v in item.values()):
            items.append(item)
    return items if items else None


def meds_to_display_df(meds_state):
    """Convert medications state to display dataframe"""
    if not meds_state or len(meds_state) == 0:
        return []
    rows = []
    for idx, med in enumerate(meds_state):
        rows.append([
            idx + 1,  # Medication number
            med.get('name', ''),
            med.get('treatment_status', ''),
            med.get('dose', ''),
            med.get('frequency', ''),
            med.get('duration', '')
        ])
    return rows


# =========================
# Imaging management functions
# =========================
def imaging_list_to_state(items):
    """Convert list of imaging dicts to State format (rows + contrast values)"""
    if not items or not isinstance(items, list):
        return [], []
    rows = []
    contrasts = []
    for it in items:
        if not isinstance(it, dict):
            continue
        rows.append([it.get(c, "") for c in IMAGING_COLS])
        contrasts.append(it.get("contrast", "No"))
    return rows if rows else empty_rows(IMAGING_COLS, 1), contrasts if contrasts else ["No"]


def state_to_imaging_items(rows, contrasts):
    """Convert State format back to list of dicts"""
    if not rows:
        return None
    items = []
    for idx, r in enumerate(rows):
        r = (list(r) + [""] * len(IMAGING_COLS))[:len(IMAGING_COLS)]
        item = {c: norm_scalar(v) for c, v in zip(IMAGING_COLS, r)}
        # Add contrast value
        if idx < len(contrasts):
            item["contrast"] = contrasts[idx]
        else:
            item["contrast"] = "No"
        if any(v is not None for v in item.values()):
            items.append(item)
    return items if items else None


# =========================
# Empty UI
# =========================
def empty_ui():
    issue_types_upd = gr.update(value=[], visible=False)
    issue_details_upd = gr.update(value="", visible=False)
 
    radio_updates = []
    for i in range(20):
        if i == 0:
            radio_updates.append(gr.update(visible=True, value="No"))
        else:
            radio_updates.append(gr.update(visible=False))
 
    return (
        None,           # annotator_id
        0,              # shard_pos_state
        "Select an annotator to start.",  # progress
        "", "", "", "", "",  # dialogue, note, icd10, icd10_desc, debug_box
        empty_rows(LAB_COLS, 1),
        empty_rows(IMAGING_COLS, 1),
        ["No"],         # imaging_contrasts_state
        empty_rows(NCS_COLS, 1),
        empty_rows(TISSUE_COLS, 1),
        empty_rows(MONITOR_COLS, 1),
        empty_rows(PREVENTION_COLS, 1),
        empty_rows(CONSERVATIVE_METHOD_COLS, 1),
        empty_rows(LIFESTYLE_COLS, 1),
        [],             # meds_state
        [],             # meds_display_df
        empty_rows(SURG_COLS, 1),
        "", "",         # complementary_therapies, external_equipment
        empty_rows(FOLLOWUP_COLS, 1),
        "",             # patient_education
        "",             # when_to_seek_medical_care
        empty_rows(REFERRAL_COLS, 1),
        "No", issue_types_upd, issue_details_upd,
        # ---- error report ----
        "",             # error_report_content
        gr.update(visible=False),  # error_report_box (hidden when no report)
    ) + tuple(radio_updates) + ("", "Please select an annotator.",)

# =========================
# Load a case
# =========================
def load_case(annotator_id, shard_pos):
    annotator_id = int(annotator_id)
    shard = assigned_indices(annotator_id)
    shard_pos = clamp_pos(int(shard_pos), len(shard))
    global_i = shard[shard_pos]
    r = ds[global_i]

    dialogue_val = r.get(DIALOGUE_COL, "") if DIALOGUE_COL else ""
    note_val = r.get(NOTE_COL, "") if NOTE_COL else ""
    icd10_val = r.get(ICD10_COL, "") if ICD10_COL else ""
    icd10_desc_val = r.get(ICD10_DESC_COL, "") if ICD10_DESC_COL else ""

    debug = (
        f"Annotator={annotator_id} | shard_pos={shard_pos} | global_row={global_i}\n"
        f"Detected: dialogue={DIALOGUE_COL}, note={NOTE_COL}, icd10={ICD10_COL}, icd10_desc={ICD10_DESC_COL}\n"
        f"Row keys: {list(r.keys())}\n"
        f"Dialogue chars: {len(safe_text(dialogue_val))}, Note chars: {len(safe_text(note_val))}"
    )

    done = len(completed_global_rows_for(annotator_id))
    total = len(shard)
    progress_txt = f"Annotator {annotator_id}: {done} done / {total} total | Now: {shard_pos+1}/{total} (global row {global_i})"


    # ---- error report for this case ----
    error_content = get_error_report_content(annotator_id, global_i)
    has_error_report = bool(error_content.strip())
    error_report_upd = gr.update(
        visible=has_error_report,
        value=error_content,
        label=f"⚠️ Error Report — Case {global_i}  (ann_{annotator_id}_errors_reports/{global_i}.txt)",
    )

    # Defaults
    labs_rows = empty_rows(LAB_COLS, 1)
    imaging_rows = empty_rows(IMAGING_COLS, 1)
    imaging_contrasts = ["No"]
    ncs_rows = empty_rows(NCS_COLS, 1)
    tissue_rows = empty_rows(TISSUE_COLS, 1)
    monitor_rows = empty_rows(MONITOR_COLS, 1)
    prevention_rows = empty_rows(PREVENTION_COLS, 1)
    conservative_rows = empty_rows(CONSERVATIVE_METHOD_COLS, 1)
    lifestyle_rows = empty_rows(LIFESTYLE_COLS, 1)
    meds_state = []
    surg_rows = empty_rows(SURG_COLS, 1)
    followup_rows = empty_rows(FOLLOWUP_COLS, 1)
    referral_rows = empty_rows(REFERRAL_COLS, 1)
    patient_education_val = ""
    when_to_seek_medical_care_val = ""
    complementary_therapies_val = ""
    external_equipment_val = ""
    report_issue_val = "No"
    issue_types_upd = gr.update(value=[], visible=False)
    issue_details_upd = gr.update(value="", visible=False)
    jump_status_msg = ""  # ADD THIS LINE
    status_msg = ""

    saved = get_saved_record(annotator_id, global_i)
    if isinstance(saved, dict):
        status_msg = "Loaded saved values ✅"
        icd10_val = saved.get("icd10", icd10_val) or icd10_val
        icd10_desc_val = saved.get("icd10_desc", icd10_desc_val) or icd10_desc_val

        ext = saved.get("extraction") or {}
        inv = ext.get("INVESTIGATIONS") or {}
        trt = ext.get("TREATMENT") or {}

        labs_rows = dict_items_to_rows(inv.get("Labs"), LAB_COLS)

        # Handle imaging with contrast
        img_items = inv.get("Imaging")
        if img_items:
            imaging_rows, imaging_contrasts = imaging_list_to_state(img_items)

        ncs_rows = dict_items_to_rows(inv.get("Nerve_Muscle_Conduction_Studies"), NCS_COLS)
        tissue_rows = dict_items_to_rows(inv.get("Tissue_sampling"), TISSUE_COLS)
        monitor_rows = dict_items_to_rows(ext.get("MONITORING"), MONITOR_COLS)

        prev_items = trt.get("Prevention")
        if isinstance(prev_items, list):
            prevention_rows = [[p or ""] for p in prev_items] if prev_items else empty_rows(PREVENTION_COLS, 1)

        cons = trt.get("Conservative") or {}
        cm = cons.get("conservative_method")
        if cm is None and isinstance(cons.get("supportive_method"), list):
            cm = cons.get("supportive_method")
        if isinstance(cm, list):
            conservative_rows = [[x or ""] for x in cm] if cm else empty_rows(CONSERVATIVE_METHOD_COLS, 1)

        life = cons.get("lifestyle_habit_modifications")
        if isinstance(life, list):
            lifestyle_rows = [[x or ""] for x in life] if life else empty_rows(LIFESTYLE_COLS, 1)

        # Medical - load into state
        med_items = trt.get("Medical")
        if isinstance(med_items, list):
            meds_state = med_list_to_state(med_items)

        surg_rows = dict_items_to_rows(trt.get("Surgical"), SURG_COLS)

        complementary_therapies_val = trt.get("COMPLEMENTARY_THERAPIES") or ext.get("COMPLEMENTARY_THERAPIES") or ""
        external_equipment_val = trt.get("EXTERNAL_EQUIPMENT") or ext.get("EXTERNAL_EQUIPMENT") or ""

        followup_rows = dict_items_to_rows(ext.get("FOLLOW_UP"), FOLLOWUP_COLS)

        patient_education_val = ext.get("PATIENT_EDUCATION") or ""
        when_to_seek_medical_care_val = ext.get("WHEN_TO_SEEK_MEDICAL_CARE") or ""

        referral_rows = dict_items_to_rows(ext.get("REFERRAL"), REFERRAL_COLS)

        issue = saved.get("issue_report")
        if isinstance(issue, dict) and issue.get("reported") is True:
            report_issue_val = "Yes"
            types = issue.get("types") if isinstance(issue.get("types"), list) else []
            details = issue.get("details") or ""
            issue_types_upd = gr.update(value=types, visible=True)
            issue_details_upd = gr.update(value=details, visible=True)

    # Create visibility updates for contrast radio buttons
    num_imaging_rows = len(imaging_rows)
    radio_updates = []
    for i in range(20):  # Max 20 rows
        if i < num_imaging_rows:
            radio_updates.append(gr.update(visible=True, value=imaging_contrasts[i] if i < len(imaging_contrasts) else "No"))
        else:
            radio_updates.append(gr.update(visible=False))

    return (
        str(annotator_id),  # annotator_id
        shard_pos,  # shard_pos_state
        progress_txt,  # progress
        safe_text(dialogue_val),  # dialogue
        safe_text(note_val),  # note
        safe_text(icd10_val),  # icd10
        safe_text(icd10_desc_val),  # icd10_desc
        debug,  # debug_box
        labs_rows,  # labs_df
        imaging_rows,  # imaging_df
        imaging_contrasts,  # imaging_contrasts_state
        ncs_rows,  # ncs_df
        tissue_rows,  # tissue_df
        monitor_rows,  # monitor_df
        prevention_rows,  # prevention_df
        conservative_rows,  # conservative_df
        lifestyle_rows,  # lifestyle_df
        meds_state,  # meds_state
        meds_to_display_df(meds_state),  # meds_display_df
        surg_rows,  # surg_df
        complementary_therapies_val,  # complementary_therapies
        external_equipment_val,  # external_equipment
        followup_rows,  # followup_df
        patient_education_val,  # patient_education
        when_to_seek_medical_care_val,  # when_to_seek_medical_care
        referral_rows,  # referral_df
        report_issue_val,  # report_issue
        issue_types_upd,  # issue_types
        issue_details_upd,  # issue_details
        # ---- error report ----
        error_content,
        error_report_upd,
    ) + tuple(radio_updates) + (jump_status_msg, status_msg,)  # ADD jump_status_msg before status_msg


def resume_case_guarded(annotator_id):
    if annotator_id is None or str(annotator_id).strip() == "":
        return empty_ui()
    pos = infer_resume_shard_pos(int(annotator_id))
    return load_case(int(annotator_id), pos)


def prev_case_guarded(annotator_id, shard_pos):
    if annotator_id is None or str(annotator_id).strip() == "":
        return empty_ui()
    shard = assigned_indices(int(annotator_id))
    shard_pos = clamp_pos(int(shard_pos) - 1, len(shard))
    return load_case(int(annotator_id), shard_pos)


# def next_case_guarded(annotator_id, shard_pos):
#     if annotator_id is None or str(annotator_id).strip() == "":
#         return empty_ui()
#     if not is_current_saved(int(annotator_id), int(shard_pos)):
#         out = list(load_case(int(annotator_id), int(shard_pos)))
#         out[-1] = "⚠️ Please click Save before going to Next."
#         return tuple(out)
#     shard = assigned_indices(int(annotator_id))
#     shard_pos = clamp_pos(int(shard_pos) + 1, len(shard))
#     return load_case(int(annotator_id), shard_pos)

def next_case_guarded(annotator_id, shard_pos):
    if annotator_id is None or str(annotator_id).strip() == "":
        return empty_ui()
    if not is_current_saved(int(annotator_id), int(shard_pos)):
        out = list(load_case(int(annotator_id), int(shard_pos)))
        out[-1] = "⚠️ Please click Save before going to Next."
        return tuple(out)
    shard = assigned_indices(int(annotator_id))
    next_pos = int(shard_pos) + 1
    if next_pos >= len(shard):  # <-- ADD THIS CHECK
        out = list(load_case(int(annotator_id), int(shard_pos)))
        out[-1] = "🎉 You've reached the end of your assigned cases!"
        return tuple(out)
    shard_pos = clamp_pos(next_pos, len(shard))
    return load_case(int(annotator_id), shard_pos)


# =========================
# Imaging handlers
# =========================
def add_imaging_row(imaging_rows, imaging_contrasts):
    """Add new imaging row with default contrast value"""
    imaging_rows = imaging_rows or []
    imaging_rows = [list(r) for r in imaging_rows] if isinstance(imaging_rows, list) else []
    imaging_rows.append(["" for _ in IMAGING_COLS])
    
    imaging_contrasts = imaging_contrasts or []
    imaging_contrasts = list(imaging_contrasts) if isinstance(imaging_contrasts, list) else []
    imaging_contrasts.append("No")
    
    # Create visibility updates for radio buttons
    num_rows = len(imaging_rows)
    radio_updates = []
    for i in range(20):  # Max 20 rows
        if i < num_rows:
            radio_updates.append(gr.update(visible=True, value=imaging_contrasts[i] if i < len(imaging_contrasts) else "No"))
        else:
            radio_updates.append(gr.update(visible=False))
    
    return [imaging_rows, imaging_contrasts] + radio_updates


def remove_imaging_row(imaging_rows, imaging_contrasts):
    """Remove last imaging row"""
    imaging_rows = imaging_rows or []
    imaging_rows = [list(r) for r in imaging_rows] if isinstance(imaging_rows, list) else []
    imaging_contrasts = imaging_contrasts or []
    imaging_contrasts = list(imaging_contrasts) if isinstance(imaging_contrasts, list) else []
    
    if len(imaging_rows) <= 1:
        imaging_rows = empty_rows(IMAGING_COLS, 1)
        imaging_contrasts = ["No"]
    else:
        imaging_rows = imaging_rows[:-1]
        imaging_contrasts = imaging_contrasts[:-1]
    
    # Create visibility updates for radio buttons
    num_rows = len(imaging_rows)
    radio_updates = []
    for i in range(20):  # Max 20 rows
        if i < num_rows:
            radio_updates.append(gr.update(visible=True, value=imaging_contrasts[i] if i < len(imaging_contrasts) else "No"))
        else:
            radio_updates.append(gr.update(visible=False))
    
    return [imaging_rows, imaging_contrasts] + radio_updates


def clear_imaging(imaging_rows, imaging_contrasts):
    """Clear all imaging rows"""
    imaging_rows = empty_rows(IMAGING_COLS, 1)
    imaging_contrasts = ["No"]
    
    # Create visibility updates for radio buttons
    radio_updates = []
    for i in range(20):  # Max 20 rows
        if i == 0:
            radio_updates.append(gr.update(visible=True, value="No"))
        else:
            radio_updates.append(gr.update(visible=False))
    
    return [imaging_rows, imaging_contrasts] + radio_updates


def update_imaging_contrast(imaging_contrasts, row_idx, new_contrast):
    """Update contrast value for a specific imaging row"""
    imaging_contrasts = list(imaging_contrasts) if isinstance(imaging_contrasts, list) else []
    # Ensure list is long enough
    while len(imaging_contrasts) <= row_idx:
        imaging_contrasts.append("No")
    imaging_contrasts[row_idx] = new_contrast
    return imaging_contrasts


# =========================
# Save + auto-advance
# =========================
def save_and_advance_guarded(
    annotator_id,
    shard_pos,
    icd10,
    icd10_desc,
    labs_rows,
    imaging_rows,
    imaging_contrasts,
    ncs_rows,
    tissue_rows,
    # tissue_sampling_ordered,
    monitor_rows,
    prevention_rows,
    conservative_rows,
    lifestyle_rows,
    meds_state,
    surg_rows,
    complementary_therapies,
    external_equipment,
    followup_rows,
    # follow_up_ordered,
    patient_education,
    when_to_seek_medical_care,
    referral_rows,
    # referral_ordered,
    report_issue,
    issue_types,
    issue_details,
):
    if annotator_id is None or str(annotator_id).strip() == "":
        return empty_ui()

    annotator_id = int(annotator_id)
    shard_pos = int(shard_pos)
    shard = assigned_indices(annotator_id)
    shard_pos = clamp_pos(shard_pos, len(shard))
    global_row_idx = shard[shard_pos]

    def single_col_df_to_list(rows):
        out = []
        for r in (rows or []):
            if not r:
                continue
            v = norm_scalar(r[0])
            if v is not None:
                out.append(v)
        return out if out else None

    labs_items = rows_to_items(labs_rows, LAB_COLS)
    imaging_items = state_to_imaging_items(imaging_rows, imaging_contrasts)
    ncs_items = rows_to_items(ncs_rows, NCS_COLS)

    # Build tissue sampling items with tissue_sampling_ordered
    # tissue_items_raw = rows_to_items(tissue_rows, TISSUE_COLS)
    # tissue_items = None
    # if tissue_items_raw:
    #     tissue_items = []
    #     for item in tissue_items_raw:
    #         item["tissue_sampling_ordered"] = tissue_sampling_ordered
    #         tissue_items.append(item)
    # elif tissue_sampling_ordered == "Yes":
    #     # If tissue_sampling_ordered is Yes but no details, create empty item
    #     tissue_items = [{"tissue_sampling_ordered": tissue_sampling_ordered}]
    
    # # Convert to dict format if there's only one item (as per original schema)
    # if tissue_items and len(tissue_items) == 1:
    #     tissue_items = tissue_items[0]

    tissue_items = rows_to_items(tissue_rows, TISSUE_COLS)

    monitor_items = rows_to_items(monitor_rows, MONITOR_COLS)
    prevention_items = single_col_df_to_list(prevention_rows)
    conservative_method_items = single_col_df_to_list(conservative_rows)
    lifestyle_items = single_col_df_to_list(lifestyle_rows)
    meds_items = state_to_med_items(meds_state)
    surg_items = rows_to_items(surg_rows, SURG_COLS)

    # Build follow-up items with follow_up_ordered
    # followup_items_raw = rows_to_items(followup_rows, FOLLOWUP_COLS)
    # followup_items = None
    # if followup_items_raw:
    #     followup_items = []
    #     for item in followup_items_raw:
    #         item["follow_up_ordered"] = follow_up_ordered
    #         followup_items.append(item)
    # elif follow_up_ordered == "Yes":
    #     # If follow_up_ordered is Yes but no details, create empty item
    #     followup_items = [{"follow_up_ordered": follow_up_ordered}]

    followup_items = rows_to_items(followup_rows, FOLLOWUP_COLS)

    # Build referral items with referral_ordered
    # referral_items_raw = rows_to_items(referral_rows, REFERRAL_COLS)
    # referral_items = None
    # if referral_items_raw:
    #     referral_items = []
    #     for item in referral_items_raw:
    #         item["referral_ordered"] = referral_ordered
    #         referral_items.append(item)
    # elif referral_ordered == "Yes":
    #     # If referral_ordered is Yes but no details, create empty item
    #     referral_items = [{"referral_ordered": referral_ordered}]

    referral_items = rows_to_items(referral_rows, REFERRAL_COLS)
    record = {
        "id": global_row_idx,
        "uuid": str(uuid.uuid4()),
        "annotator_id": annotator_id,
        "shard_pos": shard_pos,
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "row_index": global_row_idx,
        "icd10": norm_scalar(icd10),
        "icd10_desc": norm_scalar(icd10_desc),
        "issue_report": None,
        "extraction": {
            "INVESTIGATIONS": {
                "Labs": labs_items,
                "Imaging": imaging_items,
                "Nerve_Muscle_Conduction_Studies": ncs_items,
                "Tissue_sampling": tissue_items,
                
            },
            "Monitoring": monitor_items,
            "TREATMENT": {
                "Prevention": prevention_items,
                "Conservative": {
                    "conservative_method": conservative_method_items,
                    "lifestyle_habit_modifications": lifestyle_items,
                },
                "Medical": meds_items,
                "Surgical": surg_items,
                "COMPLEMENTARY_THERAPIES": norm_scalar(complementary_therapies),
                "EXTERNAL_EQUIPMENT": norm_scalar(external_equipment),
            },
            "FOLLOW_UP": followup_items,
            "PATIENT_EDUCATION": norm_scalar(patient_education),
            "WHEN_TO_SEEK_MEDICAL_CARE": norm_scalar(when_to_seek_medical_care),
            "REFERRAL": referral_items,
        },
    }

    cons = record["extraction"]["TREATMENT"]["Conservative"]
    if cons["conservative_method"] is None and cons["lifestyle_habit_modifications"] is None:
        record["extraction"]["TREATMENT"]["Conservative"] = None

    if norm_scalar(report_issue) == "Yes":
        types = issue_types if isinstance(issue_types, list) else []
        types = [t for t in types if norm_scalar(t) is not None]
        record["issue_report"] = {
            "reported": True,
            "types": types if types else None,
            "details": norm_scalar(issue_details),
        }

    out_path = out_path_for(annotator_id)
    data = load_existing_json_list(out_path)
    data = [
        x
        for x in data
        if not (isinstance(x, dict) and (x.get("id") == global_row_idx or x.get("row_index") == global_row_idx))
    ]
    data.append(record)
    atomic_save_json(out_path, data)
    _SAVED_CACHE.pop(annotator_id, None)

    done = len(completed_global_rows_for(annotator_id))
    total = len(assigned_indices(annotator_id))
    saved_msg = f"Saved ✅ annotator={annotator_id} | global_row={global_row_idx} | done: {done}/{total}"

    next_pos = infer_next_unfinished_after(annotator_id, shard_pos)
    out = list(load_case(annotator_id, next_pos))
    out[-1] = saved_msg
    return tuple(out)


# =========================
# Medication handlers
# =========================
def add_new_med(meds_state):
    """Add empty medication to state and open editor"""
    new_med = {c: "" for c in MED_COLS}
    new_med["treatment_status"] = "start"  # default
    meds_state = meds_state or []
    meds_state.append(new_med)
    new_idx = len(meds_state)
    
    # Open editor with the new medication
    return (
        meds_state,
        meds_to_display_df(meds_state),
        gr.update(visible=True),
        new_idx,  # med_select_idx
        new_med.get("name", ""),
        new_med.get("treatment_status", "start"),
        new_med.get("frequency", ""),
        new_med.get("dose", ""),
        new_med.get("duration", ""),
        new_med.get("discontinuation_criteria", ""),
        new_med.get("route", ""),
        new_med.get("dosage_form", ""),
        new_med.get("timing_instructions", ""),
        new_med.get("side_effects_contraindications", ""),
        new_med.get("drug_class", ""),
        new_med.get("condition_treated", ""),
        new_idx - 1  # med_edit_idx (0-indexed)
    )


def cancel_med_edit():
    """Cancel medication editing"""
    return gr.update(visible=False)


def save_med_edit(meds_state, edit_idx, name, treatment_status, frequency, dose, duration,
                  discontinuation, route, dosage_form, timing, side_effects, drug_class, condition_treated):
    """Save edited medication and close editor"""
    if edit_idx is None or not isinstance(meds_state, list) or edit_idx >= len(meds_state):
        return meds_state, meds_to_display_df(meds_state), gr.update(visible=False)
    
    meds_state[edit_idx] = {
        "name": name,
        "treatment_status": treatment_status,
        "frequency": frequency,
        "dose": dose,
        "duration": duration,
        "discontinuation_criteria": discontinuation,
        "route": route,
        "dosage_form": dosage_form,
        "timing_instructions": timing,
        "side_effects_contraindications": side_effects,
        "drug_class": drug_class,
        "condition_treated": condition_treated,
    }
    
    return meds_state, meds_to_display_df(meds_state), gr.update(visible=False)


def load_med_for_edit(meds_state, edit_idx):
    """Load medication data into edit form"""
    if edit_idx is None or not isinstance(meds_state, list) or edit_idx >= len(meds_state):
        return (gr.update(visible=False),) + ("",) * 12 + (None,)
    
    med = meds_state[edit_idx]
    return (
        gr.update(visible=True),
        med.get("name", ""),
        med.get("treatment_status", "start"),
        med.get("frequency", ""),
        med.get("dose", ""),
        med.get("duration", ""),
        med.get("discontinuation_criteria", ""),
        med.get("route", ""),
        med.get("dosage_form", ""),
        med.get("timing_instructions", ""),
        med.get("side_effects_contraindications", ""),
        med.get("drug_class", ""),
        med.get("condition_treated", ""),
        edit_idx
    )


def load_med_for_edit_by_number(meds_state, med_num):
    """Load medication for editing based on medication number (1-indexed)"""
    if not meds_state or not isinstance(meds_state, list):
        return (gr.update(visible=False),) + ("",) * 12 + (None,)
    
    edit_idx = int(med_num) - 1  # Convert to 0-indexed
    if edit_idx < 0 or edit_idx >= len(meds_state):
        return (gr.update(visible=False),) + ("",) * 12 + (None,)
    
    return load_med_for_edit(meds_state, edit_idx)


def delete_med(meds_state, med_num):
    """Delete medication by number (1-indexed)"""
    if not meds_state or not isinstance(meds_state, list):
        return meds_state, meds_to_display_df(meds_state)
    
    del_idx = int(med_num) - 1  # Convert to 0-indexed
    if del_idx < 0 or del_idx >= len(meds_state):
        return meds_state, meds_to_display_df(meds_state)
    
    meds_state.pop(del_idx)
    return meds_state, meds_to_display_df(meds_state)


# =========================
# CSS
# =========================
CSS = """
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}
textarea {
    font-size: 14px !important;
    line-height: 1.35 !important;
}
.smallbar .gr-button {
    min-width: 140px;
}
.error-report-box textarea {
    background-color: #fff8e1 !important;
    border: 2px solid #f59e0b !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    color: #1a1a1a !important;
}
.error-report-box label {
    color: #b45309 !important;
    font-weight: 600 !important;
}
"""

# =========================
# UI
# =========================
with gr.Blocks(title="Treatment Plan Extraction - MedSynth") as demo:
    gr.Markdown("# Treatment Plan Extraction")
    gr.Markdown("⚠️ **Please select an Annotator ID to begin.**")

    shard_pos_state = gr.State(0)
    med_edit_idx = gr.State(None)

    with gr.Row():
        annotator_id = gr.Dropdown(
            choices=[str(i) for i in range(NUM_ANNOTATORS)],
            value=None,
            label="Annotator ID (0-9)",
        )
        progress = gr.Textbox(label="Progress", interactive=False)
        btn_prev = gr.Button("⬅ Prev")
        btn_next = gr.Button("Next ➡")
        btn_save = gr.Button("Save", variant="primary")
    
    with gr.Row():
        jump_to_case = gr.Number(
            label="Jump to Case ID (global row index)", 
            value=None, 
            precision=0,
            minimum=0
        )
        btn_jump = gr.Button("Go to Case")
        jump_status = gr.Textbox(label="Jump Status", interactive=False, scale=2)

    status = gr.Textbox(label="Status", interactive=False)

    with gr.Tabs():
        with gr.Tab("Dialogue"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        dialogue_search = gr.Textbox(
                            label="🔍 Search in Dialogue",
                            placeholder="Type to highlight...",
                            scale=4,
                        )
                        dialogue_search_status = gr.Textbox(
                            label="Matches",
                            interactive=False,
                            scale=1,
                        )
                    dialogue = gr.Textbox(label="Dialogue", lines=28, interactive=False, visible=False)
                    dialogue_display = gr.HTML(label="Dialogue")
                with gr.Column(scale=1):
                    error_report_content = gr.State("")
                    error_report_box = gr.Textbox(
                        label="⚠️ Error Report",
                        lines=30,
                        interactive=False,
                        visible=False,
                        elem_classes=["error-report-box"],
                    )
        with gr.Tab("Note"):
            note = gr.Textbox(label="Note", lines=20, interactive=False)
        with gr.Tab("ICD10"):
            icd10 = gr.Textbox(label="ICD10", interactive=False)
            icd10_desc = gr.Textbox(label="ICD10_desc", interactive=False)
        with gr.Tab("Debug"):
            debug_box = gr.Textbox(label="Debug info", lines=6, interactive=False)

    gr.Markdown("## Annotation form")

    with gr.Tabs():
        with gr.Tab("Investigations"):
            gr.Markdown("### Labs (multiple orders)")
            with gr.Row(elem_classes=["smallbar"]):
                btn_labs_add = gr.Button("＋ Add lab row")
                btn_labs_remove = gr.Button("－ Remove last row")
                btn_labs_clear = gr.Button("Clear labs")
            labs_df = gr.Dataframe(value=empty_rows(LAB_COLS, 1), headers=LAB_COLS, type="array", interactive=True)

            gr.Markdown("### Imaging (multiple orders)")
            gr.Markdown("**Note:** A contrast radio button will appear for each imaging row you add")
            imaging_contrasts_state = gr.State(["No"])
            with gr.Row(elem_classes=["smallbar"]):
                btn_img_add = gr.Button("＋ Add imaging row")
                btn_img_remove = gr.Button("－ Remove last row")
                btn_img_clear = gr.Button("Clear imaging")
            imaging_df = gr.Dataframe(
                value=empty_rows(IMAGING_COLS, 1), headers=IMAGING_COLS, type="array", interactive=True
            )
            gr.Markdown("#### Contrast for each imaging order:")
            # Create contrast radio buttons for each imaging row dynamically (up to 20)
            imaging_contrast_radios = []
            for i in range(20):
                contrast_radio = gr.Radio(
                    choices=["Yes", "No"],
                    value="No",
                    label=f"Row {i+1} - Contrast",
                    interactive=True,
                    visible=(i == 0)
                )
                imaging_contrast_radios.append(contrast_radio)

            gr.Markdown("### Nerve/Muscle Conduction Studies (multiple orders)")
            with gr.Row(elem_classes=["smallbar"]):
                btn_ncs_add = gr.Button("＋ Add NCS row")
                btn_ncs_remove = gr.Button("－ Remove last row")
                btn_ncs_clear = gr.Button("Clear NCS")
            ncs_df = gr.Dataframe(value=empty_rows(NCS_COLS, 1), headers=NCS_COLS, type="array", interactive=True)

            gr.Markdown("### Tissue sampling (multiple orders)")
            # tissue_sampling_ordered = gr.Radio(
            #     choices=["Yes", "No"], value="No", label="Tissue Sampling Ordered?", interactive=True
            # )
            with gr.Row(elem_classes=["smallbar"]):
                btn_tissue_add = gr.Button("＋ Add tissue row")
                btn_tissue_remove = gr.Button("－ Remove last row")
                btn_tissue_clear = gr.Button("Clear tissue")
            tissue_df = gr.Dataframe(value=empty_rows(TISSUE_COLS, 1), headers=TISSUE_COLS, type="array", interactive=True)

            gr.Markdown("### Monitoring (multiple orders)")
            with gr.Row(elem_classes=["smallbar"]):
                btn_monitor_add = gr.Button("＋ Add monitoring row")
                btn_monitor_remove = gr.Button("－ Remove last row")
                btn_monitor_clear = gr.Button("Clear monitoring")
            monitor_df = gr.Dataframe(value=empty_rows(MONITOR_COLS, 1), headers=MONITOR_COLS, type="array", interactive=True)

        with gr.Tab("Treatment"):
            gr.Markdown("### Prevention (active prevention)")
            with gr.Row(elem_classes=["smallbar"]):
                btn_prev_add = gr.Button("＋ Add prevention row")
                btn_prev_remove = gr.Button("－ Remove last row")
                btn_prev_clear = gr.Button("Clear prevention")
            prevention_df = gr.Dataframe(value=empty_rows(PREVENTION_COLS, 1), headers=PREVENTION_COLS, type="array", interactive=True)

            gr.Markdown("### Conservative method")
            with gr.Row(elem_classes=["smallbar"]):
                btn_cons_add = gr.Button("＋ Add conservative method row")
                btn_cons_remove = gr.Button("－ Remove last row")
                btn_cons_clear = gr.Button("Clear conservative methods")
            conservative_df = gr.Dataframe(value=empty_rows(CONSERVATIVE_METHOD_COLS, 1), headers=CONSERVATIVE_METHOD_COLS, type="array", interactive=True)

            gr.Markdown("### Lifestyle / habit modifications")
            with gr.Row(elem_classes=["smallbar"]):
                btn_life_add = gr.Button("＋ Add lifestyle row")
                btn_life_remove = gr.Button("－ Remove last row")
                btn_life_clear = gr.Button("Clear lifestyle")
            lifestyle_df = gr.Dataframe(value=empty_rows(LIFESTYLE_COLS, 1), headers=LIFESTYLE_COLS, type="array", interactive=True)

            gr.Markdown("### Medical (medications)")
            meds_state = gr.State([])
            # Display saved medications as a dataframe
            meds_display_df = gr.Dataframe(
                value=[],
                headers=["#", "Name", "Status", "Dose", "Frequency", "Duration"],
                label="Saved Medications",
                interactive=False,
                wrap=True
            )
            with gr.Row(elem_classes=["smallbar"]):
                btn_med_add = gr.Button("＋ Add medication", variant="primary")
                btn_med_edit = gr.Button("✏️ Edit medication")
                btn_med_delete = gr.Button("🗑️ Delete medication")
                med_select_idx = gr.Number(label="Medication # to Edit/Delete", value=1, precision=0, minimum=1)

            with gr.Group(visible=False) as med_editor:
                gr.Markdown("#### Edit Medication")
                med_name = gr.Textbox(label="Name", placeholder="Medication name")
                with gr.Row():
                    med_treatment_status = gr.Radio(
                        ["start", "stop", "continue"], label="Treatment Status", value="start"
                    )
                    med_dose = gr.Textbox(label="Dose", placeholder="e.g., 500mg")
                    med_frequency = gr.Textbox(label="Frequency", placeholder="e.g., twice daily")
                with gr.Row():
                    med_duration = gr.Textbox(label="Duration", placeholder="e.g., 7 days")
                    med_route = gr.Textbox(label="Route", placeholder="e.g., oral, IV")
                    med_dosage_form = gr.Textbox(label="Dosage Form", placeholder="e.g., tablet, capsule")
                with gr.Row():
                    med_timing = gr.Textbox(label="Timing Instructions", placeholder="e.g., with meals")
                    med_drug_class = gr.Textbox(label="Drug Class", placeholder="e.g., antibiotic")
                with gr.Row():
                    med_discontinuation = gr.Textbox(label="Discontinuation Criteria", placeholder="When to stop")
                    med_condition_treated = gr.Textbox(label="Condition Treated", placeholder="Target condition")
                med_side_effects = gr.Textbox(label="Side Effects/Contraindications", lines=2)
                with gr.Row():
                    btn_save_med = gr.Button("Save Medication", variant="primary")
                    btn_cancel_med = gr.Button("Cancel")

            gr.Markdown("### Surgical")
            with gr.Row(elem_classes=["smallbar"]):
                btn_surg_add = gr.Button("＋ Add surgical row")
                btn_surg_remove = gr.Button("－ Remove last row")
                btn_surg_clear = gr.Button("Clear surgical")
            surg_df = gr.Dataframe(value=empty_rows(SURG_COLS, 1), headers=SURG_COLS, type="array", interactive=True)

            gr.Markdown("### Complementary Therapies & Equipment")
            complementary_therapies = gr.Textbox(label="Complementary therapies", lines=3)
            external_equipment = gr.Textbox(label="External equipment?", lines=2)

        with gr.Tab("Follow-up & Other"):
            gr.Markdown("### Follow-up (UNLIMITED rows)")
            # follow_up_ordered = gr.Radio(
            #     choices=["Yes", "No"], value="No", label="Follow-up Ordered?", interactive=True
            # )
            with gr.Row(elem_classes=["smallbar"]):
                btn_fu_add = gr.Button("＋ Add follow-up row")
                btn_fu_remove = gr.Button("－ Remove last row")
                btn_fu_clear = gr.Button("Clear follow-ups")
            followup_df = gr.Dataframe(value=empty_rows(FOLLOWUP_COLS, 1), headers=FOLLOWUP_COLS, type="array", interactive=True)

            patient_education = gr.Textbox(label="Patient education", lines=4)
            when_to_seek_medical_care = gr.Textbox(
                label="When to seek medical care",
                placeholder="When should the patient contact the doctor or go to the hospital? (e.g., Severe headache, bleeding)",
                lines=3
            )

            gr.Markdown("### Referral (UNLIMITED rows)")
            # referral_ordered = gr.Radio(
            #     choices=["Yes", "No"], value="No", label="Referral Ordered?", interactive=True
            # )
            with gr.Row(elem_classes=["smallbar"]):
                btn_ref_add = gr.Button("＋ Add referral row")
                btn_ref_remove = gr.Button("－ Remove last row")
                btn_ref_clear = gr.Button("Clear referrals")
            referral_df = gr.Dataframe(value=empty_rows(REFERRAL_COLS, 1), headers=REFERRAL_COLS, type="array", interactive=True)

        with gr.Tab("Issues"):
            report_issue = gr.Radio(["Yes", "No"], value="No", label="Report an issue in this case?")
            issue_types = gr.CheckboxGroup(ISSUE_TYPE_CHOICES, label="Issue type(s)", value=[], visible=False)
            issue_details = gr.Textbox(label="Issue details (optional)", lines=4, visible=False)

            def toggle_issue(v):
                show = (v == "Yes")
                return gr.update(visible=show), gr.update(visible=show)

            report_issue.change(toggle_issue, inputs=[report_issue], outputs=[issue_types, issue_details])

    def search_dialogue(dialogue_text, query):
        # Always render the base HTML from current dialogue text
        def render(text, q):
            import html
            escaped = html.escape(text)
            if not q or not q.strip():
                # No search — plain styled box
                return (
                    f'<div style="height:600px;overflow-y:auto;padding:10px;'
                    f'background:#1f2937;color:#f3f4f6;font-size:13px;'
                    f'line-height:1.6;white-space:pre-wrap;border-radius:6px;'
                    f'border:1px solid #374151;">{escaped}</div>'
                ), "—"
            q_strip = q.strip()
            q_lower = q_strip.lower()
            count = escaped.lower().count(q_lower)
            if count == 0:
                return (
                    f'<div style="height:600px;overflow-y:auto;padding:10px;'
                    f'background:#1f2937;color:#f3f4f6;font-size:13px;'
                    f'line-height:1.6;white-space:pre-wrap;border-radius:6px;'
                    f'border:1px solid #374151;">{escaped}</div>'
                ), "0 matches"
            # Replace all matches with highlighted spans (case-preserving)
            result = []
            i = 0
            n = len(q_strip)
            src = escaped  # work on escaped text
            src_lower = src.lower()
            while i < len(src):
                if src_lower[i:i+n] == q_lower:
                    result.append(
                        f'<mark style="background:#facc15;color:#111;'
                        f'font-weight:bold;border-radius:3px;padding:1px 3px;">'
                        f'{src[i:i+n]}</mark>'
                    )
                    i += n
                else:
                    result.append(src[i])
                    i += 1
            highlighted = "".join(result)
            return (
                f'<div style="height:600px;overflow-y:auto;padding:10px;'
                f'background:#1f2937;color:#f3f4f6;font-size:13px;'
                f'line-height:1.6;white-space:pre-wrap;border-radius:6px;'
                f'border:1px solid #374151;">{highlighted}</div>'
            ), f"{count} match{'es' if count != 1 else ''}"

        return render(dialogue_text, query)
    
    dialogue_search.change(
        search_dialogue,
        inputs=[dialogue, dialogue_search],
        outputs=[dialogue_display, dialogue_search_status],
    )

    dialogue.change(render_dialogue_html, inputs=[dialogue], outputs=[dialogue_display])
    
    # -------------------------
    # Wiring: add/remove/clear DataFrames
    # -------------------------
    btn_labs_add.click(lambda rows: df_add_row(rows, LAB_COLS), inputs=[labs_df], outputs=[labs_df])
    btn_labs_remove.click(lambda rows: df_remove_row(rows, LAB_COLS), inputs=[labs_df], outputs=[labs_df])
    btn_labs_clear.click(lambda: df_clear_rows(LAB_COLS), inputs=[], outputs=[labs_df])

    # Imaging handlers
    btn_img_add.click(
        add_imaging_row,
        inputs=[imaging_df, imaging_contrasts_state],
        outputs=[imaging_df, imaging_contrasts_state] + imaging_contrast_radios
    )
    btn_img_remove.click(
        remove_imaging_row,
        inputs=[imaging_df, imaging_contrasts_state],
        outputs=[imaging_df, imaging_contrasts_state] + imaging_contrast_radios
    )
    btn_img_clear.click(
        clear_imaging,
        inputs=[imaging_df, imaging_contrasts_state],
        outputs=[imaging_df, imaging_contrasts_state] + imaging_contrast_radios
    )

    # Wire up contrast radio buttons
    for idx, radio in enumerate(imaging_contrast_radios):
        def make_handler(row_idx):
            def handler(contrasts, val):
                return update_imaging_contrast(contrasts, row_idx, val)
            return handler
        
        radio.change(
            make_handler(idx),
            inputs=[imaging_contrasts_state, radio],
            outputs=[imaging_contrasts_state]
        )

    btn_ncs_add.click(lambda rows: df_add_row(rows, NCS_COLS), inputs=[ncs_df], outputs=[ncs_df])
    btn_ncs_remove.click(lambda rows: df_remove_row(rows, NCS_COLS), inputs=[ncs_df], outputs=[ncs_df])
    btn_ncs_clear.click(lambda: df_clear_rows(NCS_COLS), inputs=[], outputs=[ncs_df])

    btn_tissue_add.click(lambda rows: df_add_row(rows, TISSUE_COLS), inputs=[tissue_df], outputs=[tissue_df])
    btn_tissue_remove.click(lambda rows: df_remove_row(rows, TISSUE_COLS), inputs=[tissue_df], outputs=[tissue_df])
    btn_tissue_clear.click(lambda: df_clear_rows(TISSUE_COLS), inputs=[], outputs=[tissue_df])

    btn_monitor_add.click(lambda rows: df_add_row(rows, MONITOR_COLS), inputs=[monitor_df], outputs=[monitor_df])
    btn_monitor_remove.click(lambda rows: df_remove_row(rows, MONITOR_COLS), inputs=[monitor_df], outputs=[monitor_df])
    btn_monitor_clear.click(lambda: df_clear_rows(MONITOR_COLS), inputs=[], outputs=[monitor_df])

    btn_prev_add.click(lambda rows: df_add_row(rows, PREVENTION_COLS), inputs=[prevention_df], outputs=[prevention_df])
    btn_prev_remove.click(lambda rows: df_remove_row(rows, PREVENTION_COLS), inputs=[prevention_df], outputs=[prevention_df])
    btn_prev_clear.click(lambda: df_clear_rows(PREVENTION_COLS), inputs=[], outputs=[prevention_df])

    btn_cons_add.click(lambda rows: df_add_row(rows, CONSERVATIVE_METHOD_COLS), inputs=[conservative_df], outputs=[conservative_df])
    btn_cons_remove.click(lambda rows: df_remove_row(rows, CONSERVATIVE_METHOD_COLS), inputs=[conservative_df], outputs=[conservative_df])
    btn_cons_clear.click(lambda: df_clear_rows(CONSERVATIVE_METHOD_COLS), inputs=[], outputs=[conservative_df])

    btn_life_add.click(lambda rows: df_add_row(rows, LIFESTYLE_COLS), inputs=[lifestyle_df], outputs=[lifestyle_df])
    btn_life_remove.click(lambda rows: df_remove_row(rows, LIFESTYLE_COLS), inputs=[lifestyle_df], outputs=[lifestyle_df])
    btn_life_clear.click(lambda: df_clear_rows(LIFESTYLE_COLS), inputs=[], outputs=[lifestyle_df])

    # Medication handlers
    btn_med_add.click(
        add_new_med,
        inputs=[meds_state],
        outputs=[
            meds_state,
            meds_display_df,
            med_editor,
            med_select_idx,
            med_name,
            med_treatment_status,
            med_frequency,
            med_dose,
            med_duration,
            med_discontinuation,
            med_route,
            med_dosage_form,
            med_timing,
            med_side_effects,
            med_drug_class,
            med_condition_treated,
            med_edit_idx
        ]
    )

    btn_med_edit.click(
        load_med_for_edit_by_number,
        inputs=[meds_state, med_select_idx],
        outputs=[
            med_editor,
            med_name,
            med_treatment_status,
            med_frequency,
            med_dose,
            med_duration,
            med_discontinuation,
            med_route,
            med_dosage_form,
            med_timing,
            med_side_effects,
            med_drug_class,
            med_condition_treated,
            med_edit_idx
        ]
    )

    btn_med_delete.click(
        delete_med,
        inputs=[meds_state, med_select_idx],
        outputs=[meds_state, meds_display_df]
    )

    btn_cancel_med.click(
        cancel_med_edit,
        outputs=[med_editor]
    )

    btn_save_med.click(
        save_med_edit,
        inputs=[
            meds_state,
            med_edit_idx,
            med_name,
            med_treatment_status,
            med_frequency,
            med_dose,
            med_duration,
            med_discontinuation,
            med_route,
            med_dosage_form,
            med_timing,
            med_side_effects,
            med_drug_class,
            med_condition_treated
        ],
        outputs=[meds_state, meds_display_df, med_editor]
    )

    btn_surg_add.click(lambda rows: df_add_row(rows, SURG_COLS), inputs=[surg_df], outputs=[surg_df])
    btn_surg_remove.click(lambda rows: df_remove_row(rows, SURG_COLS), inputs=[surg_df], outputs=[surg_df])
    btn_surg_clear.click(lambda: df_clear_rows(SURG_COLS), inputs=[], outputs=[surg_df])

    btn_fu_add.click(lambda rows: df_add_row(rows, FOLLOWUP_COLS), inputs=[followup_df], outputs=[followup_df])
    btn_fu_remove.click(lambda rows: df_remove_row(rows, FOLLOWUP_COLS), inputs=[followup_df], outputs=[followup_df])
    btn_fu_clear.click(lambda: df_clear_rows(FOLLOWUP_COLS), inputs=[], outputs=[followup_df])

    btn_ref_add.click(lambda rows: df_add_row(rows, REFERRAL_COLS), inputs=[referral_df], outputs=[referral_df])
    btn_ref_remove.click(lambda rows: df_remove_row(rows, REFERRAL_COLS), inputs=[referral_df], outputs=[referral_df])
    btn_ref_clear.click(lambda: df_clear_rows(REFERRAL_COLS), inputs=[], outputs=[referral_df])

    # Outputs list - CRITICAL: status MUST be at the very end after all imaging contrast radios
    all_outputs = [
        annotator_id,
        shard_pos_state,
        progress,
        dialogue,
        note,
        icd10,
        icd10_desc,
        debug_box,
        labs_df,
        imaging_df,
        imaging_contrasts_state,
        ncs_df,
        tissue_df,
        monitor_df,
        prevention_df,
        conservative_df,
        lifestyle_df,
        meds_state,
        meds_display_df,
        surg_df,
        complementary_therapies,
        external_equipment,
        followup_df,
        patient_education,
        when_to_seek_medical_care,
        referral_df,
        report_issue,
        issue_types,
        issue_details,
        # ---- error report (new) ----
        error_report_content,
        error_report_box,
    ] + imaging_contrast_radios + [jump_status, status]  # status MUST be last

    demo.load(lambda: empty_ui(), inputs=[], outputs=all_outputs)

    annotator_id.change(resume_case_guarded, inputs=[annotator_id], outputs=all_outputs)

    btn_prev.click(prev_case_guarded, inputs=[annotator_id, shard_pos_state], outputs=all_outputs)

    btn_next.click(next_case_guarded, inputs=[annotator_id, shard_pos_state], outputs=all_outputs)

    btn_save.click(
        save_and_advance_guarded,
        inputs=[
            annotator_id,
            shard_pos_state,
            icd10,
            icd10_desc,
            labs_df,
            imaging_df,
            imaging_contrasts_state,
            ncs_df,
            tissue_df,
            # tissue_sampling_ordered,
            monitor_df,
            prevention_df,
            conservative_df,
            lifestyle_df,
            meds_state,
            surg_df,
            complementary_therapies,
            external_equipment,
            followup_df,
            # follow_up_ordered,
            patient_education,
            when_to_seek_medical_care,
            referral_df,
            # referral_ordered,
            report_issue,
            issue_types,
            issue_details,
        ],
        outputs=all_outputs
    )

    btn_jump.click(
        jump_to_case_guarded,
        inputs=[annotator_id, jump_to_case],
        outputs=all_outputs
    )

demo.launch(share=True, css=CSS)