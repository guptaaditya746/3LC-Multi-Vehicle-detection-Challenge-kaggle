# Bug Report: 3LC Multi-Vehicle Detection Challenge

## Summary
Analysis of 7 Python scripts in the `scripts/` directory identified **5 bugs/issues** of varying severity.

---

## Critical Issues (Should Fix Immediately)

### 1. **predict.py - Encoded Extension Handling Bug** (Line 96-104)
**Severity:** MEDIUM  
**Location:** `image_lookup_keys()` function

**Problem:**
```python
for key in list(keys):
    for extension in IMAGE_EXTENSIONS:
        encoded_extension = f"_{extension.lstrip('.')}"
        if key.lower().endswith(encoded_extension):
            base = key[: -len(encoded_extension)]
            keys.add(base)
            keys.add(f"{base}{extension}")
```

The code searches for encoded extensions like `_jpg`, `_png`, etc. in filenames. However:
- IMAGE_EXTENSIONS = `(".jpg", ".jpeg", ".png", ".bmp", ".webp")` - all lowercase with dots
- `extension.lstrip('.')` creates: `jpg`, `jpeg`, `png`, `bmp`, `webp`
- Then it looks for `_jpg`, `_jpeg`, `_png`, `_bmp`, `_webp` in filenames
- This assumes filenames were encoded with underscores instead of dots, which is non-standard

**Fix:** Clarify the intent. If this is for special filename encodings, document it. If not, remove this block.

---

### 2. **predict.py - Missing Error Handling in Result Processing** (Line 226-230)
**Severity:** MEDIUM  
**Location:** `pipeline_memory()` function

**Problem:**
```python
for result, (image_id, _) in zip(results, chunk):
    predictions[image_id] = result_to_prediction_string(result)
```

If the number of results differs from chunk length (e.g., due to a failure in `model.predict()`), the zip will silently truncate, leaving some image_ids without predictions. This would cause key errors later.

**Fix:** Add assertion or length check:
```python
results_list = list(results)
if len(results_list) != len(chunk):
    raise RuntimeError(f"Expected {len(chunk)} results, got {len(results_list)}")
for result, (image_id, _) in zip(results_list, chunk):
    predictions[image_id] = result_to_prediction_string(result)
```

---

### 3. **summarize_yolo_dataset.py - Recursive Subdirectory Handling** (Line 40)
**Severity:** LOW-MEDIUM  
**Location:** `summarize()` function

**Problem:**
```python
label_files = sorted(labels_dir.rglob("*.txt"))
```

Uses `rglob()` which recursively finds all `.txt` files in subdirectories. However:
- The code assumes all files are in the labels_dir root
- If labels are organized as `labels/train/`, `labels/val/` subdirectories, the summary will mix them together
- No indication that subdirectories are handled

**Fix:** Either:
1. Document that subdirectories are supported and mixed together, OR
2. Use `labels_dir.glob("*.txt")` for non-recursive search

---

## Moderate Issues (Worth Investigating)

### 4. **predict.py - Incomplete Prediction Dictionary Assumption** (Line 311)
**Severity:** LOW  
**Location:** `main()` function, line 311

**Problem:**
```python
"prediction_string": predictions.get(image_id, "no box")
```

After the full prediction pipeline, every image_id should be in the predictions dictionary. However:
- The code defensively uses `.get(..., "no box")` 
- If an image prediction failed silently, this masks the error
- The code produces incorrect submissions without warning

**Fix:** Add validation:
```python
missing = set(image_ids) - set(predictions.keys())
if missing:
    print(f"WARNING: {len(missing)} image predictions missing!", file=sys.stderr)
    for img_id in list(missing)[:5]:
        print(f"  - {img_id}", file=sys.stderr)
```

---

### 5. **train.py - Redundant Batch Size Calculation** (Line 225)
**Severity:** LOW  
**Location:** `pipeline_memory()` function

**Problem:**
```python
batch_size = max(1, int(batch))  # Line 214
...
results = model.predict(
    ...
    batch=min(batch_size, len(chunk)),  # Line 225
    ...
)
```

The batch size passed to `model.predict()` is already `len(chunk)` (since chunks are created with `pairs[start : start + batch_size]`), making the `min()` operation redundant. While not incorrect, it suggests unclear intent.

**Fix:** Either:
1. Pass `batch=len(chunk)` directly, or
2. Add a comment explaining why min() is needed

---

## Informational Notes

### verify_setup.py - Status Report
✓ Well-structured  
✓ Good error messages  
✓ Python version check is correct (3.9-3.13)

### validate_submission.py - Status Report  
✓ Comprehensive validation  
✓ Clear error reporting  
✓ Good test coverage appearance

### register_tables.py - Status Report
✓ Clean implementation  
✓ Proper error handling  
✓ Good exception fallbacks

### config.py - Status Report
✓ Well-designed utility functions  
✓ Proper path resolution  
✓ YAML parsing is safe

### make_experiment_entry.py - Status Report
✓ Simple and correct  
✓ Proper CSV handling with newline parameter

### tlc_compat.py - Status Report
⚠️ Complex monkey-patching could be fragile across versions  
⚠️ Mutable default state in class attribute (_ua_detrac_compat_patched)  
✓ Otherwise functionally correct

---

## Recommendations

1. **Short-term:** Fix issues #1 and #2 in predict.py before training/submission
2. **Medium-term:** Add end-to-end integration tests for the full pipeline
3. **Long-term:** Consider adding type hints and more comprehensive logging
4. **Documentation:** Document the special filename encoding logic (issue #1) if intentional

---

## Test Cases to Add

```bash
# Test with image names containing underscores (issue #1)
python scripts/predict.py --config config/competition.yaml

# Test batch processing with small batch sizes (issue #2)
# Modify config to set predict.batch = 1

# Test with organized subdirectories (issue #3 context)
python scripts/summarize_yolo_dataset.py --labels data/labels
```

