# Labeling Protocol

This protocol is meant to keep label fixes consistent across iterations and make every revision easy to justify later.

## Principles

- Fix labels only when the image evidence is clear.
- Prefer consistent annotation rules over ad hoc exceptions.
- Record every meaningful change so the training set can be audited.
- When uncertain, flag the case for review instead of guessing.

## When To Add Missing Boxes

Add a box when all of the following are true:

- the object is visibly on-road and belongs to one of the four allowed classes
- enough of the vehicle is visible to support a stable box
- the object is not already represented by another box

Do not add a box when:

- the object is too small to classify with confidence
- the visible evidence is too weak because of blur, glare, or severe occlusion
- the object is off-road and clearly outside the intended labeling scope

## When To Remove False Boxes

Remove a box when:

- the box covers background or a non-vehicle object
- the class is unsupported by the competition
- the box duplicates another annotation for the same object
- the box is attached to a reflection, poster, shadow, or artifact rather than a real vehicle

## When To Adjust Boxes

Adjust a box when:

- the target vehicle is correct but the box is too loose
- the box cuts off a meaningful part of the vehicle
- the annotation drifts onto a neighboring vehicle
- the object center is clearly misplaced

Preferred box behavior:

- include the visible extent of the vehicle
- keep overlap with neighboring vehicles as low as possible
- stay consistent across similar scenes and camera angles

## Occluded Vehicles

- Label partially occluded vehicles when the visible area still supports a reliable class and approximate extent.
- Use the visible object footprint as consistently as possible across similar cases.
- Skip extremely occluded instances when the annotation would be mostly guesswork.

## Tiny Vehicles

- Keep tiny vehicles only if they are still recognizable as one of the four classes.
- If a tiny object is visible but not classifiable with confidence, do not force a label.
- Use the same thresholding logic across the whole dataset to avoid inconsistency.

## Ambiguous Class Labels

When class identity is unclear:

- use the most defensible class based on shape and context
- prioritize consistency with the rest of the dataset
- record recurring ambiguity patterns in the data card

Common ambiguity examples to review carefully:

- car versus van
- truck versus bus at long distance
- cargo vehicles with partial side views

## Documentation Rules

For every label revision, capture:

- revision ID
- date
- table name and split
- who made the change
- why the change was needed
- how many images and boxes were touched
- any downstream impact on training or validation

## Label-Change Log Template

| Revision ID | Date | Split | Image ID | Change type | Before | After | Reason | Reviewer |
|---|---|---|---|---|---|---|---|---|
| TODO | TODO | train | TODO | adjust box | TODO | TODO | bad IoU | TODO |
