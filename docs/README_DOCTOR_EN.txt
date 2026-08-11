INTERNAL RESEARCH CANDIDATE - NOT FOR CLINICAL USE
NOT FORMALLY PROMOTED. FOR INTERNAL RESEARCH AND EVALUATION ONLY.

Full-Length Leg X-ray Automated Measurement v0.6.0
==================================================

Bundled AI models
  Bone (no knee implant)
    Version: 20260811-single-leg-v4-curated-tailqa-9e9a1de4fe98-bone-final-v1
    SHA-256: 36e8fee67c7c6bad8071a5a7ff8dbc713d76e28482c2a26798abd15ba3334862

  TKA (knee implant)
    Version: 20260811-single-leg-v4-curated-tailqa-9e9a1de4fe98-tka-final-v1
    SHA-256: 23a416f8c376156b3fa323298d3e45ae00060d619e0215754a46f2a2254a1669

  Mixed (Bone/TKA; fallback when type is unknown)
    Version: 20260811-single-leg-v4-curated-tailqa-9e9a1de4fe98-bone-tka-mixed-final-v1
    SHA-256: 5e2f5087a433aa6bc9c58d792e132d88f1098952df446512375818a779d1254e

Model selection
  Auto uses explicit Bone/TKA or implant/no-implant tokens in the image file
  or folder name. It does not diagnose an implant from image pixels. Unknown
  or conflicting names use Mixed. You may manually select Bone, TKA, or Mixed;
  a manual selection takes priority. Confirm the active model and version in
  the application before using each result.

Input scope and bilateral images
  Every newly opened image starts in the ordinary single-leg workflow. Opening
  an image does not automatically show a divider or ROI controls. Only when the
  source contains both legs, select Crop bilateral image in the top toolbar.

  Single-leg image:
    Confirm the patient's anatomical L or R. Analysis then starts automatically.

  Bilateral image:
    1. Select Crop bilateral image. Any result produced in single-leg mode is
       cleared, and the divider plus ROI controls become visible.
    2. Confirm the patient's anatomical L or R from the examination record.
    3. Separately confirm whether that target leg is on the left or right side
       of the screen. The app suggests the usual radiographic display position,
       but a doctor must verify the visible leg. Use Swap target, click the leg
       in the preview, or move the divider when needed.
    4. The ROI retains an 8%-of-image-width margin across the divider. Confirm
       that the blue ROI includes the complete target leg and excludes the
       contralateral leg.
    5. Select Confirm ROI and analyze. Inference and export remain disabled
       until this confirmation is complete.
    6. Select Return to single-leg to cancel bilateral crop mode.

  Changing anatomical side, screen side, divider, bilateral crop mode, or AI model clears
  the previous result and requires ROI confirmation again. Export is also blocked
  if a manually moved landmark lies outside the confirmed ROI.

Saving results and provenance
  Only the confirmed ROI is sent to the model for a bilateral image. Display,
  editing, and export coordinates are mapped back to the complete source image.
  The measurement JSON stores the source filename, dimensions, SHA-256, input
  scope, application version/release channel, model information, and
  manual-edit history. For a bilateral image,
  analysis.inference_roi additionally stores x0/y0/x1/y1, width, height,
  source-image coordinate space, selection method, and confirmed=true. This
  provenance lets the training importer reuse the same approved target-leg crop.

  Bilateral filenames automatically include the anatomical side. For example,
  the L result from exam.jpg is exam_L_measurement.json plus
  exam_L_measurement.png, and the R result is exam_R_measurement.json plus
  exam_R_measurement.png. You can save L, change the anatomical side to R,
  confirm the opposite ROI, and save R from the same bilateral image without
  overwriting the L files. Single-leg output retains exam_measurement.json and
  exam_measurement.png.

Starting the Windows application
  1. Use Windows 10 or 11 x64.
  2. Right-click the ZIP and select "Extract All" before starting the app.
  3. Launch KneeXrayMeasurement.exe.
  4. Keep KneeXrayMeasurement.exe and the _internal folder together.

  This research build is not code signed. If your organization's security
  policy blocks it, ask your administrator before trying again.

If a previous external model preference remains
  Open the AI model menu and select Return to Auto to restore the built-in
  models, then select one of the three model modes listed above.

Important
  - This package is an internal Research Candidate. It is not an approved
    clinical release and must not be used for clinical diagnosis or treatment.
  - Internal 5-fold OOF evaluation can contain large individual errors, and
    automatic GUI warnings do not detect every such error. Review every point,
    joint line, and angle against the source image.
  - Supports single-leg or bilateral full-length JPG, PNG, BMP, and TIFF images.
  - DICOM is not supported.
  - For an image containing both legs, always select Crop bilateral image and
    confirm the target-leg ROI before using the result.
  - The app does not infer single-leg versus bilateral scope from image geometry.
    A doctor activates bilateral crop mode only when it is needed.
  - A doctor must review all 8 points, both joint lines, and every angle.
  - Validation used case-level 5-fold validation within the same dataset; an
    independent external-site validation has not been performed.
  - Do not use this software as the sole basis for diagnosis or treatment.
  - No patient X-rays, annotations, or training code are included.
  - Images and results are processed locally and are not uploaded.
