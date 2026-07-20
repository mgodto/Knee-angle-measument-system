Full-Length Leg X-ray Automated Measurement v0.3.0 (Research Use)
================================================================

Bundled AI models
  Bone (no knee implant)
    Version: 20260720-bone-final-v1
    SHA-256: 24481410c3fd2ce2222eed422f4d519f72da95ba232570ee31a4827d45201cfd

  TKA (knee implant)
    Version: 20260720-tka-final-v1
    SHA-256: 87027887ec091068a9b91b01a881092400fed58eb8d3eeaaeddb10e8be398e5f

  Mixed (Bone/TKA; fallback when type is unknown)
    Version: 20260720-bone-tka-mixed-final-v1
    SHA-256: f0cfa67f34691f3d81da0e10f0d6ff753dcf71f5aafd278ddb6bf146efc6ba45

Model selection
  Auto uses explicit Bone/TKA or implant/no-implant tokens in the image file
  or folder name. It does not diagnose an implant from image pixels. Unknown
  or conflicting names use Mixed. You may manually select Bone, TKA, or Mixed;
  a manual selection takes priority. Confirm the active model and version in
  the application before using each result.

Starting the Windows application
  1. Use Windows 10 or 11 x64.
  2. Right-click the ZIP and select "Extract All" before starting the app.
  3. Launch KneeXrayMeasurement.exe.
  4. Keep KneeXrayMeasurement.exe and the _internal folder together.

  This research build is not code signed. If your organization's security
  policy blocks it, ask your administrator before trying again.

Important
  - Supports single-leg full-length JPG, PNG, BMP, and TIFF images.
  - DICOM and uncropped bilateral images are not supported.
  - A doctor must review all 8 points, both joint lines, and every angle.
  - Validation used case-level 5-fold validation within the same dataset; an
    independent external-site validation has not been performed.
  - Do not use this software as the sole basis for diagnosis or treatment.
  - No patient X-rays, annotations, or training code are included.
  - Images and results are processed locally and are not uploaded.
