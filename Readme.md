# Sail-CV

## Looking is measuring : embedded computer vision measurement of sails aerodynamic performance.

Understanding the interaction between wind flow and sails is essential for optimizing sailing performance.
Sailors traditionally rely on two qualitative visual cues —tell-tales and sail shape— but obtaining quantitative,
real-time measurements of these indicators remains challenging. Moreover, these visual cues are seldom recorded
in forms suitable for post-navigation analysis.
This work introduces an embedded computer-vision framework that quantitatively measures two key aerodynamic features: (1) boundary-layer behavior through continuous tell-tale state tracking, and (2) 3D sail
geometry via a photogrammetry-based reconstruction method. The two modules operate independently yet
share the same minimal hardware requirements, enabling practical, plug-and-play deployment across a broad
range of yachts.
The tell-tale tracking module —requiring only a single camera— uses a detection-plus-tracking pipeline. A
vision model is trained on a purpose-built dataset annotated with bounding boxes for attached, detached, and
leech tell-tales, as shown in Figure 1. A tracker then converts per-frame detections into time-series suitable for
aerodynamic interpretation. This machine-learning-based approach offers the robustness necessary to handle
variations in color, sail type, illumination, and object motion, showing promising behavior for reliable field use.

The 3D reconstruction module aims to recover accurate, metric point clouds of the sail surface from calibrated
stereo imagery. The method leverages two core components: (i) the ability of AI-based reconstruction models to
generate dense point correspondences between two viewpoints, and (ii) precise intrinsic and extrinsic calibration
of a general two-camera setup, enabling accurate triangulation and conversion of correspondences into 3D
coordinates. This approach eliminates the need for applied texture or detailed geometric priors on the sail.
The paper presents the details of the methods and the training of the tell-tale detector. Generic results taken
in real conditions on yachts are showed together with a more in-depth analysis of tell-tales on the rigid wings of
a model wind-powered vessel, comparing with another tell-tales detection method and pressure measurements.