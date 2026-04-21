# Final Project Proposal: Post-Fire Structural Damage Assessment Using Satellite Imagery and Deep Learning on the 2025 Los Angeles Palisades and Eaton Fires

**Tess Vu, Luciano Lu, Ming Cao**

## Problem Definition & Use Case

The Wildland-Urban Interface (WUI), as defined by the US Fire Administration, is the zone where human development meets undeveloped wildland fuels. The January 2025 Los Angeles wildfires destroyed thousands of structures across both the Pacific Palisades and Altadena communities, burning through steep chaparral-dense canyons directly into densely populated residential areas. The problem this project solves is the rapid spatial prediction of structural damage following a WUI fire event, using freely available Sentinel-2 satellite imagery and topographic context as predictive features.

The target users are civic agencies such as the Los Angeles County Department of Regional Planning, local utility providers, and environmental resilience initiatives. These stakeholders currently wait for labor-intensive CAL FIRE Damage Inspection (DINS) field teams to physically visit each structure, a process that takes weeks and places inspectors in hazardous conditions. The output is a binary segmentation mask at 20-meter resolution classifying each pixel within the fire perimeter as damaged or intact. This mask provides immediate, actionable intelligence to inform post-disaster debris removal prioritization, emergency resource allocation, and future WUI buffer planning before complete field assessments are available.

## Technical Justification

The project frames structural damage prediction as a semantic segmentation task where the model learns to associate multi-temporal spectral signatures, topographic context, and building footprint presence with field-verified damage outcomes. A positive prediction means the model has identified a pixel where the combination of spectral change between pre-fire and post-fire imagery, topographic position, and infrastructure presence correlates with CAL FIRE DINS field assessments of structural damage. Semantic segmentation is the appropriate task type because municipal planners need a spatially explicit map of predicted damage to overlay on property parcels; a tile-level classifier would provide no actionable spatial intelligence.

Three failure modes were anticipated during the design phase, and additional failure modes emerged during implementation.

The first anticipated failure mode is topographic shadowing. The fires occurred in January when the sun angle is low in the northern hemisphere. North-facing slopes in the Santa Monica Mountains and San Gabriel foothills receive deep shadow during Sentinel-2 overpasses, returning near-zero reflectance that mimics severe burn signatures. DEM-derived aspect is included as an explicit feature channel so the model can learn to distinguish shadow artifacts from genuine burn signatures.

The second anticipated failure mode is severe class imbalance. Damaged structures represent roughly 5–6% of pixels within the fire perimeter, creating a strong bias toward predicting the majority intact class. A combined loss function pairing Dice loss with Binary Focal Cross-Entropy (alpha=0.80, gamma=3.0) addresses this by heavily penalizing false negatives on the minority damaged class.

The third anticipated failure mode is target leakage. An earlier iteration of the ground truth used a Boolean union of Otsu-thresholded dNBR and DINS damage points, which introduced leakage because dNBR is also an input feature. The ground truth was revised to use DINS field assessments exclusively, reducing the Pearson correlation between the building footprint feature and the damage label from approximately 0.90 to 0.19.

A fourth failure mode, spatial leakage from patch-block misalignment, emerged during implementation. When the patch extraction size did not match the spatial block size used for cross-validation, patches straddled block boundaries and leaked information between folds. Aligning both to 64×64 pixels resolved this.

## Methodological Precedent

- Ngoua Ndong Avele, J. B., & Goryainov, V. S. (2025). Wildfire Damage Assessment over Eaton Canyon, California, Using Radar and Multispectral Datasets from Sentinel Satellites and Machine Learning Methods. _Environmental and Earth Sciences Proceedings_, 36(1), 6. <https://doi.org/10.3390/eesp2025036006>
  - This study assessed wildfire damage over the same 2025 Los Angeles fire complex using Sentinel-2 NIR (B08) and SWIR (B12) bands with a Random Forest classifier. The authors compared dNBR, RBR, and RdNBR, finding that dNBR alone achieved 78% accuracy while a fusion of all three indices reached 99%. Their finding that canyon topography created natural firebreaks for 56.76% of the study area reinforced the inclusion of DEM-derived aspect as a feature channel. For this project, the study validates using Sentinel-2 dNBR as a core predictive feature and motivates including raw NIR and SWIR bands alongside dNBR so the model can learn non-linear spectral relationships beyond what a single index captures.

- Miguel M. Pinto, Renata Libonati, Ricardo M. Trigo, Isabel F. Trigo, Carlos C. DaCamara, A deep learning approach for mapping and dating burned areas using temporal sequences of satellite images, _ISPRS Journal of Photogrammetry and Remote Sensing_, Volume 160, 2020, Pages 260-274, ISSN 0924-2716, <https://doi.org/10.1016/j.isprsjprs.2019.12.014>
  - This work introduced BA-Net, a U-Net-based architecture with 3D convolutions and LSTM layers for burned area mapping from daily multi-spectral satellite sequences across five global regions. BA-Net achieved an overall Dice score of 0.678 against medium-resolution reference maps, competitive with established MCD64A1C6 (0.687) and FireCCI51 (0.656) products despite using lower-resolution VIIRS imagery. The encoder-decoder architecture with skip connections confirmed that U-Net variants are well-suited for pixel-level burn classification because skip connections preserve fine-grained spatial boundaries. The paper also demonstrated that providing raw spectral bands alongside derived indices allows the network to learn patterns that single-index thresholds miss, motivating the multi-temporal NIR and SWIR channels in the feature tensor.

- Timothy Mayer, Ate Poortinga, Biplov Bhandari, Andrea P. Nicolau, Kel Markert, Nyein Soe Thwal, Amanda Markert, Arjen Haag, John Kilbride, Farrukh Chishtie, Amit Wadhwa, Nicholas Clinton, David Saah, Deep learning approach for Sentinel-1 surface water mapping leveraging Google Earth Engine, _ISPRS Open Journal of Photogrammetry and Remote Sensing_, Volume 2, 2021, 100005, ISSN 2667-3932, <https://doi.org/10.1016/j.ophoto.2021.100005>
  - Although focused on surface water rather than fire damage, this study provided critical guidance on U-Net training methodology for binary segmentation with satellite imagery. The authors systematically compared BCE, Dice, and combined BCE-Dice loss functions across 12 model configurations, finding that the combined BCE-Dice loss produced the highest F1-score (0.922) in independent validation. This directly informed the combined Dice plus Focal loss function used in this project. The study also validated batch normalization, dropout, and He Normal initialization as effective regularization strategies for U-Nets trained on limited satellite imagery.

## Data Plan

The project constructs a 7-channel feature tensor fusing multi-temporal optical, topographic, and civic vector data for both the Palisades Fire (Santa Monica Mountains) and Eaton Fire (San Gabriel foothills). Processing both fires roughly triples the positive-class pixel count compared to the Palisades fire alone, providing substantially more training signal for the minority damage class.

**Sentinel-2 L2A imagery** (ESA, via Element 84 STAC API): Pre-fire (Dec 2024 – Jan 6, 2025) and post-fire (Feb 2025) NIR (Band 8A, 865nm) and SWIR (Band 12, 2190nm) at 20m resolution. Scenes are filtered below 30% cloud cover, grouped by acquisition date, and mosaicked to handle tile boundaries. Cloud masking is applied using the Scene Classification Layer (SCL). dNBR is computed as the difference of pre-fire and post-fire NBR. Raw pre-fire and post-fire NIR and SWIR bands are percentile-normalized (2nd–98th) and included as separate channels.

**USGS 3DEP DEM** (via Planetary Computer STAC API): Resampled to 20m and reprojected to EPSG:32611. Aspect is derived and normalized to 0–1.

**Microsoft Open Buildings footprints** (LA County): Rasterized to the Sentinel-2 grid as a binary mask.

**CAL FIRE DINS field assessments**: Point locations of inspected structures, filtered to exclude "No Damage," buffered by 15m to account for geolocation uncertainty at 20m pixel resolution, and rasterized as the binary ground truth target. DINS assessments are used exclusively for ground truth; no spectral thresholding contributes to the label.

**NIFC FIRIS fire perimeters**: Define the AOI mask for each fire.

**Feature tensor channels (7):** dNBR, DEM Aspect (normalized), Building Footprints (binary), Pre-Fire NIR, Pre-Fire SWIR, Post-Fire NIR, Post-Fire SWIR. All layers are reprojected to EPSG:32611 at 20m resolution.

Sentinel-2 was selected because it is the only freely accessible sensor providing SWIR coverage over the study area. The 20m resolution is coarser than commercial alternatives like WorldView-3, but SWIR is essential for burn severity assessment because it is directly sensitive to vegetation moisture content. Higher-resolution alternatives like Planet SuperDove lack SWIR bands entirely.

## Modeling Approach

The baseline model is a pixel-wise Random Forest classifier from scikit-learn with class-weight balancing. Each 64×64 patch is flattened so every pixel becomes an independent 7-feature sample. No hyperparameter tuning is performed beyond the default balanced configuration, keeping the baseline deliberately simple. The baseline establishes a performance floor: any improvement from the U-Net must come from its ability to learn spatial patterns across neighboring pixels.

The primary model is a U-Net semantic segmentation architecture with a three-stage encoder (32, 64, 128 filters), a 256-filter bottleneck, and a symmetric decoder with skip connections. Each convolutional block applies two 3×3 convolutions with batch normalization, ReLU activation, standard dropout, and SpatialDropout2D. Dropout rates increase with depth (0.1 to 0.3). The output head is a single-channel sigmoid activation cast to float32 under mixed-precision training. The loss function equally weights Dice loss and Binary Focal Cross-Entropy (alpha=0.80, gamma=3.0). Training uses Adam with learning rate scaled linearly with batch size, early stopping (patience=10), and ReduceLROnPlateau (patience=5, factor=0.5). Data augmentation applies random horizontal flips, vertical flips, and 90-degree rotations, all geometrically valid for nadir satellite imagery.

## Evaluation Strategy

The model is evaluated using F1-Score (Dice Coefficient) and Intersection over Union (IoU) on the positive damaged class only. These metrics ignore true negatives, which would inflate accuracy-based metrics given the severe class imbalance (~5–6% positive prevalence). A model predicting "undamaged" everywhere would exceed 94% accuracy but score zero on both F1 and IoU.

The data is split using 5-fold spatial block cross-validation. Contiguous 64×64-pixel blocks are assigned to folds as atomic units, and blocks containing damaged pixels are distributed via stratified assignment to ensure each fold contains a representative proportion of the minority class. Patch size is aligned to block size (both 64×64) to prevent spatial leakage from patches straddling fold boundaries.

An F1-score exceeding 0.50 on held-out test folds indicates that remote sensing features carry meaningful predictive signal for structural damage. Scores approaching 0.70 or above would suggest the model could serve as a practical rapid-assessment tool for civic agencies awaiting complete field inspections. The primary limitation is small dataset size: even with both fires combined, the training set contains a limited number of 64×64 patches with positive-class pixels, producing high fold-to-fold variance that makes architecture-level improvements difficult to distinguish from random seed effects.