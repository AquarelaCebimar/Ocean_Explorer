# Ocean_Explorer V1.beta

Ocean Optics Explorer - User Manual and Documentation

Author: Ana Paula Piazza Forgiarini (anapiazzaf@gmail.com)
Modified in April 2026

# CONFIDENTIAL: RESEARCH EMBARGO NOTICE

**Project:** Ocean_Explorer V1.beta  
**Lead Researcher:** Ana Paula Piazza Forgiarini (anapiazzaf@gmail.com)  
**Institution:** Aquarela Lab - CEBIMar / University of São Paulo (USP)

### [READ BEFORE PROCEEDING]

This repository contains **unpublished research code**. Access is granted exclusively to authorized members of the current working group for collaborative development and testing purposes.

By accessing this codebase, you agree to the following terms:
1. **Confidentiality:** This code, its logic, and any generated outputs are under strict academic embargo. They must not be shared, duplicated, or distributed outside of this authorized group.
2. **Intellectual Property:** All rights to the methodology and software architecture belong to the lead researcher and the affiliating institution.
3. **Publication:** Use of this software in any publication, presentation, or report without the express written consent and co-authorship agreement of Ana Paula Piazza Forgiarini is strictly prohibited.
4. **Attribution:** Any future public release will be managed by the lead researcher upon the completion of the official peer-reviewed publication.


# How to cite:
This software is an integral part of the ongoing research of Ana Paula Piazza Forgiarini at Aquarela Lab CEBIMar/USP.
All rights reserved. This code is shared for collaborative work within the research group. Please contact the author for formal permission and specific citation guidelines before using this software, its underlying logic, or generated outputs in any presentation, report, or publication.

---

# Overview of processing phases

The app follows a four-phase workflow to transform raw radiometric and physical data into Level-3 (L3) products.

Phase 1: Data ingestion and Quality Control (QC)
-	Hyperspectral data from a profiler (ed, Lu, Es, and β) are loaded alongside physical parameters (Temperature, Conductivity, Pressure). 

-	Simultaneously, data from and External CTD/Probe (e.g., Castaway, RBR, JFE, C3) is imported. The external reference is crucial to account for profiler drift and to validate depth readings. The app uses R², RMSE, bias, and MAE metrics to compare sensors and select the most representative profile from multiple casts.

Phase 2: Filtering and Optical Layer Selection
-	Surface Irradiance (Es) Filtering: Shipboard sensors are sensitive to wave-induced motion. The app filters data based on Tilt and extreme outliers to prevent erroneous Rrs calculations.
-	Homogeneous layer identification: To calculate robust attenuation coefficients (Kd), a homogeneous layer must be selected. The app provides two key proxies for this:
1.	Brunt-Väisälä Frequency (N²): derived from TEOS-10 density gradients, high N² values indicate strong stratification (pycnoclines) where optical properties might change abruptly.
2.	Backscattering Proxy (β): Variations in β indicate changes in particle concentration, which directly affects light attenuation, and can or cannot be linked to density changes, hence why we use both proxies together to evaluate the optical depth selection.
Also, Ln(Eu) and Ln(Ed) are provided to help in this step.

	Phase 3: Mixture and Bio-Optical Analysis
-	T-S Diagram and Mixing: Mixing fractions are calculated using barycentric coordinates within a T-S mixing triangle, identifying the dominance of regional water masses.
-	Secchi-Kd Integration: Empirical Secchi Depth (Zsd) readings are linked to calculated Kd values to build site-specific models.
-	Empirical bio-optical model: The empirical KdPAR and 490 models can be derived.
-	Spectral Convolution: Hyperspectral data is convolved using satellite spectral response functions (SRFs). Following Burggraaff (2020), convolution is performed independently on Lw and Es, avoiding biases.

	Phase 4: Data merging and L3 Export
-	Final results are merged into master databases, organizing variables by depth (Radiometry, Chla, CDOM, and Water Mass percentages)


	Operation Manual

1.	Initial Setup

1.	Open your Anaconda Prompt (or Terminal) and navigate to the app folder using the cd command.
2.	Ensure the folder contains the Ocean_Explorer.py script and the srf_data/ subfolder (containing SRF CSV files)
3.	Run the command streamlit run Ocean_Explorer.py. The app will open in your default web browser as a local host.

2. Tab by tab workflow
After uploading the requested profiler files, click process files and the following tabs will appear.

1.	Tab 1 (Depth Correction): Use this if the profiler was tared incorrectly in the water. Align the N² peaks of the profiler with the external CTD reference.
2.	Tab 2 (CTD Comparison): Load ancillary data (Chl, CDOM, PAR, Secchi). Compare Profiler vs External probe CTD to validate representativeness (using the metrics in the table created)
3.	Tab 3 (Es Tilt): Apply tilt and stability filters to clean the Es spectrum.
4.	Tab 4 (Core Optics): Manually define the depth range for Kd and Klu calculation to propagate profiles to the surface (using N² and β proxies)
5.	Tab 5 (T-S diagram): define water mass end-members and calculate mixing percentages throughout the water column. Works with both the profiler and the external CTD.
6.	Tab 6 (bio-optical model): Generate site-specific regressions between Kd (PAR and 490) to estimate properties for external probes. Alternatively, use the embedded Morel model.
7.	Tab 7 (Secchi/Kd): Build empirical relationships between Zsd and propagated optical coefficients.
8.	Tab 8 (Spectral Convolution): Simulate satellite bands (e.g., OLCI, OLI, MSI)from the hyperspectral Rrs
9.	Tab 9 (Master data): Accumulate the results
-	Click “Snapshot current station to masters” after analyzing each station
-	Click “Export full campaign ZIP” to download the cumulative database
-	Click “Build L3 Station Package” for a complete folder containing physics, optics, and high-res PNG plots for a specific station
10.	Tab 10 (methods): A tab containing a brief explanation of the methods used in the calculations

3. Understanding Exported Results

L3 Station Package

-	Summary_report.txt: Metadata, main AOPs, and AVW
-	1_Profiler_Jimmy: Physical data (TEOS-10) and full vertical hyperspectral profiles
-	2_External_Probe: Raw data from the validation probe (chla, CDOM, Secchi, PAR, etc.)
-	3_Derived Optical Products: Attenuation metrics, propagated Rrs, and convolved satellite bands
-	4_Validation_and_Plots: all PNG figures generated during the session (T-S diagram, log-profiles, etc.)

Master database files (CSVs)

1.	Master Vertical Radiometry: All Ed/Lu profiles by depth.
2.	Master Vertical Ancillary: PAR, Chl, CDOM, and Water masses % by depth
3.	Summary_Hyper_Rrs: Propagated hyperspectral Rrs 
4.	Summary_convolved_Rrs: Propagated multispectral Rrs
5.	Summary_Secchi_Kd: Cumulative cross-validation table for Secchi and Ks
6.	Master_bio-optical_Model: Empirical data used to train the bio-optical models

Note: For the master database to remain consistent, always load your "Previous Master" files at the start of Tab 9 when processing new campaign days. This ensures data is appended correctly.


Expected Input Structure
1.  The application ingests pre-calibrated Level-2S CSV files. Ensure your directory contains:
2.  In-water profiler files containing keywords: temp, cond, ed, lu
3.  Optional surface reference: es
4.  Optional backscattering proxy: beta

---

## Installation & Setup

### Option A: Using Conda (Recommended)

1. **Clone the repository:**
   git clone https://github.com/AquarelaCebimar/Ocean_Explorer.git
   cd Ocean_Explorer
   
2. Create and activate the Conda environment:
   conda env create -f environment.yml
   conda activate Ocean_Optics_Explorer

3. Launch the application:
   streamlit run Ocean_Explorer.py

### Option B: Using Standard Python (pip)
1.  **Create a virtual environment:**

   python -m venv venv

# On Windows:
   venv\Scripts\activate
# On Linux/macOS:
   source venv/bin/activate

2. **Install dependencies:**
   pip install -r requirements.txt

3. **Launch the application:**
   streamlit run Ocean_Explorer.py





