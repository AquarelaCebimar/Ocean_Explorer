
# -*- coding: utf-8 -*-
"""
# Ocean_Explorer V1.beta

Author: Ana Paula Piazza Forgiarini (anapiazzaf@gmail.com)
Created on Fri Oct 10 10:46:23 2025
Modified in September 2026

# How to cite:
This software is an integral part of the ongoing research of Ana Paula Piazza Forgiarini at Aquarela Lab CEBIMar/USP.
All rights reserved. This code is shared for collaborative work within the research group. Please contact the author for formal permission and 
specific citation guidelines before using this software, its underlying logic, or generated outputs in any presentation, report, or publication.

"""

# ----------------------------- REQUIRED PACKAGES -----------------------------
# Import section for all Python libraries required for the application to function.

import streamlit as st  # Imports the Streamlit library to create the interactive web interface.
import pandas as pd  # Imports Pandas for data manipulation and analysis, especially DataFrames.
import numpy as np  # Imports NumPy for efficient numerical operations, especially with arrays.
import matplotlib.pyplot as plt  # Imports Matplotlib for creating static plots.
from matplotlib.patches import Polygon  # Imports the Polygon class to draw the mixing triangle on the T-S diagram.
import gsw  # Imports the Gibbs SeaWater (GSW) library for precise seawater property calculations (TEOS-10).
import seaborn as sns  # Imports Seaborn to create more aesthetically pleasing color palettes for the plots.
import io  # Imports the io module to handle in-memory byte buffers, used for creating zip files.
import zipfile  # Imports the zipfile module to dynamically create .zip archives.
import os  # Used to discover local spectral-response files.
import re  # Used to normalize supported input headers and filenames.
import unicodedata  # Used to make accented headers portable across input formats.
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # Imports regression metrics to compare CTD data.
from matplotlib.ticker import ScalarFormatter # Imports ScalarFormatter for scientific notation on axes.
from scipy.stats import linregress, t as student_t # Regression tools and confidence intervals.
import plotly.graph_objects as go  # Imports Plotly for interactive plots.
from scipy.integrate import simpson # Imports Simpson integral for spectral convolution


# ----------------------------- STYLE AND CONFIGURATIONS -----------------------------
# Section to define the visual appearance of the plots and the Streamlit page.

st.set_page_config(layout="wide")

# Dictionary centralizing color definitions to create a consistent visual theme (dark mode).

STYLE_CONFIG = {
    'facecolor':'#0E1117',
    'ax_facecolor':'#2C3445',
    'textcolor':'#FAFAFA',
    'gridcolor':'#5A5A5A',
    'spinecolor':'#8A8A8A',
    'n2_line_color':'#00FF00',
    'beta_line_color': '#FF8C00'
}

# Update Matplotlib's global parameters to match the dark mode theme.

plt.rcParams.update({
    'figure.facecolor': STYLE_CONFIG['facecolor'],
    'axes.facecolor': STYLE_CONFIG['ax_facecolor'],
    'axes.edgecolor': STYLE_CONFIG['spinecolor'],
    'text.color': STYLE_CONFIG['textcolor'],
    'axes.labelcolor': STYLE_CONFIG['textcolor'],
    'xtick.color': STYLE_CONFIG['textcolor'],
    'ytick.color': STYLE_CONFIG['textcolor'],
    'grid.color': STYLE_CONFIG['gridcolor']
})

# Streamlit page configuration (must be called once at the top of the script).

st.markdown(
    """
<style>
/* 1. Força todos os wrappers a permitirem scroll */
div[data-testid="stTabs"],
div[data-testid="stTabs"] > div,
div[data-testid="stTabs"] > div > div {
    overflow: visible !important;
}

/* 2. Container real das abas - Estilização da Fita */
div[data-testid="stTabs"] [role="tablist"] {
    display: flex !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    white-space: nowrap !important;
    gap: 10px !important;
    padding: 7px 14px 9px !important;
    scrollbar-width: auto !important;
    background: linear-gradient(90deg, rgba(167, 139, 250, 0.82) 0%, rgba(129, 140, 248, 0.82) 12%, rgba(96, 165, 250, 0.82) 25%, rgba(34, 211, 238, 0.82) 38%, rgba(52, 211, 153, 0.82) 52%, rgba(163, 230, 53, 0.82) 64%, rgba(250, 204, 21, 0.82) 76%, rgba(251, 146, 60, 0.82) 88%, rgba(255, 75, 75, 0.82) 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
    border-radius: 10px !important;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.22) !important;
}

/* 3. Botões das abas */
div[data-testid="stTabs"] button[role="tab"] {
    flex: 0 0 auto !important;
    white-space: nowrap !important;
    border-radius: 6px !important;
    margin: 0 !important;
    padding: 0.42rem 0.74rem !important;
    border: 1px solid transparent !important;
    background-color: transparent !important;
    transition: background-color 150ms ease, transform 150ms ease !important;
}

div[data-testid="stTabs"] button[role="tab"]:hover {
    background-color: rgba(255, 255, 255, 0.18) !important;
    transform: translateY(-1px) !important;
}

div[data-testid="stTabs"] button[role="tab"] p,
div[data-testid="stTabs"] button[role="tab"] div,
div[data-testid="stTabs"] button[role="tab"] span {
    font-size: 14px !important;
    font-weight: 600 !important;
}

/* 4. Scrollbar gradiente */
div[data-testid="stTabs"] [role="tablist"]::-webkit-scrollbar {
    height: 8px;
}
div[data-testid="stTabs"] [role="tablist"]::-webkit-scrollbar-thumb {
    background: linear-gradient(90deg, #A78BFA, #FF4B4B);
    border-radius: 5px;
}

/* 5. PALETA ESPECTRAL COM SOBREPOSIÇÃO FORÇADA DE TODAS AS TAGS */
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(1) p,
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(1) div,
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(1) span { color: #101828 !important; }

div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(2) p,
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(2) div,
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(2) span { color: #101828 !important; }

div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(3) p,
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(3) div,
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(3) span { color: #101828 !important; }

div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(4) p,
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(4) div,
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(4) span { color: #101828 !important; }

div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(5) p,
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(5) div,
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(5) span { color: #101828 !important; }

div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(6) p,
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(6) div,
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(6) span { color: #101828 !important; }

div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(7) p,
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(7) div,
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(7) span { color: #101828 !important; }

div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(8) p,
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(8) div,
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(8) span { color: #101828 !important; }

div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(9) p,
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(9) div,
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(9) span { color: #101828 !important; }

div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(10) p,
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(10) div,
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(10) span { color: #101828 !important; }

/* Force dark tab labels even when Streamlit applies an internal text-fill style. */
div[data-testid="stTabs"] [role="tablist"] > button[role="tab"]:nth-of-type(n),
div[data-testid="stTabs"] [role="tablist"] > button[role="tab"]:nth-of-type(n) * {
    color: #101828 !important;
    -webkit-text-fill-color: #101828 !important;
    text-shadow: none !important;
}

/* Streamlit versions may render a BaseWeb div instead of a button for each tab. */
html body div[data-testid="stTabs"] [data-baseweb="tab"],
html body div[data-testid="stTabs"] [role="tab"] {
    flex: 0 0 auto !important;
    margin: 0 1px !important;
    padding: 0.42rem 0.74rem !important;
    border-radius: 7px !important;
    letter-spacing: 0.01em !important;
    color: #101828 !important;
    -webkit-text-fill-color: #101828 !important;
    text-shadow: none !important;
}

html body div[data-testid="stTabs"] [data-baseweb="tab"] *,
html body div[data-testid="stTabs"] [role="tab"] * {
    color: #101828 !important;
    -webkit-text-fill-color: #101828 !important;
    opacity: 1 !important;
    text-shadow: none !important;
}
                                                                           
/* 5.5 ADICIONA OS CÍRCULOS COLORIDOS ANTES DO TEXTO DE CADA ABA */
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(1) p::before {
    content: ""; display: inline-block; width: 12px; height: 12px; background-color: #A78BFA; border-radius: 50%; margin-right: 8px; vertical-align: middle;
}
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(2) p::before {
    content: ""; display: inline-block; width: 12px; height: 12px; background-color: #818CF8; border-radius: 50%; margin-right: 8px; vertical-align: middle;
}
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(3) p::before {
    content: ""; display: inline-block; width: 12px; height: 12px; background-color: #60A5FA; border-radius: 50%; margin-right: 8px; vertical-align: middle;
}
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(4) p::before {
    content: ""; display: inline-block; width: 12px; height: 12px; background-color: #22D3EE; border-radius: 50%; margin-right: 8px; vertical-align: middle;
}
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(5) p::before {
    content: ""; display: inline-block; width: 12px; height: 12px; background-color: #34D399; border-radius: 50%; margin-right: 8px; vertical-align: middle;
}
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(6) p::before {
    content: ""; display: inline-block; width: 12px; height: 12px; background-color: #A3E635; border-radius: 50%; margin-right: 8px; vertical-align: middle;
}
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(7) p::before {
    content: ""; display: inline-block; width: 12px; height: 12px; background-color: #FACC15; border-radius: 50%; margin-right: 8px; vertical-align: middle;
}
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(8) p::before {
    content: ""; display: inline-block; width: 12px; height: 12px; background-color: #FB923C; border-radius: 50%; margin-right: 8px; vertical-align: middle;
}
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(9) p::before {
    content: ""; display: inline-block; width: 12px; height: 12px; background-color: #F87171; border-radius: 50%; margin-right: 8px; vertical-align: middle;
}
div[data-testid="stTabs"] [role="tablist"] > button:nth-of-type(10) p::before {
    content: ""; display: inline-block; width: 12px; height: 12px; background-color: #FF4B4B; border-radius: 50%; margin-right: 8px; vertical-align: middle;
}

/* 6. Destaque da Aba Ativa */
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    border-color: rgba(16, 24, 40, 0.35) !important;
    background-color: rgba(255, 255, 255, 0.30) !important;
    box-shadow: 0 1px 4px rgba(16, 24, 40, 0.18) !important;
}

/* Nested tabs group optional analyses without competing with the main workflow. */
div[data-testid="stTabs"] div[data-testid="stTabs"] [role="tablist"] {
    background: #ffffff !important;
    box-shadow: none !important;
    border-color: rgba(148, 163, 184, 0.56) !important;
}
div[data-testid="stTabs"] div[data-testid="stTabs"] [role="tablist"] > button p::before {
    display: none !important;
}
</style>
""",
    unsafe_allow_html=True,
)
# ----------------------------- CONSTANTS -----------------------------
# Define core column names and standard values used throughout the application.

PRESSURE_COL, TEMP_COL, COND_COL, WAVELENGTH_PREFIX, STATION_ID_COL = 'Depth', 'Temperature', 'Conductivity', 'X', 'station_id'
FILE_KEYWORDS = {'temp':'temp', 'cond':'cond', 'ed':'ed', 'lu':'lu', 'beta': 'beta'}
REQUIRED_PROFILER_FILE_TYPES = frozenset({'temp', 'cond', 'ed', 'lu'})
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
LW_TRANSMISSION_FACTOR = 0.54
ED_TRANSMISSION_FACTOR = 0.96
CONVOLUTION_METHOD_VERSION = 'v11_visible_integration_srf_coverage_99_5_owned_rrs'
OCI_SRF_READER_VERSION = 'OCI-SRF reader v6 (native-axis aliases)'
SCIENTIFIC_PRODUCT_WAVELENGTH_RANGE = (400.0, 700.0)
# Native PACE OCI RSRs retain weak out-of-domain wings.  A 99.5% area threshold
# keeps the official functions unchanged while excluding bands with material
# support outside the visible propagated-product domain.
MIN_SRF_DOMAIN_COVERAGE_PERCENT = 99.5
DEFAULT_KW_MODEL = 'Morel & Maritorena (2001)'
KW_MODEL_METADATA = {
    'Smith & Baker (1981)': {
        'range_nm': (300.0, 800.0),
        'short_label': 'Smith & Baker 81',
        'source_note': 'Table I selected Kw values; 300–800 nm',
    },
    'Morel & Maritorena (2001)': {
        'range_nm': (350.0, 700.0),
        'short_label': 'Morel & Maritorena 01',
        'source_note': 'Visible-range tabulation used by this app; 350–700 nm',
    },
}
ED0_PROPAGATION_MODE_LABELS = {
    'raw_fit': 'Profiler log-linear fit (raw observation / QC reference)',
    'kw_constrained': 'Kw-constrained Ed(0+) (400–700 nm)',
}
IRRADIANCE_MODE_LABELS = {
    'auto': 'Automatic hierarchy: Es → layer Ed(0+) → shallowest Ed(0+)',
    'filtered_es': 'Measured Es: filtered median',
    'original_es': 'Measured Es: original median',
    'layer_ed0_plus': 'Extrapolated Ed(0+) from the selected layer',
    'shallowest_ed0_plus': 'Ed(0+) from the shallowest valid profiler depth',
}
CONVOLUTION_IRRADIANCE_MODE_LABELS = {
    'match_rrs': 'Match the irradiance used to calculate the selected Rrs',
    **IRRADIANCE_MODE_LABELS,
}
VERTICAL_COORDINATE_LABELS = {
    'sea_pressure_dbar': 'Recorded as sea pressure (dbar) → final Depth (m)',
    'depth_m': 'Already recorded as Depth (m) → final Depth (m)',
    'absolute_pressure_dbar': 'Recorded as absolute pressure (dbar) → final Depth (m)',
}
ABSOLUTE_PRESSURE_AT_SEA_LEVEL_DBAR = 10.1325
PROFILER_FILE_TYPE_ALIASES = {
    'temp': frozenset({'temp', 'temperature'}),
    'cond': frozenset({'cond', 'conductivity'}),
    'ed': frozenset({'ed', 'irradiance', 'downwelling'}),
    'lu': frozenset({'lu', 'radiance', 'upwelling'}),
}
PROFILER_COLUMN_ALIASES = {
    STATION_ID_COL: frozenset({'stationid', 'station', 'cast', 'castid', 'profile', 'profileid'}),
    PRESSURE_COL: frozenset({
        'depth', 'depthm', 'depthmeter', 'depthmeters', 'profundidade',
        'profundidadem', 'pressure', 'pressuredbar', 'seapressure', 'pres', 'p',
    }),
    TEMP_COL: frozenset({'temperature', 'temperaturec', 'temp', 'tempc'}),
    COND_COL: frozenset({
        'conductivity', 'conductivitymscm', 'conductivitysm', 'cond', 'condmscm',
    }),
}

# ----------------------------- PROCESSING AND CALCULATION FUNCTIONS -----------------------------

def _normalise_header_token(column_name):
    """Normalizes a header for safe matching without changing the source data."""
    ascii_name = unicodedata.normalize('NFKD', str(column_name)).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z0-9]+', '', ascii_name.lower())


def _standard_wavelength_column(column_name):
    """Returns a canonical X<wavelength_nm> column name for supported wide spectra."""
    compact = re.sub(r'[\s_\-()\[\]]+', '', str(column_name).strip().lower()).replace(',', '.')
    match = re.fullmatch(r'(?:x|ed|lu|es|wl|lambda|wavelength)(\d+(?:\.\d+)?)(?:nm)?', compact)
    if match is None:
        match = re.fullmatch(r'(\d+(?:\.\d+)?)(?:nm)?', compact)
    if match is None:
        return None
    wavelength = float(match.group(1))
    if not 250.0 <= wavelength <= 2600.0:
        return None
    return f"{WAVELENGTH_PREFIX}{wavelength:g}"


def _standardize_profiler_columns(frame, required_columns, source_label):
    """Accepts documented header variants and returns the app's canonical column names."""
    frame = frame.copy()
    normalized_headers = {}
    for column in frame.columns:
        normalized_headers.setdefault(_normalise_header_token(column), []).append(column)

    rename_map = {}
    for canonical_name in PROFILER_COLUMN_ALIASES:
        candidates = list(normalized_headers.get(_normalise_header_token(canonical_name), []))
        for alias in PROFILER_COLUMN_ALIASES[canonical_name]:
            candidates.extend(normalized_headers.get(alias, []))
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) > 1:
            st.error(
                f"The {source_label} file has multiple possible columns for '{canonical_name}': "
                + ', '.join(map(str, candidates)) + '. Rename one column or remove the duplicate.'
            )
            return None
        if canonical_name in required_columns and not candidates:
            st.error(
                f"The {source_label} file is missing '{canonical_name}'. "
                f"Supported header variants include: {', '.join(sorted(PROFILER_COLUMN_ALIASES[canonical_name]))}."
            )
            return None
        if candidates and candidates[0] != canonical_name:
            rename_map[candidates[0]] = canonical_name

    for column in frame.columns:
        canonical_wavelength = _standard_wavelength_column(column)
        if canonical_wavelength is not None and canonical_wavelength != column:
            if canonical_wavelength in frame.columns or canonical_wavelength in rename_map.values():
                st.error(
                    f"The {source_label} file has duplicate spectral wavelength '{canonical_wavelength}'."
                )
                return None
            rename_map[column] = canonical_wavelength
    return frame.rename(columns=rename_map)


def _identify_profiler_file_type(filename):
    """Identifies a core profiler file from unambiguous filename tokens."""
    filename_tokens = set(re.split(r'[^a-z0-9]+', os.path.splitext(filename.lower())[0]))
    filename_tokens.discard('')
    matches = [
        file_type for file_type, aliases in PROFILER_FILE_TYPE_ALIASES.items()
        if filename_tokens.intersection(aliases)
    ]
    return matches[0] if len(matches) == 1 else None


def _to_sea_pressure(values, coordinate_type, latitude):
    """Converts a user-declared vertical coordinate to TEOS-10 sea pressure (dbar)."""
    values = pd.to_numeric(values, errors='coerce').to_numpy(dtype=float)
    if coordinate_type == 'sea_pressure_dbar':
        return values
    if coordinate_type == 'absolute_pressure_dbar':
        return values - ABSOLUTE_PRESSURE_AT_SEA_LEVEL_DBAR
    if coordinate_type == 'depth_m':
        if not np.isfinite(latitude) or not -90.0 <= latitude <= 90.0:
            raise ValueError('A station latitude between -90 and 90 degrees is required to convert depth to pressure.')
        return gsw.p_from_z(-values, latitude)
    raise ValueError(f"Unsupported vertical coordinate type: {coordinate_type}")


def _to_depth_m(values, coordinate_type, latitude):
    """Converts a user-declared vertical coordinate to positive-down depth (m)."""
    if latitude is None or not np.isfinite(latitude) or not -90.0 <= latitude <= 90.0:
        raise ValueError('A station latitude between -90 and 90 degrees is required to derive depth in metres.')
    sea_pressure = _to_sea_pressure(values, coordinate_type, latitude)
    return -gsw.z_from_p(sea_pressure, latitude)


def process_uploaded_files(
    profiler_files, es_file=None, beta_file=None,
    vertical_coordinate='sea_pressure_dbar', latitude=None,
    csv_separator=',', decimal='.',
):
    """
    Processes the uploaded in-water profiler files, Es (surface irradiance), and Beta (backscattering proxy) files.
    Matches keywords to identify each file type, cleans the data, interpolates to a common depth grid (if not already done), 
    and packages everything into a dictionary mapped by station_id.
    """
    
    # Uploaded science data must remain session-local. Do not cache this function.
    file_map = {}
    for f in profiler_files:
        if getattr(f, 'size', 0) > MAX_UPLOAD_BYTES:
            st.error(f"'{f.name}' exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
            return None, None
        key = _identify_profiler_file_type(f.name)
        if key is None:
            st.error(
                f"Could not identify exactly one profiler type for '{f.name}'. "
                "Use a filename token such as temp/temperature, cond/conductivity, ed/irradiance, or lu/radiance."
            )
            return None, None
        if key in file_map:
            st.error(f"More than one '{key}' profiler file was uploaded.")
            return None, None
        try:
            file_map[key] = io.StringIO(f.getvalue().decode("utf-8-sig"))
        except UnicodeDecodeError:
            st.error(f"'{f.name}' is not a UTF-8 CSV file.")
            return None, None
    
    if beta_file:
        if getattr(beta_file, 'size', 0) > MAX_UPLOAD_BYTES:
            st.error(f"'{beta_file.name}' exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
            return None, None
        try:
            file_map['beta'] = io.StringIO(beta_file.getvalue().decode("utf-8-sig"))
        except UnicodeDecodeError:
            st.error(f"'{beta_file.name}' is not a UTF-8 CSV file.")
            return None, None
            
    missing_core = REQUIRED_PROFILER_FILE_TYPES.difference(file_map)
    if missing_core:
        st.error(
            "Upload failed. Missing required profiler file(s): "
            + ", ".join(sorted(missing_core)) + "."
        )
        return None, None

    # Common representations of missing data to be parsed as NaNs.

    possible_NA_values = ['#########', 'NA', 'N/A', 'Not a Number', 'missing', '-9999']
    try:
        df_temp_raw = pd.read_csv(file_map['temp'], sep=csv_separator, decimal=decimal, na_values=possible_NA_values)
        df_cond_raw = pd.read_csv(file_map['cond'], sep=csv_separator, decimal=decimal, na_values=possible_NA_values)
        df_ed_raw = pd.read_csv(file_map['ed'], sep=csv_separator, decimal=decimal, na_values=possible_NA_values)
        df_lu_raw = pd.read_csv(file_map['lu'], sep=csv_separator, decimal=decimal, na_values=possible_NA_values)
        if 'beta' in file_map:
            df_beta_raw = pd.read_csv(file_map['beta'], sep=csv_separator, decimal=decimal, na_values=possible_NA_values)
        else:
            df_beta_raw = None
        if es_file:
            if getattr(es_file, 'size', 0) > MAX_UPLOAD_BYTES:
                st.error(f"'{es_file.name}' exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
                return None, None
            df_es_raw = pd.read_csv(es_file, sep=csv_separator, decimal=decimal, na_values=possible_NA_values)
        else:
            df_es_raw = None
    except Exception as e:
        st.error(f"Error reading core CSV files. Please check file format. Details: {e}")
        return None, None

    required_columns = {
        'temperature': {STATION_ID_COL, PRESSURE_COL, TEMP_COL},
        'conductivity': {STATION_ID_COL, PRESSURE_COL, COND_COL},
        'Ed': {STATION_ID_COL, PRESSURE_COL},
        'Lu': {STATION_ID_COL, PRESSURE_COL},
    }
    normalized_frames = {}
    for label, frame in {
        'temperature': df_temp_raw,
        'conductivity': df_cond_raw,
        'Ed': df_ed_raw,
        'Lu': df_lu_raw,
    }.items():
        normalized = _standardize_profiler_columns(frame, required_columns[label], label)
        if normalized is None:
            return None, None
        normalized_frames[label] = normalized
    df_temp_raw = normalized_frames['temperature']
    df_cond_raw = normalized_frames['conductivity']
    df_ed_raw = normalized_frames['Ed']
    df_lu_raw = normalized_frames['Lu']

    if df_beta_raw is not None:
        df_beta_raw = _standardize_profiler_columns(
            df_beta_raw, {STATION_ID_COL, PRESSURE_COL}, 'Beta',
        )
        if df_beta_raw is None:
            return None, None
    if df_es_raw is not None:
        df_es_raw = _standardize_profiler_columns(df_es_raw, {STATION_ID_COL}, 'Es')
        if df_es_raw is None:
            return None, None

    try:
        for frame in [df_temp_raw, df_cond_raw, df_ed_raw, df_lu_raw, df_beta_raw]:
            if frame is not None:
                frame[PRESSURE_COL] = _to_depth_m(
                    frame[PRESSURE_COL], vertical_coordinate, latitude,
                )
    except ValueError as error:
        st.error(f"Could not prepare the vertical coordinate: {error}")
        return None, None

    wave_cols_ed = [col for col in df_ed_raw.columns if col.startswith(WAVELENGTH_PREFIX)]
    wave_cols_lu = [col for col in df_lu_raw.columns if col.startswith(WAVELENGTH_PREFIX)]
    common_wave_cols = sorted(
        set(wave_cols_ed).intersection(wave_cols_lu),
        key=lambda col: float(col.replace(WAVELENGTH_PREFIX, '')),
    )
    if not common_wave_cols:
        st.error("Ed and Lu files must share at least one numeric wavelength column named X<nanometers>.")
        return None, None

    for frame, value_columns in [
        (df_temp_raw, [TEMP_COL]),
        (df_cond_raw, [COND_COL]),
        (df_ed_raw, common_wave_cols),
        (df_lu_raw, common_wave_cols),
    ]:
        for column in value_columns:
            frame[column] = pd.to_numeric(frame[column], errors='coerce')

    es_by_station = {}
    if df_es_raw is not None:
        for column in [col for col in df_es_raw.columns if col.startswith(WAVELENGTH_PREFIX)]:
            df_es_raw[column] = pd.to_numeric(df_es_raw[column], errors='coerce')
        es_by_station = {station_id: group.copy() for station_id, group in df_es_raw.groupby(STATION_ID_COL)}

    station_ids_from_temp = df_temp_raw[STATION_ID_COL].unique()
    station_data_package = {}; successful_stations = []

    for station in station_ids_from_temp:
        df_temp = df_temp_raw[df_temp_raw[STATION_ID_COL] == station]
        df_cond = df_cond_raw[df_cond_raw[STATION_ID_COL] == station]
        df_ed = df_ed_raw[df_ed_raw[STATION_ID_COL] == station]
        df_lu = df_lu_raw[df_lu_raw[STATION_ID_COL] == station]
        
        if any(df.empty for df in [df_temp, df_cond, df_ed, df_lu]):
            st.warning(f"Skipping station '{station}' due to missing core files."); continue

        wave_cols = common_wave_cols
        wavelengths_temp = np.array([float(c.replace(WAVELENGTH_PREFIX, '')) for c in wave_cols])
        
        # Identify the column closest to 490nm to use as a baseline for valid optical data.

        idx_490 = (np.abs(wavelengths_temp - 490)).argmin()
        canary_col = wave_cols[idx_490]
        
        # Clean optical data based on the canary wavelength (must exist and be > 0).

        df_ed_clean = df_ed.dropna(subset=[canary_col])
        df_ed_clean = df_ed_clean[df_ed_clean[canary_col] > 0]
        
        if df_ed_clean.empty: 
            st.warning(f"Skipping station '{station}' due to no valid Ed(490) data."); continue
        
        optical_min_depth = df_ed_clean[PRESSURE_COL].min()
        optical_max_depth = df_ed_clean[PRESSURE_COL].max()
        
        # Create a master depth grid based on temperature depths that fall within the optical depth range.

        all_temp_depths = np.sort(df_temp[PRESSURE_COL].dropna().unique())
        master_depth_m = all_temp_depths[(all_temp_depths >= optical_min_depth) & (all_temp_depths <= optical_max_depth)]
        
        if master_depth_m.size < 2: continue
    
        df_temp_clean = (
            df_temp.dropna(subset=[PRESSURE_COL, TEMP_COL])
            .groupby(PRESSURE_COL, as_index=False)[TEMP_COL].mean()
            .sort_values(PRESSURE_COL)
        )
        df_cond_clean = (
            df_cond.dropna(subset=[PRESSURE_COL, COND_COL])
            .groupby(PRESSURE_COL, as_index=False)[COND_COL].mean()
            .sort_values(PRESSURE_COL)
        )

        # Gap evaluation. Calculates the maximum difference between valid depths to detect missing data blocks
        
        if not df_temp_clean.empty:
            temp_depths_sorted = df_temp_clean[PRESSURE_COL].sort_values()
            max_temp_gap = temp_depths_sorted.diff().max()

            # If the largest gap is greater than 2 meters, emit a warning.
            if pd.notna(max_temp_gap) and max_temp_gap > 2:
                st.warning(
                    f"**Caution for station '{station}':** Missing physical data detected. "
                    f"Maximum depth gap is **{max_temp_gap:.1f} meters**. "
                    f"Data has been linearly interpolated to maintain grid consistency, but physical parameters (like N²) may be distorted in this interval."
                )

        # If no data is left (or fewer than 2 points), interpolation is impossible.
        if df_temp_clean.empty or len(df_temp_clean) < 2:
            st.warning(f"Skipping station '{station}': Not enough Temperature data (too many NaNs).")
            continue
            
        if df_cond_clean.empty or len(df_cond_clean) < 2:
            st.warning(f"Skipping station '{station}': Not enough Conductivity data.")
            continue

        # Safe interpolation mapped to the master optical depth grid
        temp_data = np.interp(master_depth_m, df_temp_clean[PRESSURE_COL], df_temp_clean[TEMP_COL])
        cond_data = np.interp(master_depth_m, df_cond_clean[PRESSURE_COL], df_cond_clean[COND_COL])

        wavelengths = np.array([float(c.replace(WAVELENGTH_PREFIX, '')) for c in wave_cols])
        
        # Group, reindex, and interpolate Ed data.

        df_ed_grouped = df_ed[[PRESSURE_COL] + wave_cols].groupby(PRESSURE_COL).mean()
        ed_df_interp = df_ed_grouped.reindex(df_ed_grouped.index.union(master_depth_m)).interpolate(method='index').loc[master_depth_m]
        ed_data = ed_df_interp.values

        # Group, reindex, and interpolate Lu data.

        df_lu_grouped = df_lu[[PRESSURE_COL] + wave_cols].groupby(PRESSURE_COL).mean()
        lu_df_interp = df_lu_grouped.reindex(df_lu_grouped.index.union(master_depth_m)).interpolate(method='index').loc[master_depth_m]
        lu_data = lu_df_interp.values
        
        # --- BLOCO DO BETA CORRIGIDO COM NUMPY (100% GARANTIDO) ---
        beta_data_interp = None
        beta_cols = []
        df_beta = None
        if df_beta_raw is not None:
          df_beta = df_beta_raw[df_beta_raw[STATION_ID_COL] == station].copy()

        if df_beta is not None and not df_beta.empty:
          # 1. Identifica colunas de dados do beta
          beta_cols = [
              col
              for col in df_beta.columns
              if col not in [PRESSURE_COL, STATION_ID_COL]
          ]

          # 2. Força profundidade e dados para numérico
          df_beta[PRESSURE_COL] = pd.to_numeric(
              df_beta[PRESSURE_COL], errors='coerce'
          )
          for col in beta_cols:
            df_beta[col] = pd.to_numeric(df_beta[col], errors='coerce')

          # 3. Limpa e ordena por profundidade
          df_beta_clean = df_beta.dropna(subset=[PRESSURE_COL]).sort_values(
              PRESSURE_COL
          )

          if not df_beta_clean.empty and len(df_beta_clean) >= 2:
            z_beta = df_beta_clean[PRESSURE_COL].values
            beta_interp_dict = {}

            # Interpolação direta e segura via numpy (imune a erros de índice do pandas)
            for col in beta_cols:
              val_col = df_beta_clean[col].values
              valid_m = ~np.isnan(val_col)
              if np.sum(valid_m) >= 2:
                beta_interp_dict[col] = np.interp(
                    master_depth_m,
                    z_beta[valid_m],
                    val_col[valid_m],
                    left=np.nan,
                    right=np.nan,
                )
              else:
                beta_interp_dict[col] = np.full(len(master_depth_m), np.nan)

            # Monta o DataFrame final alinhado à grade master
            beta_data_interp = pd.DataFrame(
                beta_interp_dict, index=master_depth_m
            ).dropna(how='all')

            if beta_data_interp.empty:
              st.warning(
                  f"For station '{station}', beta data had no overlapping depth"
                  ' with the profiler.'
              )
          else:
            st.warning(
                f"For station '{station}', the beta file contained no valid"
                ' numeric data.'
            )
                 
        # Package the successfully processed data for this station.

        es_values = None
        es_spectra = None
        if station in es_by_station:
            missing_es_wavelengths = set(wave_cols).difference(es_by_station[station].columns)
            if missing_es_wavelengths:
                st.warning(f"Ignoring Es for station '{station}': wavelength columns do not match Ed/Lu.")
            else:
                es_spectra = es_by_station[station][wave_cols].copy()
                es_values = es_spectra.median().to_numpy()

        station_data_package[station] = {
            'depth_m': master_depth_m, 'temperature': temp_data, 'conductivity': cond_data,
            'Ed_data': ed_data, 'Lu_data': lu_data, 'wavelengths': wavelengths, 
            'Es': es_values, 'Es_all_spectra': es_spectra,
            'beta_data': beta_data_interp, 'beta_cols': beta_cols,
            'vertical_coordinate_source': vertical_coordinate,
            'vertical_conversion_latitude': (
                float(latitude) if latitude is not None and np.isfinite(latitude) else None
            ),
        }
        successful_stations.append(station)
        
    return station_data_package, successful_stations


def analyze_mixture(SA, CT, vertices, names):
    """
   Calculates water mass mixing fractions using barycentric coordinates 
   based on a defined T-S (Temperature-Salinity) mixing triangle.
   """
    p1, p2, p3 = np.array(vertices[0]), np.array(vertices[1]), np.array(vertices[2])
    mixing_percentages = []
    def get_barycentric_coords(p, a, b, c):
        v0, v1, v2 = b - a, c - a, p - a
        d00, d01, d11, d20, d21 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1), np.dot(v2, v0), np.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-9: return -1, -1, -1
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return u, v, w
    for sal, temp in zip(SA, CT):
        point = np.array([sal, temp])
        
        # Only accept points that fall strictly inside (or on the edges of) the triangle

        w3, w1, w2 = get_barycentric_coords(point, p3, p1, p2)
        if w1 >= -1e-9 and w2 >= -1e-9 and w3 >= -1e-9:
            mixing_percentages.append((w1, w2, w3))
        else:
            mixing_percentages.append((np.nan, np.nan, np.nan))
    df_mix = pd.DataFrame(mixing_percentages, columns=names)
    avg_mix = df_mix.mean()
    avg_text = f"**Avg. Mixture ({len(df_mix.dropna())} points):**\n- {avg_mix[names[0]]*100:.1f}% {names[0]}\n- {avg_mix[names[1]]*100:.1f}% {names[1]}\n- {avg_mix[names[2]]*100:.1f}% {names[2]}"
    return avg_text, df_mix

def perform_convolution(prof_wl, lw_hyperspectral, surface_irradiance, srf_wl, srf_response):
    
    """
    Performs spectral convolution on Lw and its surface-irradiance denominator independently before division
    to simulate multi-spectral satellite bands (e.g., MODIS, Sentinel-3).
    
    Methodology follows Burggraaff (2020) to avoid convolution biases: 
    Reflectance (Rrs) is NOT convolved directly. Instead, the numerator (Lw) 
    and denominator are convolved independently using Simpson's rule. The
    denominator is either measured Es or an Ed(0+) estimate, matching the
    calculation of the selected hyperspectral Rrs source.
    """
    
    domain_coverage = srf_domain_coverage_percent(srf_wl, srf_response)
    if (
        not np.isfinite(domain_coverage)
        or domain_coverage < MIN_SRF_DOMAIN_COVERAGE_PERCENT
    ):
        return np.nan

    profiler_wavelengths = np.asarray(prof_wl, dtype=float).reshape(-1)
    lw_values = np.asarray(lw_hyperspectral, dtype=float).reshape(-1)
    irradiance_values = np.asarray(surface_irradiance, dtype=float).reshape(-1)
    profile_order = np.argsort(profiler_wavelengths)
    profiler_wavelengths = profiler_wavelengths[profile_order]
    lw_values = lw_values[profile_order]
    irradiance_values = irradiance_values[profile_order]
    # Eligibility uses the full native SRF. Accepted bands are integrated only
    # in the scientific domain; the <=0.5% outside wings have no valid product.
    visible = scientific_product_mask(profiler_wavelengths)
    profiler_wavelengths = profiler_wavelengths[visible]
    lw_values = lw_values[visible]
    irradiance_values = irradiance_values[visible]
    response_wavelengths = np.asarray(srf_wl, dtype=float).reshape(-1)
    response_values = np.asarray(srf_response, dtype=float).reshape(-1)
    response_order = np.argsort(response_wavelengths)
    response_wavelengths = response_wavelengths[response_order]
    response_values = response_values[response_order]

    # Step 1: Interpolate the satellite's SRF onto the profiler's wavelength grid.
    srf_interp = np.interp(
        profiler_wavelengths, response_wavelengths, response_values, left=0, right=0,
    )
    
    # Step 2: A propagated scientific product must have valid numerator and
    # denominator everywhere the selected SRF has non-zero support.
    srf_support = srf_interp > 0
    valid_radiometry = (
        np.isfinite(lw_values)
        & np.isfinite(irradiance_values)
        & (irradiance_values > 0)
    )
    if np.sum(srf_support) < 2 or not np.all(valid_radiometry[srf_support]):
        return np.nan

    # Avoid 0 × NaN outside the SRF support.
    lw_integrand = np.where(srf_support, lw_values * srf_interp, 0.0)
    ed_integrand = np.where(srf_support, irradiance_values * srf_interp, 0.0)
    
    # Step 3: Perform numerical integration using Simpson's Rule
    # This accounts for potentially non-uniform wavelength spacing (dx)
    lw_integrated = simpson(y=lw_integrand, x=profiler_wavelengths)
    ed_integrated = simpson(y=ed_integrand, x=profiler_wavelengths)
    
    # Step 4: Calculate the final band-specific Rrs. Avoid division by zero.
    if ed_integrated > 0:
        return lw_integrated / ed_integrated
    else:
        return np.nan


def assess_convolution_band_qc(prof_wl, lw_hyperspectral, surface_irradiance, srf_wl, srf_response):
    """Reports SRF/data support and flags bands excluded from the product domain."""
    profile_wavelengths = np.asarray(prof_wl, dtype=float).reshape(-1)
    lw_values = np.asarray(lw_hyperspectral, dtype=float).reshape(-1)
    irradiance_values = np.asarray(surface_irradiance, dtype=float).reshape(-1)
    profile_order = np.argsort(profile_wavelengths)
    profile_wavelengths = profile_wavelengths[profile_order]
    lw_values = lw_values[profile_order]
    irradiance_values = irradiance_values[profile_order]
    visible = scientific_product_mask(profile_wavelengths)
    profile_wavelengths = profile_wavelengths[visible]
    lw_values = lw_values[visible]
    irradiance_values = irradiance_values[visible]
    if profile_wavelengths.size < 2:
        return np.nan, 0.0, np.nan, 'Insufficient visible profiler support; not convolved'
    response_wavelengths = np.asarray(srf_wl, dtype=float)
    response = np.clip(np.asarray(srf_response, dtype=float), 0, None)
    valid_srf = np.isfinite(response_wavelengths) & np.isfinite(response)
    if np.sum(valid_srf) < 2:
        return np.nan, np.nan, np.nan, 'Invalid SRF'

    response_wavelengths = response_wavelengths[valid_srf]
    response = response[valid_srf]
    sort_order = np.argsort(response_wavelengths)
    response_wavelengths = response_wavelengths[sort_order]
    response = response[sort_order]
    full_srf_area = simpson(y=response, x=response_wavelengths)
    if not np.isfinite(full_srf_area) or full_srf_area <= 0:
        return np.nan, np.nan, np.nan, 'Invalid SRF area'

    profile_response = np.interp(
        profile_wavelengths, response_wavelengths, response, left=0, right=0,
    )
    profile_srf_area = simpson(y=profile_response, x=profile_wavelengths)
    srf_coverage_percent = 100 * profile_srf_area / full_srf_area
    valid_radiometry = (
        np.isfinite(lw_values)
        & np.isfinite(irradiance_values)
        & (irradiance_values > 0)
    )
    valid_srf_area = simpson(
        y=profile_response * valid_radiometry.astype(float), x=profile_wavelengths,
    )
    radiometry_coverage_percent = (
        100 * valid_srf_area / profile_srf_area if profile_srf_area > 0 else np.nan
    )
    band_centre = np.sum(response_wavelengths * response) / np.sum(response)

    domain_coverage = srf_domain_coverage_percent(response_wavelengths, response)
    flags = []
    if (
        not np.isfinite(domain_coverage)
        or domain_coverage < MIN_SRF_DOMAIN_COVERAGE_PERCENT
    ):
        flags.append('Outside 400–700 nm propagated-product domain; not convolved')
    if srf_coverage_percent < 98:
        flags.append('Partial SRF overlap with profiler spectrum')
    if np.sum(profile_response > 0) < 2:
        flags.append('Insufficient sampled SRF support; not convolved')
    if not np.all(valid_radiometry[profile_response > 0]):
        flags.append('Missing/non-positive radiometric support; not convolved')
    if band_centre > SCIENTIFIC_PRODUCT_WAVELENGTH_RANGE[1]:
        flags.append('NIR retained only as raw/SNR diagnostic')
    return band_centre, srf_coverage_percent, radiometry_coverage_percent, '; '.join(flags) or 'Within displayed QC range'


def convolution_history_to_dataframe(convolution_history):
    """Flattens saved convolution runs into an auditable long-form table."""
    records = []
    for run in convolution_history:
        results = run['results']
        band_centers = run['band_centers_nm']
        for (_, result), center_nm in zip(results.iterrows(), band_centers):
            records.append({
                'Run': run['run_id'],
                'Sensor': run['sensor'],
                'Rrs source': run['rrs_source'],
                'Rrs irradiance': run['rrs_irradiance'],
                'Rrs treatment in convolution': run.get('rrs_treatment', 'Not recorded'),
                'Layer Ed(0+) treatment': run.get('ed0_treatment', 'Not recorded'),
                'Convolution selection': run['convolution_selection'],
                'Convolution irradiance': run['convolution_irradiance'],
                'Band': result['Band'],
                'Band centre (nm)': center_nm,
                'Convolved_Rrs_sr-1': result['Convolved_Rrs_sr-1'],
                'SRF coverage (%)': result.get('SRF coverage (%)', np.nan),
                'Radiometric coverage (%)': result.get('Radiometric coverage (%)', np.nan),
                'QC flag': result.get('QC flag', 'Not recorded'),
            })
    return pd.DataFrame(records)


def get_spectral_display_bounds(wavelengths, extra_wavelengths=None):
    """Returns a shared spectral domain so related plots do not hide edge bands."""
    values = [np.asarray(wavelengths, dtype=float).reshape(-1)]
    if extra_wavelengths is not None:
        values.append(np.asarray(extra_wavelengths, dtype=float).reshape(-1))
    combined = np.concatenate(values)
    finite = combined[np.isfinite(combined)]
    upper = np.ceil(np.nanmax(finite) / 10) * 10 if finite.size else 700.0
    return 400.0, max(700.0, float(upper))


def place_legend_below_data(fig, ax, max_columns=4, fontsize=8):
    """Places a dense legend below the axes so it never masks a spectral feature."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    n_columns = min(max_columns, len(handles))
    n_rows = int(np.ceil(len(handles) / n_columns))
    # Leave enough of the figure for every legend row while keeping the data panel tall.
    fig.subplots_adjust(bottom=min(0.40, 0.10 + 0.07 * n_rows))
    ax.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.10),
        ncol=n_columns,
        fontsize=fontsize,
        framealpha=0.92,
        columnspacing=1.15,
        handlelength=2.0,
    )


def render_stratification_beta(station_data, phys_props_df, analysis_layers):
    """Renders the N²/beta profile used to choose optical analysis layers."""
    fig_n2, ax_n2 = plt.subplots(figsize=(6.6, 4.8))
    n2_clean = phys_props_df[['n2', 'Depth_mid_m']].dropna()
    line_n2, = ax_n2.plot(
        n2_clean['n2'] * 1e5,
        n2_clean['Depth_mid_m'],
        color=STYLE_CONFIG['n2_line_color'],
        zorder=10,
        label=r'N$^2$',
    )
    ax_n2.set_xlabel(
        r'N$^2$ (x 10$^{-5}$ s$^{-2}$)', weight='bold',
        color=STYLE_CONFIG['n2_line_color'],
    )
    ax_n2.set_ylabel("Depth (m)", weight='bold')
    ax_n2.set_title("Stratification and backscattering: use this profile to define optical layers", weight='bold')
    ax_n2.invert_yaxis()
    ax_n2.tick_params(axis='x', labelcolor=STYLE_CONFIG['n2_line_color'])

    beta_data_df = station_data.get('beta_data')
    if beta_data_df is not None and not beta_data_df.empty and 'selected_beta_col' in st.session_state:
        ax_beta = ax_n2.twiny()
        selected_beta_col = st.session_state.selected_beta_col
        beta_series = beta_data_df[selected_beta_col]
        line_beta, = ax_beta.plot(
            beta_series.values, beta_series.index,
            color=STYLE_CONFIG['beta_line_color'], zorder=9,
            label=f'Beta ({selected_beta_col})',
        )
        ax_beta.set_xlabel(
            r'$\beta$ (m$^{-1}$ sr$^{-1}$)', weight='bold',
            color=STYLE_CONFIG['beta_line_color'],
        )
        ax_beta.tick_params(axis='x', labelcolor=STYLE_CONFIG['beta_line_color'])
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))
        ax_beta.xaxis.set_major_formatter(formatter)
        ax_beta.xaxis.offsetText.set_color(STYLE_CONFIG['beta_line_color'])
        ax_n2.legend([line_n2, line_beta], [line_n2.get_label(), line_beta.get_label()], loc='best')
    else:
        ax_n2.legend(loc='best')

    layer_colors = sns.color_palette('husl', max(3, len(analysis_layers)))
    for color_index, (_, layer_data) in enumerate(sorted(analysis_layers.items())):
        ax_n2.axhspan(*layer_data['range'], facecolor=layer_colors[color_index], alpha=0.25, zorder=0)
    ax_n2.grid(True, linestyle='--')
    st.pyplot(fig_n2, use_container_width=True)
    st.session_state.fig_n2 = fig_n2


def render_rrs_spectrum(station_data, analysis_layers, derived_products, y_max_rrs, spectral_bounds):
    """Renders the Rrs panel paired with the N²/beta layer-selection guide."""
    fig_rrs, ax_rrs = plt.subplots(figsize=(6.6, 4.8))
    layer_colors = sns.color_palette('husl', max(3, len(analysis_layers)))
    if derived_products and 'rrs_df_export' in derived_products:
        rrs_df = derived_products['rrs_df_export']
        if 'initial_rrs_sr-1' in rrs_df.columns:
            ax_rrs.plot(
                rrs_df['wavelength_nm'], rrs_df['initial_rrs_sr-1'],
                color='white', linestyle='--', label='Initial Rrs(0-)',
            )
        for color_index, (layer_num, _) in enumerate(sorted(analysis_layers.items())):
            column_name = f'propagated_rrs_L{layer_num}_sr-1'
            if column_name in rrs_df.columns:
                ax_rrs.plot(
                    rrs_df['wavelength_nm'], rrs_df[column_name],
                    color=layer_colors[color_index], label=f'Prop. Rrs (L{layer_num})',
                )
    else:
        initial_rrs = calculate_Rrs(
            station_data['Ed_data'], station_data['Lu_data'],
            pressure=station_data['depth_m'], es_data=get_active_es(station_data),
        )
        ax_rrs.plot(
            station_data['wavelengths'], initial_rrs,
            color='white', linestyle='--', label='Initial Rrs(0-)',
        )
    ax_rrs.set_xlabel("Wavelength (nm)", weight='bold')
    ax_rrs.set_ylabel(r'Rrs (sr$^{-1}$)', weight='bold')
    ax_rrs.set_title("Remote Sensing Reflectance (Rrs)", weight='bold')
    ax_rrs.set_xlim(*spectral_bounds)
    ax_rrs.set_ylim(bottom=0, top=y_max_rrs)
    ax_rrs.grid(True, linestyle='--')
    ax_rrs.legend(fontsize=8)
    st.pyplot(fig_rrs, use_container_width=True)
    st.session_state.fig_rrs = fig_rrs


def render_layer_setup(station_id, station_data):
    """Renders the layer and propagated-Rrs setup next to the optical results."""
    max_depth = (
        float(np.nanmax(station_data['depth_m']))
        if station_data['depth_m'].size > 0 else 0.0
    )
    if not np.isfinite(max_depth) or max_depth <= 0:
        st.error("The selected station has no valid positive depth values.")
        return

    default_depth_range = (min(10.0, max_depth), min(20.0, max_depth))
    if default_depth_range[0] >= default_depth_range[1]:
        default_depth_range = (0.0, max_depth)

    setup_left, setup_right = st.columns([3, 2])
    with setup_left:
        selected_depth = st.slider(
            "Depth range for the analysis layer (m):",
            0.0,
            max_depth,
            default_depth_range,
            step=0.5,
            key=f"slider_{station_id}",
        )
    with setup_right:
        propagation_irradiance_mode = st.selectbox(
            "Reference irradiance for propagated Rrs:",
            options=list(IRRADIANCE_MODE_LABELS),
            format_func=lambda mode: IRRADIANCE_MODE_LABELS[mode],
            key=f"propagation_irradiance_mode_{station_id}",
            help=(
                "This selection applies to every propagated Rrs layer and is "
                "saved in the irradiance-provenance table."
            ),
        )
        ed0_propagation_mode = st.selectbox(
            "Selected-layer Ed(0+) treatment:",
            options=list(ED0_PROPAGATION_MODE_LABELS),
            format_func=lambda mode: ED0_PROPAGATION_MODE_LABELS[mode],
            key=f"ed0_propagation_mode_{station_id}",
            help=(
                "The raw fit preserves the profiler observation for QC. The Kw-constrained "
                "option uses the same Kw constraint as the propagated Kd and Klu products."
            ),
        )

    layer_num_to_add = int(st.number_input(
        "Layer number:", min_value=1, value=1, step=1,
        key=f"layer_select_{station_id}",
    ))
    z_min, z_max = selected_depth
    add_col, remove_col, clear_col = st.columns(3)

    if add_col.button(f"Analyze / update L{layer_num_to_add}", use_container_width=True):
        st.session_state.profile_specific_layers.setdefault(station_id, {})
        st.session_state.profile_specific_layers[station_id][layer_num_to_add] = {
            'range': (z_min, z_max)
        }
        st.toast(f"Layer {layer_num_to_add} saved.")
        st.rerun()

    if remove_col.button(f"Remove L{layer_num_to_add}", use_container_width=True):
        defined_layers = st.session_state.profile_specific_layers.get(station_id, {})
        if layer_num_to_add in defined_layers:
            del defined_layers[layer_num_to_add]
            model_data = st.session_state.empirical_model_data
            if not model_data.empty:
                st.session_state.empirical_model_data = model_data[
                    ~(
                        (model_data['Station_ID'] == station_id)
                        & (model_data['Layer'] == f"Layer {layer_num_to_add}")
                    )
                ]
            st.toast(f"Layer {layer_num_to_add} removed.")
            st.rerun()
        st.info("That layer is not defined for the active profile.")

    defined_layers = st.session_state.profile_specific_layers.get(station_id, {})
    if clear_col.button("Clear profile layers", use_container_width=True, disabled=not defined_layers):
        st.session_state.profile_specific_layers[station_id] = {}
        model_data = st.session_state.empirical_model_data
        if not model_data.empty:
            st.session_state.empirical_model_data = model_data[
                model_data['Station_ID'] != station_id
            ]
        st.toast("All layers for the active profile were cleared.")
        st.rerun()

    if defined_layers:
        layer_table = pd.DataFrame([
            {
                'Layer': f"L{layer_num}",
                'Depth start (m)': data['range'][0],
                'Depth end (m)': data['range'][1],
                'Rrs irradiance': IRRADIANCE_MODE_LABELS[propagation_irradiance_mode],
                'Ed(0+) treatment': ED0_PROPAGATION_MODE_LABELS[ed0_propagation_mode],
            }
            for layer_num, data in sorted(defined_layers.items())
        ])
        st.dataframe(layer_table, use_container_width=True, hide_index=True)
    else:
        st.caption("No layers have been defined yet. Define one to enable propagated Rrs and downstream model inputs.")


def get_active_es(station_data):
    """Returns filtered Es when available, otherwise the originally measured Es."""
    filtered_es = st.session_state.get('new_es_median')
    return filtered_es if filtered_es is not None else station_data.get('Es')


def resolve_surface_irradiance(
    station_data,
    selection='auto',
    z_top=None,
    z_bottom=None,
    kd_constraint=None,
    ed0_propagation_mode='raw_fit',
):
    """Resolves a user-selected irradiance source for Rrs or convolution."""
    ed_data = station_data['Ed_data']
    pressure = station_data['depth_m']

    if selection == 'filtered_es':
        return get_rrs_surface_irradiance(
            ed_data, pressure, es_data=st.session_state.get('new_es_median'),
            z_top=z_top, z_bottom=z_bottom,
            kd_constraint=kd_constraint, ed0_propagation_mode=ed0_propagation_mode,
            mode='measured_es',
            es_label='Measured Es (filtered median)',
        )
    if selection == 'original_es':
        return get_rrs_surface_irradiance(
            ed_data, pressure, es_data=station_data.get('Es'),
            z_top=z_top, z_bottom=z_bottom,
            kd_constraint=kd_constraint, ed0_propagation_mode=ed0_propagation_mode,
            mode='measured_es',
            es_label='Measured Es (original median)',
        )
    if selection == 'layer_ed0_plus':
        return get_rrs_surface_irradiance(
            ed_data, pressure, z_top=z_top, z_bottom=z_bottom,
            kd_constraint=kd_constraint, ed0_propagation_mode=ed0_propagation_mode,
            mode='layer_ed0_plus',
        )
    if selection == 'shallowest_ed0_plus':
        return get_rrs_surface_irradiance(
            ed_data, pressure, z_top=z_top, z_bottom=z_bottom,
            kd_constraint=kd_constraint, ed0_propagation_mode=ed0_propagation_mode,
            mode='shallowest_ed0_plus',
        )

    active_es_label = (
        'Measured Es (filtered median)'
        if st.session_state.get('new_es_median') is not None
        else 'Measured Es (original median)'
    )
    return get_rrs_surface_irradiance(
        ed_data, pressure, es_data=get_active_es(station_data),
        z_top=z_top, z_bottom=z_bottom,
        kd_constraint=kd_constraint, ed0_propagation_mode=ed0_propagation_mode,
        mode='auto',
        es_label=active_es_label,
    )


def calculate_kd_fit_audit(
    depths, ed_data, wavelengths, z_min, z_max, layer_num,
    kd_raw, kw_reference, kd_constrained, kw_model,
):
    """Returns fit diagnostics without changing the calculated Kd products."""
    depths = np.asarray(depths, dtype=float)
    ed_data = np.asarray(ed_data, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    layer_mask = np.isfinite(depths) & (depths >= z_min) & (depths <= z_max)
    layer_depths = depths[layer_mask]
    layer_ed = ed_data[layer_mask, :]
    audit_rows = []

    for wavelength_index, wavelength in enumerate(wavelengths):
        signal = layer_ed[:, wavelength_index]
        valid = np.isfinite(signal) & (signal > 0)
        fit_depths = layer_depths[valid]
        fit_log_ed = np.log(signal[valid])
        n_valid = int(valid.sum())

        raw_value = float(kd_raw[wavelength_index])
        kw_value = float(kw_reference[wavelength_index])
        constrained_value = float(kd_constrained[wavelength_index])
        r_squared = np.nan
        standard_error = np.nan
        ci95_low = np.nan
        ci95_high = np.nan

        if n_valid >= 2 and np.unique(fit_depths).size >= 2 and np.isfinite(raw_value):
            # The fitted model is ln(Ed) = intercept - Kd * depth.
            intercept = np.mean(fit_log_ed + raw_value * fit_depths)
            predicted = intercept - raw_value * fit_depths
            residuals = fit_log_ed - predicted
            residual_sum_squares = np.sum(residuals ** 2)
            total_sum_squares = np.sum((fit_log_ed - np.mean(fit_log_ed)) ** 2)
            if total_sum_squares > 0:
                r_squared = 1.0 - residual_sum_squares / total_sum_squares

            degrees_of_freedom = n_valid - 2
            centred_depths = fit_depths - np.mean(fit_depths)
            depth_sum_squares = np.sum(centred_depths ** 2)
            if degrees_of_freedom > 0 and depth_sum_squares > 0:
                residual_variance = residual_sum_squares / degrees_of_freedom
                standard_error = np.sqrt(residual_variance / depth_sum_squares)
                critical_t = student_t.ppf(0.975, degrees_of_freedom)
                ci95_low = raw_value - critical_t * standard_error
                ci95_high = raw_value + critical_t * standard_error

        constraint_active = bool(
            np.isfinite(raw_value) and np.isfinite(kw_value)
            and np.isfinite(constrained_value) and raw_value < kw_value
        )
        if not np.isfinite(constrained_value):
            constrained_source = 'Unavailable'
        elif constraint_active:
            constrained_source = f'Kw reference floor: {kw_model}'
        else:
            constrained_source = 'Profiler raw fit retained'

        # Regression uncertainty transfers only when the constrained product
        # is identical to the raw fit. A reference floor is not a fitted value.
        constrained_uses_raw_fit = np.isfinite(constrained_value) and not constraint_active
        constrained_r_squared = r_squared if constrained_uses_raw_fit else np.nan
        constrained_standard_error = standard_error if constrained_uses_raw_fit else np.nan
        constrained_ci95_low = ci95_low if constrained_uses_raw_fit else np.nan
        constrained_ci95_high = ci95_high if constrained_uses_raw_fit else np.nan

        audit_rows.append({
            'Layer': f'Layer {layer_num}',
            'Depth_min_m': z_min,
            'Depth_max_m': z_max,
            'Wavelength_nm': wavelength,
            'N_valid_raw_fit': n_valid,
            'Kd_raw_m-1': raw_value,
            'R2_raw_fit': r_squared,
            'Kd_raw_standard_error_m-1': standard_error,
            'Kd_raw_CI95_low_m-1': ci95_low,
            'Kd_raw_CI95_high_m-1': ci95_high,
            'One_over_Kd_raw_m': 1.0 / raw_value if np.isfinite(raw_value) and raw_value > 0 else np.nan,
            'Kw_reference_m-1': kw_value,
            'Kw_model': kw_model,
            'Kd_constrained_m-1': constrained_value,
            'One_over_Kd_constrained_m': (
                1.0 / constrained_value
                if np.isfinite(constrained_value) and constrained_value > 0 else np.nan
            ),
            'Kw_constraint_active': constraint_active,
            'Kd_constrained_source': constrained_source,
            'R2_constrained': constrained_r_squared,
            'Kd_constrained_standard_error_m-1': constrained_standard_error,
            'Kd_constrained_CI95_low_m-1': constrained_ci95_low,
            'Kd_constrained_CI95_high_m-1': constrained_ci95_high,
            'Uncertainty_method': (
                'OLS t interval on raw ln(Ed) slope; inherited only when raw fit is retained'
            ),
        })

    return pd.DataFrame(audit_rows)


def calculate_derived_products(
    station_data,
    station_id,
    analysis_layers,
    propagation_irradiance_mode='auto',
    ed0_propagation_mode='raw_fit',
):
    """
    Computes Apparent Optical Properties (AOPs) including spectral Kd, Klu, 
    propagated Rrs(0+), and Kd(PAR) for user-defined depth layers.
    """
    if not analysis_layers or not station_data: return None
    pressure, ed_data, lu_data, wavelengths = station_data['depth_m'], station_data['Ed_data'], station_data['Lu_data'], station_data['wavelengths']
    layer_k_data, layer_kw_constrained_kd, layer_k_fit_audit, results_data = {}, {}, [], []
    rrs_dict, k_dict = {'wavelength_nm': wavelengths}, {'wavelength_nm': wavelengths}
    ed0_dict = {'wavelength_nm': wavelengths}
    rrs_irradiance_audit = []
    ed0_propagation_audit = []
    selected_model = st.session_state.get('selected_kw_model', DEFAULT_KW_MODEL)
    df_kw = get_kw_data(selected_model)
    kw_interp, kw_coverage_mask = interpolate_kw_without_extrapolation(wavelengths, df_kw)
    kw_range_nm = get_kw_model_metadata(selected_model)['range_nm']
    propagation_mask = scientific_product_mask(wavelengths)
    initial_irradiance, initial_source = resolve_surface_irradiance(station_data, 'auto')
    initial_rrs = calculate_Rrs(
        ed_data, lu_data, pressure, surface_irradiance=initial_irradiance
    )
    initial_rrs = np.asarray(initial_rrs, dtype=float)
    initial_rrs[~propagation_mask] = np.nan
    rrs_dict['initial_rrs_sr-1'] = initial_rrs
    rrs_irradiance_audit.append({
        'Rrs source': 'initial_rrs_sr-1',
        'Irradiance mode': 'auto',
        'Irradiance selection': IRRADIANCE_MODE_LABELS['auto'],
        'Resolved irradiance': initial_source,
    })
    
    for layer_num, data in analysis_layers.items():
        z_min, z_max = data['range']
        kd_raw, klu_raw = calculate_k_spectra(pressure, ed_data, lu_data, wavelengths, z_min, z_max)
        
        if klu_raw is not None and kd_raw is not None:
            # The scientific product constrains both profiler-derived Kd and Klu
            # with the selected Kw model, inside the defined 400–700 nm domain.
            kd_profiler = np.full_like(kd_raw, np.nan, dtype=float)
            kd_profiler[propagation_mask] = kd_raw[propagation_mask]
            klu_profiler = np.full_like(klu_raw, np.nan, dtype=float)
            klu_profiler[propagation_mask] = klu_raw[propagation_mask]

            kw_constraint_mask = kw_coverage_mask & propagation_mask
            kd_kw_constrained = np.full_like(kd_raw, np.nan, dtype=float)
            kd_kw_constrained[kw_constraint_mask] = np.maximum(
                kd_raw[kw_constraint_mask], kw_interp[kw_constraint_mask]
            )
            klu_kw_constrained = np.full_like(klu_raw, np.nan, dtype=float)
            klu_kw_constrained[kw_constraint_mask] = np.maximum(
                klu_raw[kw_constraint_mask], kw_interp[kw_constraint_mask]
            )
            layer_k_data[layer_num] = (kd_kw_constrained, klu_kw_constrained)
            layer_kw_constrained_kd[layer_num] = kd_kw_constrained
            layer_k_fit_audit.append(calculate_kd_fit_audit(
                pressure, ed_data, wavelengths, z_min, z_max, layer_num,
                kd_raw, kw_interp, kd_kw_constrained, selected_model,
            ))
            ed0_raw_plus = extrapolate_ed0_minus(
                pressure, ed_data, z_min, z_max,
            ) / ED_TRANSMISSION_FACTOR
            ed0_kw_constrained_plus = extrapolate_ed0_minus(
                pressure, ed_data, z_min, z_max, kd_constraint=kd_kw_constrained,
            ) / ED_TRANSMISSION_FACTOR
            ed0_raw_plus[~propagation_mask] = np.nan
            ed0_kw_constrained_plus[~propagation_mask] = np.nan
            ed0_dict[f'layer_{layer_num}_ed0_plus_raw_fit'] = ed0_raw_plus
            ed0_dict[f'layer_{layer_num}_ed0_plus_kw_constrained'] = ed0_kw_constrained_plus
            ed0_propagation_audit.extend([
                {
                    'Layer': f'L{layer_num}',
                    'Ed(0+) method': ED0_PROPAGATION_MODE_LABELS['raw_fit'],
                    'Kd basis': 'Unconstrained profiler log-linear fit',
                },
                {
                    'Layer': f'L{layer_num}',
                    'Ed(0+) method': ED0_PROPAGATION_MODE_LABELS['kw_constrained'],
                    'Kd basis': f"max(profiler Kd fit, {selected_model} Kw) in 400–700 nm",
                    'Kw support (nm)': f'{kw_range_nm[0]:.0f}–{kw_range_nm[1]:.0f}',
                    'Outside support': 'Not propagated outside 400–700 nm',
                },
            ])
            propagated_irradiance, propagated_source = resolve_surface_irradiance(
                station_data,
                propagation_irradiance_mode,
                z_min,
                z_max,
                kd_constraint=kd_kw_constrained,
                ed0_propagation_mode=ed0_propagation_mode,
            )
            prop_rrs = calculate_Rrs(
                ed_data, lu_data, pressure, kd=kd_kw_constrained, klu=klu_kw_constrained,
                z_top=z_min, z_bottom=z_max, surface_irradiance=propagated_irradiance,
            )
            prop_rrs = np.asarray(prop_rrs, dtype=float)
            prop_rrs[~propagation_mask] = np.nan
            rrs_column = f'propagated_rrs_L{layer_num}_sr-1'
            rrs_dict[rrs_column] = prop_rrs
            rrs_irradiance_audit.append({
                'Rrs source': rrs_column,
                'Irradiance mode': propagation_irradiance_mode,
                'Irradiance selection': IRRADIANCE_MODE_LABELS[propagation_irradiance_mode],
                'Resolved irradiance': propagated_source,
                'Ed(0+) treatment': ED0_PROPAGATION_MODE_LABELS[ed0_propagation_mode],
                'Kd propagation': f"max(profiler Kd fit, {selected_model} Kw), 400–700 nm",
                'Klu propagation': f"max(profiler Klu fit, {selected_model} Kw), 400–700 nm",
            })
            k_dict[f'profiler_kd_raw_L{layer_num}_m-1'] = kd_profiler
            k_dict[f'kw_constrained_kd_L{layer_num}_m-1'] = kd_kw_constrained
            k_dict[f'profiler_klu_raw_L{layer_num}_m-1'] = klu_profiler
            k_dict[f'kw_constrained_klu_L{layer_num}_m-1'] = klu_kw_constrained
            idx_490 = (np.abs(wavelengths - 490)).argmin()
            results_data.append({'Layer':f"Layer {layer_num}", 'Depth Range (m)':f"{z_min}-{z_max}", 'Kd(490)':kd_kw_constrained[idx_490], 'klu(490)':klu_kw_constrained[idx_490]})
        else:
            results_data.append({'Layer':f"Layer {layer_num}", 'Depth Range (m)':f"{z_min}-{z_max}", 'Kd(490)':np.nan, 'klu(490)':np.nan})
    results_df = pd.DataFrame(results_data)
    
    # Calculate PAR using quantum conversion

    h, c, N_A = 6.62607015e-34, 2.99792458e8, 6.02214076e23; par_wl_range = np.arange(400, 701, 1)
    valid_indices = np.where((wavelengths >= 400) & (wavelengths <= 700))[0]
    sub_wl, sub_ed = wavelengths[valid_indices], ed_data[:, valid_indices]
    interp_ed = np.array([np.interp(par_wl_range, sub_wl, row) for row in sub_ed])
    conversion_factor = (par_wl_range * 1e-9) / (h * c) * (1e-6 * 1e4) / N_A * 1e6
    par_profile_profiler = np.sum(interp_ed * conversion_factor, axis=1)
    kd_par_results = calculate_kd_par(pressure, par_profile_profiler, analysis_layers)
    kd_par_df = pd.DataFrame([{'Layer': f"Layer {ln}", 'Kd(PAR)': val} for ln, val in kd_par_results.items()])
    
    # Collect data for the empirical model with context
    if not results_df.empty and not kd_par_df.empty:
        model_data_source = pd.merge(results_df, kd_par_df, on="Layer")
        
        # Explicitly define the column order
        new_model_data = pd.DataFrame({
            'Station_ID': station_id,
            'Layer': model_data_source['Layer'],
            'Kd(PAR)': model_data_source['Kd(PAR)'],
            'Kd(490)': model_data_source['Kd(490)']
        }).dropna()

        if not new_model_data.empty:
            st.session_state.empirical_model_data = pd.concat([st.session_state.empirical_model_data, new_model_data], ignore_index=True)
            st.session_state.empirical_model_data.drop_duplicates(subset=['Station_ID', 'Layer'], keep='last', inplace=True)

    return {
        "layer_k_data": layer_k_data,
        "layer_kw_constrained_kd": layer_kw_constrained_kd,
        "kd_fit_audit": (
            pd.concat(layer_k_fit_audit, ignore_index=True)
            if layer_k_fit_audit else pd.DataFrame()
        ),
        "results_df": results_df,
        "kd_par_df": kd_par_df,
        "rrs_df_export": pd.DataFrame(rrs_dict),
        "k_df_export": pd.DataFrame(k_dict),
        "rrs_irradiance_audit": pd.DataFrame(rrs_irradiance_audit),
        "ed0_df_export": pd.DataFrame(ed0_dict),
        "ed0_propagation_audit": pd.DataFrame(ed0_propagation_audit),
        "kw_model": selected_model,
        "kw_range_nm": kw_range_nm,
    }

def find_n2_peak(depth, temp, sal, lat):
    
    """
   Calculates N2 (Brunt-Väisälä frequency squared) using TEOS-10 
   and finds the depth of the maximum stability peak (pycnocline).
   """
   
    # Removes NaNs and sorts by depth
    
    df = pd.DataFrame({'p': depth, 't': temp, 'sp': sal}).dropna().sort_values('p')
    if len(df) < 5: return None, None, None

    SA = gsw.SA_from_SP(df['sp'], df['p'], -45, lat)
    CT = gsw.CT_from_t(SA, df['t'], df['p'])
    n2, p_mid = gsw.Nsquared(SA, CT, df['p'], lat)
    
    # Suaviza para evitar ruídos
    n2_smooth = pd.Series(n2).values#.rolling(window=3, center=True).mean().values
    
    # Ignores the first 1 meters to avoid surface noise/artifacts
    
    valid_idx = np.where(p_mid > 1.0)[0]
    if len(valid_idx) == 0: return p_mid, n2_smooth, None
    
    peak_idx = valid_idx[np.nanargmax(n2_smooth[valid_idx])]
    return p_mid, n2_smooth, p_mid[peak_idx]

def _handle_secchi_upload():
    
    """Callback function to process the master secchi file upload."""
    
    if st.session_state.master_secchi_loader is None:
        return
    try:
        df_loaded = pd.read_csv(st.session_state.master_secchi_loader)
        expected_cols = ['Station_ID', 'Secchi', 'Kd(PAR)_Profiler', 'Kd(490)_Profiler', 'Kd(PAR)_External']
        if all(col in df_loaded.columns for col in expected_cols):

            # Get a clean copy of the data from the CURRENT session.
            df_session = st.session_state.master_kd_secchi_df.copy()

            # Combine the session data FIRST, then the loaded data.
            combined_df = pd.concat([df_session, df_loaded], ignore_index=True)

            # De-duplicate, keeping the FIRST instance of any Station. This prioritizes any data you've added or updated in the current session.
            combined_df.drop_duplicates(subset=['Station_ID'], keep='first', inplace=True)
            
            st.session_state.master_kd_secchi_df = combined_df
            st.session_state.notification = {'type': 'success', 'message': f"Successfully loaded and merged {len(df_loaded)} master Secchi data points."}
        else:
            st.session_state.notification = {'type': 'error', 'message': "Uploaded CSV is missing required columns."}
    except Exception as e:
        st.session_state.notification = {'type': 'error', 'message': f"Failed to read file: {e}"}

def _handle_model_upload():
    
    """Callback function to process the master model file upload."""
    
    if st.session_state.master_model_loader is None:
        return
    try:
        df_loaded = pd.read_csv(st.session_state.master_model_loader)
        expected_cols = ['Station_ID', 'Layer', 'Kd(PAR)', 'Kd(490)']
        if all(col in df_loaded.columns for col in expected_cols):

            # Get a clean copy of the data from the CURRENT session.
            df_session = st.session_state.empirical_model_data.copy()

            # Combine the session data FIRST, then the loaded data.
            combined_df = pd.concat([df_session, df_loaded], ignore_index=True)

            # De-duplicate by station and layer. Since the session data came first, it is ALWAYS prioritized.
            combined_df.drop_duplicates(subset=['Station_ID', 'Layer'], keep='first', inplace=True)

            st.session_state.empirical_model_data = combined_df
            st.session_state.notification = {'type': 'success', 'message': f"Successfully loaded and merged {len(df_loaded)} model data points."}
        else:
            st.session_state.notification = {'type': 'error', 'message': "Uploaded CSV is missing required columns."}
    except Exception as e:
        st.session_state.notification = {'type': 'error', 'message': f"Failed to read file: {e}"}
        
def generate_summary_report(station_id, analysis_layers, results_df, kd_par_df, wm_profiler=None, wm_external=None):
    """Generates an auditable, source-explicit station summary for the L3 package."""

    def format_coefficient(value):
        return f"{value:.4f} m⁻¹" if pd.notna(value) else "not available"

    report = io.StringIO()
    report.write("============================================================\n")
    report.write("      OCEAN OPTICS EXPLORER - L3 SUMMARY REPORT\n")
    report.write(f"      Station ID: {station_id}\n")
    report.write("============================================================\n\n")

    report.write("--- 1. ANALYSIS CONTEXT AND PROVENANCE ---\n")
    report.write(f"Position: {st.session_state.lat:.5f}, {st.session_state.lon:.5f}\n")
    report.write(
        f"Depth offset applied to profiler: {st.session_state.get('last_applied_offset_str', '0.00')} m\n"
    )
    report_kw_model = st.session_state.get('selected_kw_model', DEFAULT_KW_MODEL)
    report_kw_range = get_kw_model_metadata(report_kw_model)['range_nm']
    report.write(
        f"Pure-water attenuation model: {report_kw_model} "
        f"({report_kw_range[0]:.0f}–{report_kw_range[1]:.0f} nm; Kd/Klu = max(profiler fit, Kw) "
        "within 400–700 nm; no end-point extrapolation)\n"
    )
    report.write(
        "Source convention: Kd(490) in Section 2 is the profiler fit constrained by the selected Kw model; "
        "Klu(490) is constrained by the same Kw model, while Kd(PAR) remains profiler-only. "
        "External CTD/probe values are reported only in Section 3.\n"
    )
    rrs_audit = (
        st.session_state.derived_products.get('rrs_irradiance_audit')
        if st.session_state.get('derived_products') else None
    )
    if rrs_audit is not None and not rrs_audit.empty:
        report.write("Rrs irradiance provenance:\n")
        for _, audit_row in rrs_audit.iterrows():
            ed0_treatment = audit_row.get('Ed(0+) treatment', 'not applicable')
            report.write(
                f"  - {audit_row['Rrs source']}: {audit_row['Resolved irradiance']} "
                f"[{audit_row['Irradiance selection']}; Ed(0+) treatment: {ed0_treatment}]\n"
            )
    report.write("\n")

    report.write("--- 2. PROFILER OPTICAL RESULTS ---\n")
    if results_df is None or results_df.empty:
        report.write("No profiler optical layer results are available.\n")
    else:
        for _, row in results_df.iterrows():
            layer_name = row['Layer']
            try:
                layer_num = int(layer_name.split(' ')[1])
                layer_range = analysis_layers.get(layer_num, {}).get('range') if analysis_layers else None
                if layer_range:
                    report.write(
                        f"{layer_name} (Depth: {layer_range[0]:.1f} m to {layer_range[1]:.1f} m):\n"
                    )
                else:
                    report.write(f"{layer_name} (Depth range not found):\n")
            except (IndexError, ValueError):
                layer_num = None
                report.write(f"{layer_name}:\n")

            report.write(f"  - Kd(490), profiler + Kw constraint: {format_coefficient(row.get('Kd(490)'))}\n")
            report.write(f"  - Klu(490), profiler + Kw constraint: {format_coefficient(row.get('klu(490)'))}\n")
            if kd_par_df is not None and not kd_par_df.empty:
                kpar_values = kd_par_df.loc[kd_par_df['Layer'] == layer_name, 'Kd(PAR)']
                if not kpar_values.empty:
                    report.write(
                        f"  - Profiler Kd(PAR): {format_coefficient(kpar_values.iloc[0])}\n"
                    )

            if layer_num is not None:
                rrs_column = f'propagated_rrs_L{layer_num}_sr-1'
                if (
                    st.session_state.get('derived_products')
                    and rrs_column in st.session_state.derived_products['rrs_df_export'].columns
                ):
                    rrs_v = st.session_state.derived_products['rrs_df_export'][rrs_column].values
                    wls = st.session_state.derived_products['rrs_df_export']['wavelength_nm'].values
                    report.write(f"  - Propagated Rrs AVW: {calculate_avw(wls, rrs_v):.1f} nm\n")
            report.write("\n")

    report.write("--- 3. EXTERNAL CTD / PROBE VALIDATION ---\n")
    comparison_table = st.session_state.get('comparison_table')
    if comparison_table is not None and not comparison_table.empty:
        report.write("CTD profile agreement:\n")
        for cast_id, row in comparison_table.iterrows():
            report.write(
                f"  - {cast_id}: Temperature R²={row.get('Temp R2', np.nan):.4f}, "
                f"RMSE={row.get('Temp RMSE', np.nan):.4f}; "
                f"Salinity R²={row.get('Sal R2', np.nan):.4f}, "
                f"RMSE={row.get('Sal RMSE', np.nan):.4f}\n"
            )
    else:
        report.write("No CTD profile-comparison result is available.\n")

    comparison_kd_df = st.session_state.get('comparison_kd_df')
    if comparison_kd_df is not None and not comparison_kd_df.empty:
        report.write("Profiler-versus-external PAR attenuation:\n")
        for layer_name, row in comparison_kd_df.iterrows():
            profiler_kpar = row.get('Kd(PAR) (Profiler)', np.nan)
            external_kpar = row.get('Kd(PAR) (External)', np.nan)
            report.write(
                f"  - {layer_name}: Profiler Kd(PAR)={format_coefficient(profiler_kpar)}; "
                f"External CTD/probe Kd(PAR)={format_coefficient(external_kpar)}"
            )
            if pd.notna(profiler_kpar) and pd.notna(external_kpar) and profiler_kpar != 0:
                difference_percent = 100 * (external_kpar - profiler_kpar) / profiler_kpar
                report.write(f"; External − Profiler={difference_percent:+.2f}%")
            report.write("\n")
    else:
        report.write("No external PAR attenuation comparison is available.\n")
    report.write("\n")

    report.write("--- 4. SATELLITE CONVOLUTION PROVENANCE ---\n")
    convolution_audit = st.session_state.get('convolution_audit')
    if convolution_audit:
        for label, value in convolution_audit.items():
            report.write(f"  - {label}: {value}\n")
    else:
        report.write("No satellite convolution was run in this session.\n")
    report.write("\n")

    report.write("--- 5. WATER MASS FRACTIONS (AVERAGE %) ---\n")
    report.write("  > Profiler:\n")
    if wm_profiler is not None:
        for name in st.session_state.wm_names:
            if name in wm_profiler.columns:
                report.write(f"    - {name}: {wm_profiler[name].mean() * 100:.1f}%\n")
    else:
        report.write("    (Not calculated)\n")

    report.write("\n  > External CTD/probe:\n")
    if wm_external is not None:
        for name in st.session_state.wm_names:
            if name in wm_external.columns:
                report.write(f"    - {name}: {wm_external[name].mean() * 100:.1f}%\n")
    else:
        report.write("    (Not linked or calculated)\n")

    report.write("\n" + "=" * 60 + "\n")
    return report.getvalue()

def calculate_avw(wavelengths, rrs_spectrum):
    
    """Calculates the Apparent Visible Wavelength (AVW) between 400 and 700 nm."""
    
    mask = (wavelengths >= 400) & (wavelengths <= 700)
    w = wavelengths[mask]
    r = rrs_spectrum[mask]
    valid = ~np.isnan(r)
    if not np.any(valid) or np.nansum(r[valid]) <= 0:
        return np.nan
    return np.nansum(r[valid] * w[valid]) / np.nansum(r[valid])

def wavelength_to_hex(wavelength):
    
    """Maps the calculated AVW to a Hexadecimal color string for UI visualization."""

    if pd.isna(wavelength): return "#808080"
    cmap = plt.get_cmap('turbo')
    norm = plt.Normalize(vmin=400, vmax=700)
    rgba = cmap(norm(wavelength))
    return f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}"

def calculate_physical_properties(_station_data, lon, lat):
    
    """Computes derived physical properties (Salinity, Density, N2) using GSW (TEOS-10)."""
    
    depth_m = _station_data['depth_m']
    temp = _station_data['temperature']
    cond = _station_data['conductivity']
    sea_pressure = gsw.p_from_z(-depth_m, lat)
    salinity = gsw.SP_from_C(cond, temp, sea_pressure)
    SA = gsw.SA_from_SP(salinity, sea_pressure, lon, lat)
    CT = gsw.CT_from_t(SA, temp, sea_pressure)
    n2, p_mid = gsw.Nsquared(SA, CT, sea_pressure)
    n2_padded = np.append(n2, [np.nan] * (len(sea_pressure) - len(n2)))
    p_mid_padded = np.append(p_mid, [np.nan] * (len(sea_pressure) - len(p_mid)))
    depth_mid_padded = np.append(
        -gsw.z_from_p(p_mid, lat), [np.nan] * (len(depth_m) - len(p_mid)),
    )
    return pd.DataFrame({
        'Depth': depth_m,
        'Pressure_dbar': sea_pressure,
        'Salinity': SA,
        'Temperature': CT,
        'n2': n2_padded,
        'Pressure_mid_dbar': p_mid_padded,
        'Depth_mid_m': depth_mid_padded,
    })

def calculate_k_spectra(pressure, ed_data, lu_data, wavelengths, z_min, z_max):
    
    """Calculates the diffuse attenuation coefficients (Kd and Klu) for a specific depth layer."""
    
    indices=np.where((pressure>=z_min)&(pressure<=z_max));
    if len(indices[0]) < 2: return None, None
    depths, ed_range, lu_range=pressure[indices], ed_data[indices], lu_data[indices]; kd, klu=[], []
    for i in range(wavelengths.shape[0]):
        ed_slice, lu_slice=ed_range[:, i], lu_range[:, i]; valid_ed, valid_lu=ed_slice > 0, lu_slice > 0
        kd.append(-np.polyfit(depths[valid_ed], np.log(ed_slice[valid_ed]), 1)[0] if np.sum(valid_ed) >= 2 else np.nan)
        klu.append(-np.polyfit(depths[valid_lu], np.log(lu_slice[valid_lu]), 1)[0] if np.sum(valid_lu) >= 2 else np.nan)
    return np.array(kd), np.array(klu)

def get_profile_metric_data(df, lat=-24.0): # Can change def lat
    
    """
    Returns (Depth, Value, Metric_Name) for plotting and profile alignment.
    Automatic fallback logic:
    1. Provided N2 (Best - from file)
    2. Calculated N2 (If Temp + Sal/Cond are available)
    3. Temperature Gradient (If only Temp is available) -> Thermocline
    """
    
    try:
        df = df.sort_values('Depth')
        
        # CASE 1: N2 PROVIDED IN FILE 

        if 'N2_Provided' in df.columns:
            valid = df.dropna(subset=['Depth', 'N2_Provided'])
            if not valid.empty:
                return valid['Depth'].values, valid['N2_Provided'].values, "N² (Do Arquivo)"

        # CASE 2: CALCULATE N2 (Needs Salinity or Conductivity)
        has_sal = 'Salinity' in df.columns
        has_cond = 'Conductivity' in df.columns
        
        if has_sal or has_cond:
            cols = ['Depth', 'Temperature'] + (['Salinity'] if has_sal else ['Conductivity'])
            valid = df.dropna(subset=cols)
            
            if not valid.empty:
                p = valid['Depth'].values
                t = valid['Temperature'].values
                
                if has_sal:
                    sp = valid['Salinity'].values
                else:
                    
                    # Convert conductivity units if necessary (uS/cm to mS/cm)

                    cond = valid['Conductivity'].values
                    if np.nanmean(cond) > 100: cond = cond / 1000.0
                    sp = gsw.SP_from_C(cond, t, p)
                
                SA = gsw.SA_from_SP(sp, p, -45, lat)
                CT = gsw.CT_from_t(SA, t, p)
                n2, p_mid = gsw.Nsquared(SA, CT, p, lat)
                
                # Light smoothing for visualization purposes
                
                n2_smooth = pd.Series(n2).rolling(window=3, center=True).mean().values
                return p_mid, n2_smooth, "N² (Calculado)"

        # CASE 3: TEMPERATURE ONLY (Thermal Gradient) 
        valid = df.dropna(subset=['Depth', 'Temperature'])
        if not valid.empty:
            p, t = valid['Depth'].values, valid['Temperature'].values
            
            # Calculates the vertical gradient (Peaks at the thermocline)
            
            dz = np.gradient(p)
            dt = np.gradient(t)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                dtdz = np.abs(dt / dz)
            
            dtdz_smooth = pd.Series(dtdz).rolling(window=5, center=True).mean().values
            
            return p, dtdz_smooth, "Thermal Gradient |dT/dz|"

        return None, None, "Error: No data"

    except Exception: return None, None, "Processing Error"
    
    
def extrapolate_ed0_minus(pressure, ed_data, z_min, z_max, kd_constraint=None):
    
    """
    Extrapolates the downwelling irradiance just below the surface, Ed(0-), 
    from a stable, user-defined sub-surface layer using log-linear regression.
    ``kd_constraint`` optionally holds Kd fixed for the Kw-constrained
    Ed(0+) estimate. The routine product constrains both Kd and Klu by Kw
    within the 400–700 nm product domain.
    Returns a 1D numpy array representing the Ed(0-) spectrum.
    """
    
    indices = np.where((pressure >= z_min) & (pressure <= z_max))
    if len(indices[0]) < 2:
        return np.full(ed_data.shape[1], np.nan)

    depths_in_layer = pressure[indices]
    ed_in_layer = ed_data[indices]
    
    ed0_extrapolated = []
    for i in range(ed_data.shape[1]):
        ed_slice = ed_in_layer[:, i]
        valid_mask = (ed_slice > 0) & (~np.isnan(ed_slice))
        
        if np.sum(valid_mask) >= 2:
            log_ed = np.log(ed_slice[valid_mask])
            valid_depths = depths_in_layer[valid_mask]
            constrained_kd = (
                np.asarray(kd_constraint, dtype=float).reshape(-1)[i]
                if kd_constraint is not None and i < np.asarray(kd_constraint).size
                else np.nan
            )
            if np.isfinite(constrained_kd) and constrained_kd >= 0:
                # ln(Ed(z)) = ln(Ed(0-)) - Kd*z. With Kd constrained, fit
                # only the intercept from the observed values in the layer.
                intercept = np.mean(log_ed + constrained_kd * valid_depths)
            else:
                # Raw fit is retained as the observation-based QC reference.
                _, intercept = np.polyfit(valid_depths, log_ed, 1)
            ed0_extrapolated.append(np.exp(intercept))
        else:
            ed0_extrapolated.append(np.nan)
            
    return np.array(ed0_extrapolated)


def get_rrs_surface_irradiance(
    ed_data,
    pressure,
    es_data=None,
    z_top=None,
    z_bottom=None,
    kd_constraint=None,
    ed0_propagation_mode='raw_fit',
    mode='auto',
    es_label='Measured Es',
):
    """Returns the Rrs denominator and its provenance.

    ``mode='auto'`` uses the auditable hierarchy: measured Es, Ed(0+)
    extrapolated from the selected layer, then shallowest valid profiler
    Ed(0+). Explicit modes never silently substitute a different source.
    """
    ed_array = np.asarray(ed_data, dtype=float)
    pressure_array = np.asarray(pressure, dtype=float)
    n_wavelengths = ed_array.shape[1]

    if mode in ('auto', 'measured_es') and es_data is not None:
        try:
            measured_es = np.asarray(es_data, dtype=float).reshape(-1)
            if (
                measured_es.size == n_wavelengths
                and np.any(np.isfinite(measured_es) & (measured_es > 0))
            ):
                return measured_es, es_label
        except (TypeError, ValueError):
            pass

    if mode == 'measured_es':
        return np.full(n_wavelengths, np.nan), f"Requested {es_label} is unavailable"

    if mode in ('auto', 'layer_ed0_plus') and z_top is not None:
        layer_end = z_bottom
        if layer_end is None:
            depths_below_top = pressure_array[pressure_array >= z_top]
            if depths_below_top.size:
                layer_end = np.nanmax(depths_below_top)

        if layer_end is not None:
            ed0_minus = extrapolate_ed0_minus(
                pressure_array,
                ed_array,
                z_top,
                layer_end,
                kd_constraint=(
                    kd_constraint if ed0_propagation_mode == 'kw_constrained' else None
                ),
            )
            ed0_plus = ed0_minus / ED_TRANSMISSION_FACTOR
            if np.any(np.isfinite(ed0_plus) & (ed0_plus > 0)):
                method_label = ED0_PROPAGATION_MODE_LABELS.get(
                    ed0_propagation_mode, ED0_PROPAGATION_MODE_LABELS['raw_fit']
                )
                return ed0_plus, f"Extrapolated Ed(0+) from selected layer ({method_label})"

    if mode == 'layer_ed0_plus':
        return np.full(n_wavelengths, np.nan), "Requested selected-layer Ed(0+) is unavailable"

    if mode in ('auto', 'shallowest_ed0_plus'):
        valid_ed_indices = np.where(np.nanmedian(ed_array, axis=1) > 0)[0]
        if len(valid_ed_indices) > 0:
            return (
                ed_array[valid_ed_indices[0], :] / ED_TRANSMISSION_FACTOR,
                "Shallowest valid profiler Ed converted to Ed(0+)",
            )

    return np.full(n_wavelengths, np.nan), "No valid irradiance for the selected source"

def load_castaway_ctd(uploaded_file):
    
    """
    Reads and parses a processed Castaway CTD file.
    - Features smart column mapping (case-insensitive).
    - Prioritizes 'calc_sal' if 'sal' is empty.
    - Correctly identifies pre-calculated 'N2' from the file.
    """
    
    try:
        # Attempts to read using comma or semicolon as separator
        
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            if df.shape[1] < 2:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=';')
        except Exception:
            return None

        # 1. Standardize column names (lowercase, stripped of whitespaces)

        df.columns = [c.strip().lower() for c in df.columns]

        # 2. Smart Mapping (Priorities). Creates a clean DataFrame to avoid duplicates and standardizes naming
        df_clean = pd.DataFrame()

        # Depth/Pressure
        
        if 'depth' in df.columns: df_clean['Depth'] = df['depth']
        elif 'press' in df.columns: df_clean['Depth'] = df['press']
        elif 'pressure' in df.columns: df_clean['Depth'] = df['pressure']
        else:
            st.error("Coluna de profundidade (depth/press) não encontrada.")
            return None

        # Temperature
        
        if 'temp' in df.columns: df_clean['Temperature'] = df['temp']
        elif 'temperature' in df.columns: df_clean['Temperature'] = df['temperature']
        else:
            st.error("Coluna de temperatura (temp) não encontrada.")
            return None

        #  N2 (From file) 
        if 'n2' in df.columns: 
            df_clean['N2_Provided'] = df['n2']
        elif 'press_n2' in df.columns: # Caso raro
            df_clean['N2_Provided'] = df['press_n2']

        # SALINITY (Prioritizes calculated salinity over raw salinity)
        
        col_sal = None
        if 'calc_sal' in df.columns and df['calc_sal'].count() > 0:
            col_sal = 'calc_sal'
        elif 'sal' in df.columns and df['sal'].count() > 0:
            col_sal = 'sal'
        elif 'salinity' in df.columns:
            col_sal = 'salinity'
            
        if col_sal:
            df_clean['Salinity'] = df[col_sal]
        
        # CONDUCTIVITY (Backup if Salinity is missing)
        if 'cond' in df.columns: df_clean['Conductivity'] = df['cond']
        elif 'conductivity' in df.columns: df_clean['Conductivity'] = df['conductivity']

        # 3. Numeric Cleaning. Corrects commas to dots for floating points and coerces to numeric
        
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                try:
                    df_clean[col] = df_clean[col].str.replace(',', '.', regex=False)
                except:
                    pass
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        # Removes rows where depth is empty
        
        return df_clean.dropna(subset=['Depth'])

    except Exception as e:
        st.error(f"Erro leitura Castaway: {e}")
        return None

def get_profile_n2_data(df, lat=-24.0): #sem smoothing
    
    """
    Returns Depth and N2 vectors for plotting and alignment.
    - If Castaway (External): Uses 'N2_Provided' WITHOUT smoothing (raw file data).
    - If Profiler (Profiler): Calculates N2 using GSW library.
    """
    
    try:
        # Sorts by depth to ensure the plot renders correctly
        df = df.sort_values('Depth')
        
        # CASE 1: CASTAWAY (N2 is already provided)
        if 'N2_Provided' in df.columns:
            
            valid = df.dropna(subset=['Depth', 'N2_Provided'])
            
            if not valid.empty:
                
                # Returns RAW data (matches vendor software output exactly)

                return valid['Depth'].values, valid['N2_Provided'].values

        # CASE 2: PROFILER (Requires computation)
        
        cols_req = ['Depth', 'Temperature']
        if 'Conductivity' in df.columns: cols_req.append('Conductivity')
        elif 'Salinity' in df.columns: cols_req.append('Salinity')
        else: return None, None
        
        df_calc = df.dropna(subset=cols_req)
        if df_calc.empty: return None, None

        p = df_calc['Depth'].values
        t = df_calc['Temperature'].values
        
        if 'Salinity' in df_calc.columns:
            sp = df_calc['Salinity'].values
        else:
            cond = df_calc['Conductivity'].values
            if np.nanmean(cond) > 100: cond = cond / 1000.0 # uS -> mS
            sp = gsw.SP_from_C(cond, t, p)
            
        SA = gsw.SA_from_SP(sp, p, -45, lat)
        CT = gsw.CT_from_t(SA, t, p)
        n2, p_mid = gsw.Nsquared(SA, CT, p, lat)
        
        # Retains light smoothing for calculated data due to numerical noise. Its is commented at the moment
        n2_smooth = pd.Series(n2).values#.rolling(window=3, center=True).mean().values 
        return p_mid, n2_smooth

    except Exception as e:
        print(f"Debug N2 Error: {e}")
        return None, None
    

def calculate_Rrs(
    ed_data,
    lu_data,
    pressure,
    es_data=None,
    kd=None,
    klu=None,
    z_top=None,
    z_bottom=None,
    surface_irradiance=None,
):
  """Calculates the 'Above-Water' Remote Sensing Reflectance (Rrs = Lw / Es),

  with a rigorous denominator hierarchy depending on data availability.
  """
  # STEP 1: Calculate Water-Leaving Radiance, Lw(0+)
  use_propagation = all(param is not None for param in [kd, klu, z_top])

  if use_propagation:
    lu_sub_list = []
    for i in range(lu_data.shape[1]):
      col_data = lu_data[:, i]
      valid_mask = (col_data > 0) & (~np.isnan(col_data))
      if not np.any(valid_mask):
        lu_sub_list.append(np.nan)
        continue
      first_valid_idx = np.argmax(valid_mask)
      z_first_valid, lu_first_valid = (
          pressure[first_valid_idx],
          col_data[first_valid_idx],
      )
      if z_top < z_first_valid:
        z_calc, lu_calc = z_first_valid, lu_first_valid
      else:
        z_calc, lu_calc = z_top, np.interp(z_top, pressure, col_data)
      val = lu_calc * np.exp(klu[i] * z_calc)
      lu_sub_list.append(val)
    lu_sub = np.array(lu_sub_list)
  else:
    valid_lu_indices = np.where(np.nanmedian(lu_data, axis=1) > 0)[0]
    lu_sub = (
        lu_data[valid_lu_indices[0], :]
        if len(valid_lu_indices) > 0
        else np.full(lu_data.shape[1], np.nan)
    )

  # Transmission of Lu(0-) to Lw(0+) across the air-sea interface
  lw_spectrum = lu_sub * LW_TRANSMISSION_FACTOR

  # STEP 2: Determine denominator (measured Es or an Ed(0+) estimate).
  # A pre-resolved spectrum preserves the explicit user selection and is also
  # used by the spectral convolution tab for traceability.
  if surface_irradiance is None:
    denominator, _ = get_rrs_surface_irradiance(
        ed_data,
        pressure,
        es_data=es_data,
        z_top=z_top if use_propagation else None,
        z_bottom=z_bottom if use_propagation else None,
    )
  else:
    denominator = np.asarray(surface_irradiance, dtype=float).reshape(-1)
    if denominator.size != lu_data.shape[1]:
      denominator = np.full(lu_data.shape[1], np.nan)

  # STEP 3: Final Rrs Calculation
  with np.errstate(divide='ignore', invalid='ignore'):
    rrs_spectrum = np.where(
        denominator > 0,
        lw_spectrum / denominator,
        np.nan,
    )

  return rrs_spectrum


def calculate_same_layer_rrs_sensitivity(
    station_data, z_top, z_bottom, kd, klu, es_mode, ed0_propagation_mode,
    ed0_kd_constraint=None,
):
    """Compare Rrs denominators without changing the selected optical layer.

    Both spectra use the same Kw-constrained Kd/Klu and therefore the same
    propagated Lw. Any difference in Rrs is consequently attributable only
    to the Es versus extrapolated Ed(0+) denominator.
    """
    es_spectrum, es_source = resolve_surface_irradiance(
        station_data,
        es_mode,
        z_top=z_top,
        z_bottom=z_bottom,
        kd_constraint=ed0_kd_constraint,
        ed0_propagation_mode=ed0_propagation_mode,
    )
    ed0_plus_spectrum, ed0_plus_source = resolve_surface_irradiance(
        station_data,
        'layer_ed0_plus',
        z_top=z_top,
        z_bottom=z_bottom,
        kd_constraint=ed0_kd_constraint,
        ed0_propagation_mode=ed0_propagation_mode,
    )

    rrs_es = calculate_Rrs(
        station_data['Ed_data'], station_data['Lu_data'], station_data['depth_m'],
        kd=kd, klu=klu, z_top=z_top, z_bottom=z_bottom,
        surface_irradiance=es_spectrum,
    )
    rrs_ed0_plus = calculate_Rrs(
        station_data['Ed_data'], station_data['Lu_data'], station_data['depth_m'],
        kd=kd, klu=klu, z_top=z_top, z_bottom=z_bottom,
        surface_irradiance=ed0_plus_spectrum,
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        relative_difference_percent = np.where(
            np.abs(rrs_es) > 0,
            100 * (rrs_ed0_plus - rrs_es) / rrs_es,
            np.nan,
        )

    return {
        'wavelength_nm': np.asarray(station_data['wavelengths'], dtype=float),
        'rrs_es': rrs_es,
        'rrs_ed0_plus': rrs_ed0_plus,
        'es_spectrum': es_spectrum,
        'ed0_plus_spectrum': ed0_plus_spectrum,
        'relative_difference_percent': relative_difference_percent,
        'es_source': es_source,
        'ed0_plus_source': ed0_plus_source,
    }

def calculate_kd_par(pressure, par_profile, analysis_layers):
    
    """
    Calculates the diffuse attenuation coefficient for PAR, Kd(PAR), 
    using a log-linear fit within defined depth layers.
    """
    
    kd_par_results = {};
    for layer_num, data in analysis_layers.items():
        z_min, z_max = data['range']; layer_indices = np.where((pressure >= z_min) & (pressure <= z_max))[0]
        if len(layer_indices) < 2: kd_par_results[layer_num] = np.nan; continue
        depths_in_layer = pressure[layer_indices]; par_in_layer = par_profile[layer_indices]; valid_par_indices = np.where(par_in_layer > 0)[0]
        if len(valid_par_indices) < 2: kd_par_results[layer_num] = np.nan; continue
        depths_for_fit = depths_in_layer[valid_par_indices]; par_for_fit = par_in_layer[valid_par_indices]; slope = np.polyfit(depths_for_fit, np.log(par_for_fit), 1)[0]; kd_par_results[layer_num] = -slope
    return kd_par_results

def get_kw_model_metadata(model_name):
    """Returns the documented wavelength support for a Kw constraint model."""
    if model_name not in KW_MODEL_METADATA:
        raise ValueError(f"Unsupported Kw model: {model_name}")
    return KW_MODEL_METADATA[model_name]


def interpolate_kw_without_extrapolation(wavelengths, df_kw):
    """Interpolates Kw only inside source support; never holds an end point constant."""
    wavelength_array = np.asarray(wavelengths, dtype=float)
    source_wavelengths = df_kw['wavelength'].to_numpy(dtype=float)
    source_kw = df_kw['Kw'].to_numpy(dtype=float)
    coverage_mask = (
        (wavelength_array >= source_wavelengths.min())
        & (wavelength_array <= source_wavelengths.max())
    )
    interpolated_kw = np.full(wavelength_array.shape, np.nan, dtype=float)
    interpolated_kw[coverage_mask] = np.interp(
        wavelength_array[coverage_mask], source_wavelengths, source_kw
    )
    return interpolated_kw, coverage_mask


def scientific_product_mask(wavelengths):
    """Returns the wavelength domain supported for propagated scientific products."""
    lower_nm, upper_nm = SCIENTIFIC_PRODUCT_WAVELENGTH_RANGE
    wavelength_array = np.asarray(wavelengths, dtype=float)
    return (wavelength_array >= lower_nm) & (wavelength_array <= upper_nm)


def srf_domain_coverage_percent(srf_wavelengths, srf_response):
    """Returns the percentage of SRF area inside the propagated-product domain."""
    wavelengths = np.asarray(srf_wavelengths, dtype=float).reshape(-1)
    response = np.clip(np.asarray(srf_response, dtype=float).reshape(-1), 0, None)
    valid = np.isfinite(wavelengths) & np.isfinite(response)
    if np.sum(valid) < 2:
        return np.nan
    wavelengths, response = wavelengths[valid], response[valid]
    order = np.argsort(wavelengths)
    wavelengths, response = wavelengths[order], response[order]
    total_area = simpson(y=response, x=wavelengths)
    if not np.isfinite(total_area) or total_area <= 0:
        return np.nan

    lower_nm, upper_nm = SCIENTIFIC_PRODUCT_WAVELENGTH_RANGE
    domain_grid = np.unique(np.concatenate([
        wavelengths[(wavelengths >= lower_nm) & (wavelengths <= upper_nm)],
        np.asarray([lower_nm, upper_nm]),
    ]))
    domain_grid = domain_grid[(domain_grid >= wavelengths.min()) & (domain_grid <= wavelengths.max())]
    if domain_grid.size < 2:
        return 0.0
    domain_response = np.interp(domain_grid, wavelengths, response, left=0, right=0)
    domain_area = simpson(y=domain_response, x=domain_grid)
    return 100 * domain_area / total_area


def get_kw_data(model_name):
    """Returns the selected clear-water Kw table without any spectral extension."""
    if "Morel" in model_name:
        # Morel & Maritorena (2001): visible-only table used by this workflow.
        data = {
            'wavelength': [350, 355, 360, 365, 370, 375, 380, 385, 390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515, 520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585, 590, 595, 600, 605, 610, 615, 620, 625, 630, 635, 640, 645, 650, 655, 660, 665, 670, 675, 680, 685, 690, 695, 700],
            'Kw': [0.0271, 0.0238, 0.0216, 0.0188, 0.0177, 0.01595, 0.0151, 0.01376, 0.01271, 0.01208, 0.01042, 0.0089, 0.00812, 0.00765, 0.00758, 0.00768, 0.0077, 0.00792, 0.00885, 0.0099, 0.01148, 0.01182, 0.01188, 0.01211, 0.01251, 0.0132, 0.01444, 0.01526, 0.0166, 0.01885, 0.02188, 0.02701, 0.03385, 0.0409, 0.04214, 0.04287, 0.04454, 0.0463, 0.04846, 0.05212, 0.05746, 0.06053, 0.0628, 0.06507, 0.07034, 0.07801, 0.09038, 0.11076, 0.13584, 0.16792, 0.2231, 0.25838, 0.26506, 0.26843, 0.27612, 0.284, 0.29218, 0.30176, 0.31134, 0.32553, 0.34052, 0.3715, 0.41048, 0.42947, 0.43946, 0.44844, 0.46543, 0.48642, 0.5164, 0.55939, 0.62438],
        }
    else:
        # Smith & Baker (1981), Table I: selected Kw values for clearest
        # ocean waters.  The original 10-nm values are preserved; profiler
        # wavelengths are linearly interpolated only within 300–800 nm.
        data = {
            'wavelength': [300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800],
            'Kw': [0.154, 0.116, 0.0944, 0.0765, 0.0637, 0.0530, 0.0439, 0.0353, 0.0267, 0.0233, 0.0209, 0.0196, 0.0184, 0.0172, 0.0170, 0.0168, 0.0176, 0.0175, 0.0194, 0.0212, 0.0271, 0.0370, 0.0489, 0.0519, 0.0568, 0.0648, 0.0717, 0.0807, 0.109, 0.158, 0.245, 0.290, 0.310, 0.320, 0.330, 0.350, 0.400, 0.430, 0.450, 0.500, 0.650, 0.834, 1.170, 1.800, 2.380, 2.470, 2.550, 2.510, 2.360, 2.160, 2.070],
        }
    return pd.DataFrame(data)

def get_comparison_stats(df_merged, col_name):
    """Calculates R2, RMSE, MAE, and Bias between internal and external sensors."""
    y_true = df_merged[f'{col_name}_external']
    y_pred = df_merged[f'{col_name}_internal']
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    
    if np.sum(mask) < 2: 
        return {'R2': 0.0, 'RMSE': 0.0, 'MAE': 0.0, 'Bias': 0.0}
    
    return {
        'R2': r2_score(y_true[mask], y_pred[mask]),
        'RMSE': np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])),
        'MAE': mean_absolute_error(y_true[mask], y_pred[mask]),
        'Bias': np.mean(y_pred[mask] - y_true[mask]) # Average deviation
    }

def get_regression_metrics(x_data, y_data):
    """Performs linear regression and returns R2, RMSE, MAE, Bias, Slope, Intercept, and P-value."""
    df = pd.DataFrame({'x': x_data, 'y': y_data}).dropna()
    
    if len(df) < 2 or df['x'].nunique() < 2:
        return {'R²': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'Bias': np.nan, 'Slope': np.nan, 'Intercept': np.nan, 'P-value': np.nan}
    
    slope, intercept, r_value, p_value, _ = linregress(df['x'], df['y'])
    y_pred = slope * df['x'] + intercept
    
    rmse = np.sqrt(mean_squared_error(df['y'], y_pred))
    mae = mean_absolute_error(df['y'], y_pred)
    bias = np.mean(y_pred - df['y']) # Calculates bias for the regression model
    
    return {
        'R²': r_value**2,
        'RMSE': rmse,
        'MAE': mae,
        'Bias': bias,
        'Slope': slope,
        'Intercept': intercept,
        'P-value': p_value
    }

def highlight_best(s):
    """Applies green background to the best statistics (Highest R2, Lowest RMSE/MAE/Bias)."""
    if s.name == 'Match Found' or s.dtype == object: 
        return [''] * len(s)
    
    is_max = s == s.max()
    is_min = s == s.min()
    is_min_abs = abs(s) == abs(s).min()

    if 'R2' in s.name:
        return ['background-color: #006400' if v else '' for v in is_max]
    elif 'Bias' in s.name:
        return ['background-color: #006400' if v else '' for v in is_min_abs]
    else: # RMSE and MAE: Lower is better
        return ['background-color: #006400' if v else '' for v in is_min]

def fig_to_buffer(fig):
    
    """Converts matplotlib figures to high-resolution PNG byte buffers for export."""
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf

def compare_ctd_profiles(df_i, df_e):
    
    """
    Interpolates internal and external CTD profiles to a common 0.5m grid 
    to allow direct statistical comparison. No smoothing filters are applied.
    """
    
    profile_columns = ['Depth', 'Temperature', 'Salinity']
    df_i = (
        df_i[profile_columns].dropna()
        .groupby('Depth', as_index=False).mean()
        .sort_values('Depth')
    )
    df_e = (
        df_e[profile_columns].dropna()
        .groupby('Depth', as_index=False).mean()
        .sort_values('Depth')
    )
    if len(df_i) < 2 or len(df_e) < 2:
        return None
    z_min = max(df_i['Depth'].min(), df_e['Depth'].min())
    z_max = min(df_i['Depth'].max(), df_e['Depth'].max())
    if z_min >= z_max: return None
    grid = np.arange(np.ceil(z_min*2)/2, np.floor(z_max*2)/2 + 0.5, 0.5)
    merged = pd.DataFrame({'Depth': grid})
    merged['Temperature_internal'] = np.interp(grid, df_i['Depth'], df_i['Temperature'])
    merged['Salinity_internal'] = np.interp(grid, df_i['Depth'], df_i['Salinity'])
    merged['Temperature_external'] = np.interp(grid, df_e['Depth'], df_e['Temperature'])
    merged['Salinity_external'] = np.interp(grid, df_e['Depth'], df_e['Salinity'])
    return merged.dropna()

def create_full_report_zip(zip_file, base_name):
    
    """Helper function to dump all generated plots and dataframes into a structured ZIP file."""
    
    figs_to_save_from_state = {"1_Stratification_Backscatter": st.session_state.get('fig_n2'), "2_Rrs_Spectra": st.session_state.get('fig_rrs'), "3_Log_Radiance_Profiles": st.session_state.get('fig_log'), "4_Attenuation_Spectra": st.session_state.get('fig_k'), "5_TS_Diagram": st.session_state.get('fig_ts'), "6_TS_Mixture_Depth_Profile": st.session_state.get('fig_mix_depth'), "7_CTD_Comparison": st.session_state.get('fig_comp')}
    for fig_name, fig_obj in figs_to_save_from_state.items():
        if fig_obj: zip_file.writestr(f"{base_name}/Plots/{fig_name}.png", fig_to_buffer(fig_obj).getvalue())
    if st.session_state.get('external_figs'):
        for param_name, fig_obj in st.session_state.external_figs.items(): zip_file.writestr(f"{base_name}/Plots/9_{param_name}_vs_LuEd.png", fig_to_buffer(fig_obj).getvalue())
    if st.session_state.get('results_df') is not None: zip_file.writestr(f"{base_name}/Metrics/K_Metrics_Summary.csv", st.session_state.results_df.to_csv(index=False))
    if st.session_state.get('kd_par_df') is not None: zip_file.writestr(f"{base_name}/Metrics/Kd_PAR_Summary.csv", st.session_state.kd_par_df.to_csv(index=False))
    if st.session_state.get('comparison_table') is not None: zip_file.writestr(f"{base_name}/Metrics/CTD_Comparison_Metrics.csv", st.session_state.comparison_table.to_csv())
    if 'convolved_rrs_df' in st.session_state and st.session_state.convolved_rrs_df is not None:
        sensor_name = st.session_state.get('convolved_sensor_name', 'convolved')
        df_to_save = st.session_state.convolved_rrs_df
        zip_file.writestr(f"{base_name}/L3_Data/Rrs_{sensor_name}.csv", df_to_save.to_csv(index=False))
    if 'mixture_df_profiler' in st.session_state and st.session_state.mixture_df_profiler is not None:
        df_to_save = st.session_state.mixture_df_profiler
        # Adiciona a coluna de profundidade para contexto completo
        df_with_depth = pd.concat([phys_props_df['Depth'], df_to_save], axis=1)
        zip_file.writestr(f"{base_name}/L3_Data/Mixture_Profile_Profiler.csv", df_with_depth.to_csv(index=False))

    if 'mixture_df_external' in st.session_state and st.session_state.mixture_df_external is not None:
        df_to_save = st.session_state.mixture_df_external
        zip_file.writestr(f"{base_name}/L3_Data/Mixture_Profile_External.csv", df_to_save.to_csv(index=False))
        
def get_srf_folder_signature(folder_path="srf_data"):
    """Returns a lightweight fingerprint so newly added local SRFs appear on rerun."""
    if not os.path.isdir(folder_path):
        return ()
    return tuple(
        (entry.name, entry.stat().st_size, entry.stat().st_mtime_ns)
        for entry in sorted(os.scandir(folder_path), key=lambda item: item.name.lower())
        if entry.is_file() and entry.name.lower().endswith((".csv", ".nc", ".nc4"))
    )


def _oci_band_label(band_value, band_index, used_labels):
    """Creates stable, readable column names from the OCI band-coordinate variable."""
    if isinstance(band_value, bytes):
        band_value = band_value.decode(errors="replace")
    try:
        label = f"OCI {float(band_value):.3f} nm"
    except (TypeError, ValueError):
        label = f"OCI band {str(band_value).strip() or band_index + 1}"
    if label in used_labels:
        label = f"{label} ({band_index + 1})"
    used_labels.add(label)
    return label


def _oci_srf_dataframe(wavelengths, bands, responses, source_name):
    """Normalizes the official OCI wavelength, band, and RSR variables to app SRF format."""
    wavelengths = np.asarray(wavelengths, dtype=float).reshape(-1)
    responses = np.asarray(responses, dtype=float)
    bands = np.asarray(bands).reshape(-1) if bands is not None else np.array([])

    if wavelengths.size < 2 or responses.ndim != 2:
        raise ValueError("OCI RSR file must contain a wavelength vector and a two-dimensional RSR matrix.")
    if responses.shape[0] == wavelengths.size:
        responses_by_wavelength = responses
    elif responses.shape[1] == wavelengths.size:
        responses_by_wavelength = responses.T
    else:
        raise ValueError(
            "The OCI RSR matrix does not have an axis matching the wavelength vector."
        )

    valid_wavelengths = np.isfinite(wavelengths)
    wavelengths = wavelengths[valid_wavelengths]
    responses_by_wavelength = responses_by_wavelength[valid_wavelengths, :]
    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    responses_by_wavelength = responses_by_wavelength[order, :]

    n_bands = responses_by_wavelength.shape[1]
    try:
        numeric_bands = np.asarray(bands, dtype=float)
    except (TypeError, ValueError):
        numeric_bands = np.array([])
    band_metadata_is_physical = (
        numeric_bands.size == n_bands
        and np.all(np.isfinite(numeric_bands))
        and np.all((numeric_bands >= 250) & (numeric_bands <= 2600))
    )
    if not band_metadata_is_physical:
        # Some OCI NetCDF files expose only a band index, not a wavelength coordinate.
        # In that case the RSR-weighted centre is the scientifically meaningful label.
        derived_centres = []
        for index in range(n_bands):
            response = responses_by_wavelength[:, index]
            valid_response = np.isfinite(response) & (response > 0)
            if np.sum(valid_response) >= 2:
                derived_centres.append(np.average(
                    wavelengths[valid_response], weights=response[valid_response]
                ))
            else:
                derived_centres.append(np.nan)
        bands = np.asarray(derived_centres)

    srf_columns = {"Wavelength": wavelengths}
    used_labels = set()
    for index, band in enumerate(bands):
        response = responses_by_wavelength[:, index]
        # Negative values are non-physical for a relative spectral response and are
        # treated as invalid rather than being silently integrated.
        if not np.any(np.isfinite(response) & (response > 0)):
            continue
        srf_columns[_oci_band_label(band, index, used_labels)] = response

    if len(srf_columns) == 1:
        raise ValueError(f"No positive OCI RSR bands were found in '{source_name}'.")
    return pd.DataFrame(srf_columns)


def _find_netcdf_variable(container, variable_name):
    """Finds a NetCDF variable by name, including variables held in nested groups."""
    target = variable_name.lower()
    direct_matches = [
        name for name in container.variables
        if name.lower() == target
    ]
    if direct_matches:
        return container.variables[direct_matches[0]]
    for group in container.groups.values():
        found = _find_netcdf_variable(group, variable_name)
        if found is not None:
            return found
    return None


def _find_hdf5_dataset(container, dataset_name, h5py_module):
    """Finds an HDF5 dataset by its final path component, independent of its group."""
    target = dataset_name.lower()
    found = []

    def visitor(path, obj):
        if (
            isinstance(obj, h5py_module.Dataset)
            and path.rsplit('/', 1)[-1].lower() == target
        ):
            found.append(obj)

    container.visititems(visitor)
    return found[0] if found else None


def _find_oci_variable(get_variable, aliases):
    """Returns the first present alias so OCI versions can retain native naming."""
    for alias in aliases:
        variable = get_variable(alias)
        if variable is not None:
            return variable
    return None


def _require_oci_variables(get_variable, source_name):
    """Returns OCI variables or an explicit, actionable file-structure error."""
    wavelengths = _find_oci_variable(
        get_variable, ("wavelengths", "wavelength", "wavelength_nm", "lambda")
    )
    bands = _find_oci_variable(
        get_variable, ("bands", "band", "band_centres", "band_centers", "band_wavelengths")
    )
    responses = _find_oci_variable(
        get_variable, ("RSR", "rsr", "relative_spectral_response")
    )
    required = {"wavelength": wavelengths, "RSR": responses}
    missing = [name for name, variable in required.items() if variable is None]
    if missing:
        raise ValueError(
            f"'{source_name}' is not a supported PACE OCI RSR file; "
            f"missing variable(s): {', '.join(missing)}."
        )
    return wavelengths, bands, responses


def load_oci_srf_netcdf(file_path):
    """Reads a NASA PACE OCI RSR NetCDF file without resampling the official responses."""
    reader_errors = []

    try:
        from netCDF4 import Dataset
        with Dataset(file_path, mode="r") as dataset:
            wavelengths, bands, responses = _require_oci_variables(
                lambda name: _find_netcdf_variable(dataset, name),
                os.path.basename(file_path),
            )
            return _oci_srf_dataframe(
                wavelengths[:], bands[:], responses[:],
                os.path.basename(file_path),
            )
    except ImportError:
        reader_errors.append("netCDF4 is not installed")
    except KeyError as error:
        raise ValueError(
            f"'{os.path.basename(file_path)}' is not a supported PACE OCI RSR file; "
            f"missing variable {error}."
        ) from error
    except OSError as error:
        reader_errors.append(f"netCDF4 could not open the file ({error})")

    # h5py is a valid fallback for these HDF5-backed NetCDF4 RSR files. It makes the
    # importer usable in lightweight scientific environments that do not ship netCDF4.
    try:
        import h5py
        with h5py.File(file_path, mode="r") as dataset:
            wavelengths, bands, responses = _require_oci_variables(
                lambda name: _find_hdf5_dataset(dataset, name, h5py),
                os.path.basename(file_path),
            )
            return _oci_srf_dataframe(
                wavelengths[:], bands[:], responses[:],
                os.path.basename(file_path),
            )
    except ImportError:
        reader_errors.append("h5py is not installed")
    except KeyError as error:
        raise ValueError(
            f"'{os.path.basename(file_path)}' is not a supported PACE OCI RSR file; "
            f"missing variable {error}."
        ) from error
    except OSError as error:
        reader_errors.append(f"h5py could not open the file ({error})")

    # PyTables is another HDF5 reader that may already be present in a scientific
    # environment. It reads the three root datasets directly, with no resampling.
    try:
        import tables
        with tables.open_file(file_path, mode="r") as dataset:
            return _oci_srf_dataframe(
                dataset.root.wavelengths.read(), dataset.root.bands.read(), dataset.root.RSR.read(),
                os.path.basename(file_path),
            )
    except ImportError:
        reader_errors.append("tables is not installed")
    except tables.NoSuchNodeError as error:
        raise ValueError(
            f"'{os.path.basename(file_path)}' is not a supported PACE OCI RSR file; "
            f"missing variable {error}."
        ) from error
    except OSError as error:
        reader_errors.append(f"tables could not open the file ({error})")

    raise ImportError(
        "PACE OCI .nc support requires netCDF4 (recommended), h5py, or tables. "
        f"Reader details: {'; '.join(reader_errors)}."
    )


@st.cache_data
def load_srf_from_folder(folder_path="srf_data", folder_signature=()):
    """Loads CSV SRFs and official PACE OCI NetCDF RSRs from the local SRF directory."""
    del folder_signature  # Included only to invalidate the Streamlit cache when files change.
    srf_data = {}
    if not os.path.isdir(folder_path):
        return srf_data

    for filename in sorted(os.listdir(folder_path), key=str.lower):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue
        suffix = os.path.splitext(filename)[1].lower()
        sensor_name = os.path.splitext(filename)[0]
        try:
            if suffix == ".csv":
                srf_data[sensor_name] = pd.read_csv(file_path)
            elif suffix in {".nc", ".nc4"}:
                srf_data[f"{sensor_name} (PACE OCI)"] = load_oci_srf_netcdf(file_path)
        except (ImportError, OSError, ValueError, pd.errors.ParserError) as error:
            st.warning(f"[{OCI_SRF_READER_VERSION}] Could not load SRF '{filename}': {error}")
    return srf_data
        
def calculate_external_kd_par(df_external, depth_col, par_col, z_min, z_max):
    
    """Calculates Kd(PAR) from an external probe's dataframe within a specific depth layer."""
    
    if df_external is None or depth_col not in df_external.columns or par_col not in df_external.columns: return np.nan
    df_external[depth_col] = pd.to_numeric(df_external[depth_col], errors='coerce'); df_external[par_col] = pd.to_numeric(df_external[par_col], errors='coerce'); df_layer = df_external.dropna(subset=[depth_col, par_col])
    df_layer = df_layer[(df_layer[depth_col] >= z_min) & (df_layer[depth_col] <= z_max)]; df_layer = df_layer[df_layer[par_col] > 0]
    if len(df_layer) < 2: return np.nan
    depths_for_fit = df_layer[depth_col]; par_for_fit = df_layer[par_col]; slope = np.polyfit(depths_for_fit, np.log(par_for_fit), 1)[0]
    return -slope

def estimate_chlorophyll_from_kdpar(kd_par):
    
    """
    Empirical Case-1 water model to estimate Chlorophyll-a from Kd(PAR).
    Subtracts the pure water contribution (0.022 m^-1) before calculation.
    """
    
    if kd_par <= 0.022: return 0.0
    kd_bio = kd_par - 0.022; a, b = 0.0512, 0.428; chlorophyll = (kd_bio / a)**(1 / b)
    return chlorophyll

def estimate_kd490_from_chlorophyll(chlorophyll):
    
    """
     Morel & Maritorena, 2001, to estimate 
    Kd(490) based on Chlorophyll-a concentration.
    """
    
    kw_490 = 0.0166; x = 0.07917; e = 0.7895; kd_490 = kw_490 + x * (chlorophyll**e)
    return kd_490


# ----------------------------- DATA UPLOAD HANDLERS -----------------------------

def _handle_master_rrs_upload():
    
    """Callback function to process the master Rrs file upload."""
    
    if st.session_state.master_rrs_loader is None:
        return
    try:
        # Read the uploaded CSV into a pandas DataFrame
        df_loaded = pd.read_csv(st.session_state.master_rrs_loader)
        
        # Store the DataFrame in the session state
        st.session_state.master_rrs_df = df_loaded
        st.toast(f"Successfully loaded {len(df_loaded.columns) - 1} historical Rrs spectra!")
    except Exception as e:
        st.toast(f"Failed to read master Rrs file: {e}")
        
def _handle_rad_upload():
    if st.session_state.up_rad_m:
        df = pd.read_csv(st.session_state.up_rad_m)
        st.session_state.master_vertical_radiometry = pd.concat([st.session_state.master_vertical_radiometry, df], ignore_index=True).drop_duplicates(subset=['Station_ID', 'Depth'], keep='last')
        st.toast("Radiometria mesclada!")

def _handle_anc_upload():
    if st.session_state.up_anc_m:
        df = pd.read_csv(st.session_state.up_anc_m)
        st.session_state.master_vertical_ancillary = pd.concat([st.session_state.master_vertical_ancillary, df], ignore_index=True).drop_duplicates(subset=['Station_ID', 'Depth'], keep='last')
        st.toast("Ancillary mesclada!")

def _handle_hyper_upload():
    if st.session_state.up_hyper_m:
        df = pd.read_csv(st.session_state.up_hyper_m)
        st.session_state.master_hyper_rrs_df = pd.concat([st.session_state.master_hyper_rrs_df, df], ignore_index=True).drop_duplicates(subset=['Station_ID'], keep='last')
        st.toast("Hyper Rrs mesclado!")

def _handle_multi_upload():
    if st.session_state.up_multi_m:
        df = pd.read_csv(st.session_state.up_multi_m)
        st.session_state.master_multi_rrs_df = pd.concat([st.session_state.master_multi_rrs_df, df], ignore_index=True).drop_duplicates(subset=['Station_ID', 'Sensor'], keep='last')
        st.toast("Multispectral mesclado!")        
        
def _handle_master_secchi_upload():
    """Callback to load and merge the Master Secchi/Kd database file."""
    if st.session_state.master_secchi_loader is not None:
        try:
            df_loaded = pd.read_csv(st.session_state.master_secchi_loader)
            expected_cols = ['Station_ID', 'Secchi', 'Kd(PAR)_Profiler', 'Kd(490)_Profiler', 'Kd(PAR)_External']
            
            if all(col in df_loaded.columns for col in expected_cols):
                # Merges with existing session data, prioritizing newly loaded data
                combined_df = pd.concat([st.session_state.master_kd_secchi_df, df_loaded], ignore_index=True)
                # Removes duplicates based on Station_ID (keeps the latest instance)
                combined_df.drop_duplicates(subset=['Station_ID'], keep='last', inplace=True)
                
                st.session_state.master_kd_secchi_df = combined_df
                st.toast("Master Secchi database successfully merged!")
            else:
                st.error("Uploaded CSV is missing the required columns.")
        except Exception as e:
            st.error(f"Failed to read the master file: {e}")
            
# ----------------------------- PLOTTING AND UTILITY FUNCTIONS -----------------------------

def plot_kd_secchi_relationship(df):
    
    """Generates comparative scatter plots and regression metrics for Secchi vs Kd."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Secchi vs Kd(PAR) (Comparative) 
    ax1.set_title("Secchi Depth vs. Kd(PAR)", weight='bold')
    ax1.set_xlabel("Secchi Depth (m)", weight='bold')
    ax1.set_ylabel("Kd(PAR) (m⁻¹)", weight='bold')

    df_par_prof = df[['Secchi', 'Kd(PAR)_Profiler']].dropna()
    df_par_ext = df[['Secchi', 'Kd(PAR)_External']].dropna()
    df_490_prof = df[['Secchi', 'Kd(490)_Profiler']].dropna()

    # Plot Profiler Data
    if not df_par_prof.empty:
        ax1.scatter(df_par_prof['Secchi'], df_par_prof['Kd(PAR)_Profiler'], ec='white', s=60, marker='o', label='Profiler', zorder=10)
        if len(df_par_prof) >= 2:
            metrics_prof = get_regression_metrics(df_par_prof['Secchi'], df_par_prof['Kd(PAR)_Profiler'])
            if not pd.isna(metrics_prof['R²']):
                x_vals = np.array(df_par_prof['Secchi'])
                ax1.plot(x_vals, metrics_prof['Intercept'] + metrics_prof['Slope'] * x_vals, '--', color='cyan', label=f'Profiler R² = {metrics_prof["R²"]:.3f}')

    # Plot External Probe Data
    if not df_par_ext.empty:
        ax1.scatter(df_par_ext['Secchi'], df_par_ext['Kd(PAR)_External'], ec='white', s=60, marker='^', label='External Probe', zorder=10)
        if len(df_par_ext) >= 2:
            metrics_ext = get_regression_metrics(df_par_ext['Secchi'], df_par_ext['Kd(PAR)_External'])
            if not pd.isna(metrics_ext['R²']):
                x_vals = np.array(df_par_ext['Secchi'])
                ax1.plot(x_vals, metrics_ext['Intercept'] + metrics_ext['Slope'] * x_vals, ':', color='yellow', label=f'External R² = {metrics_ext["R²"]:.3f}')
    
    ax1.grid(True, linestyle=':'); ax1.legend()

    # Plot 2: Secchi vs Kd(490) (Profiler Only)
    ax2.set_xlabel("Secchi Depth (m)", weight='bold')
    ax2.set_ylabel("Kd(490) (m⁻¹)", weight='bold')
    ax2.set_title("Secchi Depth vs. Kd(490) (Profiler)", weight='bold')
    
    if not df_490_prof.empty:
        ax2.scatter(df_490_prof['Secchi'], df_490_prof['Kd(490)_Profiler'], ec='white', s=50, zorder=10)
        if len(df_490_prof) >= 2:
            metrics_490 = get_regression_metrics(df_490_prof['Secchi'], df_490_prof['Kd(490)_Profiler'])
            if not pd.isna(metrics_490['R²']):
                x_min, x_max = df_490_prof['Secchi'].min(), df_490_prof['Secchi'].max()
                x_vals = np.array([x_min, x_max]) if x_min != x_max else np.array([x_min - 1, x_max + 1])  # Handles identical x-values
                ax2.plot(x_vals, metrics_490['Intercept'] + metrics_490['Slope'] * x_vals, '--', color='lime', label=f'R² = {metrics_490["R²"]:.3f}')
    
    ax2.grid(True, linestyle=':'); ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)

    # Display All Regression Metrics 
    st.markdown("**Regression Metrics**")
    
    # Only calculate metrics if there is enough data 
    if len(df_par_prof) >= 2:
        metrics_prof_par = get_regression_metrics(df_par_prof['Secchi'], df_par_prof['Kd(PAR)_Profiler'])
    else:
        metrics_prof_par = {'R²': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'Slope': np.nan, 'Intercept': np.nan, 'P-value': np.nan}

    if len(df_par_ext) >= 2:
        metrics_ext_par = get_regression_metrics(df_par_ext['Secchi'], df_par_ext['Kd(PAR)_External'])
    else:
        metrics_ext_par = {'R²': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'Slope': np.nan, 'Intercept': np.nan, 'P-value': np.nan}
        
    if len(df_490_prof) >= 2:
        metrics_prof_490 = get_regression_metrics(df_490_prof['Secchi'], df_490_prof['Kd(490)_Profiler'])
    else:
        metrics_prof_490 = {'R²': np.nan, 'RMSE': np.nan, 'Slope': np.nan, 'Intercept': np.nan, 'P-value': np.nan}
    
    df_metrics = pd.DataFrame([metrics_prof_par, metrics_ext_par, metrics_prof_490], 
                              index=['Secchi vs. Kd(PAR) (Profiler)', 
                                     'Secchi vs. Kd(PAR) (External)', 
                                     'Secchi vs. Kd(490) (Profiler)']).T
    st.dataframe(df_metrics.style.format('{:.4f}'), use_container_width=True)
    
def find_best_match(target_id, candidates):
    
    """
    Ultra-permissive fuzzy matching logic to link profiler station IDs 
    to external probe station IDs (e.g., links 'p1' to 'perfil_p1' or 'p1a').
    """
    
    if not target_id or candidates is None or len(candidates) == 0: return None
    t = str(target_id).lower().strip()
    c_list = [str(c) for c in candidates if str(c) != "nan"]
    
    # 1. Tries exact match first
    for c in c_list:
        if c.lower().strip() == t: return c
    
    # 2. Tries substring match
    for c in c_list:
        clean_c = c.lower().strip()
        if t in clean_c or clean_c in t: return c
        
    # 3. Fallback: If there's only one station in the external file, assume it's the right one
    if len(c_list) == 1: return c_list[0]
    
    return None


def plot_interactive_n2_offset(depth_ref, n2_ref, depth_target, n2_target, applied_offset, xlabel="Métrica", normalize=False):
    
    """
    Generates an interactive Plotly graph for visual alignment of stratification peaks.
    Layering order: Simulated (Background) -> Original (Middle) -> Reference (Top).
    """
    
    fig = go.Figure()

    # NORMALIZATION LOGIC
    if normalize:
        # Ref
        rmin, rmax = np.nanmin(n2_ref), np.nanmax(n2_ref)
        if rmax > rmin: n2_ref = (n2_ref - rmin) / (rmax - rmin)
        
        # Target
        tmin, tmax = np.nanmin(n2_target), np.nanmax(n2_target)
        if tmax > tmin: n2_target = (n2_target - tmin) / (tmax - tmin)
        
        xlabel = "Escala Normalizada (0 a 1)"

    #LAYER 1 (BACKGROUND): Simulated Profiler (Thick Transparent Blue)
    depth_corrected = depth_target + applied_offset
    fig.add_trace(go.Scatter(
        x=n2_target, y=depth_corrected, mode='lines', name='Simulated Offset (Blue)',
        line=dict(color='#00B4D8', width=6), # Bem grosso
        opacity=0.4, # Bem transparente
        hovertemplate='<b>Simulado</b><br>Prof: %{y:.2f}m<br>Val: %{x:.2f}<extra></extra>'
    ))

   # LAYER 2 (MIDDLE): Original Profiler (Solid Thin Red)
    fig.add_trace(go.Scatter(
        x=n2_target, y=depth_target, mode='lines', name='Original (Red)',
        line=dict(color='#FF4B4B', width=2, dash='solid'), # Sólido para ver melhor
        opacity=1.0, 
        hovertemplate='<b>Orig</b><br>Prof: %{y:.2f}m<br>Val: %{x:.2f}<extra></extra>'
    ))

    # LAYER 3 (TOP): External Reference (Green)
    fig.add_trace(go.Scatter(
        x=n2_ref, y=depth_ref, mode='lines', name='External Ref (Green)',
        line=dict(color='#00FF00', width=2),
        hovertemplate='<b>Ref</b><br>Prof: %{y:.2f}m<br>Val: %{x:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"<b>Ajuste Fino Interativo</b> (Current Offset: {applied_offset:+.2f}m)",
        xaxis_title=xlabel,
        yaxis_title="Profundidade (m)",
        # Autorange reversed 0 at the top, allowing negative depths if existent
        yaxis=dict(autorange="reversed"), 
        height=600,
        hovermode="y unified",
        template="plotly_dark",
        legend=dict(x=0.05, y=0.05, bgcolor='rgba(0,0,0,0.5)')
    )
    return fig

def get_col_index(columns, session_key):
    """Returns the index of a saved column in session_state to persist widget selections."""
    if session_key in st.session_state:
        val = st.session_state[session_key]
        if val in columns:
            return list(columns).index(val)
    return 0


def aggregate_to_vertical_masters(station_id, station_data, derived_products):
    """
    Aggregates processed data for the global campaign database, including 
    Radiometry, PAR, Chlorophyll, CDOM, and Water Mass fractions.
    """
    depths, wavelengths = station_data['depth_m'], station_data['wavelengths']
    
    # 1. MASTER VERTICAL RADIOMETRY (Ed, Lu, and apparent Lu/Ed ratio)
    
    rad_df = pd.DataFrame({'Station_ID': station_id, 'Depth': depths})
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(station_data['Ed_data'] > 0, station_data['Lu_data'] / station_data['Ed_data'], np.nan)
    for i, wl in enumerate(wavelengths):
        w = int(wl)
        rad_df[f'Ed_{w}'] = station_data['Ed_data'][:, i]
        rad_df[f'Lu_{w}'] = station_data['Lu_data'][:, i]
        rad_df[f'LuEd_{w}'] = ratio[:, i]
    
    st.session_state.master_vertical_radiometry = pd.concat([
        st.session_state.master_vertical_radiometry[st.session_state.master_vertical_radiometry['Station_ID'] != station_id], 
        rad_df], ignore_index=True, sort=False)

    # 2. MASTER ANCILLARY DATA (PAR, Chl, CDOM, WM)
    anc_df = pd.DataFrame({'Station_ID': station_id, 'Depth': depths})
    
    # A. Profiler PAR (Calculated Quantum Flux)
    h_c, c_c, Na_c = 6.626e-34, 3e8, 6.022e23
    p_idx = np.where((wavelengths >= 400) & (wavelengths <= 700))[0]
    ed_q = station_data['Ed_data'][:, p_idx] * ((wavelengths[p_idx]*1e-9)/(h_c*c_c)*(1e-6*1e4)/Na_c*1e6)
    anc_df['PAR_Profiler'] = np.sum(ed_q, axis=1)

    # Initializes external columns with NaN
    anc_df['PAR_External'] = np.nan
    anc_df['Chl_External'] = np.nan
    anc_df['CDOM_External'] = np.nan

    # B. External Probe 1 Data (PAR and Chlorophyll)
    if st.session_state.external_probe_df is not None:
        df_p1 = st.session_state.external_probe_df
        st1_col = st.session_state.get('p1_st')
        z1_col = st.session_state.get('p1_z')
        par_col = st.session_state.get('p1_par')
        chl_col = st.session_state.get('p1_chl')

        if st1_col != "None" and z1_col != "None":
            # Finds corresponding station
            match1 = find_best_match(station_id, df_p1[st1_col].dropna().unique())
            if match1:
                df_s1 = df_p1[df_p1[st1_col].astype(str) == str(match1)]
                z_ext1 = pd.to_numeric(df_s1[z1_col], errors='coerce')
                
                # Interpolates PAR
                if par_col != "None" and par_col in df_s1:
                    anc_df['PAR_External'] = np.interp(depths, z_ext1, pd.to_numeric(df_s1[par_col], errors='coerce'), left=np.nan, right=np.nan)
                
                # Interpolates Chl 
                if chl_col != "None" and chl_col in df_s1:
                    anc_df['Chl_External'] = np.interp(depths, z_ext1, pd.to_numeric(df_s1[chl_col], errors='coerce'), left=np.nan, right=np.nan)

     # C. External Probe 2 Data (CDOM / Fluorimeter)
    if st.session_state.external_bio_df is not None:
        df_p2 = st.session_state.external_bio_df
        st2_col = st.session_state.get('p2_st')
        z2_col = st.session_state.get('p2_z')
        cdom_col = st.session_state.get('p2_cdom')
        p2_vertical_coordinate = st.session_state.get('p2_vertical_coordinate', 'depth_m')

        if st2_col != "None" and z2_col != "None" and cdom_col != "None":
            match2 = find_best_match(station_id, df_p2[st2_col].dropna().unique())
            if match2:
                df_s2 = df_p2[df_p2[st2_col].astype(str) == str(match2)]
                cdom_profile = pd.DataFrame({
                    'Depth': _to_depth_m(
                        df_s2[z2_col], p2_vertical_coordinate, st.session_state.lat,
                    ),
                    'CDOM': pd.to_numeric(df_s2[cdom_col], errors='coerce'),
                }).replace([np.inf, -np.inf], np.nan).dropna()

                # Duplicate source levels are consolidated only after conversion to Depth (m).
                cdom_profile = (
                    cdom_profile.groupby('Depth', as_index=False)['CDOM']
                    .mean()
                    .sort_values('Depth')
                )

                # Interpolates CDOM exclusively against the final Depth (m) coordinate.
                if len(cdom_profile) >= 2:
                    anc_df['CDOM_External'] = np.interp(
                        depths,
                        cdom_profile['Depth'],
                        cdom_profile['CDOM'],
                        left=np.nan,
                        right=np.nan,
                    )

    # D. Water Mass Fractions (WM)
    for n in st.session_state.wm_names:
        # From Profiler
        anc_df[f'WM_{n}_Profiler'] = st.session_state.mixture_df_profiler[n].values if st.session_state.mixture_df_profiler is not None else np.nan
        
        # From External Probe
        anc_df[f'WM_{n}_External'] = np.nan
        if st.session_state.mixture_df_external is not None and st.session_state.external_probe_df is not None:
            
            if st1_col != "None" and z1_col != "None":
                 match_mix = find_best_match(station_id, st.session_state.external_probe_df[st1_col].dropna().unique())
                 if match_mix:
                     df_mix_src = st.session_state.external_probe_df[st.session_state.external_probe_df[st1_col].astype(str) == str(match_mix)]
                     # Requires Temperature and Salinity to exist (same as Tab 4 logic)
                     df_mix_src = df_mix_src.dropna(subset=[st.session_state.p1_t, st.session_state.p1_s])
                     z_mix = pd.to_numeric(df_mix_src[z1_col], errors='coerce')
                     
                     if len(z_mix) == len(st.session_state.mixture_df_external):
                         anc_df[f'WM_{n}_External'] = np.interp(depths, z_mix, st.session_state.mixture_df_external[n], left=np.nan, right=np.nan)

    st.session_state.master_vertical_ancillary = pd.concat([
        st.session_state.master_vertical_ancillary[st.session_state.master_vertical_ancillary['Station_ID'] != station_id], 
        anc_df], ignore_index=True, sort=False)
    
    # 3. MASTER HYPERSPECTRAL RRS SNAPSHOT (L1 Propagated)
    if derived_products:
        rrs_df = derived_products['rrs_df_export']
        # Identifies the highest priority Rrs column (Propagated L1 > Propagated > Initial)

        target_col = next((c for c in rrs_df.columns if 'propagated_rrs_L1' in c), 
                          next((c for c in rrs_df.columns if 'propagated' in c), 'initial_rrs_sr-1'))
        rrs_snap = pd.DataFrame([rrs_df[target_col].values], columns=[f"Rrs_{int(w)}" for w in rrs_df['wavelength_nm']])
        rrs_snap.insert(0, 'Station_ID', station_id)
        st.session_state.master_hyper_rrs_df = pd.concat([st.session_state.master_hyper_rrs_df[st.session_state.master_hyper_rrs_df['Station_ID'] != station_id], rrs_snap], ignore_index=True, sort=False)

    st.toast(f"Snapshot of '{station_id}' successfully saved to master database!")

# ---------------------------- 1. SESSION STATE INITIALIZATION ----------------------------
keys_to_init = {
    'station_data': None, 
    'station_list': [], 
    'selected_station': None, 
    'lat': -24.101879, # Can be changed so a new default lat is set
    'lon': -45.692597, # Can be changed so a new default lon is set
    'profile_specific_layers': {},
    'derived_products': None,
    'notification': None,
    'discarded_indices': [],
    'new_es_median': None,
    'convolved_rrs_df': None,
    'convolved_sensor_name': None,
    'convolved_rrs_source': None,
    'convolved_rrs_hyperspectral': None,
    'convolution_audit': None,
    'convolution_history': [],
    'convolution_method_version': None,
    'kw_policy_version': None,
    'wm_names': ["AT", "ACAS", "AC"], # Can be changed so new default water masses are set
    'wm_vertices': None,
    
    # EXTERNAL PROBES
    'external_probe_df': None,
    'external_bio_df': None, 
    
    # MIXTURE 
    'mixture_df_profiler': None,
    'mixture_df_external': None,
    'mixture_results_profiler': None,
    'mixture_results_external': None,
    
    'uploader_id': 0,
    
    # MASTER TABLES
    'master_vertical_radiometry': pd.DataFrame(columns=['Station_ID', 'Depth']),
    'master_vertical_ancillary': pd.DataFrame(columns=['Station_ID', 'Depth']),
    'master_hyper_rrs_df': pd.DataFrame(columns=['Station_ID']),
    'master_multi_rrs_df': pd.DataFrame(columns=['Station_ID', 'Sensor']),
    'master_kd_secchi_df': pd.DataFrame(columns=['Station_ID', 'Secchi', 'Kd(PAR)_Profiler', 'Kd(490)_Profiler', 'Kd(PAR)_External']),
    'empirical_model_data': pd.DataFrame(columns=['Station_ID', 'Layer', 'Kd(PAR)', 'Kd(490)']), # Importante para a Tab 5
    
    # FIGURES 
    'fig_vp': None, 'fig_ve': None, 'fig_ts': None, 'fig_n2': None, 
    'fig_rrs': None, 'fig_log': None, 'fig_k': None, 'fig_comp': None,
    
    # SELECTED COLUMNS (Prevent key errors in Tab 1)
    'p1_st': "None", 'p1_z': "None", 'p1_t': "None", 'p1_s': "None",
    'p1_vertical_coordinate': 'depth_m',
    'p1_chl': "None", 'p1_sec': "None", 'p1_par': "None",
    'p2_st': "None", 'p2_z': "None", 'p2_cdom': "None",
    'p2_vertical_coordinate': 'depth_m',
}

for key, value in keys_to_init.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Do not mix runs made with the previous denominator-reconstruction method
# with the current, physically consistent Rrs sensitivity comparison.
if st.session_state.convolution_method_version != CONVOLUTION_METHOD_VERSION:
    st.session_state.convolved_rrs_df = None
    st.session_state.convolved_sensor_name = None
    st.session_state.convolved_rrs_source = None
    st.session_state.convolved_rrs_hyperspectral = None
    st.session_state.convolution_audit = None
    st.session_state.convolution_history = []
    st.session_state.convolution_method_version = CONVOLUTION_METHOD_VERSION

# Apply the visible-only policy once, without overwriting a later deliberate
# user choice between the two reference Kw curves.
if st.session_state.kw_policy_version != CONVOLUTION_METHOD_VERSION:
    st.session_state.selected_kw_model = DEFAULT_KW_MODEL
    st.session_state.kw_policy_version = CONVOLUTION_METHOD_VERSION

# Ghost column cleanup to keep Master Ancillary organized
cols_limpeza = ['PAR_Profiler_Profiler', 'PAR_External_Probe', 'PAR_External_Probe_umol_m2_s', 'PAR_External_umol_m2_s']
for col in cols_limpeza:
    if col in st.session_state.master_vertical_ancillary.columns:
        st.session_state.master_vertical_ancillary = st.session_state.master_vertical_ancillary.drop(columns=[col])
        
        
        
# 5. UI: SIDEBAR
with st.sidebar:
    st.title("Ocean Optics Explorer")
    st.header("1. Load Data")
    st.session_state.lat = st.number_input(
        "Station latitude (°N)", min_value=-90.0, max_value=90.0,
        value=float(st.session_state.lat), format="%.5f",
        help="Used to calculate Depth (m) accurately when the input file records pressure.",
    )
    st.session_state.lon = st.number_input(
        "Station longitude (°E)", min_value=-180.0, max_value=180.0,
        value=float(st.session_state.lon), format="%.5f",
    )
    profiler_vertical_coordinate = st.selectbox(
        "How is depth recorded in the profiler file?",
        options=list(VERTICAL_COORDINATE_LABELS),
        format_func=lambda coordinate: VERTICAL_COORDINATE_LABELS[coordinate],
        help=(
            "Choose the format of the source column. After upload, the app stores and displays "
            "the vertical coordinate exclusively as Depth (m)."
        ),
        key="profiler_vertical_coordinate",
    )
    if profiler_vertical_coordinate == 'absolute_pressure_dbar':
        st.caption(
            "Use this option only when the instrument documentation confirms an absolute-pressure scale. "
            "The resulting vertical coordinate remains Depth (m)."
        )
    elif profiler_vertical_coordinate == 'sea_pressure_dbar':
        st.caption("After processing, the vertical coordinate is stored and displayed as Depth (m).")
    else:
        st.caption("Depth (m) is retained as the final coordinate for all profiles, plots, and optical products.")
    profiler_csv_left, profiler_csv_right = st.columns(2)
    profiler_separator = profiler_csv_left.radio(
        "Profiler separator", [',', ';'], horizontal=True, key="profiler_separator",
    )
    profiler_decimal = profiler_csv_right.radio(
        "Profiler decimal", ['.', ','], horizontal=True, key="profiler_decimal",
    )
    uploaded_files = st.file_uploader(
        "Upload In-Water Profiler Files (Temp, Cond, Ed, Lu)",
        accept_multiple_files=True, type="csv",
    )
    uploaded_beta_file = st.file_uploader(
        "Upload Backscattering 'Beta' File (Optional)", type="csv", key="beta_uploader",
    )
    uploaded_es_file = st.file_uploader(
        "Upload Es CSV File (Optional)", type="csv", key="es_uploader",
    )
    if uploaded_files and st.button("Process Files"):
        with st.spinner("Processing data..."):
            st.session_state.station_data, st.session_state.station_list = process_uploaded_files(
                uploaded_files, uploaded_es_file, uploaded_beta_file,
                vertical_coordinate=profiler_vertical_coordinate,
                latitude=st.session_state.lat,
                csv_separator=profiler_separator,
                decimal=profiler_decimal,
            )
            keys_to_reset = ['profile_specific_layers', 'wm_vertices', 'mixture_results', 'mixture_df', 'comparison_table', 'profiler_sel_index', 'results_df', 'kd_par_df', 'fig_n2', 'fig_rrs', 'fig_log', 'fig_k', 'fig_ts', 'fig_comp', 'external_figs', 'fig_mix_depth', 'discarded_indices', 'new_es_median', 'convolved_rrs_df', 'convolved_sensor_name', 'convolved_rrs_source', 'convolved_rrs_hyperspectral', 'convolution_audit', 'convolution_history', 'external_probe_df', 'master_kd_secchi_df', 'empirical_model_data', 'empirical_model_params', 'external_probe_filename', 'comparison_kd_df']
            for key in keys_to_reset:
                if key == 'master_kd_secchi_df': st.session_state[key] = pd.DataFrame(columns=['Station_ID', 'Secchi', 'Kd(PAR)_Profiler', 'Kd(490)_Profiler', 'Kd(PAR)_External'])
                elif key == 'empirical_model_data': st.session_state[key] = pd.DataFrame(columns=['Kd(PAR)', 'Kd(490)'])
                elif key in ['profile_specific_layers', 'external_figs']: st.session_state[key] = {}
                elif key in ['discarded_indices', 'convolution_history']: st.session_state[key] = []
                else: st.session_state[key] = None
            if st.session_state.station_data: st.session_state.selected_station=st.session_state.station_list[0]
    if st.session_state.station_data:
        st.header("2. Analysis Context")
        st.caption("Review the CTD comparison in **External Validation**, then choose the active station at the end of that tab.")
        active_station_data = st.session_state.station_data[st.session_state.selected_station]
        conversion_latitude = active_station_data.get('vertical_conversion_latitude')
        if (
            active_station_data.get('vertical_coordinate_source') != 'depth_m'
            and conversion_latitude is not None
            and not np.isclose(st.session_state.lat, conversion_latitude)
        ):
            st.warning(
                "Latitude changed after pressure was converted to depth. Reprocess the profiler files "
                "to keep the Depth (m) coordinate traceable."
            )
        if active_station_data.get('beta_data') is not None:
            with st.expander("Optional backscattering display", expanded=False):
                beta_cols_available = active_station_data['beta_cols']
                if beta_cols_available:
                    st.session_state.selected_beta_col = st.selectbox(
                        "Beta wavelength to plot:", options=beta_cols_available, index=0
                    )
        st.caption(
            "Define analysis layers and propagated-Rrs irradiance in **Core Optics**. "
            "This sidebar is reserved for loading data and switching the active profile."
        )
        st.divider()
            
        if st.button("Reset All & Restart App", type="primary", use_container_width=True):
            # 1. Increment the uploader ID so new widgets are created on rerun
            new_id = st.session_state.uploader_id + 1
            # 2. Clear all data
            st.session_state.clear()
            # 3. Restore the incremented ID
            st.session_state.uploader_id = new_id
            # 4. Rerun the app (Uploaders will now be empty)
            st.rerun()

# 6. UI: MAIN CONTENT
if not st.session_state.station_data:
    st.info("Welcome! Please upload your data files using the sidebar to begin.")
else:
    if 'previous_station' not in st.session_state or st.session_state.previous_station != st.session_state.selected_station:
       # Clear optical results on station change
       st.session_state.mixture_results = None
       st.session_state.mixture_df = None
       st.session_state.derived_products = None
       st.session_state.discarded_indices = []
       st.session_state.new_es_median = None
       st.session_state.convolved_rrs_df = None
       st.session_state.convolved_sensor_name = None
       st.session_state.convolved_rrs_source = None
       st.session_state.convolved_rrs_hyperspectral = None
       st.session_state.convolution_audit = None
       st.session_state.convolution_history = []
       
       # Critical cleanup to avoid dimension mismatch
       st.session_state.mixture_df_profiler = None
       st.session_state.mixture_df_external = None
       st.session_state.mixture_results_profiler = None
       st.session_state.mixture_results_external = None
       
       #Updates stations
       st.session_state.previous_station = st.session_state.selected_station

    # NOTIFICATION HANDLER
    if st.session_state.notification:
        msg = st.session_state.notification['message']
        icon = "✅" if st.session_state.notification['type'] == 'success' else "❌"
        
        st.toast(msg, icon=icon)
        
        # Clear the notification state
        st.session_state.notification = None # Clear the message after showing it once

    # DATA PREPARATION AND CENTRAL CALCULATIONS
    station_id = st.session_state.selected_station
    station_data = st.session_state.station_data[station_id]
    es_for_rrs = station_data.get('Es', None)
    phys_props_df = calculate_physical_properties(station_data, st.session_state.lon, st.session_state.lat)
    
    current_station_layers = st.session_state.profile_specific_layers.get(station_id, {})
    if current_station_layers:
        propagation_irradiance_mode = st.session_state.get(
            f"propagation_irradiance_mode_{station_id}", 'auto'
        )
        ed0_propagation_mode = st.session_state.get(
            f"ed0_propagation_mode_{station_id}", 'raw_fit'
        )
        st.session_state.derived_products = calculate_derived_products(
            station_data,
            station_id,
            current_station_layers,
            propagation_irradiance_mode,
            ed0_propagation_mode,
        )
    else:
        st.session_state.derived_products = None
    
    derived_products = st.session_state.get('derived_products')
    if derived_products:
        layer_k_data = derived_products['layer_k_data']
        results_df = derived_products['results_df']
        kd_par_df = derived_products['kd_par_df']
        st.session_state.results_df = results_df
        st.session_state.kd_par_df = kd_par_df
    else:
        layer_k_data = {}; results_df = None; kd_par_df = None
        st.session_state.results_df = None; st.session_state.kd_par_df = None

    # EXTERNAL KD(PAR) CALCULATION
    if derived_products and st.session_state.get('external_probe_df') is not None:
        df_ext_raw = st.session_state.external_probe_df
        # Use as chaves estáveis da Tab 1
        ext_depth_col = st.session_state.get('p1_z')
        ext_par_col = st.session_state.get('p1_par')
        
        if ext_depth_col != "None" and ext_par_col != "None" and ext_depth_col in df_ext_raw.columns:
            comparison_data = []
            model = st.session_state.get('empirical_model_params')
            for index, profiler_row in kd_par_df.iterrows():
                # Correctly parse layer number
                layer_str = profiler_row['Layer']
                layer_num = int(layer_str.split(' ')[1])
                z_min, z_max = current_station_layers[layer_num]['range']
                
                external_kd_par = calculate_external_kd_par(df_ext_raw, ext_depth_col, ext_par_col, z_min, z_max)
                
                estimated_kd490_external = np.nan
                if model and not pd.isna(external_kd_par):
                    estimated_kd490_external = model['slope'] * external_kd_par + model['intercept']
                
                comparison_data.append({
                    "Layer": layer_str,
                    "Kd(PAR) (Profiler)": profiler_row['Kd(PAR)'],
                    "Kd(PAR) (External)": external_kd_par,
                    "Kd(490) (Profiler)": results_df.loc[index, 'Kd(490)'],
                    "Estimated Kd(490) (External)": estimated_kd490_external
                })
            st.session_state.comparison_kd_df = pd.DataFrame(comparison_data).set_index('Layer')
    else:
        st.session_state.comparison_kd_df = None
        
    fig_n2, fig_rrs, fig_log, fig_k, fig_ts = (plt.figure() for i in range(5))
    
    

    # TAB CREATION
    def on_tab_change():
        st.session_state.active_tab = st.session_state.main_tabs
    
   # 1. Criação normal das abas com nomes limpos
    tab_titles = [
        "External Validation",
        "Es Quality Control",
        "Core Optics",
        "Satellite Matching",
        "Water Masses (T-S)",
        "Master Data & Reports",
        "Methods & Protocols",
        "Optional Analyses",
    ]

    (
        tab_ctd,
        tab_es,
        tab_core,
        tab_convolution,
        tab_ts,
        tab_report,
        tab_formulas,
        tab_optional,
    ) = st.tabs(tab_titles)

    with tab_optional:
        tab_tools, tab_models, tab_secchi = st.tabs([
            "Instrument Tools",
            "Bio-Optical Models",
            "Secchi & Kd",
        ])




       
    # ------------------ OPTIONAL TOOL: DEPTH CORRECTION ------------------
    
    with tab_tools:
        st.header("Optional Instrument Tools")
        st.info(
            "Use depth correction only when an independent reference indicates a profiler depth-offset error. "
            "It is intentionally kept outside the routine analysis path."
        )
        st.subheader("Depth Correction")
        st.caption("Visually align the profiles. The BLUE line should overlap the GREEN line.")
        
        if not st.session_state.station_data:
            st.warning("Please upload profiler files first.")
        else:
            c1, c2 = st.columns([1, 1])
            target_station = c1.selectbox("1. Profiler:", st.session_state.station_list, key="fix_tgt")
            uploaded_ref = c2.file_uploader("2. Reference (e.g. Castaway/External .csv):", type="csv", key="ref_upl")

            if uploaded_ref:
                if 'last_uploaded_ref' not in st.session_state or st.session_state.last_uploaded_ref != uploaded_ref.name:
                    st.session_state.df_ref_tool = load_castaway_ctd(uploaded_ref)
                    st.session_state.last_uploaded_ref = uploaded_ref.name
                
                df_ref = st.session_state.df_ref_tool
                
                if df_ref is not None:
                    if 'current_offset_val' not in st.session_state:
                        st.session_state.current_offset_val = 0.0
                    
                    if 'tool_d_ref' not in st.session_state or st.button("Reload Original Data"):
                        
                        # Process Reference
                        d_ref, val_ref, label_ref = get_profile_metric_data(df_ref, st.session_state.lat)
                        
                        if d_ref is None:
                            st.error("Error: Invalid or empty reference file.")
                            st.stop()

                        # Process Target
                        t_data = st.session_state.station_data[target_station]
                        
                        df_target = pd.DataFrame({'Depth': t_data['depth_m'], 'Temperature': t_data['temperature']})
                        
                        # Se a referência usou Gradiente, NÃO damos condutividade para o Profiler,
                        # forçando ele a usar Gradiente também.
                        if "Gradiente" not in label_ref:
                            if len(t_data['conductivity']) == len(t_data['depth_m']):
                                df_target['Conductivity'] = t_data['conductivity']
                        
                        # Process Target
                        d_tgt, val_tgt, label_tgt = get_profile_metric_data(df_target, st.session_state.lat)
                        
                        st.success(f"Comparing using: **{label_ref}**")

                        # Salva vetores
                        st.session_state.tool_d_ref = d_ref
                        st.session_state.tool_n2_ref = val_ref
                        st.session_state.tool_d_tgt = d_tgt
                        st.session_state.tool_n2_tgt = val_tgt
                        st.session_state.metric_label = label_ref
                        
                        # Tenta sugestão automática inicial
                        try:
                            # Ref (>2m)
                            mask_r = (d_ref > 2.0) & (~np.isnan(val_ref))
                            pk_r = d_ref[np.where(mask_r)[0][np.nanargmax(val_ref[mask_r])]] if np.any(mask_r) else 0.0
                            # Tgt (Todo)
                            mask_t = ~np.isnan(val_tgt)
                            pk_t = d_tgt[np.where(mask_t)[0][np.nanargmax(val_tgt[mask_t])]] if np.any(mask_t) else 0.0
                            
                            st.session_state.ruler_ref = float(pk_r)
                            st.session_state.ruler_tgt = float(pk_t)
                            st.session_state.current_offset_val = float(pk_r - pk_t)
                        except:
                            st.session_state.ruler_ref = 0.0
                            st.session_state.ruler_tgt = 0.0
                            st.session_state.current_offset_val = 0.0

                    # INTERFACE
                    st.markdown("---")
                    col_graph, col_ctrl = st.columns([3, 1])
                    
                    with col_ctrl:
                        st.subheader("Ruler & Adjustment")
                        st.markdown("Use the graph to read exact peak values.")
                        
                        val_ref = st.number_input("GREEN Peak (Reference):", value=st.session_state.get('ruler_ref', 0.0), step=0.1, format="%.2f")
                        val_tgt = st.number_input("RED Peak (Profiler):", value=st.session_state.get('ruler_tgt', 0.0), step=0.1, format="%.2f")
                        
                        if st.button("Calculate and Test Difference"):
                            diff = val_ref - val_tgt
                            st.session_state.current_offset_val = diff
                            st.rerun()
                        
                        st.markdown("---")
                        
                        final_offset = st.number_input(
                            "**Final Offset (m):**",
                            value=st.session_state.current_offset_val, 
                            step=0.1, format="%.2f",
                            help="This is the value that will be permanently applied."
                        )
                        
                        # Sincroniza
                        if final_offset != st.session_state.current_offset_val:
                            st.session_state.current_offset_val = final_offset
                            st.rerun()

                        st.markdown("---")
                        if st.button("APPLY FINAL OFFSET", type="primary"):
                            # 1. Aplica nos dados principais (Pressure, Temp, Cond, Ed, Lu)
                            t_data = st.session_state.station_data[target_station]
                            
                            new_depth_m = t_data['depth_m'] + st.session_state.current_offset_val
                            st.session_state.station_data[target_station]['depth_m'] = new_depth_m
                            
                            if t_data['beta_data'] is not None and not t_data['beta_data'].empty:
                                t_data['beta_data'].index = t_data['beta_data'].index + st.session_state.current_offset_val

                            if target_station in st.session_state.profile_specific_layers:
                                del st.session_state.profile_specific_layers[target_station]
                            
                            st.session_state.last_applied_offset_str = f"{st.session_state.current_offset_val:+.2f}"

                            st.toast(f"Corrected: {st.session_state.current_offset_val:+.2f}m em todos os sensores!")
                            
                            keys_to_clear = [
                                'tool_d_ref', 'tool_n2_ref', 
                                'tool_d_tgt', 'tool_n2_tgt', 
                                'current_offset_val', 
                                'ruler_ref', 'ruler_tgt'
                            ]
                            for k in keys_to_clear:
                                if k in st.session_state:
                                    del st.session_state[k]
                            
                            st.rerun()

                    with col_graph:
                        if 'tool_d_ref' in st.session_state:
                            
                            
                            norm_view = st.checkbox("Normalize Curves (0 to 1)", value=True, 
                                                  help="Forces curves to have the same visual width. Essential when comparing Gradients with N2.")
                            
                            d_r, n_r = st.session_state.tool_d_ref, st.session_state.tool_n2_ref
                            sort_r = np.argsort(d_r)
                            
                            d_t, n_t = st.session_state.tool_d_tgt, st.session_state.tool_n2_tgt
                            sort_t = np.argsort(d_t)
                            
                            xlabel_txt = st.session_state.get('metric_label', "Metric")
                            
                            fig = plot_interactive_n2_offset(
                                d_r[sort_r], n_r[sort_r],
                                d_t[sort_t], n_t[sort_t],
                                st.session_state.current_offset_val,
                                xlabel=xlabel_txt,
                                normalize=norm_view # <--- AQUI A MÁGICA
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Data error.")
            
            st.markdown("---")
            st.subheader("3. Export Corrected Data")
            
            if target_station in st.session_state.station_data:
                offset_suffix = st.session_state.get('last_applied_offset_str', 'corrected')
                zip_filename = f"{target_station}_Depth_Offset_{offset_suffix}m.zip"
                zip_buffer_fix = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer_fix, 'w', zipfile.ZIP_DEFLATED) as zf:
                    t_data = st.session_state.station_data[target_station]
                    
                    df_main = pd.DataFrame({'Depth_Corrected_m': t_data['depth_m']})
                    if len(t_data['temperature']) == len(df_main): df_main['Temperature'] = t_data['temperature']
                    if len(t_data['conductivity']) == len(df_main): df_main['Conductivity'] = t_data['conductivity']
                    wavelengths = t_data['wavelengths']
                    if t_data['Ed_data'].shape[0] == len(df_main):
                        for i, wl in enumerate(wavelengths): df_main[f'Ed_{int(wl)}'] = t_data['Ed_data'][:, i]
                    if t_data['Lu_data'].shape[0] == len(df_main):
                        for i, wl in enumerate(wavelengths): df_main[f'Lu_{int(wl)}'] = t_data['Lu_data'][:, i]
                    
                    zf.writestr(f"{target_station}_Profiler_Offset_{offset_suffix}.csv", df_main.to_csv(index=False).encode('utf-8'))
                    
                    # 2. Beta
                    if t_data['beta_data'] is not None:
                        df_beta_save = t_data['beta_data'].reset_index()
                        col_prof = df_beta_save.columns[0]
                        df_beta_save.rename(columns={col_prof: 'Depth_Corrected_m'}, inplace=True)
                        zf.writestr(f"{target_station}_Beta_Offset_{offset_suffix}.csv", df_beta_save.to_csv(index=False).encode('utf-8'))
                
                st.download_button(
                    label=" Baixar Dados Corrigidos (.zip)",
                    data=zip_buffer_fix.getvalue(),
                    file_name=zip_filename,
                    mime="application/zip",
                    key="dl_fixed_btn"
                )        
   
    
    # ------------------TAB 2: CTD & RADIOMETRIC COMPARISON ------------------
    with tab_ctd:
        st.header("External Validation")
        st.caption(
            "Configure the external observations and compare the profiler with the CTD before choosing the station for the optical pipeline."
        )
        st.divider()
        
        st.subheader("Probe 1: CTD, Chl and Secchi")
        c1, c2 = st.columns(2)
        sep1 = c1.radio("Separator (P1)", [',', ';'], horizontal=True, key="s_p1_final")
        dec1 = c2.radio("Decimal (P1)", ['.', ','], horizontal=True, key="d_p1_final")
        
        up_p1 = st.file_uploader("Upload Probe 1 (Castaway/RBR/JFE)", type="csv", key=f"up_p1_{st.session_state.uploader_id}")
        if up_p1:
            st.session_state.external_probe_df = pd.read_csv(up_p1, sep=sep1, decimal=dec1)
        
        if st.session_state.external_probe_df is not None:
            df1 = st.session_state.external_probe_df
            cols1 = ["None"] + list(df1.columns)
            
            st.markdown("#### Configure Columns for Probe 1")
            m = st.columns(5)
            # Usamos on_change para garantir que o valor fique preso no session_state
            st.session_state.p1_st = m[0].selectbox("Station ID", cols1, index=get_col_index(cols1, 'p1_st'), key="sel_st1_f")
            st.session_state.p1_z = m[1].selectbox("Vertical coordinate column", cols1, index=get_col_index(cols1, 'p1_z'), key="sel_z1_f")
            st.session_state.p1_t = m[2].selectbox("Temp", cols1, index=get_col_index(cols1, 'p1_t'), key="sel_t1_f")
            st.session_state.p1_s = m[3].selectbox("Salinity", cols1, index=get_col_index(cols1, 'p1_s'), key="sel_s1_f")
            st.session_state.p1_chl = m[4].selectbox("Chl-a", cols1, index=get_col_index(cols1, 'p1_chl'), key="sel_chl1_f")
            
            ce = st.columns(3)
            st.session_state.p1_sec = ce[0].selectbox("Secchi Col", cols1, index=get_col_index(cols1, 'p1_sec'), key="sel_sec1_f")
            st.session_state.p1_par = ce[1].selectbox("PAR Col", cols1, index=get_col_index(cols1, 'p1_par'), key="sel_par1_f")
            st.session_state.p1_vertical_coordinate = ce[2].selectbox(
                "How is depth recorded in this column?",
                options=list(VERTICAL_COORDINATE_LABELS),
                index=list(VERTICAL_COORDINATE_LABELS).index(
                    st.session_state.p1_vertical_coordinate,
                ),
                format_func=lambda coordinate: VERTICAL_COORDINATE_LABELS[coordinate],
                help="This selection is used only to align the external CTD profile to Depth (m).",
                key="sel_p1_vertical_coordinate",
            )

        st.divider()

        st.subheader("Probe 2: CDOM")
        c3, c4 = st.columns(2)
        sep2 = c3.radio("Separator (P2)", [',', ';'], horizontal=True, key="s_p2_final")
        dec2 = c4.radio("Decimal (P2)", ['.', ','], horizontal=True, key="d_p2_final")
        
        up_p2 = st.file_uploader("Upload Probe 2 (C3/Fluorimeter)", type="csv", key=f"up_p2_{st.session_state.uploader_id}")
        if up_p2:
            st.session_state.external_bio_df = pd.read_csv(up_p2, sep=sep2, decimal=dec2)
            
        if st.session_state.external_bio_df is not None:
            df2 = st.session_state.external_bio_df
            cols2 = ["None"] + list(df2.columns)
            st.markdown("#### Configure Columns for Probe 2")
            b = st.columns(3)
            st.session_state.p2_st = b[0].selectbox("Station ID (P2)", cols2, index=get_col_index(cols2, 'p2_st'), key="sel_st2_f")
            st.session_state.p2_z = b[1].selectbox("Vertical coordinate column (P2)", cols2, index=get_col_index(cols2, 'p2_z'), key="sel_z2_f")
            st.session_state.p2_cdom = b[2].selectbox("CDOM column", cols2, index=get_col_index(cols2, 'p2_cdom'), key="sel_cdom2_f")
            st.session_state.p2_vertical_coordinate = st.selectbox(
                "How is depth recorded in the CDOM file?",
                options=list(VERTICAL_COORDINATE_LABELS),
                index=list(VERTICAL_COORDINATE_LABELS).index(
                    st.session_state.p2_vertical_coordinate,
                ),
                format_func=lambda coordinate: VERTICAL_COORDINATE_LABELS[coordinate],
                help=(
                    "The CDOM profile is converted to Depth (m) before interpolation. "
                    "All resulting profiles and exports remain in Depth (m)."
                ),
                key="sel_p2_vertical_coordinate",
            )

        st.divider()
        
        st.subheader("Comparison Analysis (Profiler vs. Probe 1)")
        if st.session_state.external_probe_df is not None:
            casts = st.multiselect("Select casts to compare:", options=st.session_state.station_list, default=[st.session_state.selected_station], key="multi_comp")
            
            if st.button("Run Comparison Analysis", key="btn_run_comp_final", use_container_width=True):
                with st.spinner("Processing..."):
                    df_ext_raw = st.session_state.external_probe_df
                    st_col = st.session_state.p1_st
                    all_stats, all_merged_dfs = [], []
                    candidates = df_ext_raw[st_col].dropna().unique()
                    
                    for cid in casts:
                        match = find_best_match(cid, candidates)
                        if match:
                            df_e = df_ext_raw[df_ext_raw[st_col].astype(str) == str(match)].copy()
                            df_e['Depth'] = _to_depth_m(
                                df_e[st.session_state.p1_z],
                                st.session_state.p1_vertical_coordinate,
                                st.session_state.lat,
                            )
                            df_e['Temperature'] = pd.to_numeric(df_e[st.session_state.p1_t], errors='coerce')
                            df_e['Salinity'] = pd.to_numeric(df_e[st.session_state.p1_s], errors='coerce')
                            df_e.dropna(subset=['Depth', 'Temperature', 'Salinity'], inplace=True)
                            
                            df_i = calculate_physical_properties(st.session_state.station_data[cid], st.session_state.lon, st.session_state.lat)
                            merged = compare_ctd_profiles(df_i, df_e)
                            
                            if merged is not None and not merged.empty:
                                t_res = get_comparison_stats(merged, 'Temperature')
                                s_res = get_comparison_stats(merged, 'Salinity')
                                
                                all_stats.append({
                                    'Cast ID': cid, 'Match Found': str(match), 
                                    'Temp R2': t_res['R2'], 'Temp RMSE': t_res['RMSE'], 'Temp MAE': t_res['MAE'], 'Temp Bias': t_res['Bias'],
                                    'Sal R2': s_res['R2'], 'Sal RMSE': s_res['RMSE'], 'Sal MAE': s_res['MAE'], 'Sal Bias': s_res['Bias']
                                })
                                merged['Cast ID'] = cid; all_merged_dfs.append(merged)
                    
                    if all_merged_dfs:
                        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey='row')
                        ((ax_t, ax_s), (ax_td, ax_sd)) = axes
                        colors = sns.color_palette("viridis", len(all_merged_dfs))
                        for i, df in enumerate(all_merged_dfs):
                            ax_t.plot(df['Temperature_internal'], df['Depth'], color=colors[i], label=f"Profiler {df['Cast ID'].iloc[0]}")
                            ax_s.plot(df['Salinity_internal'], df['Depth'], color=colors[i])
                            ax_t.plot(
                                df['Temperature_external'], df['Depth'], 'w--', alpha=0.7, lw=1.5,
                                label='External CTD' if i == 0 else None,
                            )
                            ax_s.plot(df['Salinity_external'], df['Depth'], 'w--', alpha=0.7, lw=1.5)
                            ax_td.plot(df['Temperature_internal']-df['Temperature_external'], df['Depth'], color=colors[i])
                            ax_sd.plot(df['Salinity_internal']-df['Salinity_external'], df['Depth'], color=colors[i])
                        
                        ax_t.set_title("Temperature"); ax_s.set_title("Salinity")
                        ax_td.set_title("Temperature difference"); ax_sd.set_title("Salinity difference")
                        ax_t.set_xlabel("Temperature (°C)"); ax_s.set_xlabel("Practical salinity")
                        ax_td.set_xlabel("Profiler − CTD (°C)"); ax_sd.set_xlabel("Profiler − CTD")
                        ax_t.set_ylabel("Depth (m)"); ax_td.set_ylabel("Depth (m)")
                        ax_t.invert_yaxis(); ax_td.invert_yaxis()
                        for ax in axes.flat: ax.grid(True, linestyle=':', alpha=0.5)
                        ax_t.legend(fontsize=8); ax_td.axvline(0, color='w', alpha=0.3); ax_sd.axvline(0, color='w', alpha=0.3)
                        
                        st.session_state.fig_comp = fig
                        st.session_state.comparison_table = pd.DataFrame(all_stats).set_index('Cast ID')
                        plt.close(fig)

        if st.session_state.get('fig_comp') is not None:
            st.pyplot(st.session_state.fig_comp)
            df_disp = st.session_state.comparison_table
            num_cols = [c for c in df_disp.columns if c != 'Match Found']
            st.dataframe(df_disp.style.apply(highlight_best).format({c: '{:.4f}' for c in num_cols}), use_container_width=True)

        st.divider()
        st.subheader("Choose Station for the Optical Pipeline")
        st.caption(
            "After reviewing the CTD comparison, choose the station to carry into Es Quality Control, Core Optics, models, and reports."
        )
        active_station_index = st.session_state.station_list.index(st.session_state.selected_station)
        validation_station = st.selectbox(
            "Station to use in the downstream analysis:",
            options=st.session_state.station_list,
            index=active_station_index,
            key=f"validation_station_selector_{st.session_state.uploader_id}",
        )
        if validation_station != st.session_state.selected_station:
            st.session_state.selected_station = validation_station
            st.rerun()

    # ------------------TAB 3: ES ------------------

    with tab_es:

        st.header("Surface Irradiance (Es) Visualization")
    
        all_es_spectra = station_data.get("Es_all_spectra", None)
    
        # Clean safe index
        if all_es_spectra is not None and not all_es_spectra.empty:
            all_es_spectra = all_es_spectra.reset_index(drop=True)
            y_max_es = st.number_input("Set Y-axis Max for Es Plot:", min_value=1.0, value=200.0, step=10.0, format="%.1f", key="es_ymax_selector")
            fig_es, ax_es = plt.subplots(figsize=(8, 6))
    
        # Es is optional: keep the rest of the application available when absent.
        if all_es_spectra is None or all_es_spectra.empty:
            st.warning("No measured Es file was uploaded for this station.")
            all_es_spectra = pd.DataFrame({f'{WAVELENGTH_PREFIX}490': pd.Series(dtype=float)})
    
        # Create session state variables
        if "discarded_indices" not in st.session_state:
            st.session_state.discarded_indices = []
        if "new_es_median" not in st.session_state:
            st.session_state.new_es_median = None
    
        # 1. Identify rows that have at least one valid number (are not all NA).
        #    We use dropna(how='all') which only removes rows where EVERY column is NA.
        #    This gives us a DataFrame of only the usable spectra.
        valid_spectra_df = all_es_spectra.dropna(how='all', subset=[c for c in all_es_spectra.columns if c.startswith('X')])
        
        # 2. The original count is  the number of VALID spectra.
        original_valid_count = len(valid_spectra_df)
        
        # 3. Calculate how many of these valid spectra were kept.
        # We find the intersection of the valid indices and the kept indices.
        discard_set = set(st.session_state.discarded_indices)
        kept_indices = all_es_spectra.index.difference(discard_set)
        final_kept_valid_indices = valid_spectra_df.index.intersection(kept_indices)
        final_kept_count = len(final_kept_valid_indices)
    
        #   Summary metric 
        st.metric("Valid Spectra After Filtering", f"{final_kept_count} / {original_valid_count}")
    
    
        #   Plot Es spectra 
     
        fig_es, ax_es = plt.subplots(figsize=(8, 6))
        wavelength_values = [
            float(col.replace(WAVELENGTH_PREFIX, "")) for col in all_es_spectra.columns
        ]
        added_disc = False
        added_keep = False
    
        for idx, spectrum in all_es_spectra.iterrows():
            if idx in discard_set:
                label = "Discarded Spectra" if not added_disc else ""
                ax_es.plot(wavelength_values, spectrum.values, lw=0.5, alpha=0.35,
                           color="#FF5733", label=label)
                added_disc = True
            else:
                # We also check if the row is valid before plotting it as kept
                if not spectrum.dropna().empty:
                    label = "Kept Spectra" if not added_keep else ""
                    ax_es.plot(wavelength_values, spectrum.values, lw=0.5, alpha=0.6,
                               color="#888888", label=label)
                    added_keep = True
    
        # Original median (calculated only on valid spectra for accuracy)
        original_median_es = valid_spectra_df.median().values
        ax_es.plot(wavelength_values, original_median_es, lw=2.0, linestyle="--",
                   color="cyan", label=f"Original Median ({original_valid_count} valid spectra)")
    
        # New filtered median
        if st.session_state.new_es_median is not None:
            ax_es.plot(wavelength_values, st.session_state.new_es_median, lw=3.0,
                       color="#33FF57", label=f"New Median ({final_kept_count} spectra)")
    
        ax_es.set_xlabel("Wavelength (nm)", weight="bold")
        ax_es.set_ylabel(r"Es ($\mu$W cm$^{-2}$ nm$^{-1}$)", weight="bold")
        ax_es.set_title("Measured Es spectra", weight="bold")
        ax_es.set_xlim(380, 700)
        ax_es.set_ylim(bottom=0)
        ax_es.grid(True, linestyle='--')
        ax_es.legend()
        st.pyplot(fig_es)
    
        
        #   FILTERING PANEL 
        st.subheader("Es Stability Analysis & Hybrid Filtering")
        default_index = int(np.abs(np.array([float(c.replace("X", "")) for c in all_es_spectra.columns]) - 490).argmin())
        col_name_for_analysis = st.selectbox("Select reference wavelength for stability check:", options=list(all_es_spectra.columns), index=default_index)
        
        with st.form("hybrid_es_filters_form"):
            st.markdown("### Active Filters")
            use_tilt_filter = st.checkbox("Enable Tilt Filter", value=True)
            threshold_pct = st.number_input("Tilt threshold (% change)", 0.1, 50.0, 2.0, 0.5)
            use_low_filter = st.checkbox("Enable Low Outlier Filter", value=True)
            low_pct = st.number_input("Discard lowest %", 0.0, 49.9, 5.0, 1.0)
            use_high_filter = st.checkbox("Enable High Outlier Filter (Hybrid)", value=True)
            high_pct = st.number_input("Discard highest %", 0.0, 49.9, 5.0, 0.5)
            submitted = st.form_submit_button("Apply Filters")
    
        if submitted:
            discard_tilt = set()
            if use_tilt_filter:
                pct_change = all_es_spectra[col_name_for_analysis].pct_change().abs() * 100
                discard_tilt = set(pct_change[pct_change > threshold_pct].index)
            keep_after_tilt = all_es_spectra.index.difference(discard_tilt)
            spectra_after_tilt = all_es_spectra.loc[keep_after_tilt]
            discard_low, discard_high = set(), set()
            if not spectra_after_tilt.empty:
                hybrid_metric = (0.25 * spectra_after_tilt.max(axis=1) + 0.75 * spectra_after_tilt.sum(axis=1))
                if use_low_filter:
                    num_low = int(len(spectra_after_tilt) * (low_pct / 100))
                    discard_low = set(hybrid_metric.nsmallest(num_low).index)
                if use_high_filter:
                    num_high = int(len(spectra_after_tilt) * (high_pct / 100))
                    discard_high = set(hybrid_metric.nlargest(num_high).index)
            all_discards = sorted(list(discard_tilt | discard_low | discard_high))
            st.session_state.discarded_indices = all_discards
            keep_indices = all_es_spectra.index.difference(all_discards)
            if len(keep_indices) > 0:
                st.session_state.new_es_median = all_es_spectra.loc[keep_indices].median().values
            else:
                st.session_state.new_es_median = None
            st.rerun()
    
        if st.button("Clear All Filters & Reset"):
            st.session_state.discarded_indices = []
            st.session_state.new_es_median = None
            st.rerun()
    
    


    
    # ------------------ TAB 4: CORE OPTICAL ANALYSIS ------------------
    with tab_core:
        st.header("Core Optics")
        current_station_id = st.session_state.selected_station
        current_station_layers = st.session_state.profile_specific_layers.get(current_station_id, {})
        st.caption(
            "Routine flow: use N² and beta to define layers → inspect Lu/Ed → evaluate propagated products."
        )
        core_spectral_bounds = SCIENTIFIC_PRODUCT_WAVELENGTH_RANGE
        rrs_axis_control, rrs_axis_note = st.columns([1, 3])
        rrs_y_max = rrs_axis_control.number_input(
            "Rrs Y-axis maximum:", min_value=0.001, value=0.010,
            step=0.001, format="%.3f", key="rrs_ymax_selector",
        )
        rrs_axis_note.caption(
            f"All Core spectral plots use {core_spectral_bounds[0]:.0f}–{core_spectral_bounds[1]:.0f} nm, "
            "the validated propagated-product domain. Raw NIR data remain available in the input/QC views."
        )
        overview_left, overview_right = st.columns(2)
        with overview_left:
            st.subheader("1. Stratification and backscattering")
            render_stratification_beta(station_data, phys_props_df, current_station_layers)
        with overview_right:
            st.subheader("2. Surface Rrs")
            render_rrs_spectrum(
                station_data, current_station_layers, derived_products,
                rrs_y_max, SCIENTIFIC_PRODUCT_WAVELENGTH_RANGE,
            )

        with st.expander("3. Analysis setup: layers and propagated Rrs", expanded=True):
            render_layer_setup(current_station_id, station_data)

        st.subheader("4. Interactive in-water radiometry")
        quick_control, quick_layer_control, quick_note = st.columns([1, 1, 2])
        quick_wavelength_index = quick_control.selectbox(
            "Wavelength:",
            options=list(range(len(station_data['wavelengths']))),
            index=int(np.abs(station_data['wavelengths'] - 490).argmin()),
            format_func=lambda index: f"{station_data['wavelengths'][index]:.1f} nm",
            key="lu_ed_quicklook_wavelength",
        )
        quick_marker_layer = None
        if current_station_layers and derived_products:
            quick_marker_layer = quick_layer_control.selectbox(
                "Layer for 1/Kd:",
                options=sorted(current_station_layers),
                format_func=lambda layer_num: f"Layer {layer_num}",
                key="lu_ed_quicklook_marker_layer",
            )
        else:
            quick_layer_control.caption("Define a layer to display 1/Kd.")
        quick_note.caption(
            "The left panel is the single-wavelength ln(Lu)/ln(Ed) diagnostic; the star updates "
            "to the selected wavelength and layer. The right panel retains the in-water Lu/Ed ratio."
        )

        quick_depth = np.asarray(station_data['depth_m'], dtype=float)
        quick_ed = np.asarray(station_data['Ed_data'][:, quick_wavelength_index], dtype=float)
        quick_lu = np.asarray(station_data['Lu_data'][:, quick_wavelength_index], dtype=float)
        quick_mask = (
            np.isfinite(quick_depth) & np.isfinite(quick_ed) & np.isfinite(quick_lu)
            & (quick_ed > 0) & (quick_lu > 0)
        )
        if np.any(quick_mask):
            fig_lued, (ax_lued, ax_ratio) = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
            selected_wavelength = station_data['wavelengths'][quick_wavelength_index]
            quick_marker_note = None
            quick_marker_is_outside = False
            quick_audit_record = None
            kd_fit_audit = derived_products.get('kd_fit_audit', pd.DataFrame()) if derived_products else pd.DataFrame()
            if quick_marker_layer is not None and not kd_fit_audit.empty:
                matching_audit = kd_fit_audit[
                    (kd_fit_audit['Layer'] == f'Layer {quick_marker_layer}')
                    & np.isclose(kd_fit_audit['Wavelength_nm'], selected_wavelength)
                ]
                if not matching_audit.empty:
                    quick_audit_record = matching_audit.iloc[0]
            ax_lued.plot(np.log(quick_ed[quick_mask]), quick_depth[quick_mask], color='#4da6ff', lw=2.2, label='ln(Ed)')
            ax_lued.plot(np.log(quick_lu[quick_mask]), quick_depth[quick_mask], color='#ff9933', lw=2.2, label='ln(Lu)')
            if quick_marker_layer is not None:
                quick_k_values = derived_products.get('layer_k_data', {}).get(quick_marker_layer)
                if quick_k_values is not None:
                    quick_kd = quick_k_values[0][quick_wavelength_index]
                    quick_optical_depth = (
                        1.0 / quick_kd
                        if np.isfinite(quick_kd) and quick_kd > 0 else np.nan
                    )
                    quick_ed_depths = quick_depth[quick_mask]
                    if np.isfinite(quick_optical_depth):
                        measured_min_depth = quick_ed_depths.min()
                        measured_max_depth = quick_ed_depths.max()
                        quick_marker_note = (
                            f"1/Kd (Kw-constrained; L{quick_marker_layer}; "
                            f"{selected_wavelength:.1f} nm) = {quick_optical_depth:.2f} m."
                        )
                        if not measured_min_depth <= quick_optical_depth <= measured_max_depth:
                            quick_marker_is_outside = True
                            quick_marker_note += (
                                " This attenuation length is outside the measured profile "
                                f"({measured_min_depth:.2f}–{measured_max_depth:.2f} m), "
                                "so the marker is not drawn."
                            )
                    else:
                        quick_marker_note = (
                            f"1/Kd (Kw-constrained; L{quick_marker_layer}; "
                            f"{selected_wavelength:.1f} nm) is unavailable because Kd is not "
                            "finite and positive."
                        )
                    if (
                        np.isfinite(quick_optical_depth)
                        and quick_ed_depths.size >= 2
                        and quick_ed_depths.min() <= quick_optical_depth <= quick_ed_depths.max()
                    ):
                        quick_depth_order = np.argsort(quick_ed_depths)
                        quick_star_x = np.interp(
                            quick_optical_depth,
                            quick_ed_depths[quick_depth_order],
                            np.log(quick_ed[quick_mask])[quick_depth_order],
                        )
                        ax_lued.scatter(
                            quick_star_x,
                            quick_optical_depth,
                            marker='*',
                            s=100,
                            color='#4da6ff',
                            edgecolors='white',
                            linewidths=0.7,
                            zorder=10,
                            label=f'1/Kd constrained by Kw (L{quick_marker_layer})',
                        )
                else:
                    quick_marker_note = (
                        f"1/Kd is unavailable because Layer {quick_marker_layer} has no valid "
                        "Kd/Klu result."
                    )
            ax_lued.set_xlabel('ln(Radiance / Irradiance)', weight='bold')
            ax_lued.set_ylabel('Depth (m)', weight='bold')
            ax_lued.set_title(f'ln(Lu) and ln(Ed) at {selected_wavelength:.1f} nm', weight='bold')
            ax_lued.legend()

            quick_ratio = quick_lu[quick_mask] / quick_ed[quick_mask]
            ax_ratio.plot(quick_ratio, quick_depth[quick_mask], color='#d6e685', lw=2.2, label='Lu / Ed')
            ax_ratio.set_xlabel('Lu / Ed', weight='bold')
            ax_ratio.set_title('In-water radiance-to-irradiance ratio', weight='bold')
            ax_ratio.legend()

            quick_layer_colors = sns.color_palette('husl', max(3, len(current_station_layers)))
            for color_index, (_, layer_data) in enumerate(sorted(current_station_layers.items())):
                layer_color = quick_layer_colors[color_index]
                for axis in (ax_lued, ax_ratio):
                    axis.axhspan(*layer_data['range'], color=layer_color, alpha=0.13, zorder=0)
            ax_lued.invert_yaxis()
            for axis in (ax_lued, ax_ratio):
                axis.grid(True, linestyle=':', alpha=0.7)
            st.pyplot(fig_lued, use_container_width=True)
            plt.close(fig_lued)
            if quick_marker_note:
                if quick_marker_is_outside:
                    st.info(quick_marker_note)
                else:
                    st.caption(quick_marker_note)
            if quick_audit_record is not None:
                def format_quick_audit_value(value, digits=5):
                    return f'{value:.{digits}f}' if pd.notna(value) else 'Not applicable'

                def format_quick_audit_interval(low, high):
                    if pd.isna(low) or pd.isna(high):
                        return 'Not applicable'
                    return f'[{low:.5f}, {high:.5f}]'

                constraint_active = bool(quick_audit_record['Kw_constraint_active'])
                constrained_source = quick_audit_record['Kd_constrained_source']
                audit_display = pd.DataFrame([
                    {
                        'Product': 'Raw profiler fit',
                        'Kd (m⁻¹)': format_quick_audit_value(quick_audit_record['Kd_raw_m-1']),
                        '1/Kd (m)': format_quick_audit_value(quick_audit_record['One_over_Kd_raw_m'], 2),
                        'N': str(int(quick_audit_record['N_valid_raw_fit'])),
                        'R²': format_quick_audit_value(quick_audit_record['R2_raw_fit'], 4),
                        'SE Kd (m⁻¹)': format_quick_audit_value(
                            quick_audit_record['Kd_raw_standard_error_m-1'],
                        ),
                        '95% CI Kd (m⁻¹)': format_quick_audit_interval(
                            quick_audit_record['Kd_raw_CI95_low_m-1'],
                            quick_audit_record['Kd_raw_CI95_high_m-1'],
                        ),
                        'Source': 'Log-linear fit of ln(Ed) versus Depth',
                    },
                    {
                        'Product': 'Kw-constrained product',
                        'Kd (m⁻¹)': format_quick_audit_value(quick_audit_record['Kd_constrained_m-1']),
                        '1/Kd (m)': format_quick_audit_value(
                            quick_audit_record['One_over_Kd_constrained_m'], 2,
                        ),
                        'N': 'Not applicable' if constraint_active else str(int(quick_audit_record['N_valid_raw_fit'])),
                        'R²': format_quick_audit_value(quick_audit_record['R2_constrained'], 4),
                        'SE Kd (m⁻¹)': format_quick_audit_value(
                            quick_audit_record['Kd_constrained_standard_error_m-1'],
                        ),
                        '95% CI Kd (m⁻¹)': format_quick_audit_interval(
                            quick_audit_record['Kd_constrained_CI95_low_m-1'],
                            quick_audit_record['Kd_constrained_CI95_high_m-1'],
                        ),
                        'Source': constrained_source,
                    },
                ])
                with st.expander('Selected Kd audit: raw and Kw-constrained', expanded=True):
                    st.dataframe(audit_display, use_container_width=True, hide_index=True)
                    if constraint_active:
                        st.caption(
                            'The Kw-constrained value is a reference floor, not a second regression. '
                            'Its regression uncertainty is therefore reported as not applicable; '
                            'the raw-fit uncertainty remains visible in the first row.'
                        )
                    else:
                        st.caption(
                            'The raw profiler fit exceeds Kw, so the constrained product retains '
                            'the same value and regression uncertainty.'
                        )
        else:
            st.warning("No paired positive Lu and Ed observations are available at this wavelength.")

        st.divider()
        st.subheader("5. Advanced optical diagnostics")
        bottom_row = st.columns(2)
        plot_figsize = (5.5, 4.0)
        layer_colors = sns.color_palette("husl", max(3, len(current_station_layers)))
    
        with bottom_row[0]:
            fig_log, ax_log = plt.subplots(figsize=plot_figsize)
            wl_indices = {'490': (np.abs(station_data['wavelengths'] - 490)).argmin(), '555': (np.abs(station_data['wavelengths'] - 555)).argmin(), '600': (np.abs(station_data['wavelengths'] - 600)).argmin()}
            wl_colors = {'490': '#4da6ff', '555': '#33cc33', '600': '#ff9933'}
            for key, idx in wl_indices.items():
                wl = station_data['wavelengths'][idx]
                depth_col = station_data['depth_m']
                ed_col = station_data['Ed_data'][:, idx]
                lu_col = station_data['Lu_data'][:, idx]
                df_ed_plot = pd.DataFrame({'depth': depth_col, 'ed': ed_col}).dropna()
                df_ed_plot = df_ed_plot[df_ed_plot['ed'] > 0]
                ax_log.plot(np.log(df_ed_plot['ed']), df_ed_plot['depth'], label=f'ln(Ed) @ {wl:.1f} nm', color=wl_colors[key], linestyle='-', alpha=0.55, lw=1.0)
                df_lu_plot = pd.DataFrame({'depth': depth_col, 'lu': lu_col}).dropna()
                df_lu_plot = df_lu_plot[df_lu_plot['lu'] > 0]
                ax_log.plot(np.log(df_lu_plot['lu']), df_lu_plot['depth'], label=f'ln(Lu) @ {wl:.1f} nm', color=wl_colors[key], linestyle='--', alpha=0.55, lw=1.0)

                # Mark the first optical depth (1 / Kd) from the first defined layer.
                # Kd describes Ed, so the star is intentionally placed on the Ed profile.
                if derived_products and current_station_layers:
                    first_layer_num = min(current_station_layers)
                    first_layer_k_data = derived_products.get('layer_k_data', {}).get(first_layer_num)
                    if first_layer_k_data is not None:
                        kd_first_layer = first_layer_k_data[0]
                        kd_value = kd_first_layer[idx]
                        optical_depth = 1.0 / kd_value if np.isfinite(kd_value) and kd_value > 0 else np.nan
                        valid_ed = np.isfinite(depth_col) & np.isfinite(ed_col) & (ed_col > 0)
                        ed_depths = depth_col[valid_ed]
                        if (
                            np.isfinite(optical_depth)
                            and ed_depths.size >= 2
                            and ed_depths.min() <= optical_depth <= ed_depths.max()
                        ):
                            ed_log_values = np.log(ed_col[valid_ed])
                            star_x = np.interp(optical_depth, ed_depths, ed_log_values)
                            ax_log.scatter(
                                star_x,
                                optical_depth,
                                marker='*',
                                s=125,
                                color=wl_colors[key],
                                edgecolors='white',
                                linewidths=0.8,
                                zorder=20,
                                label=f'1/Kd (L{first_layer_num}) @ {wl:.0f} nm',
                            )
            for i, (layer_num, data) in enumerate(current_station_layers.items()):
                ax_log.axhspan(*data['range'], facecolor=layer_colors[i], alpha=0.22, zorder=0, label=f'Layer {layer_num}')
            ax_log.set_xlabel("ln(Radiance / Irradiance)", weight='bold'); ax_log.set_ylabel("Depth (m)", weight='bold'); ax_log.set_title("3. Log-Linear Profiles", weight='bold')
            ax_log.invert_yaxis(); ax_log.grid(True, linestyle='--'); ax_log.legend(fontsize=5)
            st.pyplot(fig_log, use_container_width=True); st.session_state.fig_log = fig_log
    
        with bottom_row[1]:
            st.selectbox(
                "Select pure-water attenuation model (Kw):",
                ["Morel & Maritorena (2001)", "Smith & Baker (1981)"],
                index=0,
                key="selected_kw_model",
                help=(
                    "The propagated scientific product is limited to 400–700 nm. "
                    "Morel & Maritorena (2001) is the visible-range default; Smith & Baker "
                    "(1981) is available for comparison. The selected model constrains Kd as "
                    "max(profiler Kd, Kw) and Klu as max(profiler Klu, Kw)."
                ),
            )

            fig_k, ax_k = plt.subplots(figsize=plot_figsize)
            
            current_model = st.session_state.get('selected_kw_model', DEFAULT_KW_MODEL)
            df_kw_plot = get_kw_data(current_model)
            kw_interp_plot, kw_plot_coverage = interpolate_kw_without_extrapolation(
                station_data['wavelengths'], df_kw_plot
            )
            current_model_metadata = get_kw_model_metadata(current_model)
            model_min_nm, model_max_nm = current_model_metadata['range_nm']
            
            label_name = f"Kw ({current_model_metadata['short_label']}; {model_min_nm:.0f}–{model_max_nm:.0f} nm)"
            ax_k.plot(station_data['wavelengths'], kw_interp_plot, color='white', linestyle=':', lw=2, label=label_name, zorder=5)
            if np.any(~kw_plot_coverage):
                ax_k.axvspan(
                    model_max_nm,
                    core_spectral_bounds[1],
                    color='#9CA3AF',
                    alpha=0.12,
                    zorder=0,
                    label='Outside selected Kw support',
                )
                st.caption(
                    f"{current_model}: its Kw reference is available only from "
                    f"{model_min_nm:.0f} to {model_max_nm:.0f} nm. No endpoint extrapolation is used."
                )

            if current_station_layers and derived_products:
                layer_k_data = derived_products.get('layer_k_data', {})
                for i, (layer_num, data) in enumerate(current_station_layers.items()):
                    if layer_num in layer_k_data:
                        # Both attenuation spectra are constrained by the selected Kw model.
                        kd_plot, klu_plot = layer_k_data[layer_num]
                        ax_k.plot(station_data['wavelengths'], kd_plot, color=layer_colors[i], linestyle='-', label=f'Kd constrained by Kw (L{layer_num})', zorder=10)
                        ax_k.plot(station_data['wavelengths'], klu_plot, color=layer_colors[i], linestyle='--', label=f'Klu constrained by Kw (L{layer_num})', zorder=10)
            
            ax_k.set_xlabel("Wavelength (nm)", weight='bold')
            ax_k.set_ylabel(r'Attenuation coefficient (m$^{-1}$)', weight='bold')
            ax_k.set_title("5. Profiler attenuation spectra (400–700 nm product domain)", weight='bold')
            ax_k.set_xlim(SCIENTIFIC_PRODUCT_WAVELENGTH_RANGE)
            ax_k.set_ylim(0,1)
            ax_k.grid(True, linestyle='--')
            if current_station_layers: ax_k.legend(fontsize=8)
            st.pyplot(fig_k, use_container_width=True)
            st.caption(
                "NIR measurements above 700 nm remain available for raw-data and SNR checks, "
                "but are excluded from layer-propagated Rrs and SRF convolution."
            )
            st.session_state.fig_k = fig_k
            
        st.subheader("6. Optical attenuation summary by layer")
        if not current_station_layers:
            st.info("Use the layer setup in Core Optics to analyze depth layers for the current profile.")
        else:
            if derived_products:
                attenuation_summary = derived_products['results_df'].merge(
                    derived_products['kd_par_df'][['Layer', 'Kd(PAR)']],
                    on='Layer',
                    how='left',
                ).rename(columns={
                    'Kd(490)': 'Kd(490) — Profiler + Kw constraint',
                    'klu(490)': 'Klu(490) — Profiler + Kw constraint',
                    'Kd(PAR)': 'Kd(PAR) — Profiler',
                })
                st.dataframe(
                    attenuation_summary.set_index('Layer').style.format({
                        'Kd(490) — Profiler + Kw constraint': '{:.4f}',
                        'Klu(490) — Profiler + Kw constraint': '{:.4f}',
                        'Kd(PAR) — Profiler': '{:.4f}',
                    }),
                    use_container_width=True,
                )
                with st.expander("Irradiance provenance used for each Rrs", expanded=False):
                    st.caption(
                        "This table records the requested source and the spectrum actually used "
                        "as the Rrs denominator."
                    )
                    st.dataframe(
                        derived_products['rrs_irradiance_audit'],
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption("Both selected-layer Ed(0+) products are retained for comparison and export.")
                    st.dataframe(
                        derived_products['ed0_propagation_audit'],
                        use_container_width=True,
                        hide_index=True,
                    )
    
        st.markdown("---")
        with st.expander("7. Ed(0) Comparison: Surface vs. Extrapolated/Propagated", expanded=True):
            if not current_station_layers or not derived_products:
                st.warning("Analyze at least one layer to see the comparison plot.")
            else:
                y_max_ed_comp = st.number_input("Set Y-axis Max for Irradiance Plot:", min_value=1.0, value=150.0, step=10.0, format="%.1f", key="ed_comp_ymax_selector")
                
                fig_ed_comp, ax_ed_comp = plt.subplots(figsize=(9.2, 7.2))
                
                # Plot 1: Measured Es (if available) - This is Ed(0+)
                final_es_median = get_active_es(station_data)
                if final_es_median is not None:
                    label_es = "Measured Es (Filtered)" if st.session_state.new_es_median is not None else "Measured Es (Original)"
                    ax_ed_comp.plot(station_data['wavelengths'], final_es_median, color='#62e08a', linestyle='-', label=label_es, lw=3.0, zorder=20)
                
                # Plot 2: Propagated/Extrapolated Ed(0+) for each layer
                layer_k_data = derived_products.get('layer_k_data', {})
                ed0_products = derived_products.get('ed0_df_export', pd.DataFrame())
                active_ed0_mode = st.session_state.get(
                    f"ed0_propagation_mode_{current_station_id}", 'raw_fit'
                )
                for i, (layer_num, data) in enumerate(sorted(current_station_layers.items())):
                    if layer_num in layer_k_data:
                        z_min, z_max = data['range']
                        layer_color = layer_colors[i]
                        # Calculate the in-water extrapolated value, Ed(0-)
                        ed0_minus = extrapolate_ed0_minus(station_data['depth_m'], station_data['Ed_data'], z_min, z_max)
                        
                        # Plot raw Ed(0-) plus both audit-ready Ed(0+) products.
                        if ed0_minus is not None:
                            # Plot the in-water value Ed(0-)
                            ax_ed_comp.plot(
                                station_data['wavelengths'],
                                ed0_minus,
                                color=layer_color,
                                linestyle=':',
                                label=f'L{layer_num} · Ed(0−)',
                                lw=2.0,
                                alpha=0.75,
                                zorder=18,
                            )
                            
                            ed0_plus_raw = (
                                ed0_products[f'layer_{layer_num}_ed0_plus_raw_fit'].to_numpy()
                                if f'layer_{layer_num}_ed0_plus_raw_fit' in ed0_products
                                else ed0_minus / ED_TRANSMISSION_FACTOR
                            )
                            ax_ed_comp.plot(
                                station_data['wavelengths'],
                                ed0_plus_raw,
                                color=layer_color,
                                linestyle='--',
                                label=f'L{layer_num} · raw Ed(0+)',
                                lw=2.8 if active_ed0_mode == 'raw_fit' else 1.4,
                                alpha=1.0 if active_ed0_mode == 'raw_fit' else 0.65,
                                zorder=10,
                            )
                            ed0_plus_kw = (
                                ed0_products[f'layer_{layer_num}_ed0_plus_kw_constrained'].to_numpy()
                                if f'layer_{layer_num}_ed0_plus_kw_constrained' in ed0_products
                                else np.full_like(ed0_plus_raw, np.nan)
                            )
                            ax_ed_comp.plot(
                                station_data['wavelengths'],
                                ed0_plus_kw,
                                color=layer_color,
                                linestyle='-.',
                                label=f'L{layer_num} · Kw Ed(0+)',
                                lw=2.8 if active_ed0_mode == 'kw_constrained' else 1.4,
                                alpha=1.0 if active_ed0_mode == 'kw_constrained' else 0.65,
                                zorder=11,
                            )
    
                # Plot 3: Near-surface Ed from profiler
              #valid_ed_indices = np.where(np.nanmedian(station_data['Ed_data'], axis=1) > 0)[0]
               # if len(valid_ed_indices) > 0:
                #    shallowest_ed = station_data['Ed_data'][valid_ed_indices[0], :]
                 #   ax_ed_comp.plot(station_data['wavelengths'], shallowest_ed, color='white', linestyle=':', label='Near-Surface Ed (Profiler)', lw=2.0, zorder=15)
                
                ax_ed_comp.set_xlabel("Wavelength (nm)", weight='bold')
                ax_ed_comp.set_ylabel(r'Irradiance ($\mu$W cm$^{-2}$ nm$^{-1}$)', weight='bold')
                ax_ed_comp.set_title("Comparison of Surface Irradiance Estimates", weight='bold')
                ax_ed_comp.set_xlim(core_spectral_bounds)
                ax_ed_comp.set_ylim(bottom=0, top=y_max_ed_comp)
                ax_ed_comp.grid(True, linestyle='--')
                place_legend_below_data(fig_ed_comp, ax_ed_comp, max_columns=4, fontsize=8)
                st.pyplot(fig_ed_comp, use_container_width=True)
    
                st.info("""
                **How to interpret this plot:**
                - **Measured Es (Green):** Ground truth `Ed(0+)` from the surface sensor.
                - **Raw-fit Ed(0+) (Dashed):** The observation-preserving log-linear extrapolation from each layer.
                - **Kw-constrained Ed(0+) (Dash-dot):** Uses the same Kd = max(profiler Kd, Kw) constraint as propagation. Both Kd and Klu are constrained by Kw; both Ed(0+) estimates are retained for QC and export.
                - **Extrapolated Ed(0-) (Dotted):** Uses the same layer color as its Ed(0+) counterparts, making the estimates directly comparable. It is the pure in-water surface value before conversion.
                - **Near-Surface Ed (Dotted White):** The shallowest raw measurement from the profiler, for reference.
                """)

        st.markdown("---")
        with st.expander("8. Same-layer Rrs sensitivity: measured Es vs. layer Ed(0+)", expanded=True):
            st.caption(
                "This diagnostic fixes Kd, Klu, and the optical layer. It isolates the effect of "
                "the irradiance denominator: Es versus Ed(0+) extrapolated from that same layer."
            )
            available_es_modes = []
            if st.session_state.get('new_es_median') is not None:
                available_es_modes.append('filtered_es')
            if station_data.get('Es') is not None:
                available_es_modes.append('original_es')

            sensitivity_layers = sorted(
                layer_num for layer_num in current_station_layers
                if layer_num in (derived_products or {}).get('layer_k_data', {})
            )
            if not sensitivity_layers:
                st.info("Define a layer with valid Kd and Klu to run this same-layer comparison.")
            elif not available_es_modes:
                st.info("Upload or derive a valid measured Es spectrum to compare it with layer Ed(0+).")
            else:
                sensitivity_controls = st.columns(3)
                sensitivity_layer_num = sensitivity_controls[0].selectbox(
                    "Layer to compare:",
                    options=sensitivity_layers,
                    format_func=lambda layer_num: f"Layer {layer_num}",
                    key=f"rrs_sensitivity_layer_{current_station_id}",
                )
                sensitivity_es_mode = sensitivity_controls[1].selectbox(
                    "Measured Es spectrum:",
                    options=available_es_modes,
                    format_func=lambda mode: IRRADIANCE_MODE_LABELS[mode],
                    key=f"rrs_sensitivity_es_{current_station_id}",
                )
                sensitivity_ed0_mode = sensitivity_controls[2].selectbox(
                    "Layer Ed(0+) treatment:",
                    options=list(ED0_PROPAGATION_MODE_LABELS),
                    format_func=lambda mode: ED0_PROPAGATION_MODE_LABELS[mode],
                    key=f"rrs_sensitivity_ed0_{current_station_id}",
                )

                sensitivity_layer = current_station_layers[sensitivity_layer_num]
                sensitivity_kd, sensitivity_klu = derived_products['layer_k_data'][sensitivity_layer_num]
                sensitivity_ed0_kd_constraint = (
                    derived_products.get('layer_kw_constrained_kd', {}).get(sensitivity_layer_num)
                    if sensitivity_ed0_mode == 'kw_constrained' else None
                )
                sensitivity = calculate_same_layer_rrs_sensitivity(
                    station_data,
                    sensitivity_layer['range'][0],
                    sensitivity_layer['range'][1],
                    sensitivity_kd,
                    sensitivity_klu,
                    sensitivity_es_mode,
                    sensitivity_ed0_mode,
                    ed0_kd_constraint=sensitivity_ed0_kd_constraint,
                )
                sensitivity_wavelengths = sensitivity['wavelength_nm']
                rrs_es_sensitivity = sensitivity['rrs_es']
                rrs_ed_sensitivity = sensitivity['rrs_ed0_plus']
                relative_difference = sensitivity['relative_difference_percent']
                sensitivity_mask = (
                    np.isfinite(sensitivity_wavelengths)
                    & (sensitivity_wavelengths >= 400)
                    & (sensitivity_wavelengths <= SCIENTIFIC_PRODUCT_WAVELENGTH_RANGE[1])
                    & np.isfinite(rrs_es_sensitivity)
                    & np.isfinite(rrs_ed_sensitivity)
                    & np.isfinite(relative_difference)
                )

                if not np.any(sensitivity_mask):
                    st.warning("The selected Es or layer Ed(0+) spectrum has no valid overlap from 400 to 700 nm.")
                else:
                    absolute_difference = rrs_ed_sensitivity - rrs_es_sensitivity
                    max_absolute_difference = np.nanmax(
                        np.abs(absolute_difference[sensitivity_mask])
                    )
                    median_absolute_percent = np.nanmedian(
                        np.abs(relative_difference[sensitivity_mask])
                    )
                    numerically_identical = np.allclose(
                        rrs_es_sensitivity[sensitivity_mask],
                        rrs_ed_sensitivity[sensitivity_mask],
                        rtol=1e-12,
                        atol=1e-14,
                    )
                    sensitivity_metrics = st.columns(3)
                    sensitivity_metrics[0].metric(
                        "Max |ΔRrs|",
                        f"{max_absolute_difference:.3e} sr⁻¹",
                    )
                    sensitivity_metrics[1].metric(
                        "Median |ΔRrs| / Rrs(Es)",
                        f"{median_absolute_percent:.3g}%",
                    )
                    sensitivity_metrics[2].metric(
                        "Comparison layer",
                        f"L{sensitivity_layer_num} ({sensitivity_layer['range'][0]:.1f}–{sensitivity_layer['range'][1]:.1f} m)",
                    )
                    st.caption(
                        f"Es denominator: {sensitivity['es_source']}  |  "
                        f"Ed denominator: {sensitivity['ed0_plus_source']}"
                    )
                    if numerically_identical:
                        st.warning(
                            "Rrs(Es) and Rrs(Ed 0+) are numerically identical at the displayed wavelengths. "
                            "This indicates that the two denominator spectra are identical or differ below the "
                            "calculation precision; it is not caused by Kd or Klu being the same."
                        )
                    else:
                        st.success(
                            "The two Rrs spectra differ. The plot and table below show the denominator-only "
                            "sensitivity for this fixed layer."
                        )

                    fig_rrs_sensitivity, (ax_rrs_sensitivity, ax_rrs_difference) = plt.subplots(
                        1, 2, figsize=(12, 4.4), sharex=True,
                    )
                    ax_rrs_sensitivity.plot(
                        sensitivity_wavelengths[sensitivity_mask],
                        rrs_es_sensitivity[sensitivity_mask],
                        color='#f6bd60', lw=2.4, label='Rrs using measured Es',
                    )
                    ax_rrs_sensitivity.plot(
                        sensitivity_wavelengths[sensitivity_mask],
                        rrs_ed_sensitivity[sensitivity_mask],
                        color='#4cc9f0', lw=2.0, linestyle='--', label='Rrs using layer Ed(0+)',
                    )
                    ax_rrs_sensitivity.set_title('Same propagated Lw; alternative denominators', weight='bold')
                    ax_rrs_sensitivity.set_ylabel(r'Rrs (sr$^{-1}$)', weight='bold')
                    ax_rrs_sensitivity.legend(fontsize=8)

                    ax_rrs_difference.axhline(0, color='white', lw=1, alpha=0.7)
                    ax_rrs_difference.plot(
                        sensitivity_wavelengths[sensitivity_mask],
                        relative_difference[sensitivity_mask],
                        color='#d8f3dc', lw=2.2,
                    )
                    ax_rrs_difference.set_title(r'100 × [Rrs(Ed 0+) − Rrs(Es)] / Rrs(Es)', weight='bold')
                    ax_rrs_difference.set_ylabel('Difference (%)', weight='bold')
                    for axis in (ax_rrs_sensitivity, ax_rrs_difference):
                        axis.set_xlabel('Wavelength (nm)', weight='bold')
                        axis.set_xlim(SCIENTIFIC_PRODUCT_WAVELENGTH_RANGE)
                        axis.grid(True, linestyle=':', alpha=0.7)
                    st.pyplot(fig_rrs_sensitivity, use_container_width=True)
                    plt.close(fig_rrs_sensitivity)

                    key_wavelengths = [490, 555, 600]
                    sensitivity_rows = []
                    for target_wavelength in key_wavelengths:
                        wavelength_index = int(np.abs(sensitivity_wavelengths - target_wavelength).argmin())
                        sensitivity_rows.append({
                            'Wavelength (nm)': sensitivity_wavelengths[wavelength_index],
                            'Measured Es': sensitivity['es_spectrum'][wavelength_index],
                            'Layer Ed(0+)': sensitivity['ed0_plus_spectrum'][wavelength_index],
                            'Δ irradiance (Ed - Es)': (
                                sensitivity['ed0_plus_spectrum'][wavelength_index]
                                - sensitivity['es_spectrum'][wavelength_index]
                            ),
                            'Rrs with Es (sr-1)': rrs_es_sensitivity[wavelength_index],
                            'Rrs with layer Ed(0+) (sr-1)': rrs_ed_sensitivity[wavelength_index],
                            'ΔRrs (Ed - Es) (sr-1)': absolute_difference[wavelength_index],
                            'ΔRrs / Rrs(Es) (%)': relative_difference[wavelength_index],
                        })
                    st.dataframe(
                        pd.DataFrame(sensitivity_rows).style.format({
                            'Wavelength (nm)': '{:.1f}',
                            'Measured Es': '{:.6f}',
                            'Layer Ed(0+)': '{:.6f}',
                            'Δ irradiance (Ed - Es)': '{:.3e}',
                            'Rrs with Es (sr-1)': '{:.10f}',
                            'Rrs with layer Ed(0+) (sr-1)': '{:.10f}',
                            'ΔRrs (Ed - Es) (sr-1)': '{:.3e}',
                            'ΔRrs / Rrs(Es) (%)': '{:.6f}',
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )

    # ------------------ TAB 5: T-S DIAGRAM AND WATER MASS ANALYSIS ------------------
    with tab_ts:
        st.header("T-S Diagram and Water Mass Analysis")
        
        # Retrieves selected columns from session state for external probe data

        t_col = st.session_state.get('p1_t', "None")
        s_col = st.session_state.get('p1_s', "None")
        z_col = st.session_state.get('p1_z', "None")
        st_col = st.session_state.get('p1_st', "None")
        
        df_ext_ts = None
        ext_station_label = "None"

        col_plot, col_controls = st.columns([2, 1])

        with col_controls:
            st.subheader("Configuration & Linkage")
            
            # Logic to link external CTD data to the current profiler station

            if st.session_state.external_probe_df is not None:
                if t_col != "None" and s_col != "None" and st_col != "None":
                    st.markdown("### Link External Probe Data")
                    opcoes_probe = list(st.session_state.external_probe_df[st_col].dropna().unique().astype(str))
                    
                    # Uses fuzzy matching to suggest the best external station match

                    sugestao = find_best_match(st.session_state.selected_station, opcoes_probe)
                    
                    try: idx_inicial = opcoes_probe.index(str(sugestao))
                    except: idx_inicial = 0

                    estacao_probe_manual = st.selectbox(
                        "Select corresponding External Station:",
                        options=opcoes_probe,
                        index=idx_inicial,
                        key="ts_manual_probe_select"
                    )

                    df_target_ext = st.session_state.external_probe_df[
                        st.session_state.external_probe_df[st_col].astype(str) == estacao_probe_manual
                    ].copy()
                    
                    # Prepares the external T-S dataframe

                    df_ext_ts = pd.DataFrame({
                        'Temperature': pd.to_numeric(df_target_ext[t_col], errors='coerce'),
                        'Salinity': pd.to_numeric(df_target_ext[s_col], errors='coerce'),
                        'Depth': pd.to_numeric(df_target_ext[z_col], errors='coerce') if z_col != "None" else 0
                    }).dropna(subset=['Temperature', 'Salinity'])
                    
                    ext_station_label = estacao_probe_manual
                    st.success(f"Linked to: {ext_station_label}")

            st.divider()

            # WATER MASS VERTEX DEFINITION
            with st.expander("Water Mass End-Member Definition", expanded=True):
                st.info("Define the Temperature and Salinity of your regional water masses to calculate mixing fractions.")
                c1, c2, c3 = st.columns(3)
                for i in range(3):
                    st.session_state.wm_names[i] = c1.text_input(f"Name M{i+1}", value=st.session_state.wm_names[i], key=f"n{i}")
                
                # Standard vertices (e.g., Tropical Water, South Atlantic Central Water, Coastal Water)

                wm1_s = c2.number_input("Sal 1", value=37.20, format="%.2f", key="s1")
                wm1_t = c3.number_input("Temp 1", value=25.29, format="%.2f", key="t1")
                wm2_s = c2.number_input("Sal 2", value=35.19, format="%.2f", key="s2")
                wm2_t = c3.number_input("Temp 2", value=12.67, format="%.2f", key="t2")
                wm3_s = c2.number_input("Sal 3", value=34.36, format="%.2f", key="s3")
                wm3_t = c3.number_input("Temp 3", value=27.54, format="%.2f", key="t3")
                
                if st.button("Run Mixing Analysis", type="primary", use_container_width=True):
                    
                    # Triggers the barycentric coordinate calculation for both sensors
                    st.session_state.wm_vertices = [(wm1_s, wm1_t), (wm2_s, wm2_t), (wm3_s, wm3_t)]
                    st.session_state.mixture_results_profiler, st.session_state.mixture_df_profiler = analyze_mixture(
                        phys_props_df['Salinity'], phys_props_df['Temperature'], 
                        st.session_state.wm_vertices, st.session_state.wm_names
                    )
                    if df_ext_ts is not None:
                        st.session_state.mixture_results_external, st.session_state.mixture_df_external = analyze_mixture(
                            df_ext_ts['Salinity'], df_ext_ts['Temperature'], 
                            st.session_state.wm_vertices, st.session_state.wm_names
                        )
                    st.rerun()

            if st.session_state.mixture_df_profiler is not None:
                st.markdown("### Average Fractions (%)")
                df_avg = (st.session_state.mixture_df_profiler.mean() * 100).to_frame(name="Profiler")
                if st.session_state.mixture_df_external is not None:
                    df_avg["External Probe"] = st.session_state.mixture_df_external.mean() * 100
                st.dataframe(df_avg.style.format("{:.1f}%"), use_container_width=True)

        with col_plot:
            fig_ts, ax_ts = plt.subplots(figsize=(8, 8))
            df_int = phys_props_df[['Salinity', 'Temperature', 'Depth']].dropna()
            
            # DYNAMIC AXIS LIMITS (Data + Vertices)
            s_list = [df_int['Salinity'].min(), df_int['Salinity'].max()]
            t_list = [df_int['Temperature'].min(), df_int['Temperature'].max()]
            if df_ext_ts is not None and not df_ext_ts.empty:
                s_list.extend([df_ext_ts['Salinity'].min(), df_ext_ts['Salinity'].max()])
                t_list.extend([df_ext_ts['Temperature'].min(), df_ext_ts['Temperature'].max()])
            if st.session_state.wm_vertices:
                v_s = [v[0] for v in st.session_state.wm_vertices]
                v_t = [v[1] for v in st.session_state.wm_vertices]
                s_list.extend([min(v_s), max(v_s)])
                t_list.extend([min(v_t), max(v_t)])

            s_min, s_max = min(s_list), max(s_list)
            t_min, t_max = min(t_list), max(t_list)

            # ISOPYCNAL GRID (Density Lines)
            s_grid_ax = np.linspace(s_min - 1.0, s_max + 1.0, 100)
            t_grid_ax = np.linspace(t_min - 1.5, t_max + 1.5, 100)
            S_mesh, T_mesh = np.meshgrid(s_grid_ax, t_grid_ax)
            sigma_theta = gsw.sigma0(S_mesh, T_mesh)
            cnt = ax_ts.contour(S_mesh, T_mesh, sigma_theta, colors=STYLE_CONFIG['gridcolor'], alpha=0.3, linestyles='dashed', zorder=1)
            ax_ts.clabel(cnt, inline=True, fontsize=8, fmt='%.1f')
            
            # DATA PLOTTING
            if df_ext_ts is not None and not df_ext_ts.empty:
                ax_ts.scatter(df_ext_ts['Salinity'], df_ext_ts['Temperature'], c='grey', s=12, alpha=0.4, label=f'Probe: {ext_station_label}', zorder=2)
            
            # Profiler data colored by depth to show vertical structure

            sc = ax_ts.scatter(df_int['Salinity'], df_int['Temperature'], c=df_int['Depth'], cmap='viridis_r', s=45, ec='black', lw=0.5, label='Perfilador (Profiler)', zorder=3)
            cbar = plt.colorbar(sc, ax=ax_ts, pad=0.02)
            cbar.set_label('Profundidade (m)', weight='bold'); cbar.ax.invert_yaxis()
            
            # MIXING TRIANGLE OVERLAY
            if st.session_state.wm_vertices:
               poly = Polygon(np.array(st.session_state.wm_vertices), closed=True, fill=False, edgecolor='cyan', ls='--', lw=2, zorder=4)
               ax_ts.add_patch(poly)
               for i, (sal, temp) in enumerate(st.session_state.wm_vertices):
                   ax_ts.text(sal, temp, f" {st.session_state.wm_names[i]}", color='cyan', fontsize=12, fontweight='bold', zorder=5)
            
            ax_ts.set_xlim(s_min - 0.3, s_max + 0.3)
            ax_ts.set_ylim(t_min - 0.5, t_max + 0.5)
            ax_ts.set_xlabel("Absolute Salinity (g/kg)", weight='bold')
            ax_ts.set_ylabel("Conservative Temperature (°C)", weight='bold')
            ax_ts.set_title(f"T-S Diagram: {st.session_state.selected_station}", weight='bold', pad=15)
            ax_ts.grid(True, linestyle=':', alpha=0.2)
            ax_ts.legend(loc='lower right', fontsize=9, framealpha=0.8)
            
            st.pyplot(fig_ts)
            st.session_state.fig_ts = fig_ts

        # VERTICAL MIXING PROFILES
        st.divider()
        st.subheader("Vertical Mixing Profiles (%)")
        m1, m2 = st.columns(2)
        
        with m1:
            st.markdown(f"**Profiler ({st.session_state.selected_station})**")
            if st.session_state.mixture_df_profiler is not None:
                fig_vp, ax_vp = plt.subplots(figsize=(5, 7))
                for name in st.session_state.wm_names: 
                    ax_vp.plot(st.session_state.mixture_df_profiler[name]*100, phys_props_df['Depth'], label=name, lw=2.5)
                ax_vp.invert_yaxis()
                ax_vp.set_xlabel("Proportion (%)")
                ax_vp.set_ylabel("Depth (m)")
                ax_vp.set_xlim(0, 105)
                ax_vp.legend()
                ax_vp.grid(True, ls=':')
                st.pyplot(fig_vp)
                st.session_state.fig_vp = fig_vp
            else:
                st.info("Run mixture analysis to visualize.")

        with m2:
            
            # Fraction vs Depth for the External Probe (e.g., C3/Castaway)
            st.markdown(f"**External Probe ({ext_station_label})**")
            if st.session_state.mixture_df_external is not None and df_ext_ts is not None:
                fig_ve, ax_ve = plt.subplots(figsize=(5, 7))
                z_ext = df_ext_ts['Depth']
                for name in st.session_state.wm_names: 
                    ax_ve.plot(st.session_state.mixture_df_external[name]*100, z_ext, label=name, lw=2.5)
                ax_ve.invert_yaxis()
                ax_ve.set_xlabel("Proportion (%)")
                ax_ve.set_xlim(0, 105)
                ax_ve.legend()
                ax_ve.grid(True, ls=':')
                st.pyplot(fig_ve)
                st.session_state.fig_ve = fig_ve
            else:
                st.info("External probe data unavailable or not calculated.")
                
                
    
    # ------------------ TAB 6: BIO-OPTICAL MODELS ------------------
    with tab_models:
        st.header("Bio-Optical Model Estimations")
        st.subheader("Current optical-pipeline inputs")
        current_model_data = st.session_state.empirical_model_data
        current_station_model_data = current_model_data[
            current_model_data['Station_ID'] == station_id
        ] if not current_model_data.empty else pd.DataFrame()
        pipeline_col1, pipeline_col2, pipeline_col3 = st.columns(3)
        pipeline_col1.metric("Active station", station_id)
        pipeline_col2.metric("Defined layers", len(current_station_layers))
        pipeline_col3.metric("Model-ready observations", len(current_station_model_data))
        if current_station_model_data.empty:
            st.info(
                "Define and analyze at least one layer in Core Optics to send its Kd(PAR) and Kd(490) "
                "to the site-specific empirical-model dataset."
            )
        else:
            st.success(
                "The current profile's layer results are already connected to the empirical-model dataset."
            )
            with st.expander("Inspect current station inputs and Rrs provenance", expanded=False):
                st.dataframe(current_station_model_data, use_container_width=True, hide_index=True)
                if derived_products and 'rrs_irradiance_audit' in derived_products:
                    st.caption("Rrs irradiance provenance for the current optical run")
                    st.dataframe(
                        derived_products['rrs_irradiance_audit'],
                        use_container_width=True,
                        hide_index=True,
                    )
        st.divider()
        
        st.subheader("1. Site-Specific Empirical Model: Kd(490) vs. Kd(PAR)")
        with st.expander("Generate and Manage Your Empirical Model", expanded=True):
            st.info("""
            Build a custom model using your profiler's data. This allows for persistent, long-term analysis.
            - **Load Data:** Start with your master model CSV from previous sessions.
            - **Analyze Layers:** Define layers in Core Optics to add new data points automatically.
            - **Generate Model:** Create an updated regression model.
            - **Download Data:** Save the combined old and new data to your master CSV for future use.
            """)
    
            st.markdown("#### Manage Model Data")
            col1, col2 = st.columns(2)
            with col1:
                # Callback to handle historical model data loading

                st.file_uploader(
                    "Load Master Model Data (CSV)", 
                    type="csv", 
                    key="master_model_loader",
                    on_change=_handle_model_upload
                )
            
            with col2:
                # Export current training data for future sessions

                if not st.session_state.empirical_model_data.empty:
                    st.download_button(
                        label="Download Model Data (CSV)",
                        data=st.session_state.empirical_model_data.to_csv(index=False).encode('utf-8'),
                        file_name="master_empirical_model_data.csv",
                        mime='text/csv',
                        key="download_model_data"
                    )
            st.markdown("---")
    
            if st.session_state.empirical_model_data.empty:
                st.warning("No data available yet. Analyze at least two layers with your profiler, or load a master model data file.")
            else:
                st.markdown("**Current Data for Model:**")
                
                
                # 1. Define column order
                desired_order = ['Station_ID', 'Kd(PAR)', 'Kd(490)','Layer']
                
                # 2. Create a new DataFrame with the columns in that order.
                # We also check if all desired columns exist to prevent errors.
                cols_to_display = [col for col in desired_order if col in st.session_state.empirical_model_data.columns]
                df_to_display = st.session_state.empirical_model_data[cols_to_display]
                
                # 3. Display the correctly ordered DataFrame.
                st.dataframe(df_to_display, use_container_width=True)
    
                if st.button("Generate Empirical Model from Current Data"):
                    df_model = st.session_state.empirical_model_data.dropna()
                    if len(df_model) < 2:
                        st.error("At least two valid data points are required to build a model.")
                    else:
                        # Performs linear regression to find site-specific relationship

                        slope, intercept, r_value, p_value, _ = linregress(df_model['Kd(PAR)'], df_model['Kd(490)'])
                        st.session_state.empirical_model_params = {'slope': slope, 'intercept': intercept, 'r2': r_value**2}
                        
                        st.success("Empirical model generated successfully!")
                        st.metric("Model Equation", f"Kd(490) = {slope:.4f} * Kd(PAR) + {intercept:.4f}")
                        st.metric("Model R²", f"{r_value**2:.4f}")
                        
                        # Plot the empirical regression line
                        fig_model, ax = plt.subplots()
                        ax.scatter(df_model['Kd(PAR)'], df_model['Kd(490)'], ec='white', label='Data Points')
                        x_vals = np.array(df_model['Kd(PAR)'])
                        y_vals = slope * x_vals + intercept
                        ax.plot(x_vals, y_vals, '--', color='cyan', label='Regression Line')
                        ax.set_xlabel("Kd(PAR) (m⁻¹)", weight='bold'); ax.set_ylabel("Kd(490) (m⁻¹)", weight='bold')
                        ax.set_title("Empirical Model: Kd(PAR) vs. Kd(490)", weight='bold'); ax.grid(True, linestyle=':'); ax.legend()
                        st.pyplot(fig_model)
    
        st.subheader("2. Literature Model (Morel & Maritorena, 2001)")
        with st.expander("Estimate Kd(490) from Kd(PAR) using Published Model"):
            st.markdown("""This model is most accurate for open-ocean (Case-1) waters and may differ from your site-specific model.""")
            kd_par_input = st.number_input("Enter a Kd(PAR) value (m⁻¹):", min_value=0.0, value=0.1, step=0.01, format="%.4f")
            if kd_par_input > 0:
                # Estimate Chl-a and then Kd(490) using literature coefficients

                estimated_c = estimate_chlorophyll_from_kdpar(kd_par_input)
                estimated_kd490 = estimate_kd490_from_chlorophyll(estimated_c)
                col1, col2 = st.columns(2)
                with col1: st.metric(label="Estimated Chlorophyll-a (mg/m³)", value=f"{estimated_c:.4f}")
                with col2: st.metric(label="Estimated Kd(490) (m⁻¹)", value=f"{estimated_kd490:.4f}")
  
   
    # ------------------ TAB 7: SECCHI & KD ANALYSIS ------------------

    with tab_secchi:
        st.header("Secchi & Kd Analysis")

        # --- GERENCIAMENTO DE DADOS HISTÓRICOS (MASTER) ---
        with st.expander("Manage Campaign Master Database (Historical Data)", expanded=False):
            st.info("Merge data from different days or download the current progress.")
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                st.file_uploader(
                    "Upload Master Secchi/Kd (CSV)", 
                    type="csv", 
                    key="master_secchi_loader",
                    on_change=_handle_master_secchi_upload,
                    help="Upload a previously saved .csv to continue historical analysis."
                )
            
            with col_m2:
                if not st.session_state.master_kd_secchi_df.empty:
                    csv_data = st.session_state.master_kd_secchi_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Master Campaign CSV",
                        data=csv_data,
                        file_name="Master_Campaign_Secchi_Kd.csv",
                        mime='text/csv',
                        use_container_width=True
                    )
                else:
                    st.button("Download base (empty) master csv", disabled=True, use_container_width=True)

        st.divider()

        # LIVE DATA PREVIEW
        st.subheader("1. Active Station Preview")
        if st.session_state.get('comparison_kd_df') is not None:
            current_layer = st.session_state.results_df.iloc[0]['Layer'] if st.session_state.results_df is not None else "N/A"
            p_col1, p_col2, p_col3 = st.columns(3)
            try:
                val_prof = st.session_state.comparison_kd_df.loc[current_layer, "Kd(PAR) (Profiler)"]
                val_ext = st.session_state.comparison_kd_df.loc[current_layer, "Kd(PAR) (External)"]
                val_490 = st.session_state.comparison_kd_df.loc[current_layer, "Kd(490) (Profiler)"]
                
                p_col1.metric(f"Kd(PAR) Profiler ({current_layer})", f"{val_prof:.4f} m⁻¹")
                p_col2.metric(f"Kd(PAR) External ({current_layer})", f"{val_ext:.4f} m⁻¹" if not pd.isna(val_ext) else "No Data")
                p_col3.metric(f"Kd(490) Profiler ({current_layer})", f"{val_490:.4f} m⁻¹")
            except:
                st.info("Define layers in Tab 4 to see metrics.")
        else:
            st.info("Link an external probe in Tab 2 to enable cross-validation.")
    
        st.divider()

        # LINK & SAVE
        st.subheader("2. Link Secchi & Save to Campaign")
        
        secchi_input = np.nan
        if st.session_state.external_probe_df is not None:
            
            # Logic to automatically pull Secchi values from the external probe file based on station ID

            df_s = st.session_state.external_probe_df
            st_col = st.session_state.get('p1_st')
            val_col = st.session_state.get('p1_sec')
            
            if st_col != "None" and val_col != "None" and st_col in df_s.columns:
                opcoes_probe = list(df_s[st_col].dropna().unique().astype(str))
                sugestao = find_best_match(st.session_state.selected_station, opcoes_probe)
                try: idx_sug = opcoes_probe.index(str(sugestao))
                except: idx_sug = 0

                c_vinc, c_met = st.columns([2, 1])
                with c_vinc:
                    vinc_estacao = st.selectbox("Link Secchi from Station (External Probe):", options=opcoes_probe, index=idx_sug, key="secchi_station_link_select")
                with c_met:
                    entry_mode = st.radio("Method:", ["Automated", "Manual"], horizontal=True)

                df_row = df_s[df_s[st_col].astype(str) == vinc_estacao]
                auto_val = pd.to_numeric(df_row.iloc[0][val_col], errors='coerce') if not df_row.empty else np.nan

                if entry_mode == "Automated":
                    if not pd.isna(auto_val):
                        st.success(f"Value found: **{auto_val:.2f} m**")
                        secchi_input = auto_val
                    else:
                        st.error("No Secchi value found for this station ID.")
                else:
                    secchi_input = st.number_input("Manually Enter Secchi Depth (m):", min_value=0.0, step=0.1, format="%.2f")
            else:
                secchi_input = st.number_input("Manually Enter Secchi Depth (m):", min_value=0.0, step=0.1, format="%.2f")
        else:
            secchi_input = st.number_input("Manually Enter Secchi Depth (m):", min_value=0.0, step=0.1, format="%.2f")

        
        if st.button("Commit Current Station to Master Database", type="primary", use_container_width=True):
            
            # Saves a single station's optical and Secchi data to the cumulative campaign dataframe
            if not pd.isna(secchi_input) and st.session_state.results_df is not None:
                layer_key = st.session_state.results_df.iloc[0]['Layer']
                new_row = pd.DataFrame({
                    'Station_ID': [st.session_state.selected_station], 
                    'Secchi': [secchi_input],
                    'Kd(PAR)_Profiler': [st.session_state.kd_par_df.iloc[0]['Kd(PAR)']], 
                    'Kd(490)_Profiler': [st.session_state.results_df.iloc[0]['Kd(490)']],
                    'Kd(PAR)_External': [st.session_state.comparison_kd_df.loc[layer_key, "Kd(PAR) (External)"] if st.session_state.comparison_kd_df is not None else np.nan]
                })
                
                # Mescla e evita duplicatas na sessão
                m_sec = st.session_state.master_kd_secchi_df
                st.session_state.master_kd_secchi_df = pd.concat([m_sec[m_sec['Station_ID'] != st.session_state.selected_station], new_row], ignore_index=True)
                st.toast(f"Station {st.session_state.selected_station} saved!")
            else:
                st.error("Double-check for Kd and Secchi data validity ")

        st.divider()

        st.subheader("3. Campaign Master Database")
        df_master = st.session_state.master_kd_secchi_df
        if not df_master.empty:
            st.dataframe(df_master.style.format({'Secchi': '{:.2f}', 'Kd(PAR)_Profiler': '{:.4f}', 'Kd(490)_Profiler': '{:.4f}', 'Kd(PAR)_External': '{:.4f}'}), use_container_width=True)
            
            st.subheader("4. Regression Analysis: Secchi vs. Kd")
            
            fig_secchi, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            d_Profiler = df_master[['Secchi', 'Kd(PAR)_Profiler']].dropna()
            if not d_Profiler.empty:
                ax1.scatter(d_Profiler['Secchi'], d_Profiler['Kd(PAR)_Profiler'], 
                            color='#00E5FF', ec='white', s=100, marker='o', label='Profiler (Profiler)', zorder=5)
                
                if len(d_Profiler) >= 2 and d_Profiler['Secchi'].nunique() > 1:
                    m_j = get_regression_metrics(d_Profiler['Secchi'], d_Profiler['Kd(PAR)_Profiler'])
                    x_fit = np.array([d_Profiler['Secchi'].min(), d_Profiler['Secchi'].max()])
                    ax1.plot(x_fit, m_j['Intercept'] + m_j['Slope'] * x_fit, 
                             '--', color='#00E5FF', alpha=0.8, label=f'Fit Profiler (R²={m_j["R²"]:.2f})')

            d_ext = df_master[['Secchi', 'Kd(PAR)_External']].dropna()
            if not d_ext.empty:
                ax1.scatter(d_ext['Secchi'], d_ext['Kd(PAR)_External'], 
                            color='#FF8C00', ec='white', s=100, marker='^', label='External Probe', zorder=5)
                
                if len(d_ext) >= 2 and d_ext['Secchi'].nunique() > 1:
                    m_e = get_regression_metrics(d_ext['Secchi'], d_ext['Kd(PAR)_External'])
                    x_fit_e = np.array([d_ext['Secchi'].min(), d_ext['Secchi'].max()])
                    ax1.plot(x_fit_e, m_e['Intercept'] + m_e['Slope'] * x_fit_e, 
                             ':', color='#FF8C00', alpha=0.8, label=f'Fit Probe (R²={m_e["R²"]:.2f})')
            
            ax1.set_xlabel("Secchi Depth (m)", weight='bold')
            ax1.set_ylabel(r"$K_d(PAR)$ ($m^{-1}$)", weight='bold')
            ax1.set_title("Secchi Depth vs. Kd(PAR)", weight='bold')
            ax1.legend(loc='upper right', fontsize=9, framealpha=0.6)
            ax1.grid(True, ls=':', alpha=0.4)

            d_490 = df_master[['Secchi', 'Kd(490)_Profiler']].dropna()
            if not d_490.empty:
                ax2.scatter(d_490['Secchi'], d_490['Kd(490)_Profiler'], 
                            color='#39FF14', ec='white', s=100, marker='s', label='Kd(490) Profiler', zorder=5)
                
                if len(d_490) >= 2 and d_490['Secchi'].nunique() > 1:
                    m4 = get_regression_metrics(d_490['Secchi'], d_490['Kd(490)_Profiler'])
                    x_fit4 = np.array([d_490['Secchi'].min(), d_490['Secchi'].max()])
                    ax2.plot(x_fit4, m4['Intercept'] + m4['Slope'] * x_fit4, 
                             '--', color='#39FF14', alpha=0.8, label=f'R²={m4["R²"]:.2f}')
            
            ax2.set_xlabel("Secchi Depth (m)", weight='bold')
            ax2.set_ylabel(r"$K_d(490)$ ($m^{-1}$)", weight='bold')
            ax2.set_title("Secchi Depth vs. Kd(490)", weight='bold')
            ax2.legend(loc='upper right', fontsize=9, framealpha=0.6)
            ax2.grid(True, ls=':', alpha=0.4)
            
            plt.tight_layout()
            st.pyplot(fig_secchi)
            
            if df_master['Secchi'].nunique() <= 1:
                st.warning("Regression line will appear once you have at least 2 stations with different Secchi depths.")
                
    # st.subheader("Generate L3 Ensemble Average Product")
    # ensemble_casts = st.multiselect("Select validated profiles to include in ensemble average:", options=st.session_state.comparison_table.index, default=list(st.session_state.comparison_table.index))
    
    # any_layers_defined = any(st.session_state.profile_specific_layers.get(cast) for cast in ensemble_casts)
    # disable_ensemble = (len(ensemble_casts) < 2) or not any_layers_defined
    
    # if len(ensemble_casts) < 2:
    #     st.info("Select at least two profiles above to generate an ensemble average.")
    # elif not any_layers_defined:
    #     st.warning("Please analyze at least one layer for one of the selected profiles to generate an ensemble product.")
    
    # if st.button("Generate & Download Ensemble Average Package", disabled=disable_ensemble):
    #     with st.spinner("Calculating ensemble average and building package..."):
    #         base_name = f"L3_Ensemble_Average_{len(ensemble_casts)}_profiles"
    #         zip_buffer_ens = io.BytesIO()
    #         with zipfile.ZipFile(zip_buffer_ens, 'w', zipfile.ZIP_DEFLATED) as zip_file:
    #             create_full_report_zip(zip_file, base_name)
                
    #             min_depth_ens, max_depth_ens = float('inf'), float('-inf')
    #             for cast_id in ensemble_casts:
    #                 pressure = st.session_state.station_data[cast_id]['pressure']
    #                 if pressure.size > 0:
    #                     min_depth_ens = min(min_depth_ens, pressure.min())
    #                     max_depth_ens = max(max_depth_ens, pressure.max())
                
    #             ensemble_depth_grid = np.arange(np.ceil(min_depth_ens*10)/10, np.floor(max_depth_ens*10)/10 + 0.1, 0.1)
    #             interp_ed_list, interp_lu_list = [], []
    #             wavelengths = st.session_state.station_data[ensemble_casts[0]]['wavelengths']
                
    #             for cast_id in ensemble_casts:
    #                 cast_data = st.session_state.station_data[cast_id]
    #                 interp_ed = np.array([np.interp(ensemble_depth_grid, cast_data['pressure'], cast_data['Ed_data'][:, i]) for i in range(len(wavelengths))]).T
    #                 interp_lu = np.array([np.interp(ensemble_depth_grid, cast_data['pressure'], cast_data['Lu_data'][:, i]) for i in range(len(wavelengths))]).T
    #                 interp_ed_list.append(interp_ed)
    #                 interp_lu_list.append(interp_lu)
                
    #             ed_stack, lu_stack = np.stack(interp_ed_list, axis=0), np.stack(interp_lu_list, axis=0)
    #             ed_mean, ed_std = np.mean(ed_stack, axis=0), np.std(ed_stack, axis=0)
    #             lu_mean, lu_std = np.mean(lu_stack, axis=0), np.std(lu_stack, axis=0)
    #             wvl_cols = [f"{int(w)}nm" for w in wavelengths]
                
    #             df_ed = pd.DataFrame(columns=['Depth'] + [f'{c}_{s}' for c in wvl_cols for s in ['mean', 'std']]); df_ed['Depth'] = ensemble_depth_grid; df_ed[[f'{c}_mean' for c in wvl_cols]] = ed_mean; df_ed[[f'{c}_std' for c in wvl_cols]] = ed_std
    #             df_lu = pd.DataFrame(columns=['Depth'] + [f'{c}_{s}' for c in wvl_cols for s in ['mean', 'std']]); df_lu['Depth'] = ensemble_depth_grid; df_lu[[f'{c}_mean' for c in wvl_cols]] = lu_mean; df_lu[[f'{c}_std' for c in wvl_cols]] = lu_std
    #             with np.errstate(divide='ignore', invalid='ignore'): lu_ed_ratio_mean = np.where(ed_mean > 0, lu_mean / ed_mean, np.nan)
    #             df_lu_ed = pd.DataFrame(lu_ed_ratio_mean, columns=wvl_cols); df_lu_ed.insert(0, 'Depth', ensemble_depth_grid)
                
    #             k_list, rrs_list, all_results_dfs, all_kd_par_dfs = [], [], [], []
    #             processed_casts = [] 

    #             for cast_id in ensemble_casts:
    #                 cast_layers = st.session_state.profile_specific_layers.get(cast_id, {})
    #                 if not cast_layers:
    #                     st.warning(f"Skipping '{cast_id}' in ensemble derived products: No analysis layers defined.")
    #                     continue
                    
    #                 derived_products_loop = calculate_derived_products(
    #                     st.session_state.station_data[cast_id], cast_id, cast_layers
    #                 )
                    
    #                 if derived_products_loop:
    #                     rrs_list.append(derived_products_loop['rrs_df_export'].set_index('wavelength_nm'))
    #                     k_list.append(derived_products_loop['k_df_export'].set_index('wavelength_nm'))
    #                     all_results_dfs.append(derived_products_loop['results_df'])
    #                     all_kd_par_dfs.append(derived_products_loop['kd_par_df'])
    #                     processed_casts.append(cast_id)
                        
    #             ensemble_results_df = pd.concat(all_results_dfs).groupby('Layer').mean().reset_index()
    #             ensemble_kd_par_df = pd.concat(all_kd_par_dfs).groupby('Layer').mean().reset_index()
    #             df_k_ens = pd.concat(k_list).groupby(level=0).agg(['mean', 'std']); df_k_ens.columns = ['_'.join(col) for col in df_k_ens.columns]; df_k_ens.reset_index(inplace=True)
    #             df_rrs_ens = pd.concat(rrs_list).groupby(level=0).agg(['mean', 'std']); df_rrs_ens.columns = ['_'.join(col) for col in df_rrs_ens.columns]; df_rrs_ens.reset_index(inplace=True)

    #             zip_file.writestr(f"{base_name}/L3_Data_Averaged/Ed_averaged.csv", df_ed.to_csv(index=False))
    #             zip_file.writestr(f"{base_name}/L3_Data_Averaged/Lu_averaged.csv", df_lu.to_csv(index=False))
    #             zip_file.writestr(f"{base_name}/L3_Data_Averaged/in_water_Lu_Ed_ratio_averaged.csv", df_lu_ed.to_csv(index=False))
    #             zip_file.writestr(f"{base_name}/L3_Data_Averaged/K_metrics_averaged.csv", df_k_ens.to_csv(index=False))
    #             zip_file.writestr(f"{base_name}/L3_Data_Averaged/Rrs_propagated_averaged.csv", df_rrs_ens.to_csv(index=False))
                
    #             ensemble_mixture_results = {}
    #             if st.session_state.get('wm_vertices'):
    #                 for cast_id in processed_casts:
    #                     ens_cast_data = st.session_state.station_data[cast_id]
    #                     ens_phys_props = calculate_physical_properties(ens_cast_data, st.session_state.lon, st.session_state.lat)
    #                     ens_mix_text, ens_mix_df = analyze_mixture(ens_phys_props['Salinity'], ens_phys_props['Temperature'], st.session_state.wm_vertices, st.session_state.wm_names)
    #                     ensemble_mixture_results[cast_id] = ens_mix_text
    #                     ens_mix_profile_df = pd.concat([ens_phys_props['Depth'], ens_mix_df], axis=1).dropna()
    #                     zip_file.writestr(f"{base_name}/L3_Data_Individual_Mixture_Profiles/{cast_id}_mixture_data.csv", ens_mix_profile_df.to_csv(index=False))

    #             report_text_ens = generate_summary_report(
    #                 processed_casts, 
    #                 st.session_state.profile_specific_layers, 
    #                 ensemble_results_df, 
    #                 ensemble_kd_par_df, 
    #                 ensemble_mixture_results,
    #                 empirical_model=st.session_state.get('empirical_model_params'),
    #                 external_probe_filename=st.session_state.get('external_probe_filename'),
    #                 comparison_kd_df=st.session_state.get('comparison_kd_df')
    #             )
    #             zip_file.writestr(f"{base_name}/summary_report.txt", report_text_ens)
            
    #         st.download_button(label="Download Ensemble Package", data=zip_buffer_ens.getvalue(), file_name=f"{base_name}.zip", mime="application/zip", key="download_ensemble")


    
    # ------------------ TAB 8: SPECTRAL CONVOLUTION ------------------
    with tab_convolution:
        st.header("Spectral Convolution for Satellite Matching")
        
        with st.expander("Load Historical Master Rrs Data"):
            st.file_uploader("Upload Master Rrs Data (CSV)", type="csv", key="master_rrs_loader", on_change=_handle_master_rrs_upload)
        st.divider()
    
        srf_data = load_srf_from_folder(
            folder_signature=get_srf_folder_signature("srf_data")
        )
        derived_products = st.session_state.get('derived_products')
    
        if derived_products is None or 'rrs_df_export' not in derived_products:
            st.warning("Analyze at least one layer in 'Core Optical Analysis' to generate Rrs data.")
        elif not srf_data:
            st.info(
                "No SRFs found. Add CSV SRFs or the official PACE OCI `.nc` RSR file to `srf_data` "
                "and rerun the app. OCI files retain their native response functions."
            )
        else:
            c1, c2, c3 = st.columns(3)
            selected_sensor = c1.selectbox("Select Sensor:", list(srf_data.keys()))
            df_srf = srf_data[selected_sensor]
            srf_wl_col = next((col for col in df_srf.columns if 'wave' in col.lower()), None)
            
            rrs_options = [col for col in derived_products['rrs_df_export'].columns if 'rrs' in col]
            selected_rrs_source = c2.selectbox("Select Rrs source:", rrs_options)
            
            srf_bands = [col for col in df_srf.columns if col != srf_wl_col]
            is_oci_srf = 'pace oci' in selected_sensor.lower() or selected_sensor.lower().startswith('oci')
            if is_oci_srf and srf_wl_col is not None:
                srf_wavelengths = pd.to_numeric(df_srf[srf_wl_col], errors='coerce').to_numpy(dtype=float)
                available_wavelengths = np.asarray(station_data['wavelengths'], dtype=float)
                profile_min, profile_max = np.nanmin(available_wavelengths), np.nanmax(available_wavelengths)
                oci_bands_in_profile_range = []
                for band in srf_bands:
                    response = pd.to_numeric(df_srf[band], errors='coerce').to_numpy(dtype=float)
                    valid_response = np.isfinite(srf_wavelengths) & np.isfinite(response) & (response > 0)
                    if np.sum(valid_response) < 2:
                        continue
                    centre = np.average(
                        srf_wavelengths[valid_response], weights=response[valid_response]
                    )
                    if (
                        profile_min <= centre <= profile_max
                        and SCIENTIFIC_PRODUCT_WAVELENGTH_RANGE[0] <= centre <= SCIENTIFIC_PRODUCT_WAVELENGTH_RANGE[1]
                        and srf_domain_coverage_percent(srf_wavelengths, response)
                        >= MIN_SRF_DOMAIN_COVERAGE_PERCENT
                    ):
                        oci_bands_in_profile_range.append(band)
                default_bands = oci_bands_in_profile_range
                st.caption(
                    "PACE OCI: bands with at least "
                    f"{MIN_SRF_DOMAIN_COVERAGE_PERCENT:.1f}% of their SRF area inside the 400–700 nm "
                    "propagated-product domain are selected by default. Keep or change this selection "
                    "explicitly for a reproducible run."
                )
            else:
                default_bands = [
                    band for band in srf_bands
                    if srf_domain_coverage_percent(
                        df_srf[srf_wl_col].to_numpy(dtype=float),
                        pd.to_numeric(df_srf[band], errors='coerce').to_numpy(dtype=float),
                    ) >= MIN_SRF_DOMAIN_COVERAGE_PERCENT
                ][:16]
            selected_bands = c3.multiselect("Select bands:", srf_bands, default=default_bands)
            convolution_irradiance_mode = st.selectbox(
                "Surface irradiance for convolution:",
                options=list(CONVOLUTION_IRRADIANCE_MODE_LABELS),
                format_func=lambda mode: CONVOLUTION_IRRADIANCE_MODE_LABELS[mode],
                key=f"convolution_irradiance_mode_{station_id}",
                help=(
                    "Matching the Rrs source keeps its stored Rrs. Other choices recalculate "
                    "Rrs for the same propagated Lw and the selected irradiance, then record "
                    "the comparison for audit and teaching."
                ),
            )
            if convolution_irradiance_mode == 'match_rrs':
                st.caption("Strict mode: the convolution uses the same irradiance provenance as the selected Rrs.")
            else:
                st.caption(
                    "Comparison mode: Rrs is recalculated from the same propagated Lw with the "
                    "chosen irradiance before convolution. Small changes are expected when Es and Ed(0+) "
                    "have similar spectral shapes."
                )
    
            if st.button("Perform Convolution", use_container_width=True):
                if not selected_bands:
                    st.warning("Select at least one band before performing convolution.")
                    st.stop()
                with st.spinner("Processing..."):
                    selected_layer = None
                    if selected_rrs_source.startswith('propagated_rrs_L'):
                        try:
                            selected_layer = int(
                                selected_rrs_source.split('propagated_rrs_L', 1)[1].split('_', 1)[0]
                            )
                        except (IndexError, ValueError):
                            st.error("Could not identify the layer used to calculate the selected propagated Rrs.")

                    layer_definition = (
                        current_station_layers.get(selected_layer)
                        if selected_layer is not None
                        else None
                    )
                    if selected_rrs_source.startswith('propagated_rrs_L') and layer_definition is None:
                        surface_irradiance = np.full(len(station_data['wavelengths']), np.nan)
                        irradiance_source = "Selected propagated layer is unavailable"
                        rrs_irradiance_source = "Unavailable"
                        resolved_convolution_mode = 'auto'
                    else:
                        rrs_audit = derived_products.get('rrs_irradiance_audit', pd.DataFrame())
                        matching_audit = rrs_audit.loc[
                            rrs_audit['Rrs source'] == selected_rrs_source
                        ] if not rrs_audit.empty else pd.DataFrame()
                        rrs_irradiance_mode = (
                            matching_audit['Irradiance mode'].iloc[0]
                            if not matching_audit.empty else 'auto'
                        )
                        rrs_irradiance_source = (
                            matching_audit['Resolved irradiance'].iloc[0]
                            if not matching_audit.empty else 'Not recorded'
                        )
                        resolved_convolution_mode = (
                            rrs_irradiance_mode
                            if convolution_irradiance_mode == 'match_rrs'
                            else convolution_irradiance_mode
                        )
                        convolution_ed0_mode = st.session_state.get(
                            f"ed0_propagation_mode_{station_id}", 'raw_fit'
                        )
                        convolution_kd_constraint = (
                            derived_products.get('layer_kw_constrained_kd', {}).get(selected_layer)
                            if selected_layer is not None and convolution_ed0_mode == 'kw_constrained'
                            else None
                        )
                        surface_irradiance, irradiance_source = resolve_surface_irradiance(
                            station_data,
                            resolved_convolution_mode,
                            z_top=layer_definition['range'][0] if layer_definition else None,
                            z_bottom=layer_definition['range'][1] if layer_definition else None,
                            kd_constraint=convolution_kd_constraint,
                            ed0_propagation_mode=convolution_ed0_mode,
                        )

                    if np.any(np.isfinite(surface_irradiance) & (surface_irradiance > 0)):
                        if convolution_irradiance_mode == 'match_rrs':
                            # Strict reconstruction: retain the selected Rrs and its recorded denominator.
                            rrs_hyper = derived_products['rrs_df_export'][selected_rrs_source].to_numpy(dtype=float)
                            rrs_treatment = 'Selected Rrs retained (strict provenance match)'
                        elif selected_layer is not None:
                            # Sensitivity comparison: Kd/Klu and the layer are fixed. Recalculate Rrs
                            # with the chosen denominator so the propagated Lw is physically unchanged.
                            selected_k_values = derived_products.get('layer_k_data', {}).get(selected_layer)
                            if selected_k_values is None:
                                rrs_hyper = np.full(len(station_data['wavelengths']), np.nan)
                                rrs_treatment = 'Unavailable: selected layer has no Kd/Klu'
                            else:
                                selected_kd, selected_klu = selected_k_values
                                rrs_hyper = calculate_Rrs(
                                    station_data['Ed_data'], station_data['Lu_data'], station_data['depth_m'],
                                    kd=selected_kd,
                                    klu=selected_klu,
                                    z_top=layer_definition['range'][0],
                                    z_bottom=layer_definition['range'][1],
                                    surface_irradiance=surface_irradiance,
                                )
                                rrs_treatment = (
                                    'Rrs recalculated for the selected irradiance; propagated Lw fixed'
                                )
                        else:
                            # Initial Rrs has no user-defined layer; it can still be compared for
                            # sources that do not require selected-layer Ed(0+).
                            rrs_hyper = calculate_Rrs(
                                station_data['Ed_data'], station_data['Lu_data'], station_data['depth_m'],
                                surface_irradiance=surface_irradiance,
                            )
                            rrs_treatment = 'Initial Rrs recalculated for the selected irradiance'

                        # Pandas may expose a read-only view. Mask an owned copy so
                        # convolution never modifies the stored hyperspectral Rrs.
                        rrs_hyper = np.array(rrs_hyper, dtype=float, copy=True)
                        rrs_hyper[~scientific_product_mask(station_data['wavelengths'])] = np.nan

                        if not np.any(np.isfinite(rrs_hyper)):
                            st.error(
                                "The selected Rrs and irradiance could not produce a valid hyperspectral spectrum."
                            )
                            st.stop()

                        # Reconstructed Lw is the same physical numerator for a same-layer
                        # sensitivity comparison; only the selected denominator changes Rrs.
                        lw_hyper = rrs_hyper * surface_irradiance
                        prof_wl = station_data['wavelengths']
                        srf_wl = df_srf[srf_wl_col].values
                        
                        results, band_centers = [], []
                        for band in selected_bands:
                            srf_response = df_srf[band].values
                            conv_val = perform_convolution(
                                prof_wl, lw_hyper, surface_irradiance, srf_wl, srf_response
                            )
                            (
                                band_centre,
                                srf_coverage,
                                radiometry_coverage,
                                qc_flag,
                            ) = assess_convolution_band_qc(
                                prof_wl, lw_hyper, surface_irradiance, srf_wl, srf_response,
                            )
                            results.append({
                                'Band': band,
                                'Convolved_Rrs_sr-1': conv_val,
                                'SRF coverage (%)': srf_coverage,
                                'Radiometric coverage (%)': radiometry_coverage,
                                'QC flag': qc_flag,
                            })
                            band_centers.append(band_centre)
                        
                        convolved_results = pd.DataFrame(results)
                        st.session_state.convolved_rrs_df = convolved_results
                        st.session_state.convolved_sensor_name = selected_sensor
                        st.session_state.convolved_rrs_source = selected_rrs_source
                        st.session_state.convolved_rrs_hyperspectral = rrs_hyper
                        st.session_state.convolution_irradiance_source = irradiance_source
                        st.session_state.convolution_audit = {
                            'Rrs source': selected_rrs_source,
                            'Rrs irradiance': rrs_irradiance_source,
                            'Rrs treatment in convolution': rrs_treatment,
                            'Convolution selection': CONVOLUTION_IRRADIANCE_MODE_LABELS[
                                convolution_irradiance_mode
                            ],
                            'Convolution irradiance': irradiance_source,
                            'Integration domain': '400–700 nm; at least 99.5% of full native SRF area required',
                            'Layer Ed(0+) treatment': (
                                ED0_PROPAGATION_MODE_LABELS[convolution_ed0_mode]
                                if selected_layer is not None else 'Not applicable'
                            ),
                        }
                        convolution_history = list(st.session_state.get('convolution_history') or [])
                        convolution_history.append({
                            'run_id': len(convolution_history) + 1,
                            'sensor': selected_sensor,
                            'rrs_source': selected_rrs_source,
                            'rrs_irradiance': rrs_irradiance_source,
                            'rrs_treatment': rrs_treatment,
                            'convolution_selection': CONVOLUTION_IRRADIANCE_MODE_LABELS[
                                convolution_irradiance_mode
                            ],
                            'convolution_irradiance': irradiance_source,
                            'ed0_treatment': (
                                ED0_PROPAGATION_MODE_LABELS[convolution_ed0_mode]
                                if selected_layer is not None else 'Not applicable'
                            ),
                            'band_centers_nm': np.asarray(band_centers, dtype=float),
                            'results': convolved_results.copy(),
                        })
                        st.session_state.convolution_history = convolution_history
                        st.rerun()
                    else:
                        st.error(
                            "Convolution needs a valid Es or Ed profile to reconstruct Lw from the selected Rrs."
                        )

            # RESULTS 
            if 'convolved_rrs_df' in st.session_state and st.session_state.convolved_rrs_df is not None:
                st.markdown("---")
                df_res = st.session_state.convolved_rrs_df
                if not np.any(np.isfinite(df_res['Convolved_Rrs_sr-1'])):
                    st.warning("No selected band produced a valid convolved Rrs. Review the QC flags below; only the hyperspectral curve can be displayed.")
                res_c1, res_c2 = st.columns([1, 2])
                
                with res_c1:
                    st.write(f"**Sensor:** {st.session_state.convolved_sensor_name}")
                    convolution_audit = st.session_state.get('convolution_audit')
                    if convolution_audit:
                        st.dataframe(
                            pd.DataFrame([convolution_audit]),
                            use_container_width=True,
                            hide_index=True,
                        )
                    st.dataframe(
                        df_res.style.format({
                            'Convolved_Rrs_sr-1': '{:.6f}',
                            'SRF coverage (%)': '{:.1f}',
                            'Radiometric coverage (%)': '{:.1f}',
                        }),
                        use_container_width=True,
                    )
                    flagged_bands = df_res[df_res['QC flag'] != 'Within displayed QC range']
                    if not flagged_bands.empty:
                        st.warning(
                            "Some bands require QC review or fall outside the 400–700 nm product domain. "
                            "Excluded bands are shown as blank values and are not saved as multispectral Rrs."
                        )
                    if st.button("Save to Multispectral Master"):
                        valid_convolved = df_res.loc[
                            np.isfinite(df_res['Convolved_Rrs_sr-1']),
                            ['Band', 'Convolved_Rrs_sr-1'],
                        ]
                        if valid_convolved.empty:
                            st.error("No selected band has valid 400–700 nm support to save.")
                        else:
                            new_row = valid_convolved.set_index('Band').T
                            new_row.insert(0, 'Station_ID', st.session_state.selected_station)
                            new_row.insert(1, 'Sensor', st.session_state.convolved_sensor_name)
                            st.session_state.master_multi_rrs_df = pd.concat([st.session_state.master_multi_rrs_df, new_row], ignore_index=True)
                            st.toast("Saved valid in-domain bands.")

                with res_c2:
                    fig_conv, ax = plt.subplots(figsize=(6,4))
                    # Plot Hyper
                    convolved_rrs_source = st.session_state.get(
                        'convolved_rrs_source', selected_rrs_source
                    )
                    rrs_for_plot = st.session_state.get('convolved_rrs_hyperspectral')
                    if rrs_for_plot is None:
                        rrs_for_plot = derived_products['rrs_df_export'][convolved_rrs_source]
                    ax.plot(station_data['wavelengths'], rrs_for_plot, color='white', alpha=0.5, label='Hyperspectral')
                    # Plot convolved points (Step plot style approximation)
                    current_history = st.session_state.get('convolution_history') or []
                    if current_history:
                        centers = current_history[-1]['band_centers_nm']
                    else:
                        df_s_active = srf_data[st.session_state.convolved_sensor_name]
                        centers = [
                            np.sum(df_s_active[srf_wl_col] * df_s_active[band]) / np.sum(df_s_active[band])
                            for band in df_res['Band']
                        ]
                    ax.plot(centers, df_res['Convolved_Rrs_sr-1'], color='red', marker='o', alpha=0.37, lw=2, label='Convolved (Bands)')
                    ax.set_ylim(bottom=0)
                    ax.set_xlim(*SCIENTIFIC_PRODUCT_WAVELENGTH_RANGE)
                    ax.legend(); ax.grid(True, ls=':')
                    st.pyplot(fig_conv, use_container_width=True)
                    plt.close(fig_conv)

            convolution_history = st.session_state.get('convolution_history') or []
            if convolution_history:
                st.markdown("---")
                history_title, clear_history_col = st.columns([5, 1])
                history_title.subheader("Convolution comparison by irradiance choice")
                if clear_history_col.button("Clear runs", key="clear_convolution_history"):
                    st.session_state.convolution_history = []
                    st.rerun()

                run_labels = {
                    run['run_id']: (
                        f"Run {run['run_id']} · {run['sensor']} · "
                        f"{run['convolution_irradiance']}"
                    )
                    for run in convolution_history
                }
                selected_run_ids = st.multiselect(
                    "Convolutions to display:",
                    options=list(run_labels),
                    default=list(run_labels),
                    format_func=lambda run_id: run_labels[run_id],
                    help="Each line preserves one complete convolution run from this session.",
                )
                displayed_runs = [
                    run for run in convolution_history if run['run_id'] in selected_run_ids
                ]

                if displayed_runs:
                    fig_history, ax_history = plt.subplots(figsize=(10, 5.2))
                    colors = sns.color_palette("husl", len(displayed_runs))
                    markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>']
                    for index, (run, color) in enumerate(zip(displayed_runs, colors)):
                        run_results = run['results']
                        ax_history.plot(
                            run['band_centers_nm'],
                            run_results['Convolved_Rrs_sr-1'].to_numpy(dtype=float),
                            color=color,
                            marker=markers[index % len(markers)],
                            lw=2.0,
                            markersize=6,
                            label=f"Run {run['run_id']}: {run['convolution_irradiance']}",
                        )
                    ax_history.set_title("Convolved Rrs sensitivity to irradiance source", weight='bold')
                    ax_history.set_xlabel("SRF-weighted band centre (nm)", weight='bold')
                    ax_history.set_ylabel(r"Convolved Rrs (sr$^{-1}$)", weight='bold')
                    ax_history.grid(True, linestyle=':', alpha=0.75)
                    place_legend_below_data(fig_history, ax_history, max_columns=3, fontsize=8)
                    st.pyplot(fig_history)
                    plt.close(fig_history)

                    if len(displayed_runs) > 1:
                        reference_run = displayed_runs[0]
                        reference_values = reference_run['results'][
                            ['Band', 'Convolved_Rrs_sr-1']
                        ].rename(columns={'Convolved_Rrs_sr-1': 'Reference Rrs'})
                        sensitivity_summary_rows = []
                        for comparison_run in displayed_runs[1:]:
                            if comparison_run['sensor'] != reference_run['sensor']:
                                continue
                            comparison_values = comparison_run['results'][
                                ['Band', 'Convolved_Rrs_sr-1']
                            ].rename(columns={'Convolved_Rrs_sr-1': 'Comparison Rrs'})
                            paired_values = reference_values.merge(
                                comparison_values, on='Band', how='inner'
                            )
                            if paired_values.empty:
                                continue
                            with np.errstate(divide='ignore', invalid='ignore'):
                                relative_difference = 100 * (
                                    paired_values['Comparison Rrs'] - paired_values['Reference Rrs']
                                ) / paired_values['Reference Rrs']
                            valid_difference = relative_difference[np.isfinite(relative_difference)]
                            if valid_difference.empty:
                                continue
                            sensitivity_summary_rows.append({
                                'Reference run': reference_run['run_id'],
                                'Comparison run': comparison_run['run_id'],
                                'Comparison irradiance': comparison_run['convolution_irradiance'],
                                'Median |Δ| (%)': np.median(np.abs(valid_difference)),
                                'Max |Δ| (%)': np.max(np.abs(valid_difference)),
                            })
                        if sensitivity_summary_rows:
                            st.caption(
                                f"Band-by-band difference relative to Run {reference_run['run_id']}. "
                                "This makes small, visually overlapping differences explicit."
                            )
                            st.dataframe(
                                pd.DataFrame(sensitivity_summary_rows).style.format({
                                    'Median |Δ| (%)': '{:.6f}',
                                    'Max |Δ| (%)': '{:.6f}',
                                }),
                                use_container_width=True,
                                hide_index=True,
                            )

                    with st.expander("Comparison-run audit table", expanded=False):
                        st.dataframe(
                            convolution_history_to_dataframe(displayed_runs).style.format({
                                'Band centre (nm)': '{:.1f}',
                                'Convolved_Rrs_sr-1': '{:.6f}',
                            }),
                            use_container_width=True,
                            hide_index=True,
                        )
                
                
                
                
                
    
    # ------------------ TAB 9: MASTER DATA & REPORTS ------------------
    with tab_report:
        st.header("Campaign Management & L3 Data Products")
        
        with st.expander("Step 0: Load Historical Master Tables", expanded=False):
            st.info("Suba seus arquivos Master (.csv) para mesclar dados de diferentes dias ou campanhas.")
            
            c_l1, c_l2 = st.columns(2)
            with c_l1:
                st.file_uploader("Load Vertical Radiometry", type="csv", key="up_rad_m", on_change=_handle_rad_upload)
                st.file_uploader("Load Summary Hyper Rrs", type="csv", key="up_hyper_m", on_change=_handle_hyper_upload)
                st.file_uploader("Load Summary Secchi & Kd", type="csv", key="up_secchi_central", on_change=_handle_master_secchi_upload)
            
            with c_l2:
                st.file_uploader("Load Vertical Ancillary", type="csv", key="up_anc_m", on_change=_handle_anc_upload)
                st.file_uploader("Load Summary Convolved Rrs", type="csv", key="up_multi_m", on_change=_handle_multi_upload)
                st.file_uploader("Load Master Bio-Optical Model", type="csv", key="up_model_central", on_change=_handle_model_upload)

        st.divider()

        st.subheader("1. Global Campaign Database Status")
        
        # Summary metrics of the cumulative database

        n_rad = st.session_state.master_vertical_radiometry['Station_ID'].nunique() if not st.session_state.master_vertical_radiometry.empty else 0
        n_sec = st.session_state.master_kd_secchi_df['Station_ID'].nunique() if not st.session_state.master_kd_secchi_df.empty else 0
        n_hyp = st.session_state.master_hyper_rrs_df['Station_ID'].nunique() if not st.session_state.master_hyper_rrs_df.empty else 0
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Stations (Radiometry)", n_rad)
        m_col2.metric("Stations (Secchi/Kd)", n_sec)
        m_col3.metric("Stations (Rrs Spectra)", n_hyp)

        col_agg, col_dl = st.columns(2)
        
        if col_agg.button("Snapshot Current Station to Masters", use_container_width=True, type="primary"):
            
            # Triggers the aggregation of all current results into the master tables
            if derived_products: 
                aggregate_to_vertical_masters(st.session_state.selected_station, station_data, derived_products)
                st.success(f"'{st.session_state.selected_station}' Snapshot saved in memory")
                st.rerun() 
            else:
                st.error("Analyze layers in 'Core Optical Analysis' first.")
        
        if col_dl.button("Export Full Campaign ZIP", use_container_width=True):
            
            # Packages all master tables into one ZIP for archival
            if st.session_state.master_vertical_radiometry.empty:
                st.error("Database is empty.")
            else:
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("1_Master_Vertical_Radiometry.csv", st.session_state.master_vertical_radiometry.to_csv(index=False))
                    zf.writestr("2_Master_Vertical_Ancillary.csv", st.session_state.master_vertical_ancillary.to_csv(index=False))
                    zf.writestr("3_Summary_Hyper_Rrs.csv", st.session_state.master_hyper_rrs_df.to_csv(index=False))
                    zf.writestr("4_Summary_Convolved_Rrs.csv", st.session_state.master_multi_rrs_df.to_csv(index=False))
                    zf.writestr("5_Summary_Secchi_Kd.csv", st.session_state.master_kd_secchi_df.to_csv(index=False))
                    if not st.session_state.empirical_model_data.empty:
                        zf.writestr("6_Master_BioOptical_Model.csv", st.session_state.empirical_model_data.to_csv(index=False))
                
                st.download_button("Download ZIP Package", buf.getvalue(), "Campaign_Full_Database.zip", use_container_width=True)

        st.divider()

        # L3 STATION PACKAGE ---
        st.subheader(f"2. Individual L3 Package: {st.session_state.selected_station}")
        st.info("Build Detailed L3 Package (ZIP)")
        
        # Generates a comprehensive per-station folder with all physics, optics, and plots
        if st.button("Build L3 Station Package", use_container_width=True, key="btn_l3_pro"):
            station_id = st.session_state.selected_station
            base = f"C_{station_id}_L3"
            l3_buf = io.BytesIO()
            
            s_data = st.session_state.station_data[station_id]
            w_wavelengths = s_data['wavelengths']
            wvl_cols = [f"{int(w)}nm" for w in w_wavelengths]
            
            with zipfile.ZipFile(l3_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                
                # Export Physics
                # Profiler
                df_phys_prof = calculate_physical_properties(s_data, st.session_state.lon, st.session_state.lat)
                if st.session_state.mixture_df_profiler is not None:
                    df_phys_prof = pd.concat([df_phys_prof.reset_index(drop=True), st.session_state.mixture_df_profiler.reset_index(drop=True)], axis=1)
                zf.writestr(f"{base}/1_Profiler_Profiler/PHYS_Data_Profiler.csv", df_phys_prof.to_csv(index=False))

                df_opt_prof = pd.DataFrame({'Depth_m': s_data['depth_m']})
                with np.errstate(divide='ignore', invalid='ignore'):
                    lued = np.where(s_data['Ed_data'] > 0, s_data['Lu_data'] / s_data['Ed_data'], np.nan)
                for i, wl in enumerate(wvl_cols):
                    df_opt_prof[f'Ed_{wl}'] = s_data['Ed_data'][:, i]
                    df_opt_prof[f'Lu_{wl}'] = s_data['Lu_data'][:, i]
                    df_opt_prof[f'LuEd_{wl}'] = lued[:, i]
                zf.writestr(f"{base}/1_Profiler_Profiler/OPTICAL_Profiles_Profiler.csv", df_opt_prof.to_csv(index=False))

                # External Probe
                if st.session_state.external_probe_df is not None:
                    st_col = st.session_state.p1_st
                    match = find_best_match(station_id, st.session_state.external_probe_df[st_col].dropna().unique())
                    if match:
                        df_ext_st = st.session_state.external_probe_df[st.session_state.external_probe_df[st_col].astype(str) == str(match)].copy()
                        zf.writestr(f"{base}/2_External_Probe/PHYS_Data_External_Probe_Original.csv", df_ext_st.to_csv(index=False))

                # Export Raw and derived Optics
                if derived_products:
                    profiler_aop_summary = derived_products['results_df'].copy()
                    profiler_aop_summary = profiler_aop_summary.merge(
                        derived_products['kd_par_df'][['Layer', 'Kd(PAR)']],
                        on='Layer',
                        how='left',
                    ).rename(columns={
                        'Kd(490)': 'Kd(490)_Profiler',
                        'klu(490)': 'Klu(490)_Profiler',
                        'Kd(PAR)': 'Kd(PAR)_Profiler',
                    })
                    zf.writestr(
                        f"{base}/3_Derived_Optical_Products/AOP_Summary_Metrics_Profiler.csv",
                        profiler_aop_summary.to_csv(index=False),
                    )
                    zf.writestr(f"{base}/3_Derived_Optical_Products/Rrs_PROFILER_Propagated.csv", derived_products['rrs_df_export'].to_csv(index=False))
                    zf.writestr(
                        f"{base}/3_Derived_Optical_Products/Rrs_Irradiance_Provenance.csv",
                        derived_products['rrs_irradiance_audit'].to_csv(index=False),
                    )
                    zf.writestr(
                        f"{base}/3_Derived_Optical_Products/Ed0plus_Raw_and_Kw_Constrained.csv",
                        derived_products['ed0_df_export'].to_csv(index=False),
                    )
                    zf.writestr(
                        f"{base}/3_Derived_Optical_Products/Ed0plus_Propagation_Provenance.csv",
                        derived_products['ed0_propagation_audit'].to_csv(index=False),
                    )
                    kd_fit_audit = derived_products.get('kd_fit_audit', pd.DataFrame())
                    if not kd_fit_audit.empty:
                        zf.writestr(
                            f"{base}/3_Derived_Optical_Products/Kd_Spectral_Fit_Audit_Raw_and_Kw_Constrained.csv",
                            kd_fit_audit.to_csv(index=False),
                        )
                    # Includes profiler-versus-external PAR attenuation when available.
                    if st.session_state.get('comparison_kd_df') is not None:
                        zf.writestr(
                            f"{base}/3_Derived_Optical_Products/KdPAR_Profiler_vs_External_CTD_Probe.csv",
                            st.session_state.comparison_kd_df.to_csv(),
                        )
                        
                    if 'convolved_rrs_df' in st.session_state and st.session_state.convolved_rrs_df is not None:
                        zf.writestr(f"{base}/3_Derived_Optical_Products/Rrs_SATELLITE_{st.session_state.convolved_sensor_name}_Convolved.csv", st.session_state.convolved_rrs_df.to_csv(index=False))
                        if st.session_state.get('convolution_audit'):
                            zf.writestr(
                                f"{base}/3_Derived_Optical_Products/Rrs_SATELLITE_Convolution_Provenance.csv",
                                pd.DataFrame([st.session_state.convolution_audit]).to_csv(index=False),
                            )
                        convolution_history = st.session_state.get('convolution_history') or []
                        if convolution_history:
                            zf.writestr(
                                f"{base}/3_Derived_Optical_Products/Rrs_SATELLITE_Convolution_Comparison_Runs.csv",
                                convolution_history_to_dataframe(convolution_history).to_csv(index=False),
                            )

                # VALIDATION METRICS 
                if st.session_state.get('comparison_table') is not None:                    
                    zf.writestr(f"{base}/4_Validation_and_Plots/CTD_Comparison_Metrics_ALL_CASTS.csv", st.session_state.comparison_table.to_csv())

                # Save all current session plots as high-res PNGs
                plots_to_save = {
                    "1_Stratification": st.session_state.get('fig_n2'),
                    "2_Rrs_Spectra": st.session_state.get('fig_rrs'),
                    "3_Log_Profiles": st.session_state.get('fig_log'),
                    "4_K_Spectra": st.session_state.get('fig_k'),
                    "5_TS_Diagram": st.session_state.get('fig_ts'),
                    "6_Water_Mass_Profiler": st.session_state.get('fig_vp'),
                    "7_Water_Mass_External": st.session_state.get('fig_ve'),
                    "8_CTD_Validation": st.session_state.get('fig_comp')
                }
                for name, fig in plots_to_save.items():
                    if fig: zf.writestr(f"{base}/4_Validation_and_Plots/Plots_PNG/{name}.png", fig_to_buffer(fig).getvalue())

                # Generate the final Summary Report text
                report_txt = generate_summary_report(
                    station_id, current_station_layers, 
                    st.session_state.results_df, st.session_state.kd_par_df, 
                    wm_profiler=st.session_state.mixture_df_profiler, 
                    wm_external=st.session_state.mixture_df_external
                )
                zf.writestr(f"{base}/Summary_Report.txt", report_txt)
                
            st.download_button(f"Download L3 Package for: {station_id}", l3_buf.getvalue(), f"{base}.zip", mime="application/zip", use_container_width=True)
            
    
    # ------------------ TAB 10: FORMULAS & METHODS ------------------ Needs work
    with tab_formulas:
        st.header("Formulas & Methods"); st.subheader("1. Apparent Optical Properties (Kd & klu)")
        st.markdown(r"""The diffuse attenuation coefficients for downwelling irradiance ($K_d$) and upwelling radiance ($K_u$) describe how rapidly light is attenuated with depth. They are derived from the Beer-Lambert Law, which states:$$ I(z) = I(0) \cdot e^{-K \cdot z} $$Where $I(z)$ is the intensity at depth $z$, and $I(0)$ is the intensity at the surface. By taking the natural logarithm, this becomes a linear relationship:$$ \ln(I(z)) = \ln(I(0)) - K \cdot z $$This app calculates $K_d$ and $K_u$ for each wavelength ($\lambda$) within a user-selected depth layer by performing a linear regression (using 'np.polyfit') on the natural log of the radiometric data versus depth. The coefficient $K$ is the negative of the slope of this regression. **Reference:** Mobley, C. D. (1994). *Light and Water: Radiative Transfer in Natural Waters*. Academic Press.""")
        st.subheader("2. Remote Sensing Reflectance ($R_{rs}$)"); st.markdown(f"""Remote sensing reflectance is the ratio of water-leaving radiance ($L_w$) to the downwelling irradiance ($E_d$) just above the surface: $$ R_{{rs}}(\\lambda) = \\frac{{L_w(\\lambda)}}{{E_d(\\lambda, 0^+)}} $$        The water-leaving radiance ($L_w$) is estimated from the upwelling radiance just below the surface ($L_u(\\lambda, 0^-)$) by accounting for the transmission of light across the air-sea interface. This app uses a standard approximation where the transmission factor is **{LW_TRANSMISSION_FACTOR}**:        $$ L_w \\approx {LW_TRANSMISSION_FACTOR} \\cdot L_u(0^-) $$        This coefficient approximates the term $\\frac{{1- \\rho(\\theta, n)}}{{n^2}}$, where $\\rho$ is the Fresnel reflectance and $n$ is the refractive index of water.         **Propagated $R_{{rs}}$** is calculated by using the layer-specific $K_d$ and $K_u$ to propagate the radiometric values from the top of the selected layer ($z$) back to the surface: $$ E_d(0^+) = E_d(z) \\cdot e^{{K_d \\cdot z}} $$        $$ L_u(0^-) = L_u(z) \\cdot e^{{K_u \\cdot z}} $$        These surface-equivalent values are then used to calculate the final propagated $R_{{rs}}$.
        """)
        with st.expander("Irradiance source hierarchy and audit trail", expanded=False):
            st.markdown(
                "For the automatic workflow, the denominator follows: **filtered Es**, "
                "then **original Es**, then **Ed(0+) extrapolated from the selected layer**, "
                "and finally **Ed(0+) from the shallowest valid profiler depth**.\n\n"
                "The sidebar lets you select one of these sources explicitly for propagated Rrs. "
                "In spectral convolution, *Match the irradiance used to calculate the selected Rrs* "
                "is the strict reconstruction option; every other choice is a recorded sensitivity comparison."
            )
        st.subheader("3. Stratification (Brunt-Väisälä Frequency)"); st.markdown(r"""The stratification of the water column is represented by the Brunt-Väisälä frequency squared, $N^2$. A high $N^2$ value indicates strong stratification. This value is calculated using the Gibbs SeaWater (GSW) Oceanographic Toolbox, which implements the Thermodynamic Equation of Seawater 2010 (TEOS-10). **Reference:** McDougall, T. J., & Barker, P. M. (2011). *Getting started with TEOS-10 and the Gibbs Seawater (GSW) Oceanographic Toolbox*. SCOR/IAPSO Working Group 127, ISBN 978-0-646-55606-5.""")
        st.subheader("4. Photosynthetically Available Radiation (PAR & Kd(PAR))"); st.markdown(r"""Photosynthetically Available Radiation (PAR) is the flux of photons between 400 and 700 nm. It is calculated by converting the energy of the downwelling irradiance $E_d(\lambda)$ from energy units ($\mu W/cm^2/nm$) to quanta units ($\mu mol\ photons/m^2/s/nm$) and then integrating over the wavelength range. This app follows the protocol from the ProSoft manual:$$ PAR(z) = \int_{{400nm}}^{{700nm}} E_q(\lambda, z) d\lambda $$Where $E_q$ is the spectral irradiance in quanta units. The attenuation coefficient for PAR, $K_d(PAR)$, is then calculated for each user-defined layer by performing a log-linear regression on the PAR profile versus depth, identical to the method used for spectral $K_d(\lambda)$.""")
