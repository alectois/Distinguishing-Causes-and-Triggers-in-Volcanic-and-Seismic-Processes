# Distinguishing Causes and Triggers in Volcanic and Seismic Processes

This repository contains the code, notebooks, processed datasets, figures, and result tables accompanying the thesis *Distinguishing Causes and Triggers in Volcanic and Seismic Processes*.

## Thesis summary

Volcanic crises can reflect an interaction between persistent background conditions and short-lived perturbations, but conventional causal analyses do not distinguish these roles. This thesis adapts and extends the Cause–Trigger algorithm for heterogeneous volcanic and seismological time series and applies it to two case studies: the May 2008 Etna crisis around the Wenchuan earthquake and the December 2019 Whakaari eruption.

The workflow constructs hourly monitoring datasets, defines reference and causal-analysis intervals, identifies data-driven transitions in the target variable, and evaluates candidate cause–trigger pairs across maximum lag orders and partition settings. Heterogeneous Minimum Message Length (HMML) is used as the main parent-discovery method, with PCMCI and PCMCI+ as comparison backends; multiple-testing corrections and serial-correlation sensitivity checks assess the robustness of the results.

For Etna, the most stable result identifies past local seismicity as the cause and low-frequency teleseismic-band RMS as the trigger. The Whakaari results are more sensitive to the selected transition and model specification, with the most recurrent relationships involving hydrothermal RMS and past-smoothed spectral contrast. These classifications describe operational roles among the observed monitoring proxies and should not be interpreted as proof of a unique physical mechanism.

## Repository structure

- `notebooks/` contains the dataset-construction and Cause–Trigger analysis notebooks for Etna and Whakaari.
- `src/` contains the shared workflow, causal-discovery backends, data-processing code, and reporting utilities.
- `data/` contains downloaded source data, cached inputs, and the processed hourly datasets.
- `results/` contains exported model summaries and diagnostic tables.
- `figures/` contains the generated case-study figures.

## Data sources and licensing

The Etna analysis uses the [ETNAGAS soil CO₂ dataset](https://doi.org/10.13127/etna/ecsf2002_2010), the [Mt. Etna Seismic Catalog 2000–2010](https://doi.org/10.13127/etnasc/2000_2010), and waveform data from the [Italian National Seismic Network](https://doi.org/10.13127/sd/x0fxnh7qfy). The Whakaari analysis uses GeoNet waveform and GNSS data, including the [Seismic Digital Waveform Dataset](https://doi.org/10.21420/G19Y-9D40) and [Continuous GNSS Network Time Series Dataset](https://doi.org/10.21420/30F4-1A55). Both case studies also use [Open-Meteo](https://open-meteo.com/en/license) weather data. Dataset-specific citations and processing details are provided in the construction notebooks.

The cited INGV and Open-Meteo data are distributed under CC BY 4.0. GeoNet data are distributed under CC BY 3.0 New Zealand.

The MIT License applies only to original code in this repository. Third-party datasets and adapted code remain subject to their original licenses and attribution requirements; they are not relicensed under MIT.

## Attribution

The Cause–Trigger implementation used in this repository is adapted from the code and methodology accompanying:

Hlaváčková-Schindler, K., Wöß, R., Pecorino, V., & Schindler, P. (2025). *Cause or Trigger? From Philosophy to Causal Modeling*. Zenodo. https://doi.org/10.5281/zenodo.15109084

The original Zenodo record is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). This repository modifies and extends the original implementation for the thesis use case, including volcanic datasets, diagnostic outputs, and alternative causal-discovery backends.

## License

Original code in this repository is released under the MIT License.

Some files are adapted from the Cause–Trigger algorithm implementation published on Zenodo under CC BY 4.0. Those adapted portions retain attribution to the original authors. See the file headers and the Attribution section above.

