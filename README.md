# Distinguishing Causes and Triggers in Volcanic and Seismic Processes

This repository contains the code, notebooks, processed datasets, figures, and result tables accompanying the thesis *Distinguishing Causes and Triggers in Volcanic and Seismic Processes*.

## Repository structure

- `notebooks/` contains the dataset-construction and Cause–Trigger analysis notebooks for Etna and Whakaari.
- `src/` contains the shared workflow, causal-discovery backends, data-processing code, and reporting utilities.
- `data/` contains downloaded source data, cached inputs, and the processed hourly datasets.
- `results/` contains exported model summaries and diagnostic tables.
- `figures/` contains the generated case-study figures.

## Data sources and licensing

The Etna analysis uses the [ETNAGAS soil CO₂ dataset](https://doi.org/10.13127/etna/ecsf2002_2010), the [Mt. Etna Seismic Catalog 2000–2010](https://doi.org/10.13127/etnasc/2000_2010), and waveform data from the [Italian National Seismic Network](https://doi.org/10.13127/sd/x0fxnh7qfy). The Whakaari analysis uses GeoNet waveform and GNSS data, including the [Seismic Digital Waveform Dataset](https://doi.org/10.21420/G19Y-9D40) and [Continuous GNSS Network Time Series Dataset](https://doi.org/10.21420/30F4-1A55). Both case studies also use [Open-Meteo](https://open-meteo.com/en/license) weather data. Dataset-specific citations and processing details are provided in the construction notebooks.

Processed CSV and pickle files are derivatives of the cited source datasets and were created for this study; all transformations are documented in the dataset-construction notebooks.

The cited INGV and Open-Meteo data are distributed under CC BY 4.0. GeoNet data are distributed under CC BY 3.0 New Zealand. We acknowledge the New Zealand GeoNet programme and its sponsors NHC, Earth Sciences NZ, LINZ, NEMA and MBIE for providing data/images used in this study.

The MIT License applies only to original code in this repository. Third-party datasets and adapted code remain subject to their original licenses and attribution requirements; they are not relicensed under MIT.

## Attribution

The Cause–Trigger implementation used in this repository is adapted from the code and methodology accompanying:

Hlaváčková-Schindler, K., Wöß, R., Pecorino, V., & Schindler, P. (2025). *Cause or Trigger? From Philosophy to Causal Modeling*. Zenodo. https://doi.org/10.5281/zenodo.15109084

The original Zenodo record is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). This repository modifies and extends the original implementation for the thesis use case, including volcanic datasets, diagnostic outputs, and alternative causal-discovery backends.

## License

Original code in this repository is released under the MIT License.

Some files are adapted from the Cause–Trigger algorithm implementation published on Zenodo under CC BY 4.0. Those adapted portions retain attribution to the original authors. See the file headers and the Attribution section above.

