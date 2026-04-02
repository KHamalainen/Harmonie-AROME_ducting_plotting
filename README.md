# Harmonie-AROME_ducting_plotting

# Structure of
duct_analysis/
│
├── main.py
├── config.py
│
├── io/
│   ├── read_obs.py
│   ├── preprocess_obs.py
│   └── read_model_point.py
│
├── compute/
│   ├── match_heights.py
│   ├── statistics.py
│   └── gradients.py
│
├── plots/
│   ├── timeseries.py
│   ├── scatter.py
│   ├── performance_diagram.py
│   ├── correlations.py
│   └── vertical_profiles.py
│
└── output/
    ├── data/
    └── figures/
