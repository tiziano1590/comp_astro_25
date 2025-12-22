# Daneel

A practical example to detect and characterize exoplanets.

The full documentation is at https://tiziano1590.github.io/comp_astro_25/index.html

## Installation

### Prerequisites

- Python >= 3.10

### Install from source

```bash
git clone https://github.com/tiziano1590/comp_astro_25.git
cd comp_astro_25
pip install .
```

### Development installation

```bash
git clone https://github.com/tiziano1590/comp_astro_25.git
cd comp_astro_25
pip install -e .
```

## Usage

After installation, you can run daneel from the command line:

```bash
daneel -i <input_file> [options]
```

### Command-line options

- `-i, --input`: Input parameter file (required)
- `-d, --detect`: Initialize detection algorithms for exoplanets
- `-a, --atmosphere`: Atmospheric characterization from input transmission spectrum
- `-d rf`: Run the Random Forest exoplanet detection algorithm
- `-d cnn`: Run the CNN exoplanet detection algorithm

### Examples

```bash
# Run exoplanet detection
daneel -i parameters.yaml -d

# Run atmospheric characterization
daneel -i parameters.yaml -a

# Run both detection and atmospheric analysis
daneel -i parameters.yaml -d -a

```
```bash
# Plot the transit light curve
daneel -i parameters.yaml -t

```
This command generates the transit light curve from the parameters in YAML file.

```bash
# Run Random Forest exoplanet detection
daneel -i parameters.yaml -d rf

```
```bash
# Run CNN exoplanet detection
daneel -i parameters.yaml -d cnn

```

## Input File Format

The input file should be a YAML file containing the necessary parameters for the analysis.

For Random Forest detection, the YAML file should include a `detection` section with keys:
  algorithm: rf
  dataset_path: path/to/tess_data.csv
  n_bins: 1000
  use_scaler: true
  samples_per_class: 350
  random_forest:
    n_estimators: 500
    max_depth: null
    min_samples_leaf: 1
    max_features: sqrt
    bootstrap: true
    class_weight: null

For CNN, the YAML file should include a `detection` section with keys:
  algorithm: cnn
  dataset_path: path/to/tess_data.csv
  n_bins: 1000
  
  samples_per_class: 350
  alpha: 
  gamma:
  batch_size:
  threshold:
  kernel_size:
 


## License

This project is licensed under the MIT License.

## Author

Tiziano Zingales (tiziano.zingales@unipd.it)
