# Compressible Neural Particle Method (cNPM)
## Overview
A PyTorch implementation of Neural Particle Methods for fluid dynamics simulations, including both incompressible (NPM) and compressible (cNPM) formulations for dam break scenarios.
The code supports:

- **NPM**: Standard incompressible Neural Particle Methods
- **cNPM**: Compressible Neural Particle Methods with density evolution
- **Dam break simulation**: Classical fluid dynamics benchmark problem



## Installation and Setup

### Prerequisites
- Python 3.7+
- PyTorch with CUDA support
- NumPy, SciPy, Matplotlib

### Setup Instructions
1. Clone the repository
2. Install required Python packages
3. Create the database directory: `mkdir -p ./database`
4. Ensure IRK coefficient files are present in the `IRK/` directory

## IRK Coefficient Files

The `IRK/` directory contains Implicit Runge-Kutta coefficient files:

- **Butcher_IRK4.txt**: 4th-order IRK coefficients (used when `--q 4`)
- **Butcher_IRK20.txt**: 20th-order IRK coefficients (used when `--q 20`)

These files contain the Butcher tableau coefficients for implicit Runge-Kutta time integration. Additional IRK coefficient files can be obtained from the folder 'Utilities' must be downloaded from https://github.com/maziarraissi/PINNs/



## Command Line Arguments
### Model Configuration
- `--model`: Model type (`NPM`, `cNPM`) - Default: `NPM`
- `--hidden_layers`: Network architecture (e.g., `50 50 50 50`) - Default: `[60, 60]`

### Domain and Particles
- `--L`: Domain length [m]
- `--H`: Domain height [m]
- `--dl`: Particle spacing [m]
- `--particle_distribution`: Particle arrangement (`grid`, `random`) - Default: `grid`
- `--refine_times`: Particle refinement iterations - Default: `10`

### Physical Properties
- `--rho_init`: Initial density [kg/m³] - Default: `997`
- `--mu`: Dynamic viscosity [Ns/m²] - Default: `0.001016`
- `--eta`: Compressibility parameter (cNPM only) - Default: `0.01`

### Time Integration
- `--q`: Runge-Kutta order (4 or 20) - Default: `4`
- `--dt`: Time increment [s] - Default: `0.01`
- `--t_max`: Number of time steps - Default: `20`
- `--t_start`: Start time [s] - Default: `0`

### Training Parameters
- `--epoch`: Training epochs - Default: `20000`
- `--learning_rate`: Adam learning rate - Default: `1e-4`
- `--early_stopping_flg`: Enable early stopping (0/1) - Default: `0`
- `--early_stopping_patience`: Early stopping patience - Default: `1000`
- `--lbfgs_flg`: Use L-BFGS optimizer (0/1) - Default: `0`
- `--tl_flg`: Enable transfer learning (0/1) - Default: `0`

### Output Configuration
- `--database_path`: Output directory path - Default: `./database`
- `--result_directory_name`: Result subdirectory name - Default: `result`
- `--debug`: Enable debug mode

## Usage Examples

### NPM (Incompressible) Example
```bash
python main.py \
    --model NPM \
    --dt 0.01 \
    --t_max 30 \
    --L 0.146 \
    --H 0.292 \
    --dl 0.0073 \
    --epoch 12000 \
    --hidden_layers 50 50 50 50 \
    --database_path ./database \
    --result_directory_name NPM_result
```

### cNPM (Compressible) Example
```bash
python main.py \
    --model cNPM \
    --dt 0.01 \
    --t_max 20 \
    --L 0.1 \
    --H 0.2 \
    --dl 0.005 \
    --epoch 12000 \
    --hidden_layers 50 50 50 50 \
    --eta 0.01 \
    --database_path ./database \
    --result_directory_name cNPM_result \
    --tl_flg 1
```

## Output Structure

Results are saved to `database_path/project_name/result_directory_name/`:
- `ic_bc/`: Initial conditions and boundary condition data
- `figure/`: Visualization plots (pressure, velocity, density fields)
- `weights/`: Trained neural network parameters
- `loss/`: Training loss histories

## Model Selection Guide
- **NPM**: Use for incompressible flow scenarios, faster computation, established validation
- **cNPM**: Use when density variations are important, compressible effects are significant, or pressure-density coupling is required


## Credits and Citations
This project extends the Physics-Informed Neural Networks (PINNs) framework by Maziar Raissi et al. to Neural Particle Methods (NPM) in PyTorch, focusing on compressible fluid dynamics.

We gratefully acknowledge the following resources:
- Original PINNs implementation and theory: https://github.com/maziarraissi/PINNs
- Original NPM implementation : https://gitlab.com/henningwessels/npm

These contributions were invaluable to this work, and we sincerely appreciate the authors’ kind support and guidance.
