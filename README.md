# 🏎️ F1 Qualifying Prediction System
A comprehensive machine learning system for predicting Formula 1 qualifying times using historical data and advanced analytics.

## 📋 Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Data Sources](#data-sources)
- [Model Details](#model-details)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)

## ✨ Features

### 🤖 **Machine Learning Models**
- **Linear Regression**: Fast and interpretable baseline model
- **Random Forest**: Advanced ensemble method for better accuracy
- **Cross-validation**: Robust model evaluation with 5-fold CV
- **Performance metrics**: MAE, R², RMSE for comprehensive evaluation

### 📊 **Professional Visualizations**
- **Historical Data Analysis**: Dual-panel box plots and violin plots
- **Prediction Results**: Color-coded horizontal bar charts by team
- **Team Comparisons**: Performance ranking visualizations
- **Statistical Insights**: Automated statistics display

### 🔧 **Advanced Features**
- **Automatic caching**: Fast data retrieval with FastF1 integration
- **Error handling**: Robust exception management and logging
- **Flexible configuration**: Easy customization of seasons, tracks, and drivers
- **Performance factors**: Team and driver-specific multipliers for 2025

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Required Packages
```bash
pip install fastf1 pandas numpy matplotlib seaborn scikit-learn
```

### Optional (for development)
```bash
pip install jupyter notebook ipython
```

### Clone and Setup
```bash
# Clone the repository
git clone <your-repository-url>
cd f1-prediction-system

# Create cache directory (automatic in script)
mkdir cache

# Run the system
python f1_predictions.py
```

## 🚀 Quick Start

### Basic Usage
```bash
python f1_predictions.py
```

The system will:
1. **Fetch historical data** from 2024 F1 seasons
2. **Train machine learning models** on qualifying times
3. **Generate predictions** for the 2025 season
4. **Create visualizations** of results and analysis

### Expected Output
```
🏎️  Enhanced F1 Qualifying Prediction System
==================================================

📊 Fetching training data...
✅ Successfully loaded 100 data points

📈 Visualizing data...
[Historical qualifying times visualization displays]

🤖 Training prediction model...
Model trained successfully!
Validation MAE: 0.234 seconds
Validation R²: 0.892
Cross-validation MAE: 0.198 ± 0.045 seconds

🔮 Generating predictions for upcoming race...

🏁 Predicted Qualifying Results:
================================================================================
Pos Driver              Team                     Predicted Q3   
--------------------------------------------------------------------------------
1   Max Verstappen      Red Bull Racing          88.563s
2   Sergio Perez        Red Bull Racing          88.848s
3   Charles Leclerc     Ferrari                  88.871s
...
```

## 📁 Project Structure

```
f1-prediction-system/
│
├── f1_predictions.py          # Main application file
├── cache/                     # FastF1 data cache directory
├── README.md                  # This file
├── requirements.txt           # Python dependencies
└── docs/                      # Additional documentation
    ├── model_details.md
    ├── api_reference.md
    └── examples/
```

### Core Classes

#### `F1DataManager`
Handles all data fetching and preprocessing operations.
```python
data_manager = F1DataManager()
data = data_manager.fetch_session_data(2024, 1, 'Q')  # Bahrain 2024 Qualifying
```

#### `F1Predictor`
Machine learning model for qualifying predictions.
```python
predictor = F1Predictor(model_type='random_forest')
predictor.train(historical_data)
predictions = predictor.predict_custom_grid(current_grid)
```

#### `F1Visualizer`
Professional visualization creation.
```python
visualizer = F1Visualizer()
visualizer.plot_predictions(predictions, "Race Predictions")
```

## 💡 Usage Examples

### Custom Season Analysis
```python
# Analyze specific races
sessions_config = [
    {'year': 2024, 'round': 1, 'session': 'Q'},   # Bahrain
    {'year': 2024, 'round': 5, 'session': 'Q'},   # Miami
    {'year': 2024, 'round': 9, 'session': 'Q'},   # Canada
]

data = data_manager.fetch_multiple_sessions(sessions_config)
```

### Different ML Models
```python
# Linear Regression (faster)
linear_predictor = F1Predictor(model_type='linear')

# Random Forest (more accurate)
rf_predictor = F1Predictor(model_type='random_forest')
```

### Custom Driver Lineups
```python
custom_grid = {
    'Lewis Hamilton': 'Ferrari',        # Hypothetical transfer
    'Max Verstappen': 'Red Bull Racing',
    'Charles Leclerc': 'McLaren',       # Another hypothetical
    # ... add more drivers
}

predictions = predictor.predict_custom_grid(custom_grid)
```

### Track-Specific Predictions
```python
# Monaco characteristics
monaco_config = {
    'base_time': 71.0,  # Slower, technical track
    'overtaking_difficulty': 'high'
}

# Monza characteristics
monza_config = {
    'base_time': 80.5,  # High-speed track
    'overtaking_difficulty': 'low'
}

predictions = predictor.predict_custom_grid(grid, monaco_config)
```

## ⚙️ Configuration

### Performance Factors (2025 Season)

#### Team Performance Multipliers
```python
team_factors = {
    'Red Bull Racing': 0.996,    # Dominant performance
    'Ferrari': 0.998,           # Strong contender  
    'McLaren': 0.999,           # Much improved
    'Mercedes': 0.999,          # Consistent
    'Aston Martin': 1.001,      # Midfield leader
    # ... customize as needed
}
```

#### Driver Performance Multipliers
```python
driver_factors = {
    'Max Verstappen': 0.997,    # Exceptional qualifier
    'Charles Leclerc': 0.998,   # Elite speed
    'Lando Norris': 0.999,      # Strong performer
    # ... adjust based on current form
}
```

### Logging Configuration
```python
# Adjust logging level
logging.basicConfig(level=logging.DEBUG)  # More verbose
logging.basicConfig(level=logging.WARNING)  # Less verbose
```

## 📡 Data Sources

### FastF1 API
- **Official F1 timing data** from Formula 1
- **Real-time telemetry** during race weekends
- **Historical archives** back to 2018
- **Automatic caching** for improved performance

### Data Coverage
- ✅ **Qualifying sessions** (Q1, Q2, Q3)
- ✅ **Practice sessions** (FP1, FP2, FP3)
- ✅ **Race data** (lap times, positions)
- ✅ **Weather conditions**
- ✅ **Track information**

## 🧠 Model Details

### Feature Engineering
- **Q1 and Q2 times** as primary predictors
- **Team performance factors** based on recent form
- **Driver skill multipliers** from historical analysis
- **Track-specific adjustments** for different circuits

### Model Selection
| Model | Pros | Cons | Use Case |
|-------|------|------|----------|
| **Linear Regression** | Fast, interpretable, stable | Limited complexity | Quick analysis, baseline |
| **Random Forest** | Higher accuracy, handles non-linearity | Slower, less interpretable | Production predictions |

### Validation Strategy
- **Train/Test Split**: 80/20 random split
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Metrics**: MAE (primary), R², RMSE
- **Outlier Handling**: 3-sigma clipping for extreme values

## 📈 Visualizations

### 1. Historical Data Analysis
- **Dual-panel layout**: Box plots + Violin plots
- **Outlier filtering**: Clean visualization of distributions
- **Statistical overlay**: Mean and standard deviation display

### 2. Prediction Results
- **Horizontal bar chart**: Easy reading of driver rankings
- **Team color coding**: Visual team identification
- **Time annotations**: Precise timing information

### 3. Team Performance Comparison
- **Vertical bar chart**: Clear team hierarchy
- **Gradient coloring**: Performance-based color scheme
- **Value labels**: Exact timing comparisons

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd f1-prediction-system

# Create virtual environment
python -m venv f1_env
source f1_env/bin/activate  # Linux/Mac
f1_env\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements-dev.txt
```

### Code Style
- **PEP 8** compliance for Python code
- **Type hints** for function parameters
- **Docstrings** for all classes and methods
- **Error handling** with try/except blocks

### Adding New Features
1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-feature`
3. **Add tests** for new functionality
4. **Update documentation**
5. **Submit pull request**

## 🔧 Troubleshooting

### Common Issues

#### Cache Directory Error
```
NotADirectoryError: Cache directory does not exist!
```
**Solution**: The script automatically creates the cache directory, but ensure you have write permissions.

#### No Data Available
```
❌ No training data available
```
**Solutions**:
- Check internet connection
- Verify FastF1 API is accessible
- Try different race rounds or years
- Check if the season has started

#### Memory Issues
```
MemoryError: Unable to allocate array
```
**Solutions**:
- Reduce the number of sessions in `sessions_config`
- Use `model_type='linear'` instead of `'random_forest'`
- Close other applications to free RAM

#### Import Errors
```
ModuleNotFoundError: No module named 'fastf1'
```
**Solution**: Install missing packages:
```bash
pip install fastf1 pandas numpy matplotlib seaborn scikit-learn
```

### Performance Optimization

#### Faster Data Loading
```python
# Use fewer sessions for quicker testing
sessions_config = [
    {'year': 2024, 'round': 1, 'session': 'Q'},  # Just one race
]
```

#### Reduced Model Complexity
```python
# Use Linear Regression for faster training
predictor = F1Predictor(model_type='linear')
```

#### Cache Management
```python
# Clear cache if corrupted
import shutil
shutil.rmtree('cache')  # Will be recreated automatically
```

### Debug Mode
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Add custom debug prints
logger.debug(f"Data shape: {df.shape}")
logger.debug(f"Available columns: {df.columns.tolist()}")
```

## 📚 Additional Resources

- **FastF1 Documentation**: [https://docs.fastf1.dev/](https://docs.fastf1.dev/)
- **Scikit-learn User Guide**: [https://scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html)
- **F1 Technical Regulations**: [https://www.fia.com/regulations](https://www.fia.com/regulations)
- **Formula 1 Official Site**: [https://www.formula1.com/](https://www.formula1.com/)

## 📄 License
- This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- **FastF1 Team** for the excellent F1 data API
- **Formula 1** for making timing data available
- **Scikit-learn Contributors** for machine learning tools
- **Matplotlib/Seaborn Teams** for visualization capabilities
- **Made with ❤️ for Formula 1 analytics**

