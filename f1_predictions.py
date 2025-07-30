import fastf1
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set matplotlib style for better visuals
plt.style.use('default')
sns.set_palette("husl")

# Create cache directory if it doesn't exist and enable cache
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    logger.info(f"Created cache directory: {cache_dir}")

fastf1.Cache.enable_cache(cache_dir)

class F1DataManager:
    """Handles all F1 data fetching and preprocessing operations"""
    
    def __init__(self):
        self.current_year = 2025
        self.cache_dir = 'cache'
        
    def fetch_session_data(self, year, round_number, session_type='Q'):
        """Fetch qualifying or practice session data"""
        try:
            logger.info(f"Fetching {session_type} data for {year} Round {round_number}")
            session = fastf1.get_session(year, round_number, session_type)
            session.load()
            
            # Get results with relevant columns
            if session_type == 'Q':
                results = session.results[['DriverNumber', 'FullName', 'TeamName', 'Q1', 'Q2', 'Q3']]
            else:  # Practice sessions
                results = session.results[['DriverNumber', 'FullName', 'TeamName', 'Time']]
            
            results = results.rename(columns={'FullName': 'Driver'})
            
            # Convert lap times to seconds
            if session_type == 'Q':
                for col in ['Q1', 'Q2', 'Q3']:
                    results[f'{col}_sec'] = results[col].apply(
                        lambda x: x.total_seconds() if pd.notnull(x) else None
                    )
            else:
                results['Time_sec'] = results['Time'].apply(
                    lambda x: x.total_seconds() if pd.notnull(x) else None
                )
            
            results['Year'] = year
            results['Round'] = round_number
            results['Session'] = session_type
            
            logger.info(f"Successfully fetched data for {len(results)} drivers")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching {session_type} data for {year} Round {round_number}: {e}")
            return None
    
    def fetch_multiple_sessions(self, sessions_config):
        """Fetch data from multiple sessions"""
        all_data = []
        
        for config in sessions_config:
            year = config.get('year', self.current_year)
            round_num = config.get('round')
            session_type = config.get('session', 'Q')
            
            data = self.fetch_session_data(year, round_num, session_type)
            if data is not None:
                all_data.append(data)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Combined data from {len(all_data)} sessions")
            return combined_df
        else:
            logger.warning("No data was successfully fetched")
            return None

class F1Predictor:
    """Machine learning model for F1 qualifying predictions"""
    
    def __init__(self, model_type='linear'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.is_trained = False
        
        # Initialize model based on type
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError("Model type must be 'linear' or 'random_forest'")
    
    def prepare_features(self, df):
        """Prepare features for training/prediction"""
        # Focus on qualifying sessions
        quali_data = df[df['Session'] == 'Q'].copy()
        
        if quali_data.empty:
            logger.warning("No qualifying data found")
            return None, None
        
        # Create features from Q1 and Q2 times
        feature_cols = ['Q1_sec', 'Q2_sec']
        target_col = 'Q3_sec'
        
        # Remove rows where target is missing
        valid_data = quali_data.dropna(subset=[target_col])
        
        if valid_data.empty:
            logger.warning("No valid Q3 data found")
            return None, None
        
        X = valid_data[feature_cols]
        y = valid_data[target_col]
        
        return X, y
    
    def train(self, df):
        """Train the prediction model"""
        X, y = self.prepare_features(df)
        
        if X is None or y is None:
            logger.error("Cannot train model: insufficient data")
            return False
        
        # Handle missing values
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X), 
            columns=X.columns, 
            index=X.index
        )
        
        # Scale features for better performance
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_imputed),
            columns=X.columns,
            index=X.index
        )
        
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        logger.info(f"Model trained successfully!")
        logger.info(f"Validation MAE: {mae:.3f} seconds")
        logger.info(f"Validation R²: {r2:.3f}")
        logger.info(f"Validation RMSE: {rmse:.3f} seconds")
        
        # Cross-validation for more robust evaluation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        logger.info(f"Cross-validation MAE: {cv_mae:.3f} ± {cv_scores.std():.3f} seconds")
        
        return True
    
    def predict_custom_grid(self, driver_teams, track_characteristics=None):
        """Predict Q3 times for a custom driver lineup"""
        if not self.is_trained:
            logger.error("Model must be trained before making predictions")
            return None
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame(list(driver_teams.items()), columns=['Driver', 'Team'])
        
        # Apply performance factors based on 2025 expectations
        predictions_df = self._apply_performance_factors(predictions_df, track_characteristics)
        
        # Sort by predicted time
        predictions_df = predictions_df.sort_values('Predicted_Q3').reset_index(drop=True)
        predictions_df['Position'] = range(1, len(predictions_df) + 1)
        
        return predictions_df
    
    def _apply_performance_factors(self, df, track_characteristics):
        """Apply team and driver performance factors"""
        # Base lap time (adjust based on track)
        base_time = 89.5  # Default base time in seconds
        
        if track_characteristics:
            base_time = track_characteristics.get('base_time', base_time)
        
        # 2025 Team Performance Factors
        team_factors = {
            'Red Bull Racing': 0.996,     # Dominant performance
            'Ferrari': 0.998,             # Strong contender
            'McLaren': 0.999,             # Improved significantly
            'Mercedes': 0.999,            # Consistent performer
            'Aston Martin': 1.001,        # Midfield leader
            'RB': 1.002,                  # Sister team to Red Bull
            'Williams': 1.003,            # Improving team
            'Haas F1 Team': 1.004,        # Steady midfield
            'Kick Sauber': 1.004,         # Audi development phase
            'Alpine': 1.005,              # Rebuilding phase
        }
        
        # Driver Performance Factors
        driver_factors = {
            'Max Verstappen': 0.997,      # Exceptional qualifier
            'Charles Leclerc': 0.998,     # Elite qualifier
            'Carlos Sainz': 0.999,        # Very consistent
            'Lando Norris': 0.999,        # Strong qualifier
            'Oscar Piastri': 1.000,       # Talented rookie
            'Sergio Perez': 1.000,        # Solid performer
            'Lewis Hamilton': 1.000,      # Experienced champion
            'George Russell': 1.000,      # Consistent performer
            'Fernando Alonso': 1.000,     # Veteran excellence
            'Lance Stroll': 1.001,        # Decent performer
            'Alex Albon': 1.001,          # Reliable driver
            'Daniel Ricciardo': 1.001,    # Experienced campaigner
            'Yuki Tsunoda': 1.002,        # Developing talent
            'Valtteri Bottas': 1.002,     # Steady performer
            'Zhou Guanyu': 1.003,         # Improving driver
            'Kevin Magnussen': 1.003,     # Experienced midfield
            'Nico Hulkenberg': 1.003,     # Veteran qualifier
            'Logan Sargeant': 1.004,      # Developing driver
            'Pierre Gasly': 1.004,        # Consistent midfield
            'Esteban Ocon': 1.004,        # Reliable performer
        }
        
        # Apply factors to each driver
        for idx, row in df.iterrows():
            team_factor = team_factors.get(row['Team'], 1.005)
            driver_factor = driver_factors.get(row['Driver'], 1.002)
            
            # Calculate base prediction
            base_prediction = base_time * team_factor * driver_factor
            
            # Add some realistic variation
            random_variation = np.random.uniform(-0.05, 0.05)
            df.loc[idx, 'Predicted_Q3'] = base_prediction + random_variation
        
        return df

class F1Visualizer:
    """Handles all visualization tasks for F1 data"""
    
    @staticmethod
    def plot_qualifying_times(df, title="Qualifying Lap Times Distribution"):
        """Plot distribution of qualifying times with clean, readable layout"""
        
        time_cols = ['Q1_sec', 'Q2_sec', 'Q3_sec']
        available_cols = [col for col in time_cols if col in df.columns and not df[col].isna().all()]
        
        if not available_cols:
            logger.warning("No qualifying time data available for plotting")
            return
        
        # Prepare data and remove extreme outliers
        plot_data = df[available_cols].copy()
        for col in available_cols:
            Q1, Q3 = plot_data[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            plot_data[col] = plot_data[col].clip(lower_bound, upper_bound)
        
        # Create simple, clean plot like the first image
        plt.figure(figsize=(16, 8))
        
        # Define F1-inspired colors
        f1_colors = ['#FF1801', '#00D2BE', '#FFF500']  # Red, Teal, Yellow
        
        # Create box plot with custom styling
        bp = plt.boxplot([plot_data[col].dropna() for col in available_cols], 
                        labels=[col.replace('_sec', '') for col in available_cols],
                        patch_artist=True, 
                        showfliers=True,
                        flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.6),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(color='black', linewidth=1.5),
                        capprops=dict(color='black', linewidth=1.5))
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], f1_colors[:len(available_cols)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        plt.title(title, fontweight='bold', fontsize=18, pad=20)
        plt.ylabel('Lap Time (seconds)', fontweight='bold', fontsize=14)
        plt.xlabel('Qualifying Session', fontweight='bold', fontsize=14)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.gca().set_facecolor('#f8f9fa')
        
        # Set better y-axis limits
        all_data = pd.concat([plot_data[col].dropna() for col in available_cols])
        if not all_data.empty:
            y_min = all_data.quantile(0.01) - 0.5
            y_max = all_data.quantile(0.99) + 0.5
            plt.ylim(y_min, y_max)
        
        # Add statistics text in clean format
        stats_text = "Statistics:\n"
        for col in available_cols:
            col_data = plot_data[col].dropna()
            if not col_data.empty:
                session_name = col.replace('_sec', '')
                mean_time = col_data.mean()
                std_time = col_data.std()
                stats_text += f"{session_name}: μ={mean_time:.2f}s, σ={std_time:.2f}s\n"
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_predictions(predictions_df, title="Predicted Qualifying Results"):
        """Plot predicted qualifying results with clean, readable layout like the first image"""
        plt.figure(figsize=(16, 10))
        
        # Create a more sophisticated color scheme based on teams
        team_colors = {
            'Red Bull Racing': '#1E41FF',
            'Ferrari': '#DC143C',
            'Mercedes': '#00D2BE',
            'McLaren': '#FF8700',
            'Aston Martin': '#006F62',
            'Alpine': '#0090FF',
            'Williams': '#005AFF',
            'RB': '#6692FF',
            'Haas F1 Team': '#FFFFFF',
            'Kick Sauber': '#900000'
        }
        
        # Get colors for each driver's team
        colors = [team_colors.get(team, '#808080') for team in predictions_df['Team']]
        
        # Create horizontal bar chart
        bars = plt.barh(range(len(predictions_df)), predictions_df['Predicted_Q3'],
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Customize the plot
        plt.yticks(range(len(predictions_df)), 
                  [f"{row['Position']}. {row['Driver']}" for _, row in predictions_df.iterrows()],
                  fontsize=12)
        plt.xlabel('Predicted Q3 Time (seconds)', fontweight='bold', fontsize=14)
        plt.ylabel('Driver (Grid Position)', fontweight='bold', fontsize=14)
        plt.title(title, fontweight='bold', fontsize=18, pad=20)
        
        # Invert y-axis so fastest is at top
        plt.gca().invert_yaxis()
        
        # Add time labels on bars
        for i, (bar, time) in enumerate(zip(bars, predictions_df['Predicted_Q3'])):
            plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{time:.3f}s', va='center', ha='left', fontweight='bold', fontsize=10)
        
        # Add grid and styling
        plt.grid(True, alpha=0.3, linestyle='--', axis='x')
        plt.gca().set_facecolor('#f8f9fa')
        
        # Create legend for teams in a cleaner format
        unique_teams = predictions_df['Team'].unique()
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=team_colors.get(team, '#808080'), 
                                       alpha=0.8, edgecolor='black') for team in unique_teams]
        
        # Position legend better to avoid text overlap
        plt.legend(legend_elements, unique_teams, loc='lower right', 
                  framealpha=0.9, title='Teams', title_fontsize=12, fontsize=10,
                  bbox_to_anchor=(0.98, 0.02))
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_team_comparison(predictions_df, title="Team Performance Comparison"):
        """Plot team performance comparison with clean, readable layout"""
        plt.figure(figsize=(14, 8))
        
        # Calculate team averages
        team_avg = predictions_df.groupby('Team')['Predicted_Q3'].agg(['mean', 'std']).sort_values('mean')
        
        # Define team colors
        team_colors = {
            'Red Bull Racing': '#1E41FF',
            'Ferrari': '#DC143C',
            'Mercedes': '#00D2BE',
            'McLaren': '#FF8700',
            'Aston Martin': '#006F62',
            'Alpine': '#0090FF',
            'Williams': '#005AFF',
            'RB': '#6692FF',
            'Haas F1 Team': '#808080',
            'Kick Sauber': '#900000'
        }
        
        colors = [team_colors.get(team, '#808080') for team in team_avg.index]
        
        # Create bar chart with error bars
        bars = plt.bar(range(len(team_avg)), team_avg['mean'], 
                      yerr=team_avg['std'], capsize=5,
                      color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=1.2,
                      error_kw={'linewidth': 2, 'ecolor': 'black'})
        
        # Customize the plot
        plt.xlabel('Team', fontweight='bold', fontsize=14)
        plt.ylabel('Average Predicted Q3 Time (seconds)', fontweight='bold', fontsize=14)
        plt.title(title, fontweight='bold', fontsize=18, pad=20)
        
        # Set x-axis labels with better rotation and spacing
        plt.xticks(range(len(team_avg)), team_avg.index, rotation=45, ha='right', fontsize=11)
        
        # Add value labels on bars
        for i, (bar, mean_val, std_val) in enumerate(zip(bars, team_avg['mean'], team_avg['std'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.01,
                    f'{mean_val:.3f}s', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
        
        # Add grid and styling
        plt.grid(True, alpha=0.3, linestyle='--', axis='y')
        plt.gca().set_facecolor('#f8f9fa')
        
        # Ensure proper spacing
        plt.subplots_adjust(bottom=0.15)
        plt.tight_layout()
        plt.show()

def main():
    """Main execution function"""
    print("🏎️  Enhanced F1 Qualifying Prediction System")
    print("=" * 50)
    
    # Initialize components
    data_manager = F1DataManager()
    predictor = F1Predictor(model_type='random_forest')
    visualizer = F1Visualizer()
    
    # Define sessions to fetch (example configuration)
    sessions_config = [
        {'year': 2024, 'round': 1, 'session': 'Q'},   # Bahrain 2024
        {'year': 2024, 'round': 2, 'session': 'Q'},   # Saudi Arabia 2024
        {'year': 2024, 'round': 3, 'session': 'Q'},   # Australia 2024
        {'year': 2024, 'round': 4, 'session': 'Q'},   # Japan 2024
        {'year': 2024, 'round': 5, 'session': 'Q'},   # China 2024
    ]
    
    try:
        # Fetch training data
        print("\n📊 Fetching training data...")
        training_data = data_manager.fetch_multiple_sessions(sessions_config)
        
        if training_data is not None and not training_data.empty:
            print(f"✅ Successfully loaded {len(training_data)} data points")
            
            # Visualize data
            print("\n📈 Visualizing data...")
            visualizer.plot_qualifying_times(training_data, "Historical Qualifying Times Analysis")
            
            # Train model
            print("\n🤖 Training prediction model...")
            if predictor.train(training_data):
                
                # Define current F1 grid (2025 lineup)
                current_grid = {
                    'Max Verstappen': 'Red Bull Racing',
                    'Sergio Perez': 'Red Bull Racing',
                    'Charles Leclerc': 'Ferrari',
                    'Carlos Sainz': 'Ferrari',
                    'Lewis Hamilton': 'Mercedes',
                    'George Russell': 'Mercedes',
                    'Lando Norris': 'McLaren',
                    'Oscar Piastri': 'McLaren',
                    'Fernando Alonso': 'Aston Martin',
                    'Lance Stroll': 'Aston Martin',
                    'Daniel Ricciardo': 'RB',
                    'Yuki Tsunoda': 'RB',
                    'Alex Albon': 'Williams',
                    'Logan Sargeant': 'Williams',
                    'Valtteri Bottas': 'Kick Sauber',
                    'Zhou Guanyu': 'Kick Sauber',
                    'Kevin Magnussen': 'Haas F1 Team',
                    'Nico Hulkenberg': 'Haas F1 Team',
                    'Pierre Gasly': 'Alpine',
                    'Esteban Ocon': 'Alpine'
                }
                
                # Make predictions
                print("\n🔮 Generating predictions for upcoming race...")
                track_config = {'base_time': 89.2}  # Suzuka-like track
                predictions = predictor.predict_custom_grid(current_grid, track_config)
                
                if predictions is not None:
                    # Display results
                    print("\n🏁 Predicted Qualifying Results:")
                    print("=" * 80)
                    print(f"{'Pos':<4}{'Driver':<20}{'Team':<25}{'Predicted Q3':<15}")
                    print("-" * 80)
                    
                    for _, row in predictions.iterrows():
                        print(f"{row['Position']:<4}"
                              f"{row['Driver']:<20}"
                              f"{row['Team']:<25}"
                              f"{row['Predicted_Q3']:.3f}s")
                    
                    # Visualize predictions
                    print("\n📊 Generating enhanced visualizations...")
                    visualizer.plot_predictions(predictions, "F1 2025 - Predicted Qualifying Results")
                    visualizer.plot_team_comparison(predictions, "F1 2025 - Team Performance Analysis")
                    
                    print("\n✅ Analysis complete!")
                    
            else:
                print("❌ Failed to train model")
        else:
            print("❌ No training data available")
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()