# MobiCost Analyzer: Smartphone Value Forecaster

![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MobiCost Analyzer is an intelligent web application that predicts smartphone price ranges based on technical specifications. Built with Streamlit and powered by machine learning, this tool helps consumers, retailers, and manufacturers understand market positioning and value propositions for mobile devices.

**Live Demo**: [https://mobicost-analyzer-smartphone-value-forecaster-ankit-parwatkar.streamlit.app/](https://mobicost-analyzer-smartphone-value-forecaster-ankit-parwatkar.streamlit.app/)

## Features

- 🚀 **Instant Price Range Prediction**: Classify phones into Budget, Mid-Range, Premium, or Luxury categories
- 📊 **Interactive Visualization**: Beautiful charts showing feature impact and prediction confidence
- 📱 **Real Device Recommendations**: Discover popular models in your predicted price range
- 🔍 **Feature Importance Analysis**: Understand which specifications most influence pricing
- 💡 **Marketing Recommendations**: Get strategic insights based on price category
- 🎨 **Modern UI**: Sleek dark theme with animated elements and responsive design

## How It Works

1. **Input Specifications**: Adjust technical parameters using intuitive sliders and checkboxes
2. **Predict Price Range**: Click the predict button to analyze the device
3. **View Results**: 
   - See predicted price category with confidence level
   - Explore popular models in that price range
   - Understand feature impact through visualizations
   - Get marketing strategy recommendations

## Technology Stack

### Core Components
- **Machine Learning**: XGBoost classifier trained on smartphone specifications
- **Frontend**: Streamlit for web interface with Plotly for visualizations
- **Styling**: Custom CSS with modern dark theme and animations
- **Data Processing**: Pandas and NumPy for feature engineering

### Python Libraries
```
streamlit
pandas
numpy
joblib
plotly
scikit-learn
xgboost
Pillow
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ankitparwatkar/MobiCost-Analyzer-Smartphone-Value-Forecaster.git
cd MobiCost-Analyzer-Smartphone-Value-Forecaster
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate    # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

## Deployment

The application is deployed using Streamlit Cloud. To deploy your own instance:

1. Create a GitHub repository with your code
2. Sign up for [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Click "New app" and connect your GitHub repository
4. Select branch and main file path (`app.py`)
5. Click "Deploy"

## Project Structure

```
MobiCost-Analyzer-Smartphone-Value-Forecaster/
├── app.py               # Main Streamlit application
├── model.ipynb          # Jupyter notebook for model training
├── phone_price_model.pkl # Trained machine learning model
├── scaler.pkl           # Feature scaler for preprocessing
├── requirements.txt     # Python dependencies
├── images/              # Phone model images
│   ├── galaxya14.png
│   ├── galaxys23.png
│   └── ... 
└── README.md            # This documentation
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Author

**Ankit Parwatkar**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ankitparwatkar)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ankitparwatkar)
[![Portfolio](https://img.shields.io/badge/Portfolio-%23000000.svg?style=for-the-badge&logo=firefox&logoColor=#FF7139)](https://share.streamlit.io/user/ankitparwatkar)

---

**Note**: This tool provides estimates based on market trends and specifications. Actual prices may vary based on brand, region, and market conditions.
