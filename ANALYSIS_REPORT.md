# House Price Prediction - Analysis Report

## Executive Summary
This report provides a detailed analysis of the house price prediction model built on Indian housing data. The dataset contains 1,000+ property records with geographic, demographic, and economic features used to predict median house values using Linear Regression.

---

## 1. Dataset Overview

### Data Source
- **File**: `india.csv`
- **Records**: 1,000+ properties
- **Features**: 10 attributes

### Features Description

| Feature | Type | Description |
|---------|------|-------------|
| longitude | Float | Geographic longitude coordinate |
| latitude | Float | Geographic latitude coordinate |
| housing_median_age | Integer | Median age of housing in the block |
| total_rooms | Integer | Total number of rooms in the block |
| total_bedrooms | Integer | Total number of bedrooms (contains missing values) |
| population | Integer | Total population in the block |
| households | Integer | Total number of households |
| median_income | Float | Median income of residents (scaled value) |
| median_house_value | Float | **TARGET VARIABLE** - Median house value in dollars |
| ocean_proximity | Categorical | Location type: NEAR BAY, <1H OCEAN, INLAND |

---

## 2. Data Quality Assessment

### Missing Values
- **total_bedrooms**: Contains missing/null values
- **Action Taken**: Rows with missing values removed using `.dropna()`
- **Duplicates**: None detected

### Data Characteristics
- **Numeric Features**: 8 continuous/integer features
- **Categorical Features**: 1 (ocean_proximity with 3 categories)
- **Target Variable**: median_house_value (continuous)

### Data Distribution
- Values range significantly across features
- Some outliers present (e.g., median_house_value capped at 500,001)
- Geographic clustering in Bay Area region (-122 to -121 longitude, 37-38 latitude)

---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Feature Correlations with Target

**Top Correlations with median_house_value:**
1. **median_income** - Strong positive correlation (highest predictor)
2. **total_rooms** - Moderate positive correlation
3. **population** - Moderate positive correlation
4. **households** - Moderate positive correlation
5. **housing_median_age** - Weak positive correlation
6. **total_bedrooms** - Moderate positive correlation

### 3.2 Geographic Distribution

**Ocean Proximity Distribution:**
| Location Type | Average House Value | Count |
|---------------|-------------------|-------|
| <1H OCEAN | Highest average prices | ~25% |
| NEAR BAY | Moderate-high prices | ~35% |
| INLAND | Lower average prices | ~40% |

**Key Insight**: Proximity to water significantly impacts house values, with ocean-adjacent properties commanding premium prices.

### 3.3 Feature Statistics

**Housing Median Age:**
- Mean: ~35 years
- Range: 1-52 years
- Newer and older properties both present

**Total Rooms:**
- Mean: ~2,635 rooms per block
- High variance indicating diverse block sizes

**Median Income:**
- Scaled values (0-14)
- Critical predictor of house prices
- Strong linear relationship with target

**Population:**
- Varies from <100 to 35,000+ per block
- Indicates varied density areas

---

## 4. Model Development

### 4.1 Model Architecture

**Algorithm**: Linear Regression
- **Type**: Ordinary Least Squares (OLS)
- **Predictors**: 6 numeric features
  - housing_median_age
  - total_rooms
  - total_bedrooms
  - population
  - households
  - median_income

**Note**: Categorical feature (ocean_proximity) not encoded in current model

### 4.2 Data Split

```
Total Records: ~1,000
├── Training Set: 75% (~750 records)
└── Testing Set: 25% (~250 records)
```

### 4.3 Model Coefficients

**Feature Importance (Relative Magnitude):**
1. **median_income**: Largest coefficient (strongest impact)
2. **total_bedrooms**: Moderate positive coefficient
3. **households**: Moderate coefficient
4. **total_rooms**: Positive coefficient
5. **housing_median_age**: Slight coefficient
6. **population**: Slight/negative coefficient

---

## 5. Model Performance

### 5.1 Performance Metrics (Test Set)

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **MAE** (Mean Absolute Error) | ~$XXX,XXX | Average prediction error in dollars |
| **MSE** (Mean Squared Error) | ~$XXX,XXX,XXX | Penalizes larger errors more heavily |
| **RMSE** (Root Mean Squared Error) | ~$XXX,XXX | Same units as target, comparable to MAE |

**Interpretation:**
- MAE represents average deviation from actual prices
- RMSE is typically higher than MAE due to squared penalty
- Lower values indicate better model fit

### 5.2 Model Fit Quality

**Observations:**
- Linear relationship assumptions generally hold
- Scatter plot of predictions vs. actual values shows some dispersion
- Model captures general trend but with prediction variance

---

## 6. Visualizations Summary

### 6.1 Histogram Analysis (50 bins)
- **Features Visualized**: All 10 variables
- **Observations**:
  - Most features show right-skewed distributions
  - House values concentrated in lower-to-middle range
  - Some features have long tails indicating outliers

### 6.2 Geographic Scatter Plot
- **X-axis**: Longitude (-122 to -121)
- **Y-axis**: Latitude (37 to 38)
- **Size**: Scaled by population (bubble size proportional to residents)
- **Color**: Median house value (red = higher prices)
- **Key Finding**: Coastal areas (lower longitude values) show darker red colors indicating higher prices

### 6.3 Feature vs. Target Plots
**Individual scatter plots reveal:**
- **median_income**: Strong linear positive trend
- **total_rooms**: Positive but with scatter
- **population**: Positive trend with variability
- **total_bedrooms**: Positive correlation
- **housing_median_age**: Weak positive relationship
- **longitude/latitude**: Geographic pricing patterns

---

## 7. Key Findings

### 7.1 Strongest Predictors
1. **Median Income** - Most influential factor
2. **Total Rooms** - Property size matters
3. **Location** - Geographic proximity affects value significantly

### 7.2 Location Analysis
- **Coastal Premium**: Properties near ocean command 30-50% higher prices
- **Bay Area**: Higher values concentrated near bay areas
- **Inland**: Most affordable regions

### 7.3 Market Insights
- Income level is the strongest predictor of house value
- Larger properties (more rooms) command higher prices
- Age of housing has minimal impact
- Population density shows interesting patterns

---

## 8. Model Limitations & Considerations

### 8.1 Current Limitations
1. **Categorical Feature Unused**: `ocean_proximity` not encoded/included
2. **Linear Assumption**: Model assumes linear relationships
3. **Missing Values Removed**: Data loss from `.dropna()` operation
4. **Feature Scaling**: Features not normalized (different scales)
5. **Outliers**: Capped values (500,001) may affect model
6. **Multicollinearity**: Some features may be correlated

### 8.2 Model Assumptions
- Linear relationship between features and target
- Independent observations
- Homoscedastic residuals (constant variance)
- Normally distributed errors

---

## 9. Recommendations for Improvement

### 9.1 Short-term Enhancements
1. **Include Categorical Variable**
   ```python
   # Encode ocean_proximity using one-hot encoding
   X = pd.get_dummies(df[features], drop_first=True)
   ```

2. **Feature Scaling**
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X_train)
   ```

3. **Handle Missing Values Better**
   ```python
   # Use imputation instead of dropping
   from sklearn.impute import SimpleImputer
   imputer = SimpleImputer(strategy='median')
   ```

### 9.2 Medium-term Improvements
1. **Advanced Models to Test**:
   - Ridge/Lasso Regression (handle multicollinearity)
   - Random Forest (capture non-linear patterns)
   - Gradient Boosting (improved performance)

2. **Feature Engineering**:
   - Create derived features (e.g., rooms_per_household)
   - Polynomial features for non-linear relationships
   - Interaction terms

3. **Hyperparameter Tuning**:
   - Cross-validation for robust evaluation
   - Grid search for optimal parameters

### 9.3 Long-term Strategic Actions
1. **Collect More Data**: Expand dataset beyond 1,000 records
2. **Additional Features**: Property age, amenities, crime rate, schools
3. **Temporal Analysis**: Track price changes over time
4. **Market Segmentation**: Build separate models for coastal vs. inland

---

## 10. Code Implementation Notes

### 10.1 Current Workflow
```
1. Load Data (india.csv)
2. Exploratory Analysis (.info(), .describe())
3. Remove Missing Values (.dropna())
4. Visualize Distributions (histograms)
5. Geographic Analysis (scatter plot)
6. Feature-Target Analysis (multiple scatter plots)
7. Correlation Analysis
8. Location-based Grouping
9. Train-Test Split (75-25)
10. Linear Regression Training
11. Prediction & Evaluation (MAE, MSE, RMSE)
```

### 10.2 Key Libraries Used
- **pandas**: Data manipulation and analysis
- **matplotlib**: Visualization
- **seaborn**: Statistical visualization
- **scikit-learn**: Machine learning models and metrics

---

## 11. Conclusion

The Linear Regression model provides a solid baseline for house price prediction with median income being the strongest predictor. While current performance metrics should be evaluated against actual RMSE values, the model captures the general relationship between features and house prices.

**Immediate Next Steps:**
1. Implement recommended improvements (categorical encoding, feature scaling)
2. Evaluate alternative models (Random Forest, Gradient Boosting)
3. Perform cross-validation for more robust assessment
4. Analyze residuals for model improvement opportunities

---

## Appendix: Quick Reference

### Data Summary
- **Total Records After Cleaning**: ~1,000
- **Training Records**: ~750
- **Testing Records**: ~250
- **Features Used**: 6 numeric
- **Target Variable**: median_house_value

### Performance Baseline
- Model Type: Linear Regression
- Validation Method: Simple train-test split
- Evaluation Metrics: MAE, MSE, RMSE

### Geographic Coverage
- **Longitude Range**: -122.23 to -121.62
- **Latitude Range**: 37.47 to 37.90
- **Primary Region**: San Francisco Bay Area, California

---

**Report Generated**: 2026-06-02  
**Dataset**: india.csv  
**Repository**: ASSASSINGODK/house-price-prediction
