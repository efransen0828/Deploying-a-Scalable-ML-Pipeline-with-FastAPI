# Model Card for Income Prediction Model

Model Details
    Model Type: Random Forest Classifier
    Dataset Used: Census Income Dataset
    Version: 1.0
    Training Framework: Scikit-learn
    Author: Erika Fransen
    Contact Information: efrans2@wgu.edu

Intended Use
    The Income Prediction Model is designed to predict whether an individual's annual income exceeds $50,000 based on demographic data. This model is intended for demographic analysis and
    educational purposes only. It should not be used for critical decision-making, such as employment or credit decisions, without additional safeguards and validations.

Training Data
    Source: Census Income Dataset
    Description: The dataset contains demographic and income-related features for individuals, such as age, education, workclass, marital status, occupation, and more.
    Split: 80% training, 20% testing

Evaluation Data
    Source: Same as the training dataset, split into test and training subsets.
    Description: The test set consists of 20% of the original dataset, stratified to maintain class balance.

Metrics
    Overall Performance:
        Precision: 0.7419
        Recall: 0.6384
        F1 Score: 0.6863
    Slice Performance: Metrics were calculated for unique values within each categorical feature.
    Below are a few highlights:
        Workclass:
            Best F1 Score: Without-pay (F1: 1.0000)
            Worst F1 Score: ? (F1: 0.5000)
        Education:
            Best F1 Score: 1st-4th and Preschool (F1: 1.0000)
            Worst F1 Score: 10th (F1: 0.2353)
For full slice metrics, refer to slice_output.txt.

Ethical Considerations
    Bias and Fairness:
        Some slices, particularly those with limited representation (e.g., education: 10th), have poor performance.
        Disparities in slice metrics may reflect inherent biases in the dataset, requiring careful interpretation.
    Potential Risks:
        Decisions made based on incorrect predictions could negatively impact individuals or groups, particularly those in underrepresented categories.
        Over-reliance on the model may perpetuate societal biases present in the dataset.
        
Caveats and Recommendations
    This model is trained on U.S. Census data and may not generalize well to other populations or data distributions.
    The model assumes that the provided features accurately represent the target variable, income, which may not account for all relevant factors.
    Users should regularly evaluate model performance, especially if applied to new datasets or populations.
    
Recommendations for improvement:
    Incorporate additional data sources to improve representation for underrepresented groups.
    Apply fairness-aware machine learning techniques to reduce bias in predictions.
