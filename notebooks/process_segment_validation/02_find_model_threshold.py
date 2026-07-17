""" 
Compute BirdNET confidence score threshold for a desired true positive rate.

"""
import os
import argparse
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import glm
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load data """
    data = pd.read_excel(file_path)
    data = data.dropna(subset=['classificationProbability', 'positive'])
    # Transform 'no'/'yes' to 0/1 in 'positive' column. No case sensitivity.
    data['positive'] = data['positive'].str.lower().map({'no': 0, 'yes': 1})
    return data

def fit_models(data):
    """Fit null and confidence models, and return both models."""
    null_model = glm('positive ~ 1', data=data, family=sm.families.Binomial()).fit()
    conf_model = glm('positive ~ classificationProbability', data=data, family=sm.families.Binomial()).fit()
    return null_model, conf_model

def create_aic_table(null_model, conf_model):
    """Create and return an AIC table comparing the null and confidence models."""
    aic_table = pd.DataFrame({
        'Model': ['null_model', 'conf_model'],
        'AIC': [null_model.aic, conf_model.aic]
    })
    return aic_table.sort_values(by='AIC', ascending=True)

def calculate_cutoff(conf_model, probability=0.95):
    """Calculate and return the cutoff confidence score for a given true positive rate."""
    logit_value = np.log(probability / (1 - probability))
    cutoff = (logit_value - conf_model.params['Intercept']) / conf_model.params['classificationProbability']
    return cutoff

def plot_results(
        focal_species, data, prediction_range_conf, predictions_conf, cutoff, proba, aic_table):
    """Plot the scatter plot of data, model predictions, and the cutoff line."""
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of the original data (basic matplotlib)
    plt.scatter(data['classificationProbability'], data['positive'],
                c='black', s=100, alpha=0.2)

    # Add the line for model predictions
    plt.plot(prediction_range_conf, predictions_conf, 
             linewidth=4, color=(0, 0.75, 1, 0.5))

    # Add the vertical line for the cutoff at 95% confidence
    plt.axvline(x=cutoff, color='red', linewidth=2)

    # Add textbox with aic_table information at lower right corner
    aic_text = aic_table.to_string(index=False)
    plt.gcf().text(0.87, 0.12, aic_text, fontsize=10,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   color='grey')

    # Customize the plot
    plt.title(f'Confidence scores - pr(tpr={proba}) = {round(cutoff, 3)}')
    plt.xlabel('Confidence score')
    plt.ylabel('pr(BirdNET prediction is correct)')
    plt.xlim([min(prediction_range_conf), max(prediction_range_conf)])

    # Save plot
    plt.savefig(f'{focal_species}_confidence_score_plot.png', dpi=300)
    plt.close()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process file path and perform analysis.')
    parser.add_argument('-i', '--input', required=True, help='Input file path')
    parser.add_argument('-p', '--probability', type=float, default=0.95, help='Desired true positive probability')
    args = parser.parse_args()

    # Extract the species code from the file path
    file_path = args.input
    probability = args.probability

    focal_species = os.path.basename(file_path).split('_')[0] + ' ' + os.path.basename(file_path).split('_')[1]

    # Load and preprocess the data
    data = load_data(file_path)

    # Fit models
    null_model, conf_model = fit_models(data)

    # Compute True positive rate
    tpr = data['positive'].dropna().sum() / len(data.dropna(subset=['positive']))

    # Display the AIC table
    aic_table = create_aic_table(null_model, conf_model)
    print(aic_table)

    # Calculate predictions and the cutoff value
    prediction_range_conf = np.arange(0, 1.001, 0.001)
    predictions_conf = conf_model.predict(pd.DataFrame({'classificationProbability': prediction_range_conf}))
    cutoff = calculate_cutoff(conf_model, probability)

    # Print the cutoff value
    print(f'{focal_species}')
    print(f'Cutoff for {probability*100}% confidence: {cutoff}')
    print(f'True Positive Rate: {tpr}')

    # Plot the results
    plot_results(
        focal_species, 
        data, 
        prediction_range_conf, 
        predictions_conf, 
        cutoff, 
        probability,
        aic_table,
        )

# Run the main function
if __name__ == "__main__":
    main()
