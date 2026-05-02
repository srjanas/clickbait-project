# lin_final_project

Our project is on clickbait classification. 

## Dataset

We use the clickbait dataset from Hugging Face:

`christinacdl/clickbait_detection_dataset`

In order to run the code, you will need to have the following packages downloaded:

1. python
2. numpy
3. pandas
4. sklearn
5. matplotlib
6. seaborn 
7. transformers
8. torch
9. datasets

We compare the following models:

1. Logistic Regression
2. SVM
3. LSTM
4. BERT 

We recommend running the training files in that order. Afterwards, to generate the results and the plots, run `plot_results.py`

This will generate:

- `outputs/results_table.csv`
- `outputs/metric_comparison.png`
- `outputs/*_confusion_matrix.png`
- `outputs/*_errors.csv`

## Full Workflow

To fully reproduce results, here is the complete workflow:
1. Download dataset:
- `python src/load_data.py`

2. Train models:
- `python src/train_logreg.py`
- `python src/train_svm.py`
- `python src/train_lstm.py`
- `python src/train_bert.py`

3. Generate evaluation:
- `python src/plot_results.py`

## Notes
- BERT metrics may be reused from a previous run due to computational cost.
- Random seeds are not fixed, so minor variations may occur.
- All models use the same train/validation/test splits.