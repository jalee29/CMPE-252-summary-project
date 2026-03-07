### Files:
- **summarization**: the run_summarization.py script
- **text-summarization**: trained model
- **test-00000-of-00001.parquet**: data
- **test.ipynb**: data preprocessing, for making subsets of data
- **test.csv**: subset of data for training, about 100 examples
- **val.csv**: subset of data for validation, about 50 examples

### Terminal command for running run_summarization:
```shell
python run_summarization.py `
     --model_name_or_path facebook/bart-base `
     --do_train `
     --do_eval `
     --train_file ../test.csv `
     --validation_file ../val.csv `
     --output_dir ../test-summarization ` 
     --per_device_train_batch_size=4 `
     --per_device_eval_batch_size=4 `
     --predict_with_generate `
     --text_column article `
     --summary_column abstract
```