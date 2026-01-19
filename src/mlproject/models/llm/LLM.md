### LLM training explanation

1. Prepare data(All the same steps done as for the classic ML models)
2. Create a Dataset class(ClassificationDataset) to transform our version to PyTorch version of dataset using tokenizer to encode text. Further dataset loaded into DataLoader, which is suitable for training loop
3. Load model and move to cuda device, initialize optimizer as AdamW
4. Run training loop for 2 epochs(toy demo).
5. Evaluate model using Accuracy, Precision, Recall and F1 score

### Hardware constraints

The model was trained in Google Colab environment with a T4 GPU. One epoch takes no more than 2 minutes to run. 

### Model

I chose BERT base uncased model since it is one of the most lightweight and powerful transformer models. 
-- It can be easily fine-tuned on a single small GPU as was demostrated in the notebook.
-- It has better generalization because of "uncased". It treats "Word" and "word" as the same words.


