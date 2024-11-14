# NLP_Assignment_2

# NLP_Team_13: Tokenizer Training and Model Training

---

## Overview
This project addresses two main tasks: training six different tokenizers on a 10k dataset to evaluate fertility scores and fine-tuning a language model for optimized NLP performance. For Task 1, we trained six tokenizers and selected the best-performing one based on fertility scores. In Task 2, the LLaMA model architecture was adjusted to stay under 100M parameters, using the top-performing tokenizer to tokenize the dataset. Perplexity scores were recorded at every 0.1 epoch, and a matrix of scores for each epoch was created. The model was also tested on 10 prompts to evaluate its performance.

---

## Project Tasks

---

### Task 1: Tokenizer Training
We trained six different tokenizers on samples from a previously scraped dataset to evaluate performance based on fertility scores:
1. **BertWordPieceTokenizer**
2. **SentencePieceBPETokenizer**
3. **gpt2byteleveltokenizer**
4. **ByteLevelBPETokenizer**
5. **SentencePieceUnigramTokenizer**
6. **SpaCyTokenizer**

---

### Task 1: Tokenizer Training
We trained six different tokenizers on samples from a previously scraped dataset to evaluate performance based on fertility scores:

| Tokenizer                        | Kaggle Link                               |
|-----------------------------------|-------------------------------------------|
| **BertWordPieceTokenizer**        | [🟩 Kaggle Link](https://www.kaggle.com/code/vaishnavkoka24310069/bertwordpiecetokenizer/) |
| **SentencePieceBPETokenizer**     | [🟩 Kaggle Link](https://www.kaggle.com/code/ramanand9/sentencepiecebpetokenizer/) |
| **gpt2byteleveltokenizer**        | [🟩 Kaggle Link](https://www.kaggle.com/code/vaishnavkoka24310069/gpt2byteleveltokenizer/) |
| **ByteLevelBPETokenizer**         | [🟩 Kaggle Link](https://www.kaggle.com/code/vaishnavkoka24310069/bytelevelbpetokenizer/) |
| **SentencePieceUnigramTokenizer** | [🟩 Kaggle Link](https://www.kaggle.com/code/vaishnavkoka24310069/sentencepieceunigramtokenizer/) |
| **SpaCyTokenizer**                | [🟩 Kaggle Link](https://www.kaggle.com/code/vaishnavkoka24310069/spacytokenizer/) |



Each tokenizer was evaluated on fertility, and a matrix showing fertility scores and dataset size was generated. This matrix, available in [NLP-Assignment-2_Tokenizer_Training.pdf](https://github.com/vaishnavkoka/NLP_Assignment_2/blob/main/Results/NLP-%20Assignment-2_Tokenizer_Training.pdf), showed that **SentencePieceBPETokenizer** had the lowest fertility score and was selected as the optimal tokenizer for Task 2.

---

### Task 2: Model Training

---
The **LLaMA** model architecture was adjusted to ensure fewer than 100M parameters for efficient training. Using the SentencePieceBPETokenizer, we tokenized our dataset and trained the model, logging perplexity scores at each 0.1 epoch. A matrix of perplexity scores by epoch was generated and added to the repository. Additionally, the model was tested on 10 sample prompts, with test outputs available in the repository screenshots.

---
![Model_training](https://github.com/user-attachments/assets/139e8689-b490-4076-9776-4986cf6d0071)
`Fig: model training for every 0.1 epoch`

![model_output_for_prompts](https://github.com/user-attachments/assets/23f6160b-ecae-41ac-8953-12a117a5de9a)


---

## Contributions

Each team member contributed significantly to both tasks, as detailed below:

- **Vaishnav Koka**
  - Set up and organized the Task 1 training pipeline.
  - Calculated and compiled fertility scores, generating the matrix.
  - Tested and documented model outputs for 10 prompts in Task 2.

- **Ramanand**
  - Fine-tuned the LLaMA model to meet the <100M parameter constraint.
  - Conducted training on the tokenized dataset and recorded perplexity scores at each epoch.
  - Created performance metric reports and final model output documentation.

- **Isha Jain**
  - Configured tokenizer training with multiprocessing for efficient dataset handling.
  - Conducted initial fertility analysis and supported SentencePieceBPETokenizer selection.
  - Documented Task 1 outputs and validated score accuracy.

- **Yash Sahu**
  - Developed comparison framework for fertility metrics evaluation.
  - Supported tokenizer selection and training script configuration.
  - Created fertility and perplexity matrix visualizations.
---

## Usage

1. **Install Dependencies**  
   Run `requirements.txt` to set up the environment for tokenizer and model training.

2. **Run Tasks Sequentially**  
   - **Task 1**: Execute the tokenizer training scripts and save fertility scores to the output matrix.
   - **Task 2**: Configure and fine-tune the LLaMA model with the selected tokenizer, logging perplexity scores.

3. **Access Output Files**  
   All relevant matrices, screenshots, and test outputs are available in the project repository for review.

---

## Outputs

- **Tokenizer Fertility Scores**: See [NLP-Assignment-2_Tokenizer_Training.pdf](https://github.com/vaishnavkoka/NLP_Assignment_2/blob/main/Results/NLP-%20Assignment-2_Tokenizer_Training.pdf).
  ![image](https://github.com/user-attachments/assets/1bcbeba8-9c55-4dd8-8151-eac9f08d13b2)

- **Model Perplexity Matrix**: Logged in screenshots folder, showing scores for each epoch.
- **Prompt Output Tests**: Screenshots of model responses for 10 test prompts.

---

## Acknowledgments
Our team collaborated effectively to parallelize and document each stage of the project. Special thanks to each member for their dedication to completing the tasks in a timely and efficient manner.

--- 
References: https://github.com/AamodThakur/NLP_Pre_Training
