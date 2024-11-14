# NLP_Assignment_2

## NLP_Team_13: Tokenizer Training and Model Training

### Overview
This project focuses on two main tasks: training tokenizers and fine-tuning a language model for NLP performance.

- **Task 1: Training Tokenizers**
  - Trained six different tokenizers on a 10k dataset.
  - Evaluated each tokenizer's performance using fertility scores.
  - Selected the best-performing tokenizer based on fertility score results.

- **Task 2: Fine-Tuning Language Model**
  - Adjusted the LLaMA model architecture to stay within a 100M parameter limit.
  - Used the top-performing tokenizer from Task 1 to tokenize the dataset.
  - Recorded perplexity scores at every 0.1 epoch.
  - Created a matrix of perplexity scores for each epoch.
  - Tested the fine-tuned model on 10 prompts to evaluate performance.


### Project Tasks

### Task 1.1: Tokenizer Training
We trained six different tokenizers on samples from a previously scraped dataset to evaluate performance based on fertility scores:

- **Tokenizer Fertility Scores**: See [NLP-Assignment-2_Tokenizer_Training.pdf](https://github.com/vaishnavkoka/NLP_Assignment_2/blob/main/Results/NLP-%20Assignment-2_Tokenizer_Training.pdf).
  ![image](https://github.com/user-attachments/assets/1bcbeba8-9c55-4dd8-8151-eac9f08d13b2)


| Tokenizer                        | Kaggle Link                               |
|-----------------------------------|-------------------------------------------|
| **BertWordPieceTokenizer**        | [🟩 Kaggle Link](https://www.kaggle.com/code/vaishnavkoka24310069/bertwordpiecetokenizer/) |
| **SentencePieceBPETokenizer**     | [🟩 Kaggle Link](https://www.kaggle.com/code/ramanand9/sentencepiecebpetokenizer/) |
| **gpt2byteleveltokenizer**        | [🟩 Kaggle Link](https://www.kaggle.com/code/vaishnavkoka24310069/gpt2byteleveltokenizer/) |
| **ByteLevelBPETokenizer**         | [🟩 Kaggle Link](https://www.kaggle.com/code/vaishnavkoka24310069/bytelevelbpetokenizer/) |
| **SentencePieceUnigramTokenizer** | [🟩 Kaggle Link](https://www.kaggle.com/code/vaishnavkoka24310069/sentencepieceunigramtokenizer/) |
| **SpaCyTokenizer**                | [🟩 Kaggle Link](https://www.kaggle.com/code/vaishnavkoka24310069/spacytokenizer/) |



Each tokenizer was evaluated on fertility, and a matrix showing fertility scores and dataset size was generated. This matrix, available in [NLP-Assignment-2_Tokenizer_Training.pdf](https://github.com/vaishnavkoka/NLP_Assignment_2/blob/main/Results/NLP-%20Assignment-2_Tokenizer_Training.pdf), showed that **SentencePieceBPETokenizer** had the lowest fertility score and was selected as the optimal tokenizer for Task 2.

### Task 1.2: Dataset

| Tokenizer                        | Kaggle Link                               |
|-----------------------------------|-------------------------------------------|
| **Hindi dataset 3k files**        | [🟩 Kaggle Link](https://www.kaggle.com/datasets/ramanand9/hindi-dataset-3k-files) |
| **Hindi dataset 15k files**     | [🟩 Kaggle Link](https://www.kaggle.com/datasets/ramanand9/hindi-dataset-10k-files) |
| **Hindi dataset 100k files**        | [🟩 Kaggle Link](https://www.kaggle.com/datasets/ramanand9/100k-hindi-text-files-for-nlp-task) |




### Task 2.1: Model Training

#### LLaMA Model Training and Evaluation
This project involves training a modified LLaMA model with fewer than 100M parameters for efficient training and evaluation. The model is tokenized using SentencePieceBPETokenizer.

- Tokenized dataset using SentencePieceBPETokenizer.
- Trained the model and logged *perplexity scores* at every 0.1 epoch.
- Generated a matrix of perplexity scores by epoch for analysis.
- Tested the model on 10 sample prompts.
- *Output screenshots* showcasing the model's performance are included in the repository.


| Model weights                     | Kaggle Link                               |
|-----------------------------------|-------------------------------------------|
| **Transformer weight**        | [🟩 Kaggle Link](https://www.kaggle.com/datasets/ramanand9/ps-v2-output/) |


![Model_training](https://github.com/user-attachments/assets/139e8689-b490-4076-9776-4986cf6d0071)

`Fig1: model training for every 0.1 epoch`



![training_epoch](https://github.com/user-attachments/assets/4dab3cfd-0b45-4192-ae76-cd2efeac59db)


`Fig2: Model trained on 4.7M tokens for 10 epocs in 9 hours on 2 GPU T4`



### Task 2.2: Model Prediction

![model_output_for_prompts](https://github.com/user-attachments/assets/23f6160b-ecae-41ac-8953-12a117a5de9a)

`Fig3: Prompts output`



### Contributions
Each team member contributed significantly to both tasks, as detailed below:
- **Vaishnav Koka**: Created code for Tokenizers and trained tokenizers to calculate fertility scores, and Git Hub documentation.
- **Ramanand**: Created datasets, did Transformer model training and fine-tuning, and Git Hub documentation.
- **Isha Jain**: Created code for tokenizers and trained tokenizers to calculate fertility scores, and Git Hub documentation.
- **Yash Sahu**: Created tokenizers code, transformer fine-tuning, and Git Hub documentation.


### Acknowledgments
- Our team worked in a synchronous manner so as to not burden a single person, and get the work done more effectively and in a timely manner.
- We would like to thank Amod Thakur for providing tutorial sessions.
- Of course, it wouldn't have been possible without the teaching of Mayank sir.

References: 
1. https://github.com/AamodThakur/NLP_Pre_Training
2. https://huggingface.co/learn/nlp-course/en/chapter2/4
3. https://huggingface.co/learn/nlp-course/en/chapter7/6
4. https://colab.research.google.com/drive/1YGVg_wM_pYjrsXwVtTxm5Qr1KiHecETO?usp=sharing
5. https://colab.research.google.com/drive/1K4OT3t_eLepT33uutRpjbONrz_hjoLAM?usp=sharing
6. https://drive.google.com/drive/folders/10-YOfp70fn9TOUZJ5FVLk-anfw_T8A0X
