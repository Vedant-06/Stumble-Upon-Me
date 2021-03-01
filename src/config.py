from transformers import AutoTokenizer

DEVICE = "cuda"
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
TEST_BATCH_SIZE = 1
EPOCHS = 20
RESULTS_FILE = "submissions.csv"
MODEL_PATH = "input/saved_models/model_"
TRAINING_FILE = "input/train.tsv"
TESTING_FILE = "input/test.tsv"
TOKENIZER = AutoTokenizer.from_pretrained("xlm-roberta-base")
