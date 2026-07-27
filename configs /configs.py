#MODEL AND TRAINING HYPERPARAMETERS
model_id = "google/gemma-3-4b-it"
tokenizer_id = "google/gemma-3-4b-it" #add the standard tokenizer
lr = 3e-5
TRAIN_BATCH = 8
VAL_BATCH = 4
GRAD_ACCUM = 8
NUM_EPOCHS = 1
WEIGHT_DECAY = 0.01
SAVE_STEPS = 100
EVAL_STEPS = 1000
LOGGING_STEPS = 100
OUTPUT_DIR = "./outputs_gemma"
