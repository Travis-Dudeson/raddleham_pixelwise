

### ------------------------------


# --- bayesian optimization

from tuning.bayesian_optimization import main as run_bayesian_optimization

if __name__ == "__main__":
    run_bayesian_optimization()

print("::::::::::: THIS MARKS WHEN THE WHOLE CODE STARTS :::::::::::::")



### ------------------------------
# --- for grid search


# from tuning.grid_search import main as tune_hyperparameters

# if __name__ == "__main__":
#     tune_hyperparameters()


### ------------------------------
# --- for grid search SMALL


# from tuning.small_grid_search import main as tune_hyperparameters

# if __name__ == "__main__":
#     tune_hyperparameters()


### ------------------------------

# --- for GPU

# import tensorflow as tf
# from tuning.small_grid_search import main as tune_hyperparameters

# # Check for GPU availability and log it
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# # Set GPU memory growth to avoid allocating all memory at once
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# # Enable mixed precision to improve performance on modern GPUs
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

# if __name__ == "__main__":
#     tune_hyperparameters()

































