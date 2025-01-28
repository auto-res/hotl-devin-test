from new_pruning_method import test_pruning_method, train_with_pruning_method
from src.attention_head_pruning import prune_transformer

def my_pruning_func(model):
    """Wrapper function that applies our pruning method with default settings."""
    return prune_transformer(model, prune_ratio=0.2)

if __name__ == "__main__":
    # Run the test and print any errors that occur
    try:
        train_with_pruning_method(my_pruning_func)
        print("Test passed!")
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
