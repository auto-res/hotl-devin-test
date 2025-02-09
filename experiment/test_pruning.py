from src.pruning.llama import prune_llama_model
from tests.new_pruning_method import test_pruning_method

def main():
    # Test pruning implementation
    print("Testing pruning implementation...")
    result = test_pruning_method(prune_llama_model)
    assert result, 'Pruning test failed'
    print('Pruning test passed successfully!')

if __name__ == "__main__":
    main()
