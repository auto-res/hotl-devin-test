from transformers import AutoModelForCausalLM

def inspect_model():
    model = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM-135M')
    
    print('\nModel config:')
    print(model.config)
    
    print('\nAttention module structure:')
    for name, module in model.named_modules():
        if 'attention' in name.lower():
            print(f'\n{name}:')
            for param_name, param in module.named_parameters():
                print(f'{param_name}: {param.shape}')

if __name__ == '__main__':
    inspect_model()
