import torch
import lightning
import pytorch_lightning
import sys




print("######## TORCH ########")

print("Torch version: {}", torch.__version__)
print("Lightning version: {}", lightning.__version__)

print("TORCH-GPU available:{} " , torch.cuda.is_available())

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()

    # Loop through each GPU and print its properties
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  - Total Memory: {props.total_memory / 1e9} GB")
        print(f"  - MultiProcessor Count: {props.multi_processor_count}")
else:
    print("CUDA is not available on this system.")




# Open a file and redirect stdout
with open('dene_output.txt', 'w') as f:
    original_stdout = sys.stdout
    sys.stdout = f  # Redirect stdout to the file

    print("######## TORCH ########")

    print("Torch version: {}", torch.__version__)
    print("Lightning version: {}", lightning.__version__)

    print("TORCH-GPU available:{} " , torch.cuda.is_available())

    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()

        # Loop through each GPU and print its properties
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  - Total Memory: {props.total_memory / 1e9} GB")
            print(f"  - MultiProcessor Count: {props.multi_processor_count}")
    else:
        print("CUDA is not available on this system.")

    print('#########################')

    sys.stdout = original_stdout  # Reset stdout to its original state
