import torch

def main():
    print("Hello World!")
    print("Check PyTorch version and GPU availability:")
    print(torch.__version__)
    print(torch.cuda.is_available())
    print("bye world")

if __name__ == "__main__":
    main()