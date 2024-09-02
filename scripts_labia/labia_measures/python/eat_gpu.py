import torch
import time

# Vérifier la disponibilité du GPU
if not torch.cuda.is_available():
    print("No GPU available, use cpu instead")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("Using GPU")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# Définir la taille de la matrice
matrix_size = 10240

# Créer des matrices aléatoires sur le GPU
A = torch.rand(matrix_size, matrix_size, device='cuda')
B = torch.rand(matrix_size, matrix_size, device='cuda')

# Effectuer une multiplication matricielle répétée pendant environ 1 minute
start_time = time.time()
while time.time() - start_time < 7200:
    C = torch.matmul(A, B)
    print(".")
    print("time spent : ", time.time() - start_time)

print("end")
