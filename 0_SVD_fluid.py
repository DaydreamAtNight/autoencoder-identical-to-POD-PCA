from enum import auto
import torch
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from cylinder_dataset import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device, " is in use")

# Load the fluid Dataset
flatten_data = load_img_as_matrix("./data/cylinder")
# flatten_data = flatten_data.T - flatten_data.T.mean(axis=0)
mean_data = flatten_data[250:,:].mean(axis=0)
flatten_data = flatten_data - mean_data
# flatten_data -= flatten_data.mean(axis=0)
flatten_data = torch.tensor(flatten_data.T[:,250:])
print("shape of dataset:", flatten_data.shape)  # flattened from [392, 490, 351] to [192080, 351]

flatten_data = flatten_data.to(device)
# SVD
print("-----SVD result -----")
U, S, V = torch.svd(flatten_data)
print("shape of u:", U.shape)
print("shape of s:", S.shape)
print("shape of v:", V.shape)

# truncation from first k of SVD
print("----- truncation -----")
k = 8
Uk = U[:, :k]
# Sk = np.diag(S[:k])
Sk = torch.diag(S[:k])
Vk = V[:, :k]
print("shape of uk:", Uk.shape)
print("shape of sk:", Sk.shape)
print("shape of vk:", Vk.shape)

# recover the data
print("----- recovery -----")
reconstructed = torch.matmul(Uk, torch.matmul(Sk, torch.t(Vk)))
print("shape of reconstructed:", reconstructed.shape)
print("MSE loss", torch.square(reconstructed - flatten_data).mean().cpu().numpy())
print("Max loss", (reconstructed - flatten_data).max().cpu().numpy())
print("Min loss", (reconstructed - flatten_data).min().cpu().numpy())

# energy spectral
print("----- energy spectral -----")

KE = []
KE_S = np.square(S.cpu())/ np.square(S.cpu()).sum()
for sig in range(1, S.shape[0]):
    KE_sig = np.square(S[:sig].cpu()).sum() / np.square(S.cpu()).sum()
    KE.append(KE_sig)
    if 1e-4 < KE_sig - 0.9 < 1e-3:
        print("the first", sig, "modes capture 90% of the energy")
    if 1e-5 < KE_sig - 0.99 < 1e-4:
        print("the first", sig, "modes capture 99% of the energy")
    
    print("the first", sig+1, "modes (in use) capture ", KE_sig.numpy(),
              " of the energy")

# Visualization
# plt.plot(S)
plt.grid(True)
plt.plot(KE)
plt.xlabel("k")
plt.ylabel("E")
plt.title("Energy spectral")
plt.show()

plt.bar(list(range(1,KE_S[:k].shape[0]+1)), KE_S[:k])
plt.plot(list(range(1,KE_S[:k].shape[0]+1)), KE_S[:k],'--r')
plt.grid(True)
# plt.yscale('log')
plt.xlabel("k")
plt.ylabel("E")
plt.yscale("log")
plt.title("Energy division for each mode")
plt.show()

i_images = []
o_images = []
mode_images=[]

start = 0
end = 4

print(mean_data.shape)
print(flatten_data[:, 0].cpu().shape)


for i in range(start, end):
    # Reshape the array for plotting
    i_images.append(flatten_data[:, i].cpu().numpy().reshape(-1, 392, 490) + mean_data.reshape(-1, 392, 490))
    o_images.append(reconstructed[:, i].cpu().numpy().reshape(-1, 392, 490) + mean_data.reshape(-1, 392, 490))

for i in range(k):
    mode_images.append(np.pad(Uk[:, i].cpu().reshape(-1, 392, 490),((0,0),(1,1),(1,1)),'constant', constant_values=(0.01,0.01)))

i_im = make_grid(torch.tensor(i_images),2)
i_im = i_im.numpy().transpose((1, 2, 0))
o_im = make_grid(torch.tensor(o_images),2)
o_im = o_im.numpy().transpose((1, 2, 0))
m_im = make_grid(torch.tensor(mode_images),4)
m_im = m_im.numpy().transpose((1, 2, 0))

plt.figure()
plt.imshow(i_im[:,:,0], cmap='turbo')
plt.axis("off")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(o_im[:,:,0], cmap='turbo')
plt.axis("off")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow((i_im-o_im)[:,:,0], cmap='turbo',vmin=-.04,vmax=.04)
plt.axis("off")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(m_im[:,:,0], cmap ='turbo',vmin=-.04,vmax=.04)
plt.axis("off")
plt.colorbar()
plt.show()