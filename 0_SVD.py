import torch
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = "cpu"
print(device, " is in use")

# Load the MNIST Dataset
dataset = datasets.MNIST(root="./data",
                         train=True,
                         download=False,
                         transform=transforms.ToTensor())

# Flatten from [60000, 28, 28] to [28*28, 60000], move to gpu
flatten_data = dataset.data.reshape(dataset.data.shape[0], -1).type(torch.FloatTensor) / 255  # flattened from [60000, 28, 28] to [60000, 28*28]
mean_data = flatten_data.mean(axis=0)
flatten_data = flatten_data - mean_data
flatten_data = flatten_data.T.to(device)
print("shape of dataset:", flatten_data.shape)

# SVD
print("-----SVD result -----")
U, S, V = torch.svd(flatten_data)
print("shape of u:", U.shape)
print("shape of s:", S.shape)
print("shape of v:", V.shape)

# truncation from first k of SVD
print("----- truncation -----")
k = 81
Uk = U[:, :k]
Sk = torch.diag(S[:k])
Vk = V[:, :k]
print("shape of uk:", Uk.shape)
print("range of uk:", [Uk.min(),Uk.max()]) # range of uk: [-0.2019, 0.2030]
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
    if sig == k:
        print("the first", k, "modes (in use) capture ", KE_sig.numpy(),
              " of the energy")

# Visualization
plt.plot(KE)
plt.grid(True)
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
plt.title("Energy division for each mode")
plt.show()

i_images = []
o_images = []
m_images = []
start = 0
end = 32
for i in range(start, end):
    # Reshape the array for plotting
    i_images.append(flatten_data[:, i].cpu().reshape(-1, 28, 28)+mean_data.reshape(1, 28, 28))
    o_images.append(reconstructed[:, i].cpu().reshape(-1, 28, 28)+mean_data.reshape(1, 28, 28))

for i in range(k):
    m_images.append(Uk[:, i].cpu().reshape(-1, 28, 28))

i_im = utils.make_grid(i_images)
i_im = i_im.numpy().transpose((1, 2, 0))
o_im = utils.make_grid(o_images)
o_im = o_im.numpy().transpose((1, 2, 0))
m_im = utils.make_grid(m_images,9)
m_im = m_im.numpy().transpose((1, 2, 0))

plt.figure()
plt.imshow(i_im[:,:,0],cmap="turbo",vmin=-0.2,vmax=0.2)
plt.axis("off")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(o_im[:,:,0],cmap="turbo",vmin=-0.2,vmax=0.2)
plt.axis("off")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow((i_im -o_im)[:,:,0],cmap="turbo",vmin=-0.2,vmax=0.2)
plt.axis("off")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(m_im[:,:,0],cmap="turbo",vmin=-0.2,vmax=0.2)
plt.colorbar()
plt.axis("off")
plt.show()
