from enum import auto
from torch import nn
import torch
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary
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
# flatten_data = torch.tensor(flatten_data.T[:,250:])
flatten_data = torch.tensor(flatten_data[250:,:])
flatten_data = flatten_data.T

print("shape of dataset:", flatten_data.shape)  # flattened from [392, 490, 101] to [192080, 101]

train_loader = torch.utils.data.DataLoader(
    dataset=flatten_data,  # only train for simplicity
    batch_size=1024,
    shuffle=False)

k = 8

# Creating a AE class
# 28*28 ==> k ==> 28*28
class AE(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # self.encoder = torch.nn.Sequential(torch.nn.utils.parametrizations.orthogonal(torch.nn.Linear(flatten_data.shape[1], k, bias=False)))
        # self.decoder = torch.nn.Sequential(torch.nn.utils.parametrizations.orthogonal(torch.nn.Linear(k, flatten_data.shape[1], bias=False)))
        self.encoder = torch.nn.Sequential(torch.nn.Linear(flatten_data.shape[1], k, bias=False))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(k, flatten_data.shape[1], bias=False))
        self.SVT = torch.zeros((k,k))

    def forward(self, x):
        encoded = self.encoder(x)
        svd_encoded,S,V = torch.svd(encoded)
        decoded = self.decoder(svd_encoded)
        self.SVT = V@torch.diag(S)
        return svd_encoded, decoded


class Svd_W_de(nn.Module):
    def __init__(self, SVT):
        super().__init__()
        self.SVT = SVT
    def forward(self, X):
        return X.to(device)@self.SVT.to(device)


# Model Initialization
model = AE()
torch.nn.utils.parametrizations.parametrize.register_parametrization(model.decoder[0], "weight", Svd_W_de(model.SVT))
# model.decoder[0].weight.requires_grad = False # freeze the decoder
# model.encoder[0].weight = nn.Parameter(Uk.T)
# model.decoder[0].weight = nn.Parameter(model.encoder[0].weight.data.T)
# torch.nn.init.xavier_uniform_(model.encoder[0].weight)

model = model.to(device)
summary(model, input_size=(192080, 101))

print("encoder weight shape: ", model.encoder[0].weight.shape)
print("decoder weight shape: ", model.decoder[0].weight.shape)

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()

# Using an Adam Optimizer with lr = 0.1
# optimizer = torch.optim.SGD(model.parameters(), lr=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# training
epochs = 2
losses = []
for epoch in range(epochs):
    constructs = torch.tensor([]).to(device)
    reconstructs = torch.tensor([]).to(device)
    for step, image in enumerate(train_loader):
        image = image.to(device)
        constructed, reconstructed = model(image)
        loss = loss_function(reconstructed, image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # model.decoder[0].weight = nn.Parameter(model.encoder[0].weight.data.T)

        # Storing the losses in a list for plotting
        losses.append(loss.cpu().detach().numpy())
        constructs = torch.cat((constructs, constructed), 0)
        reconstructs = torch.cat((reconstructs, reconstructed), 0)

losses = np.array(losses)
# modes = np.array(model.encoder[0].weight.data.cpu().T)
# modes = np.array(model.decoder[0].weight.data.cpu())
modes = np.array(constructs.detach().cpu())

print("mode shape: ", modes.shape)
print("laten space shape: ", constructs.cpu().shape)
print("reconstructed matrix shape: ", reconstructs.cpu().shape)
print("MSE loss", torch.square(reconstructs.cpu() - flatten_data.cpu()).mean().detach().numpy())
print("Max loss", (reconstructs.cpu() - flatten_data.cpu()).max().detach().numpy())
print("Min loss", (reconstructs.cpu() - flatten_data.cpu()).min().detach().numpy())

# vis
print(flatten_data.shape)
print(reconstructs.shape)
print(modes.shape)
# Visualization
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.plot(losses)
# plt.plot(losses[-500:])
print("average loss of last 50 steps: ", losses[-50:].mean())
print("minimum loss of last 50 steps: ", np.min(losses))
plt.show()

i_images = []
o_images = []
mode_images = []

start = 0
end = 4

for i in range(start, end):
    # Reshape the array for plotting
    i_images.append(flatten_data[:, i].cpu().detach().reshape(-1, 392, 490)+mean_data.reshape(1, 392, 490))
    o_images.append(reconstructs[:, i].cpu().detach().reshape(-1, 392, 490)+mean_data.reshape(1, 392, 490))

for i in range(k):
    mode_images.append(torch.tensor(modes[:,i]).reshape(-1, 392, 490))

i_im = make_grid(i_images,2)
i_im = i_im.numpy().transpose((1, 2, 0))
o_im = make_grid(o_images,2)
o_im = o_im.numpy().transpose((1, 2, 0))
m_im = make_grid(mode_images,2)
m_im = m_im.numpy().transpose((1, 2, 0))

plt.figure()
plt.imshow(i_im[:,:,0],cmap="turbo")
plt.colorbar()
plt.axis("off")
plt.show()

plt.figure()
plt.imshow(o_im[:,:,0],cmap="turbo")
plt.colorbar()
plt.axis("off")
plt.show()

plt.figure()
plt.imshow((i_im - o_im)[:,:,0],cmap="turbo",vmin=-0.04,vmax=0.04)
plt.colorbar()
plt.axis("off")
plt.show()

plt.figure()
plt.imshow(m_im[:,:,0],cmap="turbo",vmin=-0.04,vmax=0.04)
plt.legend("True")
plt.colorbar()
plt.axis("off")
plt.show()


# SVD
print("----- SVD -----")
flatten_data_SVD = torch.tensor(modes)
U, S, V = torch.svd(flatten_data_SVD)
print("shape of u:", U.shape)
print("shape of s:", S.shape)
print("shape of v:", V.shape)

# recover the data
print("----- recovery -----")
reconstructed_SVD = torch.matmul(U, torch.matmul(torch.diag(S), V.T))
print("shape of reconstructed:", reconstructed_SVD.shape)
print("MSE loss", torch.square(reconstructed_SVD - flatten_data_SVD).mean().cpu().numpy())

# energy spectral
print("----- energy spectral -----")

KE = []
KE_S = np.square(S.cpu())/ np.square(S.cpu()).sum()
for sig in range(1, S.shape[0]+1):
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
plt.xlabel("k")
plt.ylabel("E")
plt.yscale("log")
plt.title("Energy division for each mode")
plt.show()

i_images = []
o_images = []
m_images = []
start = 0
end = k
for i in range(start, end):
    # Reshape the array for plotting
    i_images.append(flatten_data_SVD[:, i].cpu().reshape(-1, 392, 490))
    o_images.append(reconstructed_SVD[:, i].cpu().reshape(-1, 392, 490))

for i in range(k):
    m_images.append(U[:,i].cpu().reshape(-1, 392, 490))

i_im = make_grid(i_images,2)
i_im = i_im.numpy().transpose((1, 2, 0))
o_im = make_grid(o_images,2)
o_im = o_im.numpy().transpose((1, 2, 0))
m_im = make_grid(m_images,2)
m_im = m_im.numpy().transpose((1, 2, 0))


plt.figure()
plt.imshow(m_im[:,:,0],cmap="turbo")
plt.axis("off")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(m_im[:,:,0],cmap="turbo",vmin=-0.04,vmax=0.04)
plt.axis("off")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(i_im[:,:,0],cmap="turbo",vmin=-0.04,vmax=0.04)
plt.axis("off")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(i_im[:,:,0],cmap="turbo",vmin=-0.4,vmax=0.4)
plt.axis("off")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(o_im[:,:,0],cmap="turbo",vmin=-0.04,vmax=0.04)
plt.axis("off")
plt.colorbar()
plt.show()

