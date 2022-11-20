from torch import nn
import torch
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary
from scipy.stats import ortho_group

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device, "is in use")

# Transforms images to a PyTorch Tensor
tensor_transform = transforms.Compose([transforms.ToTensor()])

# Load the MNIST Dataset
dataset = datasets.MNIST(root="./data",
                         train=True,
                         download=False,
                         transform=tensor_transform)

flatten_data = dataset.data.reshape(dataset.data.shape[0], -1).type(torch.FloatTensor) / 255  # flattened from [60000, 28, 28] to [60000, 28*28]
mean_data = flatten_data.mean(axis=0)
flatten_data = flatten_data - mean_data
print("shape of dataset:", flatten_data.shape)

train_loader = torch.utils.data.DataLoader(
    dataset=flatten_data,  # only train for simplicity
    batch_size=10240,
    shuffle=False)

k = 81
ortho_lambd = 0.01

# Creating a AE class
# 28*28 ==> k ==> 28*28
class AE(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.utils.parametrizations.orthogonal(torch.nn.Linear(flatten_data.shape[1], k, bias=False)))
        self.decoder = torch.nn.Sequential(torch.nn.utils.parametrizations.orthogonal(torch.nn.Linear(k, flatten_data.shape[1], bias=False)))
        # self.encoder = torch.nn.Sequential(torch.nn.Linear(flatten_data.shape[1], k, bias=False))
        # self.decoder = torch.nn.Sequential(torch.nn.Linear(k, flatten_data.shape[1], bias=False))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# Model Initialization
model = AE()
# model.decoder[0].weight.requires_grad = False # freeze the decoder
# model.encoder[0].weight = nn.Parameter(torch.tensor(ortho_group.rvs(dim=flatten_data.shape[0])[:,:k].T).float())
# model.decoder[0].weight = nn.Parameter(torch.tensor(ortho_group.rvs(dim=flatten_data.shape[0])[:,:k]).float())
# model.decoder[0].weight = nn.Parameter(model.encoder[0].weight.data.T)
# torch.nn.init.xavier_uniform_(model.encoder[0].weight)

summary(model, input_size=(60000, 784))

print("encoder weight shape: ", model.encoder[0].weight.shape)
print("decoder weight shape: ", model.decoder[0].weight.shape)
model = model.to(device)

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()

# Using an Adam Optimizer with lr = 0.1
# optimizer = torch.optim.SGD(model.parameters(), lr=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay = 1e-5)
# training
epochs = 100
losses = []
for epoch in range(epochs):
    constructs = torch.tensor([]).to(device)
    reconstructs = torch.tensor([]).to(device)
    for step, image in enumerate(train_loader):
        image = image.to(device)
        constructed, reconstructed = model(image)
        # loss_ortho_de = torch.square(torch.dist(model.decoder[0].weight.T @ model.decoder[0].weight, torch.eye(k).to(device)))
        # loss_ortho_en = torch.square(torch.dist(model.encoder[0].weight @ model.encoder[0].weight.T, torch.eye(k).to(device)))
        # loss = loss_function(reconstructed, image) + ortho_lambd*(loss_ortho_de)
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
modes = np.array(model.decoder[0].weight.data.cpu())
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
end = 32

for i in range(start, end):
    # Reshape the array for plotting
    i_images.append(flatten_data.T[:, i].cpu().reshape(-1, 28, 28)+mean_data.reshape(1, 28, 28))
    o_images.append(reconstructs.T[:, i].cpu().reshape(-1, 28, 28)+mean_data.reshape(1, 28, 28))

for i in range(k):
    mode_images.append(torch.tensor(modes[:,i]).reshape(-1, 28, 28))

i_im = make_grid(i_images)
i_im = i_im.numpy().transpose((1, 2, 0))
o_im = make_grid(o_images)
o_im = o_im.numpy().transpose((1, 2, 0))
m_im = make_grid(mode_images,9)
m_im = m_im.numpy().transpose((1, 2, 0))

plt.figure()
plt.imshow(i_im[:,:,0],cmap="turbo",vmin=-0.2,vmax=0.2)
plt.colorbar()
plt.axis("off")
plt.show()

plt.figure()
plt.imshow(o_im[:,:,0],cmap="turbo",vmin=-0.2,vmax=0.2)
plt.colorbar()
plt.axis("off")
plt.show()

plt.figure()
plt.imshow((i_im - o_im)[:,:,0],cmap="turbo",vmin=-0.2,vmax=0.2)
plt.colorbar()
plt.axis("off")
plt.show()

plt.figure()
plt.imshow(m_im[:,:,0],cmap="turbo",vmin=-0.2,vmax=0.2)
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
for sig in range(S.shape[0]):
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
plt.title("Energy division for each mode")
plt.show()

i_images = []
o_images = []
m_images = []
start = 0
end = k
for i in range(start, end):
    # Reshape the array for plotting
    i_images.append(flatten_data_SVD[:, i].cpu().reshape(-1, 28, 28))
    o_images.append(reconstructed_SVD[:, i].cpu().reshape(-1, 28, 28))

for i in range(k):
    m_images.append(U[:,i].cpu().reshape(-1, 28, 28))

i_im = make_grid(i_images,9)
i_im = i_im.numpy().transpose((1, 2, 0))
o_im = make_grid(o_images,9)
o_im = o_im.numpy().transpose((1, 2, 0))
m_im = make_grid(m_images,9)
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
plt.imshow(m_im[:,:,0],cmap="turbo",vmin=-0.2,vmax=0.2)
plt.axis("off")
plt.colorbar()
plt.show()