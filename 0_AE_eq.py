from enum import auto
from torch import nn
import torch
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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

# Creating a AE class
# 28*28 ==> k ==> 28*28
class AE(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(flatten_data.shape[1], k, bias=False))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(k, flatten_data.shape[1], bias=False))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# Model Initialization
model = AE()
# model.decoder[0].weight.requires_grad = False # freeze the decoder
# model.encoder[0].weight = nn.Parameter(Uk.T)
# model.decoder[0].weight = nn.Parameter(model.encoder[0].weight.data.T)
# torch.nn.init.xavier_uniform_(model.encoder[0].weight)

summary(model, input_size=(60000, 784))

print("encoder weight shape: ", model.encoder[0].weight.shape)
print("decoder weight shape: ", model.decoder[0].weight.shape)
model = model.to(device)

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()
learning_rate = 0.005
# weight_decay = 0.005 # first 6 modes range [-0.03202245, 0.03348575] MSE loss 0.051599935
# weight_decay = 0.001 # first 29 modes range [-0.09578018, 0.09122731] MSE loss 0.025854852
# weight_decay = 0.0008 # first 35 modes range [-0.11175545, 0.10226776] MSE loss 0.022978999
# weight_decay = 0.0005 # first 48 modes range [-0.12981634, 0.117246] MSE loss 0.017758733
# weight_decay = 0.0001 # first all modes range [-0.18386897, 0.22222877] MSE loss 0.008119187
weight_decay = 0.00008 # first all modes range [-0.21349238, 0.17650351] MSE loss 0.0078204945
# weight_decay = 0.00005 # different at end modes range [-0.2012432, 0.2007187] MSE loss 0.0075059803
# weight_decay = 0.00001 # different modes range [-0.23637924, 0.1923137] MSE loss 0.0073075974
# weight_decay = 0.000001 # different modes range [-0.3464441, 0.33186886] MSE loss 0.007300714
# weight_decay = 0 # different modes range [-0.39482772, 0.37963948] MSE loss 0.007299102
# learning_rate = 0.05
# weight_decay = 0.05 # 0 modes range [-2.1133948e-08, 2.1063197e-08] MSE loss 0.06725132
# weight_decay = 0.005 # first 6 modes range [-0.045366604, 0.032967832] MSE loss 0.051583923
# weight_decay = 0.0005 # noisy first 60 modes range [-0.1361872, 0.13942702] MSE loss 0.018358529
# weight_decay = 0.00005 # noisy modes range [-0.2625275, 0.26676098] MSE loss 0.01137333
# weight_decay = 0.000005 # noisy modes range [-0.3571083, 0.46167797] MSE loss 0.012289894
# weight_decay = 0 # noisy modes range [-0.46850652, 0.51445776] MSE loss 0.009620273
# learning_rate = 0.0005
# weight_decay = 0.005 # first 9 modes range [-0.04000712, 0.03910514] MSE loss 0.051576063
# weight_decay = 0.0005 # first 60 modes range [-0.11397137, 0.11686115] MSE loss 0.017709702
# weight_decay = 0.00005 # all modes range [-0.19420886, 0.19614412] MSE loss 0.0074742176
# weight_decay = 0.000005 # different modes range [-0.22619769, 0.2166732] MSE loss 0.007289214
# weight_decay = 0 # different modes range [-0.31436515, 0.27108598] MSE loss 0.0073080827
# SVD loss MSE loss 0.0072650537

# optimizer = torch.optim.SGD(model.parameters(), lr=2)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# training
epochs = 100
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
modes = np.array(model.decoder[0].weight.data.cpu())
print("mode shape: ", modes.shape)
print("laten space shape: ", constructs.cpu().shape)
print("reconstructed matrix shape: ", reconstructs.cpu().shape)
print("MSE loss", torch.square(reconstructs.cpu() - flatten_data.cpu()).mean().detach().numpy())
print("loss range", [(reconstructs.cpu() - flatten_data.cpu()).max(),(reconstructs.cpu() - flatten_data.cpu()).min()])
print("modes range", [modes.min(),modes.max()])

# Visualization
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.plot(losses)
# plt.plot(losses[-500:])
# print("average loss of last 50 steps: ", losses[-50:].mean())
# print("minimum loss of last 50 steps: ", np.min(losses))
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

# plt.figure()
# plt.imshow(i_im[:,:,0],cmap="turbo",vmin=-0.2,vmax=0.2)
# plt.colorbar()
# plt.axis("off")
# plt.show()

# plt.figure()
# plt.imshow(o_im[:,:,0],cmap="turbo",vmin=-0.2,vmax=0.2)
# plt.colorbar()
# plt.axis("off")
# plt.show()

# plt.figure()
# plt.imshow((i_im - o_im)[:,:,0],cmap="turbo",vmin=-0.2,vmax=0.2)
# plt.colorbar()
# plt.axis("off")
# plt.show()

# plt.figure()
# plt.imshow(m_im[:,:,0],cmap="turbo",vmin=-0.2,vmax=0.2)
# plt.legend("True")
# plt.colorbar()
# plt.title("modes extracted by AE\nλ = "+str(weight_decay)+", lr = "+str(learning_rate))
# plt.axis("off")
# plt.show()


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
plt.title("Energy spectral\nλ = "+str(weight_decay)+", lr = "+str(learning_rate))
# plt.savefig("./result/ablation_00005/"+"Energy spectral wd "+str(weight_decay)+", lr "+str(learning_rate)+".png")
plt.show()

plt.bar(list(range(1,KE_S[:k].shape[0]+1)), KE_S[:k])
plt.plot(list(range(1,KE_S[:k].shape[0]+1)), KE_S[:k],'--r')
plt.grid(True)
plt.xlabel("k")
plt.ylabel("E")
plt.title("Energy division for each mode\nλ = "+str(weight_decay)+", lr = "+str(learning_rate))
# plt.savefig("./result/ablation_00005/"+"Energy division wd "+str(weight_decay)+", lr "+str(learning_rate)+".png")
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
plt.imshow(i_im[:,:,0],cmap="turbo")
plt.axis("off")
plt.colorbar()
plt.title("modes extracted by AE\nλ = "+str(weight_decay)+", lr = "+str(learning_rate))
# plt.savefig("./result/ablation_00005/"+"AE modes wd "+str(weight_decay)+", lr "+str(learning_rate)+".png")
plt.show()

# plt.figure()
# plt.imshow(o_im[:,:,0],cmap="turbo",vmin=-0.2,vmax=0.2)
# plt.axis("off")
# plt.colorbar()
# plt.show()

plt.figure()
plt.imshow(m_im[:,:,0],cmap="turbo",vmin=-0.2,vmax=0.2)
# plt.title("orthogonalized modes\nλ = "+str(weight_decay)+", lr = "+str(learning_rate))
plt.axis("off")
plt.colorbar()
# plt.savefig("./result/ablation_00005/"+"orthogonalized modes wd "+str(weight_decay)+", lr "+str(learning_rate)+".png")
plt.show()
