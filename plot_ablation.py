import matplotlib.pyplot as plt
import numpy as np

lrs = [0.0005, 0.005, 0.05]
wds1 = [0.005, 0.0005, 0.00005, 0.000005, 1e-10]
wds2 = [0.005, 0.001, 0.0008, 0.0005, 0.0001, 0.00008, 0.00005, 0.00001, 0.000001, 1e-10]
wds3 = [0.005, 0.0005, 0.00005, 0.000005, 1e-10]
mr1 = np.array([[-0.04000712, 0.03910514], [-0.11397137, 0.11686115], [-0.19420886, 0.19614412], [-0.22619769, 0.2166732], [-0.31436515, 0.27108598]])
mr2 = np.array([[-0.03202245, 0.03348575], [-0.09578018, 0.09122731], [-0.11175545, 0.10226776], [-0.12981634, 0.117246], [-0.18386897, 0.22222877], [-0.21349238, 0.17650351], [-0.2012432, 0.2007187], [-0.23637924, 0.1923137], [-0.3464441, 0.33186886], [-0.39482772, 0.37963948]])
mr3 = np.array([[-0.045366604, 0.032967832], [-0.1361872, 0.13942702], [-0.2625275, 0.26676098], [-0.3571083, 0.46167797], [-0.46850652, 0.51445776]])
mse1 = [0.051576063, 0.017709702, 0.0074742176, 0.007289214, 0.0073080827]
mse2 = [0.051599935, 0.025854852, 0.022978999, 0.017758733, 0.008119187, 0.0078204945, 0.0075059803, 0.0073075974, 0.007300714, 0.007299102]
mse3 = [0.051583923, 0.018358529, 0.01137333, 0.012289894, 0.009620273]

plt.plot(wds1, mse1, '--o', label="learning rate ="+str(lrs[0]))
plt.plot(wds2, mse2,'--o', label="learning rate ="+str(lrs[1]))
plt.plot(wds3, mse3,'--o', label="learning rate ="+str(lrs[2]))
plt.xlabel("Weight Decay")
plt.ylabel("MSE error")
plt.legend()
plt.title("Reconstruction error")
plt.xscale("log")
plt.yscale("log")
plt.text(1e-10,0.003,"0", ha="center", va="center")
plt.savefig("result/ablation_00005/MSE_loss.png")
plt.show()


plt.plot(wds3, mr3[:,1],'--or', label="Vmax, learning rate ="+str(lrs[2]))
plt.plot(wds2, mr2[:,1],'--og', label="Vmax, learning rate ="+str(lrs[1]))
plt.plot(wds1, mr1[:,1], '--ob', label="Vmax, learning rate ="+str(lrs[0]))
plt.plot(wds1, mr1[:,0], '--ob', label="Vmin, learning rate ="+str(lrs[0]))
plt.plot(wds2, mr2[:,0],'--og', label="Vmin, learning rate ="+str(lrs[1]))
plt.plot(wds3, mr3[:,0],'--or', label="Vmin, learning rate ="+str(lrs[2]))
plt.xlabel("Weight Decay")
plt.ylabel("Mode range")
plt.title("Modes value range")
plt.legend()
plt.xscale("log")
plt.text(1e-10,-0.565,"0", ha="center", va="center")
plt.savefig("result/ablation_00005/Modes_range.png")
plt.show()
