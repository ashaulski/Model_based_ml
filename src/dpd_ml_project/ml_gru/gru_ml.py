import csv
from sympy import false
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.signal import welch
from pathlib import Path
from datetime import datetime
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchinfo import summary


# debug - temporary
from dpd_ml_project.siggen.SigGen import gen_lsig
from dpd_ml_project.channel.awgn import apply_awgn
from dpd_ml_project.channel.pa_model import apply_pa_model
from dpd_ml_project import config


RUN_OUTPUT_DIR = Path("runs") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_NFFT = config.Config.FFT_SIZE
_CP_LEN = config.Config.CP_LEN

# input_size = 20
# def model_info(RNN_type):
#   RNN_type(torch.randn(3, input_size))
#   print(summary(RNN_type, input_size=(1,input_size), device="cpu"))
#   print('\n')
#   for name, parameter in RNN_type.named_parameters():
#     print(name, parameter.shape)


# model_info(nn.GRU(input_size = 20, hidden_size = 50, batch_first=True))

# Network block diagram:
#   x[t] (I/Q samples)
#        |
#        v
#   [GRU: input_size=2 -> hidden_size=50]
#        |
#        v
#   [Linear: hidden_size=50 -> num_outputs=2]
#        |
#        v
#   y_hat[t] (predicted I/Q samples)
#
# Sequence-to-sequence wiring:
#   each input time step feeds the GRU, the GRU output at that step feeds the
#   fully connected layer, and the layer produces the two output values for
#   that same step.
class DpdGru(nn.Module):
    def __init__(self, RNN_type, input_size, hidden_size, num_outputs, num_layers=1, fc1_size=100, fc2_size=64, dropout=0.1):
        super().__init__() #  The super() function is used to give access to methods and properties of a parent class
        self.hidden_size = hidden_size
        self.hidden = None
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True) # batch_first=True means that the input and output tensors are provided as (batch, seq, feature)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, fc1_size),
            nn.ReLU(),
            nn.Linear(fc1_size, fc2_size),
            nn.ReLU(),
            nn.Linear(fc2_size, num_outputs)
        )

    def forward(self, x, h=None):
        # Sequence-to-sequence: return one output sample per input sample.
        rnn_out, h = self.rnn(x, h)
        out = self.head(rnn_out)
        return out, h


# --- orgenize OFDM data into batches -----
#   1. covert arrays of shape (2,N) to two tensors of shape (N,) with complex64 dtype
#   2. Splits the N//64 blocks of 64 samples.
#   3. Converts complex samples into real-valued I/Q pairs.
#   4. Reshapes each 64-sample block into 8 sequences of length 8
# -----------------------------------------
class OfdmBatcher:
    def __init__(self, ofdms, batch_size=64, drop_last=True, enhance_features=False):
        self.drop_last = drop_last
        self.enhance_features = enhance_features
        # convert multiple OFDM symbols into a single array of shape (2, N)
        ofdms = np.concatenate(ofdms, axis=1)
        # convert array of shape (2, N) to two tensors of shape (N,) with complex64 dtype
        pa_in_tens = torch.as_tensor(ofdms[0, :], dtype=torch.complex64)
        pa_out_tens = torch.as_tensor(ofdms[1, :], dtype=torch.complex64)

        self.batch_size = batch_size
        if drop_last:
            self.num_batches = pa_in_tens.shape[0] // batch_size
        else:
            self.num_batches = (pa_in_tens.shape[0] + batch_size - 1) // batch_size

        required_samples = self.num_batches * self.batch_size

        # Keep exactly 38 batches of 64 after skipping first/last 64 samples.
        self.pa_in = pa_in_tens[:required_samples]
        self.pa_out = pa_out_tens[:required_samples]

    # called on "for" loop to produce batches of data 
    def __iter__(self):
        for batch_idx in range(self.num_batches):
            start = batch_idx * self.batch_size
            end = start + self.batch_size
            pa_in_batch = self.pa_in[start:end]
            pa_out_batch = self.pa_out[start:end]

            if self.enhance_features:
                pa_in_ri = torch.stack((pa_in_batch.real, pa_in_batch.imag, pa_in_batch.real**2+pa_in_batch.imag**2,(pa_in_batch.real**2+pa_in_batch.imag**2)**2), dim=-1).view(1,self.batch_size , 4)
                pa_out_ri = torch.stack((pa_out_batch.real, pa_out_batch.imag, pa_out_batch.real**2+pa_out_batch.imag**2,(pa_out_batch.real**2+pa_out_batch.imag**2)**2), dim=-1).view(1, self.batch_size, 4)
            else:
                pa_in_ri = torch.stack((pa_in_batch.real, pa_in_batch.imag), dim=-1).view(1,self.batch_size , 2)
                pa_out_ri = torch.stack((pa_out_batch.real, pa_out_batch.imag), dim=-1).view(1, self.batch_size, 2)

            yield pa_in_ri, pa_out_ri

def get_mse(model, data_loader):
    total_loss, total_batches = 0.0, 0
    h = None

    # Evaluate without gradient tracking and restore mode after metric pass.
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for pa_in, pa_out in data_loader:
            pa_out, pa_in = pa_out.to(device), pa_in.to(device)
            pred, h = model(pa_out, h)
            total_loss += criterion(pred, pa_in).item()  # Only consider the first two features (I/Q) for loss calculation
            total_batches += 1
            h = h.detach()

    if was_training:
        model.train()

    return total_loss / max(1, total_batches)

def train(model, optimizer, num_epochs=10):
    train_mse, valid_mse = [], []
    epochs = []
    # Loop over epochs

    model = model.to(device)
    for epoch in range(num_epochs):
        # Loop over batches
        h = None
        for pa_in, pa_out in train_loader:
            pa_out, pa_in = pa_out.to(device), pa_in.to(device)
            optimizer.zero_grad()
            pred, h = model(pa_out, h)
            loss = criterion(pred, pa_in)
            loss.backward()
            optimizer.step()
            h = h.detach()
        # Save error on each epoch
        train_mse.append(get_mse(model, train_loader))
        valid_mse.append(get_mse(model, valid_loader))
        print("Epoch %d; Loss %f; Train MSE %f; Val MSE %f" % (
              epoch+1, loss, train_mse[-1], valid_mse[-1]))
    # plotting
    plt.title("Training Curve")
    plt.plot(train_mse, label="Train")
    plt.plot(valid_mse, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend(loc='best')

# configuraton parameters for the GRU model and training
enhance_features = False    # use higher order features (power and power squared) in addition to I/Q
dropout = 0.1              # dropout rate for GRU layer
hidden_gru_size = 64       # number of hidden units in the GRU layer
num_gru_layers = 2          # number of GRU layers
num_of_epochs = 20          # number of epochs to train the model
fc1_size = 64              # number of units in the first fully connected layer
fc2_size = 32               # number of units in the second fully connected layer
lrs = [2e-4, 5e-5, 5e-5]    # ADAM learning rates for each stage of training (3 stages)
num_ofdm_sym = 50           # number of OFDM symbols to generate for training, validation, and testing
if enhance_features:
    input_size = 4
    num_outputs = 4
else:
    input_size = 2
    num_outputs = 2


criterion = nn.MSELoss()

 # --- Signal generation --------------------
 # 1. genearte OFDM signal at time domain
 # 2. add AWGN to ogenerated signal
 # 3. apply PA model to the signal
 # 4. store into an array of shape (3, 2, _NFFT+_CP_LEN) 
 #      where the first dimension is train/valid/test, 
 #      the second dimension is PA input/output, 
 #      and the third dimension is the time samples
 # -----------------------------------------
gru_raw_data = np.empty((3*num_ofdm_sym, 2, _NFFT+_CP_LEN), dtype=complex)
for i in range(3*num_ofdm_sym): 
    tx_iq = gen_lsig(bypass=False, repeat_bits_every_call=False, bits_seed=123, signal_rms_dbp=-7)
    # --- AWGN channel ---
    tx_pa_in_iq = apply_awgn(tx_iq, snr_db=70, bypass=False)
    # --- PA / channel model ---
    rf_out = apply_pa_model(tx_pa_in_iq, bypass=False)
    gru_raw_data[i,0,:]= tx_pa_in_iq
    gru_raw_data[i,1,:]= rf_out

train_data = gru_raw_data[: num_ofdm_sym, :, :]
valid_data = gru_raw_data[1*num_ofdm_sym : 2*num_ofdm_sym, :, :]
test_data = gru_raw_data[2*num_ofdm_sym : 3*num_ofdm_sym, :, :]  

# Divide data into batches
train_loader = OfdmBatcher(train_data, batch_size=64, drop_last=True, enhance_features=enhance_features)
valid_loader = OfdmBatcher(valid_data, batch_size=64, drop_last=True, enhance_features=enhance_features)
test_loader = OfdmBatcher(test_data, batch_size=64, drop_last=True, enhance_features=enhance_features)

model = DpdGru(nn.GRU, input_size=input_size, hidden_size=hidden_gru_size, num_outputs=num_outputs, num_layers=num_gru_layers, fc1_size=fc1_size, fc2_size=fc2_size, dropout=dropout).to(device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=lrs[0]
)
# Train and produce training curve and validation curve
train(model, optimizer, num_epochs=num_of_epochs)


# run inference on test data and check the results
pa_in_pred_reshaped = np.array([])
model.eval()
with torch.no_grad():
    h = None
    for pa_in, pa_out in test_loader:
        pa_out, pa_in = pa_out.to(device), pa_in.to(device)
        pa_in_pred, h = model(pa_in, h)
        pa_in_pred_complex = torch.view_as_complex(pa_in_pred[..., :2]).reshape(-1)
        pa_in_pred_reshaped = np.append(pa_in_pred_reshaped, pa_in_pred_complex.cpu().numpy())

rf_out = apply_pa_model(pa_in_pred_reshaped, bypass=False)


# -------------------------------------
# meters and plot section
# -------------------------------------
n_fft = 2048

# plot time domain portion of the signal
plt.figure(figsize=(10, 5))
plt.plot(test_data.real[0,0, 64:_NFFT//2], label='Original Signal', color='blue')
plt.plot(test_data.real[0,1, 64:_NFFT//2], label='PA Output without pre distortion', color='red')
plt.plot(rf_out.real[64:_NFFT//2], label='PA Output with pre distortion', color='green')
plt.title('Magnitude Comparison')
plt.legend()
plt.tight_layout()
plt.savefig(RUN_OUTPUT_DIR / "time_domain_comparison.png", dpi=160)

# plot frequency domain magnitude
test_data_rs  = test_data[:,:,_CP_LEN//2:_NFFT+_CP_LEN//2]
test_data_fd = np.fft.fftshift(np.fft.fft(test_data_rs,axis=2),axes=2)
test_data_magdb_fd = 20*np.log10((np.mean(np.abs(test_data_fd),axis=0)))
rf_out_rs = rf_out.reshape(num_ofdm_sym,-1)[:,_CP_LEN//2:_NFFT+_CP_LEN//2]
rf_out_fd = np.fft.fftshift(np.fft.fft(rf_out_rs,axis=1),axes=1)
rf_out_magdb_fd = 20*np.log10(np.mean(np.abs(rf_out_fd),axis=0))

plt.figure(figsize=(10, 5))
plt.plot(test_data_magdb_fd[0, :], label='Original signal', color='blue')
plt.plot(test_data_magdb_fd[1, :], label='PA Output without pre distortion', color='red')
plt.plot(rf_out_magdb_fd, label='PA Output with pre distortion', color='green')
plt.title('Magnitude Comparison')
plt.legend()
plt.tight_layout()
plt.savefig(RUN_OUTPUT_DIR / "frequency_domain_comparison.png", dpi=160)

# calculate inband MSE comparison
test_data_ib_fd_no_predistorsion = test_data_fd[:,:,_NFFT//2-32:_NFFT//2+32]
mse_ib_no_predistorsion = np.mean(np.abs(test_data_ib_fd_no_predistorsion[:,0,:]-test_data_ib_fd_no_predistorsion[:,1,:])**2,axis=0)
evm_ib_db_no_predistorsion = 10*np.log10(mse_ib_no_predistorsion/np.mean(np.abs(test_data_ib_fd_no_predistorsion[:,0,:])**2,axis=0))

mse_ib_with_predistorsion = np.mean(np.abs(test_data_ib_fd_no_predistorsion[:,0,:]-rf_out_fd[:,_NFFT//2-32:_NFFT//2+32])**2,axis=0)
evm_ib_db_with_predistorsion = 10*np.log10(mse_ib_with_predistorsion/np.mean(np.abs(test_data_ib_fd_no_predistorsion[:,0,:])**2,axis=0))

# calculate time domain MSE comparison
mse_td_no_predistorsion = np.mean(np.abs(test_data_rs[:,0,_CP_LEN//2:-_CP_LEN//2]-test_data_rs[:,1,_CP_LEN//2:-_CP_LEN//2])**2,axis=(0,1))
mse_td_with_predistorsion = np.mean(np.abs(test_data_rs[:,0,_CP_LEN//2:-_CP_LEN//2]-rf_out_rs[:,_CP_LEN//2:-_CP_LEN//2])**2,axis=(0,1))
evm_td_no_predistorsion = 10*np.log10(mse_td_no_predistorsion/np.mean(np.abs(test_data_rs[:,0,_CP_LEN//2:-_CP_LEN//2])**2,axis=(0,1)))
evm_td_with_predistorsion = 10*np.log10(mse_td_with_predistorsion/np.mean(np.abs(test_data_rs[:,0,64:-64])**2,axis=(0,1)))

print("EVM time domain no DPD: ", evm_td_no_predistorsion)
print("EVM time domain with DPD: ", evm_td_with_predistorsion)


plt.figure(figsize=(10, 5))
plt.plot(evm_ib_db_no_predistorsion, label='EVM Inband no DPD', color='blue')
plt.plot(evm_ib_db_with_predistorsion, label='EVM Inband with DPD', color='red')
plt.title('Inband MSE Comparison')
plt.legend()
plt.show()

xx = 1