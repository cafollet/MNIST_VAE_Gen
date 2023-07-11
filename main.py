import argparse
import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
import threading
import time
import sys
from PIL import Image
from data import Data
from nn_gen import VariationalAutoencoder, Net


def animate() -> None:
    """
    Animated loading screen for test generation
    """
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rGenerating Tests ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rDone!             \n\n')
    sys.stdout.flush()


def train_epoch(model, data, optimizer, loss, alpha, gen_label, ind_1, ind):
    """
    Takes the nn model, the input data, optimizer, loss function, regularizer, gen_label (should be one of "gen" or
    "label"), and ind_1 and ind, the first and last indices of the batch. If gen_label is "gen", the function trains
    the VAE, whereas if the gen_label is "label", the function trains the fcnn designated to label the mnist images.
    """
    # Start training
    model.train()
    if gen_label == 'gen':

        # Generate
        gen_in_data = model(data)

        # Evaluate loss, and add kl to regularize (with regularization constant hyperparameter)
        gen_loss = loss(torch.sigmoid(gen_in_data), torch.sigmoid(data)) + (alpha * model.encoder.kl)

        # Backprop
        optimizer.zero_grad()
        gen_loss.backward()
        optimizer.step()

        training_loss = gen_loss.item()

    elif gen_label == 'label':
        # Backprop label network
        label_loss, accuracy, value = model.backprop(data, loss, optimizer, ind_1, ind)
        training_loss = label_loss
        gen_in_data = accuracy

    return training_loss, gen_in_data


if __name__ == '__main__':
    plt.close('all')  # Makes sure all plots are closed from any other previous pyplot program

    # Parsing command line arguments
    parser = argparse.ArgumentParser(description='MNIST Even Number VAE Learning and Generation Program')
    parser.add_argument('-param', help='parameter file name', default='param/param.json')
    parser.add_argument('-o', help='directory to store test samples', default='result_dir')
    parser.add_argument('-n', help='number of test samples to generate', default='100')
    args = parser.parse_args()

    # Open json parameter file
    with open(args.param) as paramfile:
        param = json.load(paramfile)
    if not os.path.isdir(args.o):
        os.makedirs(args.o)

    torch.manual_seed(0)

    n = int(args.n)

    # Defining hyperparameters
    lr = param['exec']['learning_rate']
    label_lr = param['exec']['label_learning_rate']
    d = param['exec']['latent_dimensions']
    n_hidden = param['data']['n_hidden_vae']
    n_hidden_label = param['data']['n_hidden_label']
    num_epochs = param['exec']['num_epochs']
    batch_size = param['exec']['batch_size']
    kl_reg_constant = param['exec']['regularizer_constant']

    # Defining neural network components
    data = Data()
    x_data = data.x_train
    dataset_size = len(x_data)
    in_size = len(x_data[0])
    model = VariationalAutoencoder(latent_dims=d, hidden=n_hidden)
    label_model = Net(in_size, n_hidden_label)
    loss = torch.nn.BCELoss()
    label_loss = torch.nn.NLLLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    label_optim = torch.optim.SGD(label_model.parameters(), lr=label_lr)

    # Reshape the data set to fit the accepted shape for the convolutional layer
    batch_conv_set = torch.from_numpy(np.asarray([np.asarray([x.reshape((28, 28))])for x in x_data]))
    num_batches = dataset_size/batch_size
    train_losses = np.empty(0)
    obj_vals = []
    label_losses = np.empty(0)

    for epoch in range(num_epochs):
        train_loss_sum = 0
        label_val_sum = 0
        label_acc_sum = 0
        # iterate losses of the network, and backprop the network in batches
        for i, batch in enumerate(range(0, dataset_size, batch_size)):
            batch_end = batch + batch_size
            batch_conv = batch_conv_set[batch:batch_end]
            train_loss, train_val = train_epoch(model, batch_conv, optim, loss, kl_reg_constant, 'gen', batch, batch_end)
            train_loss_sum += train_loss  # for averages
            label_val, label_acc = train_epoch(label_model, data, label_optim, label_loss, kl_reg_constant, 'label', batch, batch_end)
            label_val_sum += label_val  # for averages
            label_acc_sum += label_acc  # for averages
            obj_vals.append(train_val)
        train_losses = np.append(train_losses, train_loss)
        label_losses = np.append(label_losses, label_val)
        print('\n Epoch {}/{} ({:.0f}% Trained) \t Generator Training Loss: {:.3f} \t Label Training Loss: {:.3f}'\
              .format(epoch + 1, num_epochs, 100 * ((epoch + 1) / num_epochs), train_loss_sum / num_batches, label_val)
              + '\t Label Training Accuracy: {:.0f}'.format(100 * (label_acc_sum / num_batches)))

    # Done training, start testing/generating, so change model mode
    model.eval()
    print("\n")
    done = False
    t = threading.Thread(target=animate)
    t.start()

    with torch.no_grad():
        # reconstructing images from the dataset n times
        for iter in range(0, n):
            rand_index = np.random.randint(0, dataset_size)
            pre_enc_label = label_model.test(torch.reshape(batch_conv_set[rand_index], (1, 784)))
            pre_enc_ac_label = data.label_train[rand_index]
            rand_input = torch.reshape(batch_conv_set[rand_index], (1, 1, 28, 28))
            img_encode = model.encoder(rand_input)
            img_decode = model.decoder(img_encode)
            img_decode_flat = torch.reshape(img_decode[0, 0], (1, 784))
            rand_input_flat = torch.reshape(rand_input[0, 0], (1, 784))
            npinp = rand_input.numpy()  # Input image for imshow function
            npinp_label, npinp_prob = label_model.test(rand_input_flat)  # label and probability of that label(softmax)
            npimg = img_decode.numpy()
            npimg_label, npimg_prob = label_model.test(img_decode_flat)
            lat_noise = torch.randn(1, d)
            dec_noise = model.decoder(lat_noise)
            dec_noise_flat = torch.reshape(dec_noise[0, 0], (1, 784))
            dec_label, dec_prob = label_model.test(dec_noise_flat)
            dec_noise = dec_noise.numpy()
            plt.imshow(dec_noise[0, 0], cmap='gray')
            plt.title(f"Generated {dec_label} Image")
            plt.savefig(f'{args.o}/MNIST Test-Generated {iter+1}.pdf')
            plt.close('all')
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(npinp[0, 0], cmap='gray')
            ax2.imshow(npimg[0, 0], cmap='gray')
            plt.title(f"Encoded {npimg_label} Image (Pre-encoded: Guess: {pre_enc_label}, Actual: {pre_enc_ac_label})",
                      loc='left')
            plt.savefig(f'{args.o}/MNIST Encode-Decode {iter+1}.pdf')
            plt.close('all')

    fig, ax = plt.subplots(1, 1)
    plt.plot(train_losses, label="VAE Loss")
    plt.plot(label_losses, label="Label Loss")
    fig.suptitle('Loss Vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    fig.savefig(f'{args.o}/loss.pdf')
    time.sleep(1)
    done = True

    # Prompt user about number in command line
    with torch.no_grad():
        num = ""
        while num != "E" and num != "e":
            time.sleep(1)
            sys.stdout.flush()
            num = input("\n Generate a number from 0-9 (Type 'E' to exit): ")  # Prompt
            if num == "E" or num == "e":
                print("\n Program Exited")
            elif not (num == "0" or num == "1" or num == "2" or num == "3" or num == "4" or num == "5" or num == "6"
                      or num == "7" or num == "8" or num == "9"):
                print("Please print a number from 0-9")
            else:
                dec_label = ""
                prob = -2
                while dec_label != num or prob < -2:
                    lat_noise = torch.randn(1, d)
                    dec_noise = model.decoder(lat_noise)
                    dec_noise_flat = torch.reshape(dec_noise[0, 0], (1, 784))
                    label, prob = label_model.test(dec_noise_flat)
                    dec_label = f"{label}"
                    # if dec_label == num:
                    #     print(prob)
                    dec_noise = dec_noise.numpy()
                fig, ax = plt.subplots(1, 1)
                ax.imshow(dec_noise[0, 0], cmap='gray')
                ax.set_title(f"Generated {dec_label}")
                if not os.path.exists("result_dir/generated_numbers"):
                    os.makedirs("result_dir/generated_numbers")
                plt.savefig(f"result_dir/generated_numbers/{dec_label}.png")
                im = Image.open(f"result_dir/generated_numbers/{dec_label}.png")
                im.show()
                plt.close('all')
