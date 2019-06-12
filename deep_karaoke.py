import torch
from torch.utils.data import Dataset
import json
import numpy as np
from torchvision import transforms
from time import time
from fire import Fire
from scipy.io import wavfile
import spectrum_helper

TRAIN_PORTION = 0.8

class DeepKaraokeDataset(Dataset):
    def __init__(self, dataset_file, transform=None):
        self.data = json.load(open(dataset_file))
        self.metadata = self.data["song_samples"]
        self.sample_shape = self.data["sample_shape"]
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        input_path = self.metadata[idx]['input']
        input_data = np.load(input_path+".npy")
        output_path = self.metadata[idx]['output']
        output_data = np.load(output_path+".npy")

        sample = {'input': input_data, 'output': output_data}

        if self.transform:
            sample = self.transform(sample)
            # input_data = self.transform(input_data)
            # output_data = self.transform(output_data)

        return sample


class DeepKaraokeTrain(DeepKaraokeDataset):
    def __init__(self, dataset_file, transform=None):
        super(DeepKaraokeTrain, self).__init__(dataset_file, transform)
        self.metadata = self.metadata[:int(TRAIN_PORTION*len(self.metadata))]



class DeepKaraokeTest(DeepKaraokeDataset):
    def __init__(self, dataset_file, transform=None):
        super(DeepKaraokeTest, self).__init__(dataset_file, transform)
        self.metadata = self.metadata[int(TRAIN_PORTION*len(self.metadata)):]


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def convert(self, x):
        # type: (np.ndarray) -> torch.Tensor
        y = torch.from_numpy(x.flatten())  # type: torch.Tensor
        return y

    def __call__(self, sample):
        # return torch.from_numpy(sample[0]), torch.from_numpy(sample[1])
        try:
            return {'input': self.convert(sample['input']).float(),
                    'output': self.convert(sample['output']).float()}
        except Exception as ex:
            print(sample, ex)
            raise ex


# Define Net
from torch import nn
import torch.nn.functional as F


class Net1(nn.Module):
    def __init__(self, inputSize, innerSize):
        super(Net1, self).__init__()
        self.inputM = inputSize
        self.outputM = self.inputM
        self.innerM = innerSize
        self.layer1 = nn.Linear(self.inputM, self.innerM)
        self.layer2 = nn.Linear(self.innerM, self.innerM)
        self.layer3 = nn.Linear(self.innerM, self.outputM)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = (self.layer3(x))
        return x


class DeepKaraoke():
    def train(self, use_gpu=True, checkpoint_file=None):
        transformation = transforms.Compose([
            ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = DeepKaraokeTrain(dataset_file="metadata.json", transform=transformation)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=20,
                                                 shuffle=True, num_workers=3)
        input_size = trainset.sample_shape[0]*trainset.sample_shape[1]
        print("input size: %d" % input_size)
        inner_size = input_size
        if use_gpu:
            inner_size //= 1
        net = Net1(input_size, inner_size).float()
        cpu_device = torch.device("cpu")
        if use_gpu:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = cpu_device
        net.to(device) # move to GPU

        # criterion = nn.CrossEntropyLoss()
        # criterion = nn.BCELoss()
        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.NLLLoss(size_average=False)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.2)  # SGD + momentum

        # Train net
        t0 = time()
        n_print = 200
        losses = []
        for epoch in range(100):  # loop over the dataset multiple times
            running_loss = 0.0
            print("training epoch", epoch + 1)
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, target = data["input"], data["output"]
                inputs, target = inputs.to(device), target.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = sum([criterion(o, t) for o, t in zip(outputs, target)])
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % n_print == n_print-1:  # print every n_print mini-batches
                    norm_loss = running_loss/n_print
                    losses.append(norm_loss)
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, norm_loss))
                    running_loss = 0.0
            torch.save(net.state_dict(), "checkpoints/karaoke_{}_checkpoint_{}.torch".format("gpu" if use_gpu else "cpu", epoch+1))

        print('Finished Training. Training took %d seconds' % (time() - t0))
        print('losses:', losses)
        torch.save(net.state_dict(), "karaoke_{}.torch".format("gpu" if use_gpu else "cpu"))
        # net.to(cpu_device)
        # print('copied NN to CPU')

    def break_song(self, song_path, model_path="karaoke_gpu.torch", **kwargs):
        selectedDelta = 10
        sampleLen = spectrum_helper.sampleLen
        song = wavfile.read(song_path)[1][:, 0]
        Sxx_abs_norm = np.abs(spectrum_helper.transform_signal(song/song.std()))
        # Sxx = spectrum_helper.transform_signal(song)
        # Sxx_abs, Sxx_phase = np.abs(Sxx), np.angle(Sxx)  # type: np.ndarray, np.ndarray
        spectrogram_parts = spectrum_helper.dissect_spectrogram(Sxx_abs_norm, sampleDelta=selectedDelta)
        spectrogram_parts_flat = np.stack([x.flatten() for x in spectrogram_parts])
        spectrogram_parts_tensor = torch.from_numpy(spectrogram_parts_flat).float()  # type: torch.Tensor
        print(spectrogram_parts_tensor)

        input_size = spectrogram_parts_flat.shape[1]
        print("input size: %d" % input_size)
        inner_size = input_size
        print("Loading trained net and params")
        t0 = time()
        net = Net1(input_size, inner_size).float()
        net.load_state_dict(torch.load(model_path))
        print("Done loading net. Took {} seconds".format(time()-t0))
        t0 = time()
        output = net(spectrogram_parts_tensor)  # type: torch.Tensor
        mask_parts_flat = output.sigmoid().detach().numpy()
        print("Finished calculating mask. Took {} seconds".format(time()-t0))
        mask_parts = [np.reshape(x, [input_size//sampleLen, sampleLen]) for x in mask_parts_flat]
        np.save("mask_parts.npy", mask_parts)
        mask = spectrum_helper.assemble_mask(mask_parts, selectedDelta)
        print("mask max: {} min: {}".format(mask.max(), mask.min()))
        np.save("mask.npy", mask)

        vocals, instrumental = spectrum_helper.separate_with_mask(song, mask, force_mask_structure=True)
        wavfile.write("vocals.wav", spectrum_helper.fs, vocals)
        wavfile.write("instrumental.wav", spectrum_helper.fs, instrumental)


if __name__ == "__main__":
    Fire(DeepKaraoke)

