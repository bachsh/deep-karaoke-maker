import numpy as np
from scipy import signal
from scipy.io import wavfile
import json
from os import path, makedirs
from time import time


fs = 44100
nperseg = 2**9
window = 'hann'
# noverlap = 512
sampleLen = 20
sampleDelta = 60


def transform_signal(x):
    f, t, Sxx = signal.stft(x, fs, window=window, nperseg=nperseg)
    return Sxx


def mix_channels(x0, x1):
    x = (x0 + x1)  # type: np.array
    x_norm = x/x.std()
    Sxx0 = transform_signal(x0)
    Sxx1 = transform_signal(x1)
    mask = np.abs(Sxx1) > np.abs(Sxx0)  # type: np.ndarray
    return x_norm, mask.astype(np.int)


def separate_with_mask(x, mask, force_mask_structure=False):
    f, t, Sxx = signal.stft(x, fs, window=window, nperseg=nperseg)
    if force_mask_structure:
        Sxx = Sxx[:mask.shape[0], :mask.shape[1]]
    phase_exp = np.exp(1j*np.angle(Sxx))  # type: np.ndarray
    magnitude = np.abs(Sxx)  # type: np.ndarray
    Sxx0 = magnitude*(1-mask) * phase_exp
    Sxx1 = magnitude*mask * phase_exp
    _, x0 = signal.istft(Sxx0, fs, window=window, nperseg=nperseg)
    _, x1 = signal.istft(Sxx1, fs, window=window, nperseg=nperseg)
#     print (Sxx1 - Sxx2)
    return x0, x1


def dissect_spectrogram(x, sampleLen=sampleLen, sampleDelta=sampleDelta):
    all_parts = range((x.shape[1]-sampleLen) // sampleDelta)
    return [x[:, i * sampleDelta:(i * sampleDelta + sampleLen)] for i in all_parts]


def assemble_mask(mask_parts, sampleDelta=sampleDelta):
    mask_part_shape = mask_parts[0].shape
    sampleLen = mask_part_shape[1]
    mask = np.zeros((mask_part_shape[0], len(mask_parts)*sampleDelta + sampleLen))
    factor = np.zeros_like(mask)
    for idx, part in enumerate(mask_parts):
        mask[:, idx*sampleDelta:(idx * sampleDelta + sampleLen)] = part
        factor[:, idx*sampleDelta:(idx * sampleDelta + sampleLen)] += 1
    return mask / factor


def combine_stems(filenames):
    return sum([wavfile.read(filename)[1][:, 0] for filename in filenames])  # Take only one channel


def make_dataset(db_json_filename, datadir="data"):
    t0 = time()
    makedirs(datadir, exist_ok=True)
    metadata = []
    with open(db_json_filename) as json_file:
        data = json.load(json_file)
        base_path = data['base_path']
        for idx, song in enumerate(data['mixes']):
            print("Reading song %d, name: %s" % (idx, song['mix_path']))
            song_dir = path.join(datadir, "song{}".format(idx))
            makedirs(song_dir, exist_ok=True)
            other_stems_paths = [path.join(base_path, x) for x in song['other_stems']]
            target_stems_paths = [path.join(base_path, x) for x in song['target_stems']]
            try:
                other_stems = combine_stems(other_stems_paths)
                target_stems = combine_stems(target_stems_paths)
            except ValueError as ex:
                print(ex)
                print("Error. Skipping file")
                continue
            mix, mask = mix_channels(other_stems, target_stems)
            Sxx = transform_signal(mix)
            Sxx_real = np.real(Sxx)
            mix_parts = dissect_spectrogram(Sxx_real)
            mask_parts = dissect_spectrogram(mask)
            wavfile.write(path.join(song_dir, "mix.wav"), rate=fs, data=mix)
            for idx2, (mix_part, mask_part) in enumerate(zip(mix_parts, mask_parts)):
                input_path = path.join(song_dir, "{}.in".format(idx2))
                output_path = path.join(song_dir, "{}.out".format(idx2))
                np.save(input_path, mix_part)
                np.save(output_path, mask_part)
                metadata.append({
                    "input": input_path,
                    "output": output_path,
                    "song": song['mix_path'],
                })
    dataset_data = {
        "sample_shape": [nperseg//2+1, sampleLen],
        "song_samples": metadata,
    }
    json.dump(dataset_data, open("metadata.json", "w+"))
    print("Dataset ready. Took {} seconds".format(time()-t0))


if __name__ == "__main__":
    make_dataset("medleydb_deepkaraoke.json")
    # vocalFile = "../../MedleyDB/Audio/LizNelson_Coldwar/LizNelson_Coldwar_STEMS/LizNelson_Coldwar_STEM_02.wav"
    # sampFreq, vocals = wavfile.read(vocalFile)
    # vocals = vocals[:, 0]
    # print(vocals.shape)
    # instrumentsFile = "../../MedleyDB/Audio/LizNelson_Coldwar/LizNelson_Coldwar_STEMS/LizNelson_Coldwar_STEM_01.wav"
    # sampFreq, instruments = wavfile.read(instrumentsFile)
    # instruments = instruments[:, 0]
    # print(instruments.shape)
    # print(sampFreq)
    #
    # example_mix, example_mask = mix_channels(vocals, instruments)
    # print(example_mask.shape)
    # x1, x2 = separate_with_mask(example_mix, example_mask)
    # print(x1.shape, x2.shape)
    #
    # # mix_file = "../../MedleyDB/Audio/TheScarletBrand_LesFleursDuMal/TheScarletBrand_LesFleursDuMal_MIX.wav"
