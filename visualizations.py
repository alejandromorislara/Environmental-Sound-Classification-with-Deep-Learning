import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import soundfile as sf
from utils import calculate_envelope

def plot_urban_sound_slices(metadata_path=r"D:\Python_D\DeepLearningAudios\metadata\UrbanSound8K.csv"):
    urbansound_metadata = pd.read_csv(metadata_path)
    urbansound_metadata['duration'] = urbansound_metadata['end'] - urbansound_metadata['start']
    total_duration = urbansound_metadata.groupby('class')['duration'].sum() / 60

    urbansound_metadata['FG/BG'] = urbansound_metadata['salience'].map({1: 'FG', 2: 'BG'})
    fg_bg_counts = urbansound_metadata.groupby(['class', 'FG/BG']).size().unstack(fill_value=0)

    classes = total_duration.index.tolist()
    fg_slices = fg_bg_counts.get('FG', [0] * len(classes)).tolist()
    bg_slices = fg_bg_counts.get('BG', [0] * len(classes)).tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(classes))
    ax.bar(x, fg_slices, label='FG', color='#d3d3d3')
    ax.bar(x, bg_slices, bottom=fg_slices, label='BG', color='lightblue')

    ax.set_title("(b) Slices per class (FG/BG)", fontsize=12)
    ax.set_xlabel("Classes", fontsize=10)
    ax.set_ylabel("Slices", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()

def repeat_audio_envelope(input_path, aimed_duration=4.0, export = True,output_path='sounds'):
    y, sr = librosa.load(input_path, sr=None)

    envelope = calculate_envelope(y=y,sr=sr)

    envelope_duration = len(envelope) / sr

    num_repeats = int(np.ceil(aimed_duration / envelope_duration))

    repeated_envelope = np.tile(envelope, num_repeats)

    target_length = int(aimed_duration * sr)
    repeated_envelope = repeated_envelope[:target_length]

    repeated_audio = np.tile(y, int(np.ceil(target_length / len(y))))[:target_length]

    final_audio = repeated_audio * repeated_envelope
    if export:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        filename = os.path.basename(input_path)
        output_file_path = os.path.join(output_path, filename)
        sf.write(output_file_path, final_audio, sr)

        print(f"File {input_path} saved at {output_path}")
    
    times_audio = np.arange(len(y)) / sr
    times_envelope = np.arange(len(envelope)) / sr
    times_repeated_envelope = np.arange(len(repeated_envelope)) / sr
    times_final_audio = np.arange(len(final_audio)) / sr

    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(times_audio, y, label="Original Audio", color='blue')
    plt.title("Original Audio Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 2)
    plt.plot(times_repeated_envelope, repeated_envelope, label="Repeated Envelope", color='orange')
    plt.title("Repeated Smooth Envelope")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 3)
    plt.plot(times_final_audio, final_audio, label="Final Audio (Modulated)", color='green')
    plt.title("Final Modulated Audio Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


def plot_signal_and_envelope(audio_path, frame_size=2048, hop_length=512, figsize=(10, 6)):
    signal, sr = librosa.load(audio_path, sr=None)
    envelope = calculate_envelope(signal, sr)
    rms = librosa.feature.rms(y=signal, frame_length=frame_size, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=signal, frame_length=frame_size, hop_length=hop_length)[0]
    times_signal = np.linspace(0, len(signal) / sr, len(signal))
    times_envelope = np.linspace(0, len(signal) / sr, len(envelope))
    times_rms = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
    times_zcr = librosa.frames_to_time(range(len(zcr)), sr=sr, hop_length=hop_length)
    plt.figure(figsize=figsize)
    plt.plot(times_signal, signal, alpha=0.5, label="Waveform")
    plt.plot(times_envelope, envelope, color='red', label="Envelope (Harmonic)")
    plt.plot(times_rms, rms, color='blue', label="RMS Energy")
    plt.plot(times_zcr, zcr, color='green', label="Zero Crossing Rate (ZCR)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude / Rate")
    plt.title("Audio Signal, Envelope, RMS, and ZCR")
    plt.legend()
    plt.show()

def plot_DL_results(
    train_accuracies,
    valid_accuracies,
    train_losses,
    valid_losses,
    train_f1_scores,
    valid_f1_scores,
    train_recalls,
    valid_recalls,
    epochs_range=None
):
    # Determine epochs_range if not provided
    if epochs_range is None:
        epochs_range = range(1, len(train_accuracies) + 1)
    
    # Plotting the loss
    plt.figure(figsize=(12, 6))
    
    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, valid_losses, label='Validation Loss', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.grid(True)

    # Metrics subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy', color='g')
    plt.plot(epochs_range, valid_accuracies, label='Validation Accuracy', color='y')
    plt.plot(epochs_range, train_f1_scores, label='Train F1 Score', color='b')
    plt.plot(epochs_range, valid_f1_scores, label='Validation F1 Score', color='orange')
    plt.plot(epochs_range, train_recalls, label='Train Recall', color='r')
    plt.plot(epochs_range, valid_recalls, label='Validation Recall', color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Metrics per Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.tight_layout()
    plt.show()




def analyze_audio_features(
    audio_path, 
    n_mfcc=25, 
    n_mels=40, 
    frame_duration=0.0232, 
    hop_overlap=0.5, 
    fmin=1, 
    fmax=22050
):
    """
    Analyze audio features and plot them using default parameters from Essentia-inspired experiments.
    Includes Mel Spectrogram and its Delta 1 (first derivative).

    Parameters:
        audio_path (str): Path to the audio file.
        n_mfcc (int): Number of MFCC coefficients to compute.
        n_mels (int): Number of Mel bands for Mel spectrogram computation.
        frame_duration (float): Frame duration in seconds (default is 23.2 ms).
        hop_overlap (float): Frame overlap percentage (default is 50% overlap).
        fmin (float): Minimum frequency for Mel bands and spectral features (default is 0 Hz).
        fmax (float): Maximum frequency for Mel bands and spectral features (default is 22050 Hz).
    """
    # Load the audio file
    y, sr = librosa.load(audio_path)

    # Frame size and hop length calculations
    frame_length = int(sr * frame_duration)  # Frame duration in samples
    hop_length = int(frame_length * hop_overlap)  # Hop length for 50% overlap

    # Create a figure for plotting
    plt.figure(figsize=(14, 12))

    # **Waveform** - Display the audio waveform
    plt.subplot(3, 3, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')

    # **Zero Crossing Rate (ZCR)**
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y).T
    plt.subplot(3, 3, 2)
    plt.plot(zero_crossing_rate)
    plt.title('Zero Crossing Rate (ZCR)')

    # **Root Mean Square Energy (RMSE)**
    root_mean_square_energy = librosa.feature.rms(y=y).T
    plt.subplot(3, 3, 3)
    plt.plot(root_mean_square_energy)
    plt.title('Root Mean Square Energy (RMSE)')

    # **MFCCs**
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=frame_length, hop_length=hop_length, fmin=fmin, fmax=fmax)
    plt.subplot(3, 3, 4)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.title('MFCCs')

    # **Spectral Centroid**
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).T
    plt.subplot(3, 3, 5)
    plt.plot(spectral_centroid)
    plt.title('Spectral Centroid')

    # **Spectral Bandwidth**
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).T
    plt.subplot(3, 3, 6)
    plt.plot(spectral_bandwidth)
    plt.title('Spectral Bandwidth')

    # **Spectral Rolloff**
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).T
    plt.subplot(3, 3, 7)
    plt.plot(spectral_rolloff)
    plt.title('Spectral Rolloff')

    # **Mel Spectrogram**
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=frame_length, fmin=fmin, fmax=fmax
    )
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    plt.subplot(3, 3, 8)
    librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')

    # **Delta 1 (First Derivative) of Mel Spectrogram**
    delta_mel_spectrogram = librosa.feature.delta(mel_spectrogram_db, order=1)
    plt.subplot(3, 3, 9)
    librosa.display.specshow(delta_mel_spectrogram, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Delta 1 of Mel Spectrogram')

    # Show all the plots
    plt.tight_layout()
    plt.show()


