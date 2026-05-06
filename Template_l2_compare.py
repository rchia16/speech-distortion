import os
import pickle
import numpy as np
from scipy.signal import stft

BASE_ROOT = r"C:/Users/IDBA/Downloads/UTS_EEG/Dataloader/dataset_cache/Daniel"

def path_builder(day: int, sess: int) -> str:
    return os.path.join(BASE_ROOT, f"Day{day}", f"sess{sess}.pkl")


def patch_pickle_classes():
    try:
        import __main__
        from eeg_datasets import EEGRealtimeDataset
        __main__.EEGRealtimeDataset = EEGRealtimeDataset
        print("[Info] Patched __main__.EEGRealtimeDataset")
    except Exception as e:
        print("[Warn] Could not patch EEGRealtimeDataset automatically:", e)
        print("       If unpickling fails, confirm eeg_datasets.py is importable.")


def read_pkl(pkl_path: str):
    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)
    return dataset


def compute_label_mean_stft(dataset, nperseg=128, noverlap=96, eps=1e-8, fmax=50):
    sr = getattr(dataset, "final_sampling_rate", 250)
    all_label = np.array(dataset.all_label)

    results = {}

    for label in range(len(dataset.final_id2word)):
        idx = np.where(all_label == label)[0]
        if len(idx) == 0:
            continue

        trials = dataset.all_eeg[idx]
        tf_list = []

        for trial in trials:
            if hasattr(trial, "cpu"):
                trial = trial.cpu().numpy()
            else:
                trial = np.array(trial)

            mean = trial.mean(axis=-1, keepdims=True)
            std = trial.std(axis=-1, keepdims=True)
            trial_z = (trial - mean) / (std + eps)

            f, t, Zxx = stft(trial_z, fs=sr, nperseg=nperseg, noverlap=noverlap, axis=-1)
            tf = np.abs(Zxx)

            freq_mask = (f >= 0) & (f <= fmax)
            f_used = f[freq_mask]
            tf = tf[:, freq_mask, :]

            tf_list.append(tf)

        tf_trials = np.stack(tf_list, axis=0)
        tf_mean = tf_trials.mean(axis=0)

        results[label] = {
            "name": dataset.final_id2word[label],
            "indices": idx,
            "tf_trials": tf_trials,
            "tf_mean": tf_mean,
            "freqs": f_used,
            "times": t,
        }

    return results


def tfmean_to_time_bands_fingerprint(tf_mean, freqs):
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
        "gamma": (30, 50),
    }

    band_names = list(bands.keys())
    time_band_list = []

    for band_name, (f_low, f_high) in bands.items():
        mask = (freqs >= f_low) & (freqs < f_high)

        if np.any(mask):
            curve = tf_mean[:, mask, :].mean(axis=(0, 1))
        else:
            curve = np.zeros(tf_mean.shape[-1], dtype=float)

        time_band_list.append(curve.astype(float))

    seq = np.stack(time_band_list, axis=0).T   # (times, bands)
    return seq, band_names


def compute_single_trial_time_band_sequence(trial, sr=250, nperseg=128, noverlap=96, eps=1e-8, fmax=50):
    if hasattr(trial, "cpu"):
        trial = trial.cpu().numpy()
    else:
        trial = np.array(trial)

    mean = trial.mean(axis=-1, keepdims=True)
    std = trial.std(axis=-1, keepdims=True)
    trial_z = (trial - mean) / (std + eps)

    f, t, Zxx = stft(trial_z, fs=sr, nperseg=nperseg, noverlap=noverlap, axis=-1)
    tf = np.abs(Zxx)

    freq_mask = (f >= 0) & (f <= fmax)
    f_used = f[freq_mask]
    tf = tf[:, freq_mask, :]

    seq, band_names = tfmean_to_time_bands_fingerprint(tf, f_used)
    return seq, band_names, f_used, t


def collect_session_band_sequences(days, sessions, nperseg=128, noverlap=96, eps=1e-8, fmax=50):
    all_session_seq = []

    for day in days:
        for sess in sessions:
            pkl_path = path_builder(day, sess)
            print(f"[Collect] Loading: {pkl_path}")
            dataset = read_pkl(pkl_path)

            stft_dict = compute_label_mean_stft(
                dataset,
                nperseg=nperseg,
                noverlap=noverlap,
                eps=eps,
                fmax=fmax
            )

            label_seq = {}
            for label, info in stft_dict.items():
                seq, band_names = tfmean_to_time_bands_fingerprint(info["tf_mean"], info["freqs"])
                label_seq[label] = {
                    "name": info["name"],
                    "seq": seq,
                    "band_names": band_names
                }

            all_session_seq.append({
                "day": day,
                "sess": sess,
                "label_seq": label_seq
            })

    return all_session_seq


def build_template_mean_band_sequences(all_session_seq):
    label_seq_list = {}
    label_name = {}
    band_names_ref = None

    for item in all_session_seq:
        for label, info in item["label_seq"].items():
            if label not in label_seq_list:
                label_seq_list[label] = []
                label_name[label] = info["name"]

            if band_names_ref is None:
                band_names_ref = info["band_names"]

            label_seq_list[label].append(info["seq"])

    template_mean_seq = {}

    for label, seq_list in label_seq_list.items():
        seq_stack = np.stack(seq_list, axis=0)   # (n_sessions, times, bands)
        seq_mean = seq_stack.mean(axis=0)
        seq_std = seq_stack.std(axis=0)

        template_mean_seq[label] = {
            "name": label_name[label],
            "band_names": band_names_ref,
            "seq_mean": seq_mean,
            "seq_std": seq_std,
            "seq_all": seq_stack
        }

    return template_mean_seq


def get_one_trial_from_dataset(day, sess, label, trial_index_within_label=0):
    """
    input:
        day, sess: dataset position
        label: int label id
        trial_index_within_label: Which trial under label to load (default 0, i.e. the first one)
    return:
        trial: (channels, time)
        label_name: str
        dataset
    """
    pkl_path = path_builder(day, sess)
    print(f"[Ref] Loading one trial from: {pkl_path}")
    dataset = read_pkl(pkl_path)

    all_label = np.array(dataset.all_label)
    idx = np.where(all_label == label)[0]
    if len(idx) == 0:
        raise ValueError(f"No trial found for label={label} in Day{day} Session{sess}")

    if trial_index_within_label < 0 or trial_index_within_label >= len(idx):
        raise IndexError(
            f"trial_index_within_label={trial_index_within_label} out of range. "
            f"Available trials for label {label}: 0 ~ {len(idx)-1}"
        )

    trial = dataset.all_eeg[idx[trial_index_within_label]]
    label_name = dataset.final_id2word[label]

    if hasattr(trial, "cpu"):
        trial = trial.cpu().numpy()
    else:
        trial = np.array(trial)

    return trial, label_name, dataset


def compute_l2_to_template(input_seq, template_seq):
    if input_seq.shape != template_seq.shape:
        raise ValueError(f"Shape mismatch: input_seq {input_seq.shape}, template_seq {template_seq.shape}")

    per_time_l2 = np.linalg.norm(input_seq - template_seq, axis=1)
    mean_l2 = float(np.mean(per_time_l2))
    return per_time_l2, mean_l2


# =========================================================
# A. Prebuild template from dataset
# =========================================================
def build_template_from_dataset(
    template_days,
    template_sessions,
    nperseg=128,
    noverlap=96,
    eps=1e-8,
    fmax=50
):
    patch_pickle_classes()

    all_session_seq = collect_session_band_sequences(
        days=template_days,
        sessions=template_sessions,
        nperseg=nperseg,
        noverlap=noverlap,
        eps=eps,
        fmax=fmax
    )

    template_mean_seq = build_template_mean_band_sequences(all_session_seq)

    template_object = {
        "template_days": list(template_days),
        "template_sessions": list(template_sessions),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "eps": eps,
        "fmax": fmax,
        "template_mean_seq": template_mean_seq
    }
    return template_object


# =========================================================
# B. Compare new signal to prebuilt template
# =========================================================
def compare_signal_to_prebuilt_template(
    template_object,
    input_signal,
    input_label,
    output_mode="both",
    sr=250
):
    template_mean_seq = template_object["template_mean_seq"]

    if input_label not in template_mean_seq:
        raise ValueError(f"input_label={input_label} not found in prebuilt template")

    input_seq, band_names, freqs_used, times = compute_single_trial_time_band_sequence(
        input_signal,
        sr=sr,
        nperseg=template_object["nperseg"],
        noverlap=template_object["noverlap"],
        eps=template_object["eps"],
        fmax=template_object["fmax"]
    )

    template_seq = template_mean_seq[input_label]["seq_mean"]
    per_time_l2, mean_l2 = compute_l2_to_template(input_seq, template_seq)

    result = {
        "label_id": input_label,
        "label_name": template_mean_seq[input_label]["name"],
        "band_names": band_names,
        "times": times,
        "per_time_l2": per_time_l2,
        "mean_l2": mean_l2,
        "input_seq": input_seq,
        "template_seq": template_seq
    }

    if output_mode == "per_time":
        return {
            "label_id": result["label_id"],
            "label_name": result["label_name"],
            "times": result["times"],
            "per_time_l2": result["per_time_l2"]
        }
    elif output_mode == "mean":
        return {
            "label_id": result["label_id"],
            "label_name": result["label_name"],
            "mean_l2": result["mean_l2"]
        }
    elif output_mode == "both":
        return result
    else:
        raise ValueError("output_mode must be one of: 'per_time', 'mean', 'both'")


# =========================================================
# C. Optional: save / read template
# =========================================================
def save_template_object(template_object, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(template_object, f)
    print(f"[Saved template] {save_path}")


def load_template_object(save_path):
    with open(save_path, "rb") as f:
        template_object = pickle.load(f)
    print(f"[Loaded template] {save_path}")
    return template_object