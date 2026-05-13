"""
Guide:
    * selection first filters candidates by voice mode (female vs male), word/label,
    and candidate (global vs per-phoneme).
    * Load one candidate and uses source phoneme alignment to convert the EEG
    mismatch to per-phoneme mismatch via compute_per_phoneme_mismatch
    * for global, collapse by taking mean and then take the closest one
"""
from pathlib import Path
import pprint
from os import makedirs
from os.path import join, exists
import pickle

from Template_l2_compare_v2 import (
    compare_signal_to_prebuilt_template,
    build_template_from_dataset,
    get_one_trial_from_dataset,
)
from select_blabber_asset import (
    adapt_comparison_result,
    load_asset_index,
    normalize_voice_mode,
    scan_asset_root,
    select_asset,
)


REQUESTED_VOICE = "female"
ROOT_PATH = Path("/data/raqchia/audio-assets/speech-assets") 
OUTPUT_PATH = ROOT_PATH / Path(REQUESTED_VOICE)
ASSET_INDEX_PATH: Path | None = OUTPUT_PATH / Path("blabber_asset_index.json")
GENERATION_MODE = "global"


def load_candidates():
    if ASSET_INDEX_PATH is not None:
        return load_asset_index(ASSET_INDEX_PATH)
    if OUTPUT_PATH is not None:
        return scan_asset_root(OUTPUT_PATH)
    raise ValueError("Set ASSET_INDEX_PATH or OUTPUT_PATH before running my_test.py.")


if __name__ == "__main__":
    """
    Take from `Function call copy_v2.ipynb`
    """
    template_days = [1]
    template_sessions = range(1, 8)
    template_directory = '/data/raqchia/imaged-speech/eeg-templates'
    template_fname = 'template.pkl'

    label = 0
    test_day = 1
    test_template_fname = join(template_directory, f'day{test_day}', template_fname)
    
    for day in template_days:
        template_day_dir = join(template_directory, f'day{day}')
        makedirs(template_day_dir, exist_ok=True)
        template_fname_out = join(template_day_dir, template_fname)
        if exists(template_fname_out):
            print("already exists, skipping template creation")
            continue
        template_object = build_template_from_dataset(
            template_days=template_days,
            template_sessions=template_sessions,
            nperseg=128,
            noverlap=96,
            eps=1e-8,
            fmax=50,
        )
        with open(template_fname_out, 'wb') as f:
            pickle.dump(template_object, f)
            
    trial, label_name, dataset = get_one_trial_from_dataset(
        day=test_day,
        sess=8,
        label=label,
        trial_index_within_label=0,
    )
    print("Loaded trial label:", label_name)

    # get template object according to day and label
    print(template_fname_out)
    with open(test_template_fname, 'rb') as f:
        template_object = pickle.load(f)

    result = compare_signal_to_prebuilt_template(
        template_object=template_object,
        input_signal=trial,
        input_label=label,
        output_mode="both",
        sr=250,
    )
    pprint.pprint(result)

    comparison = adapt_comparison_result(result)
    pprint.pprint(comparison)
    candidates = load_candidates()
    selection = select_asset(
        comparison=comparison,
        candidates=candidates,
        requested_voice=normalize_voice_mode(REQUESTED_VOICE),
        generation_mode=GENERATION_MODE,
    )

    print("Selected asset:", selection["audio_path"])
    print("Selected sidecar:", selection["sidecar_path"])
    print("Generation mode:", selection["generation_mode"])
    print("Phoneme scores:", selection["phoneme_scores"])
