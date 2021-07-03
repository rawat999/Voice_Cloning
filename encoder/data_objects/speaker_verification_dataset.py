from random_cycler import RandomCycler
from speaker_batch import SpeakerBatch
from speaker import Speaker # encoder.data_objects.
#from encoder.params_data import partials_n_frames
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# TODO: improve with a pool of speakers for data efficiency

class SpeakerVerificationDataset(Dataset):
    def __init__(self, datasets_root: Path):
        self.root = datasets_root
        speaker_dirs = [f for f in self.root.glob("*") if f.is_dir()]
        if len(speaker_dirs) == 0:
            raise Exception("No speakers found. make sure you are pointing to the directory "
                            "containing all preprocessed speaker directories.")

        self.speakers = [Speaker(speaker_dir) for speaker_dir in speaker_dirs]
        self.speaker_cycler = RandomCycler(self.speakers)

    def __len__(self):
        return int(1e10)

    def get_logs(self):
        log_string = ""
        for log_fpath in self.root.glob("*.txt"):
            with log_fpath.open("r") as log_file:
                log_string += "".join(log_file.readlines())
        return log_string


class SpeakerVerificationDataLoader(DataLoader):
    def __init__(self, dataset, speakers_per_batch, utterances_per_speaker, sampler=None,
                 batch_sampler=None, num_workers=0, pin_memory=False, timeout=0,
                 worker_init_fn=None):
        self.utterances_per_speaker = utterances_per_speaker

        super().__init__(dataset, batch_size=speakers_per_batch,
                         shuffle=False, sampler=sampler,
                         batch_sampler=batch_sampler, num_workers=num_workers,
                         collate_fn=self.collate, pin_memory=pin_memory,
                         drop_last=False, timeout=timeout, worker_init_fn=worker_init_fn)

    def collate(self, speakers):
        return SpeakerBatch(speakers, self.utterances_per_speaker, partials_n_frames)


## Model parameters
model_hidden_size = 256
model_embedding_size = 256
model_num_layers = 3


## Training parameters
learning_rate_init = 1e-4
speakers_per_batch = 4
utterances_per_speaker = 5

import torch
p = Path('./dev-clean/')
data = SpeakerVerificationDataset(p)

loader = SpeakerVerificationDataLoader(p, speakers_per_batch, utterances_per_speaker)

init_step = 1
for i in data:
    #profiler.tick("Blocking, waiting for batch (threaded)")

    # Forward pass
    #inputs = torch.from_numpy(speaker_batch.data)
    print(i)
    break;
