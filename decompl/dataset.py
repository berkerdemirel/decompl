from config import *
from volleyball import *
from collective import *
import pickle


"""
Reference:
https://github.com/JacobYuan7/DIN-Group-Activity-Recognition-Benchmark/blob/main/dataset.py
"""

def return_dataset(cfg: Config, verbose: bool=True) -> Tuple[data.Dataset, data.Dataset]:
    if cfg.dataset_name == 'volleyball':
        train_anns = volley_read_dataset(cfg.data_path, cfg.train_seqs)
        train_frames = volley_all_frames(train_anns)

        test_anns = volley_read_dataset(cfg.data_path, cfg.test_seqs)
        test_frames = volley_all_frames(test_anns)

        all_anns = {**train_anns, **test_anns}
        all_tracks = pickle.load(open(cfg.data_path + '/tracks_normalized.pkl', 'rb'))

        training_set = VolleyballDataset(all_anns, all_tracks, train_frames,
                                      cfg.data_path, cfg.image_size, cfg.out_size, num_before=cfg.num_before,
                                       num_after=cfg.num_after, is_training=True, flip=True)

        validation_set = VolleyballDataset(all_anns, all_tracks, test_frames,
                                      cfg.data_path, cfg.image_size, cfg.out_size, num_before=cfg.num_before,
                                         num_after=cfg.num_after ,is_training=False, flip =False)
    elif cfg.dataset_name == 'collective':
        train_anns = collective_read_dataset(cfg.data_path, cfg.train_seqs)
        train_frames = collective_all_frames(train_anns)

        test_anns = collective_read_dataset(cfg.data_path, cfg.test_seqs)
        test_frames = collective_all_frames(test_anns)

        training_set = CollectiveDataset(train_anns,train_frames,
                                        cfg.data_path,cfg.image_size,cfg.out_size,
                                        num_frames = cfg.num_frames, is_training=True, flip=True)

        validation_set = CollectiveDataset(test_anns,test_frames,
                                        cfg.data_path,cfg.image_size,cfg.out_size,
                                        is_training=False)
    else:
        assert False
                                         
    if verbose:
        print('Reading dataset finished...')
        print('%d train samples'%len(train_frames))
        print('%d test samples'%len(test_frames))
    
    return training_set, validation_set


if __name__ == "__main__":
    start = time.time()
    cfg = Config("volleyball")
    training_set, validation_set = return_dataset(cfg, verbose=False)
    end = time.time()
    print("Reading dataset took: {} seconds".format(end-start))
    print(len(training_set))
    print(len(validation_set))
    

