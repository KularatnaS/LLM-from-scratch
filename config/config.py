def get_config():

    return {
        'data_dir': 'data/',
        'train_data_dir': 'data/train/',
        'val_data_dir': 'data/val/',
        'force_build_tokenizer': 'false',  # 'true' or 'false'
        'tokenizer_file': 'tokenizer/tokenizer.json',
        # Training parameters
        'seq_len': 128,
        'd_model': 512,
        'd_ff': 1024,
        'N': 4,
        'h': 8,
        'dropout': 0.1,
        'batch_size': 8,
        'lr': 10**-4,
        'epochs': 100,
        # model checkpoint
        'model_folder': 'weights',
        'model_basename': 'tmodel_',
        'preload': None,  # None or epoch number
        'experiment_name': 'runs/tmodel'
    }


def get_weights_file_path(model_folder, model_basename, epoch):
    return f'{model_folder}/{model_basename}{epoch}.pt'