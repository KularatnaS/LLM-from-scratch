def get_config():

    return {
        'data_dir': 'data/',
        'train_data_dir': 'data/train/',
        'val_data_dir': 'data/val/',
        'force_build_tokenizer': 'true',  # 'true' or 'false'
        'tokenizer_file': 'tokenizer/tokenizer.json',
        # Training parameters
        'seq_len': 64,
        'd_model': 512,
        'd_ff': 1024,
        'N': 6,
        'h': 8,
        'dropout': 0.0,
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
    # if epoch is a single digit, pad it with a zero
    if type(epoch) is int:
        if epoch is not None and epoch < 10:
            epoch = f'0{epoch}'
    return f'{model_folder}/{model_basename}{epoch}.pt'