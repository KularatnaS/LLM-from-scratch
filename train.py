from tqdm import tqdm

import warnings
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config.config import get_config, get_weights_file_path
from dataset.dataset import get_or_build_tokenizer, TextDataset
from model.model import build_transformer

config = get_config()
data_dir = config['data_dir']
train_data_dir = config['train_data_dir']
val_data_dir = config['val_data_dir']
seq_len = config['seq_len']
d_model = config['d_model']
batch_size = config['batch_size']
N = config['N']
h = config['h']
dropout = config['dropout']
d_ff = config['d_ff']
epochs = config['epochs']
lr = config['lr']
model_folder = config['model_folder']
model_basename = config['model_basename']
preload = config['preload']
experiment_name = config['experiment_name']
tokenizer_file = config['tokenizer_file']
force_build_tokenizer = config['force_build_tokenizer']
data_dir = config['data_dir']

tokenizer = get_or_build_tokenizer(tokenizer_file, data_dir, force_build_tokenizer)
vocab_size = tokenizer.get_vocab_size()

train_ds = TextDataset(tokenizer, train_data_dir, seq_len)
train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

val_ds = TextDataset(tokenizer, val_data_dir, seq_len)
val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    model = build_transformer(vocab_size, seq_len, d_model, N, h, dropout, d_ff).to(device)

    # Tensorboard
    writer = SummaryWriter(experiment_name)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if preload is not None:
        model_filename = get_weights_file_path(model_folder, model_basename, preload)
        print(f"Loading model weights from {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, epochs):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)  # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (batch, 1, 1, seq_len)

            # run tensors through transformer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (batch, seq_len, d_model)
            proj_output = model.project(encoder_output)  # (batch, seq_len, vocab_size)

            label = batch['label'].to(device)  # (batch, seq_len)

            # calculate loss
            # for proj_output -> (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
            # for label -> (batch, seq_len) -> (batch * seq_len)
            loss = loss_fn(proj_output.view(-1, vocab_size), label.view(-1))

            batch_iterator.set_postfix({'loss': loss.item()})
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # save model after each epoch
        model_filename = get_weights_file_path(model_folder, model_basename, f'{epoch:02d}')
        print(f"Saving model weights to {model_filename}")
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict(),
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    train_model()