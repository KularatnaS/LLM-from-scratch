from tqdm import tqdm

import warnings
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config.config import get_config, get_weights_file_path
from dataset.dataset import get_or_build_tokenizer, TextDataset, causal_mask
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
model = build_transformer(vocab_size, seq_len, d_model, N, h, dropout, d_ff).to(device)

# Tensorboard
writer = SummaryWriter(experiment_name)
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-9)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"), label_smoothing=0.1).to(device)


def train_model():
    initial_epoch = 0
    global_step = 0
    if preload is not None:
        model_filename = get_weights_file_path(model_folder, model_basename, preload)
        print(f"Loading model weights from {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        model.load_state_dict(state['model_state_dict'])

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

        # run validation
        run_validation(epoch)
        test_inference_texts = ['He found what he was looking for in his inside pocket. And then he realised that it was gone. He had lost the Marauder’s Map.',
                                'Harry and Hermione rushed up to the hospital wing to see Ron. He was lying in a bed with his eyes closed. His face was very pale.',
                                'The next day, however, Harry barely grinned once. He was in fury with Ron for',]
        for text in test_inference_texts:
            inference_test(text)

        # save model after each epoch
        if epoch % 5 == 0:
            model_filename = get_weights_file_path(model_folder, model_basename, f'{epoch:02d}')
            print(f"Saving model weights to {model_filename}")
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.state_dict(),
            }, model_filename)


def run_validation(epoch):
    model.eval()
    with torch.no_grad():
        batch_iterator_val = tqdm(val_dataloader, desc=f"Validating Epoch {epoch:02d}")
        for batch in batch_iterator_val:
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            encoder_output = model.encode(encoder_input, encoder_mask)
            proj_output = model.project(encoder_output)
            label = batch['label'].to(device)
            val_loss = loss_fn(proj_output.view(-1, vocab_size), label.view(-1))
            batch_iterator_val.set_postfix({'validation loss': val_loss.item()})
            writer.add_scalar('val loss', val_loss.item(), epoch)
            writer.flush()


def inference_test(input_text):
    model.eval()
    with torch.no_grad():
        input_encoded = tokenizer.encode(input_text).ids
        encoder_input = torch.tensor(input_encoded).unsqueeze(0).to(device)
        while True:
            if encoder_input.size(1) == seq_len:
                break
            encoder_mask = causal_mask(encoder_input.size(1)).type_as(encoder_input).to(device)

            out = model.encode(encoder_input, encoder_mask)
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=-1)
            encoder_input = torch.cat([encoder_input, torch.empty(1, 1).type_as(encoder_input).fill_(next_word.item()).to(device)], dim=1)

        # convert encoder_input to cpu and numpy
        encoder_input = encoder_input.detach().cpu().numpy()
        # decode the tokens
        decoded_tokens = tokenizer.decode(encoder_input[0].tolist())
        # print the decoded tokens
        print(decoded_tokens)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    train_model()