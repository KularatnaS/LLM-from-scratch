import torch
from config.config import get_config
from dataset.dataset import get_or_build_tokenizer, causal_mask
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
model = build_transformer(vocab_size, seq_len, d_model, N, h, dropout, d_ff).to(device)


def inference(input_text, model_path):
    state = torch.load(model_path)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    with torch.no_grad():
        input_encoded = tokenizer.encode(input_text).ids
        encoder_input = torch.tensor(input_encoded).unsqueeze(0).to(device)
        counter = 0
        while True:
            if encoder_input.size(1) == seq_len:
                # feed the last half of the sequence to the model
                encoder_input = encoder_input[:, seq_len // 2:]
            encoder_mask = causal_mask(encoder_input.size(1)).type_as(encoder_input).to(device)
            out = model.encode(encoder_input, encoder_mask)
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=-1)
            print(tokenizer.decode([next_word.item()]), end=" ")
            counter += 1
            if counter % 50 == 0:
                print("\n")
            encoder_input = torch.cat([encoder_input, torch.empty(1, 1).type_as(encoder_input).fill_(next_word.item()).to(device)], dim=1)

        # convert encoder_input to cpu and numpy
        encoder_input = encoder_input.detach().cpu().numpy()
        # decode the tokens
        # decoded_tokens = tokenizer.decode(encoder_input[0].tolist())
        # print the decoded tokens
        # print(decoded_tokens)


if __name__ == '__main__':
    # inference(<prompt>, <model_path>)
    inference("He found what he was looking for in his inside pocket", "weights/tmodel_10.pt")