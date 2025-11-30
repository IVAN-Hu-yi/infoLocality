import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.auto import tqdm

class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_prob=0.5):
        super(RNNLanguageModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_sequence, hidden=None):
        # input_sequence: (batch_size, seq_len)
        embeddings = self.embeddings(input_sequence)  # (batch_size, seq_len, embedding_dim)

        if hidden is None:
            # Initialize hidden and cell states if not provided
            h0 = torch.zeros(self.num_layers, input_sequence.size(0), self.hidden_dim).to(input_sequence.device)
            c0 = torch.zeros(self.num_layers, input_sequence.size(0), self.hidden_dim).to(input_sequence.device)
            hidden = (h0, c0)

        out, hidden = self.lstm(embeddings, hidden)
        logits = self.linear(out[:, -1, :])

        return logits, hidden

    def train_model(self, dataloader, num_epochs, learning_rate, device=None):
        """
        Trains the RNN Language Model.

        Args:
            dataloader (DataLoader): DataLoader providing batches of n-grams.
            num_epochs (int): Number of epochs to train for.
            learning_rate (float): Learning rate for the optimizer.
            device (torch.device, optional): The device (CPU/GPU) to train on. Defaults to None (inferred from model).
        """
        if device is None:
            device = next(self.parameters()).device # Get device from model parameters

        self.to(device)
        self.train() # Set the model to training mode

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        print(f"Starting training on {device} for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, ngram_batch in tqdm(
                    enumerate(dataloader), 
                    total=len(dataloader),
                    desc=f"Epoch {epoch+1}/{num_epochs}",
                    leave=True
                ):
                # ngram_batch is (batch_size, N)
                input_tokens = ngram_batch[:, :-1].to(device)
                target_tokens = ngram_batch[:, -1].to(device)

                optimizer.zero_grad()

                # Forward pass
                logits, _ = self(input_tokens) # Use self() which calls forward()

                # Calculate loss
                loss = criterion(logits, target_tokens)
                total_loss += loss.item()

                # Backward and optimize
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(dataloader)
            # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # print("Training finished!")

    def evaluate_rnn(self, model, dataloader, device):
        model.eval()
        surprisals = []

        with torch.no_grad():
            for batch in dataloader:
                input_tokens  = batch[:, :-1].to(device)
                target_tokens = batch[:, -1].to(device)

                logits, _ = model(input_tokens)
                log_probs = torch.log_softmax(logits, dim=1)

                # correct surprisal
                s = - torch.gather(log_probs, 1, target_tokens.unsqueeze(0).T).squeeze()
                s = s.detach().models.cpu().numpy().tolist()
                surprisals.extend(s[:])
        
        return surprisals


# No change needed for __call__ as it implicitly calls forward.
# The previous __call__ method for NaiveTokenizer is not affected.