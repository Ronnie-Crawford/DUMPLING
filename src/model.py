import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers):
        
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)

    def forward(self, sequences, lengths):
        
        h0 = torch.zeros(self.num_layers, sequences.size(0), self.hidden_size).to(sequences.device)
        sequences = self.embedding(sequences.long())
        lengths = lengths.cpu().to(torch.int64)
        sequences_packed = nn.utils.rnn.pack_padded_sequence(sequences, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(sequences_packed, h0)
        return h_n[-1]

class DecoderRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.variant_embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output_fc = nn.Linear(hidden_size, output_size)

    def forward(self, variant_sequences, combined_latent, variant_lengths):
        
        # Embedding variant sequences
        variant_embedded = self.variant_embedding(variant_sequences.long())
        
        # Ensure variant_lengths is a tensor and move to CPU
        if isinstance(variant_lengths, int):
            variant_lengths = torch.tensor([variant_lengths] * variant_sequences.size(0), dtype=torch.int64).cpu()
        variant_lengths = variant_lengths.cpu()
        
        # Packing variant sequences
        variant_sequences_packed = nn.utils.rnn.pack_padded_sequence(variant_embedded, variant_lengths, batch_first=True, enforce_sorted=False)
        
        # GRU processing
        gru_output_packed, _ = self.gru(variant_sequences_packed, combined_latent)
        gru_output, _ = nn.utils.rnn.pad_packed_sequence(gru_output_packed, batch_first=True)
        
        # Fully connected layer
        output_sequences = self.output_fc(gru_output)
        
        return output_sequences

class AutoencoderRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        
        super(AutoencoderRNN, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers)
        self.decoder = DecoderRNN(hidden_size, hidden_size, output_size, num_layers)
        self.predictor = PredictorNN(hidden_size * 2, hidden_size, output_size)

    def forward(self, variant, variant_lengths, wildtype, wildtype_lengths, seq_len):
        
        variant_latent = self.encoder(variant, variant_lengths)
        wildtype_latent = self.encoder(wildtype, wildtype_lengths) if wildtype is not None else None
        combined_latent = torch.cat((variant_latent, wildtype_latent), dim=1) if wildtype_latent is not None else variant_latent
        predicted_fitness = self.predictor(combined_latent)
        return self.decoder(variant, variant_latent.unsqueeze(0), seq_len), predicted_fitness


class PredictorNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        
        super(PredictorNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
