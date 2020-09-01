import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = 2
        self.hidden_size = hidden_size
        self.n_directions = 1
        self.lstm = nn.LSTM(
            input_size=embed_size, 
            hidden_size=hidden_size, 
            num_layers=self.n_layers, 
            batch_first=True, 
            dropout=0.5, 
            bidirectional=self.n_directions > 1
       )
        
        self.word_embedding = nn.Embedding(vocab_size, embed_size, sparse=False)
        self.fc = nn.Linear(self.n_directions * self.hidden_size, vocab_size)
        
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_normal(self.fc.weight)
        self.fc.bias.data.fill_(0.1)
        
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.1)
            elif 'weight' in name:
                nn.init.xavier_normal(param)
        
    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.n_layers*self.n_directions, batch_size, self.hidden_size).cuda()
        cell_state = torch.zeros(self.n_layers*self.n_directions, batch_size, self.hidden_size).cuda()
        hidden = (hidden_state, cell_state)
        return hidden
    
    def forward(self, features, captions):
        """
        features: 
            batch_size, embed_size
        captions: 
            tensor of shape batch_size, seq_len of type Long
        """
        if len(features.shape) == 2:
            batch_size, embed_size = features.shape
            # before concating, we must make sure that features are of shape: batch_size, 1, embed_size
            features = features.unsqueeze(1)  # [:, None, :]
        elif len(features.shape) == 3:
            batch_size, seq_len, embed_size = features.shape
        else:
            raise ValueError()
        h0, c0 = self.init_hidden(batch_size)
        
        # we dont need to input the last caption to the network
        if captions is not None:
            captions = captions[:, :-1]

            # map index in vocab to a vector that represent each word
            # it will get shape: batch_size, seq_len, embed_size
            captions = self.word_embedding(captions)
            if len(captions.shape) == 2:
                captions = captions.unsqueeze(1)

            # concat along seq_len
            #print(features.shape)
            #print(captions.shape)
            x = torch.cat((features, captions), 1)
        else:
            # nothing to be concated, we dont have any captions -> this is the used in inference
            x = features
        # output is of shape (batch_size, seq_len, num_directions * hidden_size)
        lstm_output, (hn, cn) = self.lstm(x, (h0, c0))
        
        # convert to the shape batch_size, seq_len, vocab_size
        output = self.fc(lstm_output)
        
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        """
        inputs: 
            tensor of shape batch_size x embed_size 
        states:
            
        max_len: int
            max length of the output caption
        
        output: 
            list of integers. Each element is the integer of the word in the sentence
        """
        result = []
        
        sos = 0
        eos = 1
        captions = None
        
        for i in range(max_len):
            decoder_output = self.forward(inputs, captions)
            
            # softmax is along the vocab_size dimension so that only one word in the vocab prevails 
            prob = nn.functional.softmax(decoder_output, 2) 
            # shape will batch_size, seq_len
            word_id = torch.argmax(prob, 2)
            
            word_id_int = int(word_id[0, -1])
            result.append(word_id_int)
            if word_id_int == eos:
                break
            
            # add dimension for batch_size
            captions = np.array(result)[None, :]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            captions = torch.Tensor(captions).long().to(device)
        
        return result
    
    def sample_beam_search(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        """
        inputs: 
            tensor of shape batch_size x embed_size 
        states:
            
        max_len: int
            max length of the output caption
        
        output: 
            list of integers. Each element is the integer of the word in the sentence
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # beam width
        b = 10
        
        # sequences of the most likely woords
        results = -1 * np.ones(max_len*b, max_len)
        # corresponding probabilities
        probs = -1 * np.ones(results.shape)
        
        sos = 0
        eos = 1
        
        ########## 1st step ##########
        decoder_output = self.forward(inputs, captions)
        
        # softmax is along the vocab_size dimension so that only one word in the vocab prevails 
        prob = nn.functional.softmax(decoder_output, 2) 
        # shape will batch_size, seq_len
        word_probs, word_idx = torch.topk(prob, k=b, dim=2, largest=True, sorted=True)

        word_probs, word_idx = word_probs[0, -1, :], word_idx[0, -1, :]
        results[:b, 0] = word_idx
        probs[:b, 0] = word_probs

        # add dimension for seq_len
        captions = results[:b, 0].unsqueeze(1)
        captions = torch.Tensor(captions).long().to(device)
        
        for i in range(max_len):
            decoder_output = self.forward(inputs, captions)
            
            # softmax is along the vocab_size dimension so that only one word in the vocab prevails 
            curr_probs = nn.functional.softmax(decoder_output, 2).cpu().detach().numpy()
            
            # TODO: joint probabiliy of top k sequences so far
            prev_seq_probs = probs[...]
            # TODO: get the tuple of length b there each element is the index of the highest new join probs
            arg_max_of_multiply(prev_seq_probs, curr_probs, n=b)
            
            word_id_int = int(word_id[0, -1])
            result.append(word_id_int)
            if word_id_int == eos:
                break
            
            # add dimension for batch_size
            captions = np.array(result)[None, :]
            captions = torch.Tensor(captions).long().to(device)
        
        return result
    
    def arg_max_of_multiply(x, y, n):
        """
        Returns:
            tuple[np.ndarray] -- tuple of ndarray
                each ndarray is index
        """
        assert len(x.shape) == 1
        assert len(y.shape) == 1
        
        res = np.dot(x.squeeze(0), y.squeeze(1))
        flat = res.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, res.shape)