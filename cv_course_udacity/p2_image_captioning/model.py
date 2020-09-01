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
        self.vocab_size = vocab_size
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

    def sample_greedy(self, inputs, states=None, max_len=20):
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        sos = 0
        eos = 1
        word_indices = np.arange(self.vocab_size, dtype="int")
        captions = None
        
        for i in range(max_len):
            decoder_output = self.forward(inputs, captions)
            
            # softmax is along the vocab_size dimension so that only one word in the vocab prevails 
            prob = nn.functional.softmax(decoder_output, 2) 
            last_word_prob = prob[0, -1, :].cpu().detach().numpy()

            word_id_int = int(np.random.choice(word_indices, p=last_word_prob))
            
            result.append(word_id_int)
            if word_id_int == eos:
                break
            
            # add dimension for batch_size
            captions = np.array(result)[None, :]
            captions = torch.Tensor(captions).long().to(device)
        
        return result
    
    def sample(self, inputs, states=None, max_len=20):
        """Beam Search
        
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
        b = 100
        
        # sequences of the most likely woords
        results = -1 * np.ones((max_len*b, max_len), "int")
        # corresponding probabilities
        probs = np.zeros(results.shape)
        
        sos = 0
        eos = 1
        word_indices = np.arange(self.vocab_size, dtype="int")
        
        ########## 1st step ##########
        decoder_output = self.forward(inputs, None)
        
        # softmax is along the vocab_size dimension so that only one word in the vocab prevails 
        prob = nn.functional.softmax(decoder_output, 2) 
        
        last_word_prob = prob[0, -1, :].cpu().detach().numpy()
        word_idx = np.random.choice(word_indices, size=b, p=last_word_prob, replace=False)
        word_probs = last_word_prob[word_idx]
        
        results[:b, 0] = word_idx
        probs[:b, 0] = word_probs

        # add dimension for seq_len
        captions = results[:b, [0]]
        captions = torch.Tensor(captions).long().to(device)
        
        for i in range(1, max_len):
            # print(i)
            # broadcast inputs as well 
            decoder_output = self.forward(torch.cat(b*[inputs]), captions)
            
            # softmax is along the vocab_size dimension so that only one word in the vocab prevails 
            curr_probs = nn.functional.softmax(decoder_output, 2).cpu().detach().numpy()
            # take only the last in the sequence
            curr_probs = curr_probs[:, -1, :]
            
            # joint probabiliy of top k sequences so far
            prev_seq_probs = probs[(i-1)*b:i*b, i-1]
            # get the list of length b where each element is the index of the highest new joint probs
            new_top_b, new_joint_mat = self.arg_max_joint_prob(prev_seq_probs, curr_probs, n=b)
            
            for j, (prev_seq_id, new_word_id) in enumerate(new_top_b):
                # copy the words of the best prev sequence in the new row
                results[i*b+j, :i] = results[(i-1)*b+prev_seq_id, :i]
                # add new word in the new column in the new row
                results[i*b+j, i] = new_word_id
                
                # do the same with probs
                probs[i*b+j, :i] = probs[(i-1)*b+prev_seq_id, :i]
                probs[i*b+j, i] = new_joint_mat[prev_seq_id, new_word_id]
            
#             if word_id_int == eos:
#                 break
            # print("results:\n", results[i*b:(i+1)*b, :i+1])
            # print("probs:\n", probs[i*b:(i+1)*b, :i+1])
            # along the batch_size dimension we put b different options
            captions = results[i*b:(i+1)*b, :i+1]
            # print("captions:\n", captions)
            assert (captions >= 0).all().all()
            captions = torch.Tensor(captions).long().to(device)
            
        # normalize all the probs and get the highest one
        
        alpha = 0.7
        normalized_probs = []
        for i in range(probs.shape[0]):
            seq_len = (i // b) + 1
            real_seq_len = get_seq_len(results[i])
            normalized_probs.append(1.0 / (real_seq_len ** alpha) * np.log(probs[i, real_seq_len-1]))
            
        best_id = np.argmax(normalized_probs)
        best = results[best_id, :get_seq_len(results[best_id])]
        
        cleaned = [int(best[0])]
        for i in range(1, len(best)):
            if best[i] == cleaned[-1]:
                continue
            else:
                cleaned.append(int(best[i]))
        
        return cleaned
    
    def arg_max_joint_prob(self, x, y, n):
        """
        x is joint probabiliy of the sequence so far
        y is the conditional probability of the new word given the seqence so far 
        
        Returns:
            list of indices. each element in the list is x,y loc
        """
        assert len(x.shape) == 1
        assert len(y.shape) == 2
        
        # we only take log of y because x is already in log
        x, y = np.log(x), np.log(y)
        
        x = x[:, None]
        added = np.exp(x + y)
        assert added.shape == y.shape 
        
        added = added / added.sum() # remove the rounding error
        
        flat = added.flatten()
        l = len(flat)
        indices = np.random.choice(l, size=n, p=flat, replace=False)
        
        top_n_indices = np.unravel_index(indices, added.shape)
        
        result = []
        for i1, i2 in zip(*top_n_indices):
            result.append((i1, i2))
            
        return result, added
    
def get_seq_len(sentence):
    l = 0
    for w in sentence:
        l += 1
        if w == 1:
            break
    return l
