import torch
import numpy as np

def greedy_decode(collation_mask, vocab, model, src, memory, max_len, start_symbol, device):
    # function to generate output sequence using greedy algorithm

    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    memory = memory.to(device)

    q_mt = 1.0
    q_mts = []

    for i in range(max_len-1):
        tgt_mask = (collation_mask.generate_square_subsequent_mask(ys.size(0), device).type(torch.bool)).to(device)
        out = model.decode_for_prediction(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(model.softmax(prob), dim=1)

        next_word = next_word.item()
        q_mt *= _.to('cpu').detach().numpy().copy()
        _ = _.to('cpu').detach().numpy().copy()
        _ = np.log2(_[0])

        q_mts.append('{:.10f}'.format(_))

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == vocab.EOS_IDX:
            break

    return ys, q_mts


def greedy_decode_moe(collation_mask, vocab, model, src, memory, max_len, start_symbol, device, num_sim, weights):
    # function to generate output sequence using greedy algorithm
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    print(weights[:,0])

    q_mt = 1.0
    q_mts = []

    for i in range(max_len-1):

        memory = memory.to(device)

        prob = None
        for j in range(0, num_sim):
            current_memory = memory[:,j::num_sim, :]

            tgt_mask = (collation_mask.generate_square_subsequent_mask(ys.size(0), device).type(torch.bool)).to(device)
            out = model.decode_for_prediction(ys, current_memory, tgt_mask)
            out = out.transpose(0, 1)
            if prob == None:
                prob = model.generator(out[:, -1])  #* weights[0,0]
            else:
                prob += model.generator(out[:, -1]) * 0.9**j#* weights[j,0]
        q, next_word = torch.max(model.softmax(prob), dim=1)

        next_word = next_word.item()
        q_mt *= q.to('cpu').detach().numpy().copy()
        q = q.to('cpu').detach().numpy().copy()
        q = np.log2(q[0])

        q_mts.append('{:.10f}'.format(q))

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == vocab.EOS_IDX:
            break

    return ys, q_mts


def greedy_decode_with_simbeam(collation_mask, vocab, model, src, memory, max_len, start_symbol, device, num_sim):
    # function to generate output sequence using greedy algorithm

    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)

    q_mt = 1.0
    q_mts = []

    for i in range(max_len-1):

        memory = memory.to(device)

        q_next_word_list = []

        for j in range(1, num_sim):
            current_memory = memory[:,j::num_sim, :]

            tgt_mask = (collation_mask.generate_square_subsequent_mask(ys.size(0), device).type(torch.bool)).to(device)
            out = model.decode_for_prediction(ys, current_memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            # q, next_word = torch.max(model.softmax(prob), dim=1)
            q_next_word_list.append(torch.max(model.softmax(prob), dim=1))
        
        q_next_word_list.sort(key = lambda x: x[0], reverse=True)
        q = q_next_word_list[0][0]
        next_word = q_next_word_list[0][1]

        next_word = next_word.item()
        q_mt *= q.to('cpu').detach().numpy().copy()
        q = q.to('cpu').detach().numpy().copy()
        q = np.log2(q[0])

        q_mts.append('{:.10f}'.format(q))

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == vocab.EOS_IDX:
            break

    return ys, q_mts