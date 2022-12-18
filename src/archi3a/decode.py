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
        out = model.decode(ys, memory, tgt_mask)
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


def greedy_decode_moe(collation_mask, vocab, model, src, memory, max_len, start_symbol, device, num_sim):
    # function to generate output sequence using greedy algorithm
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)

    q_mt = 1.0
    q_mts = []

    for i in range(max_len-1):

        memory = memory.to(device)

        prob = None
        for j in range(num_sim):
            current_memory = memory[:,j::num_sim, :]

            tgt_mask = (collation_mask.generate_square_subsequent_mask(ys.size(0), device).type(torch.bool)).to(device)
            out = model.decode(ys, current_memory, tgt_mask)
            out = out.transpose(0, 1)
            if prob == None:
                prob = model.softmax(model.generator(out[:, -1]))
            else:
                prob += model.softmax(model.generator(out[:, -1]))
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