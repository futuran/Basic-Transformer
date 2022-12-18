from audioop import reverse
import numpy as np
import nltk
import argparse
import math

import csv


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dir', required=True)
    parser.add_argument('-k', '--topk', type=int)
    parser.add_argument('-r', '--ref', default='/mnt/work/20220729_cxmi/data/ecb/ecb_test.fr.tkn')
    parser.add_argument('-s', '--src', default='/mnt/work/20220729_cxmi/data/ecb/ecb_test.en.tkn')
    parser.add_argument('-t', '--tvt', default='test')

    # 利用するランキングスコア
    parser.add_argument('--score', type=int)

    # パラメータ
    parser.add_argument('-pp', '--ppl', type=float)
    parser.add_argument('-pr', type=float)
    parser.add_argument('-pa', type=float)
    parser.add_argument('--epoch', type=int)

    # COMETスコアディレクトリへのパス
    parser.add_argument('--comet')

    args = parser.parse_args()
    return args




def main_old():

    args = parse_args()

    def load_kth(sim_num, tvt):
        with open(args.dir + '/out_{}_model_{}.sim{}.txt'.format(tvt, str(args.epoch), str(sim_num)), 'r') as f:
            out_txts = [l.strip() for l in f]   # 長さ3000のリスト
        with open(args.dir + '/out_{}_model_{}.sim{}.csv'.format(tvt,  str(args.epoch), str(sim_num)), 'r') as f:
            reader = csv.reader(f)
            out_lqmts = [[float(x) for x in l.strip().split(',')] for l in f]  # 長さ3000 * 各単語数の２次元リスト
        with open(args.comet + '/aspec_{}.sim.tkn.bpe.{}.score'.format(tvt, str(sim_num)), 'r') as f:
            comet_scores = [float(l.strip()) for l in f]   # 長さ3000のリスト

        return out_txts, out_lqmts, comet_scores

    with open(args.ref, 'r') as f:
        ref_txt = [l.strip() for l in f]
    with open(args.src, 'r') as f:
        src_txt = [l.strip() for l in f]

    out_txt_list = []           # out_txt_list[topk]
    out_lqmts_list = []         # out_lqmts_list[topk]
    comet_scores_list = []      # comet_scores_list[topk]
    for i in range(args.topk):
        out_txt_tmp, out_lqmts_tmp, comet_scores_tmp = load_kth(i+1,  args.tvt)
        out_txt_list.append(out_txt_tmp)
        out_lqmts_list.append(out_lqmts_tmp)
        comet_scores_list.append(comet_scores_tmp)

    best_ids = []
    for sent_i in range(len(ref_txt)):
        # print(sent_i)
        sent_i_ref = ref_txt[sent_i]
        sent_i_src = src_txt[sent_i]
        sent_i_txt = [out_txt_list[topk][sent_i] for topk in range(args.topk)]
        sent_i_lqmts = [out_lqmts_list[topk][sent_i] for topk in range(args.topk)]
        sent_i_comet_score = [comet_scores_list[topk][sent_i] for topk in range(args.topk)]


        if args.score == 1:
            # Score Q1 (シンプルな対数尤度に基づく手法)
            score_Q = np.array([np.sum(lqmts) for lqmts in sent_i_lqmts])
            best_id = score_Q.argmax(axis=0)
        elif args.score == 2:
            # Score Q2 (train pplに基づく手法)
            score_Q = np.array([np.log2(args.pa)**topk + (args.ppl) * np.log2(len(lqmts)) + np.sum(lqmts) for topk, lqmts in enumerate(sent_i_lqmts)])
            best_id = score_Q.argmax(axis=0)
        elif args.score == 3:
            # Score Q3 (Length Penalthに基づく手法)
            score_Q = np.array([(np.log2(args.pa)**topk + np.sum(lqmts))/(len(lqmts) ** args.pr + 1e-6) for topk, lqmts in enumerate(sent_i_lqmts)])  # 類似度減衰あり
            # print(['{:.10f}'.format(x) for x in score_Q])
            best_id = score_Q.argmax(axis=0)
        elif args.score == 301:
            # Score Q301 (Length Penalthに基づく手法/分母をbpeなしに)
            score_Q = np.array([(np.sum(lqmts))/(len(txt.replace('@@ ', '').split()) ** args.pr + 1e-6) for topk, (lqmts, txt) in enumerate(zip(sent_i_lqmts, sent_i_txt))])  # 類似度減衰あり
            # print(['{:.10f}'.format(x) for x in score_Q])
            best_id = score_Q.argmax(axis=0)
        elif args.score == 23:
            # Score Q2とQ3の組み合わせ
            score_Q = np.array([(np.log2(0.9)**topk + len(lqmts) * np.log2(w) + np.sum(lqmts))/(len(lqmts) ** 2 + 1e-6) for topk, lqmts in enumerate(sent_i_lqmts)])
            best_id = score_Q.argmax(axis=0)
        elif args.score == 4:
            # COMETスコアによる
            # print(sent_i_comet_score)
            score_Q =  np.array(sent_i_comet_score)
            best_id = score_Q.argmax(axis=0)
        elif args.score == 0:
            # sentence-BLEU
            SBLEUs = np.array([nltk.translate.bleu_score.sentence_bleu([sent_i_ref.split()], txt.replace('@@ ', '').split(), weights=(0.25, 0.25, 0.25, 0.25)) for txt in sent_i_txt])
            # print(['{:.10f}'.format(x) for x in SBLEUs])
            best_id = SBLEUs.argmax(axis=0)
        else:
            print('Error')
            exit()

        best_ids.append(best_id)

    with open(args.dir + '/best.{}.txt'.format(args.tvt), 'w') as f:
        for i, idx in enumerate(best_ids):
            f.write(str(out_txt_list[idx][i]) + '\n')


def main():

    args = parse_args()

    def load_kth(sim_num, tvt):
        with open(args.dir + '/out_{}_model_{}.sim{}.txt'.format(tvt, str(args.epoch), str(sim_num)), 'r') as f:
            out_txts = [l.strip() for l in f]   # 長さ3000のリスト
        with open(args.dir + '/out_{}_model_{}.sim{}.csv'.format(tvt,  str(args.epoch), str(sim_num)), 'r') as f:
            reader = csv.reader(f)
            out_lqmts = [[float(x) for x in l.strip().split(',')] for l in f]  # 長さ3000 * 各単語数の２次元リスト

        return out_txts, out_lqmts

    with open(args.ref, 'r') as f:
        ref_txt = [l.strip() for l in f]
    with open(args.src, 'r') as f:
        src_txt = [l.strip() for l in f]

    out_txt_list = []       # out_txt_list[topk]
    out_lqmts_list = []     # out_lqmts_list[topk]
    for i in range(args.topk):
        out_txt_tmp, out_lqmts_tmp = load_kth(i+1,  args.tvt)
        out_txt_list.append(out_txt_tmp)
        out_lqmts_list.append(out_lqmts_tmp)

    best_ids = []
    for sent_i in range(len(ref_txt)):
        # print(sent_i)
        sent_i_ref = ref_txt[sent_i]
        sent_i_src = src_txt[sent_i]
        sent_i_txt = [out_txt_list[topk][sent_i] for topk in range(args.topk)]
        sent_i_lqmts = [out_lqmts_list[topk][sent_i]
                        for topk in range(args.topk)]


        if args.score == 1:
            # Score Q1 (シンプルな対数尤度に基づく手法)
            score_Q = np.array([np.sum(lqmts) for lqmts in sent_i_lqmts])
            best_id = score_Q.argmax(axis=0)
        elif args.score == 2:
            # Score Q2 (train pplに基づく手法)
            score_Q = np.array([np.log2(args.pa)**topk + (args.ppl) * np.log2(len(lqmts)) + np.sum(lqmts) for topk, lqmts in enumerate(sent_i_lqmts)])
            best_id = score_Q.argmax(axis=0)
        elif args.score == 3:
            # Score Q3 (Length Penalthに基づく手法)
            score_Q = np.array([(np.log2(args.pa)**topk + np.sum(lqmts))/(len(lqmts) ** args.pr + 1e-6) for topk, lqmts in enumerate(sent_i_lqmts)])  # 類似度減衰あり
            # print(['{:.10f}'.format(x) for x in score_Q])
            best_id = score_Q.argmax(axis=0)
        elif args.score == 301:
            # Score Q301 (Length Penalthに基づく手法/分母をbpeなしに)
            score_Q = np.array([(np.sum(lqmts))/(len(txt.replace('@@ ', '').split()) ** args.pr + 1e-6) for topk, (lqmts, txt) in enumerate(zip(sent_i_lqmts, sent_i_txt))])  # 類似度減衰あり
            # print(['{:.10f}'.format(x) for x in score_Q])
            best_id = score_Q.argmax(axis=0)
        elif args.score == 23:
            # Score Q2とQ3の組み合わせ
            score_Q = np.array([(np.log2(0.9)**topk + len(lqmts) * np.log2(w) + np.sum(lqmts))/(len(lqmts) ** 2 + 1e-6) for topk, lqmts in enumerate(sent_i_lqmts)])
            best_id = score_Q.argmax(axis=0)
        elif args.score == 0:
            # sentence-BLEU
            SBLEUs = np.array([nltk.translate.bleu_score.sentence_bleu([sent_i_ref.split()], txt.replace('@@ ', '').split(), weights=(0.25, 0.25, 0.25, 0.25)) for txt in sent_i_txt])
            # print(['{:.10f}'.format(x) for x in SBLEUs])
            best_id = SBLEUs.argmax(axis=0)
        else:
            print('Error')
            exit()

        best_ids.append(best_id)

    with open(args.dir + '/best.{}.txt'.format(args.tvt), 'w') as f:
        for i, idx in enumerate(best_ids):
            f.write(str(out_txt_list[idx][i]) + '\n')


if __name__ == '__main__':
    main_old()
