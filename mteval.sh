REF=/mnt/work/20221004_RetrieveEditRerank-NMT/data/aspec/aspec_test.ja.tkn
HYPO1=/mnt/work/20221004_RetrieveEditRerank-NMT/Basic-Transformer/aspec.enja.nfr.labse.enh100k_to_jah2m.top2.archi1/out.txt.r
HYPO2=

/mnt/work/20220215_evals/mteval/build/bin/mteval-pairwise -i 1000 -s 100 -e BLEU RIBES -r $REF -h $HYPO1 $HYPO2