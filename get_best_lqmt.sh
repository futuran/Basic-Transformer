TVT=test
DIR=aspec.enja.nfr.ed.enh100k_to_enh100k.top2.archi1a/out_test_div
# DIR=aspec.enja.nfr.labse.enh100k_to_jah2m.top2.archi1a/out_test_div
# DIR=aspec.enja.nfr.msbert.enh100k_to_jah2m.top2.archi1a/out_test_div
# DIR=aspec.enja.nfr.mix_by_comet.top16.top2.archi1a/out_test_div

ppl=1
# ED:10
# LABSE:8
# MSBERT:9
# MIX by COMET:10

for kt in `seq 1 32`;do
    # echo $kt
    poetry run python get_best_lqmt.py  --ref /mnt/work/20221004_RetrieveEditRerank-NMT/data/aspec/aspec_test.ja.tkn \
                                        --src /mnt/work/20221004_RetrieveEditRerank-NMT/data/aspec/aspec_test.en.tkn \
                                        --dir $DIR \
                                        --topk $kt \
                                        --tvt $TVT \
                                        --score 998 \
                                        --epoch 10

    sh eval_$TVT.sh $DIR/best.$TVT.txt
done