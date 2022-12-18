TVT=test
DIR=aspec.enja.nfr.ed.enh100k_to_enh100k.top2.archi1a/out_test_div
DIR=aspec.enja.nfr.labse.enh100k_to_jah2m.top2.archi1a/out_test_div
DIR=aspec.enja.nfr.msbert.enh100k_to_jah2m.top2.archi1a/out_test_div
ppl=1

for kt in `seq 1 32`;do
    poetry run python get_best_lqmt.py  --ref /mnt/work/20221004_RetrieveEditRerank-NMT/data/aspec/aspec_test.ja.tkn \
                                        --src /mnt/work/20221004_RetrieveEditRerank-NMT/data/aspec/aspec_test.en.tkn \
                                        --dir $DIR \
                                        --topk $kt \
                                        --tvt $TVT \
                                        --ppl $ppl \
                                        -pa 1 \
                                        -pr 1 \
                                        --score 4 \
                                        --epoch 9 \
                                        --comet /mnt/work/20221004_RetrieveEditRerank-NMT/util-RetrieveEditRerank-NMT/util-comet

    # echo $kt
    sh eval_$TVT.sh $DIR/best.$TVT.txt
done