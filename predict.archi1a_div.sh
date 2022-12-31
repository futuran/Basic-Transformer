# Testデータを使って全ての類似文で推論
epoch=$1
# for rank in `seq 1 16`; do
#     # ED
#     # DIR=aspec.enja.nfr.ed.enh100k_to_enh100k.top2.archi1a
#     # VALID_SRC=/mnt/work/20221004_RetrieveEditRerank-NMT/data/aspec.ed/merge_enh100k_to_enh100k.top100.div/aspec_dev.en.tkn.bpe.$rank
#     # TEST_SRC=/mnt/work/20221004_RetrieveEditRerank-NMT/data/aspec.ed/merge_enh100k_to_enh100k.top100.div/aspec_test.en.tkn.bpe.$rank

#     # LABSE
#     # DIR=aspec.enja.nfr.labse.enh100k_to_jah2m.top2.archi1a
#     # VALID_SRC=/mnt/work/20221004_RetrieveEditRerank-NMT/data/aspec.labse/merge_enh1m_to_jah2m.top100.div/aspec_dev.en.tkn.bpe.$rank
#     # TEST_SRC=/mnt/work/20221004_RetrieveEditRerank-NMT/data/aspec.labse/merge_enh1m_to_jah2m.top100.div/aspec_test.en.tkn.bpe.$rank

#     # MSBERT
#     # DIR=aspec.enja.nfr.msbert.enh100k_to_jah2m.top2.archi1a
#     # VALID_SRC=/mnt/work/20221004_RetrieveEditRerank-NMT/data/aspec.msbert/merge_enh1m_to_jah2m.top100.div/aspec_dev.en.tkn.bpe.$rank
#     # TEST_SRC=/mnt/work/20221004_RetrieveEditRerank-NMT/data/aspec.msbert/merge_enh1m_to_jah2m.top100.div/aspec_test.en.tkn.bpe.$rank

#     mkdir $DIR/out_test_div/
#     poetry run python src/archi1a/main.py \
#         ex=$DIR \
#         ex.dataset.test.src=$TEST_SRC \
#         ex.load_checkpoint=$DIR/trained_model/model_$epoch.pt \
#         ex.out_txt=$DIR/out_test_div/out_test_model_$epoch.sim$rank.txt \
#         ex.out_lqmt=$DIR/out_test_div/out_test_model_$epoch.sim$rank \
#         do_train=False \
#         do_eval=False \
#         do_predict=True &
# done


# MIX by COMET
for rank in `seq 6 10`; do

    DIR=aspec.enja.nfr.mix_by_comet.top16.top2.archi1a
    mkdir $DIR/out_test_div/

    TEST_SRC=/mnt/work/20221004_RetrieveEditRerank-NMT/data/aspec.ed/merge_enh100k_to_enh100k.top100.div/aspec_test.en.tkn.bpe.$rank
    poetry run python src/archi1a/main.py \
        ex=$DIR \
        ex.dataset.test.src=$TEST_SRC \
        ex.load_checkpoint=$DIR/trained_model/model_$epoch.pt \
        ex.out_txt=$DIR/out_test_div/out_test_model_$epoch.sim$((3*rank-2)).txt \
        ex.out_lqmt=$DIR/out_test_div/out_test_model_$epoch.sim$((3*rank-2)) \
        do_train=False \
        do_eval=False \
        do_predict=True &

    TEST_SRC=/mnt/work/20221004_RetrieveEditRerank-NMT/data/aspec.labse/merge_enh1m_to_jah2m.top100.div/aspec_test.en.tkn.bpe.$rank
    poetry run python src/archi1a/main.py \
        ex=$DIR \
        ex.dataset.test.src=$TEST_SRC \
        ex.load_checkpoint=$DIR/trained_model/model_$epoch.pt \
        ex.out_txt=$DIR/out_test_div/out_test_model_$epoch.sim$((3*rank-1)).txt \
        ex.out_lqmt=$DIR/out_test_div/out_test_model_$epoch.sim$((3*rank-1)) \
        do_train=False \
        do_eval=False \
        do_predict=True &

    TEST_SRC=/mnt/work/20221004_RetrieveEditRerank-NMT/data/aspec.msbert/merge_enh1m_to_jah2m.top100.div/aspec_test.en.tkn.bpe.$rank
    poetry run python src/archi1a/main.py \
        ex=$DIR \
        ex.dataset.test.src=$TEST_SRC \
        ex.load_checkpoint=$DIR/trained_model/model_$epoch.pt \
        ex.out_txt=$DIR/out_test_div/out_test_model_$epoch.sim$((3*rank)).txt \
        ex.out_lqmt=$DIR/out_test_div/out_test_model_$epoch.sim$((3*rank)) \
        do_train=False \
        do_eval=False \
        do_predict=True &
done

