ARCHI=archi1a

# ED
DIR=aspec.enja.nfr.ed.enh100k_to_enh100k.top2.archi1a
VALID_SRC=/mnt/work/20221004_RetrieveEditRerank-NMT/data/aspec.ed/merge_enh100k_to_enh100k.top2/aspec_dev.en.tkn.bpe
TEST_SRC=/mnt/work/20221004_RetrieveEditRerank-NMT/data/aspec.ed/merge_enh100k_to_enh100k.top2/aspec_test.en.tkn.bpe

# MSBERT
DIR=aspec.enja.nfr.msbert.enh100k_to_jah2m.top8.archi1a
VALID_SRC=/mnt/work/20221004_RetrieveEditRerank-NMT/data/aspec.msbert/merge_enh1m_to_jah2m.top8/aspec_dev.en.tkn.bpe
TEST_SRC=/mnt/work/20221004_RetrieveEditRerank-NMT/data/aspec.msbert/merge_enh1m_to_jah2m.top8/aspec_test.en.tkn.bpe

# LABSE
# DIR=aspec.enja.nfr.labse.enh100k_to_jah2m.top2.archi1a
# VALID_SRC=/mnt/work/20221004_RetrieveEditRerank-NMT/data/aspec.labse/merge_enh1m_to_jah2m.top4/aspec_dev.en.tkn.bpe
# TEST_SRC=/mnt/work/20221004_RetrieveEditRerank-NMT/data/aspec.labse/merge_enh1m_to_jah2m.top4/aspec_test.en.tkn.bpe

mkdir $DIR/out_dev/
mkdir $DIR/out_test/

if [ $1 = 1 ]; then
    # まずValidデータを使って全てのエポックモデルで推論
    for epoch in `seq 5 20`; do
        # echo $epoch
        poetry run python src/$ARCHI/main.py \
            ex=$DIR \
            ex.dataset.test.src=$VALID_SRC \
            ex.load_checkpoint=$DIR/trained_model/model_$epoch.pt \
            ex.out_txt=$DIR/out_dev/out_dev_model_$epoch.txt \
            ex.out_lqmt=$DIR/out_dev/out_dev_model_$epoch.txt \
            do_train=False \
            do_eval=False \
            do_predict=True &
    done
    
elif [ $1 = 2 ]; then
    for epoch in `seq 5 20`; do
        sh eval_dev.sh $DIR/out_dev/out_dev_model_$epoch.txt
    done

elif [ $1 = 3 ]; then
    # 最も良いものをTestデータで推論
    epoch=$2
    poetry run python src/$ARCHI/main.py \
        ex=$DIR \
        ex.dataset.test.src=$TEST_SRC \
        ex.load_checkpoint=$DIR/trained_model/model_$epoch.pt \
        ex.out_txt=$DIR/out_test/out_test_model_$epoch.txt \
        ex.out_lqmt=$DIR/out_test/out_test_model_$epoch.txt \
        do_train=False \
        do_eval=False \
        do_predict=True
    sh eval_test.sh $DIR/out_test/out_test_model_$epoch.txt

elif [ $1 = 4 ]; then
    # archi3で推論
    epoch=$2
    ARCHI=archi3a
    poetry run python src/$ARCHI/main.py \
        ex=$DIR \
        ex.dataset.test.src=$TEST_SRC \
        ex.load_checkpoint=$DIR/trained_model/model_$epoch.pt \
        ex.out_txt=$DIR/out_test/out_test_model_$epoch.$ARCHI.txt \
        ex.out_lqmt=$DIR/out_test/out_test_model_$epoch.$ARCHI.txt \
        ex.num_sim=8 \
        do_train=False \
        do_eval=False \
        do_predict=True
    sh eval_test.sh $DIR/out_test/out_test_model_$epoch.$ARCHI.txt
fi



