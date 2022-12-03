ref=/mnt/work/20221004_RetrieveEditRerank-NMT/data/aspec/aspec_dev.ja.tkn

outfile=$1
sed -r 's/(@@ )|(@@ ?$)//g' $outfile > $outfile.r
poetry run sacrebleu $ref -i $outfile.r -b --force

