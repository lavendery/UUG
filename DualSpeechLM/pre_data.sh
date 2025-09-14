. ./path.sh

stage=100
stop_stage=1
. utils/parse_options.sh


ngpu=8
TASK='ASR'
train_set=("LibriSpeech_train" "LibriSpeech_test-clean" "LibriSpeech_test-other")
TOKEN_TYPE_ALL=("USTokenizer")
### stage 1-3: data preparation ### for LibriSpeech_train dataset

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Prepare .scp dataset"
    # this part aims to get the information about the dataset. 
    # prepare raw_wav.scp, text.scp
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "split the data for $ngpu GPUs"

    for part in ${train_set[@]}; do
        echo "split $part ... "

        for TOKEN_TYPE in ${TOKEN_TYPE_ALL[@]}; do
            echo "Current $TOKEN_TYPE ... "
            
            mkdir -p data/${TASK}/${part}/${ngpu}splits_${TOKEN_TYPE}
            split_scp=
            for n in `seq 1 $ngpu`; do
                split_scp="$split_scp data/${TASK}/${part}/${ngpu}splits_${TOKEN_TYPE}/raw_wav.${n}.scp"
            done
            utils/split_scp.pl data/${TASK}/${part}/wav.scp $split_scp

            # split text.scp based on raw_wav.scp
            utils/run.pl JOB=1:$ngpu data/${TASK}/${part}/${ngpu}splits_${TOKEN_TYPE}/log/filter_text.JOB.log \
                python3 data_scripts/filter_scp.py \
                data/${TASK}/${part}/${ngpu}splits_${TOKEN_TYPE}/raw_wav.JOB.scp data/${TASK}/${part}/text.scp \
                data/${TASK}/${part}/${ngpu}splits_${TOKEN_TYPE}/text.JOB.scp || exit 1;
        done

    done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Prepare audio sequence and text sequence"

    for part in ${train_set[@]}; do
        echo "prepare $part ... "

        for TOKEN_TYPE in ${TOKEN_TYPE_ALL[@]}; do
            echo "Current $TOKEN_TYPE ... "

            # audio sequence and text sequence
            utils/run.pl JOB=1:$ngpu data/${TASK}/${part}/${ngpu}splits_${TOKEN_TYPE}/log/codec_dump.JOB.log \
                python src/tools/extract_audio_tokens.py \
                --input-text-file  data/${TASK}/${part}/${ngpu}splits_${TOKEN_TYPE}/text.JOB.scp \
                --input-wav-file data/${TASK}/${part}/${ngpu}splits_${TOKEN_TYPE}/raw_wav.JOB.scp \
                --output-file data/${TASK}/${part}/${ngpu}splits_${TOKEN_TYPE}/ \
                --tokenizer ${TOKEN_TYPE} --rank JOB || exit 1;

        done
    done
fi