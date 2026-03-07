
dataset_name=("KSS")
ckpts=(355000)
speaker=("multi")
metric=("sim" "mos" "wer")

for dn in "${dataset_name[@]}"; do
    for sp in "${speaker[@]}"; do
        for c in "${ckpts[@]}"; do
            echo "Running: python eval_infer_batch.py -d $dn -sp $sp -c $c"
            CUDA_VISIBLE_DEVICES=1 python eval_infer_batch.py -d $dn -sp $sp -c $c

            for m in "${metric[@]}"; do
                echo "Running: python eval_PEFT-TTS_testset.py -m $m -d $dn -sp $sp -c $c"
                python eval_PEFT-TTS_testset.py -m $m -d $dn -sp $sp -c $c
            done
        done
    done
done
å