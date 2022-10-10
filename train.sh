for init_seed in {0,} # setting random seeds
do
    # training LDLM
    python train.py --epochs 30 \ # total epochs
                    --init_seed $init_seed \ # random seed
                    --postfix "wo_CLANE" \ 
                    --batch_size 48 \
                    --num_head 32 \
                    --backbone 'resnet18' \ # CNN architecture
                    --dropout 0.5 \
                    --num_workers 8 \
                    --d_model 512 \
                    --lr 6e-4 \
                    --weight_decay 1e-2 \
                    --gamma 0.96 \
                    --train_split_file SurvData/Annotations/train.txt \ # modify to your annotation file (training cohort)
                    --valid_split_file SurvData/Annotations/valid.txt \ # modify to your annotation file (internal validation cohort)
                    --data_dir SurvData/Center1Data \ # image folder (training cohort & internal validation cohort - Center 1)
                    --test_data_dir SurvData/Center234Data \ # image folder (external validation cohort - Center 2/3/4)
                    --hpd1_data_dir SurvData/Center1PData \ # image folder (prospective cohort - Center 1)
                    --output_dir "Results" # output folder

    # evaluating LDLM-BS/LDLM-1F/LDLM-2F's performance
    for eval_num_time in {1,2,3}
    do
        python eval.py --eval_num_time $eval_num_time \
                    --init_seed $init_seed \
                    --postfix "wo_CLANE" \
                    --num_head 32 \
                    --backbone 'resnet18' \
                    --num_workers 8 \
                    --d_model 512 \
                    --resume Results/MyModel_resnet18_False_512_32_0.5_${init_seed}_wo_CLANE/best.pth \ # the path of the best trained model
                    --train_split_file SurvData/Annotations/train.txt \
                    --valid_split_file SurvData/Annotations/valid.txt \
                    --data_dir SurvData/Center1Data \
                    --test_data_dir SurvData/Center234Data \
                    --hpd1_data_dir SurvData/Center1PData \
                    --output_dir "Results"
    done
done
