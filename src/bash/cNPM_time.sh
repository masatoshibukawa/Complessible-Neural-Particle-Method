for i in {1..17}
do
    python main.py \
        --database_path ./database \
        --dt 0.01 \
        --t_max 2 \
        --L 0.1 \
        --H 0.2 \
        --dl 0.005 \
        --epoch 30000 \
        --refine_times 4 \
        --hidden_layers 50 50 50 50 \
        --early_stopping_flg 1 \
        --result_directory_name cNPM_time \
        --model cNPM \
        --tl_flg 1
done