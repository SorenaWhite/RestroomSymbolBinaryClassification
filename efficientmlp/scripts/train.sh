python train.py \
--project mmlrestroomsign\
--compound_coef 1 \
--load_weights /root/autodl-tmp/efficientdet-d1.pth \
--num_workers 12 \
--batch_size 4 \
--num_epochs 500 \
--data_path /root/autodl-tmp/mmlrestroomsign/ \
--log_path /root/autodl-tmp/logs/ \
--saved_path /root/autodl-tmp/output/
