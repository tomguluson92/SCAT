
#1) ablation study with no PL (STB)
python train_coarse.py --hand_choice mano --lr 5e-4 --net reg_transformer_coarse --batch_size 96 --epoch 50 --stage 3 --l_weight_3d 100000 --l_weight_2d 10 --checkpoint_folder experiments/ablation/0309_STB --debug_img reg_0309 --vit_heads 8 --iteration 1 --pos_embed True --vit_dropout 0.0 > debug_logs/0309_STB.log


#2) ablation study with PL (STB)
python train_coarse.py --hand_choice mano --lr 5e-4 --net reg_transformer_coarse --batch_size 96 --epoch 50 --stage 3 --l_weight_3d 100000 --l_weight_2d 10 --checkpoint_folder experiments/ablation/0309_pl_STB --pl_reg True --debug_img reg_0309_pl --vit_heads 8 --iteration 1 --pos_embed True --vit_dropout 0.0 > debug_logs/0309_STB_with_PL.log