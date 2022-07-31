# #2) ablation study with no PL (STB)
# python train.py --hand_choice mano --lr 5e-4 --net reg_transformer --batch_size 96 --epoch 40 --stage 3 --l_weight_3d 100000 --l_weight_2d 10 --checkpoint_folder experiments/ablation/0223_no_pl_STB --debug_img reg_0223_2 --vit_heads 8 --iteration 3 --pos_embed True --vit_dropout 0.0 > debug_logs/0223_STB_PL.log


#3) ablation study with PL (STB)
python train.py --hand_choice mano --lr 5e-4 --net reg_transformer --batch_size 96 --epoch 40 --stage 3 --l_weight_3d 100000 --l_weight_2d 10 --checkpoint_folder experiments/ablation/20220728 --pl_reg True --debug_img 20220728 --vit_heads 8 --iteration 3 --pos_embed True --vit_dropout 0.0 --mask_rate 0.2 > debug_logs/20220728.log