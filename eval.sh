# 2) Ho3D
# python eval.py --net reg_transformer --batch_size 128 --pos_embed True --checkpoint_path_eval experiments/ablation/0220_mjm2/hand_net.pth --vit_heads 8 --iteration 3 --vit_dropout 0.0 --eval_dataset ho3d --result_dir ./experiments/0222_ablation_mjm_0.3/


# 2021.03.09 week10 visualize attention
python eval.py --net reg_transformer_coarse --batch_size 16 --pos_embed True --checkpoint_path_eval experiments/ablation/0309_STB/hand_net_final.pth --vit_heads 8 --iteration 1 --vit_dropout 0.0 --eval_dataset STB --result_dir ./experiments/0309_result_STB/