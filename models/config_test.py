import argparse


class BaseOptions():

    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--resume', type=bool, default=False, help='resume on pretrained model')
        parser.add_argument('--checkpoint_hand', required=False, default='hand_net.pth',
                            help='Path to pretrained checkpoint')
        parser.add_argument('--checkpoint_folder', required=False, default='experiments/0131_224_smplx')
        
        # MANO mean shape的inside or outside
        parser.add_argument('--outside',type=bool,default=True,help='palm or back of hand joints xyz.')   
        
        #ViT options
        parser.add_argument('--net',type=str,default="ViT",help='ViT or ViP.')
        parser.add_argument('--vit_dim',type=int,default=256)
        parser.add_argument('--vit_depth',type=int,default=3)
        parser.add_argument('--vit_heads',type=int,default=4)
        parser.add_argument('--vit_mlp_dim',type=int,default=512)
        parser.add_argument('--vit_dropout',type=float,default=0.0)
        
        #Transformer替代FC为regressor
        parser.add_argument('--feature',type=str,default='x2', help='EncoderTransformer feature level x2 or x3, [x2, x3]')
        
        #visualization options
        parser.add_argument('--debug_img',type=str,default='single',help='Choose name for debug_gt_pred_{debug_img}.png .')
        
        #train options
        parser.add_argument('--batch_size',type=int,default=32,
                            help='batch size setting')
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--epoch', type=int, default=50)
        parser.add_argument('--iteration', type=int, default=1, help='regressor iterations.')
        parser.add_argument('--stage', type=int, default=1, help='there are 3 stage: 1(for better theta), 2(train on all datasets), 3(finetune on specific dataset)')
        parser.add_argument('--order',type=str,default='SMPLX',help='Choose MANO or SMPLX.')
        parser.add_argument('--hand_choice',type=str,default='mano',help='Choose mano or smplx.')
        parser.add_argument('--smplx_model_path',type=str,default='extra_data/SMPLX_NEUTRAL.pkl')
        parser.add_argument('--mean_mano_param', type=str, default='extra_data/mean_mano_params.pkl')
        parser.add_argument('--smplx_hand_info_file',type=str,default='extra_data/SMPLX_HAND_INFO.pkl')
        parser.add_argument('--right_hand_model',type=str,default='extra_data/MANO_RIGHT.pkl')
        parser.add_argument('--use_heatmap',type=bool,default=False)
        parser.add_argument('--freeze',type=bool,default=False)
        parser.add_argument('--debug',type=bool,default=True)
        
        # ablation study. 2021.02.05 week5 星期五
        parser.add_argument('--mask_rate',type=float,default=0.0,help='mask ratio')
        parser.add_argument('--pos_embed',type=bool,default=False,help='Positional encoding')
        
        
        #loss weight
        parser.add_argument('--l_weight_3d', type=float, default=0.0) #10000.0
        parser.add_argument('--l_weight_2d', type=float, default=0.0) #10.0
        
        #data augmentation
        parser.add_argument('--motion_blur',type=bool,default=False)
        parser.add_argument('--rotation',type=bool,default=False)
        
        #eval options
        parser.add_argument('--eval_dataset', type=str, default='STB', help='choose:STB, frei, ho3d.')
        parser.add_argument('--result_dir', type=str, default='./output/eval_0219/', help='Path to save eval result')
        parser.add_argument('--checkpoint_path_eval', required=False, default='experiments/0207_iccv_1/hand_net_final.pth',
                            help='Path to pretrained checkpoint')


        self.parser = parser

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
    
    def parse_jupyter(self):
        self.opt = self.parser.parse_args([])
        return self.opt