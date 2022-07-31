import torch
from .load_STB import get_loader_STB

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

def concat_dataset(batch_size, opt):
    assert opt.stage in [1, 2, 3, 4, 5, 6], "stage must fall in 1, 2, 3, 4, 5, 6"
    
    if opt.stage == 1:
        print("[STAGE][1] pretrain, use Freihand&ho3d dataset.")
        # train_loader = torch.utils.data.DataLoader(
        #     ConcatDataset(
        #         get_loader_frei('training', batch_size, opt),  # freihand
        #         get_loader_ho3d('training', batch_size, opt),  # ho3d
        #     ),
        #     batch_size=batch_size, shuffle=True,drop_last=True,
        #     num_workers=10, pin_memory=False,
        # )
    elif opt.stage == 2:
        print("[STAGE][2] finetune, use RHD, STB, freiHand, STB, MHP.")
        # train_loader = torch.utils.data.DataLoader(
        #     ConcatDataset(
        #         get_loader_frei('training', batch_size, opt),  # freihand
        #         get_loader_ho3d('training', batch_size, opt),  # ho3d
        #         get_loader_RHD('training', batch_size, opt),  # RHD
        #         get_loader_STB('training', batch_size, opt),  # STB
        #         get_loader_MHP('training', batch_size, opt)   # MHP
        #     ),
        #     batch_size=batch_size, shuffle=True, drop_last=True,
        #     num_workers=10, pin_memory=False,
        # )
    elif opt.stage == 3:
        print("[STAGE][3] finish, finetune.")
        train_loader = torch.utils.data.DataLoader(
            ConcatDataset(
                get_loader_STB('training', batch_size, opt),  # STB
            ),
            batch_size=batch_size, shuffle=True,drop_last=True,
            num_workers=10, pin_memory=False,
        )
    elif opt.stage == 4:
        print("[STAGE][4] Ablation Study on freiHand.")
        # train_loader = torch.utils.data.DataLoader(
        #     ConcatDataset(
        #         get_loader_frei('training', batch_size, opt),
        #     ),
        #     batch_size=batch_size, shuffle=True,drop_last=True,
        #     num_workers=10, pin_memory=False,
        # )
    elif opt.stage == 5:
        print("[STAGE][5] Ablation Study on Ho-3D.")
        # train_loader = torch.utils.data.DataLoader(
        #     ConcatDataset(
        #         get_loader_ho3d('training', batch_size, opt),
        #     ),
        #     batch_size=batch_size, shuffle=True,drop_last=True,
        #     num_workers=10, pin_memory=False,
        # )
    elif opt.stage == 6:
        print("[STAGE][6] Ablation Study on MHP.")
        # train_loader = torch.utils.data.DataLoader(
        #     ConcatDataset(
        #         get_loader_MHP('training', batch_size, opt),
        #     ),
        #     batch_size=batch_size, shuffle=True,drop_last=True,
        #     num_workers=10, pin_memory=False,
        # )
    
    
    return train_loader
