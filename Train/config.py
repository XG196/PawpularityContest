import torch


## For Pawpularity ModelV3
CONFIG = dict(
    use_cuda = True,
    seed = 42,
    backbone = 'swin_base_patch4_window7_224',
    embedder = 'tf_efficientnet_b4_ns',
    train_batch_size = 16,
    valid_batch_size = 32,
    img_size = 224,
    epochs = 20,
    learning_rate = 1e-5,
    scheduler = 'CosineAnnealingLR',
    min_lr = 1e-7,
    T_max = 100,
    T_0 = 20,
#   warmup_epochs = 0,
    weight_decay = 1e-6,
    n_accumulate = 1,
    n_fold = 5,
    num_classes = 1,
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    competition = 'PetFinder',
    _wandb_kernel = 'deb'
)

# CONFIG = dict(
#     use_cuda = True,
#     seed = 42,
#     backbone = 'swin_base_patch4_window7_224',
#     embedder = 'tf_efficientnet_b4_ns',
#     train_batch_size = 16,
#     valid_batch_size = 32,
#     img_size = 224,
#     epochs = 20,
#     learning_rate = 1e-4,
#     scheduler = 'CosineAnnealingLR',
#     min_lr = 1e-6,
#     T_max = 100,
#     T_0 = 20,
# #   warmup_epochs = 0,
#     weight_decay = 1e-6,
#     n_accumulate = 1,
#     n_fold = 5,
#     num_classes = 1,
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
#     competition = 'PetFinder',
#     _wandb_kernel = 'deb'
# )