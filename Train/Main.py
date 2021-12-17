import pandas as pd
import torch.optim as optim

from config import CONFIG
from utils import *
from TrainModel import run_training
from PawpularityModel import *


ROOT_DIR = "/home/jack/hdd/x227guo/workspace/Kaggle/PawpularityContest/data"
def main():

    df = pd.read_csv(f"{ROOT_DIR}/train.csv")
    df['file_path'] = df['Id'].apply(get_train_file_path)
    feature_cols = [col for col in df.columns if col not in ['Id', 'Pawpularity', 'file_path']]

    df = create_folds(df, n_s=CONFIG['n_fold'], n_grp=14)

    # Create model
    model = PawpularityModelV3(CONFIG['backbone'], CONFIG['embedder'])


    # GPU Training is possible
    if torch.cuda.is_available() and CONFIG['use_cuda']:
        model = model.cuda()

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    # Create train loader and valid loader
    train_loader, valid_loader = prepare_loaders(fold=0, df=df, feature_cols=feature_cols)

    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = fetch_scheduler(optimizer)

    # do training and validation
    model, history = run_training(model, train_loader, valid_loader, 
                                optimizer, scheduler,
                                device=CONFIG['device'],
                                num_epochs=CONFIG['epochs'])

if __name__ == "__main__":


    main()
