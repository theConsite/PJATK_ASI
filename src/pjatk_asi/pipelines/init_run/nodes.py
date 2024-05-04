import wandb

def connect_project():
    wandb.init(project="PJATK_ASI_caret")
    return True
