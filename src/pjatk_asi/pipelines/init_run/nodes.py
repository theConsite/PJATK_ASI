import wandb

def connect_project():
    wandb.init(project="PJATK_ASI")
    wandb.login(key="8a0aeda791abab9d3da3283f4f4129c5d3223aa7")
    return True
