import wandb

def connect_project():
    wandb.login(key="8a0aeda791abab9d3da3283f4f4129c5d3223aa7")
    wandb.init(project="PJATK_ASI",  job_type="train")
    return True
