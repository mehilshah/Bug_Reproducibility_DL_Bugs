from pytorch_lightning.callbacks import ModelCheckpoint

save_model_path = './'
def checkpoint_callback():
    return ModelCheckpoint(
        filepath= save_model_path,
        save_top_k=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

checkpoint_callback()