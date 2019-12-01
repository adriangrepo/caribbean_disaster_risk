from fastai.vision import *
from fastai.callbacks import *

data_dir = Path('data')
proc_dir=data_dir/'processing'

class SaveModelCallback(TrackerCallback):
    #"A `TrackerCallback` that saves the model when monitored quantity is best."
    lowest_val_loss=0
    def __init__(self, learn:Learner, monitor:str='valid_loss', mode:str='auto', every:str='improvement', name:str='bestmodel', uid:str=''):
        super().__init__(learn, monitor=monitor, mode=mode)
        self.every,self.name = every,name
        self.uid=uid
        if self.every not in ['improvement', 'epoch']:
            warn(f'SaveModel every {self.every} is invalid, falling back to "improvement".')
            self.every = 'improvement'

    def jump_to_epoch(self, epoch:int)->None:
        try:
            self.learn.load(f'{self.name}_{epoch-1}', purge=False)
            print(f"Loaded {self.name}_{epoch-1}")
        except: print(f'Model {self.name}_{epoch-1} not found.')

    def on_epoch_end(self, epoch:int, **kwargs:Any)->None:
        #"Compare the value monitored to its best score and maybe save the model."
        if self.every=="epoch": self.learn.save(f'{self.name}_{epoch}')
        else: #every="improvement"
            current = self.get_monitor_value()
            if current is not None and self.operator(current, self.best):
                print(f'Epoch {epoch} monitor {self.monitor} current: {current} best: {self.best}')
                self.best = current
                self.lowest_val_loss=current
                self.lowest_model_name=self.name
                self.learn.save(f'{self.name}')
                with open(proc_dir/f'{self.uid}_run.txt', 'a') as run_file:
                    run_file.write(f'{epoch}: {current}: {self.name}\n')

    def on_train_end(self, **kwargs):
        '''Load the best model.
        Load an older model if no improvement
        '''
        if self.every=="improvement" and (self.learn.path/f'{self.learn.model_dir}/{self.name}.pth').is_file():
            self.learn.load(f'{self.name}', purge=False)
            if len(self.learn.recorder.val_losses) != 0:
                valid_loss=self.learn.recorder.val_losses[-1]
                if valid_loss>self.lowest_val_loss:
                    print(f'Loading previous model with loss: {self.lowest_val_loss}: {self.lowest_model_name}.')
                    self.learn.load(f'{self.lowest_model_name}', purge=False)
                    with open(proc_dir/f'{self.uid}_run.txt', 'a') as run_file:
                        run_file.write(f'Loading previous model with loss: {self.lowest_val_loss}: {self.lowest_model_name}\n')

