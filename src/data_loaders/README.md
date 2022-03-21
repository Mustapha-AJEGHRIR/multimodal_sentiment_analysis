# How to use

## CMU-MOSI

#### Simple use case :
- First you need to execute the script `get_CMU_MOSI.sh` in the data folder
- Then you can import the train val splitter, it outputs the train and val dataloaders
```python
from src.data_loaders.cmu_mosi import get_train_val
train_loader, val_loader = get_train_val()
```
- You can passe what ever argument you want to `get_train_val` to be passed to the dataset loader. Please refer to the classe `CMU_MOSI`'s source code inside the `cmu_mosi.py` file to know more about possible arguments


#### Advanced use case :
- Import the `CMU_MOSI` class directly, it is already a sub class of `torch.utils.data.Dataset`



## IEMOCAP
The data is under a licence, it is not included in the repository.