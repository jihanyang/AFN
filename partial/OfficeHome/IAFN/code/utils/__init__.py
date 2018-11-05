from .data_load import OfficeImage

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)

dataset_length = {
    'Product' : 4439,
    'Product_shared' : 1785,
    'Art' : 2427,
    'Art_shared' : 1089,
    'Real_World' : 4357,
    'Real_World_shared' : 1811,
    'Clipart' : 4365,
    'Clipart_shared' : 1675
}
        
def get_dataset_length(dataset):
    return dataset_length[dataset]

def print_args(args):
    print("==========================================")
    print("==========       CONFIG      =============")
    print("==========================================")
    for arg, content in args.__dict__.items():
        print("{}:{}".format(arg, content))
    print("\n")