from options import Options
from data.dataloader import load_data
from model.model import Model

def main():
    """ Training
    """
    opt = Options().parse()
    data = load_data(opt)
    model = Model(opt, data)
    model.train()

if __name__ == '__main__':
    main()