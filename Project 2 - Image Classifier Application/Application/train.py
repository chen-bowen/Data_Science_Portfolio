from model_spec import *
import argparse

def main():
     args = get_input_args_train()
     # get hyperparameters
     hyperparameters =  {'architecture': args.arch, 'epochs': args.epochs, 'print_every': args.print_every, 
                          'hidden_units' : args.hidden_units, 'learning_rate': args.learning_rate, 
                         'dropout_prob': args.dropout_prob}
        
     model = train_model(hyperparameters, data_dir= 'flowers', device = args.gpu)
     return model
if __name__ == '__main__':
    model = main()