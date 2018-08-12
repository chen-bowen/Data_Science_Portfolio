from model_spec import *
import argparse

def main():
     args = get_input_args_predict()
     predict(args.path_to_image, args.checkpoint, args.category_names, args. gpu, args.topk)

if __name__ == '__main__':
    main()