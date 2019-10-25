import argparse


def train(args):
    #print(f"training for {args.epochs} epochs with a learning rate = {args.lr}")
    pass

def generate(args):
    #print(f"generate to {args.path}")
    print(args.path)

# create the top-level parser
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

# create the parser for the "foo" command
parser_train = subparsers.add_parser('train')
parser_train.add_argument('-epochs', type=int, default=2)
parser_train.add_argument('-lr', type=float, default=0.001)
parser_train.set_defaults(func=train)

# create the parser for the "bar" command
parser_gen = subparsers.add_parser('generate')
parser_gen.add_argument('-path', type=str, required=True, 
    help='The path to the file where the generated image has to be stored.')
parser_gen.set_defaults(func=generate)

#args = parser.parse_args("train -lr 0.001 -epochs=4".split())
#args = parser.parse_args("generate -path z:/test.jpg".split())
args = parser.parse_args()
args.func(args)
