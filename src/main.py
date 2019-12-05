"""Running the walker machine."""

from helper import parameter_parser
from walklets import WalkletMachine

def main(args):
    """
    Walklet machine calling wrapper
    :param args: Arguments object parsed up.
    """
    WalkletMachine(args)

if __name__ == "__main__":
    args = parameter_parser()
    main(args)
