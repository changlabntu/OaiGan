import argparse
import configparser


def write_config(name, args):
    config = configparser.ConfigParser()
    for k in args.keys():
        config['DEFAULT'][k] = str(args[k])

    with open(name, 'w') as configfile:
        config.write(configfile)


def read_config(name):
    config = configparser.ConfigParser()
    config.read(name)
    opt = dict()
    for k, v in list(config['DEFAULT'].items()):
        opt[k] = v
    opt = argparse.Namespace(**opt)
    return opt