import configparser, ast
from optparse import OptionParser


def write_config(name, args):
    config = configparser.ConfigParser()
    for k in args.keys():
        config[k] = args[k]

    with open(name, 'w') as configfile:
        config.write(configfile)


def load_config(file_name, section):
    config = configparser.ConfigParser()
    config.read(file_name)
    config.sections()

    d = dict(config._sections[section])
    for k in d.keys():
        try:
            d[k] = ast.literal_eval(d[k])
        except:
            d[k] = d[k]
    return d




