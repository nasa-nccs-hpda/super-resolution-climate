import argparse

cname = "sres"
argparser = argparse.ArgumentParser(description=f'Execute workflow {cname}')
argparser.add_argument('-r', '--refresh', action='store_true')
print(argparser.parse_args())