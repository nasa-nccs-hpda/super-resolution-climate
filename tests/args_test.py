import argparse

cname = "sres"
argparser = argparse.ArgumentParser(description=f'Execute workflow {cname}')
argparser.add_argument('-r', '--refresh', action='store_true')
args: argparse.Namespace = argparser.parse_args()
print( "Refresh = " + str(args.refresh))