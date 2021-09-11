import torch
import argparse

print('transformer-implementation')
print(torch.__version__)

# 关于metavar参数的测试
parser = argparse.ArgumentParser()
parser.add_argument('--foo')
parser.add_argument('bar')
parser.parse_args('X --foo Y'.split())
parser.print_help()