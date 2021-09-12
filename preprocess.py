import argparse
import tarfile
import urllib.request
import spacy
import os
import sys
from tqdm import tqdm
from learn_bpe import learn_bpe

_TRAIN_DATA_SOURCES = [
    {
        'url': "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz",
        'trg': "news-commentary-v12.de-en.en",
        'src': "news-commentary-v12.de-en.de"
    }
]

_VAL_DATA_SOURCES = [
    {
        'url': "http://data.statmt.org/wmt17/translation-task/dev.tgz",
        'trg': "newstest2013.en",
        'src': "newstest2013.de"
    }
]

_TEST_DATA_SOURCES = [
    {
        'url': "https://storage.googleapis.com/cloud-tpu-test-datasets/transformer_data/newstest2014.tgz",
        'trg': "newstest2014.en",
        'src': "newstest2014.de"
    }
]

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b*bsize - self.n)

def file_exist(dir_name, file_name):
    for sub_dir, _, files in os.walk(dir_name):
        if file_name in files:
            return os.path.join(sub_dir, file_name)
    return None

def download_and_extract(download_dir, url, src_filename, trg_filename):
    src_path = file_exist(download_dir, src_filename)
    trg_path = file_exist(download_dir, trg_filename)
    if src_path and trg_path:
        sys.stderr.write(f'Already downloaded and extracted {url}.\n')
        return src_path, trg_path
    compressed_file = _download_file(download_dir, url)
    sys.stderr.write(f'Extracting {compressed_file}. \n')
    with tarfile.open(compressed_file, 'r:gz') as corpus_tar:
        corpus_tar.extractall(download_dir)
    src_path = file_exist(download_dir, src_filename)
    trg_path = file_exist(download_dir, trg_filename)
    if src_path and trg_path:
        return src_path, trg_path
    raise OSError(f'Download/Extraction failed for url {url} to path {download_dir}')

def _download_file(download_dir, url):
    filename = url.split('/')[-1]
    if file_exist(download_dir, filename):
        sys.stderr.write(f'Already downloaded: {url} (at {filename}).\n')
    else:
        sys.stderr.write(f'Downloading from {url} to {filename}.\n')
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)
    return filename

def get_raw_files(raw_dir, sources):
    raw_files = {'src': [], 'trg': []}
    for d in sources:
        src_file, trg_file = download_and_extract(raw_dir, d['url'], d['src'], d['trg'])
        raw_files['src'].append(src_file)
        raw_files['trg'].append(trg_file)
    return raw_files

def mkdir_if_needed(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

def compile_files(raw_dir, raw_files, prefix):
    src_fpath = os.path.join(raw_dir, f'raw-{prefix}.src')
    trg_fpath = os.path.join(raw_dir, f'raw-{prefix}.trg')
    if os.path.isfile(src_fpath) and os.path.isfile(trg_fpath):
        sys.stderr.write(f'Merged files found, skip the merging process.\n')
        return src_fpath, trg_fpath
    sys.stderr.write(f'Merge files into two files: {src_fpath} and {trg_fpath}.\n')
    with open(src_fpath, 'w') as src_outf, open(trg_fpath, 'w') as trg_outf:
        for src_inf, trg_inf in zip(raw_files['src'], raw_files['trg']):
            sys.stderr.write(f'Input files:\n' f' -SRC:{src_inf}, and\n' f'-TRG:{trg_inf}.\n')
            with open(src_inf, newline='\n') as src_inf , open(trg_inf, newline='\n') as trg_inf:
                cntr = 0
                for i, line in enumerate(src_inf):
                    cntr += 1
                    src_outf.write(line.replace('\r', ' ').strip() + '\n')
                for j, line in enumerate(trg_inf):
                    cntr -= 1
                    trg_outf.write(line.replace('\r', ' ').strip() + '\n')
                assert cntr==0, 'Number of lines in two files are inconsistent.'
    return src_fpath, trg_fpath



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--codes', required=True)
    parser.add_argument('--save_data', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--symbols', '--s', type=int, default=32000, help='Vocabulary size')
    parser.add_argument('--min_frequency', type=int, default=6, metavar='FREQ',
                        help='Stop if no symbol pair has frequency >= FREQ (default: %(default)S)')
    parser.add_argument('--dict_input', action='store_true',
                        help='If set, input file is interpreted as a dictionary where each line contains a word-count pair')
    parser.add_argument('--separator', type=str, default='@@', metavar='STR',
                        help='Separator between non-final subword units (default: %(default)s)')
    parser.add_argument('--total-symbols', '-t', action='store_true')
    args = parser.parse_args()
    print(args)
    # Create folder if needed
    mkdir_if_needed(args.raw_dir)
    mkdir_if_needed(args.data_dir)

    # Download and extract raw data
    raw_train = get_raw_files(args.raw_dir, _TRAIN_DATA_SOURCES)
    raw_val = get_raw_files(args.raw_dir, _VAL_DATA_SOURCES)
    raw_test = get_raw_files(args.raw_dir, _TEST_DATA_SOURCES)

    # Merge files into one.
    train_src, train_trg = compile_files(args.raw_dir, raw_train, args.prefix + '-train')
    val_src, val_trg = compile_files(args.raw_dir, raw_val, args.prefix + '-val')
    test_src, test_trg = compile_files(args.raw_dir, raw_test, args.prefix + '-test')

    # Build up the code from training files if not exist
    args.codes = os.path.join(args.data_dir, args.codes)
    if not os.path.isfile(args.codes):
        sys.stderr.write(f'Collect codes from training data and save to {args.codes}.\n')
        learn_bpe(raw_train['src'], + raw_train['trg'], args.codes, args.symbols, opt.min_frequency, True)
    sys.stderr.write(f'BPE codes prepared.\n')

    sys.stderr.write(f'Build up the tokenizer.\n')
    with codes.open(args.codes, encoding='utf-8') as codes:
        bpe = BPE(codes, separator=args.separator)
    sys.stderr.write(f'Encoding ...\n')





def main_wo_bpe():
    """
    Usage: python reprocess.py -lang_src de -lang_trg en -save_data multi30k_de_en.pkl -share_vocab
    """
    spacy_support_langs = ['de', 'el', 'en', 'es', 'fr', 'it', 'lt', 'nb', 'nl', 'pt']
    parser = argparse.ArgumentParser()
    parser.add_argument('-lang_src', required=True, choices=spacy_support_langs)
    parser.add_argument('-lang_trg', required=True, choices=spacy_support_langs)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-data_src', type=str, default=None)
    parser.add_argument('-data_trg', type=str, default=None)

    parser.add_argument('-max_len', type=int, default=100)
    parser.add_argument('-min_word_count', type=int, default=3)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    # parser.add_argument('-ratio', '--train_valid_test_ratio', type=int, nargs=3, metavar=(8,1,1))
    # parser.add_argument('-vocab', default=None)

    args = parser.parse_args()
    # python中any()函数, 全部都为false则返回false, 有一个为true则返回true
    # python中all()函数, 全部为true时返回true, 有一个为false是则返回false
    # assert断言, 可以在条件不满足程序运行的情况下直接返回错误，
    assert not any([args.data_src, args.data_trg]), 'Custom data input is not support now.'
    assert not any([args.data_src, args.data_trg]) or all([args.data_src, args.data_trg])
    print(args)
    src_lang_model = spacy.load(args.lang_src)
    trg_lang_model = spacy.load(args.lang_trg)
    print('done!')



if __name__ == '__main__':
    # main_wo_bpe()
    main()