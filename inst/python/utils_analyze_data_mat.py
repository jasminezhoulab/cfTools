#
# Python 3.x
#
import sys, gzip, re, string
import numpy as np
from scipy import sparse
from scipy.sparse import lil_matrix, csc_matrix
# import pandas as pd
from datetime import datetime
from itertools import combinations
# import collections
from collections import Counter
from operator import itemgetter
# from skfeature.function.similarity_based import fisher_score
# from skfeature.function.statistical_based import gini_index
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from matplotlib.backends.backend_pdf import PdfPages
# https://matplotlib.org/stable/tutorials/introductory/usage.html#the-builtin-backends
# import matplotlib as mpl
# mpl.use('ipympl')
# import matplotlib.colors


# Return the first number that satifies the requirement:
# E.g.: extract_number_after_a_substring('hypermarker_Nsf0.95Nrf1Na0.2Tsf1Trf0.2Ta1cov10cpg6Nsf0.76', 'Nsf')
#       returns '0.95', not '0.76'.
def extract_number_after_a_substring(str_, substr_):
    a = re.findall(r'%s([-+]?\d*\.\d+|\d+)'%substr_, str_)
    if len(a)>=1:
        return(a[0])
    else:
        return(None)

# Return the first number that satifies the requirement:
# The 1st number after substr_ must start from '+', and the second number must start from '-', e.g., '+0.1-0.7' or '+1-1'.
# E.g.: extract_number_after_a_substring('hyper.alpha.samplesetfreq.thresholds.n2.p4.minreadfrac+0.1-0.7', 'minreadfrac')
#       returns ('+0.1', '-0.7').
def extract_two_numbers_after_a_substring(str_, substr_):
    a = re.findall(r'%s([+]?\d*\.\d+|[+]?\d+)([-]?\d*\.\d+|[-]?\d+)'%substr_, str_)
    if len(a)>=1:
        return(a[0])
    else:
        return(None)

# Return the range that is after a substring, where range is defined as 'lower_upper', e.g. '0.8_1'
# E.g., extract_range_after_a_substring('an0.2.atrange0.8_1', 'atrange')
#       returns ('0.8', '1')
def extract_range_after_a_substring(str_, substr_):
    a = re.findall(r'%s([+-]?\d*\.\d+|[+-]?\d+)_([+-]?\d*\.\d+|[+-]?\d+)'%substr_, str_) # return [('0.8', '1')] for str_='an0.2.atrange0.8_1' and substr_='atrange'
    if len(a)>=1:
        return(a[0])
    else:
        return((None, None))

def remove_substring_followed_by_two_floats(str_, substr_):
    str_removed = re.sub(r"%s([+-]?\d*\.\d+|[+-]?\d+)([+-]?\d*\.\d+|[+-]?\d+)"%substr_, "", str_)
    return(str_removed)


# https://stackoverflow.com/questions/12929308/python-regular-expression-that-matches-floating-point-numbers
# For example, remove_substring_followed_by_float('hyper.alpha.samplesetfreq.thresholds.n2.p5.minreadfrac0.1', '.minreadfrac')
# returns 'hyper.alpha.samplesetfreq.thresholds.n2.p5'
def remove_substring_followed_by_float(str_, substr_):
    str_removed = re.sub(r"%s[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?"%substr_, "", str_)
    return(str_removed)

def remove_substring_followed_by_a_substring(str_, substr_):
    str_removed = re.sub(r"%s.*"%substr_, "", str_)
    return (str_removed)

# get the second minimum of a numpy array
def min_2nd(a):
    return(np.amin(a[a != np.amin(a)]))

# mat: a numpy 2D array, nrow X ncol
# rows_factor: a numpy 1D array, nrow X 1
# Perform operation:
#   a is mat, b is rows_factor:
#   a= np.array([[1, 2], [3, 4]])
#         array([[1, 2],
#                [3, 4]])
#   b= np.array([10, 20])
#         array([[10],
#                [20]])
#   a*b =
#         array([[10, 20],
#                [60, 80]])
# Output:
#   new_mat: a dense matrix
def normalize_matrix_by_rows(mat, rows_factor):
    mat_type = str(type(mat))
    if ('lil_matrix' in mat_type) or ('csc_matrix' in mat_type) or ('csr_matrix' in mat_type) or (
            'coo_matrix' in mat_type) or ('bsr_matrix' in mat_type):
        new_mat = mat.toarray() * rows_factor
    else:
        new_mat = mat * rows_factor
    return(new_mat)

# All values > cutoff are set to 1, otherwise 0
# mat is a 2D numpy.array or a scipy.sparse matrix
def binarize_matrix(mat, cutoff=0):
    mat_type = str(type(mat))
    if ('csc_matrix' in mat_type) or ('csr_matrix' in mat_type) or (
            'coo_matrix' in mat_type) or (
            'bsr_matrix' in mat_type):
        nnz_inds = mat.nonzero()
        if len(nnz_inds[0])==0:
            # All values are zero. So no need to binarize matrix. Keep unchanged.
            return(mat)
        if cutoff==0:
            mat_new = sparse.csr_matrix((np.ones(len(nnz_inds[0])), (nnz_inds[0], nnz_inds[1])), shape=mat.shape, dtype=mat.dtype)
        else:
            keep = np.where(mat.data>cutoff)[0]
            n_keep = len(keep)
            if n_keep == 0:
                # No values > cutoff. Keep unchanged.
                return(mat)
            else:
                mat_new = sparse.csr_matrix((np.ones(n_keep), (nnz_inds[0][keep], nnz_inds[1][keep])), shape=mat.shape)
        return(mat_new)
    else:
        mat[mat<=cutoff] = 0
        mat[mat>cutoff] = 1
    return(mat)

def remove_columns_with_zero_sums(mat):
    column_sums = mat.sum(axis=0)
    column_indexes_removed = np.where(column_sums == 0)[0]
    _, num_column = mat.shape
    columns_indexes_kept = np.arange(num_column)
    if len(column_indexes_removed) > 0:
        columns_indexes_kept = np.setdiff1d(columns_indexes_kept, column_indexes_removed)
        return( (np.delete(mat, column_indexes_removed, axis=1), columns_indexes_kept) )
    else:
        return( (mat, columns_indexes_kept) )


# input_vec_file format (nrow lines and each line has 2 columns): sample_name value
# Output vector: a 1D numpy array, 1 x nrow
def read_vector_file_with_row_labels(input_vec_file):
    vec = []
    samples_list = []
    sample2value = {}
    if input_vec_file == 'stdin':
        fin = sys.stdin
    elif input_vec_file.endswith('gz'):
        fin = gzip.open(input_vec_file, 'rt')
    else:
        fin = open(input_vec_file, 'rt')
    for line in fin:
        items = line.rstrip().split('\t')
        if len(items) != 2:
            sys.stderr.write('Error (read_vector_file): No two items in the lines.\nExit.\n')
            sys.exit(-1)
        value = float(items[1])
        samples_list.append( items[0] )
        vec.append( value )
        sample2value[items[0]] = value
    if input_vec_file != 'stdin':
        fin.close()
    return(({'samples':samples_list, 'values':np.array(vec)}, sample2value))


# input_mat_file (nrow lines and each line has 1+ncol columns): sample_name value_1 value_2 ...
# rows_factor: a numpy 1D array, size of nrow
# Output matrix file: format is the same as input_mat_file
def read_and_normalize_and_output_matrix_by_rows(input_mat_file, sample2factor, output_normalized_matrix_file):
    if input_mat_file == 'stdin':
        fin = sys.stdin
    elif input_mat_file.endswith('gz'):
        fin = gzip.open(input_mat_file, 'rt')
    else:
        fin = open(input_mat_file, 'rt')

    if output_normalized_matrix_file == 'stdout':
        fout = sys.stdout
    elif output_normalized_matrix_file.endswith('gz'):
        fout = gzip.open(output_normalized_matrix_file, 'wt')
    else:
        fout = open(output_normalized_matrix_file, 'wt')

    i = 0
    for line in fin:
        items = line.rstrip().replace('NA', 'nan').split('\t')
        sample_name = items.pop(0)
        if sample_name not in sample2factor:
            sys.stderr.write("Error (read_and_normalize_and_output_matrix_by_rows): %s does not have factor for normalization!\n"%sample_name)
            sys.exit(-1)
        profile = np.array(list(map(float, items)))
        profile *= sample2factor[sample_name]

        profile_str = '\t'.join(['%.3g'%v for v in profile]).replace('nan', 'NA')
        fout.write('%s\t%s\n'%(sample_name, profile_str))

        i += 1

    if output_normalized_matrix_file != 'stdout':
        fout.close()

    if input_mat_file != 'stdin':
        fin.close()


def get_line_number(file, exclude_first_header_line=False):
    if file.endswith('gz'):
        f = gzip.open(file, 'rt')
    else:
        f = open(file, 'rt')
    if exclude_first_header_line:
        next(f)
    n = 0
    for line in f:
        n += 1
    return(n)

# file is tab-delimited
# column_index_for_filter: index is 1-based.
# min_column_value_to_keep_line: the float
def get_line_number_with_a_column_value_filtering(file,
                                                  exclude_first_header_line=False,
                                                  column_index_for_filter=1,
                                                  min_column_value_to_keep_line=0):
    column_index_for_filter -= 1 # make index 0-based
    if file.endswith('gz'):
        f = gzip.open(file, 'rt')
    else:
        f = open(file, 'rt')
    if exclude_first_header_line:
        next(f)
    n = 0
    for line in f:
        items = line.rstrip().split('\t')
        try:
            if float(items[column_index_for_filter]) >= min_column_value_to_keep_line:
                n += 1
        except IndexError as ve:
            sys.stderr.write('%s: %s\n'%(type(ve), ve))
            sys.stderr.write('Error: column_index_for_filter (column %d) exceeds the column number (%d) in the file.\nExit.\n'%(column_index_for_filter+1, len(items)))
            sys.exit(-1)
        except ValueError as ve:
            sys.stderr.write('%s: %s\n'%(type(ve), ve))
            sys.stderr.write('Error: column %d is not number in the file.\nExit.\n'%(column_index_for_filter+1))
            sys.exit(-1)
    return(n)

# file is tab-delimited
# column1_index_for_filter: index is 1-based.
# min_column1_value_to_keep_line: the float
# column2_index_for_filter: index is 1-based.
# min_column2_value_to_keep_line: the float
def get_line_number_with_two_columns_values_filtering(file,
                                                      exclude_first_header_line=False,
                                                      column1_index_for_filter=1,
                                                      min_column1_value_to_keep_line=0,
                                                      column2_index_for_filter=1,
                                                      min_column2_value_to_keep_line=0):
    column1_index_for_filter -= 1 # make index 0-based
    column2_index_for_filter -= 1  # make index 0-based
    if file.endswith('gz'):
        f = gzip.open(file, 'rt')
    else:
        f = open(file, 'rt')
    if exclude_first_header_line:
        next(f)
    n = 0
    for line in f:
        items = line.rstrip().split('\t')
        try:
            if (float(items[column1_index_for_filter]) >= min_column1_value_to_keep_line) and \
                    (float(items[column2_index_for_filter]) >= min_column2_value_to_keep_line):
                n += 1
        except IndexError as ve:
            sys.stderr.write('%s: %s\n'%(type(ve), ve))
            sys.stderr.write('Error: column_index_for_filter (column %d or %d) exceeds the column number (%d) in the file.\nExit.\n'%(column1_index_for_filter+1,column2_index_for_filter+1, len(items)))
            sys.exit(-1)
        except ValueError as ve:
            sys.stderr.write('%s: %s\n'%(type(ve), ve))
            sys.stderr.write('Error: column %d or %d is not number in the file.\nExit.\n'%(column1_index_for_filter+1,column2_index_for_filter+1))
            sys.exit(-1)
    return(n)

# file: a tab-delimited text file (may be gzipped)
# column_index: 1-based column index
def get_specific_column_of_tab_file(file, column_index, has_header=True):
    if file.endswith('gz'):
        fid = gzip.open(file, 'rt')
    else:
        fid = open(file, 'rt')
    if has_header:
        fid.readline()
    column_index -= 1 # Make index 0-based
    ret_list = []
    for line in fid:
        try:
            items = line.rstrip().split('\t')
            v = items[column_index]
            ret_list.append(v)
        except KeyError:
            sys.stderr.write('KeyError: No column %d (1-based) in the file!\n  file: %s\nExit.\n'%(column_index+1, file))
            sys.exit(-1)
    fid.close()
    return(ret_list)

# return (l1_unique_list, l2_unique_list, l1_l2_common_list)
def diff_common_of_two_lists(l1, l2):
    set1 = set(l1)
    set2 = set(l2)
    return( (list(set1.difference(set2)),
             list(set2.difference(set1)),
             list(set1.intersection(set2))) )

# return (l1_unique_list, l2_unique_list, l1_l2_common_list)
def common_of_multi_lists(list_of_lists):
    n_lists = len(list_of_lists)
    if n_lists >= 2:
        common_set = []
        for i in range(n_lists):
            if i == 0:
                common_set = set(list_of_lists[i])
            else:
                common_set = common_set.intersection(set(list_of_lists[i]))
        common_set = list(common_set)
    elif n_lists == 1:
        common_set = list_of_lists[0]
    else:
        common_set = []
    return(common_set)

# return unique lines of file1, where unique line is defined as the line of the first column exists in file1, not in file2.
def diff_of_file1_and_file2(file1, file2, header_line=True):
    if file1.endswith('gz'):
        f1 = gzip.open(file1, 'rt')
    else:
        f1 = open(file1, 'rt')
    if file2.endswith('gz'):
        f2 = gzip.open(file2, 'rt')
    else:
        f2 = open(file2, 'rt')
    header_line_str = ''
    if header_line:
        header_line_str = next(f1).rstrip()
    lines1 = {}
    for line in f1:
        items = line.rstrip().split('\t')
        lines1[items[0]] = line.rstrip()
    lines2 = {}
    for line in f2:
        items = line.rstrip().split('\t')
        lines2[items[0]] = line.rstrip()
    lines1_keys_unique = sorted(list(set(lines1.keys()) - set(lines2.keys())))
    if len(lines1_keys_unique)==0:
        lines1_unique = []
    else:
        lines1_unique = [lines1[key] for key in lines1_keys_unique]
    f1.close()
    f2.close()
    return( (lines1_unique, header_line_str) )

# sort a list in increasing order and return index of a sorted list
def sort_list_and_return_sorted_index(l):
    if len(l)==0:
        return([])
    else:
        return(sorted(range(len(l)), key=lambda k: l[k]))

# https://devenum.com/merge-python-dictionaries-add-values-of-common-keys/
def mergeDict_by_adding_values_of_common_keys(dict1, dict2):
   return(dict(Counter(dict1) + Counter(dict2)))


# https://stackoverflow.com/questions/26910708/merging-dictionary-value-lists-in-python
# https://stackoverflow.com/questions/52562882/merge-two-dictionaries-and-keep-the-values-for-duplicate-keys-in-python
# dict1 and dict2: two dictionaries with values being sets.
# For example:
# dict1 = {'a': {2}, 'b': {3}, 'c': {55}}
# dict2 = {'a': {22}, 'b': {33}}
def mergeDict_by_union_valuesets_of_common_keys(dict1, dict2):
    a = dict(dict1)
    for k, v in dict2.items():
        a[k] = a[k].union(v) if k in a else v
    return(a)


# https://stackoverflow.com/questions/26910708/merging-dictionary-value-lists-in-python
# https://stackoverflow.com/questions/52562882/merge-two-dictionaries-and-keep-the-values-for-duplicate-keys-in-python
# dict1 and dict2: two dictionaries with values being sets.
# For example:
# dict1 = {'a': {2}, 'b': {3}, 'c': {55}}
# dict2 = {'a': {22}, 'b': {33}}
def mergeDict_by_union_valuelists_of_common_keys(dict1, dict2):
    a = dict(dict1)
    for k, v in dict2.items():
        a[k] = a[k] + v if k in a else v
    return(a)


# Input: vec is a 1D numpy.array
# Output: #non_zeros, #nan, #zeros, vector_length, fraction of non_zeros, fraction of nan, fraction of zeros
def summary_of_vector(vec):
    nan_indicator = np.isnan(vec)
    count_nan = nan_indicator.sum()
    vec_nonnan = vec[~nan_indicator]
    count_nnz = np.count_nonzero(vec_nonnan)
    size_ = float(len(vec))
    count_zero = np.count_nonzero(vec_nonnan==0)
    return( (count_nnz, count_nan, count_zero, int(size_), count_nnz/size_, count_nan/size_, count_zero/size_) )


def prepro_impute_na_by_zero(mat):
    mat_type = str(type(mat))
    if ('csc_matrix' in mat_type) or ('csr_matrix' in mat_type) or ('coo_matrix' in mat_type) or ('bsr_matrix' in mat_type):
        print('     #prepro_impute_na_by_zero (%d x %d)' % (mat.get_shape()))
        mat.data = np.nan_to_num(mat.data, nan=0) # lil_matrix cannot use this way to replace nan with zero
    else:
        print('     #prepro_impute_na_by_zero (%d x %d)' % (mat.shape))
        mat[np.isnan(mat)] = 0
    return (mat)


#
# assume all elements of sublist_ are also in list_
#
def get_indexes(sublist_, list_):
    return (np.array([list_.index(x) if x in list_ else np.nan for x in sublist_]))

def read_lines(file):
    lines = []
    if file == 'stdin':
        f = sys.stdin
    elif file.endswith('gz'):
        f = gzip.open(file, 'rt')
    else:
        f = open(file, 'rt')
    for line in f:
        lines.append(line.rstrip())
    if file != 'stdin':
        f.close()
    return(lines)


def read_lines_with_two_columns(file, delimiter='\t'):
    column1 = []
    column2 = []
    if file.endswith('gz'):
        f = gzip.open(file, 'rt')
    else:
        f = open(file, 'rt')
    i = 0
    for line in f:
        i += 1
        items = line.rstrip().split(delimiter)
        try:
            column1.append(items[0])
            column2.append(items[1])
        except IndexError:
            sys.stderr.write('Error: line %d of file does not have two columns\n  %s\nExit.\n'%(i, file))
    f.close()
    return( (column1, column2) )


def read_lines_with_first_header(file):
    lines = []
    if file.endswith('gz'):
        f = gzip.open(file, 'rt')
    else:
        f = open(file, 'rt')
    header_line = next(f).rstrip()
    for line in f:
        lines.append(line.rstrip())
    f.close()
    return( (lines, header_line) )

def read_lines_with_first_header_and_filter_lines_with_column1_number(file, upper_bound_int_of_column1=1045106):
    lines = []
    if file.endswith('gz'):
        f = gzip.open(file, 'rt')
    else:
        f = open(file, 'rt')
    header_line = next(f).rstrip()
    for line in f:
        value_of_column1 = int(line.rstrip().split('\t')[0])
        if value_of_column1 <= upper_bound_int_of_column1:
            lines.append(line.rstrip())
    f.close()
    return( (lines, header_line) )

def write_lines(fout, lines, header_line=None):
    if header_line is not None:
        fout.write('%s\n'%header_line)
    for line in lines:
        fout.write('%s\n'%line)

def count_nonzero_nonnan_max_of_rows(mat):
    _, ncol = mat.shape
    mat_type = str(type(mat))
    if ('lil_matrix' in mat_type) or ('csc_matrix' in mat_type) or ('csr_matrix' in mat_type) or (
            'coo_matrix' in mat_type) or ('bsr_matrix' in mat_type):
        mat_np_array = mat.toarray()
        nonzero_counts_of_rows = np.count_nonzero(mat_np_array > 0, axis=1)
        nonnan_counts_of_rows = np.count_nonzero(~np.isnan(mat_np_array), axis=1)
        max_of_rows = np.nanmax(mat_np_array, axis=1)
    elif 'numpy.ndarray' in mat_type:
        nonzero_counts_of_rows = np.count_nonzero(mat>0, axis=1)
        nonnan_counts_of_rows = np.count_nonzero(~np.isnan(mat), axis=1)
        max_of_rows = np.nanmax(mat, axis=1)
    nonzero_frac_of_rows = nonzero_counts_of_rows / float(ncol)
    nonnan_frac_of_rows = nonnan_counts_of_rows / float(ncol)

    return((nonzero_counts_of_rows, nonzero_frac_of_rows, nonnan_counts_of_rows, nonnan_frac_of_rows, max_of_rows))

def get_columns_index_with_nonzeros(mat):
    nrow, _ = mat.shape
    mat_type = str(type(mat))
    if ('lil_matrix' in mat_type) or ('csc_matrix' in mat_type) or ('csr_matrix' in mat_type) or ('coo_matrix' in mat_type) or ('bsr_matrix' in mat_type):
        nonzero_counts_of_columns = np.count_nonzero(mat.toarray() > 0, axis=0)  # mat>0 exclude both 0 and np.nan
    elif 'numpy.ndarray' in mat_type:
        nonzero_counts_of_columns = np.count_nonzero(mat>0, axis=0) # mat>0 exclude both 0 and np.nan
    columns_with_nonzeros = np.where(nonzero_counts_of_columns > 0)[0]
    return(columns_with_nonzeros)

def get_columns_index_with_nonzeros_memory_efficient(mat):
    nrow, _ = mat.shape
    mat_type = str(type(mat))
    if ('lil_matrix' in mat_type) or ('csc_matrix' in mat_type) or ('csr_matrix' in mat_type) or ('coo_matrix' in mat_type) or ('bsr_matrix' in mat_type):
        if 'csr_matrix' not in mat_type:
            nonnan_indicator = ~np.isnan(mat.tocsr(copy=False).data)
        else:
            nonnan_indicator = ~np.isnan(mat.data)
        keep = np.where(nonnan_indicator)[0]
        n_keep = len(keep)
        if n_keep == 0:
            # all nonzeros are actually nan
            return(np.array([]))
        nnz_inds = mat.nonzero()
        mat_binary_new = csc_matrix((np.ones(n_keep), (nnz_inds[0][keep], nnz_inds[1][keep])), shape=mat.shape, dtype=np.int8)
        nonzero_counts_of_columns = mat_binary_new.getnnz(axis=0)
        # nonzero_counts_of_columns = np.count_nonzero(mat.toarray() > 0, axis=0)  # mat>0 exclude both 0 and np.nan
    elif 'numpy.ndarray' in mat_type:
        nonzero_counts_of_columns = np.count_nonzero(mat>0, axis=0) # mat>0 exclude both 0 and np.nan
    columns_with_nonzeros = np.where(nonzero_counts_of_columns > 0)[0]
    return( columns_with_nonzeros )


def load_matrix_with_first_column_as_rownames(file):
    data = []
    rownames = []
    if file.endswith('gz'):
        f = gzip.open(file, 'rt')
    else:
        f = open(file, 'rt')
    for line in f:
        items = line.replace('NA','nan').rstrip().split('\t')
        rownames.append( items.pop(0) )
        data.append( np.array(list(map(float, items)), dtype=np.single) )
    f.close()
    return( (lil_matrix(data, dtype=np.single), rownames) )

def load_and_merge_sparse_matrix_from_many_files_of_small_matrixes(files_of_small_matrixes, n_sample):
    row_names = []
    i = 0
    for file in files_of_small_matrixes:
        data_part, row_names_part = load_matrix_with_first_column_as_rownames(file)
        if i == 0:
            _, ncol = data_part.shape
            data = lil_matrix((n_sample, ncol), dtype=np.single)
        nrow, ncol = data_part.shape
        data[i:(i+nrow), :] = data_part
        row_names += row_names_part
        i += nrow
    return( (data, row_names) )

def compact_sparse_matrix_by_removing_zero_columns(sparse_mat, column_names):
    nrow, ncol = sparse_mat.shape
    columns_index_with_nonzeros = get_columns_index_with_nonzeros(sparse_mat)
    if len(columns_index_with_nonzeros) == ncol:
        # No columns with all zeros. Keep unchanged
        return( (sparse_mat, column_names) )
    elif len(columns_index_with_nonzeros) == 0:
        # all columns are zeros and should be removed. Return empty matrix and empty column_names
        return( (lil_matrix(np.array([]), dtype=np.single), []) )
    else:
        column_names_with_nonzeros = []
        for i in columns_index_with_nonzeros:
            column_names_with_nonzeros.append(column_names[i])
        return( (sparse_mat[:,columns_index_with_nonzeros], column_names_with_nonzeros) )

def compact_sparse_matrix_by_removing_zero_columns_memory_efficient(sparse_mat, column_names):
    nrow, ncol = sparse_mat.shape
    columns_index_with_nonzeros = get_columns_index_with_nonzeros_memory_efficient(sparse_mat)
    if len(columns_index_with_nonzeros) == ncol:
        # No columns with all zeros. Keep unchanged
        return( (sparse_mat, column_names) )
    elif len(columns_index_with_nonzeros) == 0:
        # all columns are zeros and should be removed. Return empty matrix and empty column_names
        return( (lil_matrix(np.array([]), dtype=np.single), []) )
    else:
        column_names_with_nonzeros = []
        for i in columns_index_with_nonzeros:
            column_names_with_nonzeros.append(column_names[i])
        return( (sparse_mat[:,columns_index_with_nonzeros], column_names_with_nonzeros) )


# Input:
#   mat: a 2D numpy.array
#   row_names_list: a list of row names (strings)
#   fid: file identifier, such as sys.stdout or sys.stderr
# Output file format (tab delimited) with a header column:
#   row1_name value1 value2 ...
#   row2_name value1 value2 ...
#   ...
def write_matrix_with_row_labels(mat, row_names_list, fid):
    nrow, ncol = mat.shape
    for i in range(nrow):
        profile_str = '\t'.join(['%.3g' % v for v in mat[i, :]])
        profile_str = profile_str.replace('nan', 'NA')
        fid.write('%s\t%s\n'%(row_names_list[i], profile_str))

# Input:
#   mat: a 2D numpy.array
#   row_names_list: a list of row names (strings)
#   fid: file identifier, such as sys.stdout or sys.stderr
# Output file format (tab delimited) with a header column:
#   row1_name value1 value2 ...
#   row2_name value1 value2 ...
#   ...
def write_int_matrix_with_row_labels(mat, row_names_list, fid):
    nrow, ncol = mat.shape
    for i in range(nrow):
        profile_str = '\t'.join(['%.3g' % v for v in mat[i, :]])
        profile_str = profile_str.replace('nan', 'NA')
        fid.write('%s\t%s\n'%(row_names_list[i], profile_str))

# Input file format (tab delimited) with a header column:
#   row1_name value1 value2 ...
#   row2_name value1 value2 ...
#   ...
def summarize_rows_of_matrix_file_with_row_labels(input_matrix_file, output_summary_file):
    if input_matrix_file.endswith('gz'):
        f = gzip.open(input_matrix_file, 'rt')
    else:
        f = open(input_matrix_file, 'rt')
    with open(output_summary_file, 'wt') as fout:
        fout.write('row_label\tfrac_nnz\tfrac_zero\tfrac_nan\tsize\tcount_nnz\tcount_zero\tcount_nan\n')
        for line in f:
            items = line.rstrip().replace('NA','nan').split('\t')
            row_name = items.pop(0)
            row_numbers = np.array(list(map(float, items)))
            count_nnz, count_nan, count_zero, size_, frac_nnz, frac_nan, frac_zero =  summary_of_vector(row_numbers)
            fout.write('%s\t%.3g\t%.3g\t%.3g\t%d\t%d\t%d\t%d\n'%(row_name, frac_nnz, frac_zero, frac_nan, size_, count_nnz, count_zero, count_nan))
    f.close()

# Input:
#   mat: a 2D numpy.array
#   row_names_list: a list of row names (strings)
#   fid: file identifier, such as sys.stdout or sys.stderr
# Output file format (tab delimited) with a header line:
#   row1_name row2_name ...
#   value1 value2 ...
#   value1 value2 ...
#   ...
def write_transposed_matrix_with_row_labels(mat, row_names_list, fid):
    nrow, ncol = mat.shape
    fid.write('%s\n'%('\t'.join(row_names_list)))
    for j in range(ncol):
        profile_str = '\t'.join(['%.3g' % v for v in mat[:, j]])
        profile_str = profile_str.replace('nan', 'NA')
        fid.write('%s\n'%profile_str)

# The first line is header line
# The first column is sample_name. All rest columns are annotation data of this sample
# An example is below:
#
# sample,gender,age,batch
# plasma-344-F-LV,M,60,1
# liver_T-344-R-LV,M,60,1
# liver_N-344-R-LV,M,60,1
# WBC-344-R-LV,M,60,1
#
# Return:
#    samples_info: a dictionary {'sample':sample_list, 'gender':gender_list, 'age':age_list, 'batch':batch_list}
#    samples_list: a list of sample_name (the first column name)
#    annotation_feature_names_list: a list of annotation feature names (other column names), such as gender, age, batch
def parse_csv_file_with_header_line(file, delimit=','):
    with open(file) as f:
        column_names = next(f).rstrip().split(delimit)
        n_columns = len(column_names)
        # colindex2colname = {column_names.index(c):c for c in column_names}
        samples_info = {}
        for c in column_names:
            samples_info[c] = []
        for line in f:
            items = line.rstrip().split(delimit)
            for i in range(n_columns):
                samples_info[column_names[i]].append(items[i])
    annotation_feature_names_list = column_names[1:] # assume columns 2+ are the annotation feature names
    samples_info['sample'] = samples_info.pop(column_names[0])
    return((samples_info, annotation_feature_names_list))

# CSV file, with a header line, and column 1 (sample_name) and column 2 (class_name: N, CH, CC, LC, LG, LV, ST), and other columns
# The first line is header line
# The first column is sample_name, the second column is class_name. All rest columns are annotation data of this sample
# An example is below:
#
# sample,cl,gender,age,batch
# plasma-344-F-LV,LV,M,60,1
# liver_T-344-R-LV,LV,M,60,1
# liver_N-344-R-LV,LV,M,60,1
# WBC-344-R-LV,LV,M,60,1
#
# Return:
#    samples_info: a dictionary {'sample':sample_list, 'class':class_labels_list, 'gender':gender_list, 'age':age_list, 'batch':batch_list}
#    samples_list: a list of sample_name (the first column name)
#    annotation_feature_names_list: a list of annotation feature names (other column names), such as gender, age, batch
def parse_csv_file_with_header_line_and_first_two_columns(file):
    with open(file) as f:
        column_names = next(f).rstrip().split(',')
        n_columns = len(column_names)
        # colindex2colname = {column_names.index(c):c for c in column_names}
        samples_info = {}
        for c in column_names:
            samples_info[c] = []
        for line in f:
            items = line.rstrip().split(',')
            for i in range(n_columns):
                samples_info[column_names[i]].append(items[i])
    samples_info['sample'] = samples_info.pop(column_names[0])
    samples_info['class'] = samples_info.pop(column_names[1])
    annotation_feature_names_list = column_names[2:]  # assume columns 3+ are the annotation feature names
    return((samples_info, annotation_feature_names_list))


# CSV file, with a header line, and column name for samples (sample_name) and column name for class (class_name: N, CH, CC, LC, LG, LV, ST), and other columns
# The first line is header line
# An example is below:
#
# sample,cl,gender,age,batch
# plasma-344-F-LV,LV,M,60,1
# liver_T-344-R-LV,LV,M,60,1
# liver_N-344-R-LV,LV,M,60,1
# WBC-344-R-LV,LV,M,60,1
#
# Return:
#    samples_info: a dictionary {'sample':sample_list, 'class':class_labels_list, 'gender':gender_list, 'age':age_list, 'batch':batch_list}
#    samples_list: a list of sample_name (the first column name)
#    annotation_feature_names_list: a list of annotation feature names (other column names), such as gender, age, batch
def parse_csv_file_with_header_line_and_two_columns_with_specified_columnname(file,
                                                                              column_name_for_sample,
                                                                              column_name_for_class):
    with open(file) as f:
        column_names = next(f).rstrip().split(',')
        if (column_name_for_sample in column_names) and (column_name_for_class in column_names):
            column_index_for_sample = column_names.index(column_name_for_sample)
            column_index_for_class = column_names.index(column_name_for_class)
        n_columns = len(column_names)
        # colindex2colname = {column_names.index(c):c for c in column_names}
        samples_info = {}
        for c in column_names:
            samples_info[c] = []
        for line in f:
            items = line.rstrip().split(',')
            for i in range(n_columns):
                samples_info[column_names[i]].append(items[i])
    samples_info['sample'] = samples_info.pop(column_name_for_sample)
    samples_info['class'] = samples_info.pop(column_name_for_class)
    annotation_feature_names_list = column_names[2:]  # assume columns 3+ are the annotation feature names
    return((samples_info, annotation_feature_names_list))

# CSV file, with a header line, and column 1 (sample_name) and column 2 (class_name: N, CH, CC, LC, LG, LV, ST), and other columns
# The first line is header line
# The first column is sample_name, the second column is class_name. All rest columns are annotation data of this sample
# An example is below:
#
# sample,stage,pred.deg,age,batch,gender,cl,source,donor_id_biobank,cov.cpg,bis.conv,tumor.frac
# plasma-134-r1-F-CC,IIB,3.hard,78,10,F,CC,Bio-partners,2514,28.28,99.28,0.02095
# plasma-135-c12-F-CC,IVA,1.easy,57,10_11,M,CC,Bio-partners,2521,43.88,99.19,0.1783
# plasma-136-c12-F-CC,IIIB,3.hard,64,10_11,M,CC,Bio-partners,2527,21.57,99.35,0.02417
# plasma-137-r1-F-CC,IVA,1.easy,79,10,M,CC,Bio-partners,2531,20.92,99.19,0.2241
# plasma-141-r1-F-CC,IIIB,1.easy,58,11,F,CC,Bio-partners,3301,63.19,99.55,0.01558
#
# Return:
#    samples_info: a dictionary {'sample':sample_list, 'class':class_labels_list, 'stage':cancer_stage_list, 'gender':gender_list, 'age':age_list, 'batch':batch_list}
#    samples_list: a list of sample_name (the first column name)
#    annotation_feature_names_list: a list of annotation feature names (other column names), such as gender, age, batch
def parse_csv_file_with_header_line_and_three_specified_columns_for_cancer_samples(file, load_cancer_only=True, column_name_of_sample='sample', column_name_of_class='cl', column_name_of_cancer_stage='stage'):
    cancer_types_list, _, _ = get_cancer_types_list('short_name')
    with open(file) as f:
        column_names = next(f).rstrip().split(',')
        try:
            column_index_of_sample = column_names.index(column_name_of_sample)
            column_index_of_class = column_names.index(column_name_of_class)
            column_index_of_cancer_stage = column_names.index(column_name_of_cancer_stage)
        except ValueError:
            sys.stderr.write('ValueError: %s, %s, or %s does not exist in the file.\n  file: %s\nExit.\n'%(column_name_of_sample,
                                                                                                           column_name_of_class,
                                                                                                           column_name_of_cancer_stage,
                                                                                                           file))
            sys.exit(-1)
        n_columns = len(column_names)
        # colindex2colname = {column_names.index(c):c for c in column_names}
        samples_info = {}
        for c in column_names:
            samples_info[c] = []
        for line in f:
            if load_cancer_only:
                found = sum([True if c in line else False for c in cancer_types_list])
                if found == 0:
                    # this is a non-cancer sample
                    continue
            items = line.rstrip().split(',')
            for i in range(n_columns):
                samples_info[column_names[i]].append(items[i])
    samples_info['sample'] = samples_info.pop(column_name_of_sample)
    samples_info['class'] = samples_info.pop(column_name_of_class)
    samples_info['stage'] = normalize_cancer_stages(samples_info.pop(column_name_of_cancer_stage))
    annotation_feature_names_list = list(set(column_names) - set([column_name_of_sample, column_name_of_class, column_name_of_cancer_stage]))
    cancersample2cancerstage = {}
    for i in range(len(samples_info['sample'])):
        c = samples_info['class'][i]
        found = sum([True if c in cc else False for cc in cancer_types_list])
        if found != 0:
            s = samples_info['sample'][i]
            stage = samples_info['stage'][i]
            cancersample2cancerstage[s] = stage
    return((samples_info, cancersample2cancerstage, annotation_feature_names_list))


# File format: tab-delimited text file, each line a sample with the same number of feature_values.
# sample_name value_1 value_2 ...
# sample_name value_1 value_2 ...
#
# We assume the second minimum of the normalized read counts corresponds to 1 raw read count.
def convert_file_of_normalized_readcounts_to_file_of_raw_readcounts(input_file, output_file):
    with gzip.open(input_file, 'rt') as fin, gzip.open(output_file,'wt') as fout:
        for line in fin:
            items = line.rstrip().split('\t')
            sample = items.pop(0)
            normalized_values = np.array(list(map(float, items)))
            unit = min_2nd(normalized_values)
            values = (normalized_values / unit).astype(int)
            fout.write('%s\t%s\n'%(sample, '\t'.join(list(map(str,values)))))

# File format: tab-delimited text file, each line a sample with the same number of feature_values. Values are integers.
# sample_name value_1 value_2 ...
# sample_name value_1 value_2 ...
#
def parse_file_of_read_counts(file):
    data = {}
    with gzip.open(file, 'rt') as f:
        for line in f:
            items = line.rstrip().split('\t')
            sample = items.pop(0)
            values = np.array(list(map(int, items)))
            data[sample] = values
    return(data)

def nnz_count(arr_):
    return(sum(arr_!=0))

# 'data' is from the output of function 'parse_file_of_read_counts'
def summarize(data, selected_samples=[]):
    summary = {}
    if len(selected_samples) == 0:
        selected_samples = data.keys()
    for s in selected_samples:
        if s in data:
            summary[s] = [nnz_count(data[s]),
                          np.min(data[s]),
                          np.percentile(data[s],25),
                          np.median(data[s]),
                          np.percentile(data[s],75),
                          np.max(data[s])]
    summary_dataframe = pd.DataFrame.from_dict(summary,
                                               orient='index',
                                               columns=['nnz','min','percentile_25','percentile_50','percentile_75','max',])
    return(summary_dataframe)

# def print_summary(summary, samples_order=[]):
#     if len(samples_order)==0:
#         samples_order = sorted(summary.keys())
#     for s in samples_order:
#         if s in summary:

# 'd' is a dictionary: {int:int, int:int, ...}
def convert_int2int_dict_to_str(d, sep=','):
    keys_list = sorted(d.keys())
    int_keys_str = sep.join(list(map(str, keys_list)))
    int_values_str = sep.join(['%d'%d[k] for k in keys_list])
    return((int_keys_str, int_values_str))

# 'd' is a dictionary: {float_str:int, float_str:int, ...}
def convert_str2int_dict_to_str(d, sep=','):
    keys_list = sorted(d.keys())
    keys_str = sep.join(keys_list)
    int_values_str = sep.join(['%d'%d[k] for k in keys_list])
    return((keys_str, int_values_str))


# 'd' is a dictionary: {float_str:{int,int}, float_str:{int,int,int}, ...}
def convert_str2intset_dict_to_str(d, sep=','):
    sep_for_set_elements = '_'
    keys_list = sorted(d.keys())
    keys_str = sep.join(keys_list)
    intset_values_str = ''
    for k in keys_list:
        intset_values_str += sep_for_set_elements.join(['%d'%e for e in sorted(d[k])]) + sep
    return((keys_str, intset_values_str[:-1]))

# 'd' is a dictionary: {float_str:{str,str}, float_str:{str,str,str}, ...}
def convert_str2strset_dict_to_str(d, sep=','):
    sep_for_set_elements = '_'
    keys_list = sorted(d.keys())
    keys_str = sep.join(keys_list)
    intset_values_str = ''
    for k in keys_list:
        intset_values_str += sep_for_set_elements.join(sorted(d[k])) + sep
    return((keys_str, intset_values_str[:-1]))

###########
# Implement the histograms of methylation binary strings for observing the position-specific methlation patterns.
###########

# 1111111
# 1111111
# 1111011
# ...
# Assume these methylation binary strings have the same length
# Some strings may not have the same length as others, we still count them and they are unique strings from others.
def convert_methylation_str_list_to_meth_strings_histgram(meth_strings_list):
    if len(meth_strings_list)==0:
        return( (None, None) )
    unique_meth_str_2_freq = collections.Counter(meth_strings_list)
    unique_meth_strings = sorted(unique_meth_str_2_freq.keys())
    read_freq_of_unique_meth_strings = ','.join([str(unique_meth_str_2_freq[meth_str]) for meth_str in unique_meth_strings])
    return( (','.join(unique_meth_strings),
             read_freq_of_unique_meth_strings)
    )

# Mary's methy_reads_binning_file file (*.binary_meth_values) is located at /u/project/xjzhou/marysame/cf_RRBS_all_reads/data/cancerdetector/modify_for_cfRRBS/run_on_real_samples/reads_binning/theoretical_RRBS_regions_merged/binary_meth_values/*.binary_meth_values.txt.gz
# Input: Mary binary_meth_values file format:
# IMPORTANT: the lines in the file are sorted by increasing order of marker_index. This can save a lot of time.
# marker_index    cpg_locs    meth_string meth_count  unmeth_count    strand
# 2   10497,10525,10542,10563,10571,10577,10579   1111111 7   0   +
# 2   10497,10525,10542,10563,10571,10577,10579   1111111 7   0   +
# 2   10526,10543,10564,10572,10578,10580,10590   1111011 6   1   -
# 2   10526,10543,10564,10572,10578,10580,10590   1111111 7   0   -
# 72  133165,133180   01  1   1   +
# 72  133165,133180   11  2   0   +
# 72  133181,133218   11  2   0   -
# 72  133181,133218   11  2   0   -
# 223 566813,566879   00  0   2   +
# 223 566813,566879   00  0   2   +
# 223 567006,567015,567063,567091,567114,567123   111111  6   0   -
# 859 934726,934737,934748,934759 1110    3   1   +
# 859 934726,934737,934748,934759 0000    0   4   +
# 859 934738,934749,934760,934769 0000    0   4   -
# 859 934738,934749,934760,934769 0000    0   4   -
# 859 NA NA    0   0   -
# 2321    1209103,1209115,1209123,1209130,1209132,1209136,1209145,1209150 00000000    0   8   +
# 2321    1209103,1209123,1209130,1209132,1209136,1209145,1209150 0000000 0   7   +
# ...
#
# See the above and you may find marker_index=2321, it has two reads, one has 8 CpG sites and the other has only 7 CpG sites. We still tolerate it and write their histgram as below:
# 2321    +   8   8   0000000,00000000   1,1   1209103,1209115,1209123,1209130,1209132,1209136,1209145,1209150
#
# There are even markers like marker_index=223, where all CpG sites of strand+ reads are different from those of strand- reads. For these markers, if we do not output their CpG sites, then methylation strings of strand+ reads cannot be matched to strings of strand- reads
#
# Output: histograms file: each marker has a histogram with frquency for each unique methylation string
# marker_index num_cpg num_read unique_meth_strings freq_of_unique_meth_strings cpg_locs
def summarize_mary_file_binary_meth_values_for_meth_string_histgram_file(input_methy_reads_binning_file, output_meth_string_histgram_file, output_cpg_sites):
    with gzip.open(input_methy_reads_binning_file,'rt') as fin, gzip.open(output_meth_string_histgram_file, 'wt') as fout:
        next(fin) # skip the first header line
        if output_cpg_sites:
            fout.write('marker_index\tstrand\tnum_read\tnum_cpg\tunique_meth_strings\tread_freq_of_unique_meth_strings\tcpg_locs\n')
        else:
            fout.write(
                'marker_index\tstrand\tnum_read\tnum_cpg\tunique_meth_strings\tread_freq_of_unique_meth_strings\n')
        meth_string_histgram_of_marker = {'marker_index': -1,
                                          'strand+':{
                                              'methy_str_list':[],
                                              'cpg_locs':'',
                                           },
                                          'strand-': {
                                              'methy_str_list': [],
                                              'cpg_locs':'',
                                           }
                                          }
        for line in fin:
            if 'NA' in line:
                continue
            marker_index, cpg_locs, meth_string, _, _, strand = line.rstrip().split('\t')
            if marker_index != meth_string_histgram_of_marker['marker_index']:
                # A new marker begins, we need to print the old marker
                if meth_string_histgram_of_marker['marker_index']!=-1:
                    # generate and output meth_strings_histgram for strand+ reads and strand- reads
                    for strand_ in ['+', '-']:
                        unique_meth_strings, read_freq_of_unique_meth_strings = convert_methylation_str_list_to_meth_strings_histgram(meth_string_histgram_of_marker['strand'+strand_]['methy_str_list'])
                        num_reads_of_prev_marker = len(meth_string_histgram_of_marker['strand'+strand_]['methy_str_list'])
                        if num_reads_of_prev_marker == 0:
                            if output_cpg_sites:
                                fout.write('%s\t%s\t0\tNA\tNA\tNA\tNA\n'%(meth_string_histgram_of_marker['marker_index'],strand_))
                            else:
                                fout.write('%s\t%s\t0\tNA\tNA\tNA\n' % (meth_string_histgram_of_marker['marker_index'],strand_))
                        else:
                            if output_cpg_sites:
                                fout.write('%s\t%s\t%d\t%d\t%s\t%s\t%s\n'%(meth_string_histgram_of_marker['marker_index'],
                                                                           strand_,
                                                                           num_reads_of_prev_marker,
                                                                           len(meth_string_histgram_of_marker[
                                                                                   'strand'+strand_]['cpg_locs'].split(',')),
                                                                           unique_meth_strings,
                                                                           read_freq_of_unique_meth_strings,
                                                                           meth_string_histgram_of_marker['strand'+strand_][
                                                                               'cpg_locs']
                                                                           ))
                            else:
                                fout.write('%s\t%s\t%d\t%d\t%s\t%s\n' % (meth_string_histgram_of_marker['marker_index'],
                                                                         strand_,
                                                                         num_reads_of_prev_marker,
                                                                         len(meth_string_histgram_of_marker['strand'+strand_][
                                                                                 'cpg_locs'].split(',')),
                                                                         unique_meth_strings,
                                                                         read_freq_of_unique_meth_strings
                                                                         ))
                # Clear the old marker info, and initialize read counting for a new marker
                meth_string_histgram_of_marker = {'marker_index': marker_index,
                                                  'strand+': {
                                                      'methy_str_list': [],
                                                      'cpg_locs': '',
                                                  },
                                                  'strand-': {
                                                      'methy_str_list': [],
                                                      'cpg_locs': '',
                                                  }
                                                  }
            meth_string_histgram_of_marker['strand'+strand]['methy_str_list'].append(meth_string)
            meth_string_histgram_of_marker['strand'+strand]['cpg_locs'] = cpg_locs

        # At the end of the input file
        # write the last marker of the file
        if meth_string_histgram_of_marker['marker_index'] != -1:
            # generate and output meth_strings_histgram for strand+ reads and strand- reads
            for strand_ in ['+', '-']:
                unique_meth_strings, read_freq_of_unique_meth_strings = convert_methylation_str_list_to_meth_strings_histgram(
                    meth_string_histgram_of_marker['strand' + strand_]['methy_str_list'])
                num_reads_of_prev_marker = len(meth_string_histgram_of_marker['strand' + strand_]['methy_str_list'])
                if num_reads_of_prev_marker == 0:
                    if output_cpg_sites:
                        fout.write(
                            '%s\t%s\t0\tNA\tNA\tNA\tNA\n' % (meth_string_histgram_of_marker['marker_index'], strand_))
                    else:
                        fout.write(
                            '%s\t%s\t0\tNA\tNA\tNA\n' % (meth_string_histgram_of_marker['marker_index'], strand_))
                else:
                    if output_cpg_sites:
                        fout.write('%s\t%s\t%d\t%d\t%s\t%s\t%s\n' % (meth_string_histgram_of_marker['marker_index'],
                                                                     strand_,
                                                                     num_reads_of_prev_marker,
                                                                     len(meth_string_histgram_of_marker[
                                                                             'strand'+strand_]['cpg_locs'].split(',')),
                                                                     unique_meth_strings,
                                                                     read_freq_of_unique_meth_strings,
                                                                     meth_string_histgram_of_marker['strand'+strand_][
                                                                         'cpg_locs']
                                                                     ))
                    else:
                        fout.write('%s\t%s\t%d\t%d\t%s\t%s\n' % (meth_string_histgram_of_marker['marker_index'],
                                                                 strand_,
                                                                 num_reads_of_prev_marker,
                                                                 len(meth_string_histgram_of_marker['strand'+strand_][
                                                                         'cpg_locs'].split(',')),
                                                                 unique_meth_strings,
                                                                 read_freq_of_unique_meth_strings
                                                                 ))

# input meth_string_histgram_file format:
# marker_index	strand	num_read	num_cpg	unique_meth_strings	read_freq_of_unique_meth_strings
# 2	+	151	7	0011111,0100111,010110,0101111,0110011,0110110,0110111,0111011,0111110,0111111,1010011,1010111,1011110,1011111,1101010,1101011,1101101,1101111,1110011,1110100,1110101,1110110,1110111,1111000,1111010,11111,1111100,1111101,1111110,1111111	1,1,1,1,1,1,5,1,1,6,1,1,1,3,1,1,4,2,2,1,1,1,9,2,1,1,11,20,6,63
# 2	-	127	7	0011111,1000111,1001011,1001111,1011011,1011111,1101011,1101101,1101111,1110101,1110111,1111000,1111001,1111010,1111011,1111100,1111101,1111110,1111111	1,1,1,1,5,7,1,1,10,1,2,1,12,2,18,1,3,9,50
# 27	+	0	NA	NA	NA
# 27	-	9	8	011111011,11000111,11110010,11110011,11110111,11111011,11111111	1,1,1,1,2,1,2
# 61	+	0	NA	NA	NA
# 61	-	10	12	101111111011,110111111110,110111111111,111101011011,111111011011,11111101110,111111101011,111111111011,111111111111	1,2,1,1,1,1,1,1,1
# 63	+	6	5	01111,11111	2,4
# 63	-	10	5	01101,10111,11101,11111	1,1,1,7
# 65	+	1	5	11111	1
# 65	-	6	5	11111	6
# ...
#
# meth_hists: a dictionary { 'marker_strand':histgram_dictionary }, for example: {'27_+':hist_dict, '27_-':hist_dict, '63_+':hist_dict, '63_-':hist_dict}, where hist_dict is a dictionary {'01111':2, '11111':4}. The input 'meth_hists' can be an empty dictionary {}.
def load_one_meth_string_histgram_file(file, meth_hists):
    with gzip.open(file, 'rt') as f:
        next(f) # skip the first header line
        for line in f:
            items = line.rstrip().split()
            marker_index = items[0]
            strand = items[1]
            unique_meth_strings = items[4]
            read_freq_of_unique_meth_strings = items[5]
            marker_id = '%s_%s'%(marker_index, strand)
            if 'NA' in unique_meth_strings:
                if marker_id not in meth_hists:
                    meth_hists[marker_id] = {}
                continue
            meth_str_list = unique_meth_strings.split(',')
            read_freq = list(map(int, read_freq_of_unique_meth_strings.split(',')))
            n = len(meth_str_list)
            meth_hist_dict = {meth_str_list[i]: read_freq[i] for i in range(n)}
            if marker_id not in meth_hists:
                meth_hists[marker_id] = {}
            # update histgrams
            meth_hists[marker_id] = mergeDict_by_adding_values_of_common_keys(meth_hists[marker_id], meth_hist_dict)

# Input:
#   meth_histgram: a dictionary {meth_string:frequency}. For example: {'01111':2, '11111':4}
# Output:
#
def alpha_stats_of_meth_string_histgram(meth_histgram):
    alpha2freq_and_meth_string = {}
    alpha2freq = {}
    for meth_str, freq in meth_histgram.items():
        alpha = meth_str.count('1') / float(len(meth_str))
        if alpha not in alpha2freq_and_meth_string:
            alpha2freq[alpha] = freq
            alpha2freq_and_meth_string[alpha] = [(meth_str, freq)]
        else:
            alpha2freq_and_meth_string[alpha].append((meth_str, freq))
            alpha2freq[alpha] += freq
    min_alpha = min(alpha2freq.keys())
    max_alpha = max(alpha2freq.keys())
    return( (min_alpha, max_alpha, alpha2freq_and_meth_string))

# Input:
#   alpha2freq_and_meth_string: output variable of 'alpha_stats_of_meth_string_histgram'. It is a dictionary {alpha_value:list of tuples (meth_string, frequency)}. For example, {0.2:[('10000',20),('01000',10)], 1:[('11111', 7)]}
# Output:
#   meth_histgram: a dictionary structure {meth_string:freq}. For example, {'10000':20, '11111':7}.
def filter_alpha2freq_and_meth_string_by_alpha(alpha2freq_and_meth_string, alpha_cutoff, direction_to_keep_alpha2freq='>='):
    meth_histgram = {}
    if direction_to_keep_alpha2freq=='>=':
        # keep all alpha2freq and their meth_strings if their alpha>=alpha_cutoff
        for alpha, v in alpha2freq_and_meth_string.items():
            if alpha>=alpha_cutoff:
                for meth_str, freq in v:
                    meth_histgram[meth_str] = freq
    elif direction_to_keep_alpha2freq=='>':
        # keep all alpha2freq and their meth_strings if their alpha>alpha_cutoff
        for alpha, v in alpha2freq_and_meth_string.items():
            if alpha > alpha_cutoff:
                for meth_str, freq in v:
                    meth_histgram[meth_str] = freq
    elif direction_to_keep_alpha2freq=='<=':
        # keep all alpha2freq and their meth_strings if their alpha<=alpha_cutoff
        for alpha, v in alpha2freq_and_meth_string.items():
            if alpha <= alpha_cutoff:
                for meth_str, freq in v:
                    meth_histgram[meth_str] = freq
    elif direction_to_keep_alpha2freq=='<':
        # keep all alpha2freq and their meth_strings if their alpha<alpha_cutoff
        for alpha, v in alpha2freq_and_meth_string.items():
            if alpha < alpha_cutoff:
                for meth_str, freq in v:
                    meth_histgram[meth_str] = freq
    return(meth_histgram)

# Input:
#   in_file1_background and in_file2_cancer: file format is from function 'write_combined_meth_string_histgram'
# Procedure:
#   We first load each of these two files into a dictionary { 'marker_strand':histgram_dictionary }, for example: {'27_+':hist_dict, '27_-':hist_dict, '63_+':hist_dict, '63_-':hist_dict}, where hist_dict is a dictionary {'01111':2, '11111':4}. It is the output of 'combine_multi_meth_string_histgram_files' or 'load_one_meth_string_histgram_file'
#
def compare_background_vs_cancer_meth_string_histgram_files(method, in_file1_background, in_file2_cancer):
    m1_background = {}
    load_one_meth_string_histgram_file(in_file1_background, m1_background)
    m2_cancer = {}
    load_one_meth_string_histgram_file(in_file2_cancer, m2_cancer)
    marker_index_list1 = [int(m.split('_')[0]) for m in m1_background]
    marker_index_list2 = [int(m.split('_')[0]) for m in m2_cancer]
    marker_index_common_list = sorted(list(set(marker_index_list1).intersection(marker_index_list2)))
    ret_meth_hist = {}
    try:
        if 'hypo.min.alpha.diff' in method: # 'hypo.min.alpha.diff_0.3' if (min_alpha(m1_background[marker_id]) - min_alpha(m2_cancer[marker_id]))>=0.3, then we accept this marker_id and report those meth_strings in m2_cancer[marker_id] whose alpha values < min_alpha(m1_background[marker_id]).
            min_alpha_diff = float(method.split('_')[1])
            for m in marker_index_common_list:
                for strand in ['+', '-']:
                    marker_id = '%d_%s'%(m,strand)
                    if (len(m1_background[marker_id])==0) or (len(m2_cancer[marker_id])==0): continue
                    a1_min, _, alpha2freq_and_meth_string_1 = alpha_stats_of_meth_string_histgram(m1_background[marker_id])
                    a2_min, _, alpha2freq_and_meth_string_2 = alpha_stats_of_meth_string_histgram(m2_cancer[marker_id])
                    if (a1_min - a2_min) >= min_alpha_diff:
                        ret_meth_hist[marker_id] = filter_alpha2freq_and_meth_string_by_alpha(alpha2freq_and_meth_string_2, a1_min, '<')
        elif 'hyper.max.alpha.diff' in method: # 'hyper.max.alpha.diff_0.3' if max_alpha(m2_cancer[marker_id]) - max_alpha(m1_background[marker_id])>=0.3, then we accept this marker_id and report those meth_strings in m2_cancer[marker_id] whose alpha values > max_alpha(m1_background[marker_id]).
            min_alpha_diff = float(method.split('_')[1])
            for m in marker_index_common_list:
                for strand in ['+', '-']:
                    marker_id = '%d_%s'%(m,strand)
                    if (len(m1_background[marker_id])==0) or (len(m2_cancer[marker_id])==0): continue
                    _, a1_max, alpha2freq_and_meth_string_1 = alpha_stats_of_meth_string_histgram(m1_background[marker_id])
                    _, a2_max, alpha2freq_and_meth_string_2 = alpha_stats_of_meth_string_histgram(m2_cancer[marker_id])
                    if (a2_max - a1_max) >= min_alpha_diff:
                        ret_meth_hist[marker_id] = filter_alpha2freq_and_meth_string_by_alpha(alpha2freq_and_meth_string_2, a1_max, '>')
    except KeyError:
        # marker_index does not exist
        sys.stderr.write('Error: %s does not exist in one of two meth_strings_histgram_files\n  in_file1_background: %s\n  in_file2_cancer: %s\nExit.'%(marker_id, in_file1_background, in_file2_cancer))
        sys.exit(-1)
    return(ret_meth_hist)

def combine_multi_meth_string_histgram_files(files_list):
    combined_meth_hists = {}
    n = len(files_list)
    i = 0
    for filename in files_list:
        i += 1
        print('  (%d/%d) %s, '%(i, n, filename), datetime.now(), flush=True)
        load_one_meth_string_histgram_file(filename, combined_meth_hists)
    return(combined_meth_hists)

# Input:
#   combined_meth_hists: a dictionary { 'marker_strand':frequency }, for example: {'27_+':5, '27_-':2, '63_+':9, '63_-':1}. It is the output of 'combine_multi_meth_string_histgram_files' or 'load_one_meth_string_histgram_file'
#
# Output meth_string_histgram_file format:
# marker_index	strand	num_read	num_cpg	unique_meth_strings	read_freq_of_unique_meth_strings
# 2	+	151	7	0011111,0100111,010110,0101111,0110011,0110110,0110111,0111011,0111110,0111111,1010011,1010111,1011110,1011111,1101010,1101011,1101101,1101111,1110011,1110100,1110101,1110110,1110111,1111000,1111010,11111,1111100,1111101,1111110,1111111	1,1,1,1,1,1,5,1,1,6,1,1,1,3,1,1,4,2,2,1,1,1,9,2,1,1,11,20,6,63
# 2	-	127	7	0011111,1000111,1001011,1001111,1011011,1011111,1101011,1101101,1101111,1110101,1110111,1111000,1111001,1111010,1111011,1111100,1111101,1111110,1111111	1,1,1,1,5,7,1,1,10,1,2,1,12,2,18,1,3,9,50
# 27	+	0	NA	NA	NA
# 27	-	9	8	011111011,11000111,11110010,11110011,11110111,11111011,11111111	1,1,1,1,2,1,2
def write_combined_meth_string_histgram(fout, combined_meth_hists):
    marker_index_list = [int(m.split('_')[0]) for m in combined_meth_hists]
    marker_index_list = sorted(list(set(marker_index_list)))
    fout.write('marker_index\tstrand\tnum_read\tnum_cpg\tunique_meth_strings\tread_freq_of_unique_meth_strings\n')
    for m in marker_index_list:
        for strand in ['+', '-']:
            marker_id = '%d_%s'%(m,strand)
            try:
                if len(combined_meth_hists[marker_id])==0:
                    fout.write('%s\t%s\t0\tNA\tNA\tNA\n' % (m,strand))
                else:
                    t = sorted(combined_meth_hists[marker_id], key=combined_meth_hists[marker_id].__getitem__, reverse=True)
                    # for k, v in sorted(d.items(), key=lambda item: -item[1])
                    # t = sorted(combined_meth_hists[marker_id].keys())
                    out_str_for_unique_meth_strings = ','.join(t)
                    out_str_for_read_freq_of_unique_meth_strings = ','.join(['%d'%combined_meth_hists[marker_id][m] for m in t])
                    num_read = sum(combined_meth_hists[marker_id].values())
                    num_cpg = len(t[0])
                    fout.write('%s\t%s\t%d\t%d\t%s\t%s\n' % (m,
                                                             strand,
                                                             num_read,
                                                             num_cpg,
                                                             out_str_for_unique_meth_strings,
                                                             out_str_for_read_freq_of_unique_meth_strings
                                                             ))
            except KeyError:
                # marker_index does not exist
                fout.write('%s\t%s\t0\tNA\tNA\tNA\n'%(m,strand))


###########
# Re-implement the marker discovery using PCR marker concept (Mary)
###########

# convert meth_str_list=['1111111', '1101011', '1101011', '1101011'] to two dictionaries:
#    a dictionary: {unique_methylation_cpg_count:read_number}={7:1, 5:3}
#    a dictionary: {'1':1, '0.714':3}
def convert_methylation_str_list_to_distribution(meth_str_list):
    distribution_meth_counts = {}
    distribution_alpha_values = {}
    max_len = 0
    for m in meth_str_list:
        meth_count = m.count('1')
        alpha_str = '%.3g'%(meth_count / float(len(m)))
        if meth_count not in distribution_meth_counts:
            distribution_meth_counts[meth_count] = 1
        else:
            distribution_meth_counts[meth_count] += 1
        if alpha_str not in distribution_alpha_values:
            distribution_alpha_values[alpha_str] = 1
        else:
            distribution_alpha_values[alpha_str] += 1
        if max_len<len(m):
            max_len = len(m)
    return((distribution_meth_counts, distribution_alpha_values, max_len))

# Mary's methy_reads_binning_file file (*.binary_meth_values) is located at /u/project/xjzhou/marysame/cf_RRBS_all_reads/data/cancerdetector/modify_for_cfRRBS/run_on_real_samples/reads_binning/theoretical_RRBS_regions_merged/binary_meth_values/*.binary_meth_values.txt.gz
# Input: Mary binary_meth_values file format:
# IMPORTANT: the lines in the file are sorted by increasing order of marker_index. This can save a lot of time.
# marker_index    cpg_locs    meth_string meth_count  unmeth_count    strand
# 2   10497,10525,10542,10563,10571,10577,10579   1111111 7   0   +
# 2   10497,10525,10542,10563,10571,10577,10579   1111101 6   1   +
# 2   10497,10525,10542,10563,10571,10577,10579   1111111 7   0   +
# 2   10497,10525,10542,10563,10571,10577,10579   1111111 7   0   +
# 859 934726,934737,934748,934759 1110    3   1   +
# 859 934726,934737,934748,934759 0000    0   4   +
# 859 934738,934749,934760,934769 0000    0   4   -
# 859 934738,934749,934760,934769 0000    0   4   -
# 859 NA NA    0   0   -
#
# Output: Wenyuan alpha value distribution file
# marker_index num_cpg num_read unique_meth_counts freq_of_unique_meth_counts unique_alpha_values freq_of_unique_alpha_values
def summarize_mary_file_binary_meth_values_for_distribution_file(input_methy_reads_binning_file, output_distribution_file):
    with gzip.open(input_methy_reads_binning_file,'rt') as fin, gzip.open(output_distribution_file, 'wt') as fout:
        next(fin) # skip the first header line
        # distribution_of_marker = {'marker_index': None, 'num_cpg': 0,
        #                           'distribution': {'strand+':{'num_read': 0, 'unique_meth_counts_to_freq':{}},
        #                                            'strand-':{'num_read': 0, 'unique_meth_counts_to_freq':{}}
        #                                            }
        #                           }
        fout.write('marker_index\tmax_num_cpg\tnum_read\tunique_alpha_values\tread_freq_of_unique_alpha_values\tunique_meth_counts\tread_freq_of_unique_meth_counts\n')
        distribution_of_marker = {'marker_index': -1,
                                  'max_num_cpg': 0,
                                  'methy_str_list': [],
                                  'unique_meth_counts_to_freq': {},
                                  'unique_alpha_values_to_freq': {},
                                  }
        for line in fin:
            if 'NA' in line:
                continue
            marker_index, _, meth_string, _, _, _ = line.rstrip().split('\t')
            if marker_index != distribution_of_marker['marker_index']:
                # A new marker begins, we need to print the old marker
                if distribution_of_marker['marker_index']!=-1:
                    num_reads = len(distribution_of_marker['methy_str_list'])
                    distribution_of_marker['unique_meth_counts_to_freq'], distribution_of_marker['unique_alpha_values_to_freq'], distribution_of_marker['max_num_cpg'] = convert_methylation_str_list_to_distribution(distribution_of_marker['methy_str_list'])
                    # Print 'distribution_of_marker' to output
                    unique_meth_count_str, read_freq_of_unique_meth_counts_str = convert_int2int_dict_to_str(distribution_of_marker['unique_meth_counts_to_freq'])
                    unique_alpha_values_str, read_freq_of_unique_alpha_values_str = convert_str2int_dict_to_str(
                        distribution_of_marker['unique_alpha_values_to_freq'])
                    fout.write('%s\t%d\t%d\t%s\t%s\t%s\t%s\n'%(distribution_of_marker['marker_index'],
                                                               distribution_of_marker['max_num_cpg'],
                                                               num_reads,
                                                               unique_alpha_values_str,
                                                               read_freq_of_unique_alpha_values_str,
                                                               unique_meth_count_str,
                                                               read_freq_of_unique_meth_counts_str,
                                                           ))
                # Clear the old marker info, and initialize read counting for a new marker
                distribution_of_marker = {'marker_index': marker_index,
                                          'max_num_cpg': 0,
                                          'methy_str_list': [],
                                          'unique_meth_counts_to_freq': {},
                                          'unique_alpha_values_to_freq': {},
                                          }
            distribution_of_marker['methy_str_list'].append(meth_string)

        # At the end of the input file
        # write the last marker of the file
        if distribution_of_marker['marker_index'] != -1:
            num_reads = len(distribution_of_marker['methy_str_list'])
            distribution_of_marker['unique_meth_counts_to_freq'], distribution_of_marker['unique_alpha_values_to_freq'], \
            distribution_of_marker['max_num_cpg'] = convert_methylation_str_list_to_distribution(
                distribution_of_marker['methy_str_list'])
            # Print 'distribution_of_marker' to output
            unique_meth_count_str, read_freq_of_unique_meth_counts_str = convert_int2int_dict_to_str(
                distribution_of_marker['unique_meth_counts_to_freq'])
            unique_alpha_values_str, read_freq_of_unique_alpha_values_str = convert_str2int_dict_to_str(
                distribution_of_marker['unique_alpha_values_to_freq'])
            fout.write('%s\t%d\t%d\t%s\t%s\t%s\t%s\n' % (distribution_of_marker['marker_index'],
                                                         distribution_of_marker['max_num_cpg'],
                                                         num_reads,
                                                         unique_alpha_values_str,
                                                         read_freq_of_unique_alpha_values_str,
                                                         unique_meth_count_str,
                                                         read_freq_of_unique_meth_counts_str,
                                                         ))


# Identify PCR markers
# Input string: 'hypermarker_Nsf0.95Nrf1Na0.2Tsf0.2Trf0.2Ta1cov10cpg6'
def parse_parameters_of_marker_discovery(parameters_str):
    params_explanation = {'type': 'Marker type: hyper- or hypo-methylation in tumor',
                          'Nsf': 'Define a pure background (1): >=Fraction of reference normal samples have pure background noise (defined by Nrf and Na)',
                          'Nrf': 'Define a pure background (2): >=Fraction of normal reads in Nsf reference normal samples',
                          'Na': 'Define normal reads in reference normal samples, as those reads with alpha values (<=Na for hyper-marker and >=Na for hypo-marker)',
                          'Tsf': 'Define tumor signal (1): >=Fraction (float btw 0 and 1) of tumor samples have tumor reads. You may use either Tsf or TsF',
                          'TsF': 'Define tumor signal (1): >=Frequency (integer>=1) of tumor samples have tumor reads. You may use either Tsf or TsF',
                          'Trf': 'Define tumor signal (2): >=Fraction tumor reads with tumor reads',
                          'Ta': 'Define tumor reads in tumor, as those reads with alpha values (>=Ta for hyper-marker and <=Ta for hypo-marker)',
                          'cov': 'Define coverage requirement: #all_reads >= cov',
                          'cpg': 'Define CpG number requirement: #all_cpg >= cpg',
                          }
    params = {'type':None,
              'Nsf':None,
              'Nrf':None,
              'Na': None,
              'Tsf': None,
              'TsF': None,
              'Trf': None,
              'Ta': None,
              'cov': None,
              'cpg': None,
              }
    type, details = parameters_str.split('_')
    names_list = []
    names_nonexist_in_input_params_string = []
    for param_name in params.keys():
        if param_name == 'type':
            if 'hyper' in type:
                params['type'] = 'hyper'
            elif 'hypo' in type:
                params['type'] = 'hypo'
            else:
                sys.stderr.write('Error: The 1st part of %s does not contain either hyper or hypo.\nExit.\n'%parameters_str)
                sys.exit(-1)
        else:
            param_value = extract_number_after_a_substring(details, param_name)
            if param_value is None:
                if (param_name == 'Tsf') or (param_name == 'TsF'):
                    names_nonexist_in_input_params_string.append(param_name)
                    continue
                else:
                    sys.stderr.write(
                        'Error: The parameter %s is not found in input parameter string \'%s\'.\nExit.\n' %(param_name, parameters_str))
                    sys.exit(-1)
            params[param_name] = float(param_value)
        names_list.append(param_name)
    for param_name in names_nonexist_in_input_params_string:
        params.pop(param_name)
        params_explanation.pop(param_name)
    if ('Tsf' not in names_list) and ('TsF' not in names_list):
        sys.stderr.write(
            'Error: The parameter (either of Tsf and TsF) is not found in input parameter string \'%s\'.\nExit.\n' % (
            parameters_str))
        sys.exit(-1)
    params['names_list']=names_list
    return(params, params_explanation)

def print_parameters_of_marker_discovery(fid, params_values, params_explanation, str_prefix='#'):
    for params_name in params_values['names_list']:
        if params_name=='type':
            fid.write('%s%s: %s (%s)\n'%(str_prefix, params_name, params_values[params_name], params_explanation[params_name]))
        else:
            fid.write('%s%s: %g (%s)\n'%(str_prefix, params_name, params_values[params_name], params_explanation[params_name]))

def get_number_of_reads_meeting_criterion(unique_alpha_values, read_freq_of_unique_alpha_values, alpha_threshold, direction='<='):
    read_num = 0
    if direction=='<=':
        read_num = np.sum(read_freq_of_unique_alpha_values[unique_alpha_values <= alpha_threshold])
    elif direction == '>=':
        read_num = np.sum(read_freq_of_unique_alpha_values[unique_alpha_values >= alpha_threshold])
    elif direction == '<':
        read_num = np.sum(read_freq_of_unique_alpha_values[unique_alpha_values < alpha_threshold])
    elif direction == '>':
        read_num = np.sum(read_freq_of_unique_alpha_values[unique_alpha_values > alpha_threshold])
    elif direction == '==':
        read_num = np.sum(read_freq_of_unique_alpha_values[unique_alpha_values == alpha_threshold])
    return(read_num)

#
# Identify PCR markers
#
# Input:
# alpha_value_distribution file (from the output of the function 'summarize_mary_file_binary_meth_values_for_distribution_file'):
# marker_index    max_num_cpg num_read    unique_alpha_values read_freq_of_unique_alpha_values    unique_meth_counts  read_freq_of_unique_meth_counts
# 2   7   122 0.429,0.571,0.714,0.857,1   1,2,27,42,50    3,4,5,6,7   1,2,27,42,50
# 27  9   39  0.5,0.625,0.75,0.778,0.875,0.889,1  1,2,9,1,12,1,13 4,5,6,7,8   1,2,9,13,14
# 61  12  44  0.75,0.833,0.917,1  2,11,12,19  9,10,11,12  2,11,13,18
# 63  5   100 0.6,0.8,1   4,23,73 3,4,5   4,23,73
# 65  5   83  0,0.2,0.4,0.6,0.8,1 1,3,2,9,26,42   0,1,2,3,4,5 1,3,2,9,26,42
#
# params_values: a dictionary from the output of the function 'parse_parameters_of_marker_discovery'
# samples_type: 'reference_normal_plasma' or 'tumor'
#
# Output:
#   data: a dictionary marker_index -> {'cov':20, 'cov_meet_require':15, 'fraction':0.9}. If it is empty {}, then this sample fails to be used for marker discovery.
def parse_alpha_value_distribution_file_with_params_requirements(file, params_values, samples_type='reference_normal_plasma'):
    data = {}
    with gzip.open(file, 'rt') as f:
        next(f) # skip the header line
        for line in f:
            marker_index, max_num_cpg, num_read, unique_alpha_values_str, read_freq_of_unique_alpha_values_str, _, _ = line.rstrip().split('\t')
            if (int(max_num_cpg) < params_values['cpg']) or (int(num_read) < params_values['cov'] ):
                continue
            marker_index = int(marker_index)
            num_read = float(num_read)
            unique_alpha_values = np.array(list(map(float, unique_alpha_values_str.split(','))))
            read_freq_of_unique_alpha_values = np.array(list(map(int, read_freq_of_unique_alpha_values_str.split(','))))
            if 'hyper' in params_values['type']:
                # tumor reads in tumor & plasma are defined as 'alpha>=Ta'; normal reads in plasma are defined as 'alpha<=Na'
                if samples_type == 'reference_normal_plasma':
                    # identify normal reads in reference_normal_plasma
                    read_cov_of_criterion = get_number_of_reads_meeting_criterion(unique_alpha_values,
                                                                                  read_freq_of_unique_alpha_values,
                                                                                  params_values['Na'],
                                                                                  '<=')
                    if read_cov_of_criterion/num_read < params_values['Nrf']:
                        continue
                elif samples_type == 'tumor':
                    # identify tumor reads in tumor
                    read_cov_of_criterion = get_number_of_reads_meeting_criterion(unique_alpha_values,
                                                                                  read_freq_of_unique_alpha_values,
                                                                                  params_values['Ta'],
                                                                                  '>=')
                    if read_cov_of_criterion/num_read < params_values['Trf']:
                        continue
            elif 'hypo' in params_values['type']:
                # tumor reads in tumor & plasma are defined as 'alpha<=Ta'; normal reads in plasma are defined as 'alpha>=Na'
                if samples_type == 'reference_normal_plasma':
                    # identify normal reads in reference_normal_plasma
                    read_cov_of_criterion = get_number_of_reads_meeting_criterion(unique_alpha_values,
                                                                                  read_freq_of_unique_alpha_values,
                                                                                  params_values['Na'],
                                                                                  '>=')
                    if read_cov_of_criterion/num_read < params_values['Nrf']:
                        continue
                elif samples_type == 'tumor':
                    # identify tumor reads in tumor
                    read_cov_of_criterion = get_number_of_reads_meeting_criterion(unique_alpha_values,
                                                                                  read_freq_of_unique_alpha_values,
                                                                                  params_values['Ta'],
                                                                                  '<=')
                    if read_cov_of_criterion / num_read < params_values['Trf']:
                        continue
            data[marker_index] = {'cov': num_read, 'cov_meet_require': read_cov_of_criterion, 'fraction':read_cov_of_criterion/num_read}
    return(data)

# Identify PCR markers
def identify_markers_with_alpha_values(params_values,
                                       params_explanation,
                                       reference_normal_samples_list,
                                       tumor_samples_list,
                                       alpha_value_distribution_dir,
                                       input_file_suffix,
                                       output_file,
                                       tumor_type):
    num_ref_normal = float(len(reference_normal_samples_list))
    num_tumor = float(len(tumor_samples_list))
    all_markers = {} # marker_index -> {'freq_of_reference_normal_plasma':0, 'freq_of_tumor':0}
    data_normal = {} # sample -> {marker_index -> {'cov':20, 'cov_meet_require':15, 'fraction':0.9}}
    print('%d reference_normal_samples:'%num_ref_normal, flush=True)
    i = 0
    for s in reference_normal_samples_list:
        i += 1
        print('  (%d/%d) %s'%(i, num_ref_normal, s), end='', flush=True)
        file = '%s/%s.%s'%(alpha_value_distribution_dir, s, input_file_suffix)
        ret = parse_alpha_value_distribution_file_with_params_requirements(file, params_values, 'reference_normal_plasma')
        print('\t#marker: %d'%len(ret), flush=True)
        if len(ret) == 0: continue
        data_normal[s] = ret
        for m in ret.keys():
            if m not in all_markers:
                all_markers[m] = {'freq_of_reference_normal_plasma':0, 'freq_of_tumor':0}
            all_markers[m]['freq_of_reference_normal_plasma'] += 1

    data_tumor = {} # sample -> {marker_index -> {'cov':20, 'cov_meet_require':15, 'fraction':0.9}}
    print('%d tumor_samples:'%num_tumor, flush=True)
    i = 0
    for s in tumor_samples_list:
        i += 1
        print('  (%d/%d) %s' % (i, num_tumor, s), end='', flush=True)
        file = '%s/%s.%s' % (alpha_value_distribution_dir, s, input_file_suffix)
        ret = parse_alpha_value_distribution_file_with_params_requirements(file, params_values,
                                                                           'tumor')
        print('\t#marker: %d' % len(ret), flush=True)
        if len(ret) == 0: continue
        data_tumor[s] = ret
        for m in ret.keys():
            if m not in all_markers:
                all_markers[m] = {'freq_of_reference_normal_plasma':0, 'freq_of_tumor':0}
            all_markers[m]['freq_of_tumor'] += 1
    markers_identified = {}
    for m in sorted(all_markers.keys()):
        frac_of_reference_normal_plasma = all_markers[m]['freq_of_reference_normal_plasma'] / num_ref_normal
        if 'Tsf' in params_values['names_list']:
            # 'Tsf': Use fraction of tumors (float between 0 and 1)
            frac_of_tumor = all_markers[m]['freq_of_tumor'] / num_tumor
            if (frac_of_reference_normal_plasma>=params_values['Nsf']) and (frac_of_tumor>=params_values['Tsf']):
                markers_identified[m] = {'freq_of_reference_normal_plasma':all_markers[m]['freq_of_reference_normal_plasma'],
                                         'freq_of_tumor':all_markers[m]['freq_of_tumor'],
                                         'frac_of_reference_normal_plasma':frac_of_reference_normal_plasma,
                                         'frac_of_tumor':frac_of_tumor,
                                         }
        elif 'TsF' in params_values['names_list']:
            # 'Tsf': Use frequency of tumors (integer)
            frac_of_tumor = all_markers[m]['freq_of_tumor'] / num_tumor
            if (frac_of_reference_normal_plasma>=params_values['Nsf']) and (all_markers[m]['freq_of_tumor']>=params_values['TsF']):
                markers_identified[m] = {'freq_of_reference_normal_plasma':all_markers[m]['freq_of_reference_normal_plasma'],
                                         'freq_of_tumor':all_markers[m]['freq_of_tumor'],
                                         'frac_of_reference_normal_plasma':frac_of_reference_normal_plasma,
                                         'frac_of_tumor':frac_of_tumor,
                                         }
    print('Output: #marker: %d\n  %s' % (len(markers_identified), output_file), flush=True)
    with open(output_file, 'w') as fout:
        print_parameters_of_marker_discovery(fout, params_values, params_explanation, str_prefix='#')
        fout.write('marker_index\ttumor_type\tfrac_of_reference_normal_plasma\tfrac_of_tumor\tfreq_of_reference_normal_plasma\tfreq_of_tumor\n')
        for m in sorted(markers_identified.keys()):
            fout.write('%d\t%s\t%.2g\t%.2g\t%d\t%d\n'%(m,
                                                       tumor_type,
                                                       markers_identified[m]['frac_of_reference_normal_plasma'],
                                                       markers_identified[m]['frac_of_tumor'],
                                                       markers_identified[m]['freq_of_reference_normal_plasma'],
                                                       markers_identified[m]['freq_of_tumor']
                                                       ))
    return(markers_identified)

# Input:
# Format of each input marker file (generated by function 'identify_markers_with_alpha_values'):
# #type: hyper (Marker type: hyper- or hypo-methylation in tumor)
# #Nsf: 0.7 (Define a pure background (1): >=Fraction of reference normal samples have pure background noise (defined by Nrf and Na))
# #Nrf: 1 (Define a pure background (2): >=Fraction of normal reads in Nsf reference normal samples)
# #Na: 0.5 (Define normal reads in reference normal samples, as those reads with alpha values (<=Na for hyper-marker and >=Na for hypo-marker))
# #Tsf: 0.1 (Define tumor signal (1): >=Fraction (float btw 0 and 1) of tumor samples have tumor reads. You may use either Tsf or TsF)
# #Trf: 0.5 (Define tumor signal (2): >=Fraction tumor reads with tumor reads)
# #Ta: 1 (Define tumor reads in tumor, as those reads with alpha values (>=Ta for hyper-marker and <=Ta for hypo-marker))
# #cov: 5 (Define coverage requirement: #all_reads >= cov)
# #cpg: 3 (Define CpG number requirement: #all_cpg >= cpg)
# marker_index	tumor_type	frac_of_reference_normal_plasma	frac_of_tumor	freq_of_reference_normal_plasma	freq_of_tumor
# 1090	COAD	0.7	0.1	142	2
# 1821	COAD	0.76	0.2	153	4
# 3403	COAD	0.74	0.5	149	10
# ...
#
# max_marker_index: Union all markers whose marker_index <= this number.
#
# Output:
#   unified_markers: a dictionary {'marker_index':marker_index_list, 'type':marker_type_list}
def union_markers_from_multi_files(marker_files_list, max_marker_index):
    unified_markers = {'marker_index':[], 'type':[]}
    i = 0
    for marker_file in marker_files_list:
        i += 1
        sys.stderr.write('  (%d) %s\n'%(i, marker_file))
        with open(marker_file) as f:
            for line in f:
                if line.startswith('#') or line.startswith('marker'): continue
                marker_index, tumor_type, frac_of_reference_normal_plasma, frac_of_tumor, _, _ = line.rstrip().split('\t')
                marker_index = int(marker_index)
                if marker_index > max_marker_index: continue
                type_str = '%s_%s_%s'%(tumor_type, frac_of_reference_normal_plasma, frac_of_tumor)
                try:
                    idx = unified_markers['marker_index'].index(marker_index)
                    # marker_index exists
                    unified_markers['type'][idx] += ',%s'%type_str
                except ValueError:
                    # marker_index does not exist before
                    unified_markers['marker_index'].append(marker_index)
                    unified_markers['type'].append(type_str)
    sorted_index_list = sort_list_and_return_sorted_index(unified_markers['marker_index'])
    unified_markers_sorted = {'marker_index':[], 'type':[]}
    unified_markers_sorted['marker_index'] = [unified_markers['marker_index'][i] for i in sorted_index_list]
    unified_markers_sorted['type'] = [unified_markers['type'][i] for i in sorted_index_list]
    return(unified_markers_sorted)

def print_unified_markers(fid, unified_markers):
    n = len(unified_markers['marker_index'])
    fid.write('marker_index\tmarker_type\n')
    for i in range(n):
        fid.write('%d\t%s\n'%(unified_markers['marker_index'][i], unified_markers['type'][i]))

# Input file is the output of the function 'print_unified_markers'
# Output: a list of marker_index (strings)
def parse_markers(file, marker_index_upper_bound=1045106):
    markers_list = []
    with open(file) as f:
        next(f) # skip the header line
        for line in f:
            items = line.rstrip().split('\t')
            if int(items[0]) <= marker_index_upper_bound:
                markers_list.append( items[0] )
    return(markers_list)

# Input file of alpha value distributions:
# alpha_value_distribution file (from the output of the function 'summarize_mary_file_binary_meth_values_for_distribution_file'):
# marker_index    max_num_cpg num_read    unique_alpha_values read_freq_of_unique_alpha_values    unique_meth_counts  read_freq_of_unique_meth_counts
# 2   7   122 0.429,0.571,0.714,0.857,1   1,2,27,42,50    3,4,5,6,7   1,2,27,42,50
# 27  9   39  0.5,0.625,0.75,0.778,0.875,0.889,1  1,2,9,1,12,1,13 4,5,6,7,8   1,2,9,13,14
# 61  12  44  0.75,0.833,0.917,1  2,11,12,19  9,10,11,12  2,11,13,18
# 63  5   100 0.6,0.8,1   4,23,73 3,4,5   4,23,73
# 65  5   83  0,0.2,0.4,0.6,0.8,1 1,3,2,9,26,42   0,1,2,3,4,5 1,3,2,9,26,42
#
# markers_list: a list of markers_index (strings)
# marker_type: 'hyper' or 'hypo'
# Ta: float, the alpha value threshold
#
# Output:
#   profile: a 1D numpy.array, with the size == len(markers_list)
#
def generate_plasma_sample_profile_with_given_markers_by_parsing_alpha_value_distribution_file(file, markers_list, marker_type, Ta):
    n_markers = len(markers_list)
    profile = np.empty(n_markers, dtype=float)
    profile[:] = np.nan
    with gzip.open(file, 'rt') as f:
        next(f) # skip the header line
        for line in f:
            marker_index, max_num_cpg, num_read, unique_alpha_values_str, read_freq_of_unique_alpha_values_str, _, _ = line.rstrip().split('\t')
            try:
                idx = markers_list.index(marker_index)
                # marker_index exists
                unique_alpha_values = np.array(list(map(float, unique_alpha_values_str.split(','))))
                read_freq_of_unique_alpha_values = np.array(
                    list(map(int, read_freq_of_unique_alpha_values_str.split(','))))
                # identify tumor reads in tumor
                if 'hyper' in marker_type:
                    # tumor reads in tumor & plasma are defined as 'alpha>=Ta';
                    read_cov_of_criterion = get_number_of_reads_meeting_criterion(unique_alpha_values,
                                                                                  read_freq_of_unique_alpha_values,
                                                                                  Ta,
                                                                                  '>=')
                elif 'hypo' in marker_type:
                    # tumor reads in tumor & plasma are defined as 'alpha<=Ta';
                    read_cov_of_criterion = get_number_of_reads_meeting_criterion(unique_alpha_values,
                                                                                  read_freq_of_unique_alpha_values,
                                                                                  Ta,
                                                                                  '<=')
                profile[idx] = read_cov_of_criterion
            except ValueError:
                # marker_index does not exist
                continue
    return(profile)

# Input file of alpha value distributions (REQUIRE marker_index sorted):
# alpha_value_distribution file (from the output of the function 'summarize_mary_file_binary_meth_values_for_distribution_file'):
# marker_index    max_num_cpg num_read    unique_alpha_values read_freq_of_unique_alpha_values    unique_meth_counts  read_freq_of_unique_meth_counts
# 2   7   122 0.429,0.571,0.714,0.857,1   1,2,27,42,50    3,4,5,6,7   1,2,27,42,50
# 27  9   39  0.5,0.625,0.75,0.778,0.875,0.889,1  1,2,9,1,12,1,13 4,5,6,7,8   1,2,9,13,14
# 61  12  44  0.75,0.833,0.917,1  2,11,12,19  9,10,11,12  2,11,13,18
# 63  5   100 0.6,0.8,1   4,23,73 3,4,5   4,23,73
# 65  5   83  0,0.2,0.4,0.6,0.8,1 1,3,2,9,26,42   0,1,2,3,4,5 1,3,2,9,26,42
#
# markers_list: a list of markers_index (strings). REQUIRE marker_index sorted.
# marker_type: 'hyper' or 'hypo'
# Ta: float, the alpha value threshold
#
# Output:
#   profile: a 1D numpy.array, with the size == len(markers_list)
#
def generate_plasma_sample_profile_with_given_markers_by_parsing_alpha_value_distribution_file_quick_version(file, markers_list, marker_type, Ta):
    n_markers = len(markers_list)
    markers_list_int = list(map(int, markers_list))
    profile = np.empty(n_markers, dtype=float)
    profile[:] = np.nan
    current_marker_pointer_in_markers_list = 0
    with gzip.open(file, 'rt') as f:
        next(f) # skip the header line
        for line in f:
            marker_index, max_num_cpg, num_read, unique_alpha_values_str, read_freq_of_unique_alpha_values_str, _, _ = line.rstrip().split('\t')
            marker_index = int(marker_index)
            if current_marker_pointer_in_markers_list >= n_markers:
                break
            if marker_index > markers_list_int[current_marker_pointer_in_markers_list]:
                while True:
                    current_marker_pointer_in_markers_list += 1
                    if current_marker_pointer_in_markers_list >= n_markers:
                        break
                    if marker_index <= markers_list_int[current_marker_pointer_in_markers_list]:
                        break
                continue
            if current_marker_pointer_in_markers_list >= n_markers:
                break
            if marker_index < markers_list_int[current_marker_pointer_in_markers_list]:
                continue
            # now 'marker_index == markers_list_int[current_marker_pointer_in_markers_list]'
            idx = markers_list_int.index(marker_index)
            current_marker_pointer_in_markers_list += 1 # for the next round comparison
            unique_alpha_values = np.array(list(map(float, unique_alpha_values_str.split(','))))
            read_freq_of_unique_alpha_values = np.array(
                list(map(int, read_freq_of_unique_alpha_values_str.split(','))))
            # identify tumor reads in tumor
            if 'hyper' in marker_type:
                # tumor reads in tumor & plasma are defined as 'alpha>=Ta';
                read_cov_of_criterion = get_number_of_reads_meeting_criterion(unique_alpha_values,
                                                                              read_freq_of_unique_alpha_values,
                                                                              Ta,
                                                                              '>=')
            elif 'hypo' in marker_type:
                # tumor reads in tumor & plasma are defined as 'alpha<=Ta';
                read_cov_of_criterion = get_number_of_reads_meeting_criterion(unique_alpha_values,
                                                                              read_freq_of_unique_alpha_values,
                                                                              Ta,
                                                                              '<=')
            profile[idx] = read_cov_of_criterion
    return(profile)

# Input:
#   input_alpha_value_distribution_files_list: a list of input_alpha_value_distribution_files. Each file is for a sample
#   markers_list: a list of markers_index (strings)
#   marker_type: 'hyper' or 'hypo'
#   Ta: float, the alpha value threshold
#
# Output:
#   data: a 2D numpy.array. #samples X #markers. Rows are samples, columns are markers, values are tumor read counts.
#
def generate_matrix_for_many_plasma_samples_with_given_markers_by_parsing_alpha_value_distribution_files(input_alpha_value_distribution_files_list,
                                                                                                         markers_list,
                                                                                                         marker_type,
                                                                                                         Ta):
    data = []
    n_sample = len(input_alpha_value_distribution_files_list)
    for i in range(n_sample):
        input_sample_alpha_distr_file = input_alpha_value_distribution_files_list[i]
        print('  (%d/%d) %s'%(i+1, n_sample, input_sample_alpha_distr_file), flush=True)
        profile = generate_plasma_sample_profile_with_given_markers_by_parsing_alpha_value_distribution_file_quick_version(input_sample_alpha_distr_file,
                                                                                                                           markers_list,
                                                                                                                           marker_type,
                                                                                                                           Ta)
        data.append(profile)
    return(np.array(data))

###########
# Implement the marker discovery using min or max of alpha value difference (dynamic alpha threshold)
###########

# Input:
# alpha_value_distribution file (from the output of the function 'summarize_mary_file_binary_meth_values_for_distribution_file'):
# marker_index    max_num_cpg num_read    unique_alpha_values read_freq_of_unique_alpha_values    unique_meth_counts  read_freq_of_unique_meth_counts
# 2   7   122 0.429,0.571,0.714,0.857,1   1,2,27,42,50    3,4,5,6,7   1,2,27,42,50
# 27  9   39  0.5,0.625,0.75,0.778,0.875,0.889,1  1,2,9,1,12,1,13 4,5,6,7,8   1,2,9,13,14
# 61  12  44  0.75,0.833,0.917,1  2,11,12,19  9,10,11,12  2,11,13,18
# 63  5   100 0.6,0.8,1   4,23,73 3,4,5   4,23,73
# 65  5   83  0,0.2,0.4,0.6,0.8,1 1,3,2,9,26,42   0,1,2,3,4,5 1,3,2,9,26,42
# ...
#
# markers_list: None: count all total reads; ['2','10', ...]: count total reads in the given 'markers_list'
def slow_scan_one_alpha_value_distribution_file_obtain_total_reads(file, markers_list=None):
    total_reads = 0
    with gzip.open(file, 'rt') as f:
        next(f) # skip the first header line
        for line in f:
            items = line.rstrip().split()
            marker_index = items[0]
            if markers_list is None:
                num_read = int(items[2])
                total_reads += num_read
            else:
                if marker_index in markers_list:
                    num_read = int(items[2])
                    total_reads += num_read
    return(total_reads)

# Input:
# alpha_value_distribution file (from the output of the function 'summarize_mary_file_binary_meth_values_for_distribution_file'):
# marker_index    max_num_cpg num_read    unique_alpha_values read_freq_of_unique_alpha_values    unique_meth_counts  read_freq_of_unique_meth_counts
# 2   7   122 0.429,0.571,0.714,0.857,1   1,2,27,42,50    3,4,5,6,7   1,2,27,42,50
# 27  9   39  0.5,0.625,0.75,0.778,0.875,0.889,1  1,2,9,1,12,1,13 4,5,6,7,8   1,2,9,13,14
# 61  12  44  0.75,0.833,0.917,1  2,11,12,19  9,10,11,12  2,11,13,18
# 63  5   100 0.6,0.8,1   4,23,73 3,4,5   4,23,73
# 65  5   83  0,0.2,0.4,0.6,0.8,1 1,3,2,9,26,42   0,1,2,3,4,5 1,3,2,9,26,42
# ...
#
# markers_list: None: count all total reads; ['2','10', ...]: count total reads in the given 'markers_list'
#
# REQUIREMENT:
#   (1) markers_list is not EMPTY and must be sorted in increasing order of integer
#   (2) markers in alpha_value_distribution file must be sorted in the same way as in markers_list
def quick_scan_one_alpha_value_distribution_file_obtain_total_reads(file, markers_list):
    n_markers = len(markers_list)
    if n_markers == 0:
        sys.stderr('Error: the argument markers_list must not be EMPTY!\n  Func: quick_scan_one_alpha_value_distribution_file_obtain_total_reads\nExit.\n')
        sys.exit(-1)
    total_reads = 0
    curr_index_of_markers_list = 0  # 0-based index to markers_list
    with gzip.open(file, 'rt') as f:
        next(f) # skip the first header line
        for line in f:
            items = line.rstrip().split()
            marker_in_file = int(items[0])
            if marker_in_file > markers_list[curr_index_of_markers_list]:
                # move forward in markers_list, to match marker_index_in_file
                curr_index_of_markers_list += 1
                try:
                    while marker_in_file > markers_list[curr_index_of_markers_list]:
                        curr_index_of_markers_list += 1
                except IndexError as e:
                    # no marker_in_file in markers_list. We can exit from this search and report total reads
                    break # break from the loop "for line in f"
            if marker_in_file < markers_list[curr_index_of_markers_list]:
                # move forward in file, to match marker_index_in_file
                continue
            # Now we are sure marker_in_file == markers_list[curr_index_of_markers_list]
            num_read = int(items[2])
            total_reads += num_read
    return(total_reads)

# Input:
# alpha_value_distribution file (from the output of the function 'summarize_mary_file_binary_meth_values_for_distribution_file'):
# marker_index    max_num_cpg num_read    unique_alpha_values read_freq_of_unique_alpha_values    unique_meth_counts  read_freq_of_unique_meth_counts
# 2   7   122 0.429,0.571,0.714,0.857,1   1,2,27,42,50    3,4,5,6,7   1,2,27,42,50
# 27  9   39  0.5,0.625,0.75,0.778,0.875,0.889,1  1,2,9,1,12,1,13 4,5,6,7,8   1,2,9,13,14
# 61  12  44  0.75,0.833,0.917,1  2,11,12,19  9,10,11,12  2,11,13,18
# 63  5   100 0.6,0.8,1   4,23,73 3,4,5   4,23,73
# 65  5   83  0,0.2,0.4,0.6,0.8,1 1,3,2,9,26,42   0,1,2,3,4,5 1,3,2,9,26,42
# ...
#
# alpha_hists: a dictionary { marker_index:alpha_histgram_dictionary }, for example: {27:hist_dict, 63:hist_dict}, where hist_dict is a dictionary {'alpha_value':read_frequency}, e.g., {'0.7':200, '1.0':41}, where 200 means there are 200 reads with alpha==0.7 and 41 reads with alpha==1.0 among the pooled reads in all samples loaded. The input 'alpha_hists' can be an empty dictionary {}.
# marker2max_cpg_num: a dictionary {marker_index:'max_cpg_num'}.
#
def load_one_alpha_value_distribution_file(file, alpha_hists, marker2max_cpg_num):
    with gzip.open(file, 'rt') as f:
        next(f) # skip the first header line
        for line in f:
            items = line.rstrip().split()
            marker_index = int(items[0])
            max_num_cpg = items[1]
            unique_alpha_values_str = items[3]
            read_freq_of_unique_alpha_values_str = items[4]
            unique_alpha_values = unique_alpha_values_str.split(',')
            read_freq = list(map(int, read_freq_of_unique_alpha_values_str.split(',')))
            n = len(unique_alpha_values)
            alpha_hist_dict = {unique_alpha_values[i]: read_freq[i] for i in range(n)}
            if marker_index not in alpha_hists:
                alpha_hists[marker_index] = {}
            # update histgrams
            alpha_hists[marker_index] = mergeDict_by_adding_values_of_common_keys(alpha_hists[marker_index], alpha_hist_dict)
            if marker_index not in marker2max_cpg_num:
                marker2max_cpg_num[marker_index] = max_num_cpg

# Input file format is the same as input file of function 'load_one_alpha_value_distribution_file'
# Output:
# alpha_hists: a dictionary { marker_index:alpha_histgram_dictionary }, for example: {27:hist_dict, 63:hist_dict}, where hist_dict is a dictionary {'alpha_value':sample_frequency}, e.g., {'0.7':2, '1.0':1}, where the frequency 2 and 1 means there are 2 samples which have reads with alpha==0.7 and 1 sample which have reads with alpha==1.0. The input 'alpha_hists' can be an empty dictionary {}.
# marker2max_cpg_num: a dictionary {marker_index:'max_cpg_num'}.
#
def load_one_alpha_value_distribution_file_by_making_read_freq_as_ONE_for_one_sample(file, alpha_hists, marker2max_cpg_num):
    with gzip.open(file, 'rt') as f:
        next(f) # skip the first header line
        for line in f:
            items = line.rstrip().split()
            marker_index = int(items[0])
            max_num_cpg = items[1]
            unique_alpha_values_str = items[3]
            # read_freq_of_unique_alpha_values_str = items[4]
            unique_alpha_values = unique_alpha_values_str.split(',')
            # read_freq = list(map(int, read_freq_of_unique_alpha_values_str.split(',')))
            n = len(unique_alpha_values)
            alpha_hist_dict = {unique_alpha_values[i]: 1 for i in range(n)}
            if marker_index not in alpha_hists:
                alpha_hists[marker_index] = {}
            # update histgrams
            alpha_hists[marker_index] = mergeDict_by_adding_values_of_common_keys(alpha_hists[marker_index], alpha_hist_dict)
            if marker_index not in marker2max_cpg_num:
                marker2max_cpg_num[marker_index] = max_num_cpg


# Input:
#   file: format is the same as input file of function 'load_one_alpha_value_distribution_file'
#   sample_index: an integer sample_index of the input file. It will be put to the list of a unique alpha, like {alpha_value:[list of sample_index which have the reads with the same alpha_value]}. To save space, we use 1D numpy.array to replace
#
# Output:
# alpha_hists: a dictionary { marker_index:alpha_histgram_dictionary }, for example: {27:hist_dict, 63:hist_dict}, where hist_dict is a dictionary {'alpha_value':sample_frequency}, e.g., {'0.7':2, '1.0':1}, where the frequency 2 and 1 means there are 2 samples which have reads with alpha==0.7 and 1 sample which have reads with alpha==1.0. The input 'alpha_hists' can be an empty dictionary {}.
# marker2max_cpg_num: a dictionary {marker_index:'max_cpg_num'}. The input "marker2max_cpg_num" could be an empty dictionary {}.
#
def load_one_alpha_value_distribution_file_by_recording_sample_index_of_this_file(file, sample_index, alpha_hists, marker2max_cpg_num):
    with gzip.open(file, 'rt') as f:
        next(f) # skip the first header line
        for line in f:
            items = line.rstrip().split()
            marker_index = int(items[0])
            max_num_cpg = items[1]
            unique_alpha_values_str = items[3]
            # read_freq_of_unique_alpha_values_str = items[4]
            unique_alpha_values = unique_alpha_values_str.split(',')
            # read_freq = list(map(int, read_freq_of_unique_alpha_values_str.split(',')))
            n = len(unique_alpha_values)
            alpha_hist_dict = {unique_alpha_values[i]: {sample_index} for i in range(n)}
            if marker_index not in alpha_hists:
                alpha_hists[marker_index] = {}
            # update histgrams
            alpha_hists[marker_index] = mergeDict_by_union_valuesets_of_common_keys(alpha_hists[marker_index], alpha_hist_dict)
            if marker_index not in marker2max_cpg_num:
                marker2max_cpg_num[marker_index] = max_num_cpg

# Input:
#   file: format is the same as input file of function 'load_one_alpha_value_distribution_file'
#   sample_index: an integer sample_index of the input file. It will be put to the list of a unique alpha, like {alpha_value:[list of sample_index which have the reads with the same alpha_value]}. To save space, we use 1D numpy.array to replace
#
# Output:
# alpha_hists: a dictionary { marker_index:alpha_histgram_dictionary }, for example: {27:hist_dict, 63:hist_dict}, where hist_dict is a dictionary {'alpha_value':sample_frequency}, e.g., {'0.7':2, '1.0':1}, where the frequency 2 and 1 means there are 2 samples which have reads with alpha==0.7 and 1 sample which have reads with alpha==1.0. The input 'alpha_hists' can be an empty dictionary {}.
# marker2max_cpg_num: a dictionary {marker_index:'max_cpg_num'}. The input "marker2max_cpg_num" could be an empty dictionary {}.
#
def load_one_alpha_value_distribution_file_of_positive_sample_by_recording_sample_index_of_this_file_and_by_excluding_sample_index_which_appear_in_paired_negative_sample(file, sample_index, alpha_hists_positive, alpha_hists_negative, marker2max_cpg_num):
    with gzip.open(file, 'rt') as f:
        next(f) # skip the first header line
        for line in f:
            items = line.rstrip().split()
            marker_index = int(items[0])
            max_num_cpg = items[1]
            unique_alpha_values_str = items[3]
            # read_freq_of_unique_alpha_values_str = items[4]
            unique_alpha_values = unique_alpha_values_str.split(',')
            # read_freq = list(map(int, read_freq_of_unique_alpha_values_str.split(',')))
            n = len(unique_alpha_values)
            if marker_index not in alpha_hists_negative:
                # negative sample does not have this marker, so all unique alpha values can add this sample_index to their sample sets
                # We add this sample_index to the sample_set of each  unique alpha value
                alpha_hist_dict = {unique_alpha_values[i]: {sample_index} for i in range(n)}
            else:
                # negative sample has this marker, we further check each unique alpha value one by one
                # if alpha_value appears in
                alpha_hist_dict = {}
                marker_of_negative = alpha_hists_negative[marker_index]
                for i in range(n):
                    a = unique_alpha_values[i]
                    if a not in marker_of_negative:
                        alpha_hist_dict[a] = {sample_index}
            # update histgrams
            if len(alpha_hist_dict) > 0:
                if marker_index not in alpha_hists_positive:
                    alpha_hists_positive[marker_index] = {}
                alpha_hists_positive[marker_index] = mergeDict_by_union_valuesets_of_common_keys(alpha_hists_positive[marker_index], alpha_hist_dict)
            if marker_index not in marker2max_cpg_num:
                marker2max_cpg_num[marker_index] = max_num_cpg



# Input:
#   file: format is the same as the output file of function 'write_alpha_value_distribution_file_with_recorded_sample_index_set'
# marker_index    max_num_cpg num_read  unique_alpha_values read_freq_of_unique_alpha_values
# 3	8	3	0.875,1	14,3_21
# 9	13	6	0.385,0.462,0.538,0.615,0.692,0.769,0.833,0.846,0.923,1	18,8_14_18,8_11_13_14_18,8_11_14_18,8_11_14_18,8_13_14_18_24,8,8_11_13_14_18,8_11_13_14_18,8_11_13_14_18
# 15	3	1	0.667	1
# 20	4	4	0.75,1	18,12_21_23
# 25	7	1	0.857	9
# ...
#   sample_index: an integer sample_index of the input file. It will be put to the list of a unique alpha, like {alpha_value:[list of sample_index which have the reads with the same alpha_value]}. To save space, we use 1D numpy.array to replace
#
# Output:
# alpha_hists: a dictionary { marker_index:alpha_histgram_dictionary }, for example: {27:hist_dict, 63:hist_dict}, where hist_dict is a dictionary {'alpha_value':sample_frequency}, e.g., {'0.7':2, '1.0':1}, where the frequency 2 and 1 means there are 2 samples which have reads with alpha==0.7 and 1 sample which have reads with alpha==1.0. The input 'alpha_hists' can be an empty dictionary {}.
# marker2max_cpg_num: a dictionary {marker_index:'max_cpg_num'}. The input "marker2max_cpg_num" could be an empty dictionary {}.
#
def load_one_alpha_value_distribution_file_that_has_sample_index_sets(file, alpha_hists, marker2max_cpg_num):
    with gzip.open(file, 'rt') as f:
        next(f) # skip the first header line
        for line in f:
            items = line.rstrip().split()
            marker_index = int(items[0])
            max_num_cpg = items[1]
            unique_alpha_values_str = items[3]
            sample_index_sets_of_unique_alpha_values_str_list = items[4].split(',')
            unique_alpha_values = unique_alpha_values_str.split(',')
            n = len(unique_alpha_values)
            alpha_hist_dict = {unique_alpha_values[i]: set(sample_index_sets_of_unique_alpha_values_str_list[i].split('_')) for i in range(n)}
            if marker_index not in alpha_hists:
                alpha_hists[marker_index] = {}
            # update histgrams
            alpha_hists[marker_index] = mergeDict_by_union_valuesets_of_common_keys(alpha_hists[marker_index], alpha_hist_dict)
            if marker_index not in marker2max_cpg_num:
                marker2max_cpg_num[marker_index] = max_num_cpg

# Input:
#   file: format is the same as the output file of function 'write_alpha_value_distribution_file_with_recorded_sample_index_set'
# marker_index    max_num_cpg num_read  unique_alpha_values read_freq_of_unique_alpha_values
# 29	8	5	0.75,0.875,1	5:0.125,10:0.333_25:0.4_5:0.375_6:0.333_8:0.571,10:0.667_25:0.6_5:0.5_6:0.667_8:0.429
# 223	8	4	0,0.125,0.75,0.875,1	14:0.1_8:0.2,8:0.1,14:0.2_25:0.2_3:0.0909,14:0.1_25:0.2_3:0.273_8:0.2,14:0.6_25:0.6_3:0.636_8:0.5
# ...
#   sample_index: an integer sample_index of the input file. It will be put to the list of a unique alpha, like {alpha_value:[list of sample_index which have the reads with the same alpha_value]}. To save space, we use 1D numpy.array to replace
#
# Output:
# alpha_hists: a dictionary { marker_index:alpha_histgram_dictionary }, for example: {27:hist_dict, 63:hist_dict}, where hist_dict is a dictionary {'alpha_value':sample_frequency}, e.g., {'0.7':2, '1.0':1}, where the frequency 2 and 1 means there are 2 samples which have reads with alpha==0.7 and 1 sample which have reads with alpha==1.0. The input 'alpha_hists' can be an empty dictionary {}.
# marker2max_cpg_num: a dictionary {marker_index:'max_cpg_num'}. The input "marker2max_cpg_num" could be an empty dictionary {}.
#
def load_one_alpha_value_distribution_file_that_has_sample_index_sets_and_read_fractions(file, alpha_hists, marker2max_cpg_num, marker2samplenum):
    with gzip.open(file, 'rt') as f:
        next(f) # skip the first header line
        for line in f:
            items = line.rstrip().split()
            marker_index = int(items[0])
            max_num_cpg = items[1]
            sample_num = int(items[2]) # the number of unique samples that appear in 'sample_index_sets_of_unique_alpha_values_str_list'
            unique_alpha_values_str = items[3]
            sample_index_sets_of_unique_alpha_values_str_list = items[4].split(',')
            unique_alpha_values = unique_alpha_values_str.split(',')
            n = len(unique_alpha_values)
            # alpha_hist_dict = {unique_alpha_values[i]: set(sample_index_sets_of_unique_alpha_values_str_list[i].split('_')) for i in range(n)}
            alpha_hist_dict = dict([])
            for i in range(n):
                alpha_hist_dict[unique_alpha_values[i]] = dict([])
                sampleindex_and_readfraction_str_list = sample_index_sets_of_unique_alpha_values_str_list[i].split('_')
                for sampleindex_and_readfraction_str in sampleindex_and_readfraction_str_list:
                    sampleindex, readfraction = sampleindex_and_readfraction_str.split(':')
                    readfraction = float(readfraction)
                    alpha_hist_dict[unique_alpha_values[i]][sampleindex] = readfraction
            if marker_index not in alpha_hists:
                alpha_hists[marker_index] = {}
            # update histgrams
            for alpha in alpha_hist_dict:
                if alpha not in alpha_hists[marker_index]:
                    alpha_hists[marker_index][alpha] = {}
                alpha_hists[marker_index][alpha] = mergeDict_by_adding_values_of_common_keys(alpha_hists[marker_index][alpha], alpha_hist_dict[alpha])
            if marker_index not in marker2max_cpg_num:
                marker2max_cpg_num[marker_index] = max_num_cpg
            if marker_index not in marker2samplenum:
                marker2samplenum[marker_index] = sample_num

# Input:
#   file: format is the same as input file of function 'load_one_alpha_value_distribution_file'
#   sample_index: an integer sample_index of the input file. It will be put to the list of a unique alpha, like {alpha_value:[list of sample_index which have the reads with the same alpha_value]}. To save space, we use 1D numpy.array to replace
#
# Output:
# alpha_hists: a dictionary { marker_index:alpha_histgram_dictionary }, for example: {27:hist_dict, 63:hist_dict}, where hist_dict is a dictionary {'alpha_value':sample_frequency}, e.g., {'0.7':2, '1.0':1}, where the frequency 2 and 1 means there are 2 samples which have reads with alpha==0.7 and 1 sample which have reads with alpha==1.0. The input 'alpha_hists' can be an empty dictionary {}.
# marker2max_cpg_num: a dictionary {marker_index:'max_cpg_num'}. The input "marker2max_cpg_num" could be an empty dictionary {}.
#
def load_one_alpha_value_distribution_file_by_recording_sample_index_and_read_fraction_of_this_file(file, sample_index, min_read_coverage, alpha_hists, marker2max_cpg_num):
    with gzip.open(file, 'rt') as f:
        next(f) # skip the first header line
        for line in f:
            items = line.rstrip().split()
            total_read_count = float(items[2])
            if total_read_count < min_read_coverage:
                continue
            marker_index = int(items[0])
            max_num_cpg = items[1]
            unique_alpha_values_str = items[3]
            read_freq_of_unique_alpha_values_str = items[4]
            unique_alpha_values = unique_alpha_values_str.split(',')
            read_freq = list(map(int, read_freq_of_unique_alpha_values_str.split(',')))
            n = len(unique_alpha_values)
            alpha_hist_dict = {unique_alpha_values[i]:{'%d:%.3g'%(sample_index,read_freq[i]/total_read_count)} for i in range(n)}
            if marker_index not in alpha_hists:
                alpha_hists[marker_index] = {}
            # update histgrams
            alpha_hists[marker_index] = mergeDict_by_union_valuesets_of_common_keys(alpha_hists[marker_index], alpha_hist_dict)
            if marker_index not in marker2max_cpg_num:
                marker2max_cpg_num[marker_index] = max_num_cpg


#
# Input:
# alpha_hists: a dictionary { marker_index:{'alpha_threshold':threshold, 'alpha2freq':alpha_histgram_dictionary} }. for example, alpha_histgram_dictionary is {27:hist_dict, 63:hist_dict}, where hist_dict is a dictionary {'0.7':2, '1.0':4}.
# marker2max_cpg_num: a dictionary {marker_index:max_cpg_num}. NOTE: It is from the output of function 'compare_background_vs_cancer_alpha_value_distribution_files'
#
# Output:
# alpha_value_distribution_with_threshold file (from the output of the function 'load_one_alpha_value_distribution_file' or 'combine_multi_alpha_histgram_files'):
# marker_index    max_num_cpg num_read  unique_alpha_values read_freq_of_unique_alpha_values
# 2   7   122  0.429,0.571,0.714,0.857,1   1,2,27,42,50
# 27  9   39  0.5,0.625,0.75,0.778,0.875,0.889,1  1,2,9,1,12,1,13
# 61  12  44  0.75,0.833,0.917,1  2,11,12,19
# 63  5   100  0.6,0.8,1   4,23,73
# 65  5   83 0.6,0.8,1 9,26,42
# ...
#
def write_alpha_value_distribution_file(fout, alpha_hists, marker2max_cpg_num):
    marker_index_list = sorted(list(set(alpha_hists.keys())))
    fout.write(
        'marker_index\tmax_num_cpg\tnum_read\talpha_threshold\tunique_alpha_values\tread_freq_of_unique_alpha_values\n')
    for marker_index in marker_index_list:
        num_reads = sum(alpha_hists[marker_index].values())
        str_for_unique_alpha_values, read_freq_of_unique_alpha_values_str = convert_str2int_dict_to_str(alpha_hists[marker_index])
        fout.write('%d\t%s\t%d\t%s\t%s\n' % (marker_index,
                                             marker2max_cpg_num[marker_index],
                                             num_reads,
                                             str_for_unique_alpha_values,
                                             read_freq_of_unique_alpha_values_str
                                             ))


#
# Input:
# alpha_hists: a dictionary { marker_index:{'alpha_threshold':threshold, 'alpha2freq':alpha2sampleindexset_dictionary} }. for example, alpha_histgram_dictionary is {27:sampleindexset, 63:sampleindexset}, where hist_dict is a dictionary {'0.7':{1,2}, '1.0':{3,4}}.
# marker2max_cpg_num: a dictionary {marker_index:max_cpg_num}. NOTE: It is from the output of function 'compare_background_vs_cancer_alpha_value_distribution_files'
#
# Output:
# alpha_value_distribution_with_threshold file (from the output of the function 'load_one_alpha_value_distribution_file' or 'combine_multi_alpha_histgram_files'):
# marker_index    max_num_cpg num_read  unique_alpha_values read_freq_of_unique_alpha_values
# 3	8	3	0.875,1	14,3_21
# 9	13	6	0.385,0.462,0.538,0.615,0.692,0.769,0.833,0.846,0.923,1	18,8_14_18,8_11_13_14_18,8_11_14_18,8_11_14_18,8_13_14_18_24,8,8_11_13_14_18,8_11_13_14_18,8_11_13_14_18
# 15	3	1	0.667	1
# 20	4	4	0.75,1	18,12_21_23
# 25	7	1	0.857	9
# ...
#
def write_alpha_value_distribution_file_with_recorded_sample_index_set(fout, alpha_hists, marker2max_cpg_num):
    marker_index_list = sorted(list(set(alpha_hists.keys())))
    fout.write(
        'marker_index\tmax_num_cpg\tnum_read\talpha_threshold\tunique_alpha_values\tread_freq_of_unique_alpha_values\n')
    for marker_index in marker_index_list:
        a = set([])
        for v in alpha_hists[marker_index].values():
            a.update(v)
        num_reads = len(a)
        str_for_unique_alpha_values, read_freq_of_unique_alpha_values_str = convert_str2intset_dict_to_str(alpha_hists[marker_index])
        fout.write('%d\t%s\t%d\t%s\t%s\n' % (marker_index,
                                             marker2max_cpg_num[marker_index],
                                             num_reads,
                                             str_for_unique_alpha_values,
                                             read_freq_of_unique_alpha_values_str
                                             ))

#
# Input:
# alpha_hists: a dictionary { marker_index:{'alpha_threshold':threshold, 'alpha2freq':alpha2sampleindexset_dictionary} }. for example, alpha_histgram_dictionary is {27:sampleindexset_with_read_frac, 63:sampleindexset_with_read_frac}, where hist_dict is a dictionary {'0.7':{'1_0.3','2_0.1'}, '1.0':{'3_0.1','4_0.05'}}.
# marker2max_cpg_num: a dictionary {marker_index:max_cpg_num}. NOTE: It is from the output of function 'compare_background_vs_cancer_alpha_value_distribution_files'
#
# Output:
# alpha_value_distribution_with_threshold file (from the output of the function 'load_one_alpha_value_distribution_file_by_recording_sample_index_of_this_file' or 'combine_multi_alpha_histgram_files'):
# marker_index    max_num_cpg num_read  unique_alpha_values read_freq_of_unique_alpha_values
# 3	8	3	0.875,1	14,3_21
# 9	13	6	0.385,0.462,0.538,0.615,0.692,0.769,0.833,0.846,0.923,1	18:0.3,8:0.1_14:0.2_18:0.1,8:0.2_11:0.05_13:0.5_14:0.01_18:0.01,8:0.02_11:0.03_14:0.5,8:0.03_11:0.03_14:0.03_18:0.03,8:0.03_13:0.03_14:0.03_18:0.03_24:0.03,8:0.03,8:0.03_11:0.03_13:0.03_14:0.03_18:0.03,8:0.03_11:0.03_13:0.03_14:0.03_18:0.03,8:0.03_11:0.03_13:0.03_14:0.03_18:0.03
# 15	3	1	0.667	1:0.05
# 20	4	4	0.75,1	18:0.01,12:0.1_21:0.2_23:0.3
# 25	7	1	0.857	9:0.1
# ...
#
#
# For each unique alpha value's read_freq, its format is "sampleindex1:readfraction_sampleindex2:readfraction_sampleindex3:readfraction"
#
def write_alpha_value_distribution_file_with_recorded_sample_index_set_and_read_fractions(fout, alpha_hists, marker2max_cpg_num):
    marker_index_list = sorted(list(set(alpha_hists.keys())))
    fout.write(
        'marker_index\tmax_num_cpg\tnum_read\talpha_threshold\tunique_alpha_values\tread_freq_of_unique_alpha_values\n')
    for marker_index in marker_index_list:
        a = set([])
        for v in alpha_hists[marker_index].values():
            trimmed_v = [str_.split(':')[0] for str_ in v] # extract sample_index from string 'sampleindex:readfraction', such as '3:0.0045'
            a.update(trimmed_v)
        num_reads = len(a)
        str_for_unique_alpha_values, read_freq_of_unique_alpha_values_str = convert_str2strset_dict_to_str(alpha_hists[marker_index])
        fout.write('%d\t%s\t%d\t%s\t%s\n' % (marker_index,
                                             marker2max_cpg_num[marker_index],
                                             num_reads,
                                             str_for_unique_alpha_values,
                                             read_freq_of_unique_alpha_values_str
                                             ))

# Input:
#   files_list: a list of alpha histogram files (format is as the output file of function 'write_alpha_value_distribution_file')
#   frequency: Use read frequency ('read_frequency') or sample frequency ('sample_frequency') when loading and accumulating the alpha histogram files, or sample index set ('sample_index_set')
#
# alpha_hists: a dictionary { 'marker_index':alpha_histgram_dictionary }, for example: {'27':hist_dict, '63':hist_dict}, where hist_dict is a dictionary {'0.7':2, '1.0':4}. The input 'alpha_hists' can be an empty dictionary {}.
# marker2max_cpg_num: a dictionary {'marker':max_cpg_num}.
#
def combine_multi_alpha_histgram_files(files_list, frequency_type='read_frequency'):
    combined_alpha_hists = {}
    marker2max_cpg_num = {}
    n = len(files_list)
    i = 0
    for filename in files_list:
        i += 1
        print('  (%d/%d) %s, '%(i, n, filename), datetime.now(), flush=True)
        if frequency_type == 'read_frequency':
            load_one_alpha_value_distribution_file(filename, combined_alpha_hists, marker2max_cpg_num)
        elif frequency_type == 'sample_frequency':
            load_one_alpha_value_distribution_file_by_making_read_freq_as_ONE_for_one_sample(filename, combined_alpha_hists, marker2max_cpg_num)
        elif frequency_type == 'sample_index_set':
            load_one_alpha_value_distribution_file_by_recording_sample_index_of_this_file(filename, i,
                                                                                          combined_alpha_hists,
                                                                                          marker2max_cpg_num)
        elif 'sample_index_set_with_read_frac' in frequency_type:
            min_coverage = int(re.search('mincov(\d+)', frequency_type).group(1))
            load_one_alpha_value_distribution_file_by_recording_sample_index_and_read_fraction_of_this_file(filename, i, min_coverage,
                                                                                          combined_alpha_hists,
                                                                                          marker2max_cpg_num)
    return( (combined_alpha_hists,marker2max_cpg_num) )


# Input:
#   files_list: a list of alpha histogram files (format is as the output file of function 'write_alpha_value_distribution_file')
#   frequency: Use read frequency ('read_frequency'), or sample frequency ('sample_frequency') when loading and accumulating the alpha histogram files, or sample index set ('sample_index_set')
#
# alpha_hists: a dictionary { 'marker_index':alpha_histgram_dictionary }, for example: {'27':hist_dict, '63':hist_dict}, where hist_dict is a dictionary {'0.7':2, '1.0':4}. The input 'alpha_hists' can be an empty dictionary {}.
# marker2max_cpg_num: a dictionary {'marker':max_cpg_num}.
#
def combine_multi_paired_alpha_histgram_files(files_list_for_positive, files_list_for_negative, frequency_type='sample_index_set'):
    combined_alpha_hists_for_positive = {}
    marker2max_cpg_num = {}
    n = len(files_list_for_positive)
    for i in range(n):
        filename_positive = files_list_for_positive[i]
        filename_negative = files_list_for_negative[i]
        print('  (%d/%d)  '%(i+1, n), datetime.now(), '\n    +: %s\n    -: %s'%(filename_positive, filename_negative), flush=True)
        if frequency_type == 'sample_index_set':
            alpha_hists_for_negative = {}
            load_one_alpha_value_distribution_file_by_recording_sample_index_of_this_file(filename_negative, i + 1,
                                                                                          alpha_hists_for_negative,
                                                                                          marker2max_cpg_num)

            load_one_alpha_value_distribution_file_of_positive_sample_by_recording_sample_index_of_this_file_and_by_excluding_sample_index_which_appear_in_paired_negative_sample(
                filename_positive, i+1,
                combined_alpha_hists_for_positive,
                alpha_hists_for_negative,
                marker2max_cpg_num)
    return( (combined_alpha_hists_for_positive,marker2max_cpg_num) )


# Input:
#   alpha2freq: a dictionary {alpha: read_freq}. For example, {'0.2':20}
# Output:
#   meth_histgram: a dictionary structure {alpha:read_freq}. For example, {0.2:10}.
def filter_alpha2freq_by_alpha(alpha2freq, alpha_cutoff, direction_to_keep_alpha2freq='>='):
    ret_alpha2freq = {}
    if direction_to_keep_alpha2freq=='>=':
        # keep all alpha2freq and their meth_strings if their alpha>=alpha_cutoff
        for alpha, freq in alpha2freq.items():
            if float(alpha)>=alpha_cutoff:
                ret_alpha2freq[alpha] = freq
    elif direction_to_keep_alpha2freq=='>':
        # keep all alpha2freq and their meth_strings if their alpha>alpha_cutoff
        for alpha, freq in alpha2freq.items():
            if float(alpha) > alpha_cutoff:
                ret_alpha2freq[alpha] = freq
    elif direction_to_keep_alpha2freq=='<=':
        # keep all alpha2freq and their meth_strings if their alpha<=alpha_cutoff
        for alpha, freq in alpha2freq.items():
            if float(alpha) <= alpha_cutoff:
                ret_alpha2freq[alpha] = freq
    elif direction_to_keep_alpha2freq=='<':
        # keep all alpha2freq and their meth_strings if their alpha<alpha_cutoff
        for alpha, freq in alpha2freq.items():
            if float(alpha) < alpha_cutoff:
                ret_alpha2freq[alpha] = freq
    return(ret_alpha2freq)

# Input:
#   alpha_list: a list of alpha_values with a specific order (in descending or increasing order)
#   freq_cumsum: a 1D numpy.array with the same length and the same order of alpha_list, and freq_cumsum[i] is the accumulated frequency sum of the alpha_value alpha_list[i]
# Output:
#   ret_alpha2freq: a dictionary {alpha_value:freq_cum_sum}
def filter_by_freq_cumsum_and_create_alpha2freqcumsum(alpha_cutoff, alpha_list, freq_cumsum, direction_to_keep_alpha2freq='>='):
    ret_alpha2freq = {}
    n = len(alpha_list)
    if direction_to_keep_alpha2freq == '>=':
        # keep all alpha2freq and their meth_strings if their alpha>=alpha_cutoff
        for i in range(n):
            alpha = alpha_list[i]
            if float(alpha) >= alpha_cutoff:
                ret_alpha2freq[alpha] = freq_cumsum[i]
    elif direction_to_keep_alpha2freq == '>':
        # keep all alpha2freq and their meth_strings if their alpha>alpha_cutoff
        for i in range(n):
            alpha = alpha_list[i]
            if float(alpha) > alpha_cutoff:
                ret_alpha2freq[alpha] = freq_cumsum[i]
    elif direction_to_keep_alpha2freq == '<=':
        # keep all alpha2freq and their meth_strings if their alpha<=alpha_cutoff
        for i in range(n):
            alpha = alpha_list[i]
            if float(alpha) <= alpha_cutoff:
                ret_alpha2freq[alpha] = freq_cumsum[i]
    elif direction_to_keep_alpha2freq == '<':
        # keep all alpha2freq and their meth_strings if their alpha<alpha_cutoff
        for i in range(n):
            alpha = alpha_list[i]
            if float(alpha) < alpha_cutoff:
                ret_alpha2freq[alpha] = freq_cumsum[i]
    return (ret_alpha2freq)

# arr_neg, arr_pos: two 1D numpy.array with the same size. Find the index on which
#     arr_neg_cumsum[index] <= arr_neg_cumsum_threshold
#     arr_pos_cumsum[index] >= arr_pos_cumsum_threshold
# If there exist multiple index satisfying the above criteria, choose the largest index
def identify_turning_point_of_two_cumsum_array(arr_neg, arr_pos, arr_neg_cumsum_threshold, arr_pos_cumsum_threshold):
    arr_pos_cumsum = np.cumsum(arr_pos) - arr_pos_cumsum_threshold
    arr_neg_cumsum = np.cumsum(arr_neg) - arr_neg_cumsum_threshold
    n = len(arr_pos_cumsum)
    index_list = [i for i in range(n) if ((arr_pos_cumsum[i]>=0) and (arr_neg_cumsum[i]<=0))]
    if len(index_list)==0:
        # No turning point
        return(-2)
    else:
        # if (index_list[-1]+1) < (n-1):
        if (index_list[-1]) < (n - 1):
            return(index_list[-1]+1)
        else:
            if (len(index_list)>=2):
                if (index_list[-2]) < (n - 1):
                    return(index_list[-2]+1)
            # -1 means the index should be the one greater than n
            return(-1)

# Input:
#   alpha2freq_of_pos_class: a dictionary {'alpha_value':frequency} for positive class, e.g. {'0.7':2, '0.9':4}
#   alpha2freq_of_neg_class: a dictionary {'alpha_value':frequency} for negative class, e.g. {'0.7':2, '0.9':4}
#   marker_type: 'hyper' marker to identify alpha threshold which make more frequency whose alpha > threshold in positive class, than those in negative class
#                'hypo' marker to identify alpha threshold which make more frequency whose alpha < threshold in positive class, than those in negative class
def identify_alpha_threshold_by_alpha2freq_of_pos_and_neg_class(alpha2freq_of_neg_class,
                                                                alpha2freq_of_pos_class,
                                                                max_freq_cumsum_of_neg,
                                                                min_freq_cumsum_of_pos,
                                                                marker_type='hyper'):
    if 'hyper' in marker_type:
        if len(alpha2freq_of_pos_class)>0 and len(alpha2freq_of_neg_class)==0:
            alpha_threshold = 0 # any reads with any alpha values should be used. So let alpha_threshold==0 for hyper markers
            return (alpha_threshold)
        if len(alpha2freq_of_pos_class)>0 and len(alpha2freq_of_neg_class)>0:
            alpha_union_list = sorted(list(set(list(alpha2freq_of_pos_class.keys()) + list(alpha2freq_of_neg_class.keys()))), reverse=True) # decreasing order
            freq_array_of_pos = np.array([alpha2freq_of_pos_class[a] if a in alpha2freq_of_pos_class else 0 for a in alpha_union_list])
            freq_array_of_neg = np.array([alpha2freq_of_neg_class[a] if a in alpha2freq_of_neg_class else 0 for a in alpha_union_list])
    elif 'hypo' in marker_type:
        if len(alpha2freq_of_pos_class)>0 and len(alpha2freq_of_neg_class)==0:
            alpha_threshold = 1 # any reads with any alpha values should be used. So let alpha_threshold==1 for hypo markers
            return (alpha_threshold)
        if len(alpha2freq_of_pos_class)>0 and len(alpha2freq_of_neg_class)>0:
            alpha_union_list = sorted(list(set(list(alpha2freq_of_pos_class.keys()) + list(alpha2freq_of_neg_class.keys()))))  # increasing order
            freq_array_of_pos = np.array([alpha2freq_of_pos_class[a] if a in alpha2freq_of_pos_class else 0 for a in alpha_union_list])
            freq_array_of_neg = np.array([alpha2freq_of_neg_class[a] if a in alpha2freq_of_neg_class else 0 for a in alpha_union_list])
    alpha_index = identify_turning_point_of_two_cumsum_array(freq_array_of_neg, freq_array_of_pos, max_freq_cumsum_of_neg, min_freq_cumsum_of_pos)
    if alpha_index==-2:
        alpha_threshold = None
    else:
        if alpha_index==-1:
            alpha_threshold = None
            # if marker_type=='hyper':
            #     alpha_threshold = 0
            # elif marker_type == 'hypo':
            #     alpha_threshold = 1
        else:
            alpha_threshold = float(alpha_union_list[alpha_index])
            # if len(alpha2freq_of_neg_class) == 1:
            #     # Remove the following case which identifies hyper marker:
            #     # positive class: {'0.857':2, '0.286':1, '0.167':2, '0.143':4, '0':8}
            #     # negative class: {'0.167':2}
            #     # alpha_union_list: ['0.857', '0.286', '0.167', '0.143', '0']
            #     # identify_turning_point_of_two_cumsum_array returns alpha_index=4
            #     # But actually if alpha_threshold=='0', then negative class is incorrect for our purpose.
            #     if (('hyper' in marker_type) and (alpha_threshold<float(alpha2freq_of_neg_class.values[0]))) or (('hypo' in marker_type) and (alpha_threshold>float(alpha2freq_of_neg_class.values[0]))):
            #         alpha_threshold = None
    return(alpha_threshold)


# alpha_union_list: the list of alpha values, which are not only union of alpha values of pos_class and neg_class, but also with the order of alpha values, with respect to 'hyper' (in decreasing order) or 'hypo' (in increasing order).
# alpha2sampleindexset_of_neg_class, alpha2sampleindexset_of_pos_class: two dictionaries {'alpha_value':{sample_index}}
# Algorithm:
#    Step 1: accumulate sample_index_set of each alpha_value, by the alpha_value order of alpha_union_list
#    Step 2: calcualte size of accumulated sample_index_set of each alpha_value
#    Step 3: compute the following to determine the turning point. This step is the same as function 'identify_alpha_threshold_by_alpha2freq_of_pos_and_neg_class'
#            arr_neg_cumsum[index] <= arr_neg_cumsum_threshold
#            arr_pos_cumsum[index] >= arr_pos_cumsum_threshold
#            If there exist multiple index satisfying the above criteria, choose the largest index
#    Step 4: return the index of turning point (i.e., the index of alpha threshold in alpha_union_list)
#
def identify_turning_point_of_two_alpha2sampleindexset(alpha_union_list, alpha2sampleindexset_of_neg_class, alpha2sampleindexset_of_pos_class, arr_neg_cumsum_threshold, arr_pos_cumsum_threshold):
    n_alpha = len(alpha_union_list)
    # implement cumsum using set union operator
    alpha2cumset_neg = {a:set([]) for a in alpha_union_list}
    alpha2cumset_pos = dict(alpha2cumset_neg)
    for i in range(n_alpha):
        a = alpha_union_list[i]
        if i == 0:
            alpha2cumset_neg[a] = alpha2sampleindexset_of_neg_class[a] if a in alpha2sampleindexset_of_neg_class else set([])
            alpha2cumset_pos[a] = alpha2sampleindexset_of_pos_class[a] if a in alpha2sampleindexset_of_pos_class else set([])
        else:
            a_prev = alpha_union_list[i - 1]
            alpha2cumset_neg[a] = alpha2sampleindexset_of_neg_class[a].union(alpha2cumset_neg[a_prev]) if a in alpha2sampleindexset_of_neg_class else alpha2cumset_neg[a_prev]
            alpha2cumset_pos[a] = alpha2sampleindexset_of_pos_class[a].union(alpha2cumset_pos[a_prev]) if a in alpha2sampleindexset_of_pos_class else alpha2cumset_pos[a_prev]
    arr_neg_cumsum = np.array([len(alpha2cumset_neg[a]) for a in alpha_union_list])
    arr_pos_cumsum = np.array([len(alpha2cumset_pos[a]) for a in alpha_union_list])
    index_list = [i for i in range(n_alpha) if ((arr_pos_cumsum[i]>=arr_pos_cumsum_threshold) and (arr_neg_cumsum[i]<=arr_neg_cumsum_threshold))]
    if len(index_list)==0:
        # No turning point
        return( (-2, arr_neg_cumsum, arr_pos_cumsum) )
    else:
        # if (index_list[-1]+1) < (n_alpha-1):
        if (index_list[-1]) < (n_alpha - 1):
            return( (index_list[-1]+1, arr_neg_cumsum, arr_pos_cumsum) )
        else:
            if len(index_list)>=2:
                if (index_list[-2]) < (n_alpha - 1):
                    return( (index_list[-2]+1, arr_neg_cumsum, arr_pos_cumsum) )
            # -1 means the index should be the one greater than n
            return( (-1, arr_neg_cumsum, arr_pos_cumsum) )


# Input:
#   alpha2freq_of_pos_class: a dictionary {'alpha_value':set(sample_index)} for positive class, e.g. {'0.7':[1,2], '0.9':[2,4]}
#   alpha2freq_of_neg_class: a dictionary {'alpha_value':set(sample_index)} for negative class, e.g. {'0.7':[1,2], '0.9':[2,4]}
#   marker_type: 'hyper' marker to identify alpha threshold which make more sample frequency whose alpha > threshold in positive class, than those in negative class
#                'hypo' marker to identify alpha threshold which make more sample frequency whose alpha < threshold in positive class, than those in negative class
def identify_alpha_threshold_by_alpha2sampleindexset_of_pos_and_neg_class(alpha2sampleindexset_of_neg_class,
                                                                alpha2sampleindexset_of_pos_class,
                                                                max_freq_cumsum_of_neg,
                                                                min_freq_cumsum_of_pos,
                                                                marker_type='hyper'):
    if 'hyper' in marker_type:
        if len(alpha2sampleindexset_of_pos_class)>0 and len(alpha2sampleindexset_of_neg_class)==0:
            alpha_threshold = 0 # any reads with any alpha values should be used. So let alpha_threshold==0 for hyper markers
            return (alpha_threshold, [], [], [])
        if len(alpha2sampleindexset_of_pos_class)>0 and len(alpha2sampleindexset_of_neg_class)>0:
            alpha_union_list = sorted(list(set(list(alpha2sampleindexset_of_pos_class.keys()) + list(alpha2sampleindexset_of_neg_class.keys()))), reverse=True) # decreasing order
    elif 'hypo' in marker_type:
        if len(alpha2sampleindexset_of_pos_class)>0 and len(alpha2sampleindexset_of_neg_class)==0:
            alpha_threshold = 1 # any reads with any alpha values should be used. So let alpha_threshold==1 for hypo markers
            return (alpha_threshold, [], [], [])
        if len(alpha2sampleindexset_of_pos_class)>0 and len(alpha2sampleindexset_of_neg_class)>0:
            alpha_union_list = sorted(list(set(list(alpha2sampleindexset_of_pos_class.keys()) + list(alpha2sampleindexset_of_neg_class.keys()))))  # increasing order
    alpha_index, arr_neg_cumsum, arr_pos_cumsum = identify_turning_point_of_two_alpha2sampleindexset(alpha_union_list,
                                                                     alpha2sampleindexset_of_neg_class,
                                                                     alpha2sampleindexset_of_pos_class,
                                                                     max_freq_cumsum_of_neg,
                                                                     min_freq_cumsum_of_pos)
    if alpha_index==-2:
        alpha_threshold = None
    else:
        if alpha_index==-1:
            alpha_threshold = None
            # if marker_type=='hyper':
            #     alpha_threshold = 0
            # elif marker_type == 'hypo':
            #     alpha_threshold = 1
        else:
            alpha_threshold = float(alpha_union_list[alpha_index])
            # if len(alpha2freq_of_neg_class) == 1:
            #     # Remove the following case which identifies hyper marker:
            #     # positive class: {'0.857':2, '0.286':1, '0.167':2, '0.143':4, '0':8}
            #     # negative class: {'0.167':2}
            #     # alpha_union_list: ['0.857', '0.286', '0.167', '0.143', '0']
            #     # identify_turning_point_of_two_cumsum_array returns alpha_index=4
            #     # But actually if alpha_threshold=='0', then negative class is incorrect for our purpose.
            #     if (('hyper' in marker_type) and (alpha_threshold<float(alpha2freq_of_neg_class.values[0]))) or (('hypo' in marker_type) and (alpha_threshold>float(alpha2freq_of_neg_class.values[0]))):
            #         alpha_threshold = None
    return( (alpha_threshold, alpha_union_list, arr_neg_cumsum, arr_pos_cumsum) )





# Input:
#   in_file1_background and in_file2_cancer: file format is from function 'write_combined_meth_string_histgram'
# Procedure:
#   We first load each of these two files into a dictionary { 'marker':histgram_dictionary }, for example: {27:hist_dict, 63:hist_dict}, where hist_dict is a dictionary {'0.7':2, '0.9':4}. It is the output of 'load_one_alpha_value_distribution_file' or 'combine_multi_alpha_histgram_files'
#
# Output:
#   ret_marker_2_alpha2freq: a dictionary {'alpha_threshold':alpha_cutoff, 'max_cpg_num':max_cpg_num_of_the_marker, 'alpha2freq':histgram_dictionary}
#
def compare_background_vs_cancer_alpha_value_distribution_files(method, in_file1_background, in_file2_cancer):
    a1_background = {}
    marker2max_cpg_num_1 = {}
    marker2sample_num_1 = {}
    if 'samplesetfreq' in method:
        if 'readfrac' in method:
            print('This part is unused anymore, and is replaced by the corresponding part of function "compare_background_vs_cancer_alpha_value_distribution_files_with_memory_saving_way"')
            # a1_background: a dictionary {'marker_index':{'alpha_value':sample_index_set}}
            load_one_alpha_value_distribution_file_that_has_sample_index_sets_and_read_fractions(in_file1_background, a1_background, marker2max_cpg_num_1, marker2sample_num_1)
        else:
            # a1_background: a dictionary {'marker_index':{'alpha_value':sample_index_set}}
            load_one_alpha_value_distribution_file_that_has_sample_index_sets(in_file1_background, a1_background, marker2max_cpg_num_1)
    else:
        # a1_background: a dictionary {'marker_index':{'alpha_value':frequency_int}}
        load_one_alpha_value_distribution_file(in_file1_background, a1_background, marker2max_cpg_num_1)
    a2_cancer = {}
    marker2max_cpg_num_2 = {}
    if 'samplesetfreq' in method:
        # a2_cancer: a dictionary {'marker_index':{'alpha_value':sample_index_set}}
        load_one_alpha_value_distribution_file_that_has_sample_index_sets(in_file2_cancer, a2_cancer, marker2max_cpg_num_2)
    else:
        # a2_cancer: a dictionary {'marker_index':{'alpha_value':frequency_int}}
        load_one_alpha_value_distribution_file(in_file2_cancer, a2_cancer, marker2max_cpg_num_2)
    marker_index_list1 = a1_background.keys()
    marker_index_list2 = a2_cancer.keys()
    marker_index_common_list = sorted(list(set(marker_index_list1).intersection(marker_index_list2)))
    ret_marker_2_alpha2freq = {}
    try:
        if 'hypo.min.alpha.diff' in method: # 'hypo.min.alpha.diff_0.3' if (min_alpha(m1_background[marker_id]) - min_alpha(m2_cancer[marker_id]))>=0.3, then we accept this marker_id and report those meth_strings in m2_cancer[marker_id] whose alpha values < min_alpha(m1_background[marker_id]).
            min_alpha_diff = float(method.split('_')[1])
            for m in marker_index_common_list:
                if (len(a1_background[m])==0) or (len(a2_cancer[m])==0): continue
                a1_min = min(list(map(float, a1_background[m])))
                a2_min = min(list(map(float, a2_cancer[m])))
                if (a1_min - a2_min) >= min_alpha_diff:
                    ret_marker_2_alpha2freq[m] = {'alpha_threshold':a1_min, 'max_cpg_num':marker2max_cpg_num_2[m], 'alpha2freq':filter_alpha2freq_by_alpha(a2_cancer[m], a1_min, '<')}
        elif 'hyper.max.alpha.diff' in method: # 'hyper.max.alpha.diff_0.3' if max_alpha(m2_cancer[marker_id]) - max_alpha(m1_background[marker_id])>=0.3, then we accept this marker_id and report those meth_strings in m2_cancer[marker_id] whose alpha values > max_alpha(m1_background[marker_id]).
            min_alpha_diff = float(method.split('_')[1])
            for m in marker_index_common_list:
                if (len(a1_background[m])==0) or (len(a2_cancer[m])==0): continue
                a1_max = max(list(map(float, a1_background[m])))
                a2_max = max(list(map(float, a2_cancer[m])))
                if (a2_max - a1_max) >= min_alpha_diff:
                    ret_marker_2_alpha2freq[m] = {'alpha_threshold':a1_max, 'max_cpg_num':marker2max_cpg_num_2[m], 'alpha2freq':filter_alpha2freq_by_alpha(a2_cancer[m], a1_max, '>')}
        elif 'samplesetfreq' in method:
            # 'hyper.alpha.samplesetfreq.thresholds.n2.p10': hyper-methylation markers with alpha's frequency on negative class <2 and alpha's freuqency on positive class > 10. Similar to 'hypo.alpha.samplesetfreq.thresholds.n2.p10'.
            # Negative class: background class
            # Positive class: cancer class
            if 'alpha.samplesetfreq.thresholds' in method:
                marker_type, _, _, _, max_freq_cumsum_of_neg_str, min_freq_cumsum_of_pos_str = method.split('.')
                max_freq_cumsum_of_neg = int(max_freq_cumsum_of_neg_str[1:])
                min_freq_cumsum_of_pos = int(min_freq_cumsum_of_pos_str[1:])
                for m in marker_index_common_list:
                    if (len(a1_background[m]) == 0) or (len(a2_cancer[m]) == 0): continue
                    alpha_threshold, alpha_union_list, a1_freq_cumsum_background, a2_freq_cumsum_cancer = identify_alpha_threshold_by_alpha2sampleindexset_of_pos_and_neg_class(a1_background[m],
                                                                                                  a2_cancer[m],
                                                                                                  max_freq_cumsum_of_neg,
                                                                                                  min_freq_cumsum_of_pos,
                                                                                                  marker_type)
                    if alpha_threshold is not None:
                        if 'hyper' in method:
                            ret_marker_2_alpha2freq[m] = {'alpha_threshold': alpha_threshold, 'max_cpg_num': marker2max_cpg_num_2[m],
                                                          'alpha2freq': filter_by_freq_cumsum_and_create_alpha2freqcumsum(alpha_threshold, alpha_union_list, a2_freq_cumsum_cancer, '>')}
                        elif 'hypo' in method:
                            ret_marker_2_alpha2freq[m] = {'alpha_threshold': alpha_threshold, 'max_cpg_num': marker2max_cpg_num_2[m],
                                                          'alpha2freq': filter_by_freq_cumsum_and_create_alpha2freqcumsum(alpha_threshold, alpha_union_list, a2_freq_cumsum_cancer, '<')}
        else:
            if 'alpha.freq.thresholds' in method:
                # 'hyper.alpha.freq.thresholds.n2.p10': hyper-methylation markers with alpha's frequency on negative class <2 and alpha's freuqency on positive class > 10. Similar to 'hypo.alpha.freq.thresholds.n2.p10', 'hyper.alpha.freq.thresholds.n2.p4.enforce_max_output'
                # Negative class: background class
                # Positive class: cancer class
                items = method.split('.')
                marker_type = items[0]
                max_freq_cumsum_of_neg_str = items[4]
                min_freq_cumsum_of_pos_str = items[5]
                # marker_type, _, _, _, max_freq_cumsum_of_neg_str, min_freq_cumsum_of_pos_str = method.split('.')
                max_freq_cumsum_of_neg = int(max_freq_cumsum_of_neg_str[1:])
                min_freq_cumsum_of_pos = int(min_freq_cumsum_of_pos_str[1:])
                for m in marker_index_common_list:
                    if (len(a1_background[m]) == 0) or (len(a2_cancer[m]) == 0): continue
                    alpha_threshold = identify_alpha_threshold_by_alpha2freq_of_pos_and_neg_class(a1_background[m],
                                                                                                  a2_cancer[m],
                                                                                                  max_freq_cumsum_of_neg,
                                                                                                  min_freq_cumsum_of_pos,
                                                                                                  marker_type)
                    if alpha_threshold is not None:
                        if 'hyper' in method:
                            ret_marker_2_alpha2freq[m] = {'alpha_threshold': alpha_threshold, 'max_cpg_num': marker2max_cpg_num_2[m],
                                                          'alpha2freq': filter_alpha2freq_by_alpha(a2_cancer[m], alpha_threshold,
                                                                                                   '>')}
                        elif 'hypo' in method:
                            ret_marker_2_alpha2freq[m] = {'alpha_threshold': alpha_threshold, 'max_cpg_num': marker2max_cpg_num_2[m],
                                                          'alpha2freq': filter_alpha2freq_by_alpha(a2_cancer[m], alpha_threshold,
                                                                                                   '<')}
    except KeyError:
        # marker_index does not exist
        sys.stderr.write('Error: %d does not exist in one of two meth_strings_histgram_files\n  in_file1_background: %s\n  in_file2_cancer: %s\nExit.'%(m, in_file1_background, in_file2_cancer))
        sys.exit(-1)
    # ret_marker2max_cpg_num = {m: marker2max_cpg_num_2[m] for m in ret_marker_2_alpha2freq}
    return( ret_marker_2_alpha2freq )

# alpha_union_list: the list of alpha values, which are not only union of alpha values of pos_class and neg_class, but also with the order of alpha values, with respect to 'hyper' (in decreasing order) or 'hypo' (in increasing order).
# alpha2sampleindexsetandreadfraction_of_neg_class, alpha2sampleindexsetandreadfraction_of_pos_class: two dictionaries {'alpha_value':{sample_index:read_fraction}}
# Algorithm:
#    Step 1: accumulate sample_index_set of each alpha_value, by the alpha_value order of alpha_union_list
#    Step 2: calcualte size of accumulated sample_index_set of each alpha_value
#    Step 3: compute the following to determine the turning point. This step is the same as function 'identify_alpha_threshold_by_alpha2freq_of_pos_and_neg_class'
#            arr_neg_cumsum[index] <= arr_neg_cumsum_threshold
#            arr_pos_cumsum[index] >= arr_pos_cumsum_threshold
#            If there exist multiple index satisfying the above criteria, choose the largest index
#    Step 4: return the index of turning point (i.e., the index of alpha threshold in alpha_union_list)
#
def identify_turning_point_of_two_alpha2sampleindexsetandreadfraction(min_read_fraction_neg, min_read_fraction_pos, alpha_union_list, alpha2sampleindexsetandreadfraction_of_neg_class, alpha2sampleindexsetandreadfraction_of_pos_class, arr_neg_cumsum_threshold, arr_pos_cumsum_threshold):
    n_alpha = len(alpha_union_list)
    # implement cumsum using dict merge operator "mergeDict_by_adding_values_of_common_keys"
    alpha2cumdict_neg = {a:dict() for a in alpha_union_list}
    alpha2cumdict_pos = dict(alpha2cumdict_neg)
    for i in range(n_alpha):
        a = alpha_union_list[i]
        if i == 0:
            alpha2cumdict_neg[a] = alpha2sampleindexsetandreadfraction_of_neg_class[a] if a in alpha2sampleindexsetandreadfraction_of_neg_class else dict()
            alpha2cumdict_pos[a] = alpha2sampleindexsetandreadfraction_of_pos_class[a] if a in alpha2sampleindexsetandreadfraction_of_pos_class else dict()
        else:
            a_prev = alpha_union_list[i - 1]
            alpha2cumdict_neg[a] = mergeDict_by_adding_values_of_common_keys(alpha2sampleindexsetandreadfraction_of_neg_class[a], alpha2cumdict_neg[a_prev]) if a in alpha2sampleindexsetandreadfraction_of_neg_class else alpha2cumdict_neg[a_prev]
            alpha2cumdict_pos[a] = mergeDict_by_adding_values_of_common_keys(alpha2sampleindexsetandreadfraction_of_pos_class[a], alpha2cumdict_pos[a_prev]) if a in alpha2sampleindexsetandreadfraction_of_pos_class else alpha2cumdict_pos[a_prev]
    arr_neg_cumsum = np.array([sum([True if alpha2cumdict_neg[a][sample_index]>=min_read_fraction_neg else False for sample_index in alpha2cumdict_neg[a]]) for a in alpha_union_list])
    arr_pos_cumsum = np.array([sum([True if alpha2cumdict_pos[a][sample_index]>=min_read_fraction_pos else False for sample_index in alpha2cumdict_pos[a]]) for a in alpha_union_list])
    index_list = [i for i in range(n_alpha) if ((arr_pos_cumsum[i]>=arr_pos_cumsum_threshold) and (arr_neg_cumsum[i]<=arr_neg_cumsum_threshold))]
    if len(index_list)==0:
        # No turning point
        return( (-2, arr_neg_cumsum, arr_pos_cumsum) )
    else:
        # if (index_list[-1]+1) < (n_alpha-1):
        if (index_list[-1]) < (n_alpha - 1):
            return( (index_list[-1]+1, arr_neg_cumsum, arr_pos_cumsum) )
        else:
            if len(index_list)>=2:
                if (index_list[-2]) < (n_alpha - 1):
                    return( (index_list[-2]+1, arr_neg_cumsum, arr_pos_cumsum) )
            # -1 means the index should be the one greater than n
            return( (-1, arr_neg_cumsum, arr_pos_cumsum) )


def identify_alpha_threshold_by_alpha2sampleindexsetandreadfraction_of_pos_and_neg_class(unique_alpha_values_of_neg_class, sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_neg_class, unique_alpha_values_of_pos_class, sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_pos_class, max_freq_cumsum_of_neg,
                                                                min_freq_cumsum_of_pos, min_read_fraction_neg, min_read_fraction_pos,
                                                                marker_type='hyper'):
    # build the dict for alpha2sampleindexsetandreadfraction for negative class
    alpha2sampleindexsetandreadfraction_of_neg_class = {}
    for i in range(len(unique_alpha_values_of_neg_class)):
        alpha = unique_alpha_values_of_neg_class[i]
        alpha2sampleindexsetandreadfraction_of_neg_class[alpha] = {}
        items = sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_neg_class[i].split('_')
        for item in items:
            sample_index, read_fraction = item.split(':')
            read_fraction = float(read_fraction)
            alpha2sampleindexsetandreadfraction_of_neg_class[alpha][sample_index] = read_fraction
    # build the dict for alpha2sampleindexsetandreadfraction for positive class
    alpha2sampleindexsetandreadfraction_of_pos_class = {}
    for i in range(len(unique_alpha_values_of_pos_class)):
        alpha = unique_alpha_values_of_pos_class[i]
        alpha2sampleindexsetandreadfraction_of_pos_class[alpha] = {}
        items = sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_pos_class[i].split('_')
        for item in items:
            sample_index, read_fraction = item.split(':')
            read_fraction = float(read_fraction)
            alpha2sampleindexsetandreadfraction_of_pos_class[alpha][sample_index] = read_fraction

    # Process
    if 'hyper' in marker_type:
        if len(alpha2sampleindexsetandreadfraction_of_pos_class)>0 and len(alpha2sampleindexsetandreadfraction_of_neg_class)==0:
            alpha_threshold = 0 # any reads with any alpha values should be used. So let alpha_threshold==0 for hyper markers
            return (alpha_threshold, [], [], [])
        if len(alpha2sampleindexsetandreadfraction_of_pos_class)>0 and len(alpha2sampleindexsetandreadfraction_of_neg_class)>0:
            alpha_union_list = sorted(list(set(list(alpha2sampleindexsetandreadfraction_of_pos_class.keys()) + list(alpha2sampleindexsetandreadfraction_of_neg_class.keys()))), reverse=True) # decreasing order
    elif 'hypo' in marker_type:
        if len(alpha2sampleindexsetandreadfraction_of_pos_class)>0 and len(alpha2sampleindexsetandreadfraction_of_neg_class)==0:
            alpha_threshold = 1 # any reads with any alpha values should be used. So let alpha_threshold==1 for hypo markers
            return (alpha_threshold, [], [], [])
        if len(alpha2sampleindexsetandreadfraction_of_pos_class)>0 and len(alpha2sampleindexsetandreadfraction_of_neg_class)>0:
            alpha_union_list = sorted(list(set(list(alpha2sampleindexsetandreadfraction_of_pos_class.keys()) + list(alpha2sampleindexsetandreadfraction_of_neg_class.keys()))))  # increasing order
    # identify_turning_point_of_two_alpha2sampleindexsetandreadfraction
    alpha_index, arr_neg_cumsum, arr_pos_cumsum = identify_turning_point_of_two_alpha2sampleindexsetandreadfraction(min_read_fraction_neg,
                                                                                                                    min_read_fraction_pos,
                                                                                                                    alpha_union_list,
                                                                                                     alpha2sampleindexsetandreadfraction_of_neg_class,
                                                                                                     alpha2sampleindexsetandreadfraction_of_pos_class,
                                                                                                     max_freq_cumsum_of_neg,
                                                                                                     min_freq_cumsum_of_pos)
    if alpha_index == -2:
        alpha_threshold = None
    else:
        if alpha_index == -1:
            alpha_threshold = None
            # if marker_type=='hyper':
            #     alpha_threshold = 0
            # elif marker_type == 'hypo':
            #     alpha_threshold = 1
        else:
            alpha_threshold = float(alpha_union_list[alpha_index])
            # if len(alpha2freq_of_neg_class) == 1:
            #     # Remove the following case which identifies hyper marker:
            #     # positive class: {'0.857':2, '0.286':1, '0.167':2, '0.143':4, '0':8}
            #     # negative class: {'0.167':2}
            #     # alpha_union_list: ['0.857', '0.286', '0.167', '0.143', '0']
            #     # identify_turning_point_of_two_cumsum_array returns alpha_index=4
            #     # But actually if alpha_threshold=='0', then negative class is incorrect for our purpose.
            #     if (('hyper' in marker_type) and (alpha_threshold<float(alpha2freq_of_neg_class.values[0]))) or (('hypo' in marker_type) and (alpha_threshold>float(alpha2freq_of_neg_class.values[0]))):
            #         alpha_threshold = None
    return ((alpha_threshold, alpha_union_list, arr_neg_cumsum, arr_pos_cumsum))


def compare_background_vs_cancer_alpha_value_distribution_files_with_memory_saving_way(method, in_file1_background, in_file2_cancer):
    ret_marker_2_alpha2freq = {}
    if 'samplesetfreq' in method:
        if 'readfrac' in method: # 'hyper.alpha.samplesetfreq.thresholds.n2.p5.minreadfrac+0.1-0.01'
            two_read_fractions_str = extract_two_numbers_after_a_substring(method, 'minreadfrac') # extract '+0.1' and '-0.01' from 'hyper.alpha.samplesetfreq.thresholds.n2.p5.minreadfrac+0.1-0.01'
            if two_read_fractions_str is None:
                sys.stderr.write('Error: method (%s) after string minreadfrac does not have two min read fractions with format like +0.2-0.3 or +1-1.\nExit.\n'%method)
                sys.exit(-1)
            min_read_fraction_pos = abs(float(two_read_fractions_str[0]))
            min_read_fraction_neg = abs(float(two_read_fractions_str[1]))
            marker_type, _, _, _, max_freq_cumsum_of_neg_str, min_freq_cumsum_of_pos_str = remove_substring_followed_by_two_floats(
                method, '.minreadfrac').split('.')
            if 'nn' in max_freq_cumsum_of_neg_str:
                max_freq_fraction_of_neg = float(max_freq_cumsum_of_neg_str[1:])
            else:
                max_freq_cumsum_of_neg = int(max_freq_cumsum_of_neg_str[1:])
            min_freq_cumsum_of_pos = int(min_freq_cumsum_of_pos_str[1:])

            if in_file1_background.endswith('gz'):
                fid1_background = gzip.open(in_file1_background, 'rt')
            else:
                fid1_background = open(in_file1_background, 'rt')
            if in_file2_cancer.endswith('gz'):
                fid2_cancer = gzip.open(in_file2_cancer, 'rt')
            else:
                fid2_cancer = open(in_file2_cancer, 'rt')
            ### begin to process two input files and write output file
            fid1_background.readline().rstrip() # skip header line
            background_first_marker_line = fid1_background.readline()
            background_items = background_first_marker_line.rstrip().split('\t')
            background_marker_index = int(background_items[0])

            end_of_background_file = False
            fid2_cancer.readline()  # skip header line
            for cancer_line in fid2_cancer:
                cancer_items = cancer_line.rstrip().split()
                cancer_marker_index = int(cancer_items[0])
                while cancer_marker_index > background_marker_index:
                    background_line = fid1_background.readline()
                    if not background_line:
                        end_of_background_file = True
                        break
                    background_items = background_line.rstrip().split('\t')
                    background_marker_index = int(background_items[0])
                if end_of_background_file:
                    break
                if cancer_marker_index < background_marker_index:
                    continue

                # now we begin to process for cancer_marker_index == background_marker_index
                # if cancer_marker_index == 371737:
                #     print('debug 371737')
                max_cpg_num = background_items[1]
                background_sample_num = int(background_items[2])
                background_unique_alpha_values = background_items[3].split(',')
                background_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list = background_items[4].split(',')

                cancer_unique_alpha_values = cancer_items[3].split(',')
                cancer_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list = cancer_items[4].split(',')

                if 'nn' in max_freq_cumsum_of_neg_str:
                    max_freq_cumsum_of_neg = max_freq_fraction_of_neg * background_sample_num

                alpha_threshold, alpha_union_list, a1_freq_cumsum_background, a2_freq_cumsum_cancer = identify_alpha_threshold_by_alpha2sampleindexsetandreadfraction_of_pos_and_neg_class(background_unique_alpha_values, background_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list, cancer_unique_alpha_values, cancer_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list, max_freq_cumsum_of_neg, min_freq_cumsum_of_pos, min_read_fraction_neg, min_read_fraction_pos, marker_type)
                if alpha_threshold is not None:
                    if 'hyper' in method:
                        ret_marker_2_alpha2freq[cancer_marker_index] = {'alpha_threshold': alpha_threshold,
                                                      'max_cpg_num':max_cpg_num,
                                                      'alpha2freq': filter_by_freq_cumsum_and_create_alpha2freqcumsum(
                                                          alpha_threshold, alpha_union_list, a2_freq_cumsum_cancer,
                                                          '>')}
                    elif 'hypo' in method:
                        ret_marker_2_alpha2freq[cancer_marker_index] = {'alpha_threshold': alpha_threshold,
                                                      'max_cpg_num':max_cpg_num,
                                                      'alpha2freq': filter_by_freq_cumsum_and_create_alpha2freqcumsum(
                                                          alpha_threshold, alpha_union_list, a2_freq_cumsum_cancer,
                                                          '<')}

            fid1_background.close()
            fid2_cancer.close()
    return (ret_marker_2_alpha2freq)


#
# Input:
# alpha_hists: a dictionary { marker_index:{'alpha_threshold':threshold, 'max_cpg_num':cpg_num, 'alpha2freq':alpha_histgram_dictionary} }. for example, alpha_histgram_dictionary is {27:hist_dict, 63:hist_dict}, where hist_dict is a dictionary {'0.7':2, '1.0':4}. The input 'alpha_hists' can be an empty dictionary {}.
# marker2max_cpg_num: a dictionary {marker_index:max_cpg_num}. NOTE: It is from the output of function 'compare_background_vs_cancer_alpha_value_distribution_files'
#
# Output:
# alpha_value_distribution_with_threshold file (from the output of the function 'summarize_mary_file_binary_meth_values_for_distribution_file'):
# marker_index    max_num_cpg num_read  alpha_threshold  unique_alpha_values read_freq_of_unique_alpha_values
# 2   7   122  0.4  0.429,0.571,0.714,0.857,1   1,2,27,42,50
# 27  9   39  0.5  0.5,0.625,0.75,0.778,0.875,0.889,1  1,2,9,1,12,1,13
# 61  12  44  0.7 0.75,0.833,0.917,1  2,11,12,19
# 63  5   100  0.5  0.6,0.8,1   4,23,73
# 65  5   83 0.5 0.6,0.8,1 9,26,42
# ...
#
def write_alpha_value_distribution_file_with_alpha_threshold(fout, alpha_hists, frequency_type='alpha2freqeuncy_is_individual'):
    marker_index_list = sorted(list(set(alpha_hists.keys())))
    fout.write(
        'marker_index\tmax_num_cpg\tnum_read\talpha_threshold\tunique_alpha_values\tread_freq_of_unique_alpha_values\n')
    for marker_index in marker_index_list:
        if 'alpha2freqeuncy_is_individual' in frequency_type:
            num_reads = sum(alpha_hists[marker_index]['alpha2freq'].values())
        elif 'enforce_max_output' in frequency_type:
            num_reads = max(alpha_hists[marker_index]['alpha2freq'].values())
        elif 'alpha2freqeuncy_is_cumsum' in frequency_type:
            num_reads = max(alpha_hists[marker_index]['alpha2freq'].values())
        str_for_unique_alpha_values, read_freq_of_unique_alpha_values_str = convert_str2int_dict_to_str(alpha_hists[marker_index]['alpha2freq'])
        fout.write('%d\t%s\t%d\t%g\t%s\t%s\n' % (marker_index,
                                                 alpha_hists[marker_index]['max_cpg_num'],
                                                 num_reads,
                                                 alpha_hists[marker_index]['alpha_threshold'],
                                                 str_for_unique_alpha_values,
                                                 read_freq_of_unique_alpha_values_str
                                                 ))

#
# Input:
# alpha_hists: a dictionary { marker_index:{'alpha_threshold':threshold, 'max_cpg_num':cpg_num, 'alpha2freq':alpha_histgram_dictionary} }. for example, alpha_histgram_dictionary is {27:hist_dict, 63:hist_dict}, where hist_dict is a dictionary {'0.7':2, '1.0':4}. The input 'alpha_hists' can be an empty dictionary {}.
# marker2max_cpg_num: a dictionary {marker_index:max_cpg_num}. NOTE: It is from the output of function 'compare_background_vs_cancer_alpha_value_distribution_files'
#
# Output:
# alpha_value_distribution_with_threshold file (from the output of the function 'summarize_mary_file_binary_meth_values_for_distribution_file'):
# marker_index    max_num_cpg num_read  alpha_threshold_pos  alpha_threshold_neg  unique_alpha_values read_freq_of_unique_alpha_values
# 2   7   122  0.4  0.1  0.429,0.571,0.714,0.857,1   1,2,27,42,50
# 27  9   39  0.5  0.2  0.5,0.625,0.75,0.778,0.875,0.889,1  1,2,9,1,12,1,13
# 61  12  44  0.7  0.3 0.75,0.833,0.917,1  2,11,12,19
# 63  5   100  0.5  0.1  0.6,0.8,1   4,23,73
# 65  5   83 0.5  0.05 0.6,0.8,1 9,26,42
# ...
#
def write_alpha_value_distribution_file_with_two_alpha_thresholds(fout, alpha_hists, frequency_type='alpha2freqeuncy_is_individual'):
    marker_index_list = sorted(list(set(alpha_hists.keys())))
    fout.write(
        'marker_index\tmax_num_cpg\tnum_read\talpha_threshold_pos\talpha_threshold_neg\tunique_alpha_values\tread_freq_of_unique_alpha_values\n')
    for marker_index in marker_index_list:
        if 'alpha2freqeuncy_is_cumsum' in frequency_type:
            num_reads = max(alpha_hists[marker_index]['alpha2freq'].values())
        str_for_unique_alpha_values, read_freq_of_unique_alpha_values_str = convert_str2int_dict_to_str(alpha_hists[marker_index]['alpha2freq'])
        fout.write('%d\t%s\t%d\t%s\t%s\t%s\t%s\n' % (marker_index,
                                                     alpha_hists[marker_index]['max_cpg_num'],
                                                     num_reads,
                                                     alpha_hists[marker_index]['alpha_threshold_of_pos'],
                                                     alpha_hists[marker_index]['alpha_threshold_of_neg'],
                                                     str_for_unique_alpha_values,
                                                     read_freq_of_unique_alpha_values_str
                                                     ))

# Input:
# alpha_value_distribution file (from output of the function 'write_alpha_value_distribution_file_with_alpha_threshold'):
# alpha_value_distribution_with_threshold file (from the output of the function 'summarize_mary_file_binary_meth_values_for_distribution_file'):
# marker_index    max_num_cpg num_read  alpha_threshold  unique_alpha_values read_freq_of_unique_alpha_values
# 2   7   122  0.4  0.429,0.571,0.714,0.857,1   1,2,27,42,50
# 27  9   39  0.5  0.5,0.625,0.75,0.778,0.875,0.889,1  1,2,9,1,12,1,13
# 61  12  44  0.7 0.75,0.833,0.917,1  2,11,12,19
# 63  5   100  0.5  0.6,0.8,1   4,23,73
# 65  5   83 0.5 0.6,0.8,1 9,26,42
# ...
#
# markers_list: a dictionary as below. The input 'alpha_hists' can be an empty dictionary {}.
# markers_list[marker_index] = {'alpha_threshold_for_final_use': 0.3,
#                               'pair_comparison_list':['stringent.hyper.max.alpha.diff_0.2.COAD+vsSTAD-_r2_a0.5', 'stringent.hyper.max.alpha.diff_0.2.LIHC+vsSTAD-_r4_a0.5'],
#                               'alpha_threshold_list': [0.5, 0.5],
#                               'max_num_cpg': 8,
#                               'read_num_list': [2, 4]}
# markers_loaded_for_each_paired_comparison: a dictionary {paired_comparison_ID:marker_index_set}, where paired_comparison_ID can be like 'stringent.hyper.max.alpha.diff_0.2.LUNG+vsSTAD-'
# marker_type: for example, 'stringent.hyper', 'stringent.hypo', or 'hyper', 'hypo'
#
# Output:
#   markers_list: a dictionary {marker_index:info_of_alpha_thresholds_and_comparisons}, where 'info_of_alpha_thresholds_and_comparisons' is a dictionary: {'pair_comparison_list':['hypo.COAD+LIHC-_r10_a0.3','hypo.LUAD+LIHC-_r10_a0.3'], 'alpha_threshold_list':[0.3, 0.4], 'alpha_threshold_for_final_use':0.3}
#   num_of_markers_loaded_for_marker_type: a dictionary {marker_type:}
# NOTE: if we use the input argument "topK", then the lines of the file MUST be sorted by the 3rd column 'num_read' in the descending order; otherwise the file does NOT need to be sorted by the 3rd column 'num_read'. The sorted file is illustrated as below,
#
# Input file format:
# marker_index	max_num_cpg	num_read	alpha_threshold	unique_alpha_values	read_freq_of_unique_alpha_values
# 573199	15	80	0	0.2,0.267,0.333,0.4,0.429,0.533,0.6,0.667,0.714,0.733,0.8,0.857,0.867,0.929,0.933,1	1,2,1,2,1,1,2,1,1,2,2,1,17,2,20,24
# 976591	16	70	0.312	0.333,0.357,0.385,0.4,0.467,0.533,0.6,0.625,0.667,0.688,0.714,0.733,0.786,0.8,0.812,0.867,0.875,0.933,1	4,1,1,6,5,3,6,1,5,1,1,5,1,6,1,9,2,6,6
# 368467	16	66	0.312	0.333,0.375,0.438,0.5,0.562,0.625,0.688,0.75,0.812,0.867,0.875,0.938,1	2,3,4,6,3,4,6,7,7,1,8,6,9
# 156684	11	66	0.4	0.455,0.545,0.6,0.636,0.7,0.727,0.8,0.818,0.9,0.909,1	14,10,1,6,2,9,2,10,2,6,4
# 723070	5	65	0	0.2,0.4,0.6,0.8,1	6,8,14,17,20
# 923627	16	64	0.375	0.438,0.5,0.562,0.6,0.625,0.688,0.706,0.75,0.812,0.867,0.875,0.938,1	5,9,7,1,6,8,1,7,7,2,6,2,3
# 703536	15	63	0.267	0.286,0.333,0.4,0.467,0.533,0.6,0.667,0.733,0.8,0.867,0.929,0.933,1	1,4,6,3,5,4,3,3,8,5,1,6,14
# 260379	7	57	0.333	0.429,0.571,0.6,0.667,0.714,0.833,0.857,1	15,11,1,1,10,2,9,8
# 81879	16	56	0.438	0.467,0.5,0.533,0.562,0.625,0.688,0.75,0.812,0.875,0.938,1	1,7,1,6,3,6,7,6,6,7,6
# 1052811	30	55	0.161	0.179,0.194,0.2,0.226,0.258,0.276,0.3,0.452,0.484,0.516,0.545,0.586,0.611,0.633,0.645,0.655,0.667,0.677,0.688,0.7,0.71,0.733,0.739,0.742,0.774,0.793,0.806,0.833,0.839,0.846,0.85,0.862,0.871,0.9,0.903,0.933,0.935,0.968,1	1,1,1,2,2,1,1,1,2,1,1,1,1,2,2,1,1,1,1,1,2,2,1,4,1,1,3,1,1,1,1,1,1,1,2,1,4,1,1
# 544335	9	54	0.222	0.333,0.444,0.556,0.667,0.778,0.889,1	8,5,7,9,9,9,7
# 336578	2	1	0	0.5	1
# 326045	2	1	0	0.5	1
# 16614	3	1	0	0.333	1
# 960994	2	0	0
#
def load_one_alpha_value_distribution_file_with_alpha_threshold_for_marker_selection(file,
                                                                                     markers_list,
                                                                                     markers_loaded_for_each_paired_comparison,
                                                                                     marker_type_and_pair_comparison_id,
                                                                                     min_cpg_num=3,
                                                                                     min_read_num=1,
                                                                                     topK=-1):
    if marker_type_and_pair_comparison_id not in markers_loaded_for_each_paired_comparison:
        markers_loaded_for_each_paired_comparison[marker_type_and_pair_comparison_id] = set([])
    with gzip.open(file, 'rt') as f:
        next(f) # skip the first header line
        marker_rank = 0
        for line in f:
            items = line.rstrip().split()
            max_num_cpg = int(items[1])
            if max_num_cpg<min_cpg_num:
                continue
            num_read = int(items[2])
            if num_read<min_read_num:
                continue
            if (marker_rank==topK) or (len(markers_loaded_for_each_paired_comparison[marker_type_and_pair_comparison_id])==topK):
                # when topK=-1, we never break and will load all markers
                break
            marker_rank += 1 # we will accept this marker
            marker_index = int(items[0])
            alpha_threshold = items[3]
            if topK==-1:
                id = '%s_r%s_a%s' % (marker_type_and_pair_comparison_id, num_read, alpha_threshold)
            else:
                id = '%s_rank%d_r%s_a%s' % (marker_type_and_pair_comparison_id, marker_rank, num_read, alpha_threshold)
            if marker_index not in markers_list:
                alpha_threshold = float(alpha_threshold)
                markers_list[marker_index] = {'alpha_threshold_for_final_use':None,
                                              'pair_comparison_list':[id],
                                              'alpha_threshold_list': [alpha_threshold],
                                              'max_num_cpg':max_num_cpg,
                                              'read_num_list':[num_read]}
                if topK!=-1:
                    markers_list[marker_index]['marker_rank_list'] = [marker_rank]
                markers_loaded_for_each_paired_comparison[marker_type_and_pair_comparison_id].add(marker_index)
            else:
                # update markers
                alpha_threshold = float(alpha_threshold)
                markers_list[marker_index]['pair_comparison_list'].append(id)
                markers_list[marker_index]['alpha_threshold_list'].append( alpha_threshold )
                markers_list[marker_index]['read_num_list'].append( num_read )
                if topK != -1:
                    markers_list[marker_index]['marker_rank_list'].append( marker_rank )
                markers_loaded_for_each_paired_comparison[marker_type_and_pair_comparison_id].add(marker_index)

# Input file format:
# marker_index	max_num_cpg	num_read	alpha_threshold	unique_alpha_values	read_freq_of_unique_alpha_values
# 573199	15	80	0	0.2,0.267,0.333,0.4,0.429,0.533,0.6,0.667,0.714,0.733,0.8,0.857,0.867,0.929,0.933,1	1,2,1,2,1,1,2,1,1,2,2,1,17,2,20,24
# 976591	16	70	0.312	0.333,0.357,0.385,0.4,0.467,0.533,0.6,0.625,0.667,0.688,0.714,0.733,0.786,0.8,0.812,0.867,0.875,0.933,1	4,1,1,6,5,3,6,1,5,1,1,5,1,6,1,9,2,6,6
# 368467	16	66	0.312	0.333,0.375,0.438,0.5,0.562,0.625,0.688,0.75,0.812,0.867,0.875,0.938,1	2,3,4,6,3,4,6,7,7,1,8,6,9
# 156684	11	66	0.4	0.455,0.545,0.6,0.636,0.7,0.727,0.8,0.818,0.9,0.909,1	14,10,1,6,2,9,2,10,2,6,4
# 723070	5	65	0	0.2,0.4,0.6,0.8,1	6,8,14,17,20
# 923627	16	64	0.375	0.438,0.5,0.562,0.6,0.625,0.688,0.706,0.75,0.812,0.867,0.875,0.938,1	5,9,7,1,6,8,1,7,7,2,6,2,3
# 703536	15	63	0.267	0.286,0.333,0.4,0.467,0.533,0.6,0.667,0.733,0.8,0.867,0.929,0.933,1	1,4,6,3,5,4,3,3,8,5,1,6,14
# 260379	7	57	0.333	0.429,0.571,0.6,0.667,0.714,0.833,0.857,1	15,11,1,1,10,2,9,8
# 81879	16	56	0.438	0.467,0.5,0.533,0.562,0.625,0.688,0.75,0.812,0.875,0.938,1	1,7,1,6,3,6,7,6,6,7,6
# 1052811	30	55	0.161	0.179,0.194,0.2,0.226,0.258,0.276,0.3,0.452,0.484,0.516,0.545,0.586,0.611,0.633,0.645,0.655,0.667,0.677,0.688,0.7,0.71,0.733,0.739,0.742,0.774,0.793,0.806,0.833,0.839,0.846,0.85,0.862,0.871,0.9,0.903,0.933,0.935,0.968,1	1,1,1,2,2,1,1,1,2,1,1,1,1,2,2,1,1,1,1,1,2,2,1,4,1,1,3,1,1,1,1,1,1,1,2,1,4,1,1
# 544335	9	54	0.222	0.333,0.444,0.556,0.667,0.778,0.889,1	8,5,7,9,9,9,7
# 336578	2	1	0	0.5	1
# 326045	2	1	0	0.5	1
# 16614	3	1	0	0.333	1
# 960994	2	0	0
#
# NOTE: Since we use the input argument "topK", then the lines of the file MUST be sorted by the 3rd column 'num_read' in the descending order; otherwise the file does NOT need to be sorted by the 3rd column 'num_read'. The sorted file is illustrated as below,
#
# Output:
#   markers: a dictionary {'marker_index':[a list of marker_index], 'alpha_threshold':[a list of alpha thresholds], 'num_read':[a list of num_reads]}
def load_one_alpha_value_distribution_file_of_sorted_num_reads_by_topK(file, topK=100, min_cpg_num=3, min_read_num=1):
    markers = {'marker_index':[], 'alpha_threshold':[], 'num_read':[]}
    with gzip.open(file, 'rt') as f:
        next(f)  # skip the first header line
        marker_rank = 0
        for line in f:
            items = line.rstrip().split()
            max_num_cpg = int(items[1])
            if max_num_cpg < min_cpg_num:
                continue
            num_read = int(items[2])
            if num_read < min_read_num:
                continue
            if marker_rank==topK:
                break
            marker_rank += 1 # we will accept this marker
            marker_index = int(items[0])
            alpha_threshold = items[3]
            markers['marker_index'].append(marker_index)
            markers['alpha_threshold'].append(alpha_threshold)
            markers['num_read'].append(num_read)
    return(markers)


# Input:
#   list_of_paired_comparisons: a list of tuples [('COAD+vsLIHC-',filename), ('COAD+vsLUAD-',filename), ... ]
#   marker_type: 'hyper' or 'hypo', 'hyper.stringent', 'hyper.loose'
#   min_cpg_num: minimum number of CpG sites required by markers
#   output_union_markers_file
#
# Output file format:
#
def union_markers_of_paired_comparison_alpha_value_distribution_file_with_alpha_threshold(list_of_paired_comparisons,
                                                                                          marker_type,
                                                                                          min_cpg_num,
                                                                                          output_union_markers_file):
    union_markers_list = {}
    markers_loaded_for_each_paired_comparison = {}
    print('Union markers', flush=True)
    for pair_comparison_type, filename in list_of_paired_comparisons:
        print('  %s'%pair_comparison_type, flush=True)
        load_one_alpha_value_distribution_file_with_alpha_threshold_for_marker_selection(filename,
                                                                                         union_markers_list,
                                                                                         markers_loaded_for_each_paired_comparison,
                                                                                         '%s.%s'%(marker_type, pair_comparison_type),
                                                                                         min_cpg_num)
    # Determine the best alpha value threshold for each unioned marker, when the marker has multiple different alpha value thresholds
    print('Determine the best alpha value threshold for each unioned marker', flush=True)
    for marker_index in union_markers_list.keys():
        if len(union_markers_list[marker_index]['alpha_threshold_list'])==1:
            # only one alpha value threshold
            union_markers_list[marker_index]['alpha_threshold_for_final_use'] = 0 # index of alpha value in the alpha_threshold_list that will be for final use.
        else:
            # multiple alpha value thresholds
            if 'hyper' in marker_type:
                if 'stringent' in marker_type:
                    union_markers_list[marker_index]['alpha_threshold_for_final_use'] = np.argmax(union_markers_list[marker_index]['alpha_threshold_list'])
                elif 'loose' in marker_type:
                    union_markers_list[marker_index]['alpha_threshold_for_final_use'] = np.argmin(union_markers_list[marker_index]['alpha_threshold_list'])
            elif 'hypo' in marker_type:
                if 'stringent' in marker_type:
                    union_markers_list[marker_index]['alpha_threshold_for_final_use'] = np.argmin(union_markers_list[marker_index]['alpha_threshold_list'])
                elif 'loose' in marker_type:
                    union_markers_list[marker_index]['alpha_threshold_for_final_use'] = np.argmax(union_markers_list[marker_index]['alpha_threshold_list'])
    # Write to file
    print('Write to file:\n  output: %s'%output_union_markers_file, flush=True)
    with gzip.open(output_union_markers_file, 'wt') as fout:
        marker_index_list = sorted(union_markers_list.keys())
        fout.write('marker_index\talpha_threshold\tmarker_type\tmax_num_cpg\tpair_comparison_list\n')
        for marker_index in marker_index_list:
            index_of_alpha_threshold_for_use = union_markers_list[marker_index]['alpha_threshold_for_final_use']
            fout.write('%d\t%g\t%s\t%d\t%s\n'%(marker_index,
                                               union_markers_list[marker_index]['alpha_threshold_list'][index_of_alpha_threshold_for_use],
                                               union_markers_list[marker_index]['pair_comparison_list'][index_of_alpha_threshold_for_use],
                                               union_markers_list[marker_index]['max_num_cpg'],
                                               ','.join(union_markers_list[marker_index]['pair_comparison_list'])
                                               ))

# Input config file format (tab-delimited): paired_comparison_ID topK paired_comparison_alpha_value_distribution_file.
#    This input config file is generated by the code "generate_config_file_for_union_topK_markers.py"
# paired_comparison_ID_or_type topK paired_comparison_marker_file
# COAD+vsLIHC-	300	v100005/train_data/paired_comparison_hyper.alpha.freq.thresholds.n2.p10/COAD+vsLIHC-.alpha_values_distr.txt.gz
# LIHC+vsCOAD-	300	v100005/train_data/paired_comparison_hyper.alpha.freq.thresholds.n2.p10/LIHC+vsCOAD-.alpha_values_distr.txt.gz
# COAD+vsLUNG-	300	v100005/train_data/paired_comparison_hyper.alpha.freq.thresholds.n2.p10/COAD+vsLUNG-.alpha_values_distr.txt.gz
# LUNG+vsCOAD-	300	v100005/train_data/paired_comparison_hyper.alpha.freq.thresholds.n2.p10/LUNG+vsCOAD-.alpha_values_distr.txt.gz
# COAD+vsSTAD-	300	v100005/train_data/paired_comparison_hyper.alpha.freq.thresholds.n2.p10/COAD+vsSTAD-.alpha_values_distr.txt.gz
# STAD+vsCOAD-	300	v100005/train_data/paired_comparison_hyper.alpha.freq.thresholds.n2.p10/STAD+vsCOAD-.alpha_values_distr.txt.gz
# STAD+vsCOAD-	300	v100005/train_data/paired_comparison_hyper.alpha.freq.thresholds.n1.p5/STAD+vsCOAD-.alpha_values_distr.txt.gz
# LIHC+vsLUNG-	300	v100005/train_data/paired_comparison_hyper.alpha.freq.thresholds.n2.p10/LIHC+vsLUNG-.alpha_values_distr.txt.gz
# LUNG+vsLIHC-	300	v100005/train_data/paired_comparison_hyper.alpha.freq.thresholds.n2.p10/LUNG+vsLIHC-.alpha_values_distr.txt.gz
# LIHC+vsSTAD-	300	v100005/train_data/paired_comparison_hyper.alpha.freq.thresholds.n2.p10/LIHC+vsSTAD-.alpha_values_distr.txt.gz
# STAD+vsLIHC-	300	v100005/train_data/paired_comparison_hyper.alpha.freq.thresholds.n2.p10/STAD+vsLIHC-.alpha_values_distr.txt.gz
# LUNG+vsSTAD-	300	v100005/train_data/paired_comparison_hyper.alpha.freq.thresholds.n2.p10/LUNG+vsSTAD-.alpha_values_distr.txt.gz
# STAD+vsLUNG-	300	v100005/train_data/paired_comparison_hyper.alpha.freq.thresholds.n2.p10/STAD+vsLUNG-.alpha_values_distr.txt.gz
# STAD+vsLUNG-	300	v100005/train_data/paired_comparison_hyper.alpha.freq.thresholds.n1.p5/STAD+vsLUNG-.alpha_values_distr.txt.gz
#
# Output: a dictionary {paired_comparison_ID:{'topK':[300, 300], 'paired_comparison_marker_file':['n2.p10.alpha_distr.txt', 'n1.p5.alpha_distr.txt']}}
#
def load_config_file_for_union_topK_markers_of_paired_comparison_alpha_value_distribution_files(config_file):
    config = {}
    with open(config_file, 'rt') as f:
        for line in f:
            paired_comparison_ID, topK, paired_comparison_alpha_value_distribution_file = line.rstrip().split('\t')
            if paired_comparison_ID not in config:
                config[paired_comparison_ID] = {'topK':[], 'paired_comparison_marker_file':[]}
            config[paired_comparison_ID]['topK'].append( int(topK) )
            config[paired_comparison_ID]['paired_comparison_marker_file'].append(paired_comparison_alpha_value_distribution_file)
    return(config)

# Input:
#   config: a dictionary {paired_comparison_ID:{'topK':[300, 300], 'paired_comparison_marker_file':['n2.p10.alpha_distr.txt', 'n1.p5.alpha_distr.txt']}. It is the output variable of function 'load_config_file_for_union_topK_markers_of_paired_comparison_alpha_value_distribution_files'.
def union_topK_markers_of_paired_comparison_alpha_value_distribution_file_with_alpha_threshold(config,
                                                                                               marker_type,
                                                                                               min_cpg_num,
                                                                                               output_union_markers_file,
                                                                                               min_read_num=-1):
    union_markers_list = {}
    markers_loaded_for_each_paired_comparison = {}
    print('Union markers', flush=True)
    paired_comparison_ID_list = sorted(config.keys())
    for pair_comparison_type in paired_comparison_ID_list:
        paired_comparison_marker_files_list = config[pair_comparison_type]['paired_comparison_marker_file']
        print('  %s' % pair_comparison_type, flush=True)
        for i in range(len(paired_comparison_marker_files_list)):
            paired_comparison_marker_file = paired_comparison_marker_files_list[i]
            topK = config[pair_comparison_type]['topK'][i]
            # extract '5' from 'p5' in filename 'v100001/train_data/paired_comparison_hyper.alpha.freq.thresholds.n1.p5/STAD+vsLUNG-.alpha_values_distr.txt.gz'
            if min_read_num == -1:
                min_read_num = int(extract_number_after_a_substring(paired_comparison_marker_file, 'p'))
            load_one_alpha_value_distribution_file_with_alpha_threshold_for_marker_selection(paired_comparison_marker_file,
                                                                                             union_markers_list,
                                                                                             markers_loaded_for_each_paired_comparison,
                                                                                             '%s.mincpg%d.minrf%d.%s'%(marker_type,min_cpg_num,min_read_num,pair_comparison_type),
                                                                                             min_cpg_num,
                                                                                             min_read_num,
                                                                                             topK)
    # Determine the best alpha value threshold for each unioned marker, when the marker has multiple different alpha value thresholds
    print('Determine the best alpha value threshold for each unioned marker', flush=True)
    for marker_index in union_markers_list.keys():
        if len(union_markers_list[marker_index]['alpha_threshold_list']) == 1:
            # only one alpha value threshold
            union_markers_list[marker_index][
                'alpha_threshold_for_final_use'] = 0  # index of alpha value in the alpha_threshold_list that will be for final use.
        else:
            # multiple alpha value thresholds
            if 'hyper' in marker_type:
                if 'stringent' in marker_type:
                    union_markers_list[marker_index]['alpha_threshold_for_final_use'] = np.argmax(
                        union_markers_list[marker_index]['alpha_threshold_list'])
                elif 'loose' in marker_type:
                    union_markers_list[marker_index]['alpha_threshold_for_final_use'] = np.argmin(
                        union_markers_list[marker_index]['alpha_threshold_list'])
            elif 'hypo' in marker_type:
                if 'stringent' in marker_type:
                    union_markers_list[marker_index]['alpha_threshold_for_final_use'] = np.argmin(
                        union_markers_list[marker_index]['alpha_threshold_list'])
                elif 'loose' in marker_type:
                    union_markers_list[marker_index]['alpha_threshold_for_final_use'] = np.argmax(
                        union_markers_list[marker_index]['alpha_threshold_list'])
    # Write to file
    print('Write to file:\n  output: %s' % output_union_markers_file, flush=True)
    with gzip.open(output_union_markers_file, 'wt') as fout:
        marker_index_list = sorted(union_markers_list.keys())
        fout.write('marker_index\talpha_threshold\tmarker_type\tmax_num_cpg\tpair_comparison_list\n')
        for marker_index in marker_index_list:
            index_of_alpha_threshold_for_use = union_markers_list[marker_index]['alpha_threshold_for_final_use']
            fout.write('%d\t%g\t%s\t%d\t%s\n' % (marker_index,
                                                 union_markers_list[marker_index]['alpha_threshold_list'][
                                                     index_of_alpha_threshold_for_use],
                                                 union_markers_list[marker_index]['pair_comparison_list'][
                                                     index_of_alpha_threshold_for_use],
                                                 union_markers_list[marker_index]['max_num_cpg'],
                                                 ','.join(union_markers_list[marker_index]['pair_comparison_list'])
                                                 ))

# Input file is the output of the function 'print_unified_markers'
#
# marker_index    alpha_threshold marker_type max_num_cpg pair_comparison_list
# 215 0.6 stringent.hyper.max.alpha.diff_0.3.COAD+vsLUAD-_r1_a0.6 5   stringent.hyper.max.alpha.diff_0.3.COAD+vsLUAD-_r1_a0.6,stringent.hyper.max.alpha.diff_0.3.COAD+vsLUSC-_r1_a0.4
#
# Output: a list of marker_index (strings)
#
def parse_markers_with_dynamic_alpha_thresholds(file):
    markers_list = {'marker_index':[], 'alpha_threshold':[], 'header_line':None, 'lines':[]}
    if file.endswith('gz'):
        f = gzip.open(file, 'rt')
    elif file == 'stdin':
        f = sys.stdin
    else:
        f = open(file)
    markers_list['header_line'] = next(f).rstrip() # skip the header line
    for line in f:
        items = line.rstrip().split('\t')
        marker_index = int(items[0])
        markers_list['marker_index'].append( marker_index )
        markers_list['alpha_threshold'].append( float(items[1]) )
        markers_list['lines'].append( line.rstrip() )
    if file != 'stdin':
        f.close()
    return(markers_list)

# Input file is the output of the function 'print_unified_markers'
#
# marker_index    alpha_threshold marker_type max_num_cpg pair_comparison_list
# 215 0.6 stringent.hyper.max.alpha.diff_0.3.COAD+vsLUAD-_r1_a0.6 5   stringent.hyper.max.alpha.diff_0.3.COAD+vsLUAD-_r1_a0.6,stringent.hyper.max.alpha.diff_0.3.COAD+vsLUSC-_r1_a0.4
#
# Output:
#    marker_index: a list of marker_index (int)
#    marker_1based_index_in_file: a list of marker_index (1-based index int)
#    rank: a list of rank for each marker in the specified "cancer_type"
#    tumor_frequency: a list of tumor frequency that each marker appear in the specified "cancer_type"
#
def parse_markers_file_with_cancertype_and_topk(file, cancer_type, topk):
    markers_list = {'marker_index':[], 'marker_1based_index_in_file':[], 'rank_in_paired_comparison':[], 'tumor_frequency':[], 'alpha_threshold':[], 'max_num_cpg':[], 'marker_type':[]}
    if file.endswith('gz'):
        f = gzip.open(file, 'rt')
    elif file == 'stdin':
        f = sys.stdin
    else:
        f = open(file)
    header_line = next(f).rstrip()
    line_index = 0
    for line in f:
        line_index += 1
        items = line.rstrip().split('\t')
        marker_index = int(items[0])
        pair_comparison_list_str = items[4]
        if cancer_type in pair_comparison_list_str:
            pair_comparison_list = pair_comparison_list_str.split(',')
            for paired_comparison in pair_comparison_list:
                if cancer_type in paired_comparison:
                    rank = int(extract_number_after_a_substring(paired_comparison, "_rank"))
                    if rank <= topk:
                        markers_list['marker_index'].append(marker_index)
                        markers_list['marker_1based_index_in_file'].append(line_index)
                        markers_list['rank_in_paired_comparison'].append(rank)
                        tumor_frequency = int(extract_number_after_a_substring(paired_comparison, "_r"))
                        markers_list['tumor_frequency'].append(tumor_frequency)
                        markers_list['alpha_threshold'].append(items[1])
                        markers_list['max_num_cpg'].append(items[3])
                        markers_list['marker_type'].append(paired_comparison)

    if file != 'stdin':
        f.close()
    return(markers_list)

# Input file is the output of the function 'print_unified_markers'
#
# marker_index    alpha_threshold marker_type max_num_cpg pair_comparison_list
# 215 0.6 stringent.hyper.max.alpha.diff_0.3.COAD+vsLUAD-_r1_a0.6 5   stringent.hyper.max.alpha.diff_0.3.COAD+vsLUAD-_r1_a0.6,stringent.hyper.max.alpha.diff_0.3.COAD+vsLUSC-_r1_a0.4
#
# Output:
#    marker_index: a list of marker_index (int)
#    marker_1based_index_in_file: a list of marker_index (1-based index int)
#    rank: a list of rank for each marker in the specified "cancer_type"
#    tumor_frequency: a list of tumor frequency that each marker appear in the specified "cancer_type"
#
def parse_markers_file_with_pancancer_and_topk(file, topk):
    all_cancer_types_list = get_cancer_types_list('long_name')
    markers_list = {'marker_index':[], 'marker_1based_index_in_file':[], 'rank_in_paired_comparison':[], 'alpha_threshold':[], 'max_num_cpg':[], 'marker_type':[]}
    if file.endswith('gz'):
        f = gzip.open(file, 'rt')
    elif file == 'stdin':
        f = sys.stdin
    else:
        f = open(file)
    header_line = next(f).rstrip()
    line_index = 0
    for line in f:
        line_index += 1
        items = line.rstrip().split('\t')
        marker_index = int(items[0])
        pair_comparison_list_str = items[4]
        pair_comparison_list = pair_comparison_list_str.split(',')
        for paired_comparison in pair_comparison_list:
            rank = int(extract_number_after_a_substring(paired_comparison, "_rank"))
            if rank <= topk:
                markers_list['marker_index'].append(marker_index)
                markers_list['marker_1based_index_in_file'].append(line_index)
                markers_list['rank_in_paired_comparison'].append(rank)
                markers_list['alpha_threshold'].append(items[1])
                markers_list['max_num_cpg'].append(items[3])
                markers_list['marker_type'].append(pair_comparison_list_str)
                break
    if file != 'stdin':
        f.close()
    return(markers_list)


#
# write "markers_list" that is generated by function "parse_markers_file_with_cancertype_and_topk" or "parse_markers_file_with_pancancer_and_topk"
# "sorted_indexes_list": indexes are 0-based
#
def write_selected_markers(markers_list, sorted_indexes_list, output_file):
    if output_file.endswith(".gz"):
        fout = gzip.open(output_file, 'wt')
    elif output_file == "stdout":
        fout = sys.stdout
    else:
        fout = open(output_file, 'wt')
    if 'tumor_frequency' in markers_list:
        # For a specific cancer type
        fout.write("marker_index\talpha_threshold\trank\ttumor_frequency\tmarker_type\tmax_num_cpg\n")
        for i in sorted_indexes_list:
            fout.write("%d\t%s\t%d\t%d\t%s\t%s\n"%(markers_list['marker_index'][i],
                                                   markers_list['alpha_threshold'][i],
                                                   markers_list['rank_in_paired_comparison'][i],
                                                   markers_list['tumor_frequency'][i],
                                                   markers_list['marker_type'][i],
                                                   markers_list['max_num_cpg'][i]))
    else:
        # For pancancer
        fout.write("marker_index\talpha_threshold\trank\tmarker_type\tmax_num_cpg\n")
        for i in sorted_indexes_list:
            fout.write("%d\t%s\t%d\t%s\t%s\n"%(markers_list['marker_index'][i],
                                               markers_list['alpha_threshold'][i],
                                               markers_list['rank_in_paired_comparison'][i],
                                               markers_list['marker_type'][i],
                                               markers_list['max_num_cpg'][i]))
    if output_file != "stdout":
        fout.close()


# Input file of alpha value distributions (REQUIRE marker_index sorted):
# alpha_value_distribution file (from the output of the function 'summarize_mary_file_binary_meth_values_for_distribution_file'):
# marker_index    max_num_cpg num_read    unique_alpha_values read_freq_of_unique_alpha_values    unique_meth_counts  read_freq_of_unique_meth_counts
# 2   7   122 0.429,0.571,0.714,0.857,1   1,2,27,42,50    3,4,5,6,7   1,2,27,42,50
# 27  9   39  0.5,0.625,0.75,0.778,0.875,0.889,1  1,2,9,1,12,1,13 4,5,6,7,8   1,2,9,13,14
# 61  12  44  0.75,0.833,0.917,1  2,11,12,19  9,10,11,12  2,11,13,18
# 63  5   100 0.6,0.8,1   4,23,73 3,4,5   4,23,73
# 65  5   83  0,0.2,0.4,0.6,0.8,1 1,3,2,9,26,42   0,1,2,3,4,5 1,3,2,9,26,42
#
# markers_list: a list of markers_index (strings). REQUIRE marker_index sorted.
# marker_type: 'hyper' or 'hypo'
# Ta: float, the alpha value threshold
#
# Output:
#   profile: a 1D numpy.array, with the size == len(markers_list)
#
def generate_plasma_sample_profile_with_given_markers_and_dynamic_alpha_threshold_by_parsing_alpha_value_distribution_file_quick_version(file, markers_list, marker_type):
    n_markers = len(markers_list['marker_index'])
    profile = np.empty(n_markers, dtype=float)
    profile[:] = np.nan
    total_reads = 0
    current_marker_pointer_in_markers_list = 0
    with gzip.open(file, 'rt') as f:
        next(f) # skip the header line
        line = f.readline()
        while line != "":
            marker_index, max_num_cpg, num_read, unique_alpha_values_str, read_freq_of_unique_alpha_values_str, _, _ = line.rstrip().split('\t')
            marker_index = int(marker_index)
            if current_marker_pointer_in_markers_list >= n_markers:
                break
            if marker_index > markers_list['marker_index'][current_marker_pointer_in_markers_list]:
                while True:
                    current_marker_pointer_in_markers_list += 1
                    if current_marker_pointer_in_markers_list >= n_markers:
                        break
                    if marker_index <= markers_list['marker_index'][current_marker_pointer_in_markers_list]:
                        break
                continue
            if current_marker_pointer_in_markers_list >= n_markers:
                break
            if marker_index < markers_list['marker_index'][current_marker_pointer_in_markers_list]:
                line = f.readline()
                continue
            # now 'marker_index == markers_list_int[current_marker_pointer_in_markers_list]'
            # idx = markers_list['marker_index'].index(marker_index)
            idx = current_marker_pointer_in_markers_list
            Ta = markers_list['alpha_threshold'][idx]
            current_marker_pointer_in_markers_list += 1 # for the next round comparison
            unique_alpha_values = np.array(list(map(float, unique_alpha_values_str.split(','))))
            read_freq_of_unique_alpha_values = np.array(
                list(map(int, read_freq_of_unique_alpha_values_str.split(','))))
            # identify tumor reads in tumor
            if 'hyper' in marker_type:
                # tumor reads in tumor & plasma are defined as 'alpha>Ta';
                read_cov_of_criterion = get_number_of_reads_meeting_criterion(unique_alpha_values,
                                                                              read_freq_of_unique_alpha_values,
                                                                              Ta,
                                                                              '>')
            elif 'hypo' in marker_type:
                # tumor reads in tumor & plasma are defined as 'alpha<Ta';
                read_cov_of_criterion = get_number_of_reads_meeting_criterion(unique_alpha_values,
                                                                              read_freq_of_unique_alpha_values,
                                                                              Ta,
                                                                              '<')
            profile[idx] = read_cov_of_criterion
            total_reads += int(num_read)
    return( (profile, total_reads) )


# Input:
#   input_alpha_value_distribution_files_list: a list of input_alpha_value_distribution_files. Each file is for a sample
#   markers_list: a dictionary {'marker_index':[], 'alpha_threshold':[], 'line':[]}. It is the output of function 'parse_markers_with_dynamic_alpha_thresholds'
#   marker_type: 'hyper' or 'hypo'
#
# Output:
#   data: a 2D numpy.array. #samples X #markers. Rows are samples, columns are markers, values are tumor read counts.
#
def generate_matrix_for_many_plasma_samples_with_given_markers_and_dynamic_alpha_threshold_by_parsing_alpha_value_distribution_files(input_alpha_value_distribution_files_list,
                                                                                                                                     markers_list,
                                                                                                                                     marker_type):
    n_sample = len(input_alpha_value_distribution_files_list)
    total_reads_list = []
    for i in range(n_sample):
        input_sample_alpha_distr_file = input_alpha_value_distribution_files_list[i]
        # print('  (%d/%d) %s'%(i+1, n_sample, input_sample_alpha_distr_file), end="\t", flush=True)
        profile, total_reads = generate_plasma_sample_profile_with_given_markers_and_dynamic_alpha_threshold_by_parsing_alpha_value_distribution_file_quick_version(input_sample_alpha_distr_file,
                                                                                                                                                       markers_list,
                                                                                                                                                       marker_type)
        # print(datetime.now(), flush=True)
        if i==0:
            # data = csr_matrix((n_sample, len(profile)), dtype=np.single).toarray()
            data = lil_matrix((n_sample, len(profile)), dtype=np.single)
        data[i,:] = profile
        total_reads_list.append(total_reads)
    return( (data, np.array(total_reads_list)) )
    # columns_index_with_nonzeros = get_columns_index_with_nonzeros(data)
    # markers_list_with_nonzeros = {'marker_index':[], 'alpha_threshold':[], 'lines':[], 'header_line':markers_list['header_line']}
    # for i in columns_index_with_nonzeros:
    #     markers_list_with_nonzeros['marker_index'].append(markers_list['marker_index'][i])
    #     markers_list_with_nonzeros['alpha_threshold'].append(markers_list['alpha_threshold'][i])
    #     markers_list_with_nonzeros['lines'].append(markers_list['lines'][i])
    # return( (data[:,columns_index_with_nonzeros].toarray(), markers_list_with_nonzeros) )

###########
# Sone routine functions
###########

#
# Input:
#   samples_info: output of function 'parse_csv_file_with_header_line_and_first_two_columns'. We will use its two keys 'sample' and 'class'
#   classification_type: 'NvsC', 'CHvsC', 'NCHvsC' for cancer detection, and 'C4', 'C5' for cancer typing
#
# Return:
#   samples_for_use: a dictionary with keys 'positive', 'negative', 'all' and 'label_for_all' and 'unique_class_names' for cancer detection, and 'CC', ..., 'LV', 'ST', 'all' and 'label_for_all' and 'unique_class_names' for cancer typing
#
def organize_samples_for_use(samples_info, classification_type):
    all_cancer_types = ['CC', 'LC', 'LG', 'LV', 'ST']
    n_sample = len(samples_info['sample'])
    samples_for_use = None
    if 'vs' in classification_type:
        # cancer detection
        samples_for_use = {'positive':[], 'negative':[], 'all':[], 'label_for_all':[], 'unique_class_names':['N','C']}
        if classification_type == 'NvsC':
            # cancer detection: control (normal) VS case (all cancer types)
            samples_for_use['negative'] = [samples_info['sample'][i] for i in range(n_sample) if samples_info['class'][i]=='N']
        elif classification_type == 'CHvsC':
            # cancer detection: control (cirrhosis) VS case (all cancer types)
            samples_for_use['negative'] = [samples_info['sample'][i] for i in range(n_sample) if samples_info['class'][i] == 'CH']
        elif classification_type == 'NCHvsC':
            # cancer detection: control (normal + cirrhosis) VS case (all cancer types)
            samples_for_use['negative'] = [samples_info['sample'][i] for i in range(n_sample) if samples_info['class'][i] in ['N','CH']]
        else:
            sys.stderr.write('Error: classification_type (%s) is invalid in function organize_samples_for_use().\nExit.\n' % classification_type)
        samples_for_use['positive'] = [samples_info['sample'][i] for i in range(n_sample) if samples_info['class'][i] in all_cancer_types]
        samples_for_use['all'] = samples_for_use['negative'] + samples_for_use['positive']
        samples_for_use['label_for_all'] = [0] * len(samples_for_use['negative']) + [1] * len(
            samples_for_use['positive'])
    elif classification_type.startswith('C'):
        # cancer typing
        # all non-cancer samples will be excluded for 'samples_for_use'
        if classification_type == 'C4':
            # cancer typing of 4 cancer types: CC, LC_LG, LV, ST
            class_names = ['N', 'CC', 'LC_LG', 'LV', 'ST']
        elif classification_type == 'C5':
            # cancer typing of 5 cancer types: CC, LC, LG, LV, ST
            class_names = ['N', 'CC', 'LC', 'LG', 'LV', 'ST']
        else:
            sys.stderr.write('Error: classification_type (%s) is invalid in function organize_samples_for_use().\nExit.\n'%classification_type)
        samples_for_use = {'all':[], 'label_for_all':[], 'unique_class_names':class_names[1:]}
        total_sample = len(samples_info['sample'])
        for c in samples_for_use['unique_class_names']:
            c_index = class_names.index(c) # those cancer types have c_index starting from 1.
            samples_for_use[c] = [samples_info['sample'][i] for i in range(total_sample) if samples_info['class'][i] in c]
            if len(samples_for_use[c]) == 0: continue
            samples_for_use['all'] += samples_for_use[c]
            samples_for_use['label_for_all'] += [c_index] * len(samples_for_use[c])
    return(samples_for_use)

# file format
# header line (optional): sample_name feature_name_1 feature_name_2 ...
# in the following lines:
# Column 1:	sample name
# Column 2-:  value of each feature for this sample name
#
# Each row is a sample
# Last row (optional): marker_index
def read_matrix_gz_file_with_selected_samples(gz_file, selected_samples, header_feature_names=False):
    data = {'sample': [], 'mat': [], 'marker_index': [], 'label': [],
            'feature_names': []}  # , 'class_names':['N','CC','LC','LG','LV','ST']}
    overlap_samples = [] # samples are in the order of those samples in file
    with gzip.open(gz_file, 'rt') as f:
        if header_feature_names:
            data['feature_names'] = f.readline().rstrip().split('\t')[1:]  # header line is a list of feature names
        for line in f:
            items = line.rstrip().replace('NA', 'nan').split('\t')
            sample_name = items[0]  # plasma-489-F-ST
            if sample_name != 'marker_index':
                if sample_name not in selected_samples: continue
                overlap_samples.append(sample_name)
                data['mat'].append(np.array(list(map(float, items[1:]))))
            else:
                data['marker_index'] = np.array(list(map(int, items[1:])))
    samples_for_use = [s for s in selected_samples if s in overlap_samples] # samples are sorted by the order of 'selected_samples'
    index_of_samples_for_use = np.array([overlap_samples.index(s) for s in samples_for_use])
    data['sample'] = samples_for_use
    data['mat'] = np.array(data['mat'])[index_of_samples_for_use,:]
    return (data)

# file format
# header line (optional): sample_name feature_name_1 feature_name_2 ...
# in the following lines:
# Column 1:	sample name
# Column 2-:  value of each feature for this sample name
#
# Each row is a sample
# Last row (optional): marker_index
def read_matrix_gz_file_with_selected_markers_and_samples(gz_file, selected_samples, selected_markers_list, header_feature_names=False):
    selected_markers_1darray = np.array(selected_markers_list) - 1  # make marker_index 0-based
    data = {'sample': [], 'mat': [], 'marker_index': [], 'label': [],
            'feature_names': []}  # , 'class_names':['N','CC','LC','LG','LV','ST']}
    overlap_samples = [] # samples are in the order of those samples in file
    with gzip.open(gz_file, 'rt') as f:
        if header_feature_names:
            tmp = f.readline().rstrip().split('\t')[1:]  # header line is a list of feature names
            data['feature_names'] = [tmp[i] for i in selected_markers_1darray]
        else:
            data['feature_names'] = list(map(str, selected_markers_list))
        for line in f:
            items = line.rstrip().replace('NA', 'nan').split('\t')
            sample_name = items[0]  # plasma-489-F-ST
            if sample_name != 'marker_index':
                if sample_name not in selected_samples: continue
                overlap_samples.append(sample_name)
                data['mat'].append(np.array(list(map(float, items[1:])))[selected_markers_1darray])
            else:
                data['marker_index'] = np.array(list(map(int, items[1:])))[selected_markers_1darray]
    samples_for_use = [s for s in selected_samples if s in overlap_samples] # samples are sorted by the order of 'selected_samples'
    index_of_samples_for_use = np.array([overlap_samples.index(s) for s in samples_for_use])
    data['sample'] = samples_for_use
    data['mat'] = np.array(data['mat'])[index_of_samples_for_use,:]
    return (data)

# file format
# Line 1 (header): sample_name_1 sample_name_2 ...
# Column 1: marker_1_values_for_samples
# Column 2: marker_2_values_for_samples
# ...
# Each row is a marker
def read_transposed_matrix_gz_file_with_selected_samples(gz_file, selected_samples):
    data = {'sample': [], 'mat': [], 'marker_index': [], 'label': [], 'unique_class_names':[]}
    with gzip.open(gz_file, 'rt') as f:
        samples_in_file = next(f).rstrip().split('\t')  # process the header line, e.g., 'plasma-489-F-ST plasma-110-F-LG ...'
        overlap_samples = list(set(samples_in_file).intersection(selected_samples))
        if len(overlap_samples)==0:
            print('    NO selected samples exist in the file:\n      file: %s'%gz_file)
            return(data)
        samples_for_use = [s for s in selected_samples if s in overlap_samples]
        data['sample'] = samples_for_use
        index_of_samples_for_use = np.array([samples_in_file.index(s) for s in samples_for_use])
        for line in f:
            items = line.rstrip().replace('NA', 'nan').split('\t')
            profile = np.array(list(map(float, items)))
            data['mat'].append(profile[index_of_samples_for_use])
    print('   == begin to transpose the matrix ...', flush=True)
    print('      current time:', datetime.now(), flush=True)
    data['mat'] = np.transpose(np.array(data['mat']))
    print('      current time:', datetime.now(), flush=True)
    print('   == end of transposing the matrix', flush=True)
    return (data)

# file format
# Line 1 (header): sample_name_1 sample_name_2 ...
# Column 1: marker_1_values_for_samples
# Column 2: marker_2_values_for_samples
# ...
# Each row is a marker
def read_transposed_matrix_gz_file_with_selected_markers_and_samples(gz_file, selected_samples, selected_markers_list):
    data = {'sample': [], 'mat': [], 'marker_index': [], 'label': [], 'unique_class_names':[]}
    with gzip.open(gz_file, 'rt') as f:
        samples_in_file = next(f).rstrip().split('\t')  # process the header line, e.g., 'plasma-489-F-ST plasma-110-F-LG ...'
        overlap_samples = list(set(samples_in_file).intersection(selected_samples))
        if len(overlap_samples)==0:
            print('    NO selected samples exist in the file:\n      file: %s'%gz_file)
            return(data)
        samples_for_use = [s for s in selected_samples if s in overlap_samples]
        data['sample'] = samples_for_use
        index_of_samples_for_use = np.array([samples_in_file.index(s) for s in samples_for_use])
        i = 0
        for line in f:
            i += 1
            if i not in selected_markers_list: continue
            data['feature_names'].append(str(i))
            items = line.rstrip().replace('NA', 'nan').split('\t')
            data['mat'].append(np.array(list(map(float, items)))[index_of_samples_for_use])
    print('   == begin to transpose the matrix ...', flush=True)
    print('      current time:', datetime.now(), flush=True)
    data['mat'] = np.transpose(np.array(data['mat']))
    print('      current time:', datetime.now(), flush=True)
    print('   == end of transposing the matrix', flush=True)
    return (data)

## file format of selected markers: marker_index(1-based) marker_name
# 246	COAD.marker_1503
# 247	LIHC.marker_1503
# 248	LUAD.marker_1503
# 249	LUSC.marker_1503
# 250	STAD.marker_1503
# 286	COAD.marker_1561
# ...
def read_selected_markers_index_file(file):
    marker_index_list = []
    marker_name_list = []
    if file.endswith('gz'):
        f = gzip.open(file, 'rt')
    else:
        f = open(file, 'rt')
    for line in f:
        if line.startswith('#'): continue # skip header lines starting with '#'
        if line.startswith('marker'): continue  # skip header lines starting with 'marker'
        items = line.split('\t')
        marker_index = int(items[0])
        marker_name = items[1]
        marker_index_list.append(marker_index)
        marker_name_list.append(marker_name)
    f.close()
    return ((marker_index_list, marker_name_list))


#
# Input:
#
#   data: a dictionary with keys 'sample', 'mat', 'label'. This is the output of function 'read_matrix_gz_file_with_selected_samples' or 'read_transposed_matrix_gz_file_with_selected_samples'
#
#   all_sample_info: a dictionary with keys 'positive', 'negative', 'all' and 'label_for_all' and 'unique_class_names' for cancer detection, and 'CC', ..., 'LV', 'ST', 'all' and 'label_for_all' and 'unique_class_names' for cancer typing. This is the output of function 'organize_samples_for_use'
#
# Output:
#   Fill in data['label'] and data['unique_class_names']
#
def add_class_labels_to_data_by_given_sample_info(data, all_sample_info):
    for s in data['sample']:
        if s not in all_sample_info['all']:
            sys.stderr.write('Error for func add_class_labels_to_data_for_cancer_detection_by_given_sample_info:\n  %s of data does not exist in all_sample_info\nExit.\n'%s)
            sys.exit(-1)
        s_index = all_sample_info['all'].index(s)
        data['label'].append( all_sample_info['label_for_all'][s_index] )
    data['label'] = np.array(data['label'])
    data['unique_class_names'] = all_sample_info['unique_class_names']


# file format
# header line (optional): sample_name feature_name_1 feature_name_2 ...
# in the following lines:
# Column 1:	sample name
# Column 2-:  value of each feature for this sample name
#
# Each row is a sample
# Last row (optional): marker_index
def read_matrix_gz_file(gz_file, header_feature_names=False):
    data = {'sample': [], 'mat': [], 'marker_index': [], 'label': [],
            'feature_names': []}  # , 'class_names':['N','CC','LC','LG','LV','ST']}
    with gzip.open(gz_file, 'rt') as f:
        if header_feature_names:
            data['feature_names'] = f.readline().rstrip().split('\t')[1:]  # header line is a list of feature names
        for line in f:
            items = line.rstrip().replace('NA', 'nan').split('\t')
            sample_name = items[0]  # plasma-489-F-ST
            if sample_name != 'marker_index':
                data['sample'].append(sample_name)
                data['mat'].append(np.array(list(map(float, items[1:]))))
            else:
                data['marker_index'] = np.array(list(map(int, items[1:])))
    data['mat'] = np.array(data['mat'])
    return (data)


# file format
# header line (optional): sample_name feature_name_1 feature_name_2 ...
# in the following lines:
# Column 1:	sample name
# Column 2-:  value of each feature for this sample name
#
# Each row is a sample
# Last row (optional): marker_index
# 'selected_markers_list' is a list that contains selected marker_index (1-based).
def read_matrix_gz_file_with_selected_markers_list(data_gz_file, selected_markers_list, header_feature_names=False):
    selected_markers_1darray = np.array(selected_markers_list) - 1  # make marker_index 0-based
    data = {'sample': [], 'mat': [], 'marker_index': [], 'label': []}  # , 'class_names':['N','CC','LC','LG','LV','ST']}
    with gzip.open(data_gz_file, 'rt') as f:
        if header_feature_names:
            tmp = f.readline().rstrip().split('\t')[1:]  # header line is a list of feature names
            data['feature_names'] = [tmp[i] for i in selected_markers_1darray]
        for line in f:
            items = line.rstrip().replace('NA', 'nan').split('\t')
            sample_name = items[0]  # plasma-489-F-ST
            if sample_name != 'marker_index':
                data['sample'].append(sample_name)
                data['mat'].append(np.array(list(map(float, items[1:])))[selected_markers_1darray])
            else:
                data['marker_index'] = np.array(list(map(int, items[1:])))[selected_markers_1darray]
    data['mat'] = np.array(data['mat'])
    return (data)


# file format
# Line 1 (header): sample_name_1 sample_name_2 ...
# Column 1: marker_1_values_for_samples
# Column 2: marker_2_values_for_samples
# ...
# Each row is a marker
def read_transposed_matrix_gz_file(gz_file):
    data = {'sample': [], 'mat': [], 'marker_index': [], 'label': []}  # , 'class_names':['N','CC','LC','LG','LV','ST']}
    with gzip.open(gz_file, 'rt') as f:
        data['sample'] = next(f).rstrip().split(
            '\t')  # process the header line, e.g., 'plasma-489-F-ST plasma-110-F-LG ...'
        for line in f:
            items = line.rstrip().replace('NA', 'nan').split('\t')
            data['mat'].append(np.array(list(map(float, items))))
    print('   == begin to transpose the matrix ...')
    print('      current time:', datetime.now())
    sys.stdout.flush()
    data['mat'] = np.transpose(np.array(data['mat']))
    print('      current time:', datetime.now())
    print('   == end of transposing the matrix')
    sys.stdout.flush()
    return (data)


# 'selected_markers_list' is a list that contains selected marker_index (1-based).
# 'selected_markers_dict': a fast implementation, is a dictionary whose keys are marker_index (1-based), and show if this marker_index is selected (True) or not (False)
def read_transposed_matrix_gz_file_with_selected_markers_list(data_gz_file, selected_markers_list):
    data = {'sample': [], 'mat': [], 'marker_index': [], 'label': []}  # , 'class_names':['N','CC','LC','LG','LV','ST']}
    with gzip.open(data_gz_file, 'rt') as f:
        data['sample'] = next(f).rstrip().split(
            '\t')  # process the header line, e.g., 'plasma-489-F-ST plasma-110-F-LG ...'
        i = 0
        for line in f:
            i += 1
            if i not in selected_markers_list: continue
            items = line.rstrip().replace('NA', 'nan').split('\t')
            data['mat'].append(np.array(list(map(float, items))))
    print('   == begin to transpose the matrix ...')
    print('      current time:', datetime.now())
    sys.stdout.flush()
    data['mat'] = np.transpose(np.array(data['mat']))
    print('      current time:', datetime.now())
    print('   == end of transposing the matrix')
    sys.stdout.flush()
    return (data)

# 'data' is the returned value of the function 'read_matrix_gz_file()' and function 'add_class_labels_to_data()'
def get_data_of_specified_samples(samples_selected, data):
    index_samples_selected = get_indexes(samples_selected, data['sample'])
    if np.any(np.isnan(index_samples_selected)):
        index = np.where(np.isnan(index_samples_selected))[0][
            0]  # the first sample which does not occur in data samples: data['sample']
        sys.exit(
            'Error of function get_data_of_specified_samples (samples_selected, data): train sample %s does not occur data[sample]. Exit.\n' %
            samples_selected[index])
    X = data['mat'][index_samples_selected, :]
    y = data['label'][index_samples_selected]
    return ((X, y))


def get_class_labels_from_samples_names_for_cancer_detection(samples_names):
    class_names = ['N', 'C']
    labels = []
    cancer_samples_indexes = []
    i = 0
    for sample_name in samples_names:
        class_name = re.search(r'-F-[A-Z]+$', sample_name)[0].replace('-F-', '')
        if class_name == 'N':
            label = 0
        elif class_name == 'CH':
            label = 0
        else:
            label = 1
        labels.append(label)
        if class_name != 'N' and class_name != 'CH':
            cancer_samples_indexes.append(i)
        i += 1
    labels = np.array(labels)
    return ((labels, class_names))

def add_class_labels_to_data_for_cancer_detection(data):
    data['class_names'] = ['N', 'C']
    data['label'] = []
    cancer_samples_indexes = []
    i = 0
    for sample_name in data['sample']:
        class_name = re.search(r'-F-[A-Z]+$', sample_name)[0].replace('-F-', '')
        if class_name == 'N':
            label = 0
        elif class_name == 'CH':
            label = 0
        else:
            label = 1
        data['label'].append(label)
        if class_name != 'N' and class_name != 'CH':
            cancer_samples_indexes.append(i)
        i += 1
    data['label'] = np.array(data['label'])
    return (np.array(cancer_samples_indexes))

def add_class_labels_to_data_for_cancertyping(data, num_cancer_types=4):
    if num_cancer_types == 5:
        data['class_names'] = ['N', 'CC', 'LC', 'LG', 'LV', 'ST']
    elif num_cancer_types == 4:
        data['class_names'] = ['N', 'CC', 'LC_LG', 'LV', 'ST']
    data['label'] = []
    cancer_samples_indexes = []
    i = 0
    for sample_name in data['sample']:
        class_name = re.search(r'-F-[A-Z,0-9]+$', sample_name)[0].replace('-F-', '')
        # if class_name == 'N' or class_name == 'N1' or class_name=='CH' or class_name == 'CH1':
        if 'N' in class_name or 'CH' in class_name:
            label = 0
        else:
            exists = [class_name in c for c in data['class_names']]
            if True in exists:
                label = exists.index(True)
            else:
                data['class_names'].append(class_name)
                label = len(data['class_names']) - 1
            cancer_samples_indexes.append(i)
        data['label'].append(label)
        i += 1
    data['label'] = np.array(data['label'])
    return (np.array(cancer_samples_indexes))


# top_features_index are 0-based and sorted features from the most important to the least important.
def skfeature_selection_func_for_two_classes_original(X, y, feasel_fun):
    n_samples, n_features = X.shape
    if 'gini' in feasel_fun:
        scores = gini_index.gini_index(X, y)
        top_features_index = gini_index.feature_ranking(scores)
        scores = scores[top_features_index]
    elif 'fisher' in feasel_fun:
        scores = fisher_score.fisher_score(X, y)
        top_features_index = fisher_score.feature_ranking(scores)
        scores = scores[top_features_index]
    return((top_features_index, scores))

# 'type' is 'reason of selecting this feature'
# 'unified_features' are feature indexes (0-based)
def skfeature_selection_func_for_multi_classes_original(X, y, feasel_fun, topK=100):
    num_samples, num_all_features = X.shape
    if topK==-1:
        topK = num_all_features
    elif topK > num_all_features:
        topK = num_all_features
    unique_class_labels = np.unique(y)
    num_unique_labels = len(unique_class_labels)
    alt_class_label = max(unique_class_labels) + 1
    unified_features = {'union_feature_index':[], 'type':[], 'feasel_fun':feasel_fun, 'topK':topK,
                        'original':{'feature_index':[],'score':[],'type':[]}}
    print('Select OvR features for %d classes'%num_unique_labels)
    for l in unique_class_labels:
        print('  OvR_%d (%s), '%(l,str(datetime.now())), end='', flush=True)
        y_two_class = np.copy(y)
        y_two_class[np.where(y!=l)[0]] = alt_class_label # now only two class labels: l and alt_class_label
        top_features_index, scores = skfeature_selection_func_for_two_classes_original(X, y_two_class, feasel_fun)
        topK_updated = min(topK, len(top_features_index))
        top_features_index = top_features_index[0:topK_updated]
        scores = scores[0:topK_updated]
        unified_features['original']['feature_index'] += list(top_features_index)
        unified_features['original']['score'] += list(scores)
        unified_features['original']['type'] += ['OvR_%d'%(l)]*topK_updated
    print('',flush=True)
    if num_unique_labels>2:
        print('Select OvO features for %d classes' % num_unique_labels)
        ovo_class_pairs = list(combinations(unique_class_labels,2))
        for l1, l2 in ovo_class_pairs:
            print('  OvO_%dv%d (%s), '%(l1,l2,str(datetime.now())), end='', flush=True)
            samples_of_two_class = np.array(list(np.where(y==l1)[0]) + list(np.where(y==l2)[0]))
            X_of_two_class = X[samples_of_two_class,:]
            y_of_two_class = y[samples_of_two_class]
            top_features_index, scores = skfeature_selection_func_for_two_classes_original(X_of_two_class, y_of_two_class, feasel_fun)
            topK_updated = min(topK, len(top_features_index))
            top_features_index = top_features_index[0:topK_updated]
            scores = scores[0:topK_updated]
            unified_features['original']['feature_index'] += list(top_features_index)
            unified_features['original']['score'] += list(scores)
            unified_features['original']['type'] += ['OvO_%dv%d' % (l1,l2)] * topK_updated
        print('', flush=True)
    unified_features['original']['feature_index'] = np.array(unified_features['original']['feature_index'])
    unified_features['original']['score'] = np.array(unified_features['original']['score'])
    unified_features['union_feature_index'] = np.unique(unified_features['original']['feature_index'])
    num_unified_features = len(unified_features['union_feature_index'])
    print('Union %d originally selected features to obtain %d unique features for multiple classes'%(len(unified_features['original']['feature_index']),num_unified_features), flush=True)
    for i in range(num_unified_features):
        f = unified_features['union_feature_index'][i]
        duplicates_index = list( np.where(f == unified_features['original']['feature_index'])[0] )
        duplicates_type = itemgetter(*duplicates_index)(unified_features['original']['type'])
        if isinstance(duplicates_type,str):
            duplicates_type = [duplicates_type]
        unified_features['type'].append(','.join(duplicates_type))
    print('Feature index in the output is 0-based.\nDone for skfeature_selection_func_for_multi_classes()', flush=True)
    return(unified_features)

#
# top_features_index are 0-based and sorted features from the most important to the least important.
# y: a list of any of two non-negative integers
#
# Procedure:
#   Step 1. Select features with the fraction of non-zero values in positive samples > fraction of non-zero values in negative samples
#   Step 2. Among features selected in Step 1, sort features by their 'gini' or 'fisher' scores
#
def feature_selection_func_for_two_classes(X, y, samples_list_for_debug_only, direction='+1-0', feasel_fun='fisher'):
    positive_class_label = int(re.search(r'\+([0-9]*)', direction).group(1))
    negative_class_label = int(re.search(r'\-([0-9]*)', direction).group(1))
    n_samples, n_features = X.shape
    samples_positive = np.where(y == positive_class_label)[0]
    samples_negative = np.where(y == negative_class_label)[0]
    n_samples_positive = len(samples_positive)
    n_samples_negative = len(samples_negative)
    samples_of_two_class = np.array(list(samples_positive) + list(samples_negative))
    nnz_for_samples_positive = np.count_nonzero(X[samples_positive, :], axis=0)
    nnz_for_samples_negative = np.count_nonzero(X[samples_negative, :], axis=0)
    if 'gini' in feasel_fun:
        features_with_signals = np.where((nnz_for_samples_positive / n_samples_positive) > (nnz_for_samples_negative / n_samples_negative))[0]
        # features_with_signals = np.where(nnz_for_samples_positive > nnz_for_samples_negative)[0]
        if len(features_with_signals) == 0:
            return (([], [], samples_positive, samples_negative))
        scores = gini_index.gini_index(X[np.ix_(samples_of_two_class,features_with_signals)], y[samples_of_two_class])
        top_features_index = gini_index.feature_ranking(scores)
        scores = scores[top_features_index]
        return ((features_with_signals[top_features_index], scores, samples_positive, samples_negative))
    elif 'fisher' in feasel_fun:
        features_with_signals = np.where((nnz_for_samples_positive / n_samples_positive) > (nnz_for_samples_negative / n_samples_negative))[0]
        # features_with_signals = np.where(nnz_for_samples_positive > nnz_for_samples_negative)[0]
        if len(features_with_signals) == 0:
            return (([], [], samples_positive, samples_negative))
        scores = fisher_score.fisher_score(X[np.ix_(samples_of_two_class,features_with_signals)], y[samples_of_two_class])
        top_features_index = fisher_score.feature_ranking(scores)
        scores = scores[top_features_index]
        return ((features_with_signals[top_features_index], scores, samples_positive, samples_negative))
    elif 'nnzdiff' in feasel_fun:
        # difference of non-zeros in positive class and negative class. The larger this difference is, the higher ranking this marker is.
        # This method uses binary values of each feature, not their actual values.
        top_features_index = np.argsort(nnz_for_samples_negative - nnz_for_samples_positive) # rank 1 is the feature with the largest difference btw nnz of positive samples and nnz of negative samples
        sorted_difference = nnz_for_samples_positive[top_features_index] - nnz_for_samples_negative[top_features_index]
        top_features_index = top_features_index[sorted_difference > 0]
        scores = sorted_difference[:len(top_features_index)]
        return((top_features_index, scores, samples_positive, samples_negative))
    else:
        sys.stderr.write('Error: feature selection method (%s) must be gini, fisher, and cardinal_greater!\nExit.\n'%feasel_fun)
        sys.exit(-1)

#
# top_features_index are 0-based and sorted features from the most important to the least important.
# y: a list of any of two non-negative integers
#
# Procedure:
#   Step 1. Select features with the fraction of non-zero values in positive samples > fraction of non-zero values in negative samples
#   Step 2. Among features selected in Step 1, sort features by their 'gini' or 'fisher' scores
#
def feature_selection_func_for_multiclass(X, y, samples_list_for_debug_only, feasel_fun='fisher'):
    num_samples, num_all_features = X.shape
    max_topK = max(5000, num_all_features) # assume the max topK for One vs One is 5000
    unique_class_labels = np.unique(y)
    num_unique_labels = len(unique_class_labels)
    features_selected = {}
    print('Select OvO features for %d classes: %s' % (num_unique_labels,', '.join(list(map(str,unique_class_labels)))))
    ovo_class_pairs = list(combinations(unique_class_labels, 2))
    for l1, l2 in ovo_class_pairs:
        if l1!=0 and l2!=0:
            direction = '+%d-%d' % (l1, l2)
            print('  OvO_%s (%s), ' % (direction, str(datetime.now())), end='', flush=True)
            top_features_index, scores, _, _ = feature_selection_func_for_two_classes(X, y, samples_list_for_debug_only, direction,
                                                                                      feasel_fun)  # 0-based top_features_index
            features_selected['%s_OvO_%s'%(feasel_fun,direction)] = {'feature_index':top_features_index, 'score':scores}
            direction = '+%d-%d' % (l2, l1)
            print('  OvO_%s (%s), ' % (direction, str(datetime.now())), end='', flush=True)
            top_features_index, scores, _, _ = feature_selection_func_for_two_classes(X, y, samples_list_for_debug_only, direction,
                                                                                      feasel_fun)  # 0-based top_features_index
            features_selected['%s_OvO_%s'%(feasel_fun,direction)] = {'feature_index':top_features_index, 'score':scores}
            continue
        if l2==0:
            direction = '+%d-%d'%(l1, l2)
            print('  OvO_%s (%s), ' % (direction, str(datetime.now())), end='', flush=True)
            top_features_index, scores, _, _ = feature_selection_func_for_two_classes(X, y, samples_list_for_debug_only, direction,
                                                                                      feasel_fun) # 0-based top_features_index
            features_selected['%s_OvO_%s' % (feasel_fun, direction)] = {'feature_index': top_features_index,
                                                                        'score': scores}
            continue
        if l1 == 0:
            direction = '+%d-%d' % (l2, l1)
            print('  OvO_%s (%s), ' % (direction, str(datetime.now())), end='', flush=True)
            top_features_index, scores, _, _ = feature_selection_func_for_two_classes(X, y, samples_list_for_debug_only, direction,
                                                                                      feasel_fun)  # 0-based top_features_index
            features_selected['%s_OvO_%s' % (feasel_fun, direction)] = {'feature_index': top_features_index,
                                                                        'score': scores}
            continue
    print('Feature index in the output is 0-based.\nDone for feature_selection_func_for_multiclass()', flush=True)
    return(features_selected)

# Input:
#   features_selected: a dictionary with keys ('gini_OvO_+1-0', 'gini_OvO_+0-2', 'gini_OvO_+1-2', 'gini_OvO_+2-1') and values (a dictionary {'feature_index':top_features_index, 'score': scores}). top_features_index are 0-based.
#   feasel_fun: 'gini' or 'fisher'
# Output:
#   unified_features: a dictionary with keys ('union_feature_index', 'type', 'score', 'feasel_fun', 'topK')
def union_selected_features_of_multi_class(features_selected, feasel_fun, topK=10):
    unified_features = {'union_feature_index': [], 'type': [], 'score':[], 'feasel_fun': feasel_fun, 'topK': topK}
    for type, features in features_selected.items():
        topK_updated = min(topK, len(features['feature_index']))
        if topK_updated==0: continue
        for i in range(topK_updated):
            rank = i+1
            fea_index = features['feature_index'][i]
            score = features['score'][i]
            if fea_index in unified_features['union_feature_index']:
                idx = unified_features['union_feature_index'].index(fea_index)
                unified_features['type'][idx].append('%s_rank_%d'%(type,rank))
                unified_features['score'][idx].append(score)
            else:
                unified_features['union_feature_index'].append(fea_index)
                unified_features['type'].append(['%s_rank_%d'%(type,rank)])
                unified_features['score'].append([score])
    return(unified_features)

# 'unified_features' comes from function's output 'union_selected_features_of_multi_class'. In this dictionary, 'union_feature_index' are a list of 0-based feature indexes and
# print two info for each selected feature: feature_index reasons_for_selected_features
def print_unified_selected_features_for_multi_class(unified_features, class_names, out_file_id):
    num_unified_features = len(unified_features['union_feature_index'])
    out_file_id.write('#%s\n'%(', '.join(['%d:%s'%(i+1, class_names[i]) for i in range(len(class_names))])))
    for i in range(num_unified_features):
        f = unified_features['union_feature_index'][i] + 1
        types = ','.join(unified_features['type'][i])
        scores = ','.join(['%.6g'%score for score in unified_features['score'][i]])
        out_file_id.write('%d\t%s\t%s\n'%(f,types,scores))

def print_selected_features_with_scores_for_two_class(out_file_id, features, scores, topK=10):
    topK = min(len(features), topK)
    for rank in range(topK):
        out_file_id.write('%d\t%.6g\t%d\n'%(features[rank], scores[rank], rank+1))

# 'unified_features' comes from function's output 'skfeature_selection_func_for_multi_classes_original'. In this dictionary, 'feature_index' are a list of 0-based feature indexes and
# print two info for each selected feature: feature_index reasons_for_selected_features
def print_unified_selected_features_original(unified_features, out_file_id):
    num_unified_features = len(unified_features['union_feature_index'])
    for i in range(num_unified_features):
        f = unified_features['union_feature_index'][i]
        type = unified_features['type'][i]
        out_file_id.write('%d\t%s\n'%(f,type))

# 'unified_features' comes from function's output 'skfeature_selection_func_for_multi_classes_original'
# 'unified_features['original']['feature_index']' contains the duplicated features that come from different comparisons, such as OvR or OvO.
# print three info for each selected feature: feature_index score reasons_for_selected_features
# For each reason type, features are ranked from the most to least important.
def print_original_selected_features_original(unified_features, out_file_id):
    num_selected_features = len(unified_features['original']['feature_index'])
    for i in range(num_selected_features):
        f = unified_features['original']['feature_index'][i]
        type = unified_features['original']['type'][i]
        score = unified_features['original']['score'][i]
        out_file_id.write('%d\t%.6g\t%s\n'%(f,score,type))


# file format (tsv):
# Line 1: N	CC	LC	LG	LV	ST
# Line 2 (fold1): sample1,...,sample9	sample10,...,sample19	sample20,...,sample29	sample30,...,sample39	sample40,...,sample49	sample50,...,sample59
# Line 3 (fold2): ...
# ...
#
# 'index_of_fold_for_test': 1-based integer
def read_samples_folds_file(index_of_fold_for_test, samples_fold_file, exclude_normal_samples=True):
    normal_class = 'N'
    samples_test = []
    with open(samples_fold_file, 'r') as f:
        class_names = f.readline().rstrip().split('\t')  # header line is a list of class names
        if exclude_normal_samples:
            index_of_normal_class = class_names.index(normal_class)
            class_names.remove(normal_class)
        samples_train_classes = dict.fromkeys(class_names)
        for c in class_names: samples_train_classes[c] = []
        iFold = 0
        for line in f:
            iFold += 1
            items = line[:-1].split(
                '\t')  # remove only the last char (new line), but 'rstrip()' removes the last empty columns and cause the bugs
            if exclude_normal_samples:
                items.pop(index_of_normal_class)  # remove samples of normal class from items
            if iFold == index_of_fold_for_test:
                for item in items:
                    if len(item) > 0:  # must add this "if", because when item is an empty string, item.split(',') will return [''] and add it 'samples_test'
                        samples_test += item.split(',')
            else:
                for i in range(len(class_names)):
                    if i >= len(items):
                        next
                    else:
                        if len(items[i]) > 0:  # must add this "if", because when item is an empty string, item.split(',') will return [''] and add it 'samples_train'
                            samples_train_classes[class_names[i]] += items[i].split(',')
    samples_train = []
    for c in class_names:
        samples_train += samples_train_classes[c]
    samples_train = list(filter(None, samples_train))
    samples_test = list(filter(None, samples_test))

    return (samples_test, samples_train, iFold)


###########
# Implement marker (with dynamic alpha thresholds) filtering by the combined alpha-value-distribution (sample_set_frequency) of noncancer control samples
###########

# alphavalue2samplereadfractiondict: a dictionary alpha -> {sample_index:read_fraction}. For example, {'0.5': {'14':0.7, '7':0.6}, '0.75': {'6':0.3}}.
# ordered_alpha_values_list: a list of alpha values (strings) in increasing or decreasing order. For example, ['1', '0.95', '0.91', '0.86']
# min_read_fraction: minimum read fraction. This applies to the accumulated read fraction for a sample
def get_sampleset_with_accumulated_values(alphavalue2samplereadfractiondict, ordered_alpha_values_list, min_read_fraction):
    alpha2cumdict = {a: dict() for a in ordered_alpha_values_list}  # a dictionary alpha -> {'sample_index':accumualted_read_count_from_all_qualified_alpha_values}
    n_alpha = len(ordered_alpha_values_list)
    for i in range(n_alpha):
        a = ordered_alpha_values_list[i]
        if i == 0:
            alpha2cumdict[a] = alphavalue2samplereadfractiondict[a] if a in alphavalue2samplereadfractiondict else dict()
        else:
            a_prev = ordered_alpha_values_list[i - 1]
            alpha2cumdict[a] = mergeDict_by_adding_values_of_common_keys(alphavalue2samplereadfractiondict[a], alpha2cumdict[a_prev]) if a in alphavalue2samplereadfractiondict else alpha2cumdict[a_prev]
    samples_union_set = set([])
    for a in ordered_alpha_values_list:
        # >=min_read_fraction matches the rule used by marker discovery function 'identify_turning_point_of_two_alpha2sampleindexsetandreadfraction()'
        samples_union_set.update([sample for sample in alpha2cumdict[a].keys() if alpha2cumdict[a][sample] >= min_read_fraction])
    return(samples_union_set)


# Input:
#   alpha_distribution_of_one_marker: a dictionary {alpha_value_string:set_of_sample_indexes_strings, ...}. For example: {'0.5': {'14', '7'}, '0.75': {'6'}}. Or when 'minreadfrac' in alpha_distribution_type, a dictionary {alpha_value_string:set_of_sample_indexes_strings, ...}. For example: {'0.5': {'14':0.3, '7':0.2}, '0.75': {'6':0.1}}
#   alpha_distribution_type: 'samplesetfreq'
#   alpha_threshold: a float number between 0 and 1
#   compare_type: '<', '>', '<=', '>='
def get_sample_freq_of_alpha_distribution_by_alpha_threshold(alpha_distribution_of_one_marker, alpha_distribution_type, alpha_threshold, alpha_compare_type):
    if 'samplesetfreq' in alpha_distribution_type:
        # frequency based on the union of sample sets of each alpha-value
        # sorting a list of float strings actually is equivalent to sorting a list of the same float numbers. For example,
        # sorted(['0.08', '0', '1', '0.0417', '0.04', '0.24'] reverse=True) is ['1', '0.24', '0.08', '0.0417', '0.04', '0']
        # sorted([0.08, 0, 1, 0.0417, 0.04, 0.24] reverse=True) is also [1, 0.24, 0.08, 0.0417, 0.04, 0]
        # So for saving time, we directly sort the list of float strings, instead of sorting the list of float numbers.
        unique_alpha_values = list(alpha_distribution_of_one_marker.keys())
        if 'minreadfrac' in alpha_distribution_type:
            min_read_fraction = abs(float(extract_number_after_a_substring(alpha_distribution_type, 'minreadfrac')))  # extract '0.05' from 'samplesetfreq.minreadfrac0.1'
            if alpha_compare_type == '>':
                qualified_sorted_alpha_values = []
                for a in sorted(unique_alpha_values, reverse=True):
                    if float(a)>alpha_threshold:
                        qualified_sorted_alpha_values.append(a)
                    else:
                        break
                samples_union_set = get_sampleset_with_accumulated_values(alpha_distribution_of_one_marker, qualified_sorted_alpha_values, min_read_fraction)
            elif alpha_compare_type == '>=':
                qualified_sorted_alpha_values = []
                for a in sorted(unique_alpha_values, reverse=True):
                    if float(a) >= alpha_threshold:
                        qualified_sorted_alpha_values.append(a)
                    else:
                        break
                samples_union_set = get_sampleset_with_accumulated_values(alpha_distribution_of_one_marker, qualified_sorted_alpha_values, min_read_fraction)
            elif alpha_compare_type == '<':
                qualified_sorted_alpha_values = []
                for a in sorted(unique_alpha_values):
                    if float(a) < alpha_threshold:
                        qualified_sorted_alpha_values.append(a)
                    else:
                        break
                samples_union_set = get_sampleset_with_accumulated_values(alpha_distribution_of_one_marker, qualified_sorted_alpha_values, min_read_fraction)
            elif alpha_compare_type == '<=':
                qualified_sorted_alpha_values = []
                for a in sorted(unique_alpha_values):
                    if float(a) <= alpha_threshold:
                        qualified_sorted_alpha_values.append(a)
                    else:
                        break
                samples_union_set = get_sampleset_with_accumulated_values(alpha_distribution_of_one_marker, qualified_sorted_alpha_values, min_read_fraction)
            else:
                sys.stderr.write('Error in function get_sample_freq_of_alpha_distribution_by_alpha_threshold:\n  the argument alpha_compare_type (%s) is incorrect!\nExit.\n' % alpha_compare_type)
                sys.stderr.exit(-1)
            return(len(samples_union_set))
        else:
            if alpha_compare_type == '>':
                samples_union_set = set([])
                unique_alpha_values = sorted(unique_alpha_values, reverse=True)
                for a in unique_alpha_values:
                    if float(a) > alpha_threshold:
                        samples_union_set.update(alpha_distribution_of_one_marker[a])
                    else:
                        break
            elif alpha_compare_type == '>=':
                samples_union_set = set([])
                unique_alpha_values = sorted(unique_alpha_values, reverse=True)
                for a in unique_alpha_values:
                    if float(a) >= alpha_threshold:
                        samples_union_set.update(alpha_distribution_of_one_marker[a])
                    else:
                        break
            elif alpha_compare_type == '<':
                samples_union_set = set([])
                unique_alpha_values = sorted(unique_alpha_values)
                for a in unique_alpha_values:
                    if float(a) < alpha_threshold:
                        samples_union_set.update(alpha_distribution_of_one_marker[a])
                    else:
                        break
            elif alpha_compare_type == '<=':
                samples_union_set = set([])
                unique_alpha_values = sorted(unique_alpha_values)
                for a in unique_alpha_values:
                    if float(a) <= alpha_threshold:
                        samples_union_set.update(alpha_distribution_of_one_marker[a])
                    else:
                        break
            else:
                sys.stderr.write('Error in function get_sample_freq_of_alpha_distribution_by_alpha_threshold:\n  the argument alpha_compare_type (%s) is incorrect!\nExit.\n' % alpha_compare_type)
                sys.stderr.exit(-1)
            return(len(samples_union_set))
    else:
        sys.stderr.write('Error in function get_sample_freq_of_alpha_distribution_by_alpha_threshold:\n  the argument alpha_distribution_type (%s) is incorrect!\nExit.\n'%alpha_distribution_type)
        sys.stderr.exit(-1)


# Input:
#   in_file1_background and in_file2_cancer: file format is from function 'write_combined_meth_string_histgram'
#   column_index_of_alpha_threshold_in_file1: index is 1-based
def filter_markers_of_alpha_thresholds_by_control_samples_with_combined_alpha_distribution(method, column_index_of_alpha_threshold_in_file1, in_file1_markers_with_alpha_thresholds, in_file2_filter_of_control_samples_alpha_values_distr, out_markers_file, input_filters_info=({},{},{})):
    if 'alpha.samplesetfreq.thresholds' in method:
        if 'readfrac' in method:  # 'hyper.alpha.samplesetfreq.thresholds.n2.p5.minreadfrac+0.1-0.01_controlfilter.n6.minreadfrac-0.05'
            ### Use sample_index_set as the sample frequency and define a sample having tumor-sginal by using min tumor read fraction
            marker_method, filter_method = method.split('_') # marker_method='hyper.alpha.samplesetfreq.thresholds.n2.p5.minreadfrac+0.1-0.01', filter_method='controlfilter.n6.minreadfrac-0.05'
            marker_type = marker_method.split('.')[0]
            if 'nn' in filter_method:
                max_freq_frac_of_neg = float(extract_number_after_a_substring(filter_method, 'controlfilter.nn')) # extract '0.2' from 'controlfilter.nn0.2.minreadfrac-0.05', where 'nn0.2' means the sample fraction, not sample frequency (i.e., max_freq_cumsum_of_neg)
            else:
                max_freq_cumsum_of_neg = int(extract_number_after_a_substring(filter_method, 'controlfilter.n')) # extract '3' from 'controlfilter.n3.minreadfrac-0.05'
            min_read_fraction_neg = abs(float(extract_number_after_a_substring(filter_method, 'minreadfrac'))) # extract '0.05' from 'controlfilter.n6.minreadfrac-0.05'

            if len(input_filters_info[0])==0:
                a2_background = {}
                marker2max_cpg_num_2 = {}
                marker2sample_num_2 = {}
                print('begin to load filtering file of control samples ...')
                print('   start: ', datetime.now(), flush=True)
                load_one_alpha_value_distribution_file_that_has_sample_index_sets_and_read_fractions(
                    in_file2_filter_of_control_samples_alpha_values_distr, a2_background, marker2max_cpg_num_2, marker2sample_num_2)
                print('   end: ', datetime.now(), flush=True)
            else:
                print('skip loading filtering file of control samples, since it is already loaded')
                a2_background, marker2max_cpg_num_2, marker2sample_num_2 = input_filters_info
            column_index_of_alpha_threshold_in_file1 -= 1  # convert 1-based index to 0-based.
            print('begin to filter markers ...', flush=True)
            print('   start: ', datetime.now(), flush=True)
            if in_file1_markers_with_alpha_thresholds.endswith('gz'):
                fid_markers = gzip.open(in_file1_markers_with_alpha_thresholds, 'rt')
            else:
                fid_markers = open(in_file1_markers_with_alpha_thresholds, 'rt')
            if out_markers_file.endswith('gz'):
                fout = gzip.open(out_markers_file, 'wt')
            else:
                fout = open(out_markers_file, 'wt')
            header_line_of_markers = next(fid_markers)
            fout.write( header_line_of_markers )
            for line in fid_markers:
                items = line.rstrip().split('\t')
                marker = int(items[0])
                alpha_threshold = float(items[column_index_of_alpha_threshold_in_file1])
                try:
                    if 'nn' in filter_method:
                        max_freq_cumsum_of_neg = max_freq_frac_of_neg * marker2sample_num_2[marker]
                    if 'hyper' in method:
                        sample_freq = get_sample_freq_of_alpha_distribution_by_alpha_threshold(a2_background[marker], 'samplesetfreq.minreadfrac%g'%min_read_fraction_neg, alpha_threshold, '>')
                    elif 'hypo' in method:
                        sample_freq = get_sample_freq_of_alpha_distribution_by_alpha_threshold(a2_background[marker], 'samplesetfreq.minreadfrac%g'%min_read_fraction_neg, alpha_threshold, '<')
                    if sample_freq <= max_freq_cumsum_of_neg: # refer to function 'identify_turning_point_of_two_alpha2sampleindexset' for why we use <=, not <
                        fout.write(line)
                except KeyError:
                    pass
            fid_markers.close()
            fout.close()
            print('   end: ', datetime.now(), flush=True)
        else:
            ### Simply use sample_index_set as the sample frequency
            # a1_background: a dictionary {'marker_index':{'alpha_value':sample_index_set}}
            a2_background = {}
            marker2max_cpg_num_2 = {}
            load_one_alpha_value_distribution_file_that_has_sample_index_sets(in_file2_filter_of_control_samples_alpha_values_distr, a2_background, marker2max_cpg_num_2)
            print('begin to filter markers ...', flush=True)
            print('   start: ', datetime.now(), flush=True)
            column_index_of_alpha_threshold_in_file1 -= 1 # convert 1-based index to 0-based.
            # 'hyper.alpha.samplesetfreq.thresholds.n2': hyper-methylation markers with alpha's frequency on negative class <2. Similar to 'hypo.alpha.samplesetfreq.thresholds.n2'.
            marker_type, _, _, _, max_freq_cumsum_of_neg_str = method.split('.')
            max_freq_cumsum_of_neg = int(max_freq_cumsum_of_neg_str[1:])
            if in_file1_markers_with_alpha_thresholds.endswith('gz'):
                fid_markers = gzip.open(in_file1_markers_with_alpha_thresholds, 'rt')
            else:
                fid_markers = open(in_file1_markers_with_alpha_thresholds, 'rt')
            if out_markers_file.endswith('gz'):
                fout = gzip.open(out_markers_file, 'wt')
            else:
                fout = open(out_markers_file, 'wt')
            header_line_of_markers = next(fid_markers)
            fout.write( header_line_of_markers )
            for line in fid_markers:
                items = line.rstrip().split('\t')
                marker = int(items[0])
                alpha_threshold = float(items[column_index_of_alpha_threshold_in_file1])
                try:
                    if 'hyper' in method:
                        sample_freq = get_sample_freq_of_alpha_distribution_by_alpha_threshold(a2_background[marker], 'samplesetfreq', alpha_threshold, '>')
                    elif 'hypo' in method:
                        sample_freq = get_sample_freq_of_alpha_distribution_by_alpha_threshold(a2_background[marker], 'samplesetfreq', alpha_threshold, '<')
                    if sample_freq <= max_freq_cumsum_of_neg: # refer to function 'identify_turning_point_of_two_alpha2sampleindexset' for why we use <=, not <
                        fout.write(line)
                except KeyError:
                    pass
            fid_markers.close()
            fout.close()
            print('   end: ', datetime.now(), flush=True)
    return((a2_background, marker2max_cpg_num_2, marker2sample_num_2))


#################################
# Dynamic-alpha-threshold-based marker selection method for Cancer Detection
#################################

# alpha_union_list: the list of alpha values, which are not only union of alpha values of pos_class and neg_class, but also with the order of alpha values, with respect to 'hyper' (in decreasing order) or 'hypo' (in increasing order).
# alpha2sampleindexset_of_neg_class, alpha2sampleindexset_of_pos_class: two dictionaries {'alpha_value':{sample_index}}
# Algorithm:
#    Step 1: accumulate sample_index_set of each alpha_value, by the alpha_value order of alpha_union_list
#    Step 2: calcualte size of accumulated sample_index_set of each alpha_value
#    Step 3: compute the following to determine the turning point. This step is the same as function 'identify_alpha_threshold_by_alpha2freq_of_pos_and_neg_class'
#            arr_neg_cumsum[index] <= arr_neg_cumsum_threshold
#            arr_pos_cumsum[index] >= arr_pos_cumsum_threshold
#            If there exist multiple index satisfying the above criteria, choose the largest index
#    Step 4: return the index of turning point (i.e., the index of alpha threshold in alpha_union_list)
#
def identify_turning_point_of_three_alpha2sampleindexsetandreadfraction_V1(min_read_fraction_neg_backgroundplasma,
                                                                        min_read_fraction_pos_tumor,
                                                                        min_read_fraction_neg_paired_normaltissue,
                                                                        alpha_union_list,
                                                                        alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma,
                                                                        alpha2sampleindexsetandreadfraction_of_pos_tumors,
                                                                        alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues,
                                                                        max_freq_cumsum_of_neg_backgroundplasma,
                                                                        min_freq_cumsum_of_pos_tumors):
    n_alpha = len(alpha_union_list)
    # implement cumsum using set union operator
    alpha2cumdict_neg_backgroundplasma = {a:dict() for a in alpha_union_list}
    alpha2cumdict_pos_tumors = dict(alpha2cumdict_neg_backgroundplasma)
    alpha2cumdict_neg_paired_normaltissues = dict(alpha2cumdict_neg_backgroundplasma)
    for i in range(n_alpha):
        a = alpha_union_list[i]
        if i == 0:
            alpha2cumdict_neg_backgroundplasma[a] = alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma[a] if a in alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma else dict()
            alpha2cumdict_pos_tumors[a] = alpha2sampleindexsetandreadfraction_of_pos_tumors[a] if a in alpha2sampleindexsetandreadfraction_of_pos_tumors else dict()
            alpha2cumdict_neg_paired_normaltissues[a] = alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues[a] if a in alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues else dict()
        else:
            a_prev = alpha_union_list[i - 1]
            alpha2cumdict_neg_backgroundplasma[a] = mergeDict_by_adding_values_of_common_keys(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma[a], alpha2cumdict_neg_backgroundplasma[a_prev]) if a in alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma else alpha2cumdict_neg_backgroundplasma[a_prev]
            alpha2cumdict_pos_tumors[a] = mergeDict_by_adding_values_of_common_keys(alpha2sampleindexsetandreadfraction_of_pos_tumors[a], alpha2cumdict_pos_tumors[a_prev]) if a in alpha2sampleindexsetandreadfraction_of_pos_tumors else alpha2cumdict_pos_tumors[a_prev]
            alpha2cumdict_neg_paired_normaltissues[a] = mergeDict_by_adding_values_of_common_keys(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues[a], alpha2cumdict_neg_paired_normaltissues[a_prev]) if a in alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues else alpha2cumdict_neg_paired_normaltissues[a_prev]

    arr_neg_cumsum = np.array([sum([True if alpha2cumdict_neg_backgroundplasma[a][sample_index]>=min_read_fraction_neg_backgroundplasma else False for sample_index in alpha2cumdict_neg_backgroundplasma[a]]) for a in alpha_union_list])
    # The following implement the criterion of paired tissues (tumor and its matched normal tissue)
    arr_pos_cumsum = np.array([sum([True if (alpha2cumdict_pos_tumors[a][sample_index]>=min_read_fraction_pos_tumor) and (alpha2cumdict_neg_paired_normaltissues[a][sample_index]<=min_read_fraction_neg_paired_normaltissue) else False for sample_index in list(set(alpha2cumdict_pos_tumors[a].keys() & set(alpha2cumdict_neg_paired_normaltissues[a].keys())))]) for a in alpha_union_list])
    index_list = [i for i in range(n_alpha) if ((arr_pos_cumsum[i]>=min_freq_cumsum_of_pos_tumors) and (arr_neg_cumsum[i] <= max_freq_cumsum_of_neg_backgroundplasma))]
    if len(index_list) == 0:
        # No turning point
        return ((-2, arr_neg_cumsum, arr_pos_cumsum))
    else:
        # if (index_list[-1]+1) < (n_alpha-1):
        if (index_list[-1]) < (n_alpha - 1):
            return ((index_list[-1] + 1, arr_neg_cumsum, arr_pos_cumsum))
        else:
            if len(index_list) >= 2:
                if (index_list[-2]) < (n_alpha - 1):
                    return ((index_list[-2] + 1, arr_neg_cumsum, arr_pos_cumsum))
            # -1 means the index should be the one greater than n
            return ((-1, arr_neg_cumsum, arr_pos_cumsum))



def identify_alpha_threshold_by_alpha2sampleindexsetandreadfraction_of_pos_tumors_neg_paired_normaltissues_and_neg_background_V1(
        unique_alpha_values_of_neg_background,
        sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_neg_background,
        unique_alpha_values_of_pos_tumors,
        sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_pos_tumors,
        unique_alpha_values_of_neg_paired_normaltissues,
        sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_neg_paired_normaltissues,
        max_freq_cumsum_of_neg_backgroundplasma,
        min_freq_cumsum_of_pos_tumors,
        min_read_fraction_neg_backgroundplasma,
        min_read_fraction_pos_tumor,
        min_read_fraction_neg_paired_normaltissue,
        marker_type='hyper'):
    # build the dict for alpha2sampleindexsetandreadfraction for negative background plasma
    alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma = {}
    for i in range(len(unique_alpha_values_of_neg_background)):
        alpha = unique_alpha_values_of_neg_background[i]
        alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma[alpha] = {}
        items = sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_neg_background[i].split('_')
        for item in items:
            sample_index, read_fraction = item.split(':')
            read_fraction = float(read_fraction)
            alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma[alpha][sample_index] = read_fraction
    # build the dict for alpha2sampleindexsetandreadfraction for positive tumors
    alpha2sampleindexsetandreadfraction_of_pos_tumors = {}
    for i in range(len(unique_alpha_values_of_pos_tumors)):
        alpha = unique_alpha_values_of_pos_tumors[i]
        alpha2sampleindexsetandreadfraction_of_pos_tumors[alpha] = {}
        items = sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_pos_tumors[i].split('_')
        for item in items:
            sample_index, read_fraction = item.split(':')
            read_fraction = float(read_fraction)
            alpha2sampleindexsetandreadfraction_of_pos_tumors[alpha][sample_index] = read_fraction
    # build the dict for alpha2sampleindexsetandreadfraction for negative paired normal tissues
    alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues = {}
    for i in range(len(unique_alpha_values_of_neg_paired_normaltissues)):
        alpha = unique_alpha_values_of_neg_paired_normaltissues[i]
        alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues[alpha] = {}
        items = sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_neg_paired_normaltissues[i].split('_')
        for item in items:
            sample_index, read_fraction = item.split(':')
            read_fraction = float(read_fraction)
            alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues[alpha][sample_index] = read_fraction

    # Process
    alpha_union_list = []
    if 'hyper' in marker_type:
        if len(alpha2sampleindexsetandreadfraction_of_pos_tumors)>0 and (len(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma)==0 or len(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues)==0):
            alpha_threshold = 0 # any reads with any alpha values should be used. So let alpha_threshold==0 for hyper markers
            return (alpha_threshold, [], [], [])
        if len(alpha2sampleindexsetandreadfraction_of_pos_tumors)>0 and len(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma)>0 and len(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues)>0:
            alpha_union_list = sorted(list(set(list(alpha2sampleindexsetandreadfraction_of_pos_tumors.keys()) + list(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma.keys()) + list(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues.keys()))), reverse=True) # decreasing order
    elif 'hypo' in marker_type:
        if len(alpha2sampleindexsetandreadfraction_of_pos_tumors)>0 and (len(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma)==0 or len(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues)==0):
            alpha_threshold = 1 # any reads with any alpha values should be used. So let alpha_threshold==1 for hypo markers
            return (alpha_threshold, [], [], [])
        if len(alpha2sampleindexsetandreadfraction_of_pos_tumors)>0 and len(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma)>0 and len(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues)>0:
            alpha_union_list = sorted(list(set(list(alpha2sampleindexsetandreadfraction_of_pos_tumors.keys()) + list(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma.keys())+ list(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues.keys()))))  # increasing order
    if len(alpha_union_list)==0:
        sys.stderr.write('Error (identify_alpha_threshold_by_alpha2sampleindexsetandreadfraction_of_pos_tumors_neg_paired_normaltissues_and_neg_background):\n  Alpha union list of three classes are empty!\nExit.\n')
        sys.exit(-1)
    # identify_turning_point_of_three_alpha2sampleindexsetandreadfraction
    alpha_index, a1_freq_cumsum_neg_backgroundplasma, a2_freq_cumsum_pos_tumors = identify_turning_point_of_three_alpha2sampleindexsetandreadfraction_V1(
                                                                            min_read_fraction_neg_backgroundplasma,
                                                                            min_read_fraction_pos_tumor,
                                                                            min_read_fraction_neg_paired_normaltissue,
                                                                            alpha_union_list,
                                                                            alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma,
                                                                            alpha2sampleindexsetandreadfraction_of_pos_tumors,
                                                                            alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues,
                                                                            max_freq_cumsum_of_neg_backgroundplasma,
                                                                            min_freq_cumsum_of_pos_tumors)
    if alpha_index == -2:
        alpha_threshold = None
    else:
        if alpha_index == -1:
            alpha_threshold = None
            # if marker_type=='hyper':
            #     alpha_threshold = 0
            # elif marker_type == 'hypo':
            #     alpha_threshold = 1
        else:
            alpha_threshold = float(alpha_union_list[alpha_index])
            # if len(alpha2freq_of_neg_class) == 1:
            #     # Remove the following case which identifies hyper marker:
            #     # positive class: {'0.857':2, '0.286':1, '0.167':2, '0.143':4, '0':8}
            #     # negative class: {'0.167':2}
            #     # alpha_union_list: ['0.857', '0.286', '0.167', '0.143', '0']
            #     # identify_turning_point_of_two_cumsum_array returns alpha_index=4
            #     # But actually if alpha_threshold=='0', then negative class is incorrect for our purpose.
            #     if (('hyper' in marker_type) and (alpha_threshold<float(alpha2freq_of_neg_class.values[0]))) or (('hypo' in marker_type) and (alpha_threshold>float(alpha2freq_of_neg_class.values[0]))):
            #         alpha_threshold = None
    return ((alpha_threshold, alpha_union_list, a1_freq_cumsum_neg_backgroundplasma, a2_freq_cumsum_pos_tumors))

# Version 1 and Version 1.1
# Take three alpha_value_distribution files as inputs
# All three kinds of samples (normal plasma, tumors, matched/paired normal tissues) use the same dynamic-alpha-threshold
def compare_background_vs_tumor_and_paired_normaltissue_alpha_value_distribution_files_with_memory_saving_way_for_cancer_detection_V1(method,
                                                                                                              in_file1_background,
                                                                                                              in_file2_tumors,
                                                                                                              in_file3_paired_normaltissues):
    # Scan three input files to obtain the markers that appear in all three files
    # get_specific_column_of_tab_file(in_file1_background)
    ret_marker_2_alpha2freq = {}
    if 'samplesetfreq' in method:
        if 'readfrac' in method:
            # 'V1' only: Version 1, 2 parts in 'method'
            # E.g., 'hyper.alpha.samplesetfreq.triple.thresholds.nn0.2.p1.minreadfrac+tumor0.5-tissue0.25-ncplasma0.2'
            part1_method, part2_method = method.split('.minreadfrac')
            min_read_fraction_pos_tumor = abs(float(extract_number_after_a_substring(part2_method, 'tumor')))
            min_read_fraction_neg_normaltissue = abs(float(extract_number_after_a_substring(part2_method, 'tissue')))
            min_read_fraction_neg_backgroundplasma = abs(float(extract_number_after_a_substring(part2_method, 'ncplasma'))) # non-cancer plasma are background
            marker_type = part1_method.split('.')[0]
            if 'nn' in part1_method:
                max_freq_fraction_of_neg_background = float(extract_number_after_a_substring(part1_method, 'nn')) # max sample frequency, extract '0.2' from 'hyper.alpha.samplesetfreq.triple.thresholds.nn0.2.p1'
            else:
                max_freq_cumsum_of_neg_background = int(extract_number_after_a_substring(part1_method, 'n')) # max sample frequency, extract '2' from 'hyper.alpha.samplesetfreq.triple.thresholds.n2.p1'
            min_freq_cumsum_of_pos_tumors = int(extract_number_after_a_substring(part1_method, 'p')) # min sample frequency, extract '1' from 'hyper.alpha.samplesetfreq.triple.thresholds.nn0.2.p1'

            if in_file1_background.endswith('gz'):
                fid1_background = gzip.open(in_file1_background, 'rt')
            else:
                fid1_background = open(in_file1_background, 'rt')
            if in_file2_tumors.endswith('gz'):
                fid2_tumors = gzip.open(in_file2_tumors, 'rt')
            else:
                fid2_tumors = open(in_file2_tumors, 'rt')
            if in_file3_paired_normaltissues.endswith('gz'):
                fid3_paired_normaltissues = gzip.open(in_file3_paired_normaltissues, 'rt')
            else:
                fid3_paired_normaltissues = open(in_file3_paired_normaltissues, 'rt')

            ### begin to process three input files and write output file
            fid1_background.readline().rstrip()  # skip header line
            background_first_marker_line = fid1_background.readline()
            background_items = background_first_marker_line.rstrip().split('\t')
            background_marker_index = int(background_items[0])

            fid3_paired_normaltissues.readline().rstrip()  # skip header line
            paired_normaltissues_first_marker_line = fid3_paired_normaltissues.readline()
            paired_normaltissues_items = paired_normaltissues_first_marker_line.rstrip().split('\t')
            paired_normaltissues_marker_index = int(paired_normaltissues_items[0])

            end_of_background_file = False
            end_of_paired_normaltissues_file = False
            fid2_tumors.readline()  # skip header line
            for tumors_line in fid2_tumors:
                tumors_items = tumors_line.rstrip().split()
                tumors_marker_index = int(tumors_items[0])

                while tumors_marker_index > background_marker_index:
                    background_line = fid1_background.readline()
                    if not background_line:
                        end_of_background_file = True
                        break
                    background_items = background_line.rstrip().split('\t')
                    background_marker_index = int(background_items[0])
                if end_of_background_file:
                    break
                if tumors_marker_index < background_marker_index:
                    continue
                # now we arrive at tumors_marker_index == background_marker_index
                while tumors_marker_index > paired_normaltissues_marker_index:
                    paired_normaltissues_line = fid3_paired_normaltissues.readline()
                    if not paired_normaltissues_line:
                        end_of_paired_normaltissues_file = True
                        break
                    paired_normaltissues_items = paired_normaltissues_line.rstrip().split('\t')
                    paired_normaltissues_marker_index = int(paired_normaltissues_items[0])
                if end_of_paired_normaltissues_file:
                    break
                if tumors_marker_index < paired_normaltissues_marker_index:
                    continue
                # now we arrive at tumors_marker_index == background_marker_index == paired_normaltissues_marker_index

                # now we begin to process for tumors_marker_index == background_marker_index == paired_normaltissues_marker_index
                max_cpg_num = background_items[1]
                background_sample_num = int(background_items[2])
                background_unique_alpha_values = background_items[3].split(',')
                background_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list = background_items[4].split(',')

                tumors_unique_alpha_values = tumors_items[3].split(',')
                tumors_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list = tumors_items[4].split(',')

                paired_normaltissues_unique_alpha_values = paired_normaltissues_items[3].split(',')
                paired_normaltissues_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list = paired_normaltissues_items[4].split(',')

                if 'nn' in part1_method:
                    max_freq_cumsum_of_neg_background = max_freq_fraction_of_neg_background * background_sample_num

                if tumors_marker_index == 299:
                    print('debug %d'%tumors_marker_index)

                alpha_threshold, alpha_union_list, a1_freq_cumsum_background, a2_freq_cumsum_tumors  = identify_alpha_threshold_by_alpha2sampleindexsetandreadfraction_of_pos_tumors_neg_paired_normaltissues_and_neg_background_V1(
                    background_unique_alpha_values,
                    background_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list,
                    tumors_unique_alpha_values,
                    tumors_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list,
                    paired_normaltissues_unique_alpha_values,
                    paired_normaltissues_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list,
                    max_freq_cumsum_of_neg_background,
                    min_freq_cumsum_of_pos_tumors,
                    min_read_fraction_neg_backgroundplasma,
                    min_read_fraction_pos_tumor,
                    min_read_fraction_neg_normaltissue,
                    marker_type)
                if alpha_threshold is not None:
                    if 'hyper' in method:
                        ret_marker_2_alpha2freq[tumors_marker_index] = {'alpha_threshold': alpha_threshold,
                                                                        'max_cpg_num': max_cpg_num,
                                                                        'alpha2freq': filter_by_freq_cumsum_and_create_alpha2freqcumsum(
                                                                            alpha_threshold, alpha_union_list,
                                                                            a2_freq_cumsum_tumors,
                                                                            '>')}
                    elif 'hypo' in method:
                        ret_marker_2_alpha2freq[tumors_marker_index] = {'alpha_threshold': alpha_threshold,
                                                                        'max_cpg_num': max_cpg_num,
                                                                        'alpha2freq': filter_by_freq_cumsum_and_create_alpha2freqcumsum(
                                                                            alpha_threshold, alpha_union_list,
                                                                            a2_freq_cumsum_tumors,
                                                                            '<')}

            fid1_background.close()
            fid2_tumors.close()
            fid3_paired_normaltissues.close()
        return (ret_marker_2_alpha2freq)


# alpha_union_list and alpha_list_allowed_for_pairedtissues: the list of alpha values, which are not only union of alpha values of pos_class and neg_class, but also with the order of alpha values, with respect to 'hyper' (in decreasing order) or 'hypo' (in increasing order).
# alpha2sampleindexset_of_neg_class, alpha2sampleindexset_of_pos_class: two dictionaries {'alpha_value':{sample_index}}
# Algorithm:
#    Step 1: accumulate sample_index_set of each alpha_value, by the alpha_value order of alpha_union_list
#    Step 2: calcualte size of accumulated sample_index_set of each alpha_value
#    Step 3: compute the following to determine the turning point. This step is the same as function 'identify_alpha_threshold_by_alpha2freq_of_pos_and_neg_class'
#            arr_neg_cumsum[index] <= arr_neg_cumsum_threshold
#            arr_pos_cumsum[index] >= arr_pos_cumsum_threshold
#            If there exist multiple index satisfying the above criteria, choose the largest index
#    Step 4: return the index of turning point (i.e., the index of alpha threshold in alpha_union_list)
#
# In Step 3, we perform the following:
#       1. Normal plasma use the fixed alpha threshold
#       2. Tumors & their matched/paired normal tissues use the same dynamic alpha threshold, which must take value in a predefined range
#       3. The paired tissue is counted into sample frequency, if the difference of read fractions of the tumor and its
#          adjacent normal tissue >= threshold
#
# Output: a tuple of three elements
#    Element 1: index of alpha value that is used for alpha threshold, in the list 'alpha_list_allowed_for_pairedtissues'
#    Element 2 & 3: for debug use only
#
# NOTE:
#    1. 'fixed_alpha_threshold_neg_backgroundplasma' and 'alpha_list_allowed_for_pairedtissues' MUST be an element and a subset of 'alpha_union_list', respectively.
#       We assume but NOT check this.
#
def identify_turning_point_of_three_alpha2sampleindexsetandreadfraction_V2(min_read_fraction_neg_backgroundplasma,
                                                                        diff_read_fraction_pairedtissues,
                                                                        alpha_union_list,
                                                                        alpha_list_allowed_for_pairedtissues,
                                                                        fixed_alpha_threshold_neg_backgroundplasma,
                                                                        alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma,
                                                                        alpha2sampleindexsetandreadfraction_of_pos_tumors,
                                                                        alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues,
                                                                        max_freq_cumsum_of_neg_backgroundplasma,
                                                                        min_freq_cumsum_of_pos_tumors):
    n_alpha = len(alpha_union_list)
    # implement cumsum using set union operator
    alpha2cumdict_neg_backgroundplasma = {a:dict() for a in alpha_union_list}
    alpha2cumdict_pos_tumors = dict(alpha2cumdict_neg_backgroundplasma)
    alpha2cumdict_neg_paired_normaltissues = dict(alpha2cumdict_neg_backgroundplasma)
    for i in range(n_alpha):
        a = alpha_union_list[i]
        if i == 0:
            alpha2cumdict_neg_backgroundplasma[a] = alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma[a] if a in alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma else dict()
            alpha2cumdict_pos_tumors[a] = alpha2sampleindexsetandreadfraction_of_pos_tumors[a] if a in alpha2sampleindexsetandreadfraction_of_pos_tumors else dict()
            alpha2cumdict_neg_paired_normaltissues[a] = alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues[a] if a in alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues else dict()
        else:
            a_prev = alpha_union_list[i - 1]
            alpha2cumdict_neg_backgroundplasma[a] = mergeDict_by_adding_values_of_common_keys(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma[a], alpha2cumdict_neg_backgroundplasma[a_prev]) if a in alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma else alpha2cumdict_neg_backgroundplasma[a_prev]
            alpha2cumdict_pos_tumors[a] = mergeDict_by_adding_values_of_common_keys(alpha2sampleindexsetandreadfraction_of_pos_tumors[a], alpha2cumdict_pos_tumors[a_prev]) if a in alpha2sampleindexsetandreadfraction_of_pos_tumors else alpha2cumdict_pos_tumors[a_prev]
            alpha2cumdict_neg_paired_normaltissues[a] = mergeDict_by_adding_values_of_common_keys(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues[a], alpha2cumdict_neg_paired_normaltissues[a_prev]) if a in alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues else alpha2cumdict_neg_paired_normaltissues[a_prev]

    # Check if negative class meet the criterion
    neg_cumsum_for_fixed_threshold = sum([True if alpha2cumdict_neg_backgroundplasma[fixed_alpha_threshold_neg_backgroundplasma][sample_index]>=min_read_fraction_neg_backgroundplasma else False for sample_index in alpha2cumdict_neg_backgroundplasma[fixed_alpha_threshold_neg_backgroundplasma]])
    if neg_cumsum_for_fixed_threshold > max_freq_cumsum_of_neg_backgroundplasma:
        # No turning point, because negative class does not meet the criterion.
        return((-2, neg_cumsum_for_fixed_threshold, None))

    # Check positive class: the following implement the criterion of paired tissues (tumor and its matched normal tissue)
    # Fill in alpha-accumulated read fractions for the missing samples in 'alpha2cumdict_pos_tumors[alpha]' & 'alpha2cumdict_neg_paired_normaltissues[alpha]'
    for a in alpha_union_list:
        for s in list(set(alpha2cumdict_pos_tumors[a].keys() | set(alpha2cumdict_neg_paired_normaltissues[a].keys()))):
            if s not in alpha2cumdict_pos_tumors[a]:
                alpha2cumdict_pos_tumors[a][s] = 0
            if s not in alpha2cumdict_neg_paired_normaltissues[a]:
                alpha2cumdict_neg_paired_normaltissues[a][s] = 0
    n_alpha_allowed_for_pairedtissues = len(alpha_list_allowed_for_pairedtissues)
    arr_pos_cumsum = np.array([sum([True if ((alpha2cumdict_pos_tumors[a][sample_index]-alpha2cumdict_neg_paired_normaltissues[a][sample_index])>=diff_read_fraction_pairedtissues) else False for sample_index in list(set(alpha2cumdict_pos_tumors[a].keys() & set(alpha2cumdict_neg_paired_normaltissues[a].keys())))]) for a in alpha_list_allowed_for_pairedtissues])
    index_list = [i for i in range(n_alpha_allowed_for_pairedtissues) if (arr_pos_cumsum[i]>=min_freq_cumsum_of_pos_tumors)]

    if len(index_list) == 0:
        # No turning point
        return ((-2, neg_cumsum_for_fixed_threshold, arr_pos_cumsum))
    else:
        # if (index_list[-1]+1) < (n_alpha-1):
        if (index_list[-1]) < (n_alpha_allowed_for_pairedtissues - 1):
            return ((index_list[-1] + 1, neg_cumsum_for_fixed_threshold, arr_pos_cumsum))
        elif (index_list[-1]) == (n_alpha_allowed_for_pairedtissues - 1):
            # -1 means the index should be the one greater than n_alpha_allowed_for_pairedtissues
            return ((-1, neg_cumsum_for_fixed_threshold, arr_pos_cumsum))
        else:
            if len(index_list) >= 2:
                if (index_list[-2]) < (n_alpha_allowed_for_pairedtissues - 1):
                    return ((index_list[-2] + 1, neg_cumsum_for_fixed_threshold, arr_pos_cumsum))
            # -1 means the index should be the one greater than n_alpha_allowed_for_pairedtissues
            return ((-1, neg_cumsum_for_fixed_threshold, arr_pos_cumsum))



def identify_alpha_threshold_by_alpha2sampleindexsetandreadfraction_of_pos_tumors_neg_paired_normaltissues_and_neg_background_V2(
        unique_alpha_values_of_neg_background,
        sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_neg_background,
        unique_alpha_values_of_pos_tumors,
        sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_pos_tumors,
        unique_alpha_values_of_neg_paired_normaltissues,
        sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_neg_paired_normaltissues,
        max_freq_cumsum_of_neg_backgroundplasma,
        min_freq_cumsum_of_pos_tumors,
        min_read_fraction_neg_backgroundplasma,
        diff_read_fraction_pairedtissues,
        fixed_alpha_threshold_neg_backgroundplasma,
        alpha_threshold_range_pairedtissues,
        marker_type='hyper'):
    # build the dict for alpha2sampleindexsetandreadfraction for negative background plasma
    alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma = {}
    for i in range(len(unique_alpha_values_of_neg_background)):
        alpha = unique_alpha_values_of_neg_background[i]
        alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma[alpha] = {}
        items = sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_neg_background[i].split('_')
        for item in items:
            sample_index, read_fraction = item.split(':')
            read_fraction = float(read_fraction)
            alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma[alpha][sample_index] = read_fraction
    # build the dict for alpha2sampleindexsetandreadfraction for positive tumors
    alpha2sampleindexsetandreadfraction_of_pos_tumors = {}
    for i in range(len(unique_alpha_values_of_pos_tumors)):
        alpha = unique_alpha_values_of_pos_tumors[i]
        alpha2sampleindexsetandreadfraction_of_pos_tumors[alpha] = {}
        items = sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_pos_tumors[i].split('_')
        for item in items:
            sample_index, read_fraction = item.split(':')
            read_fraction = float(read_fraction)
            alpha2sampleindexsetandreadfraction_of_pos_tumors[alpha][sample_index] = read_fraction
    # build the dict for alpha2sampleindexsetandreadfraction for negative paired normal tissues
    alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues = {}
    for i in range(len(unique_alpha_values_of_neg_paired_normaltissues)):
        alpha = unique_alpha_values_of_neg_paired_normaltissues[i]
        alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues[alpha] = {}
        items = sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_neg_paired_normaltissues[i].split('_')
        for item in items:
            sample_index, read_fraction = item.split(':')
            read_fraction = float(read_fraction)
            alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues[alpha][sample_index] = read_fraction

    # Process
    alpha_union_list = []
    alpha_list_allowed_for_pairedtissues = []
    if 'hyper' in marker_type:
        if len(alpha2sampleindexsetandreadfraction_of_pos_tumors)>0 and (len(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma)==0 or len(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues)==0):
            alpha_threshold = 0 # any reads with any alpha values should be used. So let alpha_threshold==0 for hyper markers
            return (alpha_threshold, [], [], [])
        if len(alpha2sampleindexsetandreadfraction_of_pos_tumors)>0 and len(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma)>0 and len(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues)>0:
            alpha_union_list = sorted(list(set(list(alpha2sampleindexsetandreadfraction_of_pos_tumors.keys()) + list(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma.keys()) + list(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues.keys()))), reverse=True) # decreasing order
            if fixed_alpha_threshold_neg_backgroundplasma not in alpha_union_list:
                alpha_union_list += [fixed_alpha_threshold_neg_backgroundplasma]
                alpha_union_list = sorted(alpha_union_list, reverse=True) # decreasing order
            # Make 'alpha_list_allowed_for_pairedtissues' (in decreasing order), a subset of 'alpha_union_list'
            lower_bound, upper_bound = alpha_threshold_range_pairedtissues
            for a in alpha_union_list:
                if a<=upper_bound and a>=lower_bound:
                    alpha_list_allowed_for_pairedtissues.append(a)

    elif 'hypo' in marker_type:
        if len(alpha2sampleindexsetandreadfraction_of_pos_tumors)>0 and (len(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma)==0 or len(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues)==0):
            alpha_threshold = 1 # any reads with any alpha values should be used. So let alpha_threshold==1 for hypo markers
            return (alpha_threshold, [], [], [])
        if len(alpha2sampleindexsetandreadfraction_of_pos_tumors)>0 and len(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma)>0 and len(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues)>0:
            alpha_union_list = sorted(list(set(list(alpha2sampleindexsetandreadfraction_of_pos_tumors.keys()) + list(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma.keys())+ list(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues.keys()))))  # increasing order
            if fixed_alpha_threshold_neg_backgroundplasma not in alpha_union_list:
                alpha_union_list += [fixed_alpha_threshold_neg_backgroundplasma]
                alpha_union_list = sorted(alpha_union_list) # increasing order
            # Make 'alpha_list_allowed_for_pairedtissues' (in increasing order), a subset of 'alpha_union_list'
            lower_bound, upper_bound = alpha_threshold_range_pairedtissues
            for a in alpha_union_list:
                if a <= upper_bound and a >= lower_bound:
                    alpha_list_allowed_for_pairedtissues.append(a)

    if len(alpha_union_list)==0:
        sys.stderr.write('Error (identify_alpha_threshold_by_alpha2sampleindexsetandreadfraction_of_pos_tumors_neg_paired_normaltissues_and_neg_background):\n  Alpha union list of three classes are empty!\nExit.\n')
        sys.exit(-1)
    # identify_turning_point_of_three_alpha2sampleindexsetandreadfraction
    alpha_index, a1_freq_cumsum_neg_backgroundplasma, a2_freq_cumsum_pos_tumors = identify_turning_point_of_three_alpha2sampleindexsetandreadfraction_V2(
                                                                            min_read_fraction_neg_backgroundplasma,
                                                                            diff_read_fraction_pairedtissues,
                                                                            alpha_union_list,
                                                                            alpha_list_allowed_for_pairedtissues,
                                                                            fixed_alpha_threshold_neg_backgroundplasma,
                                                                            alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma,
                                                                            alpha2sampleindexsetandreadfraction_of_pos_tumors,
                                                                            alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues,
                                                                            max_freq_cumsum_of_neg_backgroundplasma,
                                                                            min_freq_cumsum_of_pos_tumors)
    if alpha_index == -2:
        alpha_threshold = None
    else:
        if alpha_index == -1:
            # -1 means the index should be the one greater than n_alpha_allowed_for_pairedtissues
            index_last = alpha_union_list.index(alpha_list_allowed_for_pairedtissues[-1])
            if index_last < (len(alpha_union_list)-1):
                alpha_threshold = float(alpha_union_list[index_last + 1])
            else:
                alpha_threshold = None
            # if marker_type=='hyper':
            #     alpha_threshold = 0
            # elif marker_type == 'hypo':
            #     alpha_threshold = 1
        else:
            alpha_threshold = float(alpha_union_list[alpha_index])
            # if len(alpha2freq_of_neg_class) == 1:
            #     # Remove the following case which identifies hyper marker:
            #     # positive class: {'0.857':2, '0.286':1, '0.167':2, '0.143':4, '0':8}
            #     # negative class: {'0.167':2}
            #     # alpha_union_list: ['0.857', '0.286', '0.167', '0.143', '0']
            #     # identify_turning_point_of_two_cumsum_array returns alpha_index=4
            #     # But actually if alpha_threshold=='0', then negative class is incorrect for our purpose.
            #     if (('hyper' in marker_type) and (alpha_threshold<float(alpha2freq_of_neg_class.values[0]))) or (('hypo' in marker_type) and (alpha_threshold>float(alpha2freq_of_neg_class.values[0]))):
            #         alpha_threshold = None
    return ((alpha_threshold, alpha_union_list, a1_freq_cumsum_neg_backgroundplasma, a2_freq_cumsum_pos_tumors))


# Version 2
# Take three alpha_value_distribution files as inputs
# All three kinds of samples (normal plasma, tumors, matched normal tissues) use two different alpha-thresholds
# 1. Normal plasma use the fixed alpha threshold 'a_n' or 'an'
# 2. Tumors & their matched/paired normal tissues use the same dynamic alpha threshold 'a_t' or 'at' (could be
#    different from 'a_n'), which must take value in a predefined range
# 3. The paired tissue is counted into sample frequency, if the difference of read fractions of the tumor and its
#    adjacent normal tissue >= threshold
#
# Explanation of the parameters in the following method ID: 3 parts that are separated by ','
#   For the example ID: 'hyper.alpha.samplesetfreq.triple_V2.thresholds.nn0.2.p1,an0.2.atrange0.8_1,readfrac+pairedtissuediff0.5-ncplasma0.2'
#   Part1:
#     'nn0.2': sample frequency of negative class (i.e., non-cancer plasma that have tumor signals) <= 20% of all samples in negative class
#     'p1': sample frequency of positive class (i.e., tissue pairs that have tumor signals) >= 1
#   Part2:
#     'an0.2': the fixed Alpha threshold of Normal plasma is 0.2
#     'atrange0.8_1': the dynamic Alpha threshold of Tissues (applied to tumors & adjacent normal tissues) is in range [0.8, 1]
#   Part3:
#     'readfrac+pairedtissuediff0.5': for positive class, difference btw fractions of reads with tumor signals for tumor & matched normal tissue >= threshold 0.5
#     '-ncplasma0.2': for negative class, "Non-Cancer plasma" has the fraction of reads with tumor signal <= 0.2
#
def compare_background_vs_tumor_and_paired_normaltissue_alpha_value_distribution_files_with_memory_saving_way_for_cancer_detection_V2(method,
                                                                                                              in_file1_background,
                                                                                                              in_file2_tumors,
                                                                                                              in_file3_paired_normaltissues):
    # Scan three input files to obtain the markers that appear in all three files
    # get_specific_column_of_tab_file(in_file1_background)
    ret_marker_2_alpha2freq = {}
    if 'samplesetfreq' in method:
        if 'readfrac' in method: # 'hyper.alpha.samplesetfreq.triple_V2.thresholds.nn0.2.p1,an0.2.atrange0.8_1,readfrac+pairedtissuediff0.5-ncplasma0.2'
            part1_method, part2_method, part3_method = method.split(',')
            marker_type = part1_method.split('.')[0]
            if 'nn' in part1_method:
                max_freq_fraction_of_neg_background = float(extract_number_after_a_substring(part1_method,'nn'))  # max sample frequency, extract '0.2' from 'hyper.alpha.samplesetfreq.triple.thresholds.nn0.2.p1'
            else:
                max_freq_cumsum_of_neg_background = int(extract_number_after_a_substring(part1_method,'n'))  # max sample frequency, extract '2' from 'hyper.alpha.samplesetfreq.triple.thresholds.n2.p1'
            min_freq_cumsum_of_pos_tumors = int(extract_number_after_a_substring(part1_method,'p'))  # min sample frequency, extract '1' from 'hyper.alpha.samplesetfreq.triple.thresholds.nn0.2.p1'
            fixed_alpha_threshold_neg_backgroundplasma = extract_number_after_a_substring(part2_method, 'an')
            alpha_threshold_range_pairedtissues = extract_range_after_a_substring(part2_method, 'atrange')
            diff_read_fraction_pairedtissues = abs(float(extract_number_after_a_substring(part3_method, 'pairedtissuediff'))) # paired tissues are tumor & its matched normal tissue
            min_read_fraction_neg_backgroundplasma = abs(float(extract_number_after_a_substring(part3_method, 'ncplasma'))) # non-cancer plasma are background

            if in_file1_background.endswith('gz'):
                fid1_background = gzip.open(in_file1_background, 'rt')
            else:
                fid1_background = open(in_file1_background, 'rt')
            if in_file2_tumors.endswith('gz'):
                fid2_tumors = gzip.open(in_file2_tumors, 'rt')
            else:
                fid2_tumors = open(in_file2_tumors, 'rt')
            if in_file3_paired_normaltissues.endswith('gz'):
                fid3_paired_normaltissues = gzip.open(in_file3_paired_normaltissues, 'rt')
            else:
                fid3_paired_normaltissues = open(in_file3_paired_normaltissues, 'rt')

            ### begin to process three input files and write output file
            fid1_background.readline().rstrip()  # skip header line
            background_first_marker_line = fid1_background.readline()
            background_items = background_first_marker_line.rstrip().split('\t')
            background_marker_index = int(background_items[0])

            fid3_paired_normaltissues.readline().rstrip()  # skip header line
            paired_normaltissues_first_marker_line = fid3_paired_normaltissues.readline()
            paired_normaltissues_items = paired_normaltissues_first_marker_line.rstrip().split('\t')
            paired_normaltissues_marker_index = int(paired_normaltissues_items[0])

            end_of_background_file = False
            end_of_paired_normaltissues_file = False
            fid2_tumors.readline()  # skip header line
            for tumors_line in fid2_tumors:
                tumors_items = tumors_line.rstrip().split()
                tumors_marker_index = int(tumors_items[0])

                while tumors_marker_index > background_marker_index:
                    background_line = fid1_background.readline()
                    if not background_line:
                        end_of_background_file = True
                        break
                    background_items = background_line.rstrip().split('\t')
                    background_marker_index = int(background_items[0])
                if end_of_background_file:
                    break
                if tumors_marker_index < background_marker_index:
                    continue
                # now we arrive at tumors_marker_index == background_marker_index
                while tumors_marker_index > paired_normaltissues_marker_index:
                    paired_normaltissues_line = fid3_paired_normaltissues.readline()
                    if not paired_normaltissues_line:
                        end_of_paired_normaltissues_file = True
                        break
                    paired_normaltissues_items = paired_normaltissues_line.rstrip().split('\t')
                    paired_normaltissues_marker_index = int(paired_normaltissues_items[0])
                if end_of_paired_normaltissues_file:
                    break
                if tumors_marker_index < paired_normaltissues_marker_index:
                    continue
                # now we arrive at tumors_marker_index == background_marker_index == paired_normaltissues_marker_index

                # now we begin to process for tumors_marker_index == background_marker_index == paired_normaltissues_marker_index
                max_cpg_num = background_items[1]
                background_sample_num = int(background_items[2])
                background_unique_alpha_values = background_items[3].split(',')
                background_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list = background_items[4].split(',')

                tumors_unique_alpha_values = tumors_items[3].split(',')
                tumors_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list = tumors_items[4].split(',')

                paired_normaltissues_unique_alpha_values = paired_normaltissues_items[3].split(',')
                paired_normaltissues_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list = paired_normaltissues_items[4].split(',')

                if 'nn' in part1_method:
                    max_freq_cumsum_of_neg_background = max_freq_fraction_of_neg_background * background_sample_num

                # if tumors_marker_index == 2597:
                #     print('debug %d'%tumors_marker_index)

                alpha_threshold, alpha_union_list, a1_freq_cumsum_background, a2_freq_cumsum_tumors  = identify_alpha_threshold_by_alpha2sampleindexsetandreadfraction_of_pos_tumors_neg_paired_normaltissues_and_neg_background_V2(
                    background_unique_alpha_values,
                    background_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list,
                    tumors_unique_alpha_values,
                    tumors_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list,
                    paired_normaltissues_unique_alpha_values,
                    paired_normaltissues_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list,
                    max_freq_cumsum_of_neg_background,
                    min_freq_cumsum_of_pos_tumors,
                    min_read_fraction_neg_backgroundplasma,
                    diff_read_fraction_pairedtissues,
                    fixed_alpha_threshold_neg_backgroundplasma,
                    alpha_threshold_range_pairedtissues,
                    marker_type)
                if alpha_threshold is not None:
                    if 'hyper' in method:
                        ret_marker_2_alpha2freq[tumors_marker_index] = {'alpha_threshold': alpha_threshold,
                                                                        'max_cpg_num': max_cpg_num,
                                                                        'alpha2freq': filter_by_freq_cumsum_and_create_alpha2freqcumsum(
                                                                            alpha_threshold, alpha_union_list,
                                                                            a2_freq_cumsum_tumors,
                                                                            '>')}
                    elif 'hypo' in method:
                        ret_marker_2_alpha2freq[tumors_marker_index] = {'alpha_threshold': alpha_threshold,
                                                                        'max_cpg_num': max_cpg_num,
                                                                        'alpha2freq': filter_by_freq_cumsum_and_create_alpha2freqcumsum(
                                                                            alpha_threshold, alpha_union_list,
                                                                            a2_freq_cumsum_tumors,
                                                                            '<')}

            fid1_background.close()
            fid2_tumors.close()
            fid3_paired_normaltissues.close()
        return (ret_marker_2_alpha2freq)


# alpha_union_list and alpha_list_allowed_for_use: the list of alpha values, which are not only union of alpha values of pos_class and neg_class, but also with the order of alpha values, with respect to 'hyper' (in decreasing order) or 'hypo' (in increasing order).
# alpha2sampleindexset_of_neg_class, alpha2sampleindexset_of_pos_class: two dictionaries {'alpha_value':{sample_index}}
# Algorithm:
#    Step 1: accumulate sample_index_set of each alpha_value, by the alpha_value order of alpha_union_list
#    Step 2: calcualte size of accumulated sample_index_set of each alpha_value
#    Step 3: compute the following to determine the turning point. This step is the same as function 'identify_alpha_threshold_by_alpha2freq_of_pos_and_neg_class'
#            arr_neg_cumsum[index] <= arr_neg_cumsum_threshold
#            arr_pos_cumsum[index] >= arr_pos_cumsum_threshold
#            If there exist multiple index satisfying the above criteria, choose the largest index
#    Step 4: return the index of turning point (i.e., the index of alpha threshold in alpha_union_list)
#
# In Step 3, we perform the following:
#       1. Normal plasma, tumors & their matched/paired normal tissues use the same dynamic alpha threshold, which must take value in a predefined range
#       2. The paired tissue is counted into sample frequency, if the difference of read fractions of the tumor and its
#          adjacent normal tissue >= threshold
#
# Output: a tuple of three elements
#    Element 1: index of alpha value that is used for alpha threshold, in the list 'alpha_list_allowed_for_use'
#    Element 2 & 3: for debug use only
#
# NOTE:
#    1. 'alpha_list_allowed_for_use' MUST be an element and a subset of 'alpha_union_list', respectively.
#       We assume but NOT check this.
#
def identify_turning_point_of_three_alpha2sampleindexsetandreadfraction_V3(min_read_fraction_neg_backgroundplasma,
                                                                        diff_read_fraction_pairedtissues,
                                                                        alpha_union_list,
                                                                        alpha_list_allowed_for_use,
                                                                        alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma,
                                                                        alpha2sampleindexsetandreadfraction_of_pos_tumors,
                                                                        alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues,
                                                                        max_freq_cumsum_of_neg_backgroundplasma,
                                                                        min_freq_cumsum_of_pos_tumors):
    n_alpha = len(alpha_union_list)
    # implement cumsum using set union operator
    alpha2cumdict_neg_backgroundplasma = {a:dict() for a in alpha_union_list}
    alpha2cumdict_pos_tumors = dict(alpha2cumdict_neg_backgroundplasma)
    alpha2cumdict_neg_paired_normaltissues = dict(alpha2cumdict_neg_backgroundplasma)
    for i in range(n_alpha):
        a = alpha_union_list[i]
        if i == 0:
            alpha2cumdict_neg_backgroundplasma[a] = alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma[a] if a in alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma else dict()
            alpha2cumdict_pos_tumors[a] = alpha2sampleindexsetandreadfraction_of_pos_tumors[a] if a in alpha2sampleindexsetandreadfraction_of_pos_tumors else dict()
            alpha2cumdict_neg_paired_normaltissues[a] = alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues[a] if a in alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues else dict()
        else:
            a_prev = alpha_union_list[i - 1]
            alpha2cumdict_neg_backgroundplasma[a] = mergeDict_by_adding_values_of_common_keys(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma[a], alpha2cumdict_neg_backgroundplasma[a_prev]) if a in alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma else alpha2cumdict_neg_backgroundplasma[a_prev]
            alpha2cumdict_pos_tumors[a] = mergeDict_by_adding_values_of_common_keys(alpha2sampleindexsetandreadfraction_of_pos_tumors[a], alpha2cumdict_pos_tumors[a_prev]) if a in alpha2sampleindexsetandreadfraction_of_pos_tumors else alpha2cumdict_pos_tumors[a_prev]
            alpha2cumdict_neg_paired_normaltissues[a] = mergeDict_by_adding_values_of_common_keys(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues[a], alpha2cumdict_neg_paired_normaltissues[a_prev]) if a in alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues else alpha2cumdict_neg_paired_normaltissues[a_prev]

    arr_neg_cumsum = np.array([sum([True if alpha2cumdict_neg_backgroundplasma[a][sample_index] >= min_read_fraction_neg_backgroundplasma else False for sample_index in alpha2cumdict_neg_backgroundplasma[a]]) for a in alpha_list_allowed_for_use])

    # Check positive class: the following implement the criterion of paired tissues (tumor and its matched normal tissue)
    # Fill in alpha-accumulated read fractions for the missing samples in 'alpha2cumdict_pos_tumors[alpha]' & 'alpha2cumdict_neg_paired_normaltissues[alpha]'
    for a in alpha_union_list:
        for s in list(set(alpha2cumdict_pos_tumors[a].keys() | set(alpha2cumdict_neg_paired_normaltissues[a].keys()))):
            if s not in alpha2cumdict_pos_tumors[a]:
                alpha2cumdict_pos_tumors[a][s] = 0
            if s not in alpha2cumdict_neg_paired_normaltissues[a]:
                alpha2cumdict_neg_paired_normaltissues[a][s] = 0
    n_alpha_allowed_for_use = len(alpha_list_allowed_for_use)
    arr_pos_cumsum = np.array([sum([True if ((alpha2cumdict_pos_tumors[a][sample_index]-alpha2cumdict_neg_paired_normaltissues[a][sample_index])>=diff_read_fraction_pairedtissues) else False for sample_index in list(set(alpha2cumdict_pos_tumors[a].keys() & set(alpha2cumdict_neg_paired_normaltissues[a].keys())))]) for a in alpha_list_allowed_for_use])

    index_list = [i for i in range(n_alpha_allowed_for_use) if ((arr_pos_cumsum[i]>=min_freq_cumsum_of_pos_tumors) and (arr_neg_cumsum[i] <= max_freq_cumsum_of_neg_backgroundplasma))]

    if len(index_list) == 0:
        # No turning point
        return ((-2, arr_neg_cumsum, arr_pos_cumsum))
    else:
        # if (index_list[-1]+1) < (n_alpha-1):
        if (index_list[-1]) < (n_alpha_allowed_for_use - 1):
            return ((index_list[-1] + 1, arr_neg_cumsum, arr_pos_cumsum))
        elif (index_list[-1]) == (n_alpha_allowed_for_use - 1):
            # -1 means the index should be the one greater than n_alpha_allowed_for_use
            return ((-1, arr_neg_cumsum, arr_pos_cumsum))
        else:
            if len(index_list) >= 2:
                if (index_list[-2]) < (n_alpha_allowed_for_use - 1):
                    return ((index_list[-2] + 1, arr_neg_cumsum, arr_pos_cumsum))
            # -1 means the index should be the one greater than n_alpha_allowed_for_use
            return ((-1, arr_neg_cumsum, arr_pos_cumsum))


def identify_alpha_threshold_by_alpha2sampleindexsetandreadfraction_of_pos_tumors_neg_paired_normaltissues_and_neg_background_V3(
        unique_alpha_values_of_neg_background,
        sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_neg_background,
        unique_alpha_values_of_pos_tumors,
        sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_pos_tumors,
        unique_alpha_values_of_neg_paired_normaltissues,
        sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_neg_paired_normaltissues,
        max_freq_cumsum_of_neg_backgroundplasma,
        min_freq_cumsum_of_pos_tumors,
        min_read_fraction_neg_backgroundplasma,
        diff_read_fraction_pairedtissues,
        alpha_threshold_range,
        marker_type='hyper'):
    # build the dict for alpha2sampleindexsetandreadfraction for negative background plasma
    alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma = {}
    for i in range(len(unique_alpha_values_of_neg_background)):
        alpha = unique_alpha_values_of_neg_background[i]
        alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma[alpha] = {}
        items = sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_neg_background[i].split('_')
        for item in items:
            sample_index, read_fraction = item.split(':')
            read_fraction = float(read_fraction)
            alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma[alpha][sample_index] = read_fraction
    # build the dict for alpha2sampleindexsetandreadfraction for positive tumors
    alpha2sampleindexsetandreadfraction_of_pos_tumors = {}
    for i in range(len(unique_alpha_values_of_pos_tumors)):
        alpha = unique_alpha_values_of_pos_tumors[i]
        alpha2sampleindexsetandreadfraction_of_pos_tumors[alpha] = {}
        items = sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_pos_tumors[i].split('_')
        for item in items:
            sample_index, read_fraction = item.split(':')
            read_fraction = float(read_fraction)
            alpha2sampleindexsetandreadfraction_of_pos_tumors[alpha][sample_index] = read_fraction
    # build the dict for alpha2sampleindexsetandreadfraction for negative paired normal tissues
    alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues = {}
    for i in range(len(unique_alpha_values_of_neg_paired_normaltissues)):
        alpha = unique_alpha_values_of_neg_paired_normaltissues[i]
        alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues[alpha] = {}
        items = sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_neg_paired_normaltissues[i].split('_')
        for item in items:
            sample_index, read_fraction = item.split(':')
            read_fraction = float(read_fraction)
            alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues[alpha][sample_index] = read_fraction

    # Process
    alpha_union_list = []
    alpha_list_allowed_for_use = []
    if 'hyper' in marker_type:
        if len(alpha2sampleindexsetandreadfraction_of_pos_tumors)>0 and (len(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma)==0 or len(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues)==0):
            alpha_threshold = 0 # any reads with any alpha values should be used. So let alpha_threshold==0 for hyper markers
            return (alpha_threshold, [], [], [])
        if len(alpha2sampleindexsetandreadfraction_of_pos_tumors)>0 and len(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma)>0 and len(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues)>0:
            alpha_union_list = sorted(list(set(list(alpha2sampleindexsetandreadfraction_of_pos_tumors.keys()) + list(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma.keys()) + list(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues.keys()))), reverse=True) # decreasing order
            # Make 'alpha_list_allowed_for_use' (in decreasing order), a subset of 'alpha_union_list'
            lower_bound, upper_bound = alpha_threshold_range
            for a in alpha_union_list:
                if a<=upper_bound and a>=lower_bound:
                    alpha_list_allowed_for_use.append(a)
            # if (lower_bound not in alpha_union_list) or (upper_bound not in alpha_union_list):
            #     alpha_union_list += [lower_bound, upper_bound]
            #     alpha_union_list = sorted(alpha_union_list, reverse=True) # decreasing order

    elif 'hypo' in marker_type:
        if len(alpha2sampleindexsetandreadfraction_of_pos_tumors)>0 and (len(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma)==0 or len(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues)==0):
            alpha_threshold = 1 # any reads with any alpha values should be used. So let alpha_threshold==1 for hypo markers
            return (alpha_threshold, [], [], [])
        if len(alpha2sampleindexsetandreadfraction_of_pos_tumors)>0 and len(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma)>0 and len(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues)>0:
            alpha_union_list = sorted(list(set(list(alpha2sampleindexsetandreadfraction_of_pos_tumors.keys()) + list(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma.keys())+ list(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues.keys()))))  # increasing order
            # Make 'alpha_list_allowed_for_use' (in increasing order), a subset of 'alpha_union_list'
            lower_bound, upper_bound = alpha_threshold_range
            for a in alpha_union_list:
                if a <= upper_bound and a >= lower_bound:
                    alpha_list_allowed_for_use.append(a)
            # if (lower_bound not in alpha_union_list) or (upper_bound not in alpha_union_list):
            #     alpha_union_list += [lower_bound, upper_bound]
            #     alpha_union_list = sorted(alpha_union_list) # increasing order

    if len(alpha_union_list)==0:
        sys.stderr.write('Error (identify_alpha_threshold_by_alpha2sampleindexsetandreadfraction_of_pos_tumors_neg_paired_normaltissues_and_neg_background_V3):\n  Alpha union list of three classes are empty!\nExit.\n')
        sys.exit(-1)
    # identify_turning_point_of_three_alpha2sampleindexsetandreadfraction
    alpha_index, a1_freq_cumsum_neg_backgroundplasma, a2_freq_cumsum_pos_tumors = identify_turning_point_of_three_alpha2sampleindexsetandreadfraction_V3(
                                                                            min_read_fraction_neg_backgroundplasma,
                                                                            diff_read_fraction_pairedtissues,
                                                                            alpha_union_list,
                                                                            alpha_list_allowed_for_use,
                                                                            alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma,
                                                                            alpha2sampleindexsetandreadfraction_of_pos_tumors,
                                                                            alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues,
                                                                            max_freq_cumsum_of_neg_backgroundplasma,
                                                                            min_freq_cumsum_of_pos_tumors)
    if alpha_index == -2:
        alpha_threshold = None
    else:
        if alpha_index == -1:
            # -1 means the index should be the one greater than n_alpha_allowed_for_pairedtissues
            index_last = alpha_union_list.index(alpha_list_allowed_for_use[-1])
            if index_last < (len(alpha_union_list)-1):
                alpha_threshold = float(alpha_union_list[index_last + 1])
            else:
                alpha_threshold = None
            # if marker_type=='hyper':
            #     alpha_threshold = 0
            # elif marker_type == 'hypo':
            #     alpha_threshold = 1
        else:
            alpha_threshold = float(alpha_union_list[alpha_index])
            # if len(alpha2freq_of_neg_class) == 1:
            #     # Remove the following case which identifies hyper marker:
            #     # positive class: {'0.857':2, '0.286':1, '0.167':2, '0.143':4, '0':8}
            #     # negative class: {'0.167':2}
            #     # alpha_union_list: ['0.857', '0.286', '0.167', '0.143', '0']
            #     # identify_turning_point_of_two_cumsum_array returns alpha_index=4
            #     # But actually if alpha_threshold=='0', then negative class is incorrect for our purpose.
            #     if (('hyper' in marker_type) and (alpha_threshold<float(alpha2freq_of_neg_class.values[0]))) or (('hypo' in marker_type) and (alpha_threshold>float(alpha2freq_of_neg_class.values[0]))):
            #         alpha_threshold = None
    return ((alpha_threshold, alpha_union_list, a1_freq_cumsum_neg_backgroundplasma, a2_freq_cumsum_pos_tumors))


# Version 3
# Take three alpha_value_distribution files as inputs
# All three kinds of samples (normal plasma, tumors, matched normal tissues) use two different alpha-thresholds
# 1. Normal plasma use the fixed alpha threshold 'a_n' or 'an'
# 2. Tumors & their matched/paired normal tissues use the same dynamic alpha threshold 'a_t' or 'at' (could be
#    different from 'a_n'), which must take value in a predefined range
# 3. The paired tissue is counted into sample frequency, if the difference of read fractions of the tumor and its
#    adjacent normal tissue >= threshold
#
# Explanation of the parameters in the following method ID: 3 parts that are separated by ','
#   For the example ID: 'hyper.alpha.samplesetfreq.triple_V3.thresholds.nn0.2.p1,arange0.7_1,readfrac+pairedtissuediff0.5-ncplasma0.2'
#   Part1:
#     'nn0.2': sample frequency of negative class (i.e., non-cancer plasma that have tumor signals) <= 20% of all samples in negative class
#     'p1': sample frequency of positive class (i.e., tissue pairs that have tumor signals) >= 1
#   Part2:
#     'arange0.7_1': the dynamic Alpha threshold of all three sample types (non-cancer plasma, tumors & adjacent normal tissues) is in range [0.7, 1]
#   Part3:
#     'readfrac+pairedtissuediff0.5': for positive class, difference btw fractions of reads with tumor signals for tumor & matched normal tissue >= threshold 0.5
#     '-ncplasma0.2': for negative class, "Non-Cancer plasma" has the fraction of reads with tumor signal <= 0.2
#
def compare_background_vs_tumor_and_paired_normaltissue_alpha_value_distribution_files_with_memory_saving_way_for_cancer_detection_V3(method,
                                                                                                              in_file1_background,
                                                                                                              in_file2_tumors,
                                                                                                              in_file3_paired_normaltissues):
    # Scan three input files to obtain the markers that appear in all three files
    # get_specific_column_of_tab_file(in_file1_background)
    ret_marker_2_alpha2freq = {}
    if 'samplesetfreq' in method:
        if 'readfrac' in method: # 'hyper.alpha.samplesetfreq.triple_V3.thresholds.nn0.2.p1,arange0.7_1,readfrac+pairedtissuediff0.5-ncplasma0.2'
            part1_method, part2_method, part3_method = method.split(',')
            marker_type = part1_method.split('.')[0]
            if 'nn' in part1_method:
                max_freq_fraction_of_neg_background = float(extract_number_after_a_substring(part1_method,'nn'))  # max sample frequency, extract '0.2' from 'hyper.alpha.samplesetfreq.triple.thresholds.nn0.2.p1'
            else:
                max_freq_cumsum_of_neg_background = int(extract_number_after_a_substring(part1_method,'n'))  # max sample frequency, extract '2' from 'hyper.alpha.samplesetfreq.triple.thresholds.n2.p1'
            min_freq_cumsum_of_pos_tumors = int(extract_number_after_a_substring(part1_method,'p'))  # min sample frequency, extract '1' from 'hyper.alpha.samplesetfreq.triple.thresholds.nn0.2.p1'
            alpha_threshold_range = extract_range_after_a_substring(part2_method, 'arange')
            diff_read_fraction_pairedtissues = abs(float(extract_number_after_a_substring(part3_method, 'pairedtissuediff'))) # paired tissues are tumor & its matched normal tissue
            min_read_fraction_neg_backgroundplasma = abs(float(extract_number_after_a_substring(part3_method, 'ncplasma'))) # non-cancer plasma are background

            if in_file1_background.endswith('gz'):
                fid1_background = gzip.open(in_file1_background, 'rt')
            else:
                fid1_background = open(in_file1_background, 'rt')
            if in_file2_tumors.endswith('gz'):
                fid2_tumors = gzip.open(in_file2_tumors, 'rt')
            else:
                fid2_tumors = open(in_file2_tumors, 'rt')
            if in_file3_paired_normaltissues.endswith('gz'):
                fid3_paired_normaltissues = gzip.open(in_file3_paired_normaltissues, 'rt')
            else:
                fid3_paired_normaltissues = open(in_file3_paired_normaltissues, 'rt')

            ### begin to process three input files and write output file
            fid1_background.readline().rstrip()  # skip header line
            background_first_marker_line = fid1_background.readline()
            background_items = background_first_marker_line.rstrip().split('\t')
            background_marker_index = int(background_items[0])

            fid3_paired_normaltissues.readline().rstrip()  # skip header line
            paired_normaltissues_first_marker_line = fid3_paired_normaltissues.readline()
            paired_normaltissues_items = paired_normaltissues_first_marker_line.rstrip().split('\t')
            paired_normaltissues_marker_index = int(paired_normaltissues_items[0])

            end_of_background_file = False
            end_of_paired_normaltissues_file = False
            fid2_tumors.readline()  # skip header line
            for tumors_line in fid2_tumors:
                tumors_items = tumors_line.rstrip().split()
                tumors_marker_index = int(tumors_items[0])

                while tumors_marker_index > background_marker_index:
                    background_line = fid1_background.readline()
                    if not background_line:
                        end_of_background_file = True
                        break
                    background_items = background_line.rstrip().split('\t')
                    background_marker_index = int(background_items[0])
                if end_of_background_file:
                    break
                if tumors_marker_index < background_marker_index:
                    continue
                # now we arrive at tumors_marker_index == background_marker_index
                while tumors_marker_index > paired_normaltissues_marker_index:
                    paired_normaltissues_line = fid3_paired_normaltissues.readline()
                    if not paired_normaltissues_line:
                        end_of_paired_normaltissues_file = True
                        break
                    paired_normaltissues_items = paired_normaltissues_line.rstrip().split('\t')
                    paired_normaltissues_marker_index = int(paired_normaltissues_items[0])
                if end_of_paired_normaltissues_file:
                    break
                if tumors_marker_index < paired_normaltissues_marker_index:
                    continue
                # now we arrive at tumors_marker_index == background_marker_index == paired_normaltissues_marker_index

                # now we begin to process for tumors_marker_index == background_marker_index == paired_normaltissues_marker_index
                max_cpg_num = background_items[1]
                background_sample_num = int(background_items[2])
                background_unique_alpha_values = background_items[3].split(',')
                background_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list = background_items[4].split(',')

                tumors_unique_alpha_values = tumors_items[3].split(',')
                tumors_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list = tumors_items[4].split(',')

                paired_normaltissues_unique_alpha_values = paired_normaltissues_items[3].split(',')
                paired_normaltissues_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list = paired_normaltissues_items[4].split(',')

                if 'nn' in part1_method:
                    max_freq_cumsum_of_neg_background = max_freq_fraction_of_neg_background * background_sample_num

                # if tumors_marker_index == 2597:
                #     print('debug %d'%tumors_marker_index)

                alpha_threshold, alpha_union_list, a1_freq_cumsum_background, a2_freq_cumsum_tumors  = identify_alpha_threshold_by_alpha2sampleindexsetandreadfraction_of_pos_tumors_neg_paired_normaltissues_and_neg_background_V3(
                    background_unique_alpha_values,
                    background_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list,
                    tumors_unique_alpha_values,
                    tumors_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list,
                    paired_normaltissues_unique_alpha_values,
                    paired_normaltissues_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list,
                    max_freq_cumsum_of_neg_background,
                    min_freq_cumsum_of_pos_tumors,
                    min_read_fraction_neg_backgroundplasma,
                    diff_read_fraction_pairedtissues,
                    alpha_threshold_range,
                    marker_type)
                if alpha_threshold is not None:
                    if 'hyper' in method:
                        ret_marker_2_alpha2freq[tumors_marker_index] = {'alpha_threshold': alpha_threshold,
                                                                        'max_cpg_num': max_cpg_num,
                                                                        'alpha2freq': filter_by_freq_cumsum_and_create_alpha2freqcumsum(
                                                                            alpha_threshold, alpha_union_list,
                                                                            a2_freq_cumsum_tumors,
                                                                            '>')}
                    elif 'hypo' in method:
                        ret_marker_2_alpha2freq[tumors_marker_index] = {'alpha_threshold': alpha_threshold,
                                                                        'max_cpg_num': max_cpg_num,
                                                                        'alpha2freq': filter_by_freq_cumsum_and_create_alpha2freqcumsum(
                                                                            alpha_threshold, alpha_union_list,
                                                                            a2_freq_cumsum_tumors,
                                                                            '<')}

            fid1_background.close()
            fid2_tumors.close()
            fid3_paired_normaltissues.close()
        return (ret_marker_2_alpha2freq)

# num_str: a string of a number
# added_number: a float number
# precision_decimal_digit: the precision of ? decimal digits
# Example 1: add_to_number_string("0.388", 0.0001, 4) returns "0.3881"
# Example 2: add_to_number_string("0.388", -0.0001, 4) returns "0.3879"
def add_to_number_string(num_str, added_number, precision_decimal_digit=4):
    new_num = float(num_str) + added_number
    return("{0:.{1}f}".format(new_num,precision_decimal_digit))

def argmax_lastoccurrence(arr, selected_index_list):
    arr_reversed = arr[selected_index_list][::-1]
    return(selected_index_list[len(arr_reversed)-np.argmax(arr_reversed)-1])


def argmin_lastoccurrence(arr, selected_index_list):
    arr_reversed = arr[selected_index_list][::-1]
    return(selected_index_list[len(arr_reversed)-np.argmin(arr_reversed)-1])



# alpha_union_list and alpha_list_allowed_for_pairedtissues: the list of alpha values, which are not only union of alpha values of pos_class and neg_class, but also with the order of alpha values, with respect to 'hyper' (in decreasing order) or 'hypo' (in increasing order).
# alpha2sampleindexset_of_neg_class, alpha2sampleindexset_of_pos_class: two dictionaries {'alpha_value':{sample_index}}
# Algorithm:
#    Step 1: accumulate sample_index_set of each alpha_value, by the alpha_value order of alpha_union_list
#    Step 2: calcualte size of accumulated sample_index_set of each alpha_value
#    Step 3: compute the following to determine the turning point. This step is the same as function 'identify_alpha_threshold_by_alpha2freq_of_pos_and_neg_class'
#            arr_neg_cumsum[index] <= arr_neg_cumsum_threshold
#            arr_pos_cumsum[index] >= arr_pos_cumsum_threshold
#            If there exist multiple index satisfying the above criteria, choose the largest index
#    Step 4: return the index of turning point (i.e., the index of alpha threshold in alpha_union_list)
#
# In Step 3, we perform the following:
#       1. Normal plasma use the fixed alpha threshold
#       2. Tumors & their matched/paired normal tissues use the same dynamic alpha threshold, which must take value in a predefined range
#       3. The paired tissue is counted into sample frequency, if the difference of read fractions of the tumor and its
#          adjacent normal tissue >= threshold
#
# Output: a tuple of three elements
#    Element 1: index of alpha value that is used for alpha threshold, in the list 'alpha_list_allowed_for_pairedtissues'
#    Element 2 & 3: for debug use only
#
# NOTE:
#    1. 'fixed_alpha_threshold_neg_backgroundplasma' and 'alpha_list_allowed_for_pairedtissues' MUST be an element and a subset of 'alpha_union_list', respectively.
#       We assume but NOT check this.
#
def identify_turning_point_of_three_alpha2sampleindexsetandreadfraction_V5(min_read_fraction_neg_backgroundplasma,
                                                                           diff_read_fraction_pairedtissues,
                                                                           alpha_union_list,
                                                                           alpha_list_allowed_for_neg_backgroundplasma,
                                                                           min_alpha_threshold_diff,
                                                                           alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma,
                                                                           alpha2sampleindexsetandreadfraction_of_pos_tumors,
                                                                           alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues,
                                                                           max_freq_cumsum_of_neg_backgroundplasma,
                                                                           min_freq_cumsum_of_pos_tumors,
                                                                           alpha_order_direction='decreasing'):
    n_alpha = len(alpha_union_list)
    # implement cumsum using set union operator for negative background normal plasma
    alpha2cumdict_neg_backgroundplasma = {a:dict() for a in alpha_union_list}
    for i in range(n_alpha):
        a = alpha_union_list[i]
        if i == 0:
            alpha2cumdict_neg_backgroundplasma[a] = alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma[a] if a in alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma else dict()
        else:
            a_prev = alpha_union_list[i - 1]
            alpha2cumdict_neg_backgroundplasma[a] = mergeDict_by_adding_values_of_common_keys(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma[a], alpha2cumdict_neg_backgroundplasma[a_prev]) if a in alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma else alpha2cumdict_neg_backgroundplasma[a_prev]
    # Check negative class: the following implement the criterion of negative background normal plasma
    n_alpha_allowed_for_neg_backgroundplasma = len(alpha_list_allowed_for_neg_backgroundplasma)
    arr_neg_cumsum = [sum([True if alpha2cumdict_neg_backgroundplasma[a][sample_index]>=min_read_fraction_neg_backgroundplasma else False for sample_index in alpha2cumdict_neg_backgroundplasma[a]]) for a in alpha_list_allowed_for_neg_backgroundplasma]
    index_list_for_neg = [i for i in range(n_alpha_allowed_for_neg_backgroundplasma) if (arr_neg_cumsum[i] <= max_freq_cumsum_of_neg_backgroundplasma)]
    if len(index_list_for_neg) == 0:
        # No turning point
        return ((None, None, arr_neg_cumsum, None))
    else:
        if alpha_order_direction == 'decreasing': # for hyper-markers
            # alpha_threshold_for_neg = add_to_number_string(alpha_list_allowed_for_neg_backgroundplasma[index_list_for_neg[-1]], -0.0001, 4)
            alpha_threshold_for_neg = add_to_number_string(alpha_list_allowed_for_neg_backgroundplasma[argmax_lastoccurrence(np.array(arr_neg_cumsum), index_list_for_neg)], -0.0001, 4)
            lower_bound_for_pos = add_to_number_string(alpha_threshold_for_neg, float(min_alpha_threshold_diff), 4)
            if lower_bound_for_pos > '1.0000':
                lower_bound_for_pos = '1.0000'
            upper_bound_for_pos = '1.0000'
            alpha_list_allowed_for_pairedtissues = []
            for a in alpha_union_list:
                if (a>=lower_bound_for_pos) and (a<=upper_bound_for_pos):
                    alpha_list_allowed_for_pairedtissues.append(a)
        elif alpha_order_direction == 'increasing': # for hypo-markers
            # alpha_threshold_for_neg = add_to_number_string(alpha_list_allowed_for_neg_backgroundplasma[index_list_for_neg[-1]], 0.0001, 4)
            alpha_threshold_for_neg = add_to_number_string(alpha_list_allowed_for_neg_backgroundplasma[argmax_lastoccurrence(np.array(arr_neg_cumsum), index_list_for_neg)], 0.0001, 4)
            upper_bound_for_pos = add_to_number_string(alpha_threshold_for_neg, -float(min_alpha_threshold_diff), 4)
            if float(upper_bound_for_pos) < 0:
                upper_bound_for_pos = '0'
            lower_bound_for_pos = '0'
            alpha_list_allowed_for_pairedtissues = []
            for a in alpha_union_list:
                if (a>=lower_bound_for_pos) and (a<=upper_bound_for_pos):
                    alpha_list_allowed_for_pairedtissues.append(a)
        else:
            sys.stderr.write('Argument Error: alpha_order_direction (%s) should be "decreasing" or "increasing" in function identify_turning_point_of_three_alpha2sampleindexsetandreadfraction_V5!\nExit.\n'%alpha_order_direction)
            sys.exit(-1)

    if len(alpha_list_allowed_for_pairedtissues)==0:
        # No turning point
        return ((None, None, arr_neg_cumsum, None))

    # implement cumsum using set union operator for tumors & matched normal tissues
    alpha2cumdict_neg_paired_normaltissues = {a: dict() for a in alpha_union_list}
    alpha2cumdict_pos_tumors = dict(alpha2cumdict_neg_paired_normaltissues)
    for i in range(n_alpha):
        a = alpha_union_list[i]
        if i == 0:
            alpha2cumdict_pos_tumors[a] = alpha2sampleindexsetandreadfraction_of_pos_tumors[a] if a in alpha2sampleindexsetandreadfraction_of_pos_tumors else dict()
            alpha2cumdict_neg_paired_normaltissues[a] = alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues[a] if a in alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues else dict()
        else:
            a_prev = alpha_union_list[i - 1]
            alpha2cumdict_pos_tumors[a] = mergeDict_by_adding_values_of_common_keys(alpha2sampleindexsetandreadfraction_of_pos_tumors[a], alpha2cumdict_pos_tumors[a_prev]) if a in alpha2sampleindexsetandreadfraction_of_pos_tumors else alpha2cumdict_pos_tumors[a_prev]
            alpha2cumdict_neg_paired_normaltissues[a] = mergeDict_by_adding_values_of_common_keys(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues[a], alpha2cumdict_neg_paired_normaltissues[a_prev]) if a in alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues else alpha2cumdict_neg_paired_normaltissues[a_prev]
    # Check positive class: the following implement the criterion of paired tissues (tumor and its matched normal tissue)
    # Fill in alpha-accumulated read fractions for the missing samples in 'alpha2cumdict_pos_tumors[alpha]' & 'alpha2cumdict_neg_paired_normaltissues[alpha]'
    for a in alpha_union_list:
        for s in list(set(alpha2cumdict_pos_tumors[a].keys() | set(alpha2cumdict_neg_paired_normaltissues[a].keys()))):
            if s not in alpha2cumdict_pos_tumors[a]:
                alpha2cumdict_pos_tumors[a][s] = 0
            if s not in alpha2cumdict_neg_paired_normaltissues[a]:
                alpha2cumdict_neg_paired_normaltissues[a][s] = 0
    n_alpha_allowed_for_pairedtissues = len(alpha_list_allowed_for_pairedtissues)
    arr_pos_cumsum = np.array([sum([True if ((alpha2cumdict_pos_tumors[a][sample_index]-alpha2cumdict_neg_paired_normaltissues[a][sample_index])>=diff_read_fraction_pairedtissues) else False for sample_index in list(set(alpha2cumdict_pos_tumors[a].keys() & set(alpha2cumdict_neg_paired_normaltissues[a].keys())))]) for a in alpha_list_allowed_for_pairedtissues])
    index_list_for_pos = [i for i in range(n_alpha_allowed_for_pairedtissues) if (arr_pos_cumsum[i]>=min_freq_cumsum_of_pos_tumors)]
    if len(index_list_for_pos) == 0:
        # No turning point
        return ((None, None, arr_neg_cumsum, arr_pos_cumsum))
    else:
        if alpha_order_direction == 'decreasing': # for hyper-markers
            # alpha_threshold_for_pos = add_to_number_string(alpha_list_allowed_for_pairedtissues[index_list_pos[-1]], -0.0001, 4)
            alpha_threshold_for_pos = add_to_number_string(alpha_list_allowed_for_pairedtissues[argmax_lastoccurrence(np.array(arr_pos_cumsum), index_list_for_pos)], -0.0001, 4)
        elif alpha_order_direction == 'increasing': # for hypo-markers
            # alpha_threshold_for_pos = add_to_number_string(alpha_list_allowed_for_pairedtissues[index_list_pos[-1]], 0.0001, 4)
            alpha_threshold_for_pos = add_to_number_string(alpha_list_allowed_for_pairedtissues[argmax_lastoccurrence(np.array(arr_pos_cumsum), index_list_for_pos)], 0.0001, 4)
        else:
            sys.stderr.write('Argument Error: alpha_order_direction (%s) should be "decreasing" or "increasing" in function identify_turning_point_of_three_alpha2sampleindexsetandreadfraction_V5!\nExit.\n' % alpha_order_direction)
            sys.exit(-1)
        return ((alpha_threshold_for_neg, alpha_threshold_for_pos, arr_neg_cumsum, arr_pos_cumsum))


def identify_alpha_threshold_by_alpha2sampleindexsetandreadfraction_of_pos_tumors_neg_paired_normaltissues_and_neg_background_V5(
        unique_alpha_values_of_neg_background,
        sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_neg_background,
        unique_alpha_values_of_pos_tumors,
        sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_pos_tumors,
        unique_alpha_values_of_neg_paired_normaltissues,
        sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_neg_paired_normaltissues,
        max_freq_cumsum_of_neg_backgroundplasma,
        min_freq_cumsum_of_pos_tumors,
        min_read_fraction_neg_backgroundplasma,
        diff_read_fraction_pairedtissues,
        alpha_threshold_range_neg_backgroundplasma,
        min_alpha_threshold_diff,
        marker_type='hyper'):
    # build the dict for alpha2sampleindexsetandreadfraction for negative background plasma
    alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma = {}
    for i in range(len(unique_alpha_values_of_neg_background)):
        alpha = unique_alpha_values_of_neg_background[i]
        alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma[alpha] = {}
        items = sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_neg_background[i].split('_')
        for item in items:
            sample_index, read_fraction = item.split(':')
            read_fraction = float(read_fraction)
            alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma[alpha][sample_index] = read_fraction
    # build the dict for alpha2sampleindexsetandreadfraction for positive tumors
    alpha2sampleindexsetandreadfraction_of_pos_tumors = {}
    for i in range(len(unique_alpha_values_of_pos_tumors)):
        alpha = unique_alpha_values_of_pos_tumors[i]
        alpha2sampleindexsetandreadfraction_of_pos_tumors[alpha] = {}
        items = sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_pos_tumors[i].split('_')
        for item in items:
            sample_index, read_fraction = item.split(':')
            read_fraction = float(read_fraction)
            alpha2sampleindexsetandreadfraction_of_pos_tumors[alpha][sample_index] = read_fraction
    # build the dict for alpha2sampleindexsetandreadfraction for negative paired normal tissues
    alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues = {}
    for i in range(len(unique_alpha_values_of_neg_paired_normaltissues)):
        alpha = unique_alpha_values_of_neg_paired_normaltissues[i]
        alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues[alpha] = {}
        items = sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_neg_paired_normaltissues[i].split('_')
        for item in items:
            sample_index, read_fraction = item.split(':')
            read_fraction = float(read_fraction)
            alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues[alpha][sample_index] = read_fraction

    # Process
    alpha_union_list = []
    alpha_list_allowed_for_neg_backgroundplasma = []
    if 'hyper' in marker_type:
        if len(alpha2sampleindexsetandreadfraction_of_pos_tumors)>0 and (len(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma)==0 or len(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues)==0):
            # alpha_threshold = 0 # any reads with any alpha values should be used. So let alpha_threshold==0 for hyper markers
            return( (None, None, [], [], []) )
        if len(alpha2sampleindexsetandreadfraction_of_pos_tumors)>0 and len(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma)>0 and len(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues)>0:
            alpha_union_list = sorted(list(set(list(alpha2sampleindexsetandreadfraction_of_pos_tumors.keys()) + list(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma.keys()) + list(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues.keys()))), reverse=True) # decreasing order
            # Make 'alpha_list_allowed_for_pairedtissues' (in decreasing order), a subset of 'alpha_union_list'
            lower_bound, upper_bound = alpha_threshold_range_neg_backgroundplasma
            for a in alpha_union_list:
                if a<=upper_bound and a>=lower_bound:
                    alpha_list_allowed_for_neg_backgroundplasma.append(a)
        alpha_order_direction = 'decreasing'

    elif 'hypo' in marker_type:
        if len(alpha2sampleindexsetandreadfraction_of_pos_tumors)>0 and (len(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma)==0 or len(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues)==0):
            # alpha_threshold = 1 # any reads with any alpha values should be used. So let alpha_threshold==1 for hypo markers
            return( (None, None, [], [], []) )
        if len(alpha2sampleindexsetandreadfraction_of_pos_tumors)>0 and len(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma)>0 and len(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues)>0:
            alpha_union_list = sorted(list(set(list(alpha2sampleindexsetandreadfraction_of_pos_tumors.keys()) + list(alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma.keys())+ list(alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues.keys()))))  # increasing order
            # Make 'alpha_list_allowed_for_pairedtissues' (in increasing order), a subset of 'alpha_union_list'
            lower_bound, upper_bound = alpha_threshold_range_neg_backgroundplasma
            for a in alpha_union_list:
                if a <= upper_bound and a >= lower_bound:
                    alpha_list_allowed_for_neg_backgroundplasma.append(a)
        alpha_order_direction = 'increasing'
    else:
        sys.stderr.write('Argument Error: marker_type (%s) should include "hyper" or "hypo" in function identify_alpha_threshold_by_alpha2sampleindexsetandreadfraction_of_pos_tumors_neg_paired_normaltissues_and_neg_background_V5!\nExit.\n' % marker_type)
        sys.exit(-1)

    if len(alpha_union_list)==0:
        sys.stderr.write('Error (identify_alpha_threshold_by_alpha2sampleindexsetandreadfraction_of_pos_tumors_neg_paired_normaltissues_and_neg_background_V5):\n  Alpha union list of three classes are empty!\nExit.\n')
        sys.exit(-1)
    if len(alpha_list_allowed_for_neg_backgroundplasma) == 0:
        return( (None, None, [], [], []) )
    # identify_turning_point_of_three_alpha2sampleindexsetandreadfraction
    alpha_threshold_for_neg_backgroundplasma, alpha_threshold_for_pos_tumors, a1_freq_cumsum_neg_backgroundplasma, a2_freq_cumsum_pos_tumors = identify_turning_point_of_three_alpha2sampleindexsetandreadfraction_V5(
        min_read_fraction_neg_backgroundplasma,
        diff_read_fraction_pairedtissues,
        alpha_union_list,
        alpha_list_allowed_for_neg_backgroundplasma,
        min_alpha_threshold_diff,
        alpha2sampleindexsetandreadfraction_of_neg_backgroundplasma,
        alpha2sampleindexsetandreadfraction_of_pos_tumors,
        alpha2sampleindexsetandreadfraction_of_neg_paired_normaltissues,
        max_freq_cumsum_of_neg_backgroundplasma,
        min_freq_cumsum_of_pos_tumors,
        alpha_order_direction)
    return ((alpha_threshold_for_neg_backgroundplasma, alpha_threshold_for_pos_tumors, alpha_union_list, a1_freq_cumsum_neg_backgroundplasma, a2_freq_cumsum_pos_tumors))



# Version 5
# Take three alpha_value_distribution files as inputs
# All three kinds of samples (normal plasma, tumors, matched normal tissues) use two different alpha-thresholds
# 1. Normal plasma use the dynamic alpha threshold 'a_n' or 'an', which must take value in a predefined range
# 2. Tumors & their matched/paired normal tissues use another dynamic alpha threshold 'a_t' or 'at' (could be
#    different from 'a_n'), which has a FIXED difference from 'a_n' or 'an'
# 3. The paired tissue is counted into sample frequency, if the difference of read fractions of the tumor and its
#    adjacent normal tissue >= threshold
#
# Explanation of the parameters in the following method ID: 3 parts that are separated by ','
#   For the example ID: 'hyper.alpha.samplesetfreq.triple_V3.thresholds.nn0.2.p1,arange0.7_1,readfrac+pairedtissuediff0.5-ncplasma0.2'
#   Part1:
#     'nn0.2': sample frequency of negative class (i.e., non-cancer plasma that have tumor signals) <= 20% of all samples in negative class
#     'p1': sample frequency of positive class (i.e., tissue pairs that have tumor signals) >= 1
#   Part2:
#     'anrange0_0.5': the dynamic Alpha threshold of Normal plasma is in RANGE [0, 0.5]
#     'adiff0.5': minimum DIFFerence of the dynamic Alpha threshold of Tissues (applied to tumors & adjacent normal tissues) and the dynamic Alpha threshold of normal plasma is 0.5. For hyper-markers, a_tissue - a_normal >= min_a_diff = , e.g., a_tissue - 0.1 >= 0.5; for hypo-markers, a_normal - a_tissue >= min_a_diff, e.g., 0.8 - a_tissue >= 0.5.
#   Part3:
#     'readfrac+pairedtissuediff0.5': for positive class, difference btw fractions of reads with tumor signals for tumor & matched normal tissue >= threshold 0.5
#     '-ncplasma0.2': for negative class, "Non-Cancer plasma" has the fraction of reads with tumor signal <= 0.2
#
def compare_background_vs_tumor_and_paired_normaltissue_alpha_value_distribution_files_with_memory_saving_way_for_cancer_detection_V5(method,
                                                                                                              in_file1_background,
                                                                                                              in_file2_tumors,
                                                                                                              in_file3_paired_normaltissues):
    # Scan three input files to obtain the markers that appear in all three files
    # get_specific_column_of_tab_file(in_file1_background)
    ret_marker_2_alpha2freq = {}
    if 'samplesetfreq' in method:
        if 'readfrac' in method: # 'hyper.alpha.samplesetfreq.triple_V5.thresholds.nn0.2.p1,anrange0_0.5.adiff0.5,readfrac+pairedtissuediff0.3-ncplasma0.2'
            part1_method, part2_method, part3_method = method.split(',')
            marker_type = part1_method.split('.')[0]
            if 'nn' in part1_method:
                max_freq_fraction_of_neg_background = float(extract_number_after_a_substring(part1_method,'nn'))  # max sample frequency, extract '0.2' from 'hyper.alpha.samplesetfreq.triple.thresholds.nn0.2.p1'
            else:
                max_freq_cumsum_of_neg_background = int(extract_number_after_a_substring(part1_method,'n'))  # max sample frequency, extract '2' from 'hyper.alpha.samplesetfreq.triple.thresholds.n2.p1'
            min_freq_cumsum_of_pos_tumors = int(extract_number_after_a_substring(part1_method,'p'))  # min sample frequency, extract '1' from 'hyper.alpha.samplesetfreq.triple.thresholds.nn0.2.p1'
            alpha_threshold_range_neg_backgroundplasma = extract_range_after_a_substring(part2_method, 'anrange')
            min_alpha_threshold_diff = extract_number_after_a_substring(part2_method, 'adiff')
            diff_read_fraction_pairedtissues = abs(float(extract_number_after_a_substring(part3_method, 'pairedtissuediff'))) # paired tissues are tumor & its matched normal tissue
            min_read_fraction_neg_backgroundplasma = abs(float(extract_number_after_a_substring(part3_method, 'ncplasma'))) # non-cancer plasma are background

            if in_file1_background.endswith('gz'):
                fid1_background = gzip.open(in_file1_background, 'rt')
            else:
                fid1_background = open(in_file1_background, 'rt')
            if in_file2_tumors.endswith('gz'):
                fid2_tumors = gzip.open(in_file2_tumors, 'rt')
            else:
                fid2_tumors = open(in_file2_tumors, 'rt')
            if in_file3_paired_normaltissues.endswith('gz'):
                fid3_paired_normaltissues = gzip.open(in_file3_paired_normaltissues, 'rt')
            else:
                fid3_paired_normaltissues = open(in_file3_paired_normaltissues, 'rt')

            ### begin to process three input files and write output file
            fid1_background.readline().rstrip()  # skip header line
            background_first_marker_line = fid1_background.readline()
            background_items = background_first_marker_line.rstrip().split('\t')
            background_marker_index = int(background_items[0])

            fid3_paired_normaltissues.readline().rstrip()  # skip header line
            paired_normaltissues_first_marker_line = fid3_paired_normaltissues.readline()
            paired_normaltissues_items = paired_normaltissues_first_marker_line.rstrip().split('\t')
            paired_normaltissues_marker_index = int(paired_normaltissues_items[0])

            end_of_background_file = False
            end_of_paired_normaltissues_file = False
            fid2_tumors.readline()  # skip header line
            for tumors_line in fid2_tumors:
                tumors_items = tumors_line.rstrip().split()
                tumors_marker_index = int(tumors_items[0])

                while tumors_marker_index > background_marker_index:
                    background_line = fid1_background.readline()
                    if not background_line:
                        end_of_background_file = True
                        break
                    background_items = background_line.rstrip().split('\t')
                    background_marker_index = int(background_items[0])
                if end_of_background_file:
                    break
                if tumors_marker_index < background_marker_index:
                    continue
                # now we arrive at tumors_marker_index == background_marker_index
                while tumors_marker_index > paired_normaltissues_marker_index:
                    paired_normaltissues_line = fid3_paired_normaltissues.readline()
                    if not paired_normaltissues_line:
                        end_of_paired_normaltissues_file = True
                        break
                    paired_normaltissues_items = paired_normaltissues_line.rstrip().split('\t')
                    paired_normaltissues_marker_index = int(paired_normaltissues_items[0])
                if end_of_paired_normaltissues_file:
                    break
                if tumors_marker_index < paired_normaltissues_marker_index:
                    continue
                # now we arrive at tumors_marker_index == background_marker_index == paired_normaltissues_marker_index

                # now we begin to process for tumors_marker_index == background_marker_index == paired_normaltissues_marker_index
                max_cpg_num = background_items[1]
                background_sample_num = int(background_items[2])
                background_unique_alpha_values = background_items[3].split(',')
                background_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list = background_items[4].split(',')

                tumors_unique_alpha_values = tumors_items[3].split(',')
                tumors_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list = tumors_items[4].split(',')

                paired_normaltissues_unique_alpha_values = paired_normaltissues_items[3].split(',')
                paired_normaltissues_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list = paired_normaltissues_items[4].split(',')

                if 'nn' in part1_method:
                    max_freq_cumsum_of_neg_background = max_freq_fraction_of_neg_background * background_sample_num

                # if tumors_marker_index == 1332:
                #     print('debug %d'%tumors_marker_index)

                alpha_threshold_for_neg_backgroundplasma, alpha_threshold_for_pos_tumors, alpha_union_list, a1_freq_cumsum_background, a2_freq_cumsum_tumors  = identify_alpha_threshold_by_alpha2sampleindexsetandreadfraction_of_pos_tumors_neg_paired_normaltissues_and_neg_background_V5(
                    background_unique_alpha_values,
                    background_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list,
                    tumors_unique_alpha_values,
                    tumors_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list,
                    paired_normaltissues_unique_alpha_values,
                    paired_normaltissues_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list,
                    max_freq_cumsum_of_neg_background,
                    min_freq_cumsum_of_pos_tumors,
                    min_read_fraction_neg_backgroundplasma,
                    diff_read_fraction_pairedtissues,
                    alpha_threshold_range_neg_backgroundplasma,
                    min_alpha_threshold_diff,
                    marker_type)
                if alpha_threshold_for_pos_tumors is not None:
                    if 'hyper' in method:
                        ret_marker_2_alpha2freq[tumors_marker_index] = {'alpha_threshold_of_neg': alpha_threshold_for_neg_backgroundplasma,
                                                                        'alpha_threshold_of_pos': alpha_threshold_for_pos_tumors,
                                                                        'max_cpg_num': max_cpg_num,
                                                                        'alpha2freq': filter_by_freq_cumsum_and_create_alpha2freqcumsum(
                                                                            float(alpha_threshold_for_pos_tumors), alpha_union_list,
                                                                            a2_freq_cumsum_tumors,
                                                                            '>')}
                    elif 'hypo' in method:
                        ret_marker_2_alpha2freq[tumors_marker_index] = {'alpha_threshold_of_neg': alpha_threshold_for_neg_backgroundplasma,
                                                                        'alpha_threshold_of_pos': alpha_threshold_for_pos_tumors,
                                                                        'max_cpg_num': max_cpg_num,
                                                                        'alpha2freq': filter_by_freq_cumsum_and_create_alpha2freqcumsum(
                                                                            float(alpha_threshold_for_pos_tumors), alpha_union_list,
                                                                            a2_freq_cumsum_tumors,
                                                                            '<')}

            fid1_background.close()
            fid2_tumors.close()
            fid3_paired_normaltissues.close()
        return (ret_marker_2_alpha2freq)

####################################################################################
####################################################################################
## functions for samples
####################################################################################
####################################################################################

# cancer stages are I, IA, IA1, IA2, IA3, IB, II, IIA, IIB, IIC, IIIA, IIIB, IIIC, IV, IVA, IVB
# standard cancer stages are I, II, III, IV
def normalize_cancer_stages(cancer_stages_list):
    normalized_cancer_stages_list = [stage.replace('A','').replace('B','').replace('C','').replace('1','').replace('2','').replace('3','') for stage in cancer_stages_list]
    return(normalized_cancer_stages_list)

def normalize_cancer_stage(cancer_stage):
    normalized_cancer_stage = cancer_stage.replace('A','').replace('B','').replace('C','').replace('1','').replace('2','').replace('3','')
    return(normalized_cancer_stage)

def get_cancer_types_list(short_or_long_name='short_name'):
    cancer_types_list = []
    if short_or_long_name == 'short_name':
        cancer_types_list = ['CC', 'LV', 'LC', 'LG', 'ST']
    elif short_or_long_name == 'long_name':
        cancer_types_list = ['COAD', 'LIHC', 'LUSC', 'LUAD', 'STAD']
    short2long = {'CC':'COAD', 'LV':'LIHC', 'LC':'LUSC', 'LG':'LUAD', 'ST':'STAD'}
    long2short = {'COAD': 'CC', 'LIHC': 'LV', 'LUSC': 'LC', 'LUAD': 'LG', 'STAD': 'ST'}
    return(cancer_types_list, short2long, long2short)

# Input:
#   longname_type: 'HumanName' refers to long name such as 'Noncancer', 'Lung Cancer'; 'AbbreviateName' refers to long name such as 'COAD', 'LUAD', 'LIHC'
def get_ordered_classes_list(cancer_type_num=4, include_control_class=True, longname_type='HumanName'):
    class_list_for_short = []
    if include_control_class:
        class_list_for_short.append('N_CH') # 'N_CH' represents "normal" and "cirrhosis and other liver diseases"
    if cancer_type_num == 4:
        class_list_for_short += ['CC', 'LV', 'LC_LG', 'ST']
        if longname_type == 'HumanName':
            short2long = {'N_CH':'Noncancer', 'CC': 'Colon Cancer', 'LV': 'Liver Cancer', 'LC_LG': 'Lung Cancer', 'ST': 'Stomach Cancer','CC_LV_LC_LG_ST':'Pan-Cancer'}
            long2short = {'Noncancer':'N_CH', 'COAD': 'Colon Cancer', 'LIHC': 'Liver Cancer', 'Lung Cancer': 'LC_LG', 'Stomach Cancer': 'ST', 'Pan-Cancer':'CC_LV_LC_LG_ST'}
        elif longname_type == 'AbbreviateName':
            short2long = {'N_CH':'Noncancer', 'CC': 'COAD', 'LV': 'LIHC', 'LC_LG': 'LUNG', 'ST': 'STAD', 'CC_LV_LC_LG_ST':'PANCANCER'}
            long2short = {'Noncancer':'N_CH', 'COAD': 'CC', 'LIHC': 'LV', 'LUNG': 'LC_LG', 'STAD': 'ST', 'PANCANCER':'CC_LV_LC_LG_ST'}
    elif cancer_type_num == 5:
        class_list_for_short += ['CC', 'LV', 'LC', 'LG', 'ST']
        if longname_type == 'HumanName':
            short2long = {'N_CH':'Noncancer', 'CC': 'Colon Cancer', 'LV': 'Liver Cancer', 'LC': 'Lung Squamous Cell Carcinoma', 'LG':'Lung Adenocarcinoma', 'ST': 'Stomach Cancer', 'CC_LV_LC_LG_ST':'Pan-Cancer'}
            long2short = {'Noncancer':'N_CH', 'COAD': 'Colon Cancer', 'LIHC': 'Liver Cancer', 'Lung Squamous Cell Carcinoma': 'LC', 'Lung Adenocarcinoma':'LG', 'Stomach Cancer': 'ST', 'Pan-Cancer':'CC_LV_LC_LG_ST'}
        elif longname_type == 'AbbreviateName':
            short2long = {'N_CH':'Noncancer', 'CC': 'COAD', 'LV': 'LIHC', 'LC': 'LUSC', 'LG': 'LUAD', 'ST': 'STAD', 'CC_LV_LC_LG_ST':'PANCANCER'}
            long2short = {'Noncancer':'N_CH', 'COAD': 'CC', 'LIHC': 'LV', 'LUSC': 'LC', 'LUAD': 'LG', 'STAD': 'ST', 'PANCANCER':'CC_LV_LC_LG_ST'}
    class_list_for_long = [short2long[c] for c in class_list_for_short]
    return (class_list_for_short, class_list_for_long, short2long, long2short)

# Input
#   selected_samples_list: a list of sample names that we want to extract their annotation info
#   sample_annot_table: a dictionary generated from function 'parse_csv_file_with_header_line'
def reorder_samples_list_by_class_age_and_cancer_stages(selected_samples_list, sample_annot_table, cancer_type_num=4, include_control_class=True):
    class_list_for_short, class_list_for_long, class_short2long, class_long2short = get_ordered_classes_list(cancer_type_num, include_control_class, 'HumanName')
    df_sample_annot = pd.DataFrame.from_dict(sample_annot_table).set_index('sample')
    df_sample_annot.sort_values(['class', 'stage', 'age'], inplace=True, ascending = [True, True, True])
    # Sort the selected_samples_list and return the ordered samples list
    selected_samples_list_sorted = {'sample':[], 'age':[], 'class':[], 'stage':[]}
    for c_short in class_list_for_short:
        for s in df_sample_annot.index.values: # index is sample name
            c = df_sample_annot.loc[s, 'class']
            if (c in c_short) and (s in selected_samples_list): # sometimes, c='CH' and c='N_CH', so we should use (c in c_short)
                selected_samples_list_sorted['sample'].append(s)
                selected_samples_list_sorted['age'].append(df_sample_annot.loc[s, 'age'])
                selected_samples_list_sorted['class'].append( class_short2long[c_short] )
                selected_samples_list_sorted['stage'].append(df_sample_annot.loc[s, 'stage'])
    return(selected_samples_list_sorted, class_list_for_long)


# Input
#   selected_samples_list: a list of sample names that we want to extract their annotation info
#   sample_annot_table: a dictionary generated from function 'parse_csv_file_with_header_line'
def reorder_samples_list_of_noncancer_and_one_cancertype_by_class_age_and_cancer_stages(sample_annot_table, cancer_type_longname="LUNG", cancer_type_num=4):
    class_list_for_short, class_list_for_long, class_short2long, class_long2short = get_ordered_classes_list(cancer_type_num, True, 'AbbreviateName')
    df_sample_annot = pd.DataFrame.from_dict(sample_annot_table).set_index('sample')
    df_sample_annot.sort_values(['class', 'stage', 'age'], inplace=True, ascending = [True, True, True])
    # Sort the selected_samples_list and return the ordered samples list
    classes_shortnames_list_for_retrieval = [class_long2short['Noncancer'], class_long2short[cancer_type_longname]] # "LUNG" -> "LC_LG" and "Noncancer" -> "N_CH"
    selected_samples_list_sorted = {'sample':[], 'age':[], 'class':[], 'stage':[]}
    for c_short in classes_shortnames_list_for_retrieval:
        for s in df_sample_annot.index.values: # index is sample name
            c = df_sample_annot.loc[s, 'class']
            if (c in c_short): # sometimes, c='CH' and c='N_CH', so we should use (c in c_short)
                selected_samples_list_sorted['sample'].append(s)
                selected_samples_list_sorted['age'].append(df_sample_annot.loc[s, 'age'])
                selected_samples_list_sorted['class'].append( class_short2long[c_short] )
                selected_samples_list_sorted['stage'].append(df_sample_annot.loc[s, 'stage'])
    class_list_for_long_for_retrieval = [class_short2long[c] for c in classes_shortnames_list_for_retrieval]
    return(selected_samples_list_sorted, class_list_for_long_for_retrieval)


# Input
#   selected_samples_list_with_order: a list of sample names that has its own sample order and we want to extract their annotation info
#   sample_annot_table: a dictionary generated from function 'parse_csv_file_with_header_line'
def reorder_samples_list_by_class_and_provided_order(selected_samples_list_with_order, sample_annot_table, cancer_type_num=4, include_control_class=True):
    class_list_for_short, class_list_for_long, class_short2long, class_long2short = get_ordered_classes_list(cancer_type_num, include_control_class, 'HumanName')
    df_sample_annot = pd.DataFrame.from_dict(sample_annot_table).set_index('sample')
    df_sample_annot.sort_values(['class', 'stage', 'age'], inplace=True, ascending = [True, True, True])
    # Sort the selected_samples_list and return the ordered samples list
    selected_samples_list_sorted = {'sample':[], 'age':[], 'class':[], 'stage':[]}
    for c_short in class_list_for_short:
        for s in selected_samples_list_with_order:
            c = df_sample_annot.loc[s, 'class']
            if c in c_short: # sometimes, c='CH' and c='N_CH', so we should use (c in c_short)
                selected_samples_list_sorted['sample'].append(s)
                selected_samples_list_sorted['age'].append(df_sample_annot.loc[s, 'age'])
                selected_samples_list_sorted['class'].append( class_short2long[c_short] )
                selected_samples_list_sorted['stage'].append(df_sample_annot.loc[s, 'stage'])
    return(selected_samples_list_sorted, class_list_for_long)

####################################################################################
####################################################################################
## Plot functions
####################################################################################
####################################################################################

# For plotting heatmap
def convert_2_or_4_or_5_classes_to_colors(class_labels_of_samples, unique_classnames_list):
    # https://www.w3schools.com/colors/colors_picker.asp
    unknown_class_color = 'black'
    unique_colors_list = []
    if len(unique_classnames_list)==2:
        unique_colors_list = ['springgreen', 'lightcoral']
    elif len(unique_classnames_list)==4:
        # unique_colors_list = ['red', 'blue', 'orange', '#8000ff']
        unique_colors_list = ['lightcoral', 'cornflowerblue', 'goldenrod', 'plum']
    elif len(unique_classnames_list)==5:
        # unique_colors_list = ['red', 'blue', 'orange', '#8000ff', '#00ffff']
        unique_colors_list = ['springgreen', 'lightcoral', 'cornflowerblue', 'goldenrod', 'plum', '#00ffff']
    colors_of_samples = []
    for sc in class_labels_of_samples:
        found = -1
        for c in unique_classnames_list:
            if sc in c:
                found = unique_classnames_list.index(c)
                break
        if found==-1:
            colors_of_samples.append(unknown_class_color)
        else:
            colors_of_samples.append(unique_colors_list[found])
    return(colors_of_samples, unique_colors_list)

# uniquecancerstage2color_dict: cancer_stage -> RGB_color_tuple
def make_colors_for_cancer_stages_list():
    # https://stackoverflow.com/questions/61816216/seaborn-clustermap-with-two-row-colors
    # https://rgbcolorcode.com/color/converter/
    unique_cancer_stages_list = "I,II,III,IV,_".split(",") # "_" indicates unknown stage
    # color_palette = sns.color_palette("Reds", 4, .75) + [(1,1,1)] # list of RGB_color_tuples
    # https://matplotlib.org/3.1.1/gallery/color/named_colors.html
    # color_palette = ['gold', 'darkorange', 'firebrick', 'blueviolet', 'white']
    color_palette = ['cyan', 'darkturquoise', 'cornflowerblue', 'darkblue', 'white'] # for stage "I,II,III,IV,_"
    uniquecancerstage2color_dict = dict(zip(unique_cancer_stages_list, color_palette))
    return(uniquecancerstage2color_dict, unique_cancer_stages_list, color_palette)

def convert_classes_to_colors_and_with_cancer_stages(class_labels_of_samples, cancer_stages_of_samples, unique_classnames_list):
    # https://www.w3schools.com/colors/colors_picker.asp
    unknown_class_color = 'black'
    unique_colors_list_of_class_labels = []
    if len(unique_classnames_list)==2:
        unique_colors_list_of_class_labels = ['springgreen', 'lightcoral']
    elif len(unique_classnames_list)==4:
        # unique_colors_list = ['red', 'blue', 'orange', '#8000ff']
        unique_colors_list_of_class_labels = ['lightcoral', 'cornflowerblue', 'goldenrod', 'plum']
    elif len(unique_classnames_list)==5:
        # unique_colors_list = ['red', 'blue', 'orange', '#8000ff', '#00ffff']
        unique_colors_list_of_class_labels = ['springgreen', 'lightcoral', 'cornflowerblue', 'goldenrod', 'plum', '#00ffff']
    colors_of_class_labels_for_samples = []
    for sc in class_labels_of_samples:
        found = -1
        for c in unique_classnames_list:
            if sc in c:
                found = unique_classnames_list.index(c)
                break
        if found==-1:
            colors_of_class_labels_for_samples.append(unknown_class_color)
        else:
            colors_of_class_labels_for_samples.append(unique_colors_list_of_class_labels[found])

    uniquecancerstage2color_dict, unique_list_of_cancer_stages, unique_colors_list_of_cancer_stages = make_colors_for_cancer_stages_list()
    colors_of_cancer_stages_for_samples = [uniquecancerstage2color_dict[stage] for stage in cancer_stages_of_samples]
    color_info = {'colors_of_class_labels_for_samples':colors_of_class_labels_for_samples,
                  'colors_of_cancer_stages_for_samples':colors_of_cancer_stages_for_samples,
                  'unique_colors_list_of_class_labels':unique_colors_list_of_class_labels,
                  'unique_colors_list_of_cancer_stages':unique_colors_list_of_cancer_stages,
                  'unique_cancer_stages':unique_list_of_cancer_stages[:4] }
    return(color_info)

# For plotting bars & error bars
# Call "output_pdf_pages = PdfPages(outptu_pdf_filename)" before calling this function and
# call "output_pdf_pages.close()" after calling this function
def plot_bars_with_error_bars(values_list, errors_list, output_pdf_pages, x_label, x_ticklabels, y_label, y_ticks, y_ticklabels=None, title="", color_list=None):
    n_values = len(values_list)
    if color_list is None:
        # https://matplotlib.org/stable/gallery/color/named_colors.html
        all_color_list = 'darkorange mediumblue darkcyan darkmagenta darkred dimgray'.split(' ')
        color_list = [all_color_list[i % len(all_color_list)] for i in range(n_values)]
    errorbar_color = 'dimgray'
    lw = 0.5
    # page_width = 3  # inch
    # page_height = 3  # inch
    fontsize = 8
    plt.rcParams.update(
        {'font.family': 'sans-serif', 'font.sans-serif': 'Tahoma', 'font.size': fontsize, 'font.weight': "bold"})
    fig, ax = plt.subplots()
    x_pos = np.arange(n_values)
    ax.bar(x_pos,
           values_list,
           yerr=errors_list,
           width=lw,
           capsize=5,
           align='center', alpha=0.5, color=color_list, ecolor=errorbar_color)
    ax.set_xlabel(x_label, fontweight='bold', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_ticklabels, rotation='vertical')
    ax.set_ylim((min(y_ticks), max(y_ticks)+0.02))
    ax.set_yticks(y_ticks)
    if y_ticklabels is None:
        y_ticklabels = ['%g'%y for y in y_ticks]
    ax.set_yticklabels(y_ticklabels)
    ax.set_ylabel(y_label, fontweight='bold', fontsize=14)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.yaxis.grid(True, linewidth=0.2, linestyle='--')
    # plt.show()
    plt.tight_layout()
    output_pdf_pages.savefig(fig)
    plt.cla()
    plt.close(fig)  # To save memory, remove a specific figure instance from the pylab state machine, and allow it to be garbage collected.
