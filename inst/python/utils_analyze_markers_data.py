#
# Python 3.x
#
import sys, gzip, re
import numpy as np
from scipy.sparse import lil_matrix
# import pandas as pd
from datetime import datetime
from itertools import combinations
# import collections
from collections import Counter
from operator import itemgetter
# from skfeature.function.similarity_based import fisher_score
# from skfeature.function.statistical_based import gini_index

# Return the first number that satifies the requirement:
# E.g.: extract_number_after_a_substring('hypermarker_Nsf0.95Nrf1Na0.2Tsf1Trf0.2Ta1cov10cpg6Nsf0.76', 'Nsf')
#       returns '0.95', not '0.76'.
def extract_number_after_a_substring(str_, substr_):
    a = re.findall(r'%s([-+]?\d*\.\d+|\d+)'%substr_, str_)
    if len(a)>=1:
        return(a[0])
    else:
        return(None)

# get the second minimum of a numpy array
def min_2nd(a):
    return(np.amin(a[a != np.amin(a)]))

# sort a list in increasing order and return index of a sorted list
def sort_list_and_return_sorted_index(l):
    if len(l)==0:
        return([])
    else:
        return(sorted(range(len(l)), key=lambda k: l[k]))

# https://devenum.com/merge-python-dictionaries-add-values-of-common-keys/
def mergeDict_by_adding_values_of_common_keys(dict1, dict2):
   return(dict(Counter(dict1) + Counter(dict2)))

def prepro_impute_na_by_zero(mat):
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
    if file.endswith('gz'):
        f = gzip.open(file, 'rt')
    else:
        f = open(file, 'rt')
    for line in f:
        lines.append(line.rstrip())
    f.close()
    return(lines)

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
        data.append( np.array(list(map(float, items)), dtype=np.float) )
    f.close()
    return( (lil_matrix(data, dtype=np.float), rownames) )

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
        profile_str = '\t'.join(['%g' % v for v in mat[i, :]])
        profile_str = profile_str.replace('nan', 'NA')
        fid.write('%s\t%s\n'%(row_names_list[i], profile_str))

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
            marker_index, meth_string = line.rstrip().split('\t')
            # marker_index, _, meth_string, _, _, _ = line.rstrip().split('\t')
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
# alpha_hists: a dictionary { marker_index:alpha_histgram_dictionary }, for example: {27:hist_dict, 63:hist_dict}, where hist_dict is a dictionary {'0.7':2, '1.0':4}. The input 'alpha_hists' can be an empty dictionary {}.
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
        num_reads = sum(alpha_hists[marker_index]['alpha2freq'].values())
        str_for_unique_alpha_values, read_freq_of_unique_alpha_values_str = convert_str2int_dict_to_str(alpha_hists[marker_index]['alpha2freq'])
        fout.write('%d\t%s\t%d\t%s\t%s\n' % (marker_index,
                                             marker2max_cpg_num[marker_index],
                                             num_reads,
                                             str_for_unique_alpha_values,
                                             read_freq_of_unique_alpha_values_str
                                             ))

#
# alpha_hists: a dictionary { 'marker_index':alpha_histgram_dictionary }, for example: {'27':hist_dict, '63':hist_dict}, where hist_dict is a dictionary {'0.7':2, '1.0':4}. The input 'alpha_hists' can be an empty dictionary {}.
# marker2max_cpg_num: a dictionary {'marker':max_cpg_num}.
#
def combine_multi_alpha_histgram_files(files_list):
    combined_alpha_hists = {}
    marker2max_cpg_num = {}
    n = len(files_list)
    i = 0
    for filename in files_list:
        i += 1
        print('  (%d/%d) %s, '%(i, n, filename), datetime.now(), flush=True)
        load_one_alpha_value_distribution_file(filename, combined_alpha_hists, marker2max_cpg_num)
    return( (combined_alpha_hists,marker2max_cpg_num) )


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
#   in_file1_background and in_file2_cancer: file format is from function 'write_combined_meth_string_histgram'
# Procedure:
#   We first load each of these two files into a dictionary { 'marker_strand':histgram_dictionary }, for example: {27:hist_dict, 63:hist_dict}, where hist_dict is a dictionary {'0.7':2, '0.9':4}. It is the output of 'load_one_alpha_value_distribution_file' or 'combine_multi_alpha_histgram_files'
#
# Output:
#   ret_marker_2_alpha2freq: a dictionary {'alpha_threshold':alpha_cutoff, 'max_cpg_num':max_cpg_num_of_the_marker, 'alpha2freq':histgram_dictionary}
#
def compare_background_vs_cancer_alpha_value_distribution_files(method, in_file1_background, in_file2_cancer):
    a1_background = {}
    marker2max_cpg_num_1 = {}
    load_one_alpha_value_distribution_file(in_file1_background, a1_background, marker2max_cpg_num_1)
    a2_cancer = {}
    marker2max_cpg_num_2 = {}
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
    except KeyError:
        # marker_index does not exist
        sys.stderr.write('Error: %d does not exist in one of two meth_strings_histgram_files\n  in_file1_background: %s\n  in_file2_cancer: %s\nExit.'%(m, in_file1_background, in_file2_cancer))
        sys.exit(-1)
    # ret_marker2max_cpg_num = {m: marker2max_cpg_num_2[m] for m in ret_marker_2_alpha2freq}
    return( ret_marker_2_alpha2freq )

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
def write_alpha_value_distribution_file_with_alpha_threshold(fout, alpha_hists):
    marker_index_list = sorted(list(set(alpha_hists.keys())))
    fout.write(
        'marker_index\tmax_num_cpg\tnum_read\talpha_threshold\tunique_alpha_values\tread_freq_of_unique_alpha_values\n')
    for marker_index in marker_index_list:
        num_reads = sum(alpha_hists[marker_index]['alpha2freq'].values())
        str_for_unique_alpha_values, read_freq_of_unique_alpha_values_str = convert_str2int_dict_to_str(alpha_hists[marker_index]['alpha2freq'])
        fout.write('%d\t%s\t%d\t%g\t%s\t%s\n' % (marker_index,
                                                 alpha_hists[marker_index]['max_cpg_num'],
                                                 num_reads,
                                                 alpha_hists[marker_index]['alpha_threshold'],
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
# alpha_hists: a dictionary { marker_index:alpha_histgram_dictionary }, for example: {27:hist_dict, 63:hist_dict}, where hist_dict is a dictionary {'0.7':2, '1.0':4}. The input 'alpha_hists' can be an empty dictionary {}.
# marker2max_cpg_num: a dictionary {marker_index:'max_cpg_num'}.
#
# Output:
#   markers: a dictionary {'pair_comparison_list':['hypo.COAD+LIHC-_r10_a0.3','hypo.LUAD+LIHC-_r10_a0.3'], 'alpha_threshold_list':[0.3, 0.4], 'alpha_threshold_for_final_use':0.3}
def load_one_alpha_value_distribution_file_with_alpha_threshold_for_marker_selection(file, markers_list, marker_type, min_cpg_num):
    markers = {}
    with gzip.open(file, 'rt') as f:
        next(f) # skip the first header line
        for line in f:
            items = line.rstrip().split()
            max_num_cpg = int(items[1])
            if max_num_cpg<min_cpg_num:
                continue
            marker_index = int(items[0])
            num_read = items[2]
            alpha_threshold = items[3]
            if marker_index not in markers_list:
                alpha_threshold = float(alpha_threshold)
                markers_list[marker_index] = {'alpha_threshold_for_final_use':None,
                                              'pair_comparison_list':['%s_r%s_a%s'%(marker_type, num_read, alpha_threshold)],
                                              'alpha_threshold_list': [alpha_threshold],
                                              'max_num_cpg':max_num_cpg}
            else:
                # update markers
                alpha_threshold = float(alpha_threshold)
                markers_list[marker_index]['pair_comparison_list'].append('%s_r%s_a%s'%(marker_type,num_read,alpha_threshold))
                markers_list[marker_index]['alpha_threshold_list'].append( alpha_threshold )
    return(markers_list)

# Input:
#   list_of_paired_comparisons: a list of tuples [('COAD+vsLIHC-',filename), ('COAD+vsLUAD-',filename), ... ]
#   marker_type: 'hyper' or 'hypo'
#   min_cpg_num: minimum number of CpG sites required by markers
#   output_union_markers_file
#
# Output file format:
#
def union_markers_of_paired_comparison_alpha_value_distribution_file_with_alpha_threshold(list_of_paired_comparisons, marker_type, min_cpg_num, output_union_markers_file):
    union_markers_list = {}
    print('Union markers', flush=True)
    for pair_comparison_type, filename in list_of_paired_comparisons:
        print('  %s'%pair_comparison_type, flush=True)
        load_one_alpha_value_distribution_file_with_alpha_threshold_for_marker_selection(filename,
                                                                                         union_markers_list,
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
    else:
        f = open(file)
    markers_list['header_line'] = next(f).rstrip() # skip the header line
    for line in f:
        items = line.rstrip().split('\t')
        marker_index = int(items[0])
        markers_list['marker_index'].append( marker_index )
        markers_list['alpha_threshold'].append( float(items[1]) )
        markers_list['lines'].append( line.rstrip() )
    f.close()
    return(markers_list)


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
    current_marker_pointer_in_markers_list = 0
    with gzip.open(file, 'rt') as f:
        next(f) # skip the header line
        for line in f:
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
                continue
            # now 'marker_index == markers_list_int[current_marker_pointer_in_markers_list]'
            idx = markers_list['marker_index'].index(marker_index)
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
    return(profile)


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
    for i in range(n_sample):
        input_sample_alpha_distr_file = input_alpha_value_distribution_files_list[i]
        print('  (%d/%d) %s'%(i+1, n_sample, input_sample_alpha_distr_file), end="\t", flush=True)
        profile = generate_plasma_sample_profile_with_given_markers_and_dynamic_alpha_threshold_by_parsing_alpha_value_distribution_file_quick_version(input_sample_alpha_distr_file,
                                                                                                                                                       markers_list,
                                                                                                                                                       marker_type)
        print(datetime.now(), flush=True)
        if i==0:
            # data = csr_matrix((n_sample, len(profile)), dtype=np.single).toarray()
            data = lil_matrix((n_sample, len(profile)), dtype=np.single)
        data[i,:] = profile
    return(data)
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
            data['mat'].append(np.array(list(map(float, items)))[index_of_samples_for_use])
    print('   == begin to transpose the matrix ...', flush=True)
    print('      current time:', datetime.now(), flush=True)
    data['mat'] = np.transpose(np.array(data['mat']))
    print('      current time:', datetime.now(), flush=True)
    print('   == end of transposing the matrix', flush=True)
    return (data)

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
# ongoing
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
