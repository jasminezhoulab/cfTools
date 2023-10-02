import sys
# sys.path.append("../analyze_marker_data")
from utils_analyze_markers_data import summarize_mary_file_binary_meth_values_for_distribution_file

debug = False
# debug = True
if debug:
    input_reads_binning_file = './input.data/binary_meth_values_files/debug.binary_meth_values.txt.gz'
    output_alpha_value_distribution_file = './debug.alpha_distr.txt.gz'
else:
    input_reads_binning_file = sys.argv[1]
    output_alpha_value_distribution_file = sys.argv[2]

# print('input: %s'%input_reads_binning_file)
# print('output: %s'%output_alpha_value_distribution_file)
summarize_mary_file_binary_meth_values_for_distribution_file(input_reads_binning_file, output_alpha_value_distribution_file)
# print('py_done')
