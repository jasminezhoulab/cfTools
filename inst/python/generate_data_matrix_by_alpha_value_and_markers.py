#
# Python 3.x
#
# Deployed to the working directory: /u/project/xjzhou/wenyuan/data/reads_alpha_distribution_pipeline
#
import sys, gzip
import numpy as np
sys.path.append(".")
from utils_analyze_data_mat import read_lines, parse_markers_with_dynamic_alpha_thresholds, generate_matrix_for_many_plasma_samples_with_given_markers_and_dynamic_alpha_threshold_by_parsing_alpha_value_distribution_files, write_int_matrix_with_row_labels, normalize_matrix_by_rows, count_nonzero_nonnan_max_of_rows, write_lines

debug = False
# debug = True
if debug:
    work_dir1 = '/u/project/xjzhou/wenyuan/data/markers_cancerdetection_alpha_distribution_setdiff_method_pipeline_by_sample_freq_and_readfraction_v8/pipeline_for_lungtumors-normaltissues_by_pairs_vs_normal_plasma'
    work_dir2 = '/u/scratch/w/wenyuan/temp'
    marker_type = 'stringent.hypo.nn0.2.p1.anrange0.5_1.adiff0.5.readfrac+pairedtissuediff0.3-ncplasma0.2.top500.mincpg3'
    input_markers_file_with_dynamic_alpha_threshold = '%s/../lungtumors-normaltissues_by_pairs_t9_vs_normal_plasma/v228_chdf_cov15_bs98.7/stringent.hypo.nn0.2.p1.anrange0.5_1.adiff0.5.readfrac+pairedtissuediff0.3-ncplasma0.2.top500.mincpg3.union_markers.alpha_values_distr.txt.gz'%work_dir1
    input_plasma_samples_list_file = '/u/scratch/w/wenyuan/temp/samples_408/samples.aaa'
    output_raw_matrix_file = '%s/samples.aaa.mat.gz'%work_dir2
    output_normalized_matrix_file = '%s/samples.aaa.cpm.mat.gz'%work_dir2
else:
    # marker_type = sys.argv[1]
    marker_type = "hypo_cfsort"
    input_markers_file_with_dynamic_alpha_threshold = sys.argv[1]
    input_plasma_samples_list_file = sys.argv[2]
    output_raw_matrix_file = sys.argv[3]
    output_normalized_matrix_file = ''
    # if len(sys.argv) == 6:
    #     output_normalized_matrix_file = sys.argv[5]

normalize_type = 'None'  # default is not to use normalization.
if len(output_normalized_matrix_file) > 0:
    all_normalized_types_list = 'cpm.bpm'.split('.')
    for t in all_normalized_types_list:
        if t in output_normalized_matrix_file:
            normalize_type = t
            break
    if normalize_type == 'None':
        sys.stderr.write('Error: argument output_normalized_matrix_file (see below) does not contain [%s].\nExit.\n'%(', '.join(all_normalized_types_list)))


# print('REQUIRE: marker_index in both markers_file and alpha_values_distribution_files are sorted in increasing order of integeter!')

# print('Parse samples list file\n  %s'%input_plasma_samples_list_file, flush=True)
samples_info = {'samples':[], 'alpha_values_files':[]}
# samples_info['samples'] = read_lines(input_plasma_samples_list_file)
samples_info['samples'] = [input_plasma_samples_list_file]

for s in samples_info['samples']:
    samples_info['alpha_values_files'].append( '%s'%(s) )
# print('  #samples: %d'%len(samples_info['samples']), flush=True)

# print('Parse markers list file\n  %s'%input_markers_file_with_dynamic_alpha_threshold, flush=True)
markers_list = parse_markers_with_dynamic_alpha_thresholds(input_markers_file_with_dynamic_alpha_threshold)
# print('  #markers: %d'%len(markers_list['lines']))

# print('Parse alpha value distribution file of each sample', flush=True)
data, total_reads_of_samples = generate_matrix_for_many_plasma_samples_with_given_markers_and_dynamic_alpha_threshold_by_parsing_alpha_value_distribution_files(samples_info['alpha_values_files'],
                                                                                                                                                                markers_list,
                                                                                                                                                                marker_type)

# print('  #original_markers: %d'%len(markers_list['lines']))

# print('Output raw matrix file\n  matrix: %s' % (output_raw_matrix_file), flush=True)
with gzip.open(output_raw_matrix_file, 'wt') as fout:
    write_int_matrix_with_row_labels(data.toarray(), samples_info['samples'], fout)

if len(output_normalized_matrix_file) > 0:
    print('\n==========\nNormalize raw matrix and output ...\n')
    if normalize_type == 'cpb':
        rows_factor = 10**9 / total_reads_of_samples
        print('Normalize row factor: %s (10^9 / total_read_count)' % normalize_type, flush=True)
    elif normalize_type == 'cpm':
        rows_factor = 10**6 / total_reads_of_samples
        print('Normalize row factor: %s (10^6 / total_read_count)' % normalize_type, flush=True)
    else:
        sys.stderr.write('Error: normalize_type should be cpb (count per billion reads) or cpm (count per million reads)\nExit.\n')
        sys.exit(-1)
    rows_factor[rows_factor==np.inf] = 1 # For those rows whose samples' total reads are zero, row_factor is set to one.
    rows_factor = rows_factor.reshape((len(rows_factor), 1)) # Convert to array of size nrow X 1
    data_normalized = normalize_matrix_by_rows(data, rows_factor)

    print('Output normalized matrix file\n  matrix: %s' % (output_normalized_matrix_file), flush=True)
    with gzip.open(output_normalized_matrix_file, 'wt') as fout:
        write_int_matrix_with_row_labels(data_normalized, samples_info['samples'], fout)

# print('Done.')
