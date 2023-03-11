######### Software Disclaimer #############
# This software is provided free of charge to the academic research community for non-commercial research and educational purposes only. For information on the use for a commercial purpose or by a commercial or for-profit entity, please contact Prof. Xiangong Jasmine Zhou (https://zhoulab.dgsom.ucla.edu/), and Email: xjzhou@mednet.ucla.edu).
#
# CancerDetector is copyrighted by the UCLA. It can be freely used for educational and research purposes by non-profit institutions and U.S. government agencies only. It cannot be used for commercial purposes. In accordance with the terms herein, UCLA grants to LICENSEE and LICENSEE accepts from UCLA, a non-exclusive, non-modifiable, non-transferable license to use the Software for LICENSEE's evaluation and internal research purposes only. LICENSEE recognizes and understands that it shall have no rights whatsoever to distribute the Software unless such rights are granted to LICENSEE by UCLA in a separate agreement.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
#

import sys, re, os
from scipy.special import gamma
# import numpy as np

def beta_function(x, y):
	return gamma(x) * gamma(y) / gamma(x+y)

def calc_read_likelihood(methy_states_str, alpha, beta, B):
	num_ones = methy_states_str.count('1')
	num_zeros = methy_states_str.count('0')
	if num_ones==0:
		likelihood_ones=1
		likelihood_zeros = (beta_function(alpha,beta+1)/B)**num_zeros
	elif num_zeros==0:
		likelihood_ones = (beta_function(1+alpha,beta)/B)**num_ones
		likelihood_zeros = 1
	else:
		likelihood_ones = (beta_function(1+alpha,beta)/B)**num_ones
		likelihood_zeros = (beta_function(alpha,beta+1)/B)**num_zeros
	return likelihood_ones * likelihood_zeros

#
# Read marker file
# Each line is a marker bin (or feature). All columns are delimited by TAB. There is one header line.
# Column 1: marker_index, 1-based index. Only marker bins are included, those complementary bins do not appear in this file.
# Column 2: chr
# Column 3: start coordinate of bin (1-base)
# Column 4: end coordinate of bin (1-base). The range of the bin is [start, end)
# Column 5+: paired values for this marker (each column is a class). The first class is tumor and the second is normal class. Each class has a pair of values (alpha and beta shape parameters of a beta distribution), which are delimited by ":"

#For example:
#marker_index	chr	start	end	tumor	normal_plasma
#1	chr1	1	402	3.89692838577676:6.41446579743991	37.6159266954451:12.0867424085961
#2	chr1	101359	101646	2.87841390045233:2.93349657854305	36.4032438100824:3.1110956610252
#3	chr1	102391	102860	2.84669435249341:2.76935335266805	31.3421864311187:3.7799000264534
#4	chr1	113512	114254	3.17828827147782:2.688981888202	59.6678114550981:7.71569189859518
#5	chr1	473277	473599	4.15708661124289:2.71858405253333	3.38883865010887:32.5214534266035
#...
#
# input file is TAB-delimited plain text. The file has a header line
#
def read_paired_values_file_of_bins(file):
	bins2pairedvalues = {}
	bins_index_list = []
	with open(file) as f:
		column_names = next(f).rstrip().split('\t')
		class_names = column_names[4:]
		n_items = len(column_names)
		for c in class_names:
			bins2pairedvalues[c] = {}
		for line in f:
			items = line.rstrip().split('\t')
			bin_index = items[0]
			bins_index_list.append( bin_index )
			for i in range(4, n_items):
				class_name = column_names[i]
				alpha, beta = map(float, re.split(':|,', items[i]))
				if (alpha+beta)>160:
					# gamma(x+y+1) in scipy will be possibly "inf", so we approximate it, by scaling alpha and beta so that alpha+beta=160
					alpha = alpha*160.0/(alpha+beta)
					beta = beta*160.0/(alpha+beta)
				bins2pairedvalues[class_name][bin_index] = [alpha, beta, beta_function(alpha, beta)]
	return bins2pairedvalues, class_names, bins_index_list

reads_binning_file = sys.argv[1]
markers_file = sys.argv[2]
output_dir = sys.argv[3]
id = sys.argv[4]
output_file = os.path.join(output_dir, id+".likelihood.txt")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

bins2pairedvalues, class_names, bins_index_list = read_paired_values_file_of_bins( markers_file )

#
# input reads_binning file format:
# Each line is a read. All columns are delimited by TAB. There is one header line.
# Column 1: marker_index
# Column 2: number of CpG sites in the read
# Column 3: a binary vector of methylation status for all CpG sites in this read (no delimit). For example, 00111100
#
# output file format:
# Each line is a read. All columns are delimited by TAB. There is one header line.
# Column 1: marker index (1-base)
# Columns 2 and 3: each column is a probability value of a class. 
#


# if reads_binning_file == 'stdin':
# 	fin = sys.stdout
# elif reads_binning_file.endswith('gz'): ###commented by Ran
# 	fin = gzip.open(reads_binning_file)
# else:
# 	fin = open(reads_binning_file)

fin = open(reads_binning_file,'r')
# fin.next() # skip header line
lines = fin.readlines()[1:]

with open(output_file, 'w') as temp_file:
	temp_file.write('marker_index\t%s'%('\t'.join(class_names))+"\n")
# print('marker_index\t%s'%('\t'.join(class_names)))
for line in lines:
	# print line.rstrip()
	marker_index, _, methylation_states = line.rstrip().split('\t')
# Calculate the likelihood of each read that belongs to a class
#    the formula is B(x+alpha, 1-x+beta)/B(alpha,beta)
#    where B() is beta function, x is methylation status of a CpG site (0 or 1), alpha and beta are a pair of values for describing a Beta-distribution provided by the input parameter "bins2pairedvalues".
	if marker_index not in bins_index_list: continue
	likelihoods_list = []
	# print '%s\t%s'%(marker_index, methylation_states)
	for c in class_names:
		alpha, beta, B = bins2pairedvalues[c][marker_index]
		likelihood = calc_read_likelihood(methylation_states, alpha, beta, B)
		likelihoods_list.append( likelihood )
		# print '%s: a=%g, b=%g, ll=%g'%(c,alpha,beta,likelihood)
	with open(output_file, 'a') as temp_file:
		temp_file.write('%s\t%s'%(marker_index, '\t'.join(['%g'%ll for ll in likelihoods_list]))+"\n")
	# print('%s\t%s'%(marker_index, '\t'.join(['%g'%ll for ll in likelihoods_list])))
if reads_binning_file != 'stdin':
	fin.close()

