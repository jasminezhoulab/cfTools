######### Software Disclaimer #############
# This software is provided free of charge to the academic research community for non-commercial research and educational purposes only. For information on the use for a commercial purpose or by a commercial or for-profit entity, please contact Prof. Xiangong Jasmine Zhou (https://zhoulab.dgsom.ucla.edu/), and Email: xjzhou@mednet.ucla.edu).
#
# CancerDetector is copyrighted by the UCLA. It can be freely used for educational and research purposes by non-profit institutions and U.S. government agencies only. It cannot be used for commercial purposes. In accordance with the terms herein, UCLA grants to LICENSEE and LICENSEE accepts from UCLA, a non-exclusive, non-modifiable, non-transferable license to use the Software for LICENSEE's evaluation and internal research purposes only. LICENSEE recognizes and understands that it shall have no rights whatsoever to distribute the Software unless such rights are granted to LICENSEE by UCLA in a separate agreement.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
#

import sys, os
import numpy as np
np.seterr(divide = 'ignore') ##########RH

def ismember(A, B): # A and B are numpy.array
	return [np.sum(B==a) for a in A]

def loadReadProbilities(fileName):
	markerIdReads = []
	p = []
	with open(fileName) as f:
		next(f) # skip the first header line
		for line in f:
			items = line.rstrip().split('\t')
			markerIdReads.append( int(float(items[0])) )
			p.append( np.array( list(map(float,items[1:])) ) )
	f.close()
	return (np.array(markerIdReads), np.matrix(p))

def grid_search( p, nGrid=1000):
	(N, M) = p.shape # N=#reads, M=2 (#class)
	theta = np.linspace(0, 1, num=nGrid+1)
	theta = np.matrix( np.vstack( (theta, 1-theta) ) )
	obj = np.sum( np.log( p*theta ), axis=0 ) # column sums
	indMax = np.nanargmax( obj )
	theta_best = [float(theta[:,indMax][0]), float(theta[:,indMax][1]) ]
	return ( theta_best, float(obj[:,indMax]) )

def grid_bins( markerIdReads, p ):
	theta_bins = []
	max_obj_bins = []
	marker_ids = np.unique( markerIdReads )
	[N, M] = p.shape
	for i in range(len(marker_ids)):
		marker_id = marker_ids[i]
		p_marker = p[markerIdReads==marker_id, :]
		(theta_best, obj_max) = grid_search( p_marker )
		theta_bins.append( theta_best )
		max_obj_bins.append( obj_max )
	return (np.array(theta_bins), np.array(max_obj_bins), marker_ids)

readProbFile = sys.argv[1]
lambda_ = float( sys.argv[2] )
output_dir = sys.argv[3]
id = sys.argv[4]
if output_dir!="" and id!="":
	output_file = os.path.join(output_dir, id+".tumor_burden.txt")

markerIdReads, p = loadReadProbilities( readProbFile )

# Remove the reads whose probabilities to each class is zero, especially when #class=2, they will lead to NAN in computation and so need to be removed before computation
goodReadsIdx = np.where(np.sum(p, axis=1) !=0)[0]
p = p[goodReadsIdx, :]
markerIdReads = markerIdReads[goodReadsIdx]

# Step 1. infer tumor burden of each marker ...
theta_bins_both, max_obj_bins, marker_ids = grid_bins( markerIdReads, p )
theta_bins = theta_bins_both[:,1] # the 2nd column is theta of normal samples
theta_std = np.std(theta_bins, ddof=1)

# Step 2. infer tumor burden over all markers ...
# after 20 round, the tumor burden should converge
nRound = 20
p_goodmarkers = p
good_marker_ids = marker_ids

n_good_marker_prev = 0
for round in range(nRound):
	thetaVectorUpdateBest, max_objective_value_best = grid_search( p_goodmarkers )
	theta_cutoff_biomarker = thetaVectorUpdateBest[1] - theta_std * lambda_ # lambda_ is a parameter to adjust the normal cfDNA fraction. We suppose the 2nd element of 'thetaVectorUpdateBest' is the fraction of normal cfDNA
	good_marker_ids = marker_ids[theta_bins >= theta_cutoff_biomarker]
	if len(good_marker_ids) != n_good_marker_prev:
		n_good_marker_prev = len(good_marker_ids)
	else:
		break
	good_reads_indexes = np.where( ismember(markerIdReads, good_marker_ids) )[0];
	p_goodmarkers = p[good_reads_indexes, :]

if output_dir!="" and id!="":
	with open(output_file, 'w') as temp_file:
		temp_file.write('%6.4f\t%6.4f\n'%(thetaVectorUpdateBest[0], thetaVectorUpdateBest[1]))

# print("cfDNA tumor burden: %6.4f"%(thetaVectorUpdateBest[0]))
# print("normal cfDNA fraction: %6.4f"%(thetaVectorUpdateBest[1]))
# sys.stdout.write('%6.4f\t%6.4f\n'%(thetaVectorUpdateBest[0], thetaVectorUpdateBest[1]))

