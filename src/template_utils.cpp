#ifndef TEMPLATE_UTILS_H
#define TEMPLATE_UTILS_H

#include <Rcpp.h>
using namespace Rcpp;

// print vector of any primary data type to the format, e.g., 10469,10471,10484,10489,10493,10497,10525,10542
template <typename T> void print_vec(std::ostream& of, std::vector<T> & v, std::string delimit=",", std::string prefix="") {
	switch (v.size()) {
		case 0:
			return;
		case 1:
			of << prefix << v[0];
			break;
		default: // v.size()>=2
			int i;
			of << prefix;
			for (i=0; i<v.size()-1; i++) of << v[i] << delimit;
			of << v[i];
	}
}

#endif

