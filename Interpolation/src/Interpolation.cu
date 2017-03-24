/*
 ============================================================================
 Name        : Interpolation.cu
 Author      : Gavin Buffington
 Version     :
 Copyright   : yes
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <cmath>

double linear(double x, double x1, double f1, double x2, double f2);

int main()  {
	std::cout << linear(0.5, 0, 0, 1, 2);
	return 0;
}

double linear(double x, double x1, double f1, double x2, double f2)  {
	return (f2-f1)*(x-x1) / (x2-x1) + f1;
}
