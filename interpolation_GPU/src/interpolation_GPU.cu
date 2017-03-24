/*
 ============================================================================
 Name        : Interpolation_CPU.cu
 Author      : Gavin Buffington
 Version     :
 Copyright   : yes
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <cstdlib>
#include <cmath>
#include <ctime>

double linear(double x, double x1, double f1, double x2, double f2);
int search(double xx, int n, double *x);
int interpolate(int n, double *xx, double *ff, double *x, double *f);

int main()  {
	int n = 4 * 65536;
	int nm1 = n - 1;
	double *x = new double[nm1];
	double *f = new double[nm1];
	double *xx = new double[n];
	double *ff = new double[n];

	double xMin = 0.;
	double xMax = 100.;
	double dx = (xMax - xMin)/(n-1.);

	double r;
	srand (time(NULL));

	for (int i=0; i<n; i++)  {
		x[i] = i * dx;
		f[i] = x[i] * x[i];
	}

	for (int i=0; i<nm1; i++)  {
		x[i] = i * dx;
		f[i] = x[i] * x[i];
		r = dx * ((rand() % 10) - 5.)/20.;
		xx[i] = x[i] + (dx/2.) + r;
//		std::cout << r << "    " << xx[i] << "\n";
	}

	std::cout << "start\n";
	time_t startTime;
	time(&startTime);

	interpolate(n, xx, ff, x, f);

//	for (int i=0; i<nm1; i++)  {
//		std::cout << xx[i] << "\t" << ff[i] << "\t\t" << x[i] << "\t" << f[i] << "\n";
//	}

	time_t stopTime;
	time(&stopTime);
	time_t runTime = difftime(stopTime,startTime);
	std::cout << runTime << " s\n";

	delete [] x;
	delete [] f;
	delete [] xx;
	delete [] ff;

	return 0;
}

double linear(double x, double x1, double f1, double x2, double f2)  {
	return (f2-f1)*(x-x1) / (x2-x1) + f1;
}

int search(double xx, int n, double *x)  {
	int index;
	for (int i=0; i<n; i++)  {
		if (x[i] > xx)  {
			index = i - 1;
			break;
		}
	}
	return index;
}

int interpolate(int n, double *xx, double *ff, double *x, double *f)  {
	int nm1 = n - 1;
	int index;
	for (int i=0; i<nm1; i++)  {
		index = search(xx[i], n, x);
		ff[i] = linear(xx[i], x[index], f[index], x[index+1], f[index+1]);
	}
	return 0;
}
