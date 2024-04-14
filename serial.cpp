#include <cstdlib> 
#include <ctime>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <utility>
#include <stdexcept>
#include <sstream>
#include <chrono>

using namespace std;

void train(double* dataset, double* w) {
	double prev_loss = 0.0;
	double diff = 1.0;
	while (diff > 1e-5) {

		double loss = 0.0;

		for (int i = 0; i < 1000000; i += 128) {
			double* grad = new double[11];
			for (int ii = i; ii < min(i+128, 1000000); ++ii) {
				double y = dataset[ii*12 + 11];
			    double y_hat = 0.0;
				for (int j = 0; j < 11; ++j) {
				    y_hat += dataset[ii*12 + j] * w[j];
			    }
				double diff = y_hat - y;
			    loss += diff * diff;
				for (int j = 0; j < 11; ++j) {
					grad[j] += 2 * diff * dataset[ii*12 + j];
				}
			}

			for (int j = 0; j < 11; ++j) {
				w[j] -= 0.0000001 * (grad[j] / 32);
			}
			delete[] grad;
		}
		loss = loss / 1000000;
		diff = abs(prev_loss - loss);
		prev_loss = loss;
		//cout << "Loss " << loss << endl;
		//cout << "Diff " << diff << endl;
	}
}