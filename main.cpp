#include "common.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

// =================
// Helper Functions
// =================

double* read_csv(string filename) {
	double* result = new double[1000000 * 12];

	ifstream myFile(filename);

	if(!myFile.is_open()) throw runtime_error("Could not open file");

	string line, colname;
    double val;

    if(myFile.good()) {
    	getline(myFile, line);
    }

    int i = 0;
    while(getline(myFile, line)) {
    	stringstream ss(line);

    	int j = 0;
    	while(ss >> val) {
    		if (j == 10) {
    			result[i*12 + 10] = 1;
    			result[i*12 + 11] = val / 10;
    			//cout << val / 10 << endl;
    		} else {
    		    result[i*12 + j] = val;
    		    j++;
    		}
    		if(ss.peek() == ',') ss.ignore();
    	}
    	i++;
    }

    myFile.close();

    return result;
}

// ==============
// Main Function
// ==============

int main(int argc, char** argv) {
    double* dataset = read_csv("multiLinearRawData.csv");
    double* w = new double[11];

	for (int i = 0; i < 11; i++) {
		w[i] = 1.0;
	}

    // Algorithm
    auto start_time = std::chrono::steady_clock::now();

    train(dataset, w);

    auto end_time = std::chrono::steady_clock::now();

    double loss = 0.0;
	double y_avg = 0.0;

	for (int i = 0; i < 1000000; ++i) {
		double y = dataset[i*12 + 11];
		double y_hat = 0.0;
		y_avg += y;

		for (int j = 0; j < 11; ++j) {
			y_hat += dataset[i*12 + j] * w[j];
		}

		double diff = y_hat - y;
		loss += diff * diff;
	}

	y_avg /= 1000000;
	double mean = 0.0;
	for (int i = 0; i < 1000000; ++i) {
		double y = dataset[i*12 + 11];
		double diff = y - y_avg;
		mean += diff * diff;
	}
	cout << y_avg << endl;
	cout << loss << endl;
	cout << mean << endl;
	double r_2 = 1 - loss / mean;
	cout << r_2 << endl;

    chrono::duration<double> timediff = end_time - start_time;
    double seconds = timediff.count();
	cout << "Simulation Time = " << seconds << "\n";

	delete[] dataset;
	delete[] w;

	return 0;
}
