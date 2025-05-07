#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <chrono> // Include chrono for timing

using namespace std;
using namespace chrono;

int main() {
    const int SIZE = 10;
    vector<int> data(SIZE);

    // Fill vector with random integers
    srand(time(0));
    for (int i = 0; i < SIZE; ++i) {
        data[i] = rand() % 10000;
    }

    int minVal = data[0];
    int maxVal = data[0];
    long long sum = 0;

    // Measure time for parallel reduction
    auto start = high_resolution_clock::now();
    #pragma omp parallel for reduction(min:minVal)

    for (int i = 0; i < SIZE; ++i) {
        if (data[i] < minVal) minVal = data[i];
    }
    for (int i = 0; i < SIZE; ++i) cout<<data[i]<<" ";
    cout<<"\n"<<endl;
    auto end = high_resolution_clock::now();
    auto minTime = duration_cast<nanoseconds>(end - start).count();

    start = high_resolution_clock::now();
    #pragma omp parallel for reduction(max:maxVal)
    for (int i = 0; i < SIZE; ++i) {
        if (data[i] > maxVal) maxVal = data[i];
    }
    end = high_resolution_clock::now();
    auto maxTime = duration_cast<nanoseconds>(end - start).count();

    start = high_resolution_clock::now();
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < SIZE; ++i) {
        sum += data[i];
    }
    end = high_resolution_clock::now();
    auto sumTime = duration_cast<nanoseconds>(end - start).count();

    start = high_resolution_clock::now();
    double average = static_cast<double>(sum) / SIZE;
    end = high_resolution_clock::now();
    auto avgTime = duration_cast<nanoseconds>(end - start).count();

    // Print results
    cout << "Parallel Reduction Results:\n";
    cout << "Min: " << minVal << " (Time: " << minTime << " ns)\n";
    cout << "Max: " << maxVal << " (Time: " << maxTime << " ns)\n";
    cout << "Sum: " << sum << " (Time: " << sumTime << " ns)\n";
    cout << "Average: " << average << " (Time: " << avgTime << " ns)\n";

    return 0;
}