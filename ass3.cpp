#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <chrono>
#include <climits>

void parallel_min_avg(const std::vector<int>& data, int& min_val, double& avg_val) {
    int local_min = INT_MAX;
    long long sum = 0;

    #pragma omp parallel for reduction(min:local_min) reduction(+:sum)
    for (int i = 0; i < data.size(); i++) {
        if (data[i] < local_min)
            local_min = data[i];
        sum += data[i];
    }

    min_val = local_min;
    avg_val = static_cast<double>(sum) / data.size();
}

void parallel_max_sum(const std::vector<int>& data, int& max_val, long long& sum_val) {
    int local_max = INT_MIN;
    long long total_sum = 0;

    #pragma omp parallel for reduction(max:local_max) reduction(+:total_sum)
    for (int i = 0; i < data.size(); i++) {
        if (data[i] > local_max)
            local_max = data[i];
        total_sum += data[i];
    }

    max_val = local_max;
    sum_val = total_sum;
}

int main() {
    const int data_size = 1000000;
    std::vector<int> data(data_size);

    for (int i = 0; i < data_size; i++) {
        data[i] = rand() % 100;
    }

    int min_val, max_val;
    double avg_val;
    long long sum_val;

    // Measure time for min and average
    auto start_min_avg = std::chrono::high_resolution_clock::now();
    parallel_min_avg(data, min_val, avg_val);
    auto end_min_avg = std::chrono::high_resolution_clock::now();
    auto duration_min_avg = std::chrono::duration_cast<std::chrono::nanoseconds>(end_min_avg - start_min_avg).count();

    // Measure time for max and sum
    auto start_max_sum = std::chrono::high_resolution_clock::now();
    parallel_max_sum(data, max_val, sum_val);
    auto end_max_sum = std::chrono::high_resolution_clock::now();
    auto duration_max_sum = std::chrono::duration_cast<std::chrono::nanoseconds>(end_max_sum - start_max_sum).count();

    // Results
    std::cout << "Minimum value: " << min_val << std::endl;
    std::cout << "Average value: " << avg_val << std::endl;
    std::cout << "Time for min and average: " << duration_min_avg << " ns" << std::endl;

    std::cout << "Maximum value: " << max_val << std::endl;
    std::cout << "Sum value: " << sum_val << std::endl;
    std::cout << "Time for max and sum: " << duration_max_sum << " ns" << std::endl;

    return 0;
}
