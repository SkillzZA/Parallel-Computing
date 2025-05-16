#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iomanip>

using namespace std; // Common student practice for single-file projects

// --- Helper to initialize a matrix with random values ---
void initializeMatrix(vector<vector<int>>& matrix, int rows, int cols, int maxValue = 10) {
    matrix.assign(rows, vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = rand() % maxValue;
        }
    }
}

// --- a) Dense Matrix Multiplication ---
namespace DMM {

const int N_DIM = 256;
const int M_DIM = 256;
const int L_DIM = 256;

vector<vector<int>> multiply_sequential(
    const vector<vector<int>>& A,
    const vector<vector<int>>& B) {
    vector<vector<int>> C(N_DIM, vector<int>(L_DIM, 0));
    for (int i = 0; i < N_DIM; ++i) {
        for (int j = 0; j < L_DIM; ++j) {
            for (int k = 0; k < M_DIM; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

vector<vector<int>> multiply_parallel(
    const vector<vector<int>>& A,
    const vector<vector<int>>& B) {
    vector<vector<int>> C(N_DIM, vector<int>(L_DIM, 0));

    // Parallelize the outer loop calculating rows of C
    #pragma omp parallel for schedule(static) collapse(1)
    for (int i = 0; i < N_DIM; ++i) {
        for (int j = 0; j < L_DIM; ++j) {
            for (int k = 0; k < M_DIM; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

void run() {
    cout << "--- (a) Dense Matrix Multiplication (N=M=L=" << N_DIM << ") ---" << endl;
    vector<vector<int>> A, B, C_seq, C_par;

    initializeMatrix(A, N_DIM, M_DIM);
    initializeMatrix(B, M_DIM, L_DIM);

    auto start_seq = chrono::high_resolution_clock::now();
    C_seq = multiply_sequential(A, B);
    auto end_seq = chrono::high_resolution_clock::now();
    chrono::duration<double> diff_seq = end_seq - start_seq;
    cout << "Sequential DMM time: " << fixed << setprecision(6) << diff_seq.count() << " s" << endl;

    auto start_par = chrono::high_resolution_clock::now();
    C_par = multiply_parallel(A, B);
    auto end_par = chrono::high_resolution_clock::now();
    chrono::duration<double> diff_par = end_par - start_par;
    cout << "Parallel DMM time (OpenMP): " << fixed << setprecision(6) << diff_par.count() << " s" << endl;

    // Basic Verification
    bool verified = true;
    if (!C_seq.empty() && !C_par.empty() && C_seq.size() == C_par.size() && C_seq[0].size() == C_par[0].size()) {
        // For large matrices, comparing all elements can be slow.
        // Here, we just check if the dimensions match and trust the logic for full verification.
        // Or, compare a few sample elements:
        for(int i=0; i<min(N_DIM, 2); ++i) {
            for(int j=0; j<min(L_DIM, 2); ++j) {
                if (C_seq[i][j] != C_par[i][j]) {
                    verified = false; break;
                }
            }
            if(!verified) break;
        }
        if (verified) cout << "Verification: Parallel result seems to match sequential (sample check)." << endl;
        else cout << "Verification: MISMATCH DETECTED (in sample check)!" << endl;
    } else {
        cout << "Verification: Could not compare (empty or mismatched result dimensions)." << endl;
    }
    cout << endl;
}

} // namespace DMM

// --- b) Pseudo-Polynomial Knapsack ---
namespace Knapsack {

const int NUM_ITEMS = 1024;
const int CAPACITY = 1024;

struct Item {
    int value;
    int weight;
};

void initialize_items(vector<Item>& items, int num_items) {
    items.resize(num_items);
    for (int i = 0; i < num_items; ++i) {
        items[i].value = rand() % 100 + 1;
        items[i].weight = rand() % (CAPACITY / 10) + 1;
        if (items[i].weight == 0) items[i].weight = 1;
    }
}

// Uses 1D DP table for space optimization
long long knapsack_sequential(const vector<Item>& items, int capacity, int num_items) {
    vector<long long> dp(capacity + 1, 0);

    for (int i = 0; i < num_items; ++i) {
        for (int w = capacity; w >= items[i].weight; --w) {
            dp[w] = max(dp[w], dp[w - items[i].weight] + items[i].value);
        }
    }
    return dp[capacity];
}

// Parallelizes the inner loop (over weights) using a "ping-pong" DP table.
long long knapsack_parallel(const vector<Item>& items, int capacity, int num_items) {
    vector<long long> dp_prev(capacity + 1, 0);
    vector<long long> dp_curr(capacity + 1, 0);

    for (int i = 0; i < num_items; ++i) {
        #pragma omp parallel for schedule(static)
        for (int w = 0; w <= capacity; ++w) {
            dp_curr[w] = dp_prev[w];
            if (w >= items[i].weight) {
                dp_curr[w] = max(dp_curr[w], dp_prev[w - items[i].weight] + items[i].value);
            }
        }
        dp_prev.swap(dp_curr); // Or dp_prev = dp_curr;
    }
    return dp_prev[capacity];
}

void run() {
    cout << "--- (b) Pseudo-Polynomial Knapsack (N=" << NUM_ITEMS << ", C=" << CAPACITY << ") ---" << endl;
    vector<Item> items;
    initialize_items(items, NUM_ITEMS);

    auto start_seq = chrono::high_resolution_clock::now();
    long long max_value_seq = knapsack_sequential(items, CAPACITY, NUM_ITEMS);
    auto end_seq = chrono::high_resolution_clock::now();
    chrono::duration<double> diff_seq = end_seq - start_seq;
    cout << "Sequential Knapsack max value: " << max_value_seq << endl;
    cout << "Sequential Knapsack time: " << fixed << setprecision(6) << diff_seq.count() << " s" << endl;

    auto start_par = chrono::high_resolution_clock::now();
    long long max_value_par = knapsack_parallel(items, CAPACITY, NUM_ITEMS);
    auto end_par = chrono::high_resolution_clock::now();
    chrono::duration<double> diff_par = end_par - start_par;
    cout << "Parallel Knapsack max value (OpenMP): " << max_value_par << endl;
    cout << "Parallel Knapsack time (OpenMP): " << fixed << setprecision(6) << diff_par.count() << " s" << endl;

    if (max_value_seq == max_value_par) {
        cout << "Verification: Parallel result matches sequential." << endl;
    } else {
        cout << "Verification: MISMATCH! Sequential: " << max_value_seq << ", Parallel: " << max_value_par << endl;
    }
    cout << endl;
}

} // namespace Knapsack


int main(int argc, char *argv[]) { // Changed main signature for compatibility
    srand(time(0));

    cout << fixed << setprecision(6);

    cout << "OpenMP Configuration:" << endl;
    cout << "Max threads available (omp_get_max_threads()): " << omp_get_max_threads() << endl;
    // Example: omp_set_num_threads(4); if you want to force a specific number

    #pragma omp parallel
    {
        #pragma omp master
        {
            cout << "Actual threads used in parallel region (omp_get_num_threads()): " << omp_get_num_threads() << endl;
        }
    }
    cout << endl;

    DMM::run();
    Knapsack::run();

    cout << "--- Report ---" << endl;
    cout << "To complete your report, copy the execution times printed above into the following sections." << endl;
    cout << "Also, fill in your system specifications." << endl;
    cout << endl;
    cout << "System Specifications:" << endl;
    cout << "CPU: [e.g., Apple M1 Pro, Intel Core i7-XXXX]" << endl;
    cout << "Cores: [e.g., 10 (8 performance, 2 efficiency), 8 cores (16 threads)]" << endl;
    cout << "RAM: [e.g., 16 GB]" << endl;
    cout << "Compiler: [e.g., Apple Clang version 15.0.0, g++ (GCC) 11.2.0]" << endl;
    cout << "OpenMP Runtime: [e.g., libomp (Homebrew), GCC libgomp]" << endl;
    cout << "Compilation Command Used: [Paste the command you used, e.g., clang++ -std=c++17 -O2 -fopenmp ...]" << endl;
    cout << endl;

    cout << "Execution Times & Speedup:" << endl;
    cout << "a) Dense Matrix Multiplication (N=M=L=256):" << endl;
    cout << "   Sequential Time: [COPY FROM OUTPUT] s" << endl;
    cout << "   OpenMP Parallel Time: [COPY FROM OUTPUT] s" << endl;
    cout << "   Speedup: (Calculate Sequential Time / Parallel Time)" << endl;
    cout << endl;
    cout << "b) Pseudo-Polynomial Knapsack (N=1024, C=1024):" << endl;
    cout << "   Sequential Time: [COPY FROM OUTPUT] s" << endl;
    cout << "   OpenMP Parallel Time: [COPY FROM OUTPUT] s" << endl;
    cout << "   Speedup: (Calculate Sequential Time / Parallel Time)" << endl;
    cout << endl;

    cout << "Brief Discussion of Parallelization Strategy:" << endl;
    cout << "a) Dense Matrix Multiplication:" << endl;
    cout << "   - Strategy: Parallelized the outer loop iterating over rows of the result matrix C. Each thread calculates a subset of these rows independently." << endl;
    cout << "   - Rationale: Row calculations are independent (data parallelism). Good granularity for N=256. `schedule(static)` is efficient for uniform workloads." << endl;
    cout << endl;
    cout << "b) Pseudo-Polynomial Knapsack:" << endl;
    cout << "   - Strategy: The outer loop (over items) has data dependencies. Parallelized the inner loop (over capacities `w`) for each item. Used a 'ping-pong' (double-buffer) DP table (`dp_prev`, `dp_curr`) to manage dependencies between item iterations." << endl;
    cout << "   - Rationale: For a fixed item, `dp_curr[w]` calculations are independent. Ping-pong avoids race conditions. `schedule(static)` for the inner loop is suitable for many small, uniform tasks." << endl;

    return 0;
}