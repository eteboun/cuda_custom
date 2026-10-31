#include <iostream>
#include <stdexcept>

#include "tests.cuh"

int main() {
    try {
        int failures = runBenchmarks();

        return failures == 0 ? 0 : 1;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
