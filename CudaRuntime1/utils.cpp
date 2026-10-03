#include <stdio.h>
#include <vector>
#include "utils.h"

size_t calcNumel(const std::vector<int>& v) {
    size_t numel = 1;

    for (int dim : v) {
        numel *= dim;
    }

    return numel;
}