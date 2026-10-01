
#pragma once
#include <iostream>

void check(bool ok, const char* name) {
    ++checks;
    if (!ok) {
        ++failures;
        std::cout << "FAILED: " << name << '\n';
    }
}

void checkNear(double actual, double expected, double tol, const char* name) {
    ++checks;
    if (std::abs(actual - expected) > tol) {
        ++failures;
        std::cout << "FAILED: " << name << '\n';
    }
}