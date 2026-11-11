#include "../src/gpu/parameters.h"
#include <iostream>

using namespace gpu_gfn;
static void emit(const integer &a) { std::cout << a.get_str(); }
static void emit(const weight &w) {
    std::cout << '[';
    emit(w.whole);
    std::cout << ',';
    emit(w.fraction);
    std::cout << ']';
}
template <class T> static void emit(const std::vector<T> &a) {
    std::cout << '[';
    for (size_t j = 0; j < a.size(); ++j) {
        if (j)
            std::cout << ',';
        emit(a[j]);
    }
    std::cout << ']';
}
int main() {
    uint32_t base;
    unsigned length, pack, ratio, fields;
    while (std::cin >> base >> length >> pack >> ratio >> fields) {
        std::vector<uint32_t> primes(fields);
        for (uint32_t &p : primes)
            if (!(std::cin >> p))
                return 2;
        try {
            const auto a = admit(base, length, pack, ratio, primes);
            std::cout << "{\"accepted\":true,\"B\":";
            emit(a.radix);
            std::cout << ",\"P\":";
            emit(a.modulus);
            std::cout << ",\"EP\":";
            emit(a.extra_remainder);
            std::cout << ",\"Tmax\":";
            emit(a.maximum_carry);
            std::cout << ",\"groups\":" << a.groups;
            std::cout << ",\"e\":\"" << a.error.get_str() << '"';
            std::cout << ",\"H\":";
            emit(a.cofactors);
            std::cout << ",\"iv\":";
            emit(a.inverses);
            std::cout << ",\"E\":";
            emit(a.remainders);
            std::cout << ",\"quotient\":";
            emit(a.quotient);
            std::cout << ",\"tail\":";
            emit(a.tail);
            std::cout << ",\"extra_quotient\":";
            emit(a.extra_quotient);
            std::cout << ",\"extra_tail\":";
            emit(a.extra_tail);
            std::cout << ",\"bm\":";
            emit(a.radix_montgomery);
            std::cout << ",\"epm\":";
            emit(a.extra_montgomery);
            std::cout << ",\"matrix\":";
            emit(a.matrix);
            std::cout << "}\n";
        } catch (const std::invalid_argument &) {
            std::cout << "{\"accepted\":false}\n";
        }
    }
}
