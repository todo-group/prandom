#include <iostream>
#include <random>
#include <omp.h>
#include "prandom.hpp"

int main() {
  std::cout << "number of threads = " << omp_get_max_threads() << std::endl;

  int seed = 1234;
  prandom::mt19937 engines(seed);
  std::uniform_real_distribution<> dist(0.0, 1.0);

  std::size_t count = 1 << 16;
  double s = 0.0;
  #pragma omp parallel reduction(+: s)
  {
    auto& engine = engines(omp_get_thread_num());
    #pragma omp for
    for (std::size_t i = 0; i < count; ++i) {
      s += dist(engine);
    }
  }
  #pragma omp barrier
  s /= count;
  std::cout << "average = " << s << std::endl;
}
