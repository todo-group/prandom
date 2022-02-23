// incorrect parallel implementation

#include <iostream>
#include <random>
#include <omp.h>
#include <standards/accumulator.hpp>
#include <standards/timer.hpp>

int main() {
  double max_elapsed = 1.0;
  std::size_t iter = 8;

  int num_threads = omp_get_max_threads();
  int seed = 1234;
  std::seed_seq seq { seed, 0, 0 };
  std::mt19937 engine(seq);
  std::uniform_real_distribution<> dist(0.0, 1.0);

  std::cout << "# num_threads count average error speed\n";

  std::cout << "# num_threads count mean variance elapsed perf perf/thread\n";
  std::size_t count = 1 << 4;
  double elapsed;
  do {
    standards::accumulator res;
    elapsed = 0.0;
    for (std::size_t j = 0; j < iter; ++j) {
      double s = 0.0;
      standards::timer t;
      #pragma omp parallel for reduction(+: s)
      for (std::size_t i = 0; i < count; ++i) {
        s += dist(engine);
      }
      #pragma omp barrier
      elapsed += t.elapsed();
      res << (s / count);
    }
    elapsed /= iter;
    double perf = count / elapsed;
    std::cout << num_threads << ' ' << count << ' ' << res.mean() << ' ' << res.variance() << ' ' << elapsed << ' ' << perf << ' ' << perf / num_threads << std::endl;
    count *= 2;
  } while (count > 0 && elapsed < max_elapsed);
}
