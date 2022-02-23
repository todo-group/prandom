// paralell random number generator

#pragma once

#include <memory>
#include <random>
#include <omp.h>

namespace prandom {

template<class ENGINE>
class prandom {
public:
  typedef ENGINE engine_t;
  prandom(int s) { this->seed(s); }
  void seed(int s) {
    int num_threads = omp_get_max_threads();
    engines_.resize(num_threads);
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      std::seed_seq seq { s, 0, tid };
      engines_[tid].reset(new engine_t(seq));
    }
  }
  engine_t& operator()(int tid) { return *engines_[tid]; }
private:
  std::vector<std::shared_ptr<engine_t>> engines_;
};

typedef prandom<std::mt19937> mt19937;
typedef prandom<std::mt19937_64> mt19937_64;

} // end namespace prandom