// paralell random number generator

#pragma once

#include <memory>
#include <random>
#ifdef _OPENMP
# include <omp.h>
#endif

namespace prandom {

template<class ENGINE>
class prandom {
public:
  typedef ENGINE engine_type;
  typedef typename engine_type::result_type result_type;

  prandom(result_type s = engine_type::default_seed) { this->seed(s); }
  template<class COMM>
  prandom(result_type s, COMM comm) { this->seed(s, comm); }

  void seed(result_type s = engine_type::default_seed) { this->seed_impl(s, 0); }
  template<class COMM>
  void seed(result_type s, COMM comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    this->seed_impl(s, rank);
  }

  engine_type& operator()(std::size_t tid = 0) { return *engines_[tid]; }

protected:
  void seed_impl(result_type s, int pid) {
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    engines_.resize(num_threads);
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      std::seed_seq seq { s, result_type(pid), result_type(tid) };
      engines_[tid].reset(new engine_type(seq));
    }
#else
    engines_.resize(1);
    std::seed_seq seq { s, pid, 0 };
    engines_[0].reset(new engine_t(seq));
#endif
  }
  
private:
  std::vector<std::shared_ptr<engine_type>> engines_;
};

typedef prandom<std::mt19937> mt19937;
typedef prandom<std::mt19937_64> mt19937_64;

} // end namespace prandom
