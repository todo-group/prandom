#include <iostream>
#include <random>
#include <omp.h>
#include <mpi.h>
#include "prandom.hpp"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int nproc, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
    std::cout << "number of processes = " << nproc << std::endl
              << "number of threads = " << omp_get_max_threads() << std::endl;

  int seed = 1234;
  prandom::mt19937 engines(seed, MPI_COMM_WORLD);
  std::uniform_real_distribution<> dist(0.0, 1.0);

  std::size_t count = (1 << 16);
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

  double ave;
  MPI_Reduce(&s, &ave, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  ave /= nproc;
  if (rank == 0) std::cout << "average = " << ave << std::endl;

  MPI_Finalize();
}
