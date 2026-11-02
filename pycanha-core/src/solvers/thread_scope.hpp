#pragma once

#include <mkl_service.h>

// The OpenMP runtime MKL is linked with (Intel's). Declared here instead of
// including <omp.h>, which may belong to the compiler's own runtime.
extern "C" {
void omp_set_num_threads(int num_threads);
int omp_get_max_threads();
}

namespace pycanha::solvers::detail {

/// Sets the threads of the calling thread for one MKL call and restores the
/// previous setting afterwards. MKL's LAPACK follows MKL's setting, but the
/// two-level PARDISO factorization follows OpenMP's, which defaults to every
/// logical CPU: both are set. 0 keeps MKL's setting (MKL_NUM_THREADS, or the
/// physical cores) and applies it to OpenMP too.
class ThreadScope {
  public:
    explicit ThreadScope(int threads)
        : _previous_mkl(threads > 0 ? mkl_set_num_threads_local(threads) : -1),
          _previous_openmp(omp_get_max_threads()) {
        omp_set_num_threads(threads > 0 ? threads : mkl_get_max_threads());
    }
    ~ThreadScope() {
        omp_set_num_threads(_previous_openmp);
        if (_previous_mkl >= 0) {
            mkl_set_num_threads_local(_previous_mkl);
        }
    }
    ThreadScope(const ThreadScope&) = delete;
    ThreadScope& operator=(const ThreadScope&) = delete;
    ThreadScope(ThreadScope&&) = delete;
    ThreadScope& operator=(ThreadScope&&) = delete;

  private:
    int _previous_mkl;
    int _previous_openmp;
};

}  // namespace pycanha::solvers::detail
