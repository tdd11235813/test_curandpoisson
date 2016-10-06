#include "cuda_globals.hpp"

#include <iostream>
#include <fstream>

template<typename T>
float run_cuda( const Parameters& _parameters )
{
  Data<T> data;
  curandGenerator_t gen;
  cudaEvent_t custart, cuend;
  float ms=0.f;

  data.poisson_numbers_h = new T[_parameters.n];

  CHECK_CUDA( cudaEventCreate(&custart) );
  CHECK_CUDA( cudaEventCreate(&cuend) );

  CHECK_CUDA(curandCreateGenerator(&gen,
                                    CURAND_RNG_PSEUDO_DEFAULT));
  /* Set seed */
  CHECK_CUDA(curandSetPseudoRandomGeneratorSeed(
                gen, 1234ULL));

  /* Allocate n unsigned ints on device */
  CHECK_CUDA(cudaMalloc(&data.poisson_numbers_d,
                       _parameters.n * sizeof(T)));

  CHECK_CUDA(cudaEventRecord(custart));

  /* Generate n unsigned ints on device */
  CHECK_CUDA(curandGeneratePoisson(gen,
                                    data.poisson_numbers_d,
                                    _parameters.n,
                                    _parameters.lambda));
  CHECK_CUDA(cudaEventRecord(cuend));

  CHECK_CUDA( cudaEventSynchronize(cuend) );
  CHECK_CUDA( cudaEventElapsedTime(&ms, custart, cuend) );

  CHECK_CUDA( cudaMemcpy(data.poisson_numbers_h, data.poisson_numbers_d, _parameters.n*sizeof(T), cudaMemcpyDeviceToHost) );

  CHECK_CUDA( cudaEventDestroy(custart) );
  CHECK_CUDA( cudaEventDestroy(cuend) );
  CHECK_CUDA( cudaFree(data.poisson_numbers_d) );


  std::ofstream fs;

  fs.open("dump.csv", std::ofstream::out);
  for( auto i=0; i<_parameters.n; ++i ) {
    fs << data.poisson_numbers_h[i] << std::endl;
  }
  fs.close();
  std::cout <<_parameters.n<< " Poisson numbers dumped to dump.csv." << std::endl;
  delete[] data.poisson_numbers_h;

  return ms;
}

int main(int argc, char** argv)
{
  Parameters parameters;
  if(argc>=2)
    parameters.n = atoi(argv[1]);
  if(argc==3)
    parameters.lambda = atof(argv[2]);
  float ms = run_cuda<unsigned>(parameters);
  std::cout << std::endl << parameters.n << " Poisson numbers with lambda = " << parameters.lambda << std::endl;
  std::cout << " ... generated on device in: " << ms << " ms" << std::endl;
  return 0;
}