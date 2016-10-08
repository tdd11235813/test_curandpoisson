#include "cuda_globals.hpp"

#include <iostream>
#include <fstream>


__global__ void setup_kernel(Parameters _params, curandState* _state)
{
  int i;
  for (i = blockIdx.x * blockDim.x + threadIdx.x;
       i < _params.n;
       i += blockDim.x * gridDim.x)
  {
    curand_init(1234, i, 0, &_state[i]);
  }
}


template<typename T>
__global__ void d_generate_poisson_numbers(Data<T> _data,
                                           Parameters _params,
                                           curandState* _state)
{
  int i;
  for (i = blockIdx.x * blockDim.x + threadIdx.x;
       i < _params.n;
       i += blockDim.x * gridDim.x)
  {
    /* Copy state to local memory for efficiency */
    curandState localState = _state[i];
    /* Simulate queue in time */
    /* Draw number of new customers depending on API */
    _data.poisson_numbers_d[i] = curand_poisson(&localState,
                                                _params.lambda);
    /* Copy state back to global memory */
    _state[i] = localState;
  }
}


template<typename T>
float run_cuda( const Parameters& _parameters )
{
  Data<T> data;
  curandGenerator_t gen;
  cudaEvent_t custart, cuend;
  float ms=0.f;

  CHECK_CUDA( cudaEventCreate(&custart) );
  CHECK_CUDA( cudaEventCreate(&cuend) );
  CHECK_CUDA(cudaEventRecord(custart));

  /* Allocate n unsigned ints on device */
  CHECK_CUDA(cudaMalloc(&data.poisson_numbers_d,
                        _parameters.n * sizeof(T)));


  if(_parameters.api_mode==0) {
    CHECK_CUDA(curandCreateGenerator(&gen,
                                     CURAND_RNG_PSEUDO_DEFAULT));
    /* Set seed */
    CHECK_CUDA(curandSetPseudoRandomGeneratorSeed(
                 gen, 1234ULL));


    /* Generate n unsigned ints on device */
    CHECK_CUDA(curandGeneratePoisson(gen,
                                     data.poisson_numbers_d,
                                     _parameters.n,
                                     _parameters.lambda));
  }else if(_parameters.api_mode==1) {
    curandState *devStates;
    dim3 threads(128);
    dim3 blocks(1024);
    /* Allocate space for prng states on device */
    CHECK_CUDA(cudaMalloc(&devStates, _parameters.n *
                         sizeof(curandState)));

    setup_kernel<<<blocks, threads>>>(_parameters, devStates);
    d_generate_poisson_numbers<<<blocks, threads>>>(data, _parameters, devStates);
    CHECK_CUDA( cudaFree(devStates) );
  }

  CHECK_CUDA(cudaEventRecord(cuend));

  CHECK_CUDA( cudaEventSynchronize(cuend) );
  CHECK_CUDA( cudaEventElapsedTime(&ms, custart, cuend) );

  if(_parameters.dump) {

    data.poisson_numbers_h = new T[_parameters.n];
    CHECK_CUDA( cudaMemcpy(data.poisson_numbers_h, data.poisson_numbers_d, _parameters.n*sizeof(T), cudaMemcpyDeviceToHost) );
    std::ofstream fs;

    fs.open("dump.csv", std::ofstream::out);
    for( auto i=0; i<_parameters.n; ++i ) {
      fs << data.poisson_numbers_h[i] << std::endl;
    }
    fs.close();
    std::cout <<_parameters.n<< " Poisson numbers dumped to dump.csv." << std::endl;
    delete[] data.poisson_numbers_h;
  }

  CHECK_CUDA( cudaEventDestroy(custart) );
  CHECK_CUDA( cudaEventDestroy(cuend) );
  CHECK_CUDA( cudaFree(data.poisson_numbers_d) );

  return ms;
}

int main(int argc, char** argv)
{
  Parameters parameters;
  if(argc>=2)
    parameters.n = atoi(argv[1]);
  if(argc>=3)
    parameters.lambda = atof(argv[2]);
  if(argc>=4)
    parameters.dump = atoi(argv[3]);
  if(argc>=5)
    parameters.api_mode = atoi(argv[4]);
  std::cout << listCudaDevices().str();
  std::cout << "n        = " << parameters.n << std::endl
            << "lambda   = " << parameters.lambda << std::endl
            << "dump?    = " << (parameters.dump?"yes":"no") << std::endl
            << "API mode = " << (parameters.api_mode?"device API":"host API") << std::endl;
  float ms = run_cuda<unsigned>(parameters);
  std::cout << std::endl << parameters.n << " Poisson numbers with lambda = " << parameters.lambda << std::endl;
  std::cout << " ... generated on device in: " << ms << " ms" << std::endl;
  return 0;
}