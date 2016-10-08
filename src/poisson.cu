#include "cuda_globals.hpp"

#include <iostream>
#include <fstream>

template<typename TRandGenerator>
__global__ void setup_kernel(Parameters _params, TRandGenerator* _state)
{
  int i;
  for (i = blockIdx.x * blockDim.x + threadIdx.x;
       i < _params.n;
       i += blockDim.x * gridDim.x)
  {
    curand_init(1234, i, 0, &_state[i]);
  }
}

template<typename TRandGenerator, typename T>
__global__ void d_generate_poisson_numbers(Data<T> _data,
                                           Parameters _params,
                                           TRandGenerator* _state);

template<typename TRandGenerator>
__global__ void d_generate_poisson_numbers(Data<unsigned> _data,
                                           Parameters _params,
                                           TRandGenerator* _state)
{
  int i;
  for (i = blockIdx.x * blockDim.x + threadIdx.x;
       i < _params.n;
       i += blockDim.x * gridDim.x)
  {
    TRandGenerator localState = _state[i];
    _data.poisson_numbers_d[i] = curand_poisson(&localState,
                                                _params.lambda);
    _state[i] = localState;
  }
}

/// more efficient
template<typename TRandGenerator>
__global__ void d_generate_poisson_numbers(Data<uint4> _data,
                                           Parameters _params,
                                           TRandGenerator* _state)
{
  int i;
  for (i = blockIdx.x * blockDim.x + threadIdx.x;
       i < _params.n;
       i += blockDim.x * gridDim.x)
  {
    TRandGenerator localState = _state[i];
    _data.poisson_numbers_d[i] = curand_poisson4(&localState,
                                                 _params.lambda);
    _state[i] = localState;
  }
}


float run_cuda( const Parameters& _parameters )
{
  Data<unsigned> data;
  cudaEvent_t custart, cuend;
  dim3 threads(128);
  dim3 blocks(1024);
  float ms=0.f;

  CHECK_CUDA( cudaEventCreate(&custart) );
  CHECK_CUDA( cudaEventCreate(&cuend) );
  CHECK_CUDA( cudaEventRecord(custart) );

  switch(_parameters.api_mode) {
  case 0: // host
  {
    curandGenerator_t gen;
    /* Allocate n unsigned ints on device */
    CHECK_CUDA(cudaMalloc(&data.poisson_numbers_d,
                          _parameters.n * sizeof(unsigned)));

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
  }
  break;

  case 1: // device
  {
    curandState* devStates;
    /* Allocate n unsigned ints on device */
    CHECK_CUDA(cudaMalloc(&data.poisson_numbers_d,
                          _parameters.n * sizeof(unsigned)));
    /* Allocate space for prng states on device */
    CHECK_CUDA(cudaMalloc(&devStates, _parameters.n *
                          sizeof(curandState)));

    setup_kernel<<<blocks, threads>>>(_parameters, devStates);
    d_generate_poisson_numbers<<<blocks, threads>>>(data, _parameters, devStates);
    CHECK_LAST( "Kernel failure.");
    CHECK_CUDA( cudaFree(devStates) );
  }
  break;

  case 2: // device (Philox uint4)
  {
    Data<uint4> data4;
    curandStatePhilox4_32_10_t* devStates;
    Parameters params4 = _parameters;
    params4.n = (_parameters.n+3)/4;
    /* Allocate n unsigned ints on device */
    CHECK_CUDA(cudaMalloc(&data4.poisson_numbers_d,
                          params4.n * sizeof(uint4)));
    /* Allocate space for prng states on device */
    CHECK_CUDA(cudaMalloc(&devStates, params4.n *
                          sizeof(curandStatePhilox4_32_10_t)));

    setup_kernel<<<blocks, threads>>>(params4, devStates);
    d_generate_poisson_numbers<<<blocks, threads>>>(data4, params4, devStates);
    CHECK_LAST( "Kernel failure.");
    CHECK_CUDA( cudaFree(devStates) );

    data.poisson_numbers_d = reinterpret_cast<unsigned*>(data4.poisson_numbers_d);
  }
  break;

  default:
    throw std::runtime_error("Wrong API mode.");
  }

  CHECK_CUDA(cudaEventRecord(cuend));

  CHECK_CUDA( cudaEventSynchronize(cuend) );
  CHECK_CUDA( cudaEventElapsedTime(&ms, custart, cuend) );

  if(_parameters.dump) {

    data.poisson_numbers_h = new unsigned[_parameters.n];
    CHECK_CUDA( cudaMemcpy(data.poisson_numbers_h, data.poisson_numbers_d, _parameters.n*sizeof(unsigned), cudaMemcpyDeviceToHost) );
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
            << "API mode = " << (parameters.api_mode==2?"device API (more efficient w Philox uint4)" :
                                 parameters.api_mode==1?"device API":"host API") << std::endl;

  float ms = run_cuda(parameters);

  std::cout << std::endl << parameters.n << " Poisson numbers with lambda = " << parameters.lambda << std::endl;
  std::cout << " ... generated on device in: " << ms << " ms" << std::endl;
  return 0;
}