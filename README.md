# Lanczos
Parallel implementations of Lanczos algorithm (Famously used as a sub routine in recursive spectral bisection algorithm)

## Build NOMP lanczos
```mkdir build; cd build; cmake -DENABLE_NOMP=ON -D CMAKE_C_COMPILER=nompcc ..; make; cd -```  

## Build SYCL lanczos
```sh
mkdir build; cd build; cmake -DENABLE_SYCL=ON -D CMAKE_CXX_COMPILER=icpx ..; make; cd -
```
