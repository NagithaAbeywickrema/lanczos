# lanczos
Parallel implementations of Lanczos algorithm (Famously used as a sub routine in recursive spectral bisection algorithm)

# Build 
```nompcc ../src/lanczos-nomp.c ../src/main.c ../src/mtx.c ../src/print-helper.c ../src/lanczos-aux.c```

```./a.out --nomp-backend opencl --nomp-platform 2 --nomp-device 0``` 
