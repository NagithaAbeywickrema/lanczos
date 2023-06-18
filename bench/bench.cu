#include "bench.h"

void free_(void **p) { free(*p), *p = NULL; }

double *create_host_vec(int size) {
  double *x = tcalloc(double, size);
  for (int i = 0; i < size; i++)
    x[i] = (rand() + 1.0) / RAND_MAX;

  return x;
}

int inc(int i) {
  return (int)(1.1 * i);
  if (i < 1000)
    return i + 1;
  else
    return (int)(1.03 * i);
}
FILE *open_file(char *suffix) {
  char fname[2 * BUFSIZ];
  strncpy(fname, "lanczos", BUFSIZ);
  strncat(fname, "_", 2);
  strncat(fname, suffix, BUFSIZ);
  strncat(fname, ".txt", 5);

  FILE *fp = fopen(fname, "a");
  if (!fp)
    printf("Not found \n");
  return fp;
}