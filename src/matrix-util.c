#include "matrix-util.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SPARSE_MATRIX_DENSITY 0.2

typedef struct {
  unsigned row, column;
  double val;
} mm_struct;

void get_tokens(char *str, unsigned *row, unsigned *column, double *val) {
  char *token = strtok(str, " "), *endptr;
  *row = strtol(token, &endptr, 10);
  assert(endptr != token);
  token = strtok(NULL, " ");
  *column = strtol(token, &endptr, 10);
  assert(endptr != token);
  *val = -1;
  token = strtok(NULL, " ");
}

void get_header(char *str, unsigned *row, unsigned *column, double *val) {
  char *token = strtok(str, " "), *endptr;

  *row = strtol(token, &endptr, 10);
  assert(endptr != token);
  token = strtok(NULL, " ");
  *column = strtol(token, &endptr, 10);
  assert(endptr != token);
  token = strtok(NULL, " ");
  *val = strtof(token, &endptr);
  assert(endptr != token);
  token = strtok(NULL, " ");
}

int mm_struct_cmp(const void *a, const void *b) {
  const mm_struct *aa = (const mm_struct *)a;
  const mm_struct *bb = (const mm_struct *)b;
  if (aa->row != bb->row)
    return aa->row - bb->row;
  else
    return aa->column - bb->column;
}

void mm_to_csr(char *file_name, unsigned **row_ptrs, unsigned **columns,
               double **vals, unsigned *m, unsigned *n, unsigned *val_count) {
  // Open the file
  FILE *fp = fopen(file_name, "r");
  if (fp == NULL) {
    fprintf(stderr, "Error opening file '%s'", file_name);
    exit(1);
  }

  // Parse the file line by line
  mm_struct *mm;
  unsigned index = 0, header = 0;
  char buffer[1024];
  while (fgets(buffer, 1024, fp) != NULL) {
    // Skip comments
    if (buffer[0] == '%')
      continue;

    // Parse header
    if (!header) {
      double val_count_f;
      get_header(buffer, m, n, &val_count_f);
      *val_count = ((unsigned)val_count_f) * 2 + *n;
      index = *n;
      mm = (mm_struct *)calloc(*val_count, sizeof(mm_struct));
      for (unsigned i = 0; i < *n; i++) {
        mm_struct element = {i + 1, i + 1, 0};
        mm[i] = element;
      }
      header = 1;
      continue;
    }

    // Parse line
    unsigned row, column;
    double val;
    get_tokens(buffer, &row, &column, &val);
    mm_struct element = {row, column, val};
    mm_struct element2 = {column, row, val};
    mm[index++] = element;
    mm[index++] = element2;
    mm[row - 1].val += 1;
    mm[column - 1].val += 1;
  }

  // Close the file
  fclose(fp);

  // Sort matrix market structure
  qsort(mm, *val_count, sizeof(mm_struct), mm_struct_cmp);

  // Convert format from MM to CSR
  unsigned current_row = 1;
  unsigned row_val_count = 0;
  *row_ptrs = (unsigned *)calloc(((*m) + 1), sizeof(unsigned));
  *columns = (unsigned *)calloc(*val_count, sizeof(unsigned));
  *vals = (double *)calloc(*val_count, sizeof(double));
  *row_ptrs[0] = 0;
  for (unsigned i = 0; i < (*val_count); i++) {
    if (current_row != mm[i].row) {
      (*row_ptrs)[current_row] = row_val_count;
      current_row = mm[i].row;
    }
    row_val_count++;
    (*columns)[i] = mm[i].column - 1;
    (*vals)[i] = mm[i].val;
  }
  (*row_ptrs)[current_row] = row_val_count;
}
