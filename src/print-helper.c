#include "print-helper.h"

void print_matrix(double *matrix, int size1, int size2) {
  printf("Matrix\n");
  for (int i = 0; i < size1; i++) {
    for (int j = 0; j < size2; j++) {
      printf("%f ", matrix[i * size1 + j]);
    }
    printf("\n");
  }
}

void print_eigen_vals(double *eigen_vals, int size) {
  printf("eigen_vals\n");
  for (int i = 0; i < size; i++) {
    printf("%f\n", eigen_vals[i]);
  }
}

void print_kernel_time(time_struct *time_measure) {
  printf("vec_norm      :- time = %e , Num of itteration = %ld\n",
         time_measure->vec_norm->time, time_measure->vec_norm->no_of_itt);
  printf("vec_dot       :- time = %e , Num of itteration = %ld\n",
         time_measure->vec_dot->time, time_measure->vec_dot->no_of_itt);
  printf("vec_sclr_mul  :- time = %e , Num of itteration = %ld\n",
         time_measure->vec_sclr_mul->time,
         time_measure->vec_sclr_mul->no_of_itt);
  printf("vec_copy      :- time = %e , Num of itteration = %ld\n",
         time_measure->vec_copy->time, time_measure->vec_copy->no_of_itt);
  printf("spmv          :- time = %e , Num of itteration = %ld\n",
         time_measure->spmv->time, time_measure->spmv->no_of_itt);
  printf("calc_w        :- time = %e , Num of itteration = %ld\n",
         time_measure->calc_w->time, time_measure->calc_w->no_of_itt);
  printf("calc_w_init   :- time = %e , Num of itteration = %ld\n",
         time_measure->calc_w_init->time, time_measure->calc_w_init->no_of_itt);
}
