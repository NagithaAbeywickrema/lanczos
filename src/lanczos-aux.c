#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define EPS 1e-12

int tqli(double *eVectors, double *eValues, int n, double *diagonal,
         double *upper, int id) {
  if (n == 0)
    return 0;

  double *d = (double *)calloc(2 * n, sizeof(double));
  double *e = d + n;
  int i;
  for (i = 0; i < n; i++)
    d[i] = diagonal[i];
  for (i = 0; i < n - 1; i++)
    e[i] = upper[i];
  e[n - 1] = 0.0;

  for (i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++)
      eVectors[i * n + j] = 0;
    eVectors[i * n + i] = 1;
  }

  int j, k, l, iter, m;
  for (l = 0; l < n; l++) {
    iter = 0;
    do {
      for (m = l; m < n - 1; m++) {
        double dd = fabs(d[m]) + fabs(d[m + 1]);
        /* Should use a tolerance for this check */
        if (fabs(e[m]) / dd < EPS)
          break;
      }

      if (m != l) {
        if (iter++ == 30) {
          if (id == 0)
            printf("Too many iterations.\n");
          // vec_copy(*eValues, d);
          for (i = 0; i < n; i++)
            eValues[i] = d[i];
          return 1;
        }

        double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
        double r = sqrt(g * g + 1.0);

        g = d[m] - d[l] + e[l] / (g + ((r > 0) ? fabs(r) : -fabs(r)));
        double s = 1.0, c = 1.0, p = 0.0;

        for (i = m - 1; i >= l; i--) {
          double f = s * e[i];
          double b = c * e[i];

          if (fabs(f) >= fabs(g)) {
            c = g / f;
            r = sqrt(c * c + 1.0);
            e[i + 1] = f * r;
            s = 1.0 / r;
            c = c * s;
          } else {
            s = f / g;
            r = sqrt(s * s + 1.0);
            e[i + 1] = g * r;
            c = 1.0 / r;
            s = s * c;
          }

          g = d[i + 1] - p;
          r = (d[i] - g) * s + 2.0 * c * b;
          p = s * r;
          d[i + 1] = g + p;
          g = c * r - b;
          /* Find eigenvectors */
          for (k = 0; k < n; k++) {
            f = eVectors[k * n + i + 1];
            eVectors[k * n + i + 1] = s * eVectors[k * n + i] + c * f;
            eVectors[k * n + i] = c * eVectors[k * n + i] - s * f;
          }
          /* Done with eigenvectors */
        }

        if (r < EPS && i >= l)
          continue;

        d[l] -= p;
        e[l] = g;
        e[m] = 0.0;
      }
    } while (m != l);
  }

  /* Orthnormalize eigenvectors -- Just normalize? */
  for (i = 0; i < n; i++) {
    for (j = 0; j < i; j++) {
      double tmp = eVectors[i * n + j];
      eVectors[i * n + j] = eVectors[j * n + i];
      eVectors[j * n + i] = tmp;
    }
  }

  for (k = 0; k < n; k++) {
    e[k] = 0;
    for (unsigned int i = 0; i < n; i++)
      e[k] += eVectors[k * n + i] * eVectors[k * n + i];
    if (e[k] > 0.0)
      e[k] = sqrt(fabs(e[k]));
    double scale = 1.0 / e[k];
    for (unsigned int i = 0; i < n; i++)
      eVectors[k * n + i] *= scale;
  }

  // vec_copy(*eValues, d);
  for (i = 0; i < n; i++)
    eValues[i] = d[i];

  free(d);

  return 0;
}
