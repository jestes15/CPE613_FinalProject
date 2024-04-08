#ifndef MEMORY_H_
#define MEMORY_H_

#include <stdlib.h>

double *matrix1D(int np)
{
    double *a;

    a = (double *)calloc(np, sizeof(double));

    return a;
}

double **matrix2D(int nm, int np)
{
    double **m;

    m = (double **)calloc(nm, sizeof(double *));
    for (int i = 0; i < nm; i++)
        m[i] = (double *)calloc(np, sizeof(double));

    return m;
}

void free2D(double **f, int nm)
{
    for (int i = 0; i < nm; i++)
        delete[] f[i];

    delete[] f;
}

#endif /* MEMORY_H_ */
