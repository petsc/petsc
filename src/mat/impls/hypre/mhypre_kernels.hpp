#pragma once

#include <../src/mat/impls/hypre/mhypre.h>

// Zero the specified n rows in rows[] of the hypre CSRMatrix (i, j, a) and replace the diagonal entry with diag
__global__ static void ZeroRows(PetscInt n, const PetscInt rows[], const HYPRE_Int i[], const HYPRE_Int j[], HYPRE_Complex a[], HYPRE_Complex diag)
{
  PetscInt k     = blockDim.x * blockIdx.x + threadIdx.x; // k-th entry in rows[]
  PetscInt c     = blockDim.y * blockIdx.y + threadIdx.y; // c-th nonzero in row rows[k]
  PetscInt gridx = gridDim.x * blockDim.x;
  PetscInt gridy = gridDim.y * blockDim.y;
  for (; k < n; k += gridx) {
    PetscInt r  = rows[k]; // r-th row of the matrix
    PetscInt nz = i[r + 1] - i[r];
    for (; c < nz; c += gridy) {
      if (r == j[i[r] + c]) a[i[r] + c] = diag;
      else a[i[r] + c] = 0.0;
    }
  }
}
