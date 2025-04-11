#pragma once

#include <petscvec.h>
#include <petscmat.h>

/*
  Dense linear algebra for vectors and matrices that are _cyclically indexed_.

  For a Vec, this means that it has a fixed size m, but it is for storing data from a sliding window of _history
  indices_.  Rather than rearranging the vector as the window slides, we insert the value v_i for history index i in the
  location v[i % m].

  This has the following effects on linear algebra:

  - Sometimes we are dealing with a window that is smaller than m, e.g. doing y[i % m] <- alpha x[i % m], but not all
    values in [0, m) are active.  We want the linear algebra routines to leave the inactive indices
    untouched.

  - We sometimes need operations that work on the upper triangle of a matrix _with respect to history indices_, i.e.
    operating on the values

      A[i % m, j % m], i <= j

  The functions defined here dispatch linear algebra operations, described by the range of history indices [oldest, next) that
  they act on, to the appropriate backend blas-like (BLAS, cuBLAS, hipBLAS) libraries.
 */

PETSC_INTERN PetscLogEvent AXPBY_Cyc;
PETSC_INTERN PetscLogEvent DMV_Cyc;
PETSC_INTERN PetscLogEvent DSV_Cyc;
PETSC_INTERN PetscLogEvent TRSV_Cyc;
PETSC_INTERN PetscLogEvent GEMV_Cyc;
PETSC_INTERN PetscLogEvent HEMV_Cyc;

/* These methods are intended for small vectors / dense matrices that are either sequential or have all of their entries on the first rank */
// y <- alpha * x + beta * y
PETSC_INTERN PetscErrorCode VecAXPBYCyclic(PetscInt, PetscInt, PetscScalar, Vec, PetscScalar, Vec);
// y <- alpha * (A .* x) + beta * y
PETSC_INTERN PetscErrorCode VecDMVCyclic(PetscBool, PetscInt, PetscInt, PetscScalar, Vec, Vec, PetscScalar, Vec);
// y <- (x ./ A)
PETSC_INTERN PetscErrorCode VecDSVCyclic(PetscBool, PetscInt, PetscInt, Vec, Vec, Vec);
// y <- (triu(A) \ x)
PETSC_INTERN PetscErrorCode MatSeqDenseTRSVCyclic(PetscBool, PetscInt, PetscInt, Mat, Vec, Vec);
// y <- alpha * A * x + beta * y
PETSC_INTERN PetscErrorCode MatSeqDenseGEMVCyclic(PetscBool, PetscInt, PetscInt, PetscScalar, Mat, Vec, PetscScalar, Vec);
// y <- alpha * symm(A) * x + beta * y   [sym(A) = triu(A) + striu(A)^H]
PETSC_INTERN PetscErrorCode MatSeqDenseHEMVCyclic(PetscInt, PetscInt, PetscScalar, Mat, Vec, PetscScalar, Vec);
// A[i,:] <- alpha * x + beta * A[i,:]
PETSC_INTERN PetscErrorCode MatSeqDenseRowAXPBYCyclic(PetscInt, PetscInt, PetscScalar, Vec, PetscScalar, Mat, PetscInt);

/* These methods are intended for for tall matrices of column vectors where we would like to compute products with ranges of the
   vectors.  The column layout should place all columns on the first rank.  Because they may involve MPI communication,
   they rely on implementations from the dense matrix backends */
PETSC_INTERN PetscErrorCode MatMultColumnRange(Mat, Vec, Vec, PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode MatMultAddColumnRange(Mat, Vec, Vec, Vec, PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode MatMultHermitianTransposeColumnRange(Mat, Vec, Vec, PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode MatMultHermitianTransposeAddColumnRange(Mat, Vec, Vec, Vec, PetscInt, PetscInt);
