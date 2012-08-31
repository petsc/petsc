#ifndef _matcliqueimpl_h
#define _matcliqueimpl_h

#include <clique.hpp>
#include <petsc-private/matimpl.h>

#if defined (PETSC_USE_COMPLEX)
typedef cliq::Complex<PetscReal> PetscCliqScalar;
#else
typedef PetscScalar PetscCliqScalar;
#endif

typedef struct {
  MatStructure   matstruc;
  PetscBool      CleanUpClique;  /* Boolean indicating if we call Clique clean step */
  MPI_Comm       cliq_comm;      /* Clique MPI communicator                         */
  PetscInt       cutoff;         /* maximum size of leaf node */
  PetscInt       numDistSeps;    /* number of distributed separators to try */
  PetscInt       numSeqSeps;     /* number of sequential separators to try */

  cliq::DistSparseMatrix<PetscCliqScalar>  *cmat;  /* Clique sparse matrix */
  cliq::DistMap                            *inverseMap;
  cliq::DistSymmInfo                       *info;
  cliq::DistSymmFrontTree<PetscCliqScalar> *frontTree;
  cliq::DistVector<PetscCliqScalar>        *rhs;
  cliq::DistNodalVector<PetscCliqScalar>   *xNodal;

  PetscErrorCode (*Destroy)(Mat);
} Mat_Clique;

#endif
