#if !defined(_matcliqueimpl_h)
#define _matcliqueimpl_h

#include <petsc/private/matimpl.h>
#include <petscmatelemental.h>

typedef struct {
  MatStructure matstruc;
  PetscBool    CleanUp;        /* Boolean indicating if we call Elue clean step */
  MPI_Comm     comm;           /*  MPI communicator                         */
  PetscInt     cutoff;         /* maximum size of leaf node */
  PetscInt     numDistSeps;    /* number of distributed separators to try */
  PetscInt     numSeqSeps;     /* number of sequential separators to try */

  El::DistSparseMatrix<PetscElemScalar>  *cmat;  /* Elue sparse matrix */
  El::DistMap                            *inverseMap;
  El::DistMultiVec<PetscElemScalar>      *rhs;
} Mat_SparseElemental;

#endif
