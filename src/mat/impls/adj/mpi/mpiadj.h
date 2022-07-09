
#if !defined(MPIADJ_H)
#define MPIADJ_H
#include <petsc/private/matimpl.h>
#include <petsc/private/hashsetij.h>

/*
  MATMPIAdj format - Compressed row storage for storing adjacency lists, and possibly weights
                     This is for grid reorderings (to reduce bandwidth)
                     grid partitionings, etc.
*/

typedef struct {
  PetscHSetIJ ht;

  /*
     once the matrix is assembled (either by calling MatAssemblyBegin/End() or MatMPIAdjSetPreallocation() or MatCreateMPIAdj()
     then the data structures below are valid and cannot be changed
  */
  PetscInt    nz;
  PetscInt    *diag;                   /* pointers to diagonal elements, if they exist */
  PetscInt    *i;                      /* pointer to beginning of each row */
  PetscInt    *j;                      /* column values: j + i[k] is start of row k */
  PetscInt    *values;                 /* numerical values */
  PetscBool   useedgeweights;          /* if edge weights are used  */
  PetscBool   symmetric;               /* user indicates the nonzero structure is symmetric */
  PetscBool   freeaij;                 /* free a, i,j at destroy */
  PetscBool   freeaijwithfree;         /* use free() to free i,j instead of PetscFree() */
  PetscScalar *rowvalues;              /* scalar work space for MatGetRow() */
  PetscInt    rowvalues_alloc;
} Mat_MPIAdj;

#endif
