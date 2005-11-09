
#include "src/mat/impls/aij/seq/aij.h"

typedef struct {
  PetscInt    nz;     
  PetscInt    m;        /* number of rows */
  PetscInt    rmax;     /* maximum number of columns in a row */
  PetscInt    ncols;    /* number of columns in each row */
  PetscInt    *icols;   /* columns of nonzeros, stored one column at a time */ 
  PetscScalar *acols;   /* values of nonzeros, stored as icols */

  /* We need to keep a pointer to MatAssemblyEnd_AIJ because we 
   * actually want to call this function from within the 
   * MatAssemblyEnd_CRL function.  Similarly, we also need 
   * MatDestroy_AIJ and MatDuplicate_AIJ. */
  PetscErrorCode (*AssemblyEnd)(Mat,MatAssemblyType);
  PetscErrorCode (*MatDestroy)(Mat);
  PetscErrorCode (*MatDuplicate)(Mat,MatDuplicateOption,Mat*);

  /* these are only needed for the parallel case */
  Vec        xwork,fwork;   
  VecScatter xscat;  /* gathers the locally needed part of global vector */
} Mat_CRL;
