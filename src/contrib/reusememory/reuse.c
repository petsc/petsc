#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: reuse.c,v 1.1 1997/04/10 14:38:41 bsmith Exp bsmith $";
#endif

#include "src/mat/impls/aij/mpi/mpiaij.h"


#undef __FUNC__  
#define __FUNC__ "MatReleaseValuesMemory_SeqAIJ"
/*
    MatReleaseValuesMemory_SeqAIJ - Frees the memory used to store the sparse matrices
        values. Retains the nonzero structure.

    Input Parameter:
.     mat - the matrix, must be a SEQAIJ matrix


*/
int MatReleaseValuesMemory_SeqAIJ(Mat mat)
{ 
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *) mat->data;

  if (mat->type != MATSEQAIJ) SETERRQ(1,1,"Wrong matrix type");

  /* 
     Assume that matrices with seperate allocation for i, j, and a 
     are ready to free.
  */
  if (!aij->singlemalloc) {
    PetscFree(aij->a);
    aij->a = 0; /* so we don't accidently reuse it */
  /*
     Otherwise we have to allocate new locations for i and j and copy 
     them over 
  */
  } else {
    int *new_i, *new_j;
    new_i = (int *) PetscMalloc( (aij->m+1)*sizeof(int) ); CHKPTRQ(new_i);
    new_j = (int *) PetscMalloc( (aij->nz+1)*sizeof(int) ); CHKPTRQ(new_j);
    PetscMemcpy(new_i,aij->i,(aij->m+1)*sizeof(int));
    PetscMemcpy(new_j,aij->j,(aij->nz)*sizeof(int));
    aij->i = new_i;
    aij->j = new_j;
    PetscFree(aij->a);
    aij->a = 0; /* so we don't accidently reuse it */
    aij->singlemalloc = PETSC_FALSE;
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatReleaseValuesMemory"
/*
    MatReleaseValuesMemory - Frees the memory used to store the sparse matrices
        values. Retains the nonzero structure.

    Input Parameter:
.     mat - the matrix, must be a MPIAIJ matrix

    Notes: You MUST call MatRestoreValuesMemory() before using the matrix 
       again.
*/
int MatReleaseValuesMemory(Mat mat)
{ 
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  int        ierr;

  if (mat->type != MATMPIAIJ) SETERRQ(1,1,"Wrong matrix type");

  ierr = MatReleaseValuesMemory_SeqAIJ(aij->A); CHKERRQ(ierr);
  ierr = MatReleaseValuesMemory_SeqAIJ(aij->B); CHKERRQ(ierr);

  return 0;
}

/* ====================================================================================*/

#undef __FUNC__  
#define __FUNC__ "MatRestoreValuesMemory_SeqAIJ"
/*
    MatRestoreValuesMemory_SeqAIJ - Restores the memory used to store the sparse matrices
        values. Retains the nonzero structure.

    Input Parameter:
.     mat - the matrix, must be a SEQAIJ matrix

*/
int MatRestoreValuesMemory_SeqAIJ(Mat mat)
{ 
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *) mat->data;

  if (mat->type != MATSEQAIJ) SETERRQ(1,1,"Wrong matrix type");

  aij->a = (Scalar *) PetscMalloc( (aij->nz+1)*sizeof(Scalar) ); CHKPTRQ(aij->a);

  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreValuesMemory"
/*
    MatRestoreValuesMemory - Restores the memory used to store the sparse matrices
        values. 

    Input Parameter:
.     mat - the matrix, must be a MPIAIJ matrix

*/
int MatRestoreValuesMemory(Mat mat)
{ 
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  int        ierr;

  if (mat->type != MATMPIAIJ) SETERRQ(1,1,"Wrong matrix type");

  ierr = MatRestoreValuesMemory_SeqAIJ(aij->A); CHKERRQ(ierr);
  ierr = MatRestoreValuesMemory_SeqAIJ(aij->B); CHKERRQ(ierr);

  return 0;
}
<
