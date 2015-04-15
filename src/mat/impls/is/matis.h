
#if !defined(__is_h)
#define __is_h

#include <petscsf.h>
#include <petsc/private/matimpl.h>

typedef struct {
  Mat                    A;             /* the local Neumann matrix */
  VecScatter             ctx;           /* update ghost points for matrix vector product */
  Vec                    x,y;           /* work space for ghost values for matrix vector product */
  ISLocalToGlobalMapping mapping;
  int                    rstart,rend;   /* local row ownership */
  PetscBool              pure_neumann;
  PetscSF                sf;
  PetscInt               sf_nroots,sf_nleaves;
  PetscInt               *sf_rootdata,*sf_leafdata;
} Mat_IS;

PETSC_EXTERN PetscErrorCode MatISSetMPIXAIJPreallocation_Private(Mat,Mat,PetscBool);
#endif




