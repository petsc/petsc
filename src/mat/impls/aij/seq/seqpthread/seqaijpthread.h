
#if !defined(__AIJPTHREAD_H)
#define __AIJPTHREAD_H

typedef struct {
  MatScalar *aa;
  PetscInt  *ai;
  PetscInt  *aj;
  PetscInt  *adiag;
  PetscInt  rstart;
  PetscInt  nz;
  PetscScalar *x,*y,*z;
  PetscInt   nrows;
  PetscInt   nonzerorow;
  PetscBool  missing_diag,find_d;
  PetscInt   d;
  PetscInt   nzerodiags;
  PetscInt   *zerodiags;
  InsertMode is;
}Mat_KernelData;

Mat_KernelData *mat_kerneldatap;
Mat_KernelData **mat_pdata;

EXTERN_C_BEGIN
extern PetscErrorCode MatSeqAIJPThreadSetPreallocation_SeqAIJPThread(Mat,PetscInt,const PetscInt*);
EXTERN_C_END

#endif
