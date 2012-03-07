
#if !defined(__AIJPTHREAD_H)
#define __AIJPTHREAD_H

typedef struct {
  MatScalar *aa;
  PetscInt  *ai;
  PetscInt  *aj;
  PetscInt  *adiag;
  PetscInt  nz;
  PetscScalar *x,*y,*z;
  PetscInt   nrows;
  PetscInt   nonzerorow;
}Mat_KernelData;

Mat_KernelData *mat_kerneldatap;
Mat_KernelData **mat_pdata;

#endif
