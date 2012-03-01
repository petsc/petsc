
#if !defined(__AIJPTHREAD_H)
#define __AIJPTHREAD_H

typedef struct {
  PetscInt   *rstart;       /* List of starting row number for each thread */
  PetscInt   *nrows;        /* List of nrows for each thread */
  PetscInt   nthreads;      /* Number of threads to use for matrix operations */
  PetscInt   *cpu_affinity; /* CPU affinity of threads */
  PetscInt   *nz;           /* List of number of nonzeros for each thread */
}Mat_SeqAIJPThread;

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
