
/*
        Provides an interface to the IBM RS6000 Essl sparse solver

*/
#include <../src/mat/impls/aij/seq/aij.h>

/* #include <essl.h> This doesn't work!  */

PETSC_EXTERN void dgss(int*,int*,double*,int*,int*,int*,double*,double*,int*);
PETSC_EXTERN void dgsf(int*,int*,int*,double*,int*,int*,int*,int*,double*,double*,double*,int*);

typedef struct {
  int         n,nz;
  PetscScalar *a;
  int         *ia;
  int         *ja;
  int         lna;
  int         iparm[5];
  PetscReal   rparm[5];
  PetscReal   oparm[5];
  PetscScalar *aux;
  int         naux;

  PetscBool   CleanUpESSL;
} Mat_Essl;

PetscErrorCode MatDestroy_Essl(Mat A)
{
  PetscErrorCode ierr;
  Mat_Essl       *essl=(Mat_Essl*)A->data;

  PetscFunctionBegin;
  if (essl->CleanUpESSL) {
    ierr = PetscFree4(essl->a,essl->aux,essl->ia,essl->ja);CHKERRQ(ierr);
  }
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_Essl(Mat A,Vec b,Vec x)
{
  Mat_Essl       *essl = (Mat_Essl*)A->data;
  PetscScalar    *xx;
  PetscErrorCode ierr;
  int            nessl,zero = 0;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(A->cmap->n,&nessl);CHKERRQ(ierr);
  ierr = VecCopy(b,x);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  dgss(&zero,&nessl,essl->a,essl->ia,essl->ja,&essl->lna,xx,essl->aux,&essl->naux);
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorNumeric_Essl(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat_SeqAIJ     *aa  =(Mat_SeqAIJ*)(A)->data;
  Mat_Essl       *essl=(Mat_Essl*)(F)->data;
  PetscErrorCode ierr;
  int            nessl,i,one = 1;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(A->rmap->n,&nessl);CHKERRQ(ierr);
  /* copy matrix data into silly ESSL data structure (1-based Frotran style) */
  for (i=0; i<A->rmap->n+1; i++) essl->ia[i] = aa->i[i] + 1;
  for (i=0; i<aa->nz; i++) essl->ja[i] = aa->j[i] + 1;

  ierr = PetscArraycpy(essl->a,aa->a,aa->nz);CHKERRQ(ierr);

  /* set Essl options */
  essl->iparm[0] = 1;
  essl->iparm[1] = 5;
  essl->iparm[2] = 1;
  essl->iparm[3] = 0;
  essl->rparm[0] = 1.e-12;
  essl->rparm[1] = 1.0;

  ierr = PetscOptionsGetReal(NULL,((PetscObject)A)->prefix,"-matessl_lu_threshold",&essl->rparm[1],NULL);CHKERRQ(ierr);

  dgsf(&one,&nessl,&essl->nz,essl->a,essl->ia,essl->ja,&essl->lna,essl->iparm,essl->rparm,essl->oparm,essl->aux,&essl->naux);

  F->ops->solve     = MatSolve_Essl;
  (F)->assembled    = PETSC_TRUE;
  (F)->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorSymbolic_Essl(Mat B,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  Mat_Essl       *essl;
  PetscReal      f = 1.0;

  PetscFunctionBegin;
  essl = (Mat_Essl*)(B->data);

  /* allocate the work arrays required by ESSL */
  f    = info->fill;
  ierr = PetscBLASIntCast(a->nz,&essl->nz);CHKERRQ(ierr);
  ierr = PetscBLASIntCast((PetscInt)(a->nz*f),&essl->lna);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(100 + 10*A->rmap->n,&essl->naux);CHKERRQ(ierr);

  /* since malloc is slow on IBM we try a single malloc */
  ierr = PetscMalloc4(essl->lna,&essl->a,essl->naux,&essl->aux,essl->lna,&essl->ia,essl->lna,&essl->ja);CHKERRQ(ierr);

  essl->CleanUpESSL = PETSC_TRUE;

  ierr = PetscLogObjectMemory((PetscObject)B,essl->lna*(2*sizeof(int)+sizeof(PetscScalar)) + essl->naux*sizeof(PetscScalar));CHKERRQ(ierr);

  B->ops->lufactornumeric = MatLUFactorNumeric_Essl;
  PetscFunctionReturn(0);
}

PetscErrorCode MatFactorGetSolverType_essl(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERESSL;
  PetscFunctionReturn(0);
}

/*MC
  MATSOLVERESSL - "essl" - Provides direct solvers (LU) for sequential matrices
                              via the external package ESSL.

  If ESSL is installed (see the manual for
  instructions on how to declare the existence of external packages),

  Works with MATSEQAIJ matrices

   Level: beginner

.seealso: PCLU, PCFactorSetMatSolverType(), MatSolverType
M*/

PETSC_EXTERN PetscErrorCode MatGetFactor_seqaij_essl(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  PetscErrorCode ierr;
  Mat_Essl       *essl;

  PetscFunctionBegin;
  if (A->cmap->N != A->rmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"matrix must be square");
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  ierr = PetscStrallocpy("essl",&((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);

  ierr = PetscNewLog(B,&essl);CHKERRQ(ierr);

  B->data                  = essl;
  B->ops->lufactorsymbolic = MatLUFactorSymbolic_Essl;
  B->ops->destroy          = MatDestroy_Essl;
  B->ops->getinfo          = MatGetInfo_External;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_essl);CHKERRQ(ierr);

  B->factortype = MAT_FACTOR_LU;
  ierr = PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&B->preferredordering[MAT_FACTOR_LU]);CHKERRQ(ierr);
  ierr = PetscFree(B->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERESSL,&B->solvertype);CHKERRQ(ierr);

  *F            = B;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_Essl(void)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatSolverTypeRegister(MATSOLVERESSL,MATSEQAIJ,          MAT_FACTOR_LU,MatGetFactor_seqaij_essl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
