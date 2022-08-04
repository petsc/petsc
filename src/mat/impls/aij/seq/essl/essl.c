
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
  Mat_Essl       *essl=(Mat_Essl*)A->data;

  PetscFunctionBegin;
  if (essl->CleanUpESSL) PetscCall(PetscFree4(essl->a,essl->aux,essl->ia,essl->ja));
  PetscCall(PetscFree(A->data));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_Essl(Mat A,Vec b,Vec x)
{
  Mat_Essl       *essl = (Mat_Essl*)A->data;
  PetscScalar    *xx;
  int            nessl,zero = 0;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(A->cmap->n,&nessl));
  PetscCall(VecCopy(b,x));
  PetscCall(VecGetArray(x,&xx));
  dgss(&zero,&nessl,essl->a,essl->ia,essl->ja,&essl->lna,xx,essl->aux,&essl->naux);
  PetscCall(VecRestoreArray(x,&xx));
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorNumeric_Essl(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat_SeqAIJ     *aa  =(Mat_SeqAIJ*)(A)->data;
  Mat_Essl       *essl=(Mat_Essl*)(F)->data;
  int            nessl,i,one = 1;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(A->rmap->n,&nessl));
  /* copy matrix data into silly ESSL data structure (1-based Frotran style) */
  for (i=0; i<A->rmap->n+1; i++) essl->ia[i] = aa->i[i] + 1;
  for (i=0; i<aa->nz; i++) essl->ja[i] = aa->j[i] + 1;

  PetscCall(PetscArraycpy(essl->a,aa->a,aa->nz));

  /* set Essl options */
  essl->iparm[0] = 1;
  essl->iparm[1] = 5;
  essl->iparm[2] = 1;
  essl->iparm[3] = 0;
  essl->rparm[0] = 1.e-12;
  essl->rparm[1] = 1.0;

  PetscCall(PetscOptionsGetReal(NULL,((PetscObject)A)->prefix,"-matessl_lu_threshold",&essl->rparm[1],NULL));

  dgsf(&one,&nessl,&essl->nz,essl->a,essl->ia,essl->ja,&essl->lna,essl->iparm,essl->rparm,essl->oparm,essl->aux,&essl->naux);

  F->ops->solve     = MatSolve_Essl;
  (F)->assembled    = PETSC_TRUE;
  (F)->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorSymbolic_Essl(Mat B,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  Mat_Essl       *essl;
  PetscReal      f = 1.0;

  PetscFunctionBegin;
  essl = (Mat_Essl*)(B->data);

  /* allocate the work arrays required by ESSL */
  f    = info->fill;
  PetscCall(PetscBLASIntCast(a->nz,&essl->nz));
  PetscCall(PetscBLASIntCast((PetscInt)(a->nz*f),&essl->lna));
  PetscCall(PetscBLASIntCast(100 + 10*A->rmap->n,&essl->naux));

  /* since malloc is slow on IBM we try a single malloc */
  PetscCall(PetscMalloc4(essl->lna,&essl->a,essl->naux,&essl->aux,essl->lna,&essl->ia,essl->lna,&essl->ja));

  essl->CleanUpESSL = PETSC_TRUE;

  PetscCall(PetscLogObjectMemory((PetscObject)B,essl->lna*(2*sizeof(int)+sizeof(PetscScalar)) + essl->naux*sizeof(PetscScalar)));

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

.seealso: `PCLU`, `PCFactorSetMatSolverType()`, `MatSolverType`
M*/

PETSC_EXTERN PetscErrorCode MatGetFactor_seqaij_essl(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  Mat_Essl       *essl;

  PetscFunctionBegin;
  PetscCheck(A->cmap->N == A->rmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"matrix must be square");
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&B));
  PetscCall(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,A->rmap->n,A->cmap->n));
  PetscCall(PetscStrallocpy("essl",&((PetscObject)B)->type_name));
  PetscCall(MatSetUp(B));

  PetscCall(PetscNewLog(B,&essl));

  B->data                  = essl;
  B->ops->lufactorsymbolic = MatLUFactorSymbolic_Essl;
  B->ops->destroy          = MatDestroy_Essl;
  B->ops->getinfo          = MatGetInfo_External;

  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_essl));

  B->factortype = MAT_FACTOR_LU;
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&B->preferredordering[MAT_FACTOR_LU]));
  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERESSL,&B->solvertype));

  *F            = B;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_Essl(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERESSL,MATSEQAIJ,          MAT_FACTOR_LU,MatGetFactor_seqaij_essl));
  PetscFunctionReturn(0);
}
