
/*
   Provides an interface to the ML smoothed Aggregation
   Note: Something non-obvious breaks -pc_mg_type ADDITIVE for parallel runs
                                    Jed Brown, see [PETSC #18321, #18449].
*/
#include <petsc-private/pcimpl.h>   /*I "petscpc.h" I*/
#include <../src/ksp/pc/impls/mg/mgimpl.h>                    /*I "petscpcmg.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

#include <math.h>
EXTERN_C_BEGIN
/* HAVE_CONFIG_H flag is required by ML include files */
#if !defined(HAVE_CONFIG_H)
#define HAVE_CONFIG_H
#endif
#include <ml_include.h>
EXTERN_C_END

typedef enum {PCML_NULLSPACE_AUTO,PCML_NULLSPACE_USER,PCML_NULLSPACE_BLOCK,PCML_NULLSPACE_SCALAR} PCMLNullSpaceType;
static const char *const PCMLNullSpaceTypes[] = {"AUTO","USER","BLOCK","SCALAR","PCMLNullSpaceType","PCML_NULLSPACE_",0};

/* The context (data structure) at each grid level */
typedef struct {
  Vec        x,b,r;           /* global vectors */
  Mat        A,P,R;
  KSP        ksp;
} GridCtx;

/* The context used to input PETSc matrix into ML at fine grid */
typedef struct {
  Mat          A;      /* Petsc matrix in aij format */
  Mat          Aloc;   /* local portion of A to be used by ML */
  Vec          x,y;
  ML_Operator  *mlmat;
  PetscScalar  *pwork; /* tmp array used by PetscML_comm() */
} FineGridCtx;

/* The context associates a ML matrix with a PETSc shell matrix */
typedef struct {
  Mat          A;       /* PETSc shell matrix associated with mlmat */
  ML_Operator  *mlmat;  /* ML matrix assorciated with A */
  Vec          y, work;
} Mat_MLShell;

/* Private context for the ML preconditioner */
typedef struct {
  ML                *ml_object;
  ML_Aggregate      *agg_object;
  GridCtx           *gridctx;
  FineGridCtx       *PetscMLdata;
  PetscInt          Nlevels,MaxNlevels,MaxCoarseSize,CoarsenScheme,EnergyMinimization;
  PetscReal         Threshold,DampingFactor,EnergyMinimizationDropTol;
  PetscBool         SpectralNormScheme_Anorm,BlockScaling,EnergyMinimizationCheap,Symmetrize,OldHierarchy,KeepAggInfo,Reusable;
  PetscBool         reuse_interpolation;
  PCMLNullSpaceType nulltype;
  PetscMPIInt       size; /* size of communicator for pc->pmat */
} PC_ML;

#undef __FUNCT__
#define __FUNCT__ "PetscML_getrow"
static int PetscML_getrow(ML_Operator *ML_data, int N_requested_rows, int requested_rows[],int allocated_space, int columns[], double values[], int row_lengths[])
{
  PetscErrorCode ierr;
  PetscInt       m,i,j,k=0,row,*aj;
  PetscScalar    *aa;
  FineGridCtx    *ml=(FineGridCtx*)ML_Get_MyGetrowData(ML_data);
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)ml->Aloc->data;

  ierr = MatGetSize(ml->Aloc,&m,PETSC_NULL); if (ierr) return(0);
  for (i = 0; i<N_requested_rows; i++) {
    row   = requested_rows[i];
    row_lengths[i] = a->ilen[row];
    if (allocated_space < k+row_lengths[i]) return(0);
    if ( (row >= 0) || (row <= (m-1)) ) {
      aj = a->j + a->i[row];
      aa = a->a + a->i[row];
      for (j=0; j<row_lengths[i]; j++){
        columns[k]  = aj[j];
        values[k++] = aa[j];
      }
    }
  }
  return(1);
}

#undef __FUNCT__
#define __FUNCT__ "PetscML_comm"
static PetscErrorCode PetscML_comm(double p[],void *ML_data)
{
  PetscErrorCode ierr;
  FineGridCtx    *ml=(FineGridCtx*)ML_data;
  Mat            A=ml->A;
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscMPIInt    size;
  PetscInt       i,in_length=A->rmap->n,out_length=ml->Aloc->cmap->n;
  PetscScalar    *array;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr);
  if (size == 1) return 0;

  ierr = VecPlaceArray(ml->y,p);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->Mvctx,ml->y,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,ml->y,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecResetArray(ml->y);CHKERRQ(ierr);
  ierr = VecGetArray(a->lvec,&array);CHKERRQ(ierr);
  for (i=in_length; i<out_length; i++){
    p[i] = array[i-in_length];
  }
  ierr = VecRestoreArray(a->lvec,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscML_matvec"
static int PetscML_matvec(ML_Operator *ML_data,int in_length,double p[],int out_length,double ap[])
{
  PetscErrorCode ierr;
  FineGridCtx    *ml=(FineGridCtx*)ML_Get_MyMatvecData(ML_data);
  Mat            A=ml->A, Aloc=ml->Aloc;
  PetscMPIInt    size;
  PetscScalar    *pwork=ml->pwork;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr);
  if (size == 1){
    ierr = VecPlaceArray(ml->x,p);CHKERRQ(ierr);
  } else {
    for (i=0; i<in_length; i++) pwork[i] = p[i];
    PetscML_comm(pwork,ml);
    ierr = VecPlaceArray(ml->x,pwork);CHKERRQ(ierr);
  }
  ierr = VecPlaceArray(ml->y,ap);CHKERRQ(ierr);
  ierr = MatMult(Aloc,ml->x,ml->y);CHKERRQ(ierr);
  ierr = VecResetArray(ml->x);CHKERRQ(ierr);
  ierr = VecResetArray(ml->y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_ML"
static PetscErrorCode MatMult_ML(Mat A,Vec x,Vec y)
{
  PetscErrorCode   ierr;
  Mat_MLShell      *shell;
  PetscScalar      *xarray,*yarray;
  PetscInt         x_length,y_length;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void **)&shell);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xarray);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yarray);CHKERRQ(ierr);
  x_length = shell->mlmat->invec_leng;
  y_length = shell->mlmat->outvec_leng;
  ML_Operator_Apply(shell->mlmat,x_length,xarray,y_length,yarray);
  ierr = VecRestoreArray(x,&xarray);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_ML"
/* Computes y = w + A * x
   It is possible that w == y, but not x == y
*/
static PetscErrorCode MatMultAdd_ML(Mat A,Vec x,Vec w,Vec y)
{
  Mat_MLShell   *shell;
  PetscScalar   *xarray,*yarray;
  PetscInt       x_length,y_length;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A, (void **) &shell);CHKERRQ(ierr);
  if (y == w) {
    if (!shell->work) {
      ierr = VecDuplicate(y, &shell->work);CHKERRQ(ierr);
    }
    ierr = VecGetArray(x,           &xarray);CHKERRQ(ierr);
    ierr = VecGetArray(shell->work, &yarray);CHKERRQ(ierr);
    x_length = shell->mlmat->invec_leng;
    y_length = shell->mlmat->outvec_leng;
    ML_Operator_Apply(shell->mlmat, x_length, xarray, y_length, yarray);
    ierr = VecRestoreArray(x,           &xarray);CHKERRQ(ierr);
    ierr = VecRestoreArray(shell->work, &yarray);CHKERRQ(ierr);
    ierr = VecAXPY(y, 1.0, shell->work);CHKERRQ(ierr);
  } else {
    ierr = VecGetArray(x, &xarray);CHKERRQ(ierr);
    ierr = VecGetArray(y, &yarray);CHKERRQ(ierr);
    x_length = shell->mlmat->invec_leng;
    y_length = shell->mlmat->outvec_leng;
    ML_Operator_Apply(shell->mlmat, x_length, xarray, y_length, yarray);
    ierr = VecRestoreArray(x, &xarray);CHKERRQ(ierr);
    ierr = VecRestoreArray(y, &yarray);CHKERRQ(ierr);
    ierr = VecAXPY(y, 1.0, w);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* newtype is ignored since only handles one case */
#undef __FUNCT__
#define __FUNCT__ "MatConvert_MPIAIJ_ML"
static PetscErrorCode MatConvert_MPIAIJ_ML(Mat A,MatType newtype,MatReuse scall,Mat *Aloc)
{
  PetscErrorCode  ierr;
  Mat_MPIAIJ      *mpimat=(Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ      *mat,*a=(Mat_SeqAIJ*)(mpimat->A)->data,*b=(Mat_SeqAIJ*)(mpimat->B)->data;
  PetscInt        *ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j;
  PetscScalar     *aa=a->a,*ba=b->a,*ca;
  PetscInt        am=A->rmap->n,an=A->cmap->n,i,j,k;
  PetscInt        *ci,*cj,ncols;

  PetscFunctionBegin;
  if (am != an) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"A must have a square diagonal portion, am: %d != an: %d",am,an);

  if (scall == MAT_INITIAL_MATRIX){
    ierr = PetscMalloc((1+am)*sizeof(PetscInt),&ci);CHKERRQ(ierr);
    ci[0] = 0;
    for (i=0; i<am; i++){
      ci[i+1] = ci[i] + (ai[i+1] - ai[i]) + (bi[i+1] - bi[i]);
    }
    ierr = PetscMalloc((1+ci[am])*sizeof(PetscInt),&cj);CHKERRQ(ierr);
    ierr = PetscMalloc((1+ci[am])*sizeof(PetscScalar),&ca);CHKERRQ(ierr);

    k = 0;
    for (i=0; i<am; i++){
      /* diagonal portion of A */
      ncols = ai[i+1] - ai[i];
      for (j=0; j<ncols; j++) {
        cj[k]   = *aj++;
        ca[k++] = *aa++;
      }
      /* off-diagonal portion of A */
      ncols = bi[i+1] - bi[i];
      for (j=0; j<ncols; j++) {
        cj[k]   = an + (*bj); bj++;
        ca[k++] = *ba++;
      }
    }
    if (k != ci[am]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"k: %d != ci[am]: %d",k,ci[am]);

    /* put together the new matrix */
    an = mpimat->A->cmap->n+mpimat->B->cmap->n;
    ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,am,an,ci,cj,ca,Aloc);CHKERRQ(ierr);

    /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
    /* Since these are PETSc arrays, change flags to free them as necessary. */
    mat = (Mat_SeqAIJ*)(*Aloc)->data;
    mat->free_a       = PETSC_TRUE;
    mat->free_ij      = PETSC_TRUE;

    mat->nonew    = 0;
  } else if (scall == MAT_REUSE_MATRIX){
    mat=(Mat_SeqAIJ*)(*Aloc)->data;
    ci = mat->i; cj = mat->j; ca = mat->a;
    for (i=0; i<am; i++) {
      /* diagonal portion of A */
      ncols = ai[i+1] - ai[i];
      for (j=0; j<ncols; j++) *ca++ = *aa++;
      /* off-diagonal portion of A */
      ncols = bi[i+1] - bi[i];
      for (j=0; j<ncols; j++) *ca++ = *ba++;
    }
  } else SETERRQ1(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONG,"Invalid MatReuse %d",(int)scall);
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatDestroy_Shell(Mat);
#undef __FUNCT__
#define __FUNCT__ "MatDestroy_ML"
static PetscErrorCode MatDestroy_ML(Mat A)
{
  PetscErrorCode ierr;
  Mat_MLShell    *shell;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void **)&shell);CHKERRQ(ierr);
  ierr = VecDestroy(&shell->y);CHKERRQ(ierr);
  if (shell->work) {ierr = VecDestroy(&shell->work);CHKERRQ(ierr);}
  ierr = PetscFree(shell);CHKERRQ(ierr);
  ierr = MatDestroy_Shell(A);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatWrapML_SeqAIJ"
static PetscErrorCode MatWrapML_SeqAIJ(ML_Operator *mlmat,MatReuse reuse,Mat *newmat)
{
  struct ML_CSR_MSRdata *matdata = (struct ML_CSR_MSRdata *)mlmat->data;
  PetscErrorCode        ierr;
  PetscInt              m=mlmat->outvec_leng,n=mlmat->invec_leng,*nnz = PETSC_NULL,nz_max;
  PetscInt              *ml_cols=matdata->columns,*ml_rowptr=matdata->rowptr,*aj,i,j,k;
  PetscScalar           *ml_vals=matdata->values,*aa;

  PetscFunctionBegin;
  if (!mlmat->getrow) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"mlmat->getrow = NULL");
  if (m != n){ /* ML Pmat and Rmat are in CSR format. Pass array pointers into SeqAIJ matrix */
    if (reuse){
      Mat_SeqAIJ  *aij= (Mat_SeqAIJ*)(*newmat)->data;
      aij->i = ml_rowptr;
      aij->j = ml_cols;
      aij->a = ml_vals;
    } else {
      /* sort ml_cols and ml_vals */
      ierr = PetscMalloc((m+1)*sizeof(PetscInt),&nnz);
      for (i=0; i<m; i++) {
        nnz[i] = ml_rowptr[i+1] - ml_rowptr[i];
      }
      aj = ml_cols; aa = ml_vals;
      for (i=0; i<m; i++){
        ierr = PetscSortIntWithScalarArray(nnz[i],aj,aa);CHKERRQ(ierr);
        aj += nnz[i]; aa += nnz[i];
      }
      ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,m,n,ml_rowptr,ml_cols,ml_vals,newmat);CHKERRQ(ierr);
      ierr = PetscFree(nnz);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }

  /* ML Amat is in MSR format. Copy its data into SeqAIJ matrix */
  if (reuse) {
    for (nz_max=0,i=0; i<m; i++) nz_max = PetscMax(nz_max,ml_cols[i+1] - ml_cols[i] + 1);
  } else {
    ierr = MatCreate(PETSC_COMM_SELF,newmat);CHKERRQ(ierr);
    ierr = MatSetSizes(*newmat,m,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetType(*newmat,MATSEQAIJ);CHKERRQ(ierr);

    ierr = PetscMalloc((m+1)*sizeof(PetscInt),&nnz);
    nz_max = 1;
    for (i=0; i<m; i++) {
      nnz[i] = ml_cols[i+1] - ml_cols[i] + 1;
      if (nnz[i] > nz_max) nz_max = nnz[i];
    }
    ierr = MatSeqAIJSetPreallocation(*newmat,0,nnz);CHKERRQ(ierr);
  }
  ierr = PetscMalloc2(nz_max,PetscScalar,&aa,nz_max,PetscInt,&aj);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    PetscInt ncols;
    k = 0;
    /* diagonal entry */
    aj[k] = i; aa[k++] = ml_vals[i];
    /* off diagonal entries */
    for (j=ml_cols[i]; j<ml_cols[i+1]; j++){
      aj[k] = ml_cols[j]; aa[k++] = ml_vals[j];
    }
    ncols = ml_cols[i+1] - ml_cols[i] + 1;
    /* sort aj and aa */
    ierr = PetscSortIntWithScalarArray(ncols,aj,aa);CHKERRQ(ierr);
    ierr = MatSetValues(*newmat,1,&i,ncols,aj,aa,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscFree2(aa,aj);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatWrapML_SHELL"
static PetscErrorCode MatWrapML_SHELL(ML_Operator *mlmat,MatReuse reuse,Mat *newmat)
{
  PetscErrorCode ierr;
  PetscInt       m,n;
  ML_Comm        *MLcomm;
  Mat_MLShell    *shellctx;

  PetscFunctionBegin;
  m = mlmat->outvec_leng;
  n = mlmat->invec_leng;

  if (reuse){
    ierr = MatShellGetContext(*newmat,(void **)&shellctx);CHKERRQ(ierr);
    shellctx->mlmat = mlmat;
    PetscFunctionReturn(0);
  }

  MLcomm = mlmat->comm;
  ierr = PetscNew(Mat_MLShell,&shellctx);CHKERRQ(ierr);
  ierr = MatCreateShell(MLcomm->USR_comm,m,n,PETSC_DETERMINE,PETSC_DETERMINE,shellctx,newmat);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*newmat,MATOP_MULT,(void(*)(void))MatMult_ML);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*newmat,MATOP_MULT_ADD,(void(*)(void))MatMultAdd_ML);CHKERRQ(ierr);
  shellctx->A         = *newmat;
  shellctx->mlmat     = mlmat;
  shellctx->work      = PETSC_NULL;
  ierr = VecCreate(MLcomm->USR_comm,&shellctx->y);CHKERRQ(ierr);
  ierr = VecSetSizes(shellctx->y,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(shellctx->y);CHKERRQ(ierr);
  (*newmat)->ops->destroy = MatDestroy_ML;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatWrapML_MPIAIJ"
static PetscErrorCode MatWrapML_MPIAIJ(ML_Operator *mlmat,MatReuse reuse,Mat *newmat)
{
  struct ML_CSR_MSRdata *matdata = (struct ML_CSR_MSRdata *)mlmat->data;
  PetscInt              *ml_cols=matdata->columns,*aj;
  PetscScalar           *ml_vals=matdata->values,*aa;
  PetscErrorCode        ierr;
  PetscInt              i,j,k,*gordering;
  PetscInt              m=mlmat->outvec_leng,n,nz_max,row;
  Mat                   A;

  PetscFunctionBegin;
  if (!mlmat->getrow) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"mlmat->getrow = NULL");
  n = mlmat->invec_leng;
  if (m != n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"m %d must equal to n %d",m,n);

  if (reuse) {
    A = *newmat;
    for (nz_max=0,i=0; i<m; i++) nz_max = PetscMax(nz_max,ml_cols[i+1] - ml_cols[i] + 1);
  } else {
    PetscInt *nnzA,*nnzB,*nnz;
    ierr = MatCreate(mlmat->comm->USR_comm,&A);CHKERRQ(ierr);
    ierr = MatSetSizes(A,m,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetType(A,MATMPIAIJ);CHKERRQ(ierr);
    ierr = PetscMalloc3(m,PetscInt,&nnzA,m,PetscInt,&nnzB,m,PetscInt,&nnz);CHKERRQ(ierr);

    nz_max = 0;
    for (i=0; i<m; i++){
      nnz[i] = ml_cols[i+1] - ml_cols[i] + 1;
      if (nz_max < nnz[i]) nz_max = nnz[i];
      nnzA[i] = 1; /* diag */
      for (j=ml_cols[i]; j<ml_cols[i+1]; j++){
        if (ml_cols[j] < m) nnzA[i]++;
      }
      nnzB[i] = nnz[i] - nnzA[i];
    }
    ierr = MatMPIAIJSetPreallocation(A,0,nnzA,0,nnzB);CHKERRQ(ierr);
    ierr = PetscFree3(nnzA,nnzB,nnz);
  }

  /* insert mat values -- remap row and column indices */
  nz_max++;
  ierr = PetscMalloc2(nz_max,PetscScalar,&aa,nz_max,PetscInt,&aj);CHKERRQ(ierr);
  /* create global row numbering for a ML_Operator */
  ML_build_global_numbering(mlmat,&gordering,"rows");
  for (i=0; i<m; i++) {
    PetscInt ncols;
    row = gordering[i];
    k = 0;
    /* diagonal entry */
    aj[k] = row; aa[k++] = ml_vals[i];
    /* off diagonal entries */
    for (j=ml_cols[i]; j<ml_cols[i+1]; j++){
      aj[k] = gordering[ml_cols[j]]; aa[k++] = ml_vals[j];
    }
    ncols = ml_cols[i+1] - ml_cols[i] + 1;
    ierr = MatSetValues(A,1,&row,ncols,aj,aa,INSERT_VALUES);CHKERRQ(ierr);
  }
  ML_free(gordering);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *newmat = A;

  ierr = PetscFree2(aa,aj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "PCReset_ML"
PetscErrorCode PCReset_ML(PC pc)
{
  PetscErrorCode  ierr;
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_ML           *pc_ml = (PC_ML*)mg->innerctx;
  PetscInt        level,fine_level=pc_ml->Nlevels-1;

  PetscFunctionBegin;
  ML_Aggregate_Destroy(&pc_ml->agg_object);
  ML_Destroy(&pc_ml->ml_object);

  if (pc_ml->PetscMLdata) {
    ierr = PetscFree(pc_ml->PetscMLdata->pwork);CHKERRQ(ierr);
    ierr = MatDestroy(&pc_ml->PetscMLdata->Aloc);CHKERRQ(ierr);
    ierr = VecDestroy(&pc_ml->PetscMLdata->x);CHKERRQ(ierr);
    ierr = VecDestroy(&pc_ml->PetscMLdata->y);CHKERRQ(ierr);
  }
  ierr = PetscFree(pc_ml->PetscMLdata);CHKERRQ(ierr);

  if (pc_ml->gridctx) {
    for (level=0; level<fine_level; level++){
      if (pc_ml->gridctx[level].A){ierr = MatDestroy(&pc_ml->gridctx[level].A);CHKERRQ(ierr);}
      if (pc_ml->gridctx[level].P){ierr = MatDestroy(&pc_ml->gridctx[level].P);CHKERRQ(ierr);}
      if (pc_ml->gridctx[level].R){ierr = MatDestroy(&pc_ml->gridctx[level].R);CHKERRQ(ierr);}
      if (pc_ml->gridctx[level].x){ierr = VecDestroy(&pc_ml->gridctx[level].x);CHKERRQ(ierr);}
      if (pc_ml->gridctx[level].b){ierr = VecDestroy(&pc_ml->gridctx[level].b);CHKERRQ(ierr);}
      if (pc_ml->gridctx[level+1].r){ierr = VecDestroy(&pc_ml->gridctx[level+1].r);CHKERRQ(ierr);}
    }
  }
  ierr = PetscFree(pc_ml->gridctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   PCSetUp_ML - Prepares for the use of the ML preconditioner
                    by setting data structures and options.

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
extern PetscErrorCode PCSetFromOptions_MG(PC);
extern PetscErrorCode PCReset_MG(PC);

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_ML"
PetscErrorCode PCSetUp_ML(PC pc)
{
  PetscErrorCode  ierr;
  PetscMPIInt     size;
  FineGridCtx     *PetscMLdata;
  ML              *ml_object;
  ML_Aggregate    *agg_object;
  ML_Operator     *mlmat;
  PetscInt        nlocal_allcols,Nlevels,mllevel,level,level1,m,fine_level,bs;
  Mat             A,Aloc;
  GridCtx         *gridctx;
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_ML           *pc_ml = (PC_ML*)mg->innerctx;
  PetscBool       isSeq, isMPI;
  KSP             smoother;
  PC              subpc;
  PetscInt        mesh_level, old_mesh_level;

  PetscFunctionBegin;
  A = pc->pmat;
  ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr);

  if (pc->setupcalled) {
    if (pc->flag == SAME_NONZERO_PATTERN && pc_ml->reuse_interpolation) {
      /*
       Reuse interpolaton instead of recomputing aggregates and updating the whole hierarchy. This is less expensive for
       multiple solves in which the matrix is not changing too quickly.
       */
      ml_object = pc_ml->ml_object;
      gridctx = pc_ml->gridctx;
      Nlevels = pc_ml->Nlevels;
      fine_level = Nlevels - 1;
      gridctx[fine_level].A = A;

      ierr = PetscObjectTypeCompare((PetscObject) A, MATSEQAIJ, &isSeq);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject) A, MATMPIAIJ, &isMPI);CHKERRQ(ierr);
      if (isMPI){
        ierr = MatConvert_MPIAIJ_ML(A,PETSC_NULL,MAT_INITIAL_MATRIX,&Aloc);CHKERRQ(ierr);
      } else if (isSeq) {
        Aloc = A;
        ierr = PetscObjectReference((PetscObject)Aloc);CHKERRQ(ierr);
      } else SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONG, "Matrix type '%s' cannot be used with ML. ML can only handle AIJ matrices.",((PetscObject)A)->type_name);

      ierr = MatGetSize(Aloc,&m,&nlocal_allcols);CHKERRQ(ierr);
      PetscMLdata = pc_ml->PetscMLdata;
      ierr = MatDestroy(&PetscMLdata->Aloc);CHKERRQ(ierr);
      PetscMLdata->A    = A;
      PetscMLdata->Aloc = Aloc;
      ML_Init_Amatrix(ml_object,0,m,m,PetscMLdata);
      ML_Set_Amatrix_Matvec(ml_object,0,PetscML_matvec);

      mesh_level = ml_object->ML_finest_level;
      while (ml_object->SingleLevel[mesh_level].Rmat->to) {
        old_mesh_level = mesh_level;
        mesh_level = ml_object->SingleLevel[mesh_level].Rmat->to->levelnum;

        /* clean and regenerate A */
        mlmat = &(ml_object->Amat[mesh_level]);
        ML_Operator_Clean(mlmat);
        ML_Operator_Init(mlmat,ml_object->comm);
        ML_Gen_AmatrixRAP(ml_object, old_mesh_level, mesh_level);
      }

      level = fine_level - 1;
      if (size == 1) { /* convert ML P, R and A into seqaij format */
        for (mllevel=1; mllevel<Nlevels; mllevel++){
          mlmat = &(ml_object->Amat[mllevel]);
          ierr = MatWrapML_SeqAIJ(mlmat,MAT_REUSE_MATRIX,&gridctx[level].A);CHKERRQ(ierr);
          level--;
        }
      } else { /* convert ML P and R into shell format, ML A into mpiaij format */
        for (mllevel=1; mllevel<Nlevels; mllevel++){
          mlmat  = &(ml_object->Amat[mllevel]);
          ierr = MatWrapML_MPIAIJ(mlmat,MAT_REUSE_MATRIX,&gridctx[level].A);CHKERRQ(ierr);
          level--;
        }
      }

      for (level=0; level<fine_level; level++) {
        if (level > 0){
          ierr = PCMGSetResidual(pc,level,PCMGDefaultResidual,gridctx[level].A);CHKERRQ(ierr);
        }
        ierr = KSPSetOperators(gridctx[level].ksp,gridctx[level].A,gridctx[level].A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      }
      ierr = PCMGSetResidual(pc,fine_level,PCMGDefaultResidual,gridctx[fine_level].A);CHKERRQ(ierr);
      ierr = KSPSetOperators(gridctx[fine_level].ksp,gridctx[level].A,gridctx[fine_level].A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);

      ierr = PCSetUp_MG(pc);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    } else {
      /* since ML can change the size of vectors/matrices at any level we must destroy everything */
      ierr = PCReset_ML(pc);CHKERRQ(ierr);
      ierr = PCReset_MG(pc);CHKERRQ(ierr);
    }
  }

  /* setup special features of PCML */
  /*--------------------------------*/
  /* covert A to Aloc to be used by ML at fine grid */
  pc_ml->size = size;
  ierr = PetscObjectTypeCompare((PetscObject) A, MATSEQAIJ, &isSeq);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) A, MATMPIAIJ, &isMPI);CHKERRQ(ierr);
  if (isMPI){
    ierr = MatConvert_MPIAIJ_ML(A,PETSC_NULL,MAT_INITIAL_MATRIX,&Aloc);CHKERRQ(ierr);
  } else if (isSeq) {
    Aloc = A;
    ierr = PetscObjectReference((PetscObject)Aloc);CHKERRQ(ierr);
  } else SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONG, "Matrix type '%s' cannot be used with ML. ML can only handle AIJ matrices.",((PetscObject)A)->type_name);

  /* create and initialize struct 'PetscMLdata' */
  ierr = PetscNewLog(pc,FineGridCtx,&PetscMLdata);CHKERRQ(ierr);
  pc_ml->PetscMLdata = PetscMLdata;
  ierr = PetscMalloc((Aloc->cmap->n+1)*sizeof(PetscScalar),&PetscMLdata->pwork);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_SELF,&PetscMLdata->x);CHKERRQ(ierr);
  ierr = VecSetSizes(PetscMLdata->x,Aloc->cmap->n,Aloc->cmap->n);CHKERRQ(ierr);
  ierr = VecSetType(PetscMLdata->x,VECSEQ);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_SELF,&PetscMLdata->y);CHKERRQ(ierr);
  ierr = VecSetSizes(PetscMLdata->y,A->rmap->n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(PetscMLdata->y,VECSEQ);CHKERRQ(ierr);
  PetscMLdata->A    = A;
  PetscMLdata->Aloc = Aloc;

  /* create ML discretization matrix at fine grid */
  /* ML requires input of fine-grid matrix. It determines nlevels. */
  ierr = MatGetSize(Aloc,&m,&nlocal_allcols);CHKERRQ(ierr);
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  ML_Create(&ml_object,pc_ml->MaxNlevels);
  ML_Comm_Set_UsrComm(ml_object->comm,((PetscObject)A)->comm);
  pc_ml->ml_object = ml_object;
  ML_Init_Amatrix(ml_object,0,m,m,PetscMLdata);
  ML_Set_Amatrix_Getrow(ml_object,0,PetscML_getrow,PetscML_comm,nlocal_allcols);
  ML_Set_Amatrix_Matvec(ml_object,0,PetscML_matvec);

  ML_Set_Symmetrize(ml_object,pc_ml->Symmetrize ? ML_YES : ML_NO);

  /* aggregation */
  ML_Aggregate_Create(&agg_object);
  pc_ml->agg_object = agg_object;

  {
    MatNullSpace mnull;
    ierr = MatGetNearNullSpace(A,&mnull);CHKERRQ(ierr);
    if (pc_ml->nulltype == PCML_NULLSPACE_AUTO) {
      if (mnull) pc_ml->nulltype = PCML_NULLSPACE_USER;
      else if (bs > 1) pc_ml->nulltype = PCML_NULLSPACE_BLOCK;
      else pc_ml->nulltype = PCML_NULLSPACE_SCALAR;
    }
    switch (pc_ml->nulltype) {
    case PCML_NULLSPACE_USER: {
      PetscScalar *nullvec;
      const PetscScalar *v;
      PetscBool has_const;
      PetscInt i,j,mlocal,nvec,M;
      const Vec *vecs;
      if (!mnull) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_USER,"Must provide explicit null space using MatSetNearNullSpace() to use user-specified null space");
      ierr = MatGetSize(A,&M,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatGetLocalSize(Aloc,&mlocal,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatNullSpaceGetVecs(mnull,&has_const,&nvec,&vecs);CHKERRQ(ierr);
      ierr = PetscMalloc((nvec+!!has_const)*mlocal*sizeof(*nullvec),&nullvec);CHKERRQ(ierr);
      if (has_const) for (i=0; i<mlocal; i++) nullvec[i] = 1.0/M;
      for (i=0; i<nvec; i++) {
        ierr = VecGetArrayRead(vecs[i],&v);CHKERRQ(ierr);
        for (j=0; j<mlocal; j++) nullvec[(i+!!has_const)*mlocal + j] = v[j];
        ierr = VecRestoreArrayRead(vecs[i],&v);CHKERRQ(ierr);
      }
      ierr = ML_Aggregate_Set_NullSpace(agg_object,bs,nvec+!!has_const,nullvec,mlocal);CHKERRQ(ierr);
      ierr = PetscFree(nullvec);CHKERRQ(ierr);
    } break;
    case PCML_NULLSPACE_BLOCK:
      ierr = ML_Aggregate_Set_NullSpace(agg_object,bs,bs,0,0);CHKERRQ(ierr);
      break;
    case PCML_NULLSPACE_SCALAR:
      break;
    default: SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Unknown null space type");
    }
  }
  ML_Aggregate_Set_MaxCoarseSize(agg_object,pc_ml->MaxCoarseSize);
  /* set options */
  switch (pc_ml->CoarsenScheme) {
  case 1:
    ML_Aggregate_Set_CoarsenScheme_Coupled(agg_object);break;
  case 2:
    ML_Aggregate_Set_CoarsenScheme_MIS(agg_object);break;
  case 3:
    ML_Aggregate_Set_CoarsenScheme_METIS(agg_object);break;
  }
  ML_Aggregate_Set_Threshold(agg_object,pc_ml->Threshold);
  ML_Aggregate_Set_DampingFactor(agg_object,pc_ml->DampingFactor);
  if (pc_ml->SpectralNormScheme_Anorm){
    ML_Set_SpectralNormScheme_Anorm(ml_object);
  }
  agg_object->keep_agg_information      = (int)pc_ml->KeepAggInfo;
  agg_object->keep_P_tentative          = (int)pc_ml->Reusable;
  agg_object->block_scaled_SA           = (int)pc_ml->BlockScaling;
  agg_object->minimizing_energy         = (int)pc_ml->EnergyMinimization;
  agg_object->minimizing_energy_droptol = (double)pc_ml->EnergyMinimizationDropTol;
  agg_object->cheap_minimizing_energy   = (int)pc_ml->EnergyMinimizationCheap;

  if (pc_ml->OldHierarchy) {
    Nlevels = ML_Gen_MGHierarchy_UsingAggregation(ml_object,0,ML_INCREASING,agg_object);
  } else {
    Nlevels = ML_Gen_MultiLevelHierarchy_UsingAggregation(ml_object,0,ML_INCREASING,agg_object);
  }
  if (Nlevels<=0) SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Nlevels %d must > 0",Nlevels);
  pc_ml->Nlevels = Nlevels;
  fine_level = Nlevels - 1;

  ierr = PCMGSetLevels(pc,Nlevels,PETSC_NULL);CHKERRQ(ierr);
  /* set default smoothers */
  for (level=1; level<=fine_level; level++){
    if (size == 1){
      ierr = PCMGGetSmoother(pc,level,&smoother);CHKERRQ(ierr);
      ierr = KSPSetType(smoother,KSPRICHARDSON);CHKERRQ(ierr);
      ierr = KSPGetPC(smoother,&subpc);CHKERRQ(ierr);
      ierr = PCSetType(subpc,PCSOR);CHKERRQ(ierr);
    } else {
      ierr = PCMGGetSmoother(pc,level,&smoother);CHKERRQ(ierr);
      ierr = KSPSetType(smoother,KSPRICHARDSON);CHKERRQ(ierr);
      ierr = KSPGetPC(smoother,&subpc);CHKERRQ(ierr);
      ierr = PCSetType(subpc,PCSOR);CHKERRQ(ierr);
    }
  }
  ierr = PetscObjectOptionsBegin((PetscObject)pc);CHKERRQ(ierr);
  ierr = PCSetFromOptions_MG(pc);CHKERRQ(ierr); /* should be called in PCSetFromOptions_ML(), but cannot be called prior to PCMGSetLevels() */
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscMalloc(Nlevels*sizeof(GridCtx),&gridctx);CHKERRQ(ierr);
  pc_ml->gridctx = gridctx;

  /* wrap ML matrices by PETSc shell matrices at coarsened grids.
     Level 0 is the finest grid for ML, but coarsest for PETSc! */
  gridctx[fine_level].A = A;

  level = fine_level - 1;
  if (size == 1){ /* convert ML P, R and A into seqaij format */
    for (mllevel=1; mllevel<Nlevels; mllevel++){
      mlmat = &(ml_object->Pmat[mllevel]);
      ierr  = MatWrapML_SeqAIJ(mlmat,MAT_INITIAL_MATRIX,&gridctx[level].P);CHKERRQ(ierr);
      mlmat = &(ml_object->Rmat[mllevel-1]);
      ierr  = MatWrapML_SeqAIJ(mlmat,MAT_INITIAL_MATRIX,&gridctx[level].R);CHKERRQ(ierr);

      mlmat = &(ml_object->Amat[mllevel]);
      ierr  = MatWrapML_SeqAIJ(mlmat,MAT_INITIAL_MATRIX,&gridctx[level].A);CHKERRQ(ierr);
      level--;
    }
  } else { /* convert ML P and R into shell format, ML A into mpiaij format */
    for (mllevel=1; mllevel<Nlevels; mllevel++){
      mlmat  = &(ml_object->Pmat[mllevel]);
      ierr = MatWrapML_SHELL(mlmat,MAT_INITIAL_MATRIX,&gridctx[level].P);CHKERRQ(ierr);
      mlmat  = &(ml_object->Rmat[mllevel-1]);
      ierr = MatWrapML_SHELL(mlmat,MAT_INITIAL_MATRIX,&gridctx[level].R);CHKERRQ(ierr);

      mlmat  = &(ml_object->Amat[mllevel]);
      ierr = MatWrapML_MPIAIJ(mlmat,MAT_INITIAL_MATRIX,&gridctx[level].A);CHKERRQ(ierr);
      level--;
    }
  }

  /* create vectors and ksp at all levels */
  for (level=0; level<fine_level; level++){
    level1 = level + 1;
    ierr = VecCreate(((PetscObject)gridctx[level].A)->comm,&gridctx[level].x);CHKERRQ(ierr);
    ierr = VecSetSizes(gridctx[level].x,gridctx[level].A->cmap->n,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetType(gridctx[level].x,VECMPI);CHKERRQ(ierr);
    ierr = PCMGSetX(pc,level,gridctx[level].x);CHKERRQ(ierr);

    ierr = VecCreate(((PetscObject)gridctx[level].A)->comm,&gridctx[level].b);CHKERRQ(ierr);
    ierr = VecSetSizes(gridctx[level].b,gridctx[level].A->rmap->n,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetType(gridctx[level].b,VECMPI);CHKERRQ(ierr);
    ierr = PCMGSetRhs(pc,level,gridctx[level].b);CHKERRQ(ierr);

    ierr = VecCreate(((PetscObject)gridctx[level1].A)->comm,&gridctx[level1].r);CHKERRQ(ierr);
    ierr = VecSetSizes(gridctx[level1].r,gridctx[level1].A->rmap->n,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetType(gridctx[level1].r,VECMPI);CHKERRQ(ierr);
    ierr = PCMGSetR(pc,level1,gridctx[level1].r);CHKERRQ(ierr);

    if (level == 0){
      ierr = PCMGGetCoarseSolve(pc,&gridctx[level].ksp);CHKERRQ(ierr);
    } else {
      ierr = PCMGGetSmoother(pc,level,&gridctx[level].ksp);CHKERRQ(ierr);
    }
  }
  ierr = PCMGGetSmoother(pc,fine_level,&gridctx[fine_level].ksp);CHKERRQ(ierr);

  /* create coarse level and the interpolation between the levels */
  for (level=0; level<fine_level; level++){
    level1 = level + 1;
    ierr = PCMGSetInterpolation(pc,level1,gridctx[level].P);CHKERRQ(ierr);
    ierr = PCMGSetRestriction(pc,level1,gridctx[level].R);CHKERRQ(ierr);
    if (level > 0){
      ierr = PCMGSetResidual(pc,level,PCMGDefaultResidual,gridctx[level].A);CHKERRQ(ierr);
    }
    ierr = KSPSetOperators(gridctx[level].ksp,gridctx[level].A,gridctx[level].A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  ierr = PCMGSetResidual(pc,fine_level,PCMGDefaultResidual,gridctx[fine_level].A);CHKERRQ(ierr);
  ierr = KSPSetOperators(gridctx[fine_level].ksp,gridctx[level].A,gridctx[fine_level].A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  /* setupcalled is set to 0 so that MG is setup from scratch */
  pc->setupcalled = 0;
  ierr = PCSetUp_MG(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCDestroy_ML - Destroys the private context for the ML preconditioner
   that was created with PCCreate_ML().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
#undef __FUNCT__
#define __FUNCT__ "PCDestroy_ML"
PetscErrorCode PCDestroy_ML(PC pc)
{
  PetscErrorCode  ierr;
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_ML           *pc_ml= (PC_ML*)mg->innerctx;

  PetscFunctionBegin;
  ierr = PCReset_ML(pc);CHKERRQ(ierr);
  ierr = PetscFree(pc_ml);CHKERRQ(ierr);
  ierr = PCDestroy_MG(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_ML"
PetscErrorCode PCSetFromOptions_ML(PC pc)
{
  PetscErrorCode  ierr;
  PetscInt        indx,PrintLevel;
  const char      *scheme[] = {"Uncoupled","Coupled","MIS","METIS"};
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_ML           *pc_ml = (PC_ML*)mg->innerctx;
  PetscMPIInt     size;
  MPI_Comm        comm = ((PetscObject)pc)->comm;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscOptionsHead("ML options");CHKERRQ(ierr);
  PrintLevel    = 0;
  indx          = 0;
  ierr = PetscOptionsInt("-pc_ml_PrintLevel","Print level","ML_Set_PrintLevel",PrintLevel,&PrintLevel,PETSC_NULL);CHKERRQ(ierr);
  ML_Set_PrintLevel(PrintLevel);
  ierr = PetscOptionsInt("-pc_ml_maxNlevels","Maximum number of levels","None",pc_ml->MaxNlevels,&pc_ml->MaxNlevels,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_ml_maxCoarseSize","Maximum coarsest mesh size","ML_Aggregate_Set_MaxCoarseSize",pc_ml->MaxCoarseSize,&pc_ml->MaxCoarseSize,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-pc_ml_CoarsenScheme","Aggregate Coarsen Scheme","ML_Aggregate_Set_CoarsenScheme_*",scheme,4,scheme[0],&indx,PETSC_NULL);CHKERRQ(ierr);
  pc_ml->CoarsenScheme = indx;
  ierr = PetscOptionsReal("-pc_ml_DampingFactor","P damping factor","ML_Aggregate_Set_DampingFactor",pc_ml->DampingFactor,&pc_ml->DampingFactor,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_ml_Threshold","Smoother drop tol","ML_Aggregate_Set_Threshold",pc_ml->Threshold,&pc_ml->Threshold,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_ml_SpectralNormScheme_Anorm","Method used for estimating spectral radius","ML_Set_SpectralNormScheme_Anorm",pc_ml->SpectralNormScheme_Anorm,&pc_ml->SpectralNormScheme_Anorm,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_ml_Symmetrize","Symmetrize aggregation","ML_Set_Symmetrize",pc_ml->Symmetrize,&pc_ml->Symmetrize,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_ml_BlockScaling","Scale all dofs at each node together","None",pc_ml->BlockScaling,&pc_ml->BlockScaling,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-pc_ml_nullspace","Which type of null space information to use","None",PCMLNullSpaceTypes,(PetscEnum)pc_ml->nulltype,(PetscEnum*)&pc_ml->nulltype,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_ml_EnergyMinimization","Energy minimization norm type (0=no minimization; see ML manual for 1,2,3; -1 and 4 undocumented)","None",pc_ml->EnergyMinimization,&pc_ml->EnergyMinimization,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_ml_reuse_interpolation","Reuse the interpolation operators when possible (cheaper, weaker when matrix entries change a lot)","None",pc_ml->reuse_interpolation,&pc_ml->reuse_interpolation,PETSC_NULL);CHKERRQ(ierr);
  /*
    The following checks a number of conditions.  If we let this stuff slip by, then ML's error handling will take over.
    This is suboptimal because it amounts to calling exit(1) so we check for the most common conditions.

    We also try to set some sane defaults when energy minimization is activated, otherwise it's hard to find a working
    combination of options and ML's exit(1) explanations don't help matters.
  */
  if (pc_ml->EnergyMinimization < -1 || pc_ml->EnergyMinimization > 4) SETERRQ(comm,PETSC_ERR_ARG_OUTOFRANGE,"EnergyMinimization must be in range -1..4");
  if (pc_ml->EnergyMinimization == 4 && size > 1) SETERRQ(comm,PETSC_ERR_SUP,"Energy minimization type 4 does not work in parallel");
  if (pc_ml->EnergyMinimization == 4) {ierr = PetscInfo(pc,"Mandel's energy minimization scheme is experimental and broken in ML-6.2");CHKERRQ(ierr);}
  if (pc_ml->EnergyMinimization) {
    ierr = PetscOptionsReal("-pc_ml_EnergyMinimizationDropTol","Energy minimization drop tolerance","None",pc_ml->EnergyMinimizationDropTol,&pc_ml->EnergyMinimizationDropTol,PETSC_NULL);CHKERRQ(ierr);
  }
  if (pc_ml->EnergyMinimization == 2) {
    /* According to ml_MultiLevelPreconditioner.cpp, this option is only meaningful for norm type (2) */
    ierr = PetscOptionsBool("-pc_ml_EnergyMinimizationCheap","Use cheaper variant of norm type 2","None",pc_ml->EnergyMinimizationCheap,&pc_ml->EnergyMinimizationCheap,PETSC_NULL);CHKERRQ(ierr);
  }
  /* energy minimization sometimes breaks if this is turned off, the more classical stuff should be okay without it */
  if (pc_ml->EnergyMinimization) pc_ml->KeepAggInfo = PETSC_TRUE;
  ierr = PetscOptionsBool("-pc_ml_KeepAggInfo","Allows the preconditioner to be reused, or auxilliary matrices to be generated","None",pc_ml->KeepAggInfo,&pc_ml->KeepAggInfo,PETSC_NULL);CHKERRQ(ierr);
  /* Option (-1) doesn't work at all (calls exit(1)) if the tentative restriction operator isn't stored. */
  if (pc_ml->EnergyMinimization == -1) pc_ml->Reusable = PETSC_TRUE;
  ierr = PetscOptionsBool("-pc_ml_Reusable","Store intermedaiate data structures so that the multilevel hierarchy is reusable","None",pc_ml->Reusable,&pc_ml->Reusable,PETSC_NULL);CHKERRQ(ierr);
  /*
    ML's C API is severely underdocumented and lacks significant functionality.  The C++ API calls
    ML_Gen_MultiLevelHierarchy_UsingAggregation() which is a modified copy (!?) of the documented function
    ML_Gen_MGHierarchy_UsingAggregation().  This modification, however, does not provide a strict superset of the
    functionality in the old function, so some users may still want to use it.  Note that many options are ignored in
    this context, but ML doesn't provide a way to find out which ones.
   */
  ierr = PetscOptionsBool("-pc_ml_OldHierarchy","Use old routine to generate hierarchy","None",pc_ml->OldHierarchy,&pc_ml->OldHierarchy,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCCreate_ML - Creates a ML preconditioner context, PC_ML,
   and sets this as the private data within the generic preconditioning
   context, PC, that was created within PCCreate().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCCreate()
*/

/*MC
     PCML - Use algebraic multigrid preconditioning. This preconditioner requires you provide
       fine grid discretization matrix. The coarser grid matrices and restriction/interpolation
       operators are computed by ML, with the matrices coverted to PETSc matrices in aij format
       and the restriction/interpolation operators wrapped as PETSc shell matrices.

   Options Database Key:
   Multigrid options(inherited)
+  -pc_mg_cycles <1>: 1 for V cycle, 2 for W-cycle (MGSetCycles)
.  -pc_mg_smoothup <1>: Number of post-smoothing steps (MGSetNumberSmoothUp)
.  -pc_mg_smoothdown <1>: Number of pre-smoothing steps (MGSetNumberSmoothDown)
   -pc_mg_type <multiplicative>: (one of) additive multiplicative full kascade
   ML options:
.  -pc_ml_PrintLevel <0>: Print level (ML_Set_PrintLevel)
.  -pc_ml_maxNlevels <10>: Maximum number of levels (None)
.  -pc_ml_maxCoarseSize <1>: Maximum coarsest mesh size (ML_Aggregate_Set_MaxCoarseSize)
.  -pc_ml_CoarsenScheme <Uncoupled>: (one of) Uncoupled Coupled MIS METIS
.  -pc_ml_DampingFactor <1.33333>: P damping factor (ML_Aggregate_Set_DampingFactor)
.  -pc_ml_Threshold <0>: Smoother drop tol (ML_Aggregate_Set_Threshold)
-  -pc_ml_SpectralNormScheme_Anorm <false>: Method used for estimating spectral radius (ML_Set_SpectralNormScheme_Anorm)

   Level: intermediate

  Concepts: multigrid

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCMGType,
           PCMGSetLevels(), PCMGGetLevels(), PCMGSetType(), MPSetCycles(), PCMGSetNumberSmoothDown(),
           PCMGSetNumberSmoothUp(), PCMGGetCoarseSolve(), PCMGSetResidual(), PCMGSetInterpolation(),
           PCMGSetRestriction(), PCMGGetSmoother(), PCMGGetSmootherUp(), PCMGGetSmootherDown(),
           PCMGSetCyclesOnLevel(), PCMGSetRhs(), PCMGSetX(), PCMGSetR()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCCreate_ML"
PetscErrorCode  PCCreate_ML(PC pc)
{
  PetscErrorCode  ierr;
  PC_ML           *pc_ml;
  PC_MG           *mg;

  PetscFunctionBegin;
  /* PCML is an inherited class of PCMG. Initialize pc as PCMG */
  ierr = PCSetType(pc,PCMG);CHKERRQ(ierr); /* calls PCCreate_MG() and MGCreate_Private() */
  ierr = PetscObjectChangeTypeName((PetscObject)pc,PCML);CHKERRQ(ierr);
  /* Since PCMG tries to use DM assocated with PC must delete it */
  ierr = DMDestroy(&pc->dm);CHKERRQ(ierr);
  mg = (PC_MG*)pc->data;
  mg->galerkin = 2;             /* Use Galerkin, but it is computed externally */

  /* create a supporting struct and attach it to pc */
  ierr = PetscNewLog(pc,PC_ML,&pc_ml);CHKERRQ(ierr);
  mg->innerctx = pc_ml;

  pc_ml->ml_object     = 0;
  pc_ml->agg_object    = 0;
  pc_ml->gridctx       = 0;
  pc_ml->PetscMLdata   = 0;
  pc_ml->Nlevels       = -1;
  pc_ml->MaxNlevels    = 10;
  pc_ml->MaxCoarseSize = 1;
  pc_ml->CoarsenScheme = 1;
  pc_ml->Threshold     = 0.0;
  pc_ml->DampingFactor = 4.0/3.0;
  pc_ml->SpectralNormScheme_Anorm = PETSC_FALSE;
  pc_ml->size          = 0;

  /* overwrite the pointers of PCMG by the functions of PCML */
  pc->ops->setfromoptions = PCSetFromOptions_ML;
  pc->ops->setup          = PCSetUp_ML;
  pc->ops->reset          = PCReset_ML;
  pc->ops->destroy        = PCDestroy_ML;
  PetscFunctionReturn(0);
}
EXTERN_C_END
