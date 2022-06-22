
/*
     Routines that call the kernel minpack coloring subroutines
*/

#include <petsc/private/matimpl.h>
#include <petsc/private/isimpl.h>
#include <../src/mat/color/impls/minpack/color.h>

/*
    MatFDColoringDegreeSequence_Minpack - Calls the MINPACK routine seqr() that
      computes the degree sequence required by MINPACK coloring routines.
*/
PETSC_INTERN PetscErrorCode MatFDColoringDegreeSequence_Minpack(PetscInt m,const PetscInt *cja,const PetscInt *cia,const PetscInt *rja,const PetscInt *ria,PetscInt **seq)
{
  PetscInt       *work;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(m,&work));
  PetscCall(PetscMalloc1(m,seq));

  MINPACKdegr(&m,cja,cia,rja,ria,*seq,work);

  PetscCall(PetscFree(work));
  PetscFunctionReturn(0);
}

/*
    MatFDColoringMinimumNumberofColors_Private - For a given sparse
        matrix computes the minimum number of colors needed.

*/
PetscErrorCode MatFDColoringMinimumNumberofColors_Private(PetscInt m,PetscInt *ia,PetscInt *minc)
{
  PetscInt i,c = 0;

  PetscFunctionBegin;
  for (i=0; i<m; i++) c = PetscMax(c,ia[i+1]-ia[i]);
  *minc = c;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatColoringApply_SL(MatColoring mc,ISColoring *iscoloring)
{
  PetscInt        *list,*work,clique,*seq,*coloring,n;
  const PetscInt  *ria,*rja,*cia,*cja;
  PetscInt        ncolors,i;
  PetscBool       done;
  Mat             mat = mc->mat;
  Mat             mat_seq = mat;
  PetscMPIInt     size;
  MPI_Comm        comm;
  ISColoring      iscoloring_seq;
  PetscInt        bs = 1,rstart,rend,N_loc,nc;
  ISColoringValue *colors_loc;
  PetscBool       flg1,flg2;

  PetscFunctionBegin;
  PetscCheck(mc->dist == 2,PETSC_COMM_SELF,PETSC_ERR_SUP,"SL may only do distance 2 coloring");
  /* this is ugly way to get blocksize but cannot call MatGetBlockSize() because AIJ can have bs > 1 */
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)mat,MATSEQBAIJ,&flg1));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)mat,MATMPIBAIJ,&flg2));
  if (flg1 || flg2) {
    PetscCall(MatGetBlockSize(mat,&bs));
  }

  PetscCall(PetscObjectGetComm((PetscObject)mat,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  if (size > 1) {
    /* create a sequential iscoloring on all processors */
    PetscCall(MatGetSeqNonzeroStructure(mat,&mat_seq));
  }

  PetscCall(MatGetRowIJ(mat_seq,1,PETSC_FALSE,PETSC_TRUE,&n,&ria,&rja,&done));
  PetscCall(MatGetColumnIJ(mat_seq,1,PETSC_FALSE,PETSC_TRUE,&n,&cia,&cja,&done));
  PetscCheck(done,PETSC_COMM_SELF,PETSC_ERR_SUP,"Ordering requires IJ");

  PetscCall(MatFDColoringDegreeSequence_Minpack(n,cja,cia,rja,ria,&seq));

  PetscCall(PetscMalloc2(n,&list,4*n,&work));

  MINPACKslo(&n,cja,cia,rja,ria,seq,list,&clique,work,work+n,work+2*n,work+3*n);

  PetscCall(PetscMalloc1(n,&coloring));
  MINPACKseq(&n,cja,cia,rja,ria,list,coloring,&ncolors,work);

  PetscCall(PetscFree2(list,work));
  PetscCall(PetscFree(seq));
  PetscCall(MatRestoreRowIJ(mat_seq,1,PETSC_FALSE,PETSC_TRUE,NULL,&ria,&rja,&done));
  PetscCall(MatRestoreColumnIJ(mat_seq,1,PETSC_FALSE,PETSC_TRUE,NULL,&cia,&cja,&done));

  /* shift coloring numbers to start at zero and shorten */
  PetscCheck(ncolors <= IS_COLORING_MAX-1,PETSC_COMM_SELF,PETSC_ERR_SUP,"Maximum color size exceeded");
  {
    ISColoringValue *s = (ISColoringValue*) coloring;
    for (i=0; i<n; i++) {
      s[i] = (ISColoringValue) (coloring[i]-1);
    }
    PetscCall(MatColoringPatch(mat_seq,ncolors,n,s,iscoloring));
  }

  if (size > 1) {
    PetscCall(MatDestroySeqNonzeroStructure(&mat_seq));

    /* convert iscoloring_seq to a parallel iscoloring */
    iscoloring_seq = *iscoloring;
    rstart         = mat->rmap->rstart/bs;
    rend           = mat->rmap->rend/bs;
    N_loc          = rend - rstart; /* number of local nodes */

    /* get local colors for each local node */
    PetscCall(PetscMalloc1(N_loc+1,&colors_loc));
    for (i=rstart; i<rend; i++) {
      colors_loc[i-rstart] = iscoloring_seq->colors[i];
    }
    /* create a parallel iscoloring */
    nc   = iscoloring_seq->n;
    PetscCall(ISColoringCreate(comm,nc,N_loc,colors_loc,PETSC_OWN_POINTER,iscoloring));
    PetscCall(ISColoringDestroy(&iscoloring_seq));
  }
  PetscFunctionReturn(0);
}

/*MC
  MATCOLORINGSL - implements the SL (smallest last) coloring routine

   Level: beginner

   Notes:
    Supports only distance two colorings (for computation of Jacobians)

          This is a sequential algorithm

   References:
.  * - TF Coleman and J More, "Estimation of sparse Jacobian matrices and graph coloring," SIAM Journal on Numerical Analysis, vol. 20, no. 1,
   pp. 187-209, 1983.

.seealso: `MatColoringCreate()`, `MatColoring`, `MatColoringSetType()`, `MATCOLORINGGREEDY`, `MatColoringType`
M*/

PETSC_EXTERN PetscErrorCode MatColoringCreate_SL(MatColoring mc)
{
    PetscFunctionBegin;
    mc->dist                = 2;
    mc->data                = NULL;
    mc->ops->apply          = MatColoringApply_SL;
    mc->ops->view           = NULL;
    mc->ops->destroy        = NULL;
    mc->ops->setfromoptions = NULL;
    PetscFunctionReturn(0);
}

static PetscErrorCode MatColoringApply_LF(MatColoring mc,ISColoring *iscoloring)
{
  PetscInt        *list,*work,*seq,*coloring,n;
  const PetscInt  *ria,*rja,*cia,*cja;
  PetscInt        n1, none,ncolors,i;
  PetscBool       done;
  Mat             mat = mc->mat;
  Mat             mat_seq = mat;
  PetscMPIInt     size;
  MPI_Comm        comm;
  ISColoring      iscoloring_seq;
  PetscInt        bs = 1,rstart,rend,N_loc,nc;
  ISColoringValue *colors_loc;
  PetscBool       flg1,flg2;

  PetscFunctionBegin;
  PetscCheck(mc->dist == 2,PETSC_COMM_SELF,PETSC_ERR_SUP,"LF may only do distance 2 coloring");
  /* this is ugly way to get blocksize but cannot call MatGetBlockSize() because AIJ can have bs > 1 */
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)mat,MATSEQBAIJ,&flg1));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)mat,MATMPIBAIJ,&flg2));
  if (flg1 || flg2) {
    PetscCall(MatGetBlockSize(mat,&bs));
  }

  PetscCall(PetscObjectGetComm((PetscObject)mat,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  if (size > 1) {
    /* create a sequential iscoloring on all processors */
    PetscCall(MatGetSeqNonzeroStructure(mat,&mat_seq));
  }

  PetscCall(MatGetRowIJ(mat_seq,1,PETSC_FALSE,PETSC_TRUE,&n,&ria,&rja,&done));
  PetscCall(MatGetColumnIJ(mat_seq,1,PETSC_FALSE,PETSC_TRUE,&n,&cia,&cja,&done));
  PetscCheck(done,PETSC_COMM_SELF,PETSC_ERR_SUP,"Ordering requires IJ");

  PetscCall(MatFDColoringDegreeSequence_Minpack(n,cja,cia,rja,ria,&seq));

  PetscCall(PetscMalloc2(n,&list,4*n,&work));

  n1   = n - 1;
  none = -1;
  MINPACKnumsrt(&n,&n1,seq,&none,list,work+2*n,work+n);
  PetscCall(PetscMalloc1(n,&coloring));
  MINPACKseq(&n,cja,cia,rja,ria,list,coloring,&ncolors,work);

  PetscCall(PetscFree2(list,work));
  PetscCall(PetscFree(seq));

  PetscCall(MatRestoreRowIJ(mat_seq,1,PETSC_FALSE,PETSC_TRUE,NULL,&ria,&rja,&done));
  PetscCall(MatRestoreColumnIJ(mat_seq,1,PETSC_FALSE,PETSC_TRUE,NULL,&cia,&cja,&done));

  /* shift coloring numbers to start at zero and shorten */
  PetscCheck(ncolors <= IS_COLORING_MAX-1,PETSC_COMM_SELF,PETSC_ERR_SUP,"Maximum color size exceeded");
  {
    ISColoringValue *s = (ISColoringValue*) coloring;
    for (i=0; i<n; i++) s[i] = (ISColoringValue) (coloring[i]-1);
    PetscCall(MatColoringPatch(mat_seq,ncolors,n,s,iscoloring));
  }

  if (size > 1) {
    PetscCall(MatDestroySeqNonzeroStructure(&mat_seq));

    /* convert iscoloring_seq to a parallel iscoloring */
    iscoloring_seq = *iscoloring;
    rstart         = mat->rmap->rstart/bs;
    rend           = mat->rmap->rend/bs;
    N_loc          = rend - rstart; /* number of local nodes */

    /* get local colors for each local node */
    PetscCall(PetscMalloc1(N_loc+1,&colors_loc));
    for (i=rstart; i<rend; i++) colors_loc[i-rstart] = iscoloring_seq->colors[i];

    /* create a parallel iscoloring */
    nc   = iscoloring_seq->n;
    PetscCall(ISColoringCreate(comm,nc,N_loc,colors_loc,PETSC_OWN_POINTER,iscoloring));
    PetscCall(ISColoringDestroy(&iscoloring_seq));
  }
  PetscFunctionReturn(0);
}

/*MC
  MATCOLORINGLF - implements the LF (largest first) coloring routine

   Level: beginner

   Notes:
    Supports only distance two colorings (for computation of Jacobians)

          This is a sequential algorithm

   References:
.  * - TF Coleman and J More, "Estimation of sparse Jacobian matrices and graph coloring," SIAM Journal on Numerical Analysis, vol. 20, no. 1,
   pp. 187-209, 1983.

.seealso: `MatColoringCreate()`, `MatColoring`, `MatColoringSetType()`, `MATCOLORINGGREEDY`, `MatColoringType`
M*/

PETSC_EXTERN PetscErrorCode MatColoringCreate_LF(MatColoring mc)
{
    PetscFunctionBegin;
    mc->dist                = 2;
    mc->data                = NULL;
    mc->ops->apply          = MatColoringApply_LF;
    mc->ops->view           = NULL;
    mc->ops->destroy        = NULL;
    mc->ops->setfromoptions = NULL;
    PetscFunctionReturn(0);
}

static PetscErrorCode MatColoringApply_ID(MatColoring mc,ISColoring *iscoloring)
{
  PetscInt        *list,*work,clique,*seq,*coloring,n;
  const PetscInt  *ria,*rja,*cia,*cja;
  PetscInt        ncolors,i;
  PetscBool       done;
  Mat             mat = mc->mat;
  Mat             mat_seq = mat;
  PetscMPIInt     size;
  MPI_Comm        comm;
  ISColoring      iscoloring_seq;
  PetscInt        bs = 1,rstart,rend,N_loc,nc;
  ISColoringValue *colors_loc;
  PetscBool       flg1,flg2;

  PetscFunctionBegin;
  PetscCheck(mc->dist == 2,PETSC_COMM_SELF,PETSC_ERR_SUP,"IDO may only do distance 2 coloring");
  /* this is ugly way to get blocksize but cannot call MatGetBlockSize() because AIJ can have bs > 1 */
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)mat,MATSEQBAIJ,&flg1));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)mat,MATMPIBAIJ,&flg2));
  if (flg1 || flg2) {
    PetscCall(MatGetBlockSize(mat,&bs));
  }

  PetscCall(PetscObjectGetComm((PetscObject)mat,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  if (size > 1) {
    /* create a sequential iscoloring on all processors */
    PetscCall(MatGetSeqNonzeroStructure(mat,&mat_seq));
  }

  PetscCall(MatGetRowIJ(mat_seq,1,PETSC_FALSE,PETSC_TRUE,&n,&ria,&rja,&done));
  PetscCall(MatGetColumnIJ(mat_seq,1,PETSC_FALSE,PETSC_TRUE,&n,&cia,&cja,&done));
  PetscCheck(done,PETSC_COMM_SELF,PETSC_ERR_SUP,"Ordering requires IJ");

  PetscCall(MatFDColoringDegreeSequence_Minpack(n,cja,cia,rja,ria,&seq));

  PetscCall(PetscMalloc2(n,&list,4*n,&work));

  MINPACKido(&n,&n,cja,cia,rja,ria,seq,list,&clique,work,work+n,work+2*n,work+3*n);

  PetscCall(PetscMalloc1(n,&coloring));
  MINPACKseq(&n,cja,cia,rja,ria,list,coloring,&ncolors,work);

  PetscCall(PetscFree2(list,work));
  PetscCall(PetscFree(seq));

  PetscCall(MatRestoreRowIJ(mat_seq,1,PETSC_FALSE,PETSC_TRUE,NULL,&ria,&rja,&done));
  PetscCall(MatRestoreColumnIJ(mat_seq,1,PETSC_FALSE,PETSC_TRUE,NULL,&cia,&cja,&done));

  /* shift coloring numbers to start at zero and shorten */
  PetscCheck(ncolors <= IS_COLORING_MAX-1,PETSC_COMM_SELF,PETSC_ERR_SUP,"Maximum color size exceeded");
  {
    ISColoringValue *s = (ISColoringValue*) coloring;
    for (i=0; i<n; i++) {
      s[i] = (ISColoringValue) (coloring[i]-1);
    }
    PetscCall(MatColoringPatch(mat_seq,ncolors,n,s,iscoloring));
  }

  if (size > 1) {
    PetscCall(MatDestroySeqNonzeroStructure(&mat_seq));

    /* convert iscoloring_seq to a parallel iscoloring */
    iscoloring_seq = *iscoloring;
    rstart         = mat->rmap->rstart/bs;
    rend           = mat->rmap->rend/bs;
    N_loc          = rend - rstart; /* number of local nodes */

    /* get local colors for each local node */
    PetscCall(PetscMalloc1(N_loc+1,&colors_loc));
    for (i=rstart; i<rend; i++) {
      colors_loc[i-rstart] = iscoloring_seq->colors[i];
    }
    /* create a parallel iscoloring */
    nc   = iscoloring_seq->n;
    PetscCall(ISColoringCreate(comm,nc,N_loc,colors_loc,PETSC_OWN_POINTER,iscoloring));
    PetscCall(ISColoringDestroy(&iscoloring_seq));
  }
  PetscFunctionReturn(0);
}

/*MC
  MATCOLORINGID - implements the ID (incidence degree) coloring routine

   Level: beginner

   Notes:
    Supports only distance two colorings (for computation of Jacobians)

          This is a sequential algorithm

   References:
.  * - TF Coleman and J More, "Estimation of sparse Jacobian matrices and graph coloring," SIAM Journal on Numerical Analysis, vol. 20, no. 1,
   pp. 187-209, 1983.

.seealso: `MatColoringCreate()`, `MatColoring`, `MatColoringSetType()`, `MATCOLORINGGREEDY`, `MatColoringType`
M*/

PETSC_EXTERN PetscErrorCode MatColoringCreate_ID(MatColoring mc)
{
    PetscFunctionBegin;
    mc->dist                = 2;
    mc->data                = NULL;
    mc->ops->apply          = MatColoringApply_ID;
    mc->ops->view           = NULL;
    mc->ops->destroy        = NULL;
    mc->ops->setfromoptions = NULL;
    PetscFunctionReturn(0);
}
