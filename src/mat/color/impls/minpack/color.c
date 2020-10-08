
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(m,&work);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,seq);CHKERRQ(ierr);

  MINPACKdegr(&m,cja,cia,rja,ria,*seq,work);

  ierr = PetscFree(work);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;
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
  if (mc->dist != 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SL may only do distance 2 coloring");
  /* this is ugly way to get blocksize but cannot call MatGetBlockSize() because AIJ can have bs > 1 */
  ierr = PetscObjectBaseTypeCompare((PetscObject)mat,MATSEQBAIJ,&flg1);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)mat,MATMPIBAIJ,&flg2);CHKERRQ(ierr);
  if (flg1 || flg2) {
    ierr = MatGetBlockSize(mat,&bs);CHKERRQ(ierr);
  }

  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size > 1) {
    /* create a sequential iscoloring on all processors */
    ierr = MatGetSeqNonzeroStructure(mat,&mat_seq);CHKERRQ(ierr);
  }

  ierr = MatGetRowIJ(mat_seq,1,PETSC_FALSE,PETSC_TRUE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatGetColumnIJ(mat_seq,1,PETSC_FALSE,PETSC_TRUE,&n,&cia,&cja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Ordering requires IJ");

  ierr = MatFDColoringDegreeSequence_Minpack(n,cja,cia,rja,ria,&seq);CHKERRQ(ierr);

  ierr = PetscMalloc2(n,&list,4*n,&work);CHKERRQ(ierr);

  MINPACKslo(&n,cja,cia,rja,ria,seq,list,&clique,work,work+n,work+2*n,work+3*n);

  ierr = PetscMalloc1(n,&coloring);CHKERRQ(ierr);
  MINPACKseq(&n,cja,cia,rja,ria,list,coloring,&ncolors,work);

  ierr = PetscFree2(list,work);CHKERRQ(ierr);
  ierr = PetscFree(seq);CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(mat_seq,1,PETSC_FALSE,PETSC_TRUE,NULL,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatRestoreColumnIJ(mat_seq,1,PETSC_FALSE,PETSC_TRUE,NULL,&cia,&cja,&done);CHKERRQ(ierr);

  /* shift coloring numbers to start at zero and shorten */
  if (ncolors > IS_COLORING_MAX-1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Maximum color size exceeded");
  {
    ISColoringValue *s = (ISColoringValue*) coloring;
    for (i=0; i<n; i++) {
      s[i] = (ISColoringValue) (coloring[i]-1);
    }
    ierr = MatColoringPatch(mat_seq,ncolors,n,s,iscoloring);CHKERRQ(ierr);
  }

  if (size > 1) {
    ierr = MatDestroySeqNonzeroStructure(&mat_seq);CHKERRQ(ierr);

    /* convert iscoloring_seq to a parallel iscoloring */
    iscoloring_seq = *iscoloring;
    rstart         = mat->rmap->rstart/bs;
    rend           = mat->rmap->rend/bs;
    N_loc          = rend - rstart; /* number of local nodes */

    /* get local colors for each local node */
    ierr = PetscMalloc1(N_loc+1,&colors_loc);CHKERRQ(ierr);
    for (i=rstart; i<rend; i++) {
      colors_loc[i-rstart] = iscoloring_seq->colors[i];
    }
    /* create a parallel iscoloring */
    nc   = iscoloring_seq->n;
    ierr = ISColoringCreate(comm,nc,N_loc,colors_loc,PETSC_OWN_POINTER,iscoloring);CHKERRQ(ierr);
    ierr = ISColoringDestroy(&iscoloring_seq);CHKERRQ(ierr);
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
.  1. - TF Coleman and J More, "Estimation of sparse Jacobian matrices and graph coloring," SIAM Journal on Numerical Analysis, vol. 20, no. 1,
   pp. 187-209, 1983.

.seealso: MatColoringCreate(), MatColoring, MatColoringSetType(), MATCOLORINGGREEDY, MatColoringType
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
  PetscErrorCode  ierr;
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
  if (mc->dist != 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"LF may only do distance 2 coloring");
  /* this is ugly way to get blocksize but cannot call MatGetBlockSize() because AIJ can have bs > 1 */
  ierr = PetscObjectBaseTypeCompare((PetscObject)mat,MATSEQBAIJ,&flg1);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)mat,MATMPIBAIJ,&flg2);CHKERRQ(ierr);
  if (flg1 || flg2) {
    ierr = MatGetBlockSize(mat,&bs);CHKERRQ(ierr);
  }

  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size > 1) {
    /* create a sequential iscoloring on all processors */
    ierr = MatGetSeqNonzeroStructure(mat,&mat_seq);CHKERRQ(ierr);
  }

  ierr = MatGetRowIJ(mat_seq,1,PETSC_FALSE,PETSC_TRUE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatGetColumnIJ(mat_seq,1,PETSC_FALSE,PETSC_TRUE,&n,&cia,&cja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Ordering requires IJ");

  ierr = MatFDColoringDegreeSequence_Minpack(n,cja,cia,rja,ria,&seq);CHKERRQ(ierr);

  ierr = PetscMalloc2(n,&list,4*n,&work);CHKERRQ(ierr);

  n1   = n - 1;
  none = -1;
  MINPACKnumsrt(&n,&n1,seq,&none,list,work+2*n,work+n);
  ierr = PetscMalloc1(n,&coloring);CHKERRQ(ierr);
  MINPACKseq(&n,cja,cia,rja,ria,list,coloring,&ncolors,work);

  ierr = PetscFree2(list,work);CHKERRQ(ierr);
  ierr = PetscFree(seq);CHKERRQ(ierr);

  ierr = MatRestoreRowIJ(mat_seq,1,PETSC_FALSE,PETSC_TRUE,NULL,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatRestoreColumnIJ(mat_seq,1,PETSC_FALSE,PETSC_TRUE,NULL,&cia,&cja,&done);CHKERRQ(ierr);

  /* shift coloring numbers to start at zero and shorten */
  if (ncolors > IS_COLORING_MAX-1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Maximum color size exceeded");
  {
    ISColoringValue *s = (ISColoringValue*) coloring;
    for (i=0; i<n; i++) s[i] = (ISColoringValue) (coloring[i]-1);
    ierr = MatColoringPatch(mat_seq,ncolors,n,s,iscoloring);CHKERRQ(ierr);
  }

  if (size > 1) {
    ierr = MatDestroySeqNonzeroStructure(&mat_seq);CHKERRQ(ierr);

    /* convert iscoloring_seq to a parallel iscoloring */
    iscoloring_seq = *iscoloring;
    rstart         = mat->rmap->rstart/bs;
    rend           = mat->rmap->rend/bs;
    N_loc          = rend - rstart; /* number of local nodes */

    /* get local colors for each local node */
    ierr = PetscMalloc1(N_loc+1,&colors_loc);CHKERRQ(ierr);
    for (i=rstart; i<rend; i++) colors_loc[i-rstart] = iscoloring_seq->colors[i];

    /* create a parallel iscoloring */
    nc   = iscoloring_seq->n;
    ierr = ISColoringCreate(comm,nc,N_loc,colors_loc,PETSC_OWN_POINTER,iscoloring);CHKERRQ(ierr);
    ierr = ISColoringDestroy(&iscoloring_seq);CHKERRQ(ierr);
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
.  1. - TF Coleman and J More, "Estimation of sparse Jacobian matrices and graph coloring," SIAM Journal on Numerical Analysis, vol. 20, no. 1,
   pp. 187-209, 1983.

.seealso: MatColoringCreate(), MatColoring, MatColoringSetType(), MATCOLORINGGREEDY, MatColoringType
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
  PetscErrorCode  ierr;
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
  if (mc->dist != 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"IDO may only do distance 2 coloring");
  /* this is ugly way to get blocksize but cannot call MatGetBlockSize() because AIJ can have bs > 1 */
  ierr = PetscObjectBaseTypeCompare((PetscObject)mat,MATSEQBAIJ,&flg1);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)mat,MATMPIBAIJ,&flg2);CHKERRQ(ierr);
  if (flg1 || flg2) {
    ierr = MatGetBlockSize(mat,&bs);CHKERRQ(ierr);
  }

  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size > 1) {
    /* create a sequential iscoloring on all processors */
    ierr = MatGetSeqNonzeroStructure(mat,&mat_seq);CHKERRQ(ierr);
  }

  ierr = MatGetRowIJ(mat_seq,1,PETSC_FALSE,PETSC_TRUE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatGetColumnIJ(mat_seq,1,PETSC_FALSE,PETSC_TRUE,&n,&cia,&cja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Ordering requires IJ");

  ierr = MatFDColoringDegreeSequence_Minpack(n,cja,cia,rja,ria,&seq);CHKERRQ(ierr);

  ierr = PetscMalloc2(n,&list,4*n,&work);CHKERRQ(ierr);

  MINPACKido(&n,&n,cja,cia,rja,ria,seq,list,&clique,work,work+n,work+2*n,work+3*n);

  ierr = PetscMalloc1(n,&coloring);CHKERRQ(ierr);
  MINPACKseq(&n,cja,cia,rja,ria,list,coloring,&ncolors,work);

  ierr = PetscFree2(list,work);CHKERRQ(ierr);
  ierr = PetscFree(seq);CHKERRQ(ierr);

  ierr = MatRestoreRowIJ(mat_seq,1,PETSC_FALSE,PETSC_TRUE,NULL,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatRestoreColumnIJ(mat_seq,1,PETSC_FALSE,PETSC_TRUE,NULL,&cia,&cja,&done);CHKERRQ(ierr);

  /* shift coloring numbers to start at zero and shorten */
  if (ncolors > IS_COLORING_MAX-1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Maximum color size exceeded");
  {
    ISColoringValue *s = (ISColoringValue*) coloring;
    for (i=0; i<n; i++) {
      s[i] = (ISColoringValue) (coloring[i]-1);
    }
    ierr = MatColoringPatch(mat_seq,ncolors,n,s,iscoloring);CHKERRQ(ierr);
  }

  if (size > 1) {
    ierr = MatDestroySeqNonzeroStructure(&mat_seq);CHKERRQ(ierr);

    /* convert iscoloring_seq to a parallel iscoloring */
    iscoloring_seq = *iscoloring;
    rstart         = mat->rmap->rstart/bs;
    rend           = mat->rmap->rend/bs;
    N_loc          = rend - rstart; /* number of local nodes */

    /* get local colors for each local node */
    ierr = PetscMalloc1(N_loc+1,&colors_loc);CHKERRQ(ierr);
    for (i=rstart; i<rend; i++) {
      colors_loc[i-rstart] = iscoloring_seq->colors[i];
    }
    /* create a parallel iscoloring */
    nc   = iscoloring_seq->n;
    ierr = ISColoringCreate(comm,nc,N_loc,colors_loc,PETSC_OWN_POINTER,iscoloring);CHKERRQ(ierr);
    ierr = ISColoringDestroy(&iscoloring_seq);CHKERRQ(ierr);
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
.  1. - TF Coleman and J More, "Estimation of sparse Jacobian matrices and graph coloring," SIAM Journal on Numerical Analysis, vol. 20, no. 1,
   pp. 187-209, 1983.

.seealso: MatColoringCreate(), MatColoring, MatColoringSetType(), MATCOLORINGGREEDY, MatColoringType
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
