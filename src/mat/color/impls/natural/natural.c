#include <petsc/private/matimpl.h>      /*I "petscmat.h"  I*/
#include <petsc/private/isimpl.h>

static PetscErrorCode MatColoringApply_Natural(MatColoring mc,ISColoring *iscoloring)
{
  PetscErrorCode  ierr;
  PetscInt        start,end,i,bs = 1,n;
  ISColoringValue *colors;
  MPI_Comm        comm;
  PetscBool       flg1,flg2;
  Mat             mat     = mc->mat;
  Mat             mat_seq = mc->mat;
  PetscMPIInt     size;
  ISColoring      iscoloring_seq;
  ISColoringValue *colors_loc;
  PetscInt        rstart,rend,N_loc,nc;

  PetscFunctionBegin;
  /* this is ugly way to get blocksize but cannot call MatGetBlockSize() because AIJ can have bs > 1 */
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATSEQBAIJ,&flg1);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATMPIBAIJ,&flg2);CHKERRQ(ierr);
  if (flg1 || flg2) {
    ierr = MatGetBlockSize(mat,&bs);CHKERRQ(ierr);
  }

  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size > 1) {
    /* create a sequential iscoloring on all processors */
    ierr = MatGetSeqNonzeroStructure(mat,&mat_seq);CHKERRQ(ierr);
  }

  ierr = MatGetSize(mat_seq,&n,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(mat_seq,&start,&end);CHKERRQ(ierr);
  n    = n/bs;
  PetscAssertFalse(n > IS_COLORING_MAX-1,PETSC_COMM_SELF,PETSC_ERR_SUP,"Maximum color size exceeded");

  start = start/bs;
  end   = end/bs;
  ierr  = PetscMalloc1(end-start+1,&colors);CHKERRQ(ierr);
  for (i=start; i<end; i++) {
    colors[i-start] = (ISColoringValue)i;
  }
  ierr = ISColoringCreate(comm,n,end-start,colors,PETSC_OWN_POINTER,iscoloring);CHKERRQ(ierr);

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
  MATCOLORINGNATURAL - implements a trivial coloring routine with one color per column

  Level: beginner

.seealso: MatColoringCreate(), MatColoring, MatColoringSetType(), MatColoringType
M*/
PETSC_EXTERN PetscErrorCode MatColoringCreate_Natural(MatColoring mc)
{
    PetscFunctionBegin;
    mc->data                = NULL;
    mc->ops->apply          = MatColoringApply_Natural;
    mc->ops->view           = NULL;
    mc->ops->destroy        = NULL;
    mc->ops->setfromoptions = NULL;
    PetscFunctionReturn(0);
}
