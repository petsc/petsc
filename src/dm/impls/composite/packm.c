
#include <../src/dm/impls/composite/packimpl.h>       /*I  "petscdmcomposite.h"  I*/

static PetscErrorCode DMCreateMatrix_Composite_Nest(DM dm,Mat *J)
{
  const DM_Composite           *com = (DM_Composite*)dm->data;
  const struct DMCompositeLink *rlink,*clink;
  IS                           *isg;
  Mat                          *submats;
  PetscInt                     i,j,n;

  PetscFunctionBegin;
  n = com->nDM;                 /* Total number of entries */

  /* Explicit index sets are not required for MatCreateNest, but getting them here allows MatNest to do consistency
   * checking and allows ISEqual to compare by identity instead of by contents. */
  CHKERRQ(DMCompositeGetGlobalISs(dm,&isg));

  /* Get submatrices */
  CHKERRQ(PetscMalloc1(n*n,&submats));
  for (i=0,rlink=com->next; rlink; i++,rlink=rlink->next) {
    for (j=0,clink=com->next; clink; j++,clink=clink->next) {
      Mat sub = NULL;
      if (i == j) {
        CHKERRQ(DMCreateMatrix(rlink->dm,&sub));
      } else PetscCheckFalse(com->FormCoupleLocations,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Cannot manage off-diagonal parts yet");
      submats[i*n+j] = sub;
    }
  }

  CHKERRQ(MatCreateNest(PetscObjectComm((PetscObject)dm),n,isg,n,isg,submats,J));

  /* Disown references */
  for (i=0; i<n; i++) CHKERRQ(ISDestroy(&isg[i]));
  CHKERRQ(PetscFree(isg));

  for (i=0; i<n*n; i++) {
    if (submats[i]) CHKERRQ(MatDestroy(&submats[i]));
  }
  CHKERRQ(PetscFree(submats));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateMatrix_Composite_AIJ(DM dm,Mat *J)
{
  PetscErrorCode         ierr;
  DM_Composite           *com = (DM_Composite*)dm->data;
  struct DMCompositeLink *next;
  PetscInt               m,*dnz,*onz,i,j,mA;
  Mat                    Atmp;
  PetscMPIInt            rank;
  PetscBool              dense = PETSC_FALSE;

  PetscFunctionBegin;
  /* use global vector to determine layout needed for matrix */
  m = com->n;

  CHKERRQ(MatCreate(PetscObjectComm((PetscObject)dm),J));
  CHKERRQ(MatSetSizes(*J,m,m,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatSetType(*J,dm->mattype));

  /*
     Extremely inefficient but will compute entire Jacobian for testing
  */
  CHKERRQ(PetscOptionsGetBool(((PetscObject)dm)->options,((PetscObject)dm)->prefix,"-dmcomposite_dense_jacobian",&dense,NULL));
  if (dense) {
    PetscInt    rstart,rend,*indices;
    PetscScalar *values;

    mA   = com->N;
    CHKERRQ(MatMPIAIJSetPreallocation(*J,mA,NULL,mA-m,NULL));
    CHKERRQ(MatSeqAIJSetPreallocation(*J,mA,NULL));

    CHKERRQ(MatGetOwnershipRange(*J,&rstart,&rend));
    CHKERRQ(PetscMalloc2(mA,&values,mA,&indices));
    CHKERRQ(PetscArrayzero(values,mA));
    for (i=0; i<mA; i++) indices[i] = i;
    for (i=rstart; i<rend; i++) {
      CHKERRQ(MatSetValues(*J,1,&i,mA,indices,values,INSERT_VALUES));
    }
    CHKERRQ(PetscFree2(values,indices));
    CHKERRQ(MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY));
    PetscFunctionReturn(0);
  }

  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));
  ierr = MatPreallocateInitialize(PetscObjectComm((PetscObject)dm),m,m,dnz,onz);CHKERRQ(ierr);
  /* loop over packed objects, handling one at at time */
  next = com->next;
  while (next) {
    PetscInt       nc,rstart,*ccols,maxnc;
    const PetscInt *cols,*rstarts;
    PetscMPIInt    proc;

    CHKERRQ(DMCreateMatrix(next->dm,&Atmp));
    CHKERRQ(MatGetOwnershipRange(Atmp,&rstart,NULL));
    CHKERRQ(MatGetOwnershipRanges(Atmp,&rstarts));
    CHKERRQ(MatGetLocalSize(Atmp,&mA,NULL));

    maxnc = 0;
    for (i=0; i<mA; i++) {
      CHKERRQ(MatGetRow(Atmp,rstart+i,&nc,NULL,NULL));
      maxnc = PetscMax(nc,maxnc);
      CHKERRQ(MatRestoreRow(Atmp,rstart+i,&nc,NULL,NULL));
    }
    CHKERRQ(PetscMalloc1(maxnc,&ccols));
    for (i=0; i<mA; i++) {
      CHKERRQ(MatGetRow(Atmp,rstart+i,&nc,&cols,NULL));
      /* remap the columns taking into how much they are shifted on each process */
      for (j=0; j<nc; j++) {
        proc = 0;
        while (cols[j] >= rstarts[proc+1]) proc++;
        ccols[j] = cols[j] + next->grstarts[proc] - rstarts[proc];
      }
      CHKERRQ(MatPreallocateSet(com->rstart+next->rstart+i,nc,ccols,dnz,onz));
      CHKERRQ(MatRestoreRow(Atmp,rstart+i,&nc,&cols,NULL));
    }
    CHKERRQ(PetscFree(ccols));
    CHKERRQ(MatDestroy(&Atmp));
    next = next->next;
  }
  if (com->FormCoupleLocations) {
    CHKERRQ((*com->FormCoupleLocations)(dm,NULL,dnz,onz,__rstart,__nrows,__start,__end));
  }
  CHKERRQ(MatMPIAIJSetPreallocation(*J,0,dnz,0,onz));
  CHKERRQ(MatSeqAIJSetPreallocation(*J,0,dnz));
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  if (dm->prealloc_only) PetscFunctionReturn(0);

  next = com->next;
  while (next) {
    PetscInt          nc,rstart,row,maxnc,*ccols;
    const PetscInt    *cols,*rstarts;
    const PetscScalar *values;
    PetscMPIInt       proc;

    CHKERRQ(DMCreateMatrix(next->dm,&Atmp));
    CHKERRQ(MatGetOwnershipRange(Atmp,&rstart,NULL));
    CHKERRQ(MatGetOwnershipRanges(Atmp,&rstarts));
    CHKERRQ(MatGetLocalSize(Atmp,&mA,NULL));
    maxnc = 0;
    for (i=0; i<mA; i++) {
      CHKERRQ(MatGetRow(Atmp,rstart+i,&nc,NULL,NULL));
      maxnc = PetscMax(nc,maxnc);
      CHKERRQ(MatRestoreRow(Atmp,rstart+i,&nc,NULL,NULL));
    }
    CHKERRQ(PetscMalloc1(maxnc,&ccols));
    for (i=0; i<mA; i++) {
      CHKERRQ(MatGetRow(Atmp,rstart+i,&nc,(const PetscInt**)&cols,&values));
      for (j=0; j<nc; j++) {
        proc = 0;
        while (cols[j] >= rstarts[proc+1]) proc++;
        ccols[j] = cols[j] + next->grstarts[proc] - rstarts[proc];
      }
      row  = com->rstart+next->rstart+i;
      CHKERRQ(MatSetValues(*J,1,&row,nc,ccols,values,INSERT_VALUES));
      CHKERRQ(MatRestoreRow(Atmp,rstart+i,&nc,(const PetscInt**)&cols,&values));
    }
    CHKERRQ(PetscFree(ccols));
    CHKERRQ(MatDestroy(&Atmp));
    next = next->next;
  }
  if (com->FormCoupleLocations) {
    PetscInt __rstart;
    CHKERRQ(MatGetOwnershipRange(*J,&__rstart,NULL));
    CHKERRQ((*com->FormCoupleLocations)(dm,*J,NULL,NULL,__rstart,0,0,0));
  }
  CHKERRQ(MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateMatrix_Composite(DM dm,Mat *J)
{
  PetscBool              usenest;
  ISLocalToGlobalMapping ltogmap;

  PetscFunctionBegin;
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(DMSetUp(dm));
  CHKERRQ(PetscStrcmp(dm->mattype,MATNEST,&usenest));
  if (usenest) {
    CHKERRQ(DMCreateMatrix_Composite_Nest(dm,J));
  } else {
    CHKERRQ(DMCreateMatrix_Composite_AIJ(dm,J));
  }

  CHKERRQ(DMGetLocalToGlobalMapping(dm,&ltogmap));
  CHKERRQ(MatSetLocalToGlobalMapping(*J,ltogmap,ltogmap));
  CHKERRQ(MatSetDM(*J,dm));
  PetscFunctionReturn(0);
}
