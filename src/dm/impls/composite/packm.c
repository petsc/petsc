
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
  PetscCall(DMCompositeGetGlobalISs(dm,&isg));

  /* Get submatrices */
  PetscCall(PetscMalloc1(n*n,&submats));
  for (i=0,rlink=com->next; rlink; i++,rlink=rlink->next) {
    for (j=0,clink=com->next; clink; j++,clink=clink->next) {
      Mat sub = NULL;
      if (i == j) {
        PetscCall(DMCreateMatrix(rlink->dm,&sub));
      } else PetscCheck(!com->FormCoupleLocations,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Cannot manage off-diagonal parts yet");
      submats[i*n+j] = sub;
    }
  }

  PetscCall(MatCreateNest(PetscObjectComm((PetscObject)dm),n,isg,n,isg,submats,J));

  /* Disown references */
  for (i=0; i<n; i++) PetscCall(ISDestroy(&isg[i]));
  PetscCall(PetscFree(isg));

  for (i=0; i<n*n; i++) {
    if (submats[i]) PetscCall(MatDestroy(&submats[i]));
  }
  PetscCall(PetscFree(submats));
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

  PetscCall(MatCreate(PetscObjectComm((PetscObject)dm),J));
  PetscCall(MatSetSizes(*J,m,m,PETSC_DETERMINE,PETSC_DETERMINE));
  PetscCall(MatSetType(*J,dm->mattype));

  /*
     Extremely inefficient but will compute entire Jacobian for testing
  */
  PetscCall(PetscOptionsGetBool(((PetscObject)dm)->options,((PetscObject)dm)->prefix,"-dmcomposite_dense_jacobian",&dense,NULL));
  if (dense) {
    PetscInt    rstart,rend,*indices;
    PetscScalar *values;

    mA   = com->N;
    PetscCall(MatMPIAIJSetPreallocation(*J,mA,NULL,mA-m,NULL));
    PetscCall(MatSeqAIJSetPreallocation(*J,mA,NULL));

    PetscCall(MatGetOwnershipRange(*J,&rstart,&rend));
    PetscCall(PetscMalloc2(mA,&values,mA,&indices));
    PetscCall(PetscArrayzero(values,mA));
    for (i=0; i<mA; i++) indices[i] = i;
    for (i=rstart; i<rend; i++) {
      PetscCall(MatSetValues(*J,1,&i,mA,indices,values,INSERT_VALUES));
    }
    PetscCall(PetscFree2(values,indices));
    PetscCall(MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY));
    PetscFunctionReturn(0);
  }

  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));
  ierr = MatPreallocateInitialize(PetscObjectComm((PetscObject)dm),m,m,dnz,onz);PetscCall(ierr);
  /* loop over packed objects, handling one at at time */
  next = com->next;
  while (next) {
    PetscInt       nc,rstart,*ccols,maxnc;
    const PetscInt *cols,*rstarts;
    PetscMPIInt    proc;

    PetscCall(DMCreateMatrix(next->dm,&Atmp));
    PetscCall(MatGetOwnershipRange(Atmp,&rstart,NULL));
    PetscCall(MatGetOwnershipRanges(Atmp,&rstarts));
    PetscCall(MatGetLocalSize(Atmp,&mA,NULL));

    maxnc = 0;
    for (i=0; i<mA; i++) {
      PetscCall(MatGetRow(Atmp,rstart+i,&nc,NULL,NULL));
      maxnc = PetscMax(nc,maxnc);
      PetscCall(MatRestoreRow(Atmp,rstart+i,&nc,NULL,NULL));
    }
    PetscCall(PetscMalloc1(maxnc,&ccols));
    for (i=0; i<mA; i++) {
      PetscCall(MatGetRow(Atmp,rstart+i,&nc,&cols,NULL));
      /* remap the columns taking into how much they are shifted on each process */
      for (j=0; j<nc; j++) {
        proc = 0;
        while (cols[j] >= rstarts[proc+1]) proc++;
        ccols[j] = cols[j] + next->grstarts[proc] - rstarts[proc];
      }
      PetscCall(MatPreallocateSet(com->rstart+next->rstart+i,nc,ccols,dnz,onz));
      PetscCall(MatRestoreRow(Atmp,rstart+i,&nc,&cols,NULL));
    }
    PetscCall(PetscFree(ccols));
    PetscCall(MatDestroy(&Atmp));
    next = next->next;
  }
  if (com->FormCoupleLocations) {
    PetscCall((*com->FormCoupleLocations)(dm,NULL,dnz,onz,__rstart,__nrows,__start,__end));
  }
  PetscCall(MatMPIAIJSetPreallocation(*J,0,dnz,0,onz));
  PetscCall(MatSeqAIJSetPreallocation(*J,0,dnz));
  ierr = MatPreallocateFinalize(dnz,onz);PetscCall(ierr);

  if (dm->prealloc_only) PetscFunctionReturn(0);

  next = com->next;
  while (next) {
    PetscInt          nc,rstart,row,maxnc,*ccols;
    const PetscInt    *cols,*rstarts;
    const PetscScalar *values;
    PetscMPIInt       proc;

    PetscCall(DMCreateMatrix(next->dm,&Atmp));
    PetscCall(MatGetOwnershipRange(Atmp,&rstart,NULL));
    PetscCall(MatGetOwnershipRanges(Atmp,&rstarts));
    PetscCall(MatGetLocalSize(Atmp,&mA,NULL));
    maxnc = 0;
    for (i=0; i<mA; i++) {
      PetscCall(MatGetRow(Atmp,rstart+i,&nc,NULL,NULL));
      maxnc = PetscMax(nc,maxnc);
      PetscCall(MatRestoreRow(Atmp,rstart+i,&nc,NULL,NULL));
    }
    PetscCall(PetscMalloc1(maxnc,&ccols));
    for (i=0; i<mA; i++) {
      PetscCall(MatGetRow(Atmp,rstart+i,&nc,(const PetscInt**)&cols,&values));
      for (j=0; j<nc; j++) {
        proc = 0;
        while (cols[j] >= rstarts[proc+1]) proc++;
        ccols[j] = cols[j] + next->grstarts[proc] - rstarts[proc];
      }
      row  = com->rstart+next->rstart+i;
      PetscCall(MatSetValues(*J,1,&row,nc,ccols,values,INSERT_VALUES));
      PetscCall(MatRestoreRow(Atmp,rstart+i,&nc,(const PetscInt**)&cols,&values));
    }
    PetscCall(PetscFree(ccols));
    PetscCall(MatDestroy(&Atmp));
    next = next->next;
  }
  if (com->FormCoupleLocations) {
    PetscInt __rstart;
    PetscCall(MatGetOwnershipRange(*J,&__rstart,NULL));
    PetscCall((*com->FormCoupleLocations)(dm,*J,NULL,NULL,__rstart,0,0,0));
  }
  PetscCall(MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateMatrix_Composite(DM dm,Mat *J)
{
  PetscBool              usenest;
  ISLocalToGlobalMapping ltogmap;

  PetscFunctionBegin;
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(PetscStrcmp(dm->mattype,MATNEST,&usenest));
  if (usenest) {
    PetscCall(DMCreateMatrix_Composite_Nest(dm,J));
  } else {
    PetscCall(DMCreateMatrix_Composite_AIJ(dm,J));
  }

  PetscCall(DMGetLocalToGlobalMapping(dm,&ltogmap));
  PetscCall(MatSetLocalToGlobalMapping(*J,ltogmap,ltogmap));
  PetscCall(MatSetDM(*J,dm));
  PetscFunctionReturn(0);
}
