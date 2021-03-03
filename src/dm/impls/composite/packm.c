
#include <../src/dm/impls/composite/packimpl.h>       /*I  "petscdmcomposite.h"  I*/

static PetscErrorCode DMCreateMatrix_Composite_Nest(DM dm,Mat *J)
{
  const DM_Composite           *com = (DM_Composite*)dm->data;
  const struct DMCompositeLink *rlink,*clink;
  PetscErrorCode               ierr;
  IS                           *isg;
  Mat                          *submats;
  PetscInt                     i,j,n;

  PetscFunctionBegin;
  n = com->nDM;                 /* Total number of entries */

  /* Explicit index sets are not required for MatCreateNest, but getting them here allows MatNest to do consistency
   * checking and allows ISEqual to compare by identity instead of by contents. */
  ierr = DMCompositeGetGlobalISs(dm,&isg);CHKERRQ(ierr);

  /* Get submatrices */
  ierr = PetscMalloc1(n*n,&submats);CHKERRQ(ierr);
  for (i=0,rlink=com->next; rlink; i++,rlink=rlink->next) {
    for (j=0,clink=com->next; clink; j++,clink=clink->next) {
      Mat sub = NULL;
      if (i == j) {
        ierr = DMCreateMatrix(rlink->dm,&sub);CHKERRQ(ierr);
      } else if (com->FormCoupleLocations) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Cannot manage off-diagonal parts yet");
      submats[i*n+j] = sub;
    }
  }

  ierr = MatCreateNest(PetscObjectComm((PetscObject)dm),n,isg,n,isg,submats,J);CHKERRQ(ierr);

  /* Disown references */
  for (i=0; i<n; i++) {ierr = ISDestroy(&isg[i]);CHKERRQ(ierr);}
  ierr = PetscFree(isg);CHKERRQ(ierr);

  for (i=0; i<n*n; i++) {
    if (submats[i]) {ierr = MatDestroy(&submats[i]);CHKERRQ(ierr);}
  }
  ierr = PetscFree(submats);CHKERRQ(ierr);
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

  ierr = MatCreate(PetscObjectComm((PetscObject)dm),J);CHKERRQ(ierr);
  ierr = MatSetSizes(*J,m,m,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(*J,dm->mattype);CHKERRQ(ierr);

  /*
     Extremely inefficient but will compute entire Jacobian for testing
  */
  ierr = PetscOptionsGetBool(((PetscObject)dm)->options,((PetscObject)dm)->prefix,"-dmcomposite_dense_jacobian",&dense,NULL);CHKERRQ(ierr);
  if (dense) {
    PetscInt    rstart,rend,*indices;
    PetscScalar *values;

    mA   = com->N;
    ierr = MatMPIAIJSetPreallocation(*J,mA,NULL,mA-m,NULL);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(*J,mA,NULL);CHKERRQ(ierr);

    ierr = MatGetOwnershipRange(*J,&rstart,&rend);CHKERRQ(ierr);
    ierr = PetscMalloc2(mA,&values,mA,&indices);CHKERRQ(ierr);
    ierr = PetscArrayzero(values,mA);CHKERRQ(ierr);
    for (i=0; i<mA; i++) indices[i] = i;
    for (i=rstart; i<rend; i++) {
      ierr = MatSetValues(*J,1,&i,mA,indices,values,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = PetscFree2(values,indices);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRMPI(ierr);
  ierr = MatPreallocateInitialize(PetscObjectComm((PetscObject)dm),m,m,dnz,onz);CHKERRQ(ierr);
  /* loop over packed objects, handling one at at time */
  next = com->next;
  while (next) {
    PetscInt       nc,rstart,*ccols,maxnc;
    const PetscInt *cols,*rstarts;
    PetscMPIInt    proc;

    ierr = DMCreateMatrix(next->dm,&Atmp);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(Atmp,&rstart,NULL);CHKERRQ(ierr);
    ierr = MatGetOwnershipRanges(Atmp,&rstarts);CHKERRQ(ierr);
    ierr = MatGetLocalSize(Atmp,&mA,NULL);CHKERRQ(ierr);

    maxnc = 0;
    for (i=0; i<mA; i++) {
      ierr  = MatGetRow(Atmp,rstart+i,&nc,NULL,NULL);CHKERRQ(ierr);
      maxnc = PetscMax(nc,maxnc);
      ierr  = MatRestoreRow(Atmp,rstart+i,&nc,NULL,NULL);CHKERRQ(ierr);
    }
    ierr = PetscMalloc1(maxnc,&ccols);CHKERRQ(ierr);
    for (i=0; i<mA; i++) {
      ierr = MatGetRow(Atmp,rstart+i,&nc,&cols,NULL);CHKERRQ(ierr);
      /* remap the columns taking into how much they are shifted on each process */
      for (j=0; j<nc; j++) {
        proc = 0;
        while (cols[j] >= rstarts[proc+1]) proc++;
        ccols[j] = cols[j] + next->grstarts[proc] - rstarts[proc];
      }
      ierr = MatPreallocateSet(com->rstart+next->rstart+i,nc,ccols,dnz,onz);CHKERRQ(ierr);
      ierr = MatRestoreRow(Atmp,rstart+i,&nc,&cols,NULL);CHKERRQ(ierr);
    }
    ierr = PetscFree(ccols);CHKERRQ(ierr);
    ierr = MatDestroy(&Atmp);CHKERRQ(ierr);
    next = next->next;
  }
  if (com->FormCoupleLocations) {
    ierr = (*com->FormCoupleLocations)(dm,NULL,dnz,onz,__rstart,__nrows,__start,__end);CHKERRQ(ierr);
  }
  ierr = MatMPIAIJSetPreallocation(*J,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*J,0,dnz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  if (dm->prealloc_only) PetscFunctionReturn(0);

  next = com->next;
  while (next) {
    PetscInt          nc,rstart,row,maxnc,*ccols;
    const PetscInt    *cols,*rstarts;
    const PetscScalar *values;
    PetscMPIInt       proc;

    ierr  = DMCreateMatrix(next->dm,&Atmp);CHKERRQ(ierr);
    ierr  = MatGetOwnershipRange(Atmp,&rstart,NULL);CHKERRQ(ierr);
    ierr  = MatGetOwnershipRanges(Atmp,&rstarts);CHKERRQ(ierr);
    ierr  = MatGetLocalSize(Atmp,&mA,NULL);CHKERRQ(ierr);
    maxnc = 0;
    for (i=0; i<mA; i++) {
      ierr  = MatGetRow(Atmp,rstart+i,&nc,NULL,NULL);CHKERRQ(ierr);
      maxnc = PetscMax(nc,maxnc);
      ierr  = MatRestoreRow(Atmp,rstart+i,&nc,NULL,NULL);CHKERRQ(ierr);
    }
    ierr = PetscMalloc1(maxnc,&ccols);CHKERRQ(ierr);
    for (i=0; i<mA; i++) {
      ierr = MatGetRow(Atmp,rstart+i,&nc,(const PetscInt**)&cols,&values);CHKERRQ(ierr);
      for (j=0; j<nc; j++) {
        proc = 0;
        while (cols[j] >= rstarts[proc+1]) proc++;
        ccols[j] = cols[j] + next->grstarts[proc] - rstarts[proc];
      }
      row  = com->rstart+next->rstart+i;
      ierr = MatSetValues(*J,1,&row,nc,ccols,values,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(Atmp,rstart+i,&nc,(const PetscInt**)&cols,&values);CHKERRQ(ierr);
    }
    ierr = PetscFree(ccols);CHKERRQ(ierr);
    ierr = MatDestroy(&Atmp);CHKERRQ(ierr);
    next = next->next;
  }
  if (com->FormCoupleLocations) {
    PetscInt __rstart;
    ierr = MatGetOwnershipRange(*J,&__rstart,NULL);CHKERRQ(ierr);
    ierr = (*com->FormCoupleLocations)(dm,*J,NULL,NULL,__rstart,0,0,0);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateMatrix_Composite(DM dm,Mat *J)
{
  PetscErrorCode         ierr;
  PetscBool              usenest;
  ISLocalToGlobalMapping ltogmap;

  PetscFunctionBegin;
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = PetscStrcmp(dm->mattype,MATNEST,&usenest);CHKERRQ(ierr);
  if (usenest) {
    ierr = DMCreateMatrix_Composite_Nest(dm,J);CHKERRQ(ierr);
  } else {
    ierr = DMCreateMatrix_Composite_AIJ(dm,J);CHKERRQ(ierr);
  }

  ierr = DMGetLocalToGlobalMapping(dm,&ltogmap);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*J,ltogmap,ltogmap);CHKERRQ(ierr);
  ierr = MatSetDM(*J,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
