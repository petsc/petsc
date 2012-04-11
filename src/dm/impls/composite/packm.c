
#include "packimpl.h" /*I   "petscdmcomposite.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DMCreateMatrix_Composite_Nest"
static PetscErrorCode DMCreateMatrix_Composite_Nest(DM dm,const MatType mtype,Mat *J)
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
  ierr = PetscMalloc(n*n*sizeof(Mat),&submats);CHKERRQ(ierr);
  for (i=0,rlink=com->next; rlink; i++,rlink=rlink->next) {
    for (j=0,clink=com->next; clink; j++,clink=clink->next) {
      Mat sub = PETSC_NULL;
      if (i == j) {
        ierr = DMCreateMatrix(rlink->dm,PETSC_NULL,&sub);CHKERRQ(ierr);
      } else if (com->FormCoupleLocations) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"Cannot manage off-diagonal parts yet");
      submats[i*n+j] = sub;
    }
  }

  ierr = MatCreateNest(((PetscObject)dm)->comm,n,isg,n,isg,submats,J);CHKERRQ(ierr);

  /* Disown references */
  for (i=0; i<n; i++) {ierr = ISDestroy(&isg[i]);CHKERRQ(ierr);}
  ierr = PetscFree(isg);CHKERRQ(ierr);

  for (i=0; i<n*n; i++) {
    if (submats[i]) {ierr = MatDestroy(&submats[i]);CHKERRQ(ierr);}
  }
  ierr = PetscFree(submats);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateMatrix_Composite_AIJ"
static PetscErrorCode DMCreateMatrix_Composite_AIJ(DM dm,const MatType mtype,Mat *J)
{
  PetscErrorCode         ierr;
  DM_Composite           *com = (DM_Composite*)dm->data;
  struct DMCompositeLink *next = com->next;
  PetscInt               m,*dnz,*onz,i,j,mA;
  Mat                    Atmp;
  PetscMPIInt            rank;
  PetscBool              dense = PETSC_FALSE; 

  PetscFunctionBegin;
  /* use global vector to determine layout needed for matrix */
  m = com->n;

  ierr = MatCreate(((PetscObject)dm)->comm,J);CHKERRQ(ierr);
  ierr = MatSetSizes(*J,m,m,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(*J,mtype);CHKERRQ(ierr);

  /*
     Extremely inefficient but will compute entire Jacobian for testing
  */
  ierr = PetscOptionsGetBool(((PetscObject)dm)->prefix,"-dmcomposite_dense_jacobian",&dense,PETSC_NULL);CHKERRQ(ierr);
  if (dense) {
    PetscInt    rstart,rend,*indices;
    PetscScalar *values;

    mA    = com->N;
    ierr = MatMPIAIJSetPreallocation(*J,mA,PETSC_NULL,mA-m,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(*J,mA,PETSC_NULL);CHKERRQ(ierr);

    ierr = MatGetOwnershipRange(*J,&rstart,&rend);CHKERRQ(ierr);
    ierr = PetscMalloc2(mA,PetscScalar,&values,mA,PetscInt,&indices);CHKERRQ(ierr);
    ierr = PetscMemzero(values,mA*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0; i<mA; i++) indices[i] = i;
    for (i=rstart; i<rend; i++) {
      ierr = MatSetValues(*J,1,&i,mA,indices,values,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = PetscFree2(values,indices);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = MPI_Comm_rank(((PetscObject)dm)->comm,&rank);CHKERRQ(ierr);
  ierr = MatPreallocateInitialize(((PetscObject)dm)->comm,m,m,dnz,onz);CHKERRQ(ierr);
  /* loop over packed objects, handling one at at time */
  next = com->next;
  while (next) {
    PetscInt       nc,rstart,*ccols,maxnc;
    const PetscInt *cols,*rstarts;
    PetscMPIInt    proc;

    ierr = DMCreateMatrix(next->dm,mtype,&Atmp);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(Atmp,&rstart,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatGetOwnershipRanges(Atmp,&rstarts);CHKERRQ(ierr);
    ierr = MatGetLocalSize(Atmp,&mA,PETSC_NULL);CHKERRQ(ierr);

    maxnc = 0;
    for (i=0; i<mA; i++) {
      ierr  = MatGetRow(Atmp,rstart+i,&nc,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      ierr  = MatRestoreRow(Atmp,rstart+i,&nc,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      maxnc = PetscMax(nc,maxnc);
    }
    ierr = PetscMalloc(maxnc*sizeof(PetscInt),&ccols);CHKERRQ(ierr);
    for (i=0; i<mA; i++) {
      ierr = MatGetRow(Atmp,rstart+i,&nc,&cols,PETSC_NULL);CHKERRQ(ierr);
      /* remap the columns taking into how much they are shifted on each process */
      for (j=0; j<nc; j++) {
        proc = 0;
        while (cols[j] >= rstarts[proc+1]) proc++;
        ccols[j] = cols[j] + next->grstarts[proc] - rstarts[proc];
      }
      ierr = MatPreallocateSet(com->rstart+next->rstart+i,nc,ccols,dnz,onz);CHKERRQ(ierr);
      ierr = MatRestoreRow(Atmp,rstart+i,&nc,&cols,PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = PetscFree(ccols);CHKERRQ(ierr);
    ierr = MatDestroy(&Atmp);CHKERRQ(ierr);
    next = next->next;
  }
  if (com->FormCoupleLocations) {
    ierr = (*com->FormCoupleLocations)(dm,PETSC_NULL,dnz,onz,__rstart,__nrows,__start,__end);CHKERRQ(ierr);
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

    ierr = DMCreateMatrix(next->dm,mtype,&Atmp);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(Atmp,&rstart,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatGetOwnershipRanges(Atmp,&rstarts);CHKERRQ(ierr);
    ierr = MatGetLocalSize(Atmp,&mA,PETSC_NULL);CHKERRQ(ierr);
    maxnc = 0;
    for (i=0; i<mA; i++) {
      ierr  = MatGetRow(Atmp,rstart+i,&nc,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      ierr  = MatRestoreRow(Atmp,rstart+i,&nc,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      maxnc = PetscMax(nc,maxnc);
    }
    ierr = PetscMalloc(maxnc*sizeof(PetscInt),&ccols);CHKERRQ(ierr);
    for (i=0; i<mA; i++) {
      ierr = MatGetRow(Atmp,rstart+i,&nc,(const PetscInt **)&cols,&values);CHKERRQ(ierr);
      for (j=0; j<nc; j++) {
        proc = 0;
        while (cols[j] >= rstarts[proc+1]) proc++;
        ccols[j] = cols[j] + next->grstarts[proc] - rstarts[proc];
      }
      row  = com->rstart+next->rstart+i;
      ierr = MatSetValues(*J,1,&row,nc,ccols,values,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(Atmp,rstart+i,&nc,(const PetscInt **)&cols,&values);CHKERRQ(ierr);
    }
    ierr = PetscFree(ccols);CHKERRQ(ierr);
    ierr = MatDestroy(&Atmp);CHKERRQ(ierr);
    next = next->next;
  }
  if (com->FormCoupleLocations) {
    PetscInt __rstart;
    ierr = MatGetOwnershipRange(*J,&__rstart,PETSC_NULL);CHKERRQ(ierr);
    ierr = (*com->FormCoupleLocations)(dm,*J,PETSC_NULL,PETSC_NULL,__rstart,0,0,0);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateMatrix_Composite"
PetscErrorCode DMCreateMatrix_Composite(DM dm,const MatType mtype,Mat *J)
{
  PetscErrorCode         ierr;
  PetscBool              usenest;
  ISLocalToGlobalMapping ltogmap,ltogmapb;

  PetscFunctionBegin;
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = PetscStrcmp(mtype,MATNEST,&usenest);CHKERRQ(ierr);
  if (usenest) {
    ierr = DMCreateMatrix_Composite_Nest(dm,mtype,J);CHKERRQ(ierr);
  } else {
    ierr = DMCreateMatrix_Composite_AIJ(dm,mtype?mtype:MATAIJ,J);CHKERRQ(ierr);
  }

  ierr = DMGetLocalToGlobalMapping(dm,&ltogmap);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMappingBlock(dm,&ltogmapb);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*J,ltogmap,ltogmap);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(*J,ltogmapb,ltogmapb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
