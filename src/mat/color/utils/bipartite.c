#include <petsc/private/matimpl.h>      /*I "petscmat.h"  I*/
#include <petscsf.h>

PETSC_EXTERN PetscErrorCode MatColoringCreateBipartiteGraph(MatColoring mc,PetscSF *etoc,PetscSF *etor)
{
  PetscInt          nentries,ncolentries,idx;
  PetscInt          i,j,rs,re,cs,ce,cn;
  PetscInt          *rowleaf,*colleaf,*rowdata;
  PetscInt          ncol;
  const PetscScalar *vcol;
  const PetscInt    *icol;
  const PetscInt    *coldegrees,*rowdegrees;
  Mat               m = mc->mat;

  PetscFunctionBegin;
  CHKERRQ(MatGetOwnershipRange(m,&rs,&re));
  CHKERRQ(MatGetOwnershipRangeColumn(m,&cs,&ce));
  cn = ce-cs;
  nentries=0;
  for (i=rs;i<re;i++) {
    CHKERRQ(MatGetRow(m,i,&ncol,NULL,&vcol));
    for (j=0;j<ncol;j++) {
      nentries++;
    }
    CHKERRQ(MatRestoreRow(m,i,&ncol,NULL,&vcol));
  }
  CHKERRQ(PetscMalloc1(nentries,&rowleaf));
  CHKERRQ(PetscMalloc1(nentries,&rowdata));
  idx=0;
  for (i=rs;i<re;i++) {
    CHKERRQ(MatGetRow(m,i,&ncol,&icol,&vcol));
    for (j=0;j<ncol;j++) {
      rowleaf[idx] = icol[j];
      rowdata[idx] = i;
      idx++;
    }
    CHKERRQ(MatRestoreRow(m,i,&ncol,&icol,&vcol));
  }
  PetscCheckFalse(idx != nentries,PetscObjectComm((PetscObject)m),PETSC_ERR_NOT_CONVERGED,"Bad number of entries %" PetscInt_FMT " vs %" PetscInt_FMT,idx,nentries);
  CHKERRQ(PetscSFCreate(PetscObjectComm((PetscObject)m),etoc));
  CHKERRQ(PetscSFCreate(PetscObjectComm((PetscObject)m),etor));

  CHKERRQ(PetscSFSetGraphLayout(*etoc,m->cmap,nentries,NULL,PETSC_COPY_VALUES,rowleaf));
  CHKERRQ(PetscSFSetFromOptions(*etoc));

  /* determine the number of entries in the column matrix */
  CHKERRQ(PetscLogEventBegin(MATCOLORING_Comm,*etoc,0,0,0));
  CHKERRQ(PetscSFComputeDegreeBegin(*etoc,&coldegrees));
  CHKERRQ(PetscSFComputeDegreeEnd(*etoc,&coldegrees));
  CHKERRQ(PetscLogEventEnd(MATCOLORING_Comm,*etoc,0,0,0));
  ncolentries=0;
  for (i=0;i<cn;i++) {
    ncolentries += coldegrees[i];
  }
  CHKERRQ(PetscMalloc1(ncolentries,&colleaf));

  /* create the one going the other way by building the leaf set */
  CHKERRQ(PetscLogEventBegin(MATCOLORING_Comm,*etoc,0,0,0));
  CHKERRQ(PetscSFGatherBegin(*etoc,MPIU_INT,rowdata,colleaf));
  CHKERRQ(PetscSFGatherEnd(*etoc,MPIU_INT,rowdata,colleaf));
  CHKERRQ(PetscLogEventEnd(MATCOLORING_Comm,*etoc,0,0,0));

  /* this one takes mat entries in *columns* to rows -- you never have to actually be able to order the leaf entries. */
  CHKERRQ(PetscSFSetGraphLayout(*etor,m->rmap,ncolentries,NULL,PETSC_COPY_VALUES,colleaf));
  CHKERRQ(PetscSFSetFromOptions(*etor));

  CHKERRQ(PetscLogEventBegin(MATCOLORING_Comm,*etor,0,0,0));
  CHKERRQ(PetscSFComputeDegreeBegin(*etor,&rowdegrees));
  CHKERRQ(PetscSFComputeDegreeEnd(*etor,&rowdegrees));
  CHKERRQ(PetscLogEventEnd(MATCOLORING_Comm,*etor,0,0,0));

  CHKERRQ(PetscFree(rowdata));
  CHKERRQ(PetscFree(rowleaf));
  CHKERRQ(PetscFree(colleaf));
  PetscFunctionReturn(0);
}
