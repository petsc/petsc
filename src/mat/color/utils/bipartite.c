#include <petsc/private/matimpl.h>      /*I "petscmat.h"  I*/
#include <petscsf.h>

PETSC_EXTERN PetscErrorCode MatColoringCreateBipartiteGraph(MatColoring mc,PetscSF *etoc,PetscSF *etor)
{
  PetscErrorCode    ierr;
  PetscInt          nentries,ncolentries,idx;
  PetscInt          i,j,rs,re,cs,ce,cn;
  PetscInt          *rowleaf,*colleaf,*rowdata;
  PetscInt          ncol;
  const PetscScalar *vcol;
  const PetscInt    *icol;
  const PetscInt    *coldegrees,*rowdegrees;
  Mat               m = mc->mat;

  PetscFunctionBegin;
  ierr = MatGetOwnershipRange(m,&rs,&re);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(m,&cs,&ce);CHKERRQ(ierr);
  cn = ce-cs;
  nentries=0;
  for (i=rs;i<re;i++) {
    ierr = MatGetRow(m,i,&ncol,NULL,&vcol);CHKERRQ(ierr);
    for (j=0;j<ncol;j++) {
      nentries++;
    }
    ierr = MatRestoreRow(m,i,&ncol,NULL,&vcol);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(nentries,&rowleaf);CHKERRQ(ierr);
  ierr = PetscMalloc1(nentries,&rowdata);CHKERRQ(ierr);
  idx=0;
  for (i=rs;i<re;i++) {
    ierr = MatGetRow(m,i,&ncol,&icol,&vcol);CHKERRQ(ierr);
    for (j=0;j<ncol;j++) {
      rowleaf[idx] = icol[j];
      rowdata[idx] = i;
      idx++;
    }
    ierr = MatRestoreRow(m,i,&ncol,&icol,&vcol);CHKERRQ(ierr);
  }
  PetscCheckFalse(idx != nentries,PetscObjectComm((PetscObject)m),PETSC_ERR_NOT_CONVERGED,"Bad number of entries %" PetscInt_FMT " vs %" PetscInt_FMT,idx,nentries);
  ierr = PetscSFCreate(PetscObjectComm((PetscObject)m),etoc);CHKERRQ(ierr);
  ierr = PetscSFCreate(PetscObjectComm((PetscObject)m),etor);CHKERRQ(ierr);

  ierr = PetscSFSetGraphLayout(*etoc,m->cmap,nentries,NULL,PETSC_COPY_VALUES,rowleaf);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(*etoc);CHKERRQ(ierr);

  /* determine the number of entries in the column matrix */
  ierr = PetscLogEventBegin(MATCOLORING_Comm,*etoc,0,0,0);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeBegin(*etoc,&coldegrees);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(*etoc,&coldegrees);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MATCOLORING_Comm,*etoc,0,0,0);CHKERRQ(ierr);
  ncolentries=0;
  for (i=0;i<cn;i++) {
    ncolentries += coldegrees[i];
  }
  ierr = PetscMalloc1(ncolentries,&colleaf);CHKERRQ(ierr);

  /* create the one going the other way by building the leaf set */
  ierr = PetscLogEventBegin(MATCOLORING_Comm,*etoc,0,0,0);CHKERRQ(ierr);
  ierr = PetscSFGatherBegin(*etoc,MPIU_INT,rowdata,colleaf);CHKERRQ(ierr);
  ierr = PetscSFGatherEnd(*etoc,MPIU_INT,rowdata,colleaf);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MATCOLORING_Comm,*etoc,0,0,0);CHKERRQ(ierr);

  /* this one takes mat entries in *columns* to rows -- you never have to actually be able to order the leaf entries. */
  ierr = PetscSFSetGraphLayout(*etor,m->rmap,ncolentries,NULL,PETSC_COPY_VALUES,colleaf);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(*etor);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MATCOLORING_Comm,*etor,0,0,0);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeBegin(*etor,&rowdegrees);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(*etor,&rowdegrees);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MATCOLORING_Comm,*etor,0,0,0);CHKERRQ(ierr);

  ierr = PetscFree(rowdata);CHKERRQ(ierr);
  ierr = PetscFree(rowleaf);CHKERRQ(ierr);
  ierr = PetscFree(colleaf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
