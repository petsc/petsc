#include <petsc/private/matimpl.h>      /*I "petscmat.h"  I*/
#include <petscsf.h>

PETSC_EXTERN PetscErrorCode MatColoringCreateBipartiteGraph(MatColoring,PetscSF *,PetscSF *);

PETSC_EXTERN PetscErrorCode MatColoringTestValid(MatColoring mc,ISColoring coloring)
{
  PetscErrorCode ierr;
  Mat            m=mc->mat;
  PetscSF        etor,etoc;
  PetscInt       s,e;
  PetscInt       ncolors,nrows,ncols;
  IS             *colors;
  PetscInt       i,j,k,l;
  PetscInt       *staterow,*statecol,*statespread;
  PetscInt       nindices;
  const PetscInt *indices;
  PetscInt       dist=mc->dist;
  const PetscInt *degrees;
  PetscInt       *stateleafrow,*stateleafcol,nleafrows,nleafcols,idx,nentries,maxcolors;
  MPI_Datatype   itype;

  PetscFunctionBegin;
  ierr = MatColoringGetMaxColors(mc,&maxcolors);CHKERRQ(ierr);
  ierr = PetscDataTypeToMPIDataType(PETSC_INT,&itype);CHKERRQ(ierr);
  /* get the communication structures and the colors */
  ierr = MatColoringCreateBipartiteGraph(mc,&etoc,&etor);CHKERRQ(ierr);
  ierr = ISColoringGetIS(coloring,&ncolors,&colors);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(etor,&nrows,&nleafrows,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(etoc,&ncols,&nleafcols,NULL,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(m,&s,&e);CHKERRQ(ierr);
  ierr = PetscMalloc1(ncols,&statecol);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrows,&staterow);CHKERRQ(ierr);
  ierr = PetscMalloc1(nleafcols,&stateleafcol);CHKERRQ(ierr);
  ierr = PetscMalloc1(nleafrows,&stateleafrow);CHKERRQ(ierr);

  for (l=0;l<ncolors;l++) {
    if (l > maxcolors) break;
    for (k=0;k<ncols;k++) {
      statecol[k] = -1;
    }
    ierr = ISGetLocalSize(colors[l],&nindices);CHKERRQ(ierr);
    ierr = ISGetIndices(colors[l],&indices);CHKERRQ(ierr);
    for (k=0;k<nindices;k++) {
      statecol[indices[k]-s] = indices[k];
    }
    ierr = ISRestoreIndices(colors[l],&indices);CHKERRQ(ierr);
    statespread = statecol;
    for (k=0;k<dist;k++) {
      if (k%2 == 1) {
        ierr = PetscSFComputeDegreeBegin(etor,&degrees);CHKERRQ(ierr);
        ierr = PetscSFComputeDegreeEnd(etor,&degrees);CHKERRQ(ierr);
        nentries=0;
        for(i=0;i<nrows;i++) {
          nentries += degrees[i];
        }
        idx=0;
        for(i=0;i<nrows;i++) {
          for (j=0;j<degrees[i];j++) {
            stateleafrow[idx] = staterow[i];
            idx++;
          }
          statecol[i]=0.;
        }
        if (idx != nentries) SETERRQ2(PetscObjectComm((PetscObject)mc),PETSC_ERR_NOT_CONVERGED,"Bad number of entries %d vs %d",idx,nentries);
        ierr = PetscLogEventBegin(MATCOLORING_Comm,mc,0,0,0);CHKERRQ(ierr);
        ierr = PetscSFReduceBegin(etoc,itype,stateleafrow,statecol,MPI_MAX);CHKERRQ(ierr);
        ierr = PetscSFReduceEnd(etoc,itype,stateleafrow,statecol,MPI_MAX);CHKERRQ(ierr);
        ierr = PetscLogEventEnd(MATCOLORING_Comm,mc,0,0,0);CHKERRQ(ierr);
        statespread = statecol;
      } else {
        ierr = PetscSFComputeDegreeBegin(etoc,&degrees);CHKERRQ(ierr);
        ierr = PetscSFComputeDegreeEnd(etoc,&degrees);CHKERRQ(ierr);
        nentries=0;
        for(i=0;i<ncols;i++) {
          nentries += degrees[i];
        }
        idx=0;
        for(i=0;i<ncols;i++) {
          for (j=0;j<degrees[i];j++) {
            stateleafcol[idx] = statecol[i];
            idx++;
          }
          staterow[i]=0.;
        }
        if (idx != nentries) SETERRQ2(PetscObjectComm((PetscObject)mc),PETSC_ERR_NOT_CONVERGED,"Bad number of entries %d vs %d",idx,nentries);
        ierr = PetscLogEventBegin(MATCOLORING_Comm,mc,0,0,0);CHKERRQ(ierr);
        ierr = PetscSFReduceBegin(etor,itype,stateleafcol,staterow,MPI_MAX);CHKERRQ(ierr);
        ierr = PetscSFReduceEnd(etor,itype,stateleafcol,staterow,MPI_MAX);CHKERRQ(ierr);
        ierr = PetscLogEventEnd(MATCOLORING_Comm,mc,0,0,0);CHKERRQ(ierr);
        statespread = staterow;
      }
    }
    for (k=0;k<nindices;k++) {
      if (statespread[indices[k]-s] != indices[k]) {
        ierr = PetscPrintf(PetscObjectComm((PetscObject)mc),"%d of color %d conflicts with %d\n",indices[k],l,statespread[indices[k]-s]);CHKERRQ(ierr);
      }
    }
    ierr = ISRestoreIndices(colors[l],&indices);CHKERRQ(ierr);
  }
  ierr = PetscFree(statecol);CHKERRQ(ierr);
  ierr = PetscFree(staterow);CHKERRQ(ierr);
  ierr = PetscFree(stateleafcol);CHKERRQ(ierr);
  ierr = PetscFree(stateleafrow);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&etor);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&etoc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
