#include <petsc/private/matimpl.h>      /*I "petscmat.h"  I*/
#include <petscsf.h>

PETSC_EXTERN PetscErrorCode MatColoringCreateBipartiteGraph(MatColoring,PetscSF *,PetscSF *);

PETSC_EXTERN PetscErrorCode MatColoringTest(MatColoring mc,ISColoring coloring)
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
  MPI_Datatype   itype = MPIU_INT;

  PetscFunctionBegin;
  ierr = MatColoringGetMaxColors(mc,&maxcolors);CHKERRQ(ierr);
  /* get the communication structures and the colors */
  ierr = MatColoringCreateBipartiteGraph(mc,&etoc,&etor);CHKERRQ(ierr);
  ierr = ISColoringGetIS(coloring,PETSC_USE_POINTER,&ncolors,&colors);CHKERRQ(ierr);
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
        for (i=0;i<nrows;i++) {
          nentries += degrees[i];
        }
        idx=0;
        for (i=0;i<nrows;i++) {
          for (j=0;j<degrees[i];j++) {
            stateleafrow[idx] = staterow[i];
            idx++;
          }
          statecol[i]=0.;
        }
        PetscCheckFalse(idx != nentries,PetscObjectComm((PetscObject)mc),PETSC_ERR_NOT_CONVERGED,"Bad number of entries %" PetscInt_FMT " vs %" PetscInt_FMT,idx,nentries);
        ierr = PetscLogEventBegin(MATCOLORING_Comm,mc,0,0,0);CHKERRQ(ierr);
        ierr = PetscSFReduceBegin(etoc,itype,stateleafrow,statecol,MPI_MAX);CHKERRQ(ierr);
        ierr = PetscSFReduceEnd(etoc,itype,stateleafrow,statecol,MPI_MAX);CHKERRQ(ierr);
        ierr = PetscLogEventEnd(MATCOLORING_Comm,mc,0,0,0);CHKERRQ(ierr);
        statespread = statecol;
      } else {
        ierr = PetscSFComputeDegreeBegin(etoc,&degrees);CHKERRQ(ierr);
        ierr = PetscSFComputeDegreeEnd(etoc,&degrees);CHKERRQ(ierr);
        nentries=0;
        for (i=0;i<ncols;i++) {
          nentries += degrees[i];
        }
        idx=0;
        for (i=0;i<ncols;i++) {
          for (j=0;j<degrees[i];j++) {
            stateleafcol[idx] = statecol[i];
            idx++;
          }
          staterow[i]=0.;
        }
        PetscCheckFalse(idx != nentries,PetscObjectComm((PetscObject)mc),PETSC_ERR_NOT_CONVERGED,"Bad number of entries %" PetscInt_FMT " vs %" PetscInt_FMT,idx,nentries);
        ierr = PetscLogEventBegin(MATCOLORING_Comm,mc,0,0,0);CHKERRQ(ierr);
        ierr = PetscSFReduceBegin(etor,itype,stateleafcol,staterow,MPI_MAX);CHKERRQ(ierr);
        ierr = PetscSFReduceEnd(etor,itype,stateleafcol,staterow,MPI_MAX);CHKERRQ(ierr);
        ierr = PetscLogEventEnd(MATCOLORING_Comm,mc,0,0,0);CHKERRQ(ierr);
        statespread = staterow;
      }
    }
    for (k=0;k<nindices;k++) {
      if (statespread[indices[k]-s] != indices[k]) {
        ierr = PetscPrintf(PetscObjectComm((PetscObject)mc),"%" PetscInt_FMT " of color %" PetscInt_FMT " conflicts with %" PetscInt_FMT "\n",indices[k],l,statespread[indices[k]-s]);CHKERRQ(ierr);
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

PETSC_EXTERN PetscErrorCode MatISColoringTest(Mat A,ISColoring iscoloring)
{
  PetscErrorCode ierr;
  PetscInt       nn,c,i,j,M,N,nc,nnz,col,row;
  const PetscInt *cia,*cja,*cols;
  IS             *isis;
  MPI_Comm       comm;
  PetscMPIInt    size;
  PetscBool      done;
  PetscBT        table;

  PetscFunctionBegin;
  ierr = ISColoringGetIS(iscoloring,PETSC_USE_POINTER,&nn,&isis);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  PetscCheckFalse(size > 1,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only support sequential matrix");

  ierr = MatGetColumnIJ(A,0,PETSC_FALSE,PETSC_FALSE,&N,&cia,&cja,&done);CHKERRQ(ierr);
  PetscCheckFalse(!done,PETSC_COMM_SELF,PETSC_ERR_SUP,"Ordering requires IJ");

  ierr = MatGetSize(A,&M,NULL);CHKERRQ(ierr);
  ierr = PetscBTCreate(M,&table);CHKERRQ(ierr);
  for (c=0; c<nn; c++) { /* for each color */
    ierr = ISGetSize(isis[c],&nc);CHKERRQ(ierr);
    if (nc <= 1) continue;

    ierr = PetscBTMemzero(M,table);CHKERRQ(ierr);
    ierr = ISGetIndices(isis[c],&cols);CHKERRQ(ierr);
    for (j=0; j<nc; j++) { /* for each column */
      col = cols[j];
      nnz = cia[col+1] - cia[col];
      for (i=0; i<nnz; i++) {
        row = cja[cia[col]+i];
        if (PetscBTLookupSet(table,row)) {
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"color %" PetscInt_FMT ", col %" PetscInt_FMT ": row %" PetscInt_FMT " already in this color",c,col,row);
        }
      }
    }
    ierr = ISRestoreIndices(isis[c],&cols);CHKERRQ(ierr);
  }
  ierr = PetscBTDestroy(&table);CHKERRQ(ierr);

  ierr = MatRestoreColumnIJ(A,1,PETSC_FALSE,PETSC_TRUE,NULL,&cia,&cja,&done);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
