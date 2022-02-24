#include <petsc/private/matimpl.h>      /*I "petscmat.h"  I*/
#include <petscsf.h>

PETSC_EXTERN PetscErrorCode MatColoringCreateBipartiteGraph(MatColoring,PetscSF *,PetscSF *);

PETSC_EXTERN PetscErrorCode MatColoringTest(MatColoring mc,ISColoring coloring)
{
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
  CHKERRQ(MatColoringGetMaxColors(mc,&maxcolors));
  /* get the communication structures and the colors */
  CHKERRQ(MatColoringCreateBipartiteGraph(mc,&etoc,&etor));
  CHKERRQ(ISColoringGetIS(coloring,PETSC_USE_POINTER,&ncolors,&colors));
  CHKERRQ(PetscSFGetGraph(etor,&nrows,&nleafrows,NULL,NULL));
  CHKERRQ(PetscSFGetGraph(etoc,&ncols,&nleafcols,NULL,NULL));
  CHKERRQ(MatGetOwnershipRangeColumn(m,&s,&e));
  CHKERRQ(PetscMalloc1(ncols,&statecol));
  CHKERRQ(PetscMalloc1(nrows,&staterow));
  CHKERRQ(PetscMalloc1(nleafcols,&stateleafcol));
  CHKERRQ(PetscMalloc1(nleafrows,&stateleafrow));

  for (l=0;l<ncolors;l++) {
    if (l > maxcolors) break;
    for (k=0;k<ncols;k++) {
      statecol[k] = -1;
    }
    CHKERRQ(ISGetLocalSize(colors[l],&nindices));
    CHKERRQ(ISGetIndices(colors[l],&indices));
    for (k=0;k<nindices;k++) {
      statecol[indices[k]-s] = indices[k];
    }
    CHKERRQ(ISRestoreIndices(colors[l],&indices));
    statespread = statecol;
    for (k=0;k<dist;k++) {
      if (k%2 == 1) {
        CHKERRQ(PetscSFComputeDegreeBegin(etor,&degrees));
        CHKERRQ(PetscSFComputeDegreeEnd(etor,&degrees));
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
        CHKERRQ(PetscLogEventBegin(MATCOLORING_Comm,mc,0,0,0));
        CHKERRQ(PetscSFReduceBegin(etoc,itype,stateleafrow,statecol,MPI_MAX));
        CHKERRQ(PetscSFReduceEnd(etoc,itype,stateleafrow,statecol,MPI_MAX));
        CHKERRQ(PetscLogEventEnd(MATCOLORING_Comm,mc,0,0,0));
        statespread = statecol;
      } else {
        CHKERRQ(PetscSFComputeDegreeBegin(etoc,&degrees));
        CHKERRQ(PetscSFComputeDegreeEnd(etoc,&degrees));
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
        CHKERRQ(PetscLogEventBegin(MATCOLORING_Comm,mc,0,0,0));
        CHKERRQ(PetscSFReduceBegin(etor,itype,stateleafcol,staterow,MPI_MAX));
        CHKERRQ(PetscSFReduceEnd(etor,itype,stateleafcol,staterow,MPI_MAX));
        CHKERRQ(PetscLogEventEnd(MATCOLORING_Comm,mc,0,0,0));
        statespread = staterow;
      }
    }
    for (k=0;k<nindices;k++) {
      if (statespread[indices[k]-s] != indices[k]) {
        CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)mc),"%" PetscInt_FMT " of color %" PetscInt_FMT " conflicts with %" PetscInt_FMT "\n",indices[k],l,statespread[indices[k]-s]));
      }
    }
    CHKERRQ(ISRestoreIndices(colors[l],&indices));
  }
  CHKERRQ(PetscFree(statecol));
  CHKERRQ(PetscFree(staterow));
  CHKERRQ(PetscFree(stateleafcol));
  CHKERRQ(PetscFree(stateleafrow));
  CHKERRQ(PetscSFDestroy(&etor));
  CHKERRQ(PetscSFDestroy(&etoc));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatISColoringTest(Mat A,ISColoring iscoloring)
{
  PetscInt       nn,c,i,j,M,N,nc,nnz,col,row;
  const PetscInt *cia,*cja,*cols;
  IS             *isis;
  MPI_Comm       comm;
  PetscMPIInt    size;
  PetscBool      done;
  PetscBT        table;

  PetscFunctionBegin;
  CHKERRQ(ISColoringGetIS(iscoloring,PETSC_USE_POINTER,&nn,&isis));

  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));
  CHKERRMPI(MPI_Comm_size(comm,&size));
  PetscCheckFalse(size > 1,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only support sequential matrix");

  CHKERRQ(MatGetColumnIJ(A,0,PETSC_FALSE,PETSC_FALSE,&N,&cia,&cja,&done));
  PetscCheckFalse(!done,PETSC_COMM_SELF,PETSC_ERR_SUP,"Ordering requires IJ");

  CHKERRQ(MatGetSize(A,&M,NULL));
  CHKERRQ(PetscBTCreate(M,&table));
  for (c=0; c<nn; c++) { /* for each color */
    CHKERRQ(ISGetSize(isis[c],&nc));
    if (nc <= 1) continue;

    CHKERRQ(PetscBTMemzero(M,table));
    CHKERRQ(ISGetIndices(isis[c],&cols));
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
    CHKERRQ(ISRestoreIndices(isis[c],&cols));
  }
  CHKERRQ(PetscBTDestroy(&table));

  CHKERRQ(MatRestoreColumnIJ(A,1,PETSC_FALSE,PETSC_TRUE,NULL,&cia,&cja,&done));
  PetscFunctionReturn(0);
}
