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
  PetscCall(MatColoringGetMaxColors(mc,&maxcolors));
  /* get the communication structures and the colors */
  PetscCall(MatColoringCreateBipartiteGraph(mc,&etoc,&etor));
  PetscCall(ISColoringGetIS(coloring,PETSC_USE_POINTER,&ncolors,&colors));
  PetscCall(PetscSFGetGraph(etor,&nrows,&nleafrows,NULL,NULL));
  PetscCall(PetscSFGetGraph(etoc,&ncols,&nleafcols,NULL,NULL));
  PetscCall(MatGetOwnershipRangeColumn(m,&s,&e));
  PetscCall(PetscMalloc1(ncols,&statecol));
  PetscCall(PetscMalloc1(nrows,&staterow));
  PetscCall(PetscMalloc1(nleafcols,&stateleafcol));
  PetscCall(PetscMalloc1(nleafrows,&stateleafrow));

  for (l=0;l<ncolors;l++) {
    if (l > maxcolors) break;
    for (k=0;k<ncols;k++) {
      statecol[k] = -1;
    }
    PetscCall(ISGetLocalSize(colors[l],&nindices));
    PetscCall(ISGetIndices(colors[l],&indices));
    for (k=0;k<nindices;k++) {
      statecol[indices[k]-s] = indices[k];
    }
    PetscCall(ISRestoreIndices(colors[l],&indices));
    statespread = statecol;
    for (k=0;k<dist;k++) {
      if (k%2 == 1) {
        PetscCall(PetscSFComputeDegreeBegin(etor,&degrees));
        PetscCall(PetscSFComputeDegreeEnd(etor,&degrees));
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
        PetscCall(PetscLogEventBegin(MATCOLORING_Comm,mc,0,0,0));
        PetscCall(PetscSFReduceBegin(etoc,itype,stateleafrow,statecol,MPI_MAX));
        PetscCall(PetscSFReduceEnd(etoc,itype,stateleafrow,statecol,MPI_MAX));
        PetscCall(PetscLogEventEnd(MATCOLORING_Comm,mc,0,0,0));
        statespread = statecol;
      } else {
        PetscCall(PetscSFComputeDegreeBegin(etoc,&degrees));
        PetscCall(PetscSFComputeDegreeEnd(etoc,&degrees));
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
        PetscCall(PetscLogEventBegin(MATCOLORING_Comm,mc,0,0,0));
        PetscCall(PetscSFReduceBegin(etor,itype,stateleafcol,staterow,MPI_MAX));
        PetscCall(PetscSFReduceEnd(etor,itype,stateleafcol,staterow,MPI_MAX));
        PetscCall(PetscLogEventEnd(MATCOLORING_Comm,mc,0,0,0));
        statespread = staterow;
      }
    }
    for (k=0;k<nindices;k++) {
      if (statespread[indices[k]-s] != indices[k]) {
        PetscCall(PetscPrintf(PetscObjectComm((PetscObject)mc),"%" PetscInt_FMT " of color %" PetscInt_FMT " conflicts with %" PetscInt_FMT "\n",indices[k],l,statespread[indices[k]-s]));
      }
    }
    PetscCall(ISRestoreIndices(colors[l],&indices));
  }
  PetscCall(PetscFree(statecol));
  PetscCall(PetscFree(staterow));
  PetscCall(PetscFree(stateleafcol));
  PetscCall(PetscFree(stateleafrow));
  PetscCall(PetscSFDestroy(&etor));
  PetscCall(PetscSFDestroy(&etoc));
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
  PetscCall(ISColoringGetIS(iscoloring,PETSC_USE_POINTER,&nn,&isis));

  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCheckFalse(size > 1,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only support sequential matrix");

  PetscCall(MatGetColumnIJ(A,0,PETSC_FALSE,PETSC_FALSE,&N,&cia,&cja,&done));
  PetscCheck(done,PETSC_COMM_SELF,PETSC_ERR_SUP,"Ordering requires IJ");

  PetscCall(MatGetSize(A,&M,NULL));
  PetscCall(PetscBTCreate(M,&table));
  for (c=0; c<nn; c++) { /* for each color */
    PetscCall(ISGetSize(isis[c],&nc));
    if (nc <= 1) continue;

    PetscCall(PetscBTMemzero(M,table));
    PetscCall(ISGetIndices(isis[c],&cols));
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
    PetscCall(ISRestoreIndices(isis[c],&cols));
  }
  PetscCall(PetscBTDestroy(&table));

  PetscCall(MatRestoreColumnIJ(A,1,PETSC_FALSE,PETSC_TRUE,NULL,&cia,&cja,&done));
  PetscFunctionReturn(0);
}
