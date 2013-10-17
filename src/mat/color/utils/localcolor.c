#include <petsc-private/matimpl.h>      /*I "petscmat.h"  I*/
#include <petscsf.h>



#undef __FUNCT__
#define __FUNCT__ "MatColoringLocalColor"
PETSC_EXTERN PetscErrorCode MatColoringLocalColor(MatColoring mc,PetscSF etoc,PetscSF etor,PetscReal *wts,ISColoringValue *color, ISColoringValue *maxcolor)
{
  PetscInt       nrows,ncols,ncolentries,nrowentries,idx,neighoffset;
  PetscInt          i,j,k;
  PetscInt          dist = mc->dist;
  PetscInt          totalcolors;
  PetscBool         *colormask;
  PetscErrorCode    ierr;
  PetscBool         *rowseen,*colseen;
  const PetscInt    *rowdegrees;
  PetscInt          *rowoffsets;
  const PetscInt    *coldegrees;
  PetscInt          *coloffsets;
  PetscInt          offset;
  PetscInt          *ll_ptr;
  PetscInt          *ll_idx;
  PetscReal         *swts;
  PetscInt          *sidx;
  PetscInt          unused;
  PetscInt          rowlist,collist;
  PetscInt          swp;
  PetscMPIInt       rank;
  const PetscSFNode *colentries,*rowentries;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)mc),&rank);CHKERRQ(ierr);

  ierr = PetscSFGetGraph(etoc,&ncols,&ncolentries,NULL,&colentries);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(etor,&nrows,&nrowentries,NULL,&rowentries);CHKERRQ(ierr);

  ierr = PetscMalloc6(nrows,PetscBool,&rowseen,
                      ncols,PetscBool,&colseen,
                      ncols,PetscInt,&coloffsets,
                      nrows,PetscInt,&rowoffsets,
                      2*ncols,PetscInt,&ll_ptr,
                      2*ncols,PetscInt,&ll_idx);CHKERRQ(ierr);

  ierr = PetscMalloc2(ncols,PetscInt,&sidx,
                      ncols,PetscReal,&swts);CHKERRQ(ierr);

  ierr = PetscSFComputeDegreeBegin(etoc,&rowdegrees);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(etoc,&rowdegrees);CHKERRQ(ierr);

  ierr = PetscSFComputeDegreeBegin(etor,&coldegrees);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(etor,&coldegrees);CHKERRQ(ierr);

  /* sort by weight */
  for (i=0;i<ncols;i++) {
    sidx[i] = i;
    swts[i] = wts[i];
  }

  ierr = PetscSortRealWithPermutation(ncols,swts,sidx);CHKERRQ(ierr);
  for (i=0;i<ncols/2;i++) {
    swp = sidx[i];
    sidx[i] = sidx[ncols-1-i];
    sidx[ncols-1-i] = swp;
  }

  /* set up the "unused" linked list */
  unused = 0;
  ll_ptr[2*ncols-1] = -1;
  for (i=0;i<2*ncols-1;i++) {
    ll_ptr[i] = i+1;
  }

  /* initialize the offsets */
  offset=0;
  for (i=0;i<ncols;i++) {
    coloffsets[i] = offset;
    offset+=coldegrees[i];
    colseen[i] = PETSC_FALSE;
  }
  offset=0;
  for (i=0;i<nrows;i++) {
    rowoffsets[i] = offset;
    offset+=rowdegrees[i];
    rowseen[i] = PETSC_FALSE;
  }

  /* discover the maximum current color */
  totalcolors = 1;
  for (i=0;i<ncols;i++) {
    if (color[i] > totalcolors-1 && color[i] != IS_COLORING_MAX) totalcolors = color[i]+1;
  }
  if (totalcolors < 10) totalcolors=10;
  ierr = PetscMalloc(sizeof(PetscBool)*totalcolors,&colormask);CHKERRQ(ierr);

  /* alternate between rows and columns to get the distance k minimum coloring */
  for (i=0;i<ncols;i++) {
    collist = -1;
    rowlist = -1;
    if (color[sidx[i]] == IS_COLORING_MAX) {
      for (j=0;j<totalcolors;j++) colormask[j] = PETSC_FALSE;
      swp = unused;
      unused = ll_ptr[unused];
      ll_ptr[swp] = collist;
      ll_idx[swp] = sidx[i];
      collist = swp;
      colseen[sidx[i]] = PETSC_TRUE;
      for (k=0;k<=dist;k++) {
        if (k % 2 == 0) {
          while (collist >= 0) {
            if (k != dist) {
              for (j=0;j<coldegrees[ll_idx[collist]];j++) {
                neighoffset = coloffsets[ll_idx[collist]]+j;
                idx = colentries[neighoffset].index;
                if (colentries[neighoffset].rank == rank && !rowseen[idx]) {
                  swp = unused;
                  unused = ll_ptr[unused];
                  ll_ptr[swp] = rowlist;
                  ll_idx[swp] = idx;
                  rowlist = swp;
                  rowseen[idx] = PETSC_TRUE;
                }
              }
            }
            if (color[ll_idx[collist]] != IS_COLORING_MAX) colormask[color[ll_idx[collist]]] = PETSC_TRUE;
            colseen[ll_idx[collist]] = PETSC_FALSE;
            swp = collist;
            collist = ll_ptr[collist];
            ll_ptr[swp] = unused;
            unused = swp;
          }
        } else {
          while (rowlist >= 0) {
            if (k != dist) {
              for (j=0;j<rowdegrees[ll_idx[rowlist]];j++) {
                neighoffset = rowoffsets[ll_idx[rowlist]]+j;
                idx = rowentries[neighoffset].index;
                if (rowentries[neighoffset].rank == rank && !colseen[idx]) {
                  swp = unused;
                  unused = ll_ptr[unused];
                  ll_ptr[swp] = collist;
                  ll_idx[swp] = idx;
                  collist = swp;
                  colseen[idx] = PETSC_TRUE;
                }
              }
            }
            if (color[ll_idx[rowlist]] != IS_COLORING_MAX) colormask[color[ll_idx[rowlist]]] = PETSC_TRUE;
            rowseen[ll_idx[rowlist]] = PETSC_FALSE;
            swp = rowlist;
            rowlist = ll_ptr[rowlist];
            ll_ptr[swp] = unused;
            unused = swp;
          }
        }
      }
      color[sidx[i]] = totalcolors;
      for (k=0;k<totalcolors;k++) {
        if (!colormask[k]) {color[sidx[i]] = k; break;}
      }
      if (color[sidx[i]] >= mc->maxcolors && mc->maxcolors > 0) color[sidx[i]] = mc->maxcolors;
      if (color[sidx[i]] > *maxcolor) *maxcolor = color[sidx[i]];
      if (color[sidx[i]] > totalcolors-1) {
        totalcolors *= 2;
        ierr = PetscFree(colormask);CHKERRQ(ierr);
        ierr = PetscMalloc(sizeof(PetscBool)*totalcolors,&colormask);CHKERRQ(ierr);
      }
    }
    if (collist != -1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_NOT_CONVERGED,"Likely error in local coloring BFS -- column queue still has %d\n",collist);
    if (rowlist != -1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_NOT_CONVERGED,"Likely error in local coloring BFS -- row queue still has %d\n",rowlist);
  }
  for (i=0;i<ncols;i++) {
    if (colseen[i]) {SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_NOT_CONVERGED,"Likely error in local coloring BFS -- column %d still seen\n",i);}
  }
  for (i=0;i<nrows;i++) {
    if (rowseen[i]) {SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_NOT_CONVERGED,"Likely error in local coloring BFS -- row %d still seen\n",i);}
  }

  ierr = PetscFree6(rowseen,colseen,coloffsets,rowoffsets,ll_ptr,ll_idx);CHKERRQ(ierr);
  ierr = PetscFree2(sidx,swts);CHKERRQ(ierr);
  ierr = PetscFree(colormask);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringDiscoverBoundary"
PETSC_EXTERN PetscErrorCode MatColoringDiscoverBoundary(MatColoring mc,PetscSF etoc,PetscSF etor,PetscInt *nboundary,PetscInt **boundary)
{
  PetscInt       nrows,ncols,ncolentries,nrowentries,idx,bidx,neighoffset;
  PetscInt          i,j,k;
  PetscInt          dist = mc->dist;
  PetscBool         onBoundary;
  PetscErrorCode    ierr;
  PetscBool         *rowseen,*colseen;
  const PetscInt    *rowdegrees;
  PetscInt          *rowoffsets;
  const PetscInt    *coldegrees;
  PetscInt          *coloffsets;
  PetscInt          offset;
  PetscInt          *ll_ptr;
  PetscInt          *ll_idx;
  PetscInt          unused;
  PetscInt          rowlist,collist;
  PetscInt          swp;
  PetscMPIInt       rank;
  const PetscSFNode *colentries,*rowentries;

  PetscFunctionBegin;
  *nboundary = 0;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)mc),&rank);CHKERRQ(ierr);

  ierr = PetscSFGetGraph(etoc,&ncols,&ncolentries,NULL,&colentries);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(etor,&nrows,&nrowentries,NULL,&rowentries);CHKERRQ(ierr);

  ierr = PetscMalloc6(nrows,PetscBool,&rowseen,
                      ncols,PetscBool,&colseen,
                      ncols,PetscInt,&coloffsets,
                      nrows,PetscInt,&rowoffsets,
                      2*ncols,PetscInt,&ll_ptr,
                      2*ncols,PetscInt,&ll_idx);CHKERRQ(ierr);

  ierr = PetscSFComputeDegreeBegin(etoc,&rowdegrees);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(etoc,&rowdegrees);CHKERRQ(ierr);

  ierr = PetscSFComputeDegreeBegin(etor,&coldegrees);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(etor,&coldegrees);CHKERRQ(ierr);

  /* set up the "unused" linked list -- double the size of the number of items as in tiny or large distance cases we may have a clique */
  unused = 0;
  ll_ptr[2*ncols-1] = -1;
  for (i=0;i<2*ncols-1;i++) {
    ll_ptr[i] = i+1;
  }

  /* initialize the offsets */
  offset=0;
  for (i=0;i<ncols;i++) {
    coloffsets[i] = offset;
    offset+=coldegrees[i];
    colseen[i] = PETSC_FALSE;
  }
  offset=0;
  for (i=0;i<nrows;i++) {
    rowoffsets[i] = offset;
    offset+=rowdegrees[i];
    rowseen[i] = PETSC_FALSE;
  }

  /* count the number of boundary nodes */
  for (i=0;i<ncols;i++) {
    onBoundary = PETSC_FALSE;
    collist = -1;
    rowlist = -1;
    swp = unused;
    unused = ll_ptr[unused];
    ll_ptr[swp] = collist;
    ll_idx[swp] = i;
    collist = swp;
    colseen[i] = PETSC_TRUE;
    for (k=0;k<=dist;k++) {
      if (k % 2 == 0) {
        while (collist >= 0) {
          if (k != dist) {
            for (j=0;j<coldegrees[ll_idx[collist]];j++) {
              neighoffset = coloffsets[ll_idx[collist]]+j;
              idx = colentries[neighoffset].index;
              if (colentries[neighoffset].rank == rank) {
                if (!rowseen[idx] && !onBoundary) {
                  swp = unused;
                  unused = ll_ptr[unused];
                  ll_ptr[swp] = rowlist;
                  rowlist = swp;
                  ll_idx[swp] = idx;
                  rowseen[idx] = PETSC_TRUE;
                }
              } else {
                onBoundary = PETSC_TRUE;
              }
            }
          }
          colseen[ll_idx[collist]] = PETSC_FALSE;
          swp = collist;
          collist = ll_ptr[collist];
          ll_ptr[swp] = unused;
          unused = swp;
        }
      } else {
        while (rowlist >= 0) {
          if (k != dist) {
            for (j=0;j<rowdegrees[ll_idx[rowlist]];j++) {
              neighoffset = rowoffsets[ll_idx[rowlist]]+j;
              if (rowentries[neighoffset].rank == rank) {
                idx = rowentries[neighoffset].index;
                if (!colseen[idx] && !onBoundary) {
                  swp = unused;
                  unused = ll_ptr[unused];
                  ll_ptr[swp] = collist;
                  ll_idx[swp] = idx;
                  colseen[idx] = PETSC_TRUE;
                  collist = swp;
                }
              } else {
                onBoundary = PETSC_TRUE;
              }
            }
          }
          rowseen[ll_idx[rowlist]] = PETSC_FALSE;
          swp = rowlist;
          rowlist = ll_ptr[rowlist];
          ll_ptr[swp] = unused;
          unused = swp;
        }
      }
    }
    if (onBoundary) {(*nboundary)++;}
    if (collist != -1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_NOT_CONVERGED,"Likely error in boundary count BFS -- column queue still has %d\n",collist);
    if (rowlist != -1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_NOT_CONVERGED,"Likely error in boundary count BFS -- row queue still has %d\n",collist);
  }
  for (i=0;i<ncols;i++) {
    if (colseen[i]) {SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_NOT_CONVERGED,"Likely error in boundary count BFS -- column %d still seen\n",i);}
  }
  for (i=0;i<nrows;i++) {
    if (rowseen[i]) {SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_NOT_CONVERGED,"Likely error in boundary count BFS -- row %d still seen\n",i);}
  }

  ierr = PetscMalloc(sizeof(PetscInt)*(*nboundary),boundary);CHKERRQ(ierr);

  /* set the boundary nodes */
  bidx=0;
  for (i=0;i<ncols;i++) {
    onBoundary = PETSC_FALSE;
    collist = -1;
    rowlist = -1;
    swp = unused;
    unused = ll_ptr[unused];
    ll_ptr[swp] = collist;
    ll_idx[swp] = i;
    collist = swp;
    colseen[i] = PETSC_TRUE;
    for (k=0;k<=dist;k++) {
      if (k % 2 == 0) {
        while (collist >= 0) {
          if (k != dist) {
            for (j=0;j<coldegrees[ll_idx[collist]];j++) {
              neighoffset = coloffsets[ll_idx[collist]]+j;
              idx = colentries[neighoffset].index;
              if (colentries[neighoffset].rank == rank) {
                if (!rowseen[idx] && !onBoundary) {
                  swp = unused;
                  unused = ll_ptr[unused];
                  ll_ptr[swp] = rowlist;
                  rowlist = swp;
                  ll_idx[swp] = idx;
                  rowseen[idx] = PETSC_TRUE;
                }
              } else {
                onBoundary = PETSC_TRUE;
              }
            }
          }
          colseen[ll_idx[collist]] = PETSC_FALSE;
          swp = collist;
          collist = ll_ptr[collist];
          ll_ptr[swp] = unused;
          unused = swp;
        }
      } else {
        while (rowlist >= 0) {
          if (k != dist) {
            for (j=0;j<rowdegrees[ll_idx[rowlist]];j++) {
              neighoffset = rowoffsets[ll_idx[rowlist]]+j;
              if (rowentries[neighoffset].rank == rank) {
                idx = rowentries[neighoffset].index;
                if (!colseen[idx] && !onBoundary) {
                  swp = unused;
                  unused = ll_ptr[unused];
                  ll_ptr[swp] = collist;
                  ll_idx[swp] = idx;
                  colseen[idx] = PETSC_TRUE;
                  collist = swp;
                }
              } else {
                onBoundary = PETSC_TRUE;
              }
            }
          }
          rowseen[ll_idx[rowlist]] = PETSC_FALSE;
          swp = rowlist;
          rowlist = ll_ptr[rowlist];
          ll_ptr[swp] = unused;
          unused = swp;
        }
      }
    }
    if (onBoundary) {(*boundary)[bidx] = i; bidx++;}
  }
  for (i=0;i<ncols;i++) {
    if (colseen[i]) {SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_NOT_CONVERGED,"Likely error in boundary set BFS -- column %d still seen\n",i);}
  }
  for (i=0;i<nrows;i++) {
    if (rowseen[i]) {SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_NOT_CONVERGED,"Likely error in boundary set BFS -- row %d still seen\n",i);}
  }
  if (bidx != *nboundary) {SETERRQ(PetscObjectComm((PetscObject)mc),PETSC_ERR_NOT_CONVERGED,"Number of boundary nodes not matched");}
  ierr = PetscFree6(rowseen,colseen,coloffsets,rowoffsets,ll_ptr,ll_idx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
