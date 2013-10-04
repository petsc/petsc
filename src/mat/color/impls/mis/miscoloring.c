#include <petsc-private/matimpl.h>      /*I "petscmat.h"  I*/
#include <petscsf.h>


#undef __FUNCT__
#define __FUNCT__ "MISInitialize_Private"
PetscErrorCode MISInitialize_Private(MatColoring mc,PetscSF etoc,PetscReal *wts,ISColoringValue *color)
{
  PetscErrorCode ierr;
  PetscInt       i,ncols;
  PetscRandom    rand;
  PetscReal      r;
  const PetscInt *coldegrees;

  PetscFunctionBegin;
  /* each weight should be the degree plus a random perturbation */
  ierr = PetscSFGetGraph(etoc,&ncols,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeBegin(etoc,&coldegrees);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(etoc,&coldegrees);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)mc),&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  for (i=0;i<ncols;i++) {
    ierr = PetscRandomGetValueReal(rand,&r);CHKERRQ(ierr);
    wts[i] = coldegrees[i] + r;
    color[i] = IS_COLORING_MAX;
  }
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MISBipartiteSF_Private"
PetscErrorCode MISBipartiteSF_Private(Mat m,PetscSF *mattorow,PetscSF *mattocol)
{
  PetscErrorCode ierr;
  PetscInt       nentries,ncolentries,idx;
  PetscInt       i,j,rs,re,cs,ce,cn;
  PetscInt       *rowleaf,*colleaf,*rowdata;
  PetscInt       ncol;
  const PetscInt *icol;
  const PetscInt *coldegrees;

  PetscFunctionBegin;
  ierr = MatGetOwnershipRange(m,&rs,&re);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(m,&cs,&ce);CHKERRQ(ierr);
  cn = ce-cs;
  nentries=0;
  for (i=rs;i<re;i++) {
    ierr = MatGetRow(m,i,&ncol,NULL,NULL);CHKERRQ(ierr);
    nentries += ncol;
    ierr = MatRestoreRow(m,i,&ncol,NULL,NULL);CHKERRQ(ierr);
  }
  ierr = PetscMalloc(sizeof(PetscInt)*nentries,&rowleaf);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*nentries,&rowdata);CHKERRQ(ierr);
  idx=0;
  for (i=rs;i<re;i++) {
    ierr = MatGetRow(m,i,&ncol,&icol,NULL);CHKERRQ(ierr);
    for (j=0;j<ncol;j++) {
      rowleaf[idx] = icol[j];
      rowdata[idx] = i;
      idx++;
    }
    ierr = MatRestoreRow(m,i,&ncol,&icol,NULL);CHKERRQ(ierr);
  }
  if (idx != nentries) SETERRQ2(PetscObjectComm((PetscObject)m),PETSC_ERR_NOT_CONVERGED,"Bad number of entries %d vs %d",idx,nentries);
  ierr = PetscSFCreate(PetscObjectComm((PetscObject)m),mattorow);CHKERRQ(ierr);
  ierr = PetscSFCreate(PetscObjectComm((PetscObject)m),mattocol);CHKERRQ(ierr);
  ierr = PetscSFSetGraphLayout(*mattorow,m->cmap,nentries,NULL,PETSC_COPY_VALUES,rowleaf);CHKERRQ(ierr);

  /* determine the number of entries in the column matrix */
  ierr = PetscSFComputeDegreeBegin(*mattorow,&coldegrees);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(*mattorow,&coldegrees);CHKERRQ(ierr);
  ncolentries=0;
  for (i=0;i<cn;i++) {
    ncolentries += coldegrees[i];
  }
  ierr = PetscMalloc(sizeof(PetscInt)*ncolentries,&colleaf);CHKERRQ(ierr);

  /* create the one going the other way by building the leaf set */
  ierr = PetscSFGatherBegin(*mattorow,MPI_INT,rowdata,colleaf);CHKERRQ(ierr);
  ierr = PetscSFGatherEnd(*mattorow,MPI_INT,rowdata,colleaf);CHKERRQ(ierr);

  /* this one takes mat entries in *columns* to rows -- you never have to actually be able to order the leaf entries. */
  ierr = PetscSFSetGraphLayout(*mattocol,m->rmap,ncolentries,NULL,PETSC_COPY_VALUES,colleaf);CHKERRQ(ierr);
  ierr = PetscFree(rowdata);CHKERRQ(ierr);
  ierr = PetscFree(rowleaf);CHKERRQ(ierr);
  ierr = PetscFree(colleaf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MISGreatestWeight_Private"
PetscErrorCode MISGreatestWeight_Private(MatColoring mc,PetscInt dist,PetscSF etoc,PetscSF etor,PetscReal *wtsin,PetscReal *maxwts)
{
  PetscInt       nrows,ncols,nentries,idx;
  PetscInt       i,j,k;
  const PetscInt *degrees;
  PetscReal      *ewts,*wtsrow,*wtscol;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nentries=0;
  ierr = PetscSFGetGraph(etor,&nrows,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(etoc,&ncols,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*nrows,&wtsrow);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*ncols,&wtscol);CHKERRQ(ierr);
  for (i=0;i<ncols;i++) {
    wtscol[i] = wtsin[i];
  }
  for (k=0;k<dist;k++) {
    if (k%2 == 1) {
      /* second step takes the row weights to the column weights */
      ierr = PetscSFComputeDegreeBegin(etor,&degrees);CHKERRQ(ierr);
      ierr = PetscSFComputeDegreeEnd(etor,&degrees);CHKERRQ(ierr);
      nentries=0;
      for(i=0;i<nrows;i++) {
          nentries += degrees[i];
      }
      idx=0;
      ierr = PetscMalloc(sizeof(PetscReal)*nentries,&ewts);CHKERRQ(ierr);
      for(i=0;i<nrows;i++) {
        for (j=0;j<degrees[i];j++) {
          ewts[idx] = wtsrow[i];
          idx++;
        }
        wtscol[i]=0.;
      }
      if (idx != nentries) SETERRQ2(PetscObjectComm((PetscObject)mc),PETSC_ERR_NOT_CONVERGED,"Bad number of entries %d vs %d",idx,nentries);
      ierr = PetscSFReduceBegin(etoc,MPI_DOUBLE,ewts,wtscol,MPI_MAX);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(etoc,MPI_DOUBLE,ewts,wtscol,MPI_MAX);CHKERRQ(ierr);
      ierr = PetscFree(ewts);CHKERRQ(ierr);
    } else {
      /* first step takes the column weights to the row weights */
      ierr = PetscSFComputeDegreeBegin(etoc,&degrees);CHKERRQ(ierr);
      ierr = PetscSFComputeDegreeEnd(etoc,&degrees);CHKERRQ(ierr);
      nentries=0;
      for(i=0;i<ncols;i++) {
          nentries += degrees[i];
      }
      idx=0;
      ierr = PetscMalloc(sizeof(PetscReal)*nentries,&ewts);CHKERRQ(ierr);
      for(i=0;i<ncols;i++) {
        for (j=0;j<degrees[i];j++) {
          ewts[idx] = wtscol[i];
          idx++;
        }
        wtsrow[i]=0.;
      }
      if (idx != nentries) SETERRQ2(PetscObjectComm((PetscObject)mc),PETSC_ERR_NOT_CONVERGED,"Bad number of entries %d vs %d",idx,nentries);
      ierr = PetscSFReduceBegin(etor,MPI_DOUBLE,ewts,wtsrow,MPI_MAX);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(etor,MPI_DOUBLE,ewts,wtsrow,MPI_MAX);CHKERRQ(ierr);
      ierr = PetscFree(ewts);CHKERRQ(ierr);
    }
  }
  if (mc->dist % 2 == 1) {
    /* if it's an odd number of steps, copy out the square part */
    for (i=0;i<ncols;i++) {
      if (i < nrows) {
        maxwts[i] = wtsrow[i];
      } else {
        maxwts[i] = 0;
      }
    }
  } else {
    for (i=0;i<ncols;i++) {
      maxwts[i] = wtscol[i];
    }
  }
  ierr = PetscFree(wtsrow);CHKERRQ(ierr);
  ierr = PetscFree(wtscol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MISSpreadState_Private"
PetscErrorCode MISSpreadState_Private(MatColoring mc,PetscInt dist,PetscSF etoc,PetscSF etor,PetscInt *statein,PetscInt *stateout)
{
  PetscInt       nrows,ncols,nentries,idx;
  PetscInt       i,j,k;
  const PetscInt *degrees;
  PetscInt       *estate;
  PetscInt       *staterow,*statecol;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSFGetGraph(etor,&nrows,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(etoc,&ncols,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*nrows,&staterow);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*ncols,&statecol);CHKERRQ(ierr);
  for (i=0;i<ncols;i++) {
    statecol[i] = statein[i];
  }
  for (k=0;k<dist;k++) {
    if (k%2 == 1) {
      ierr = PetscSFComputeDegreeBegin(etor,&degrees);CHKERRQ(ierr);
      ierr = PetscSFComputeDegreeEnd(etor,&degrees);CHKERRQ(ierr);
      nentries=0;
      for(i=0;i<nrows;i++) {
        nentries += degrees[i];
      }
      idx=0;
      ierr = PetscMalloc(sizeof(PetscInt)*nentries,&estate);CHKERRQ(ierr);
      for(i=0;i<nrows;i++) {
        for (j=0;j<degrees[i];j++) {
          estate[idx] = staterow[i];
          idx++;
        }
        statecol[i]=0;
      }
      if (idx != nentries) SETERRQ2(PetscObjectComm((PetscObject)mc),PETSC_ERR_NOT_CONVERGED,"Bad number of entries %d vs %d",idx,nentries);
      ierr = PetscSFReduceBegin(etoc,MPIU_INT,estate,statecol,MPIU_MAX);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(etoc,MPIU_INT,estate,statecol,MPIU_MAX);CHKERRQ(ierr);
      ierr = PetscFree(estate);CHKERRQ(ierr);
    } else {
      ierr = PetscSFComputeDegreeBegin(etoc,&degrees);CHKERRQ(ierr);
      ierr = PetscSFComputeDegreeEnd(etoc,&degrees);CHKERRQ(ierr);
      nentries=0;
      for(i=0;i<ncols;i++) {
        nentries += degrees[i];
      }
      idx=0;
      ierr = PetscMalloc(sizeof(PetscInt)*nentries,&estate);CHKERRQ(ierr);
      for(i=0;i<ncols;i++) {
        for (j=0;j<degrees[i];j++) {
          estate[idx] = statecol[i];
          idx++;
        }
        staterow[i]=0;
      }
      if (idx != nentries) SETERRQ2(PetscObjectComm((PetscObject)mc),PETSC_ERR_NOT_CONVERGED,"Bad number of entries %d vs %d",idx,nentries);
      ierr = PetscSFReduceBegin(etor,MPIU_INT,estate,staterow,MPI_MAX);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(etor,MPIU_INT,estate,staterow,MPI_MAX);CHKERRQ(ierr);
      ierr = PetscFree(estate);CHKERRQ(ierr);
    }
  }
  if (mc->dist % 2 == 1) {
    /* if it's an odd number of steps, copy out the square part */
    for (i=0;i<ncols;i++) {
      if (i < nrows) {
        stateout[i] = staterow[i];
      } else {
        stateout[i] = 0;
      }
    }
  } else {
    for (i=0;i<ncols;i++) {
      stateout[i] = statecol[i];
    }
  }
  ierr = PetscFree(staterow);CHKERRQ(ierr);
  ierr = PetscFree(statecol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MISCompute_Private"
PetscErrorCode MISCompute_Private(MatColoring mc,PetscSF etor,PetscSF etoc,ISColoringValue curcolor,PetscReal *wts,ISColoringValue *colors,PetscInt *nadded_global)
{
  PetscInt        nr,nc,i;
  PetscInt        *state,*spreadstate;
  PetscErrorCode  ierr;
  PetscReal       *wtsinit,*wtsspread;
  PetscInt        nadded,misadded,misadded_global;

  PetscFunctionBegin;
  nadded=0;
  misadded_global=1;
  ierr = PetscSFGetGraph(etor,&nr,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(etoc,&nc,NULL,NULL,NULL);CHKERRQ(ierr);

  ierr = PetscMalloc(sizeof(PetscReal)*nc,&wtsinit);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*nc,&wtsspread);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*nc,&state);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*nc,&spreadstate);CHKERRQ(ierr);

  for (i=0;i<nc;i++) {
    if (colors[i] == IS_COLORING_MAX) {
      wtsinit[i] = wts[i];
    } else {
      /* already has a color and thus isn't in the graph any longer */
      wtsinit[i] = 0.;
    }
    state[i]=0;
    wtsspread[i]=0.;
    spreadstate[i]=0;
  }
  while (misadded_global > 0) {
    ierr = MISGreatestWeight_Private(mc,mc->dist,etoc,etor,wtsinit,wtsspread);CHKERRQ(ierr);
    misadded = 0;
    for (i=0;i<nc;i++) {
      if (wtsinit[i] >= wtsspread[i] && wtsinit[i] != 0.) {
        /* pick this one */
        colors[i] = curcolor;
        nadded++;
        state[i] = 1;
        wtsinit[i] = 0.;
        misadded++;
      } else {
        state[i] = 0;
      }
    }
    misadded_global = 0;
    ierr = MPI_Allreduce(&misadded,&misadded_global,1,MPI_INT,MPI_SUM,PetscObjectComm((PetscObject)mc));CHKERRQ(ierr);
    ierr = MISSpreadState_Private(mc,mc->dist,etoc,etor,state,spreadstate);CHKERRQ(ierr);
    for (i=0;i<nc;i++) {
      /* eliminated */
      if (colors[i] == IS_COLORING_MAX && state[i] == 0 && spreadstate[i] == 1) {
        wtsinit[i] = 0.;
      }
    }
  }
  *nadded_global=0;
  ierr = MPI_Allreduce(&nadded,nadded_global,1,MPI_INT,MPI_SUM,PetscObjectComm((PetscObject)mc));CHKERRQ(ierr);
  ierr = PetscFree(wtsinit);CHKERRQ(ierr);
  ierr = PetscFree(wtsspread);CHKERRQ(ierr);
  ierr = PetscFree(state);CHKERRQ(ierr);
  ierr = PetscFree(spreadstate);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringApply_MIS"
PETSC_EXTERN PetscErrorCode MatColoringApply_MIS(MatColoring mc,ISColoring *iscoloring)
{
  PetscErrorCode  ierr;
  PetscSF         etor,etoc;
  PetscReal       *wts;
  ISColoringValue curcolor;
  ISColoringValue *color;
  PetscInt        i,nadded,nadded_total,nrows,ncols,ncolstotal;

  PetscFunctionBegin;
  nadded=1;
  nadded_total=0;
  ierr = MatGetSize(mc->mat,NULL,&ncolstotal);CHKERRQ(ierr);
  ierr = MISBipartiteSF_Private(mc->mat,&etoc,&etor);
  ierr = PetscSFGetGraph(etor,&nrows,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(etoc,&ncols,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*ncols,&wts);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*ncols,&color);CHKERRQ(ierr);
  ierr = MISInitialize_Private(mc,etoc,wts,color);CHKERRQ(ierr);
  curcolor=0;
  for (i=0;(i<mc->maxcolors || mc->maxcolors == 0) && (nadded_total < ncolstotal);i++) {
    ierr = MISCompute_Private(mc,etor,etoc,curcolor,wts,color,&nadded);CHKERRQ(ierr);
    nadded_total += nadded;
    if (!nadded && nadded_total != ncolstotal) {SETERRQ(PetscObjectComm((PetscObject)mc),PETSC_ERR_NOT_CONVERGED,"MatColoringApply_MIS made no progress");}
    curcolor++;
  }
  for (i=0;i<ncols;i++) {
    /* set up a dummy color if the coloring has been truncated */
    if (color[i] == IS_COLORING_MAX) color[i] = curcolor;
  }
  ierr = PetscFree(wts);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&etor);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&etoc);CHKERRQ(ierr);
  ierr = ISColoringCreate(PetscObjectComm((PetscObject)mc),curcolor,ncols,color,iscoloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringCreate_MIS"
PETSC_EXTERN PetscErrorCode MatColoringCreate_MIS(MatColoring mc)
{
    PetscFunctionBegin;
    mc->data                = NULL;
    mc->ops->apply          = MatColoringApply_MIS;
    mc->ops->view           = NULL;
    mc->ops->destroy        = NULL;
    mc->ops->setfromoptions = NULL;
    PetscFunctionReturn(0);
}
