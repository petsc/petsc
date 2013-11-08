#include <petsc-private/matimpl.h>      /*I "petscmat.h"  I*/
#include <petscsf.h>

PETSC_EXTERN PetscErrorCode MatColoringCreateBipartiteGraph(MatColoring,PetscSF *,PetscSF *);
PETSC_EXTERN PetscErrorCode MatColoringDiscoverBoundary(MatColoring,PetscSF,PetscSF,PetscInt *,PetscInt**);

typedef struct {
  PetscSF         etoc;
  PetscSF         etor;
  PetscReal       *wts;
  PetscReal       *wtsinit;
  PetscReal       *wtscol;
  PetscReal       *wtsrow;
  PetscReal       *wtsleafrow;
  PetscReal       *wtsleafcol;
  PetscReal       *wtsspread;
  PetscInt        *state;
  PetscInt        *statecol;
  PetscInt        *staterow;
  PetscInt        *stateleafcol;
  PetscInt        *stateleafrow;
  PetscInt        *statespread;
  ISColoringValue *color;
} MC_MIS;

#undef __FUNCT__
#define __FUNCT__ "MISCreateWeights_Private"
PetscErrorCode MISCreateWeights_Private(MatColoring mc)
{
  MC_MIS         *mis = (MC_MIS *)mc->data;
  PetscErrorCode ierr;
  PetscInt       i,ncols;
  PetscRandom    rand;
  PetscReal      *wts = mis->wts;
  PetscReal      r;
  const PetscInt *coldegrees;
  PetscSF        etoc=mis->etoc;

  PetscFunctionBegin;
  /* each weight should be the degree plus a random perturbation */
  ierr = PetscSFGetGraph(etoc,&ncols,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeBegin(etoc,&coldegrees);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(etoc,&coldegrees);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)mc),&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  for (i=0;i<ncols;i++) {
    ierr = PetscRandomGetValueReal(rand,&r);CHKERRQ(ierr);
    wts[i] = coldegrees[i] + PetscAbsReal(r);
  }
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MISInitialize_Private"
PetscErrorCode MISInitialize_Private(MatColoring mc)
{
  MC_MIS         *mis = (MC_MIS *)mc->data;
  PetscInt       i,croot,cleaf,rroot,rleaf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatColoringCreateBipartiteGraph(mc,&mis->etoc,&mis->etor);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(mis->etoc,&croot,&cleaf,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(mis->etor,&rroot,&rleaf,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc7(croot,PetscReal,&mis->wts,
                      croot,PetscReal,&mis->wtsinit,
                      croot,PetscReal,&mis->wtscol,
                      rroot,PetscReal,&mis->wtsrow,
                      croot,PetscReal,&mis->wtsspread,
                      cleaf,PetscReal,&mis->wtsleafcol,
                      rleaf,PetscReal,&mis->wtsleafrow);CHKERRQ(ierr);
  ierr = PetscMalloc6(croot,PetscInt,&mis->state,
                      croot,PetscInt,&mis->statecol,
                      rroot,PetscInt,&mis->staterow,
                      croot,PetscInt,&mis->statespread,
                      cleaf,PetscInt,&mis->stateleafcol,
                      rleaf,PetscInt,&mis->stateleafrow);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(ISColoringValue)*croot,&mis->color);CHKERRQ(ierr);
  for (i=0;i<croot;i++) {
    mis->color[i] = IS_COLORING_MAX;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringDestroy_MIS"
PetscErrorCode MatColoringDestroy_MIS(MatColoring mc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(mc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MISTearDown_Private"
PetscErrorCode MISTearDown_Private(MatColoring mc)
{
  MC_MIS         *mis = (MC_MIS *)mc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSFDestroy(&mis->etoc);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&mis->etor);CHKERRQ(ierr);
  ierr = PetscFree7(mis->wts,
                    mis->wtsinit,
                    mis->wtscol,
                    mis->wtsrow,
                    mis->wtsspread,
                    mis->wtsleafcol,
                    mis->wtsleafrow);CHKERRQ(ierr);
  ierr = PetscFree6(mis->state,
                    mis->statecol,
                    mis->staterow,
                    mis->statespread,
                    mis->stateleafcol,
                    mis->stateleafrow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MISGreatestWeight_Private"
PetscErrorCode MISGreatestWeight_Private(MatColoring mc,PetscReal *wtsin,PetscReal *maxwts)
{
  MC_MIS         *mis = (MC_MIS *)mc->data;
  PetscInt       nrows,ncols,nentries,idx,dist=mc->dist;
  PetscInt       i,j,k;
  const PetscInt *degrees;
  PetscReal      *ewts,*wtsrow=mis->wtsrow,*wtscol=mis->wtscol;
  PetscSF        etoc=mis->etoc,etor=mis->etor;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nentries=0;
  ierr = PetscSFGetGraph(etor,&nrows,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(etoc,&ncols,NULL,NULL,NULL);CHKERRQ(ierr);
  for (i=0;i<ncols;i++) {
    wtscol[i] = wtsin[i];
  }
  for (k=0;k<dist;k++) {
    if (k%2 == 1) {
      /* second step takes the row weights to the column weights */
      ierr = PetscLogEventBegin(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
      ierr = PetscSFComputeDegreeBegin(etor,&degrees);CHKERRQ(ierr);
      ierr = PetscSFComputeDegreeEnd(etor,&degrees);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
      nentries=0;
      for(i=0;i<nrows;i++) {
          nentries += degrees[i];
      }
      idx=0;
      ewts = mis->wtsleafrow;
      for(i=0;i<nrows;i++) {
        for (j=0;j<degrees[i];j++) {
          ewts[idx] = wtsrow[i];
          idx++;
        }
        wtscol[i]=0.;
      }
      if (idx != nentries) SETERRQ2(PetscObjectComm((PetscObject)mc),PETSC_ERR_NOT_CONVERGED,"Bad number of entries %d vs %d",idx,nentries);
      ierr = PetscLogEventBegin(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
      ierr = PetscSFReduceBegin(etoc,MPI_DOUBLE,ewts,wtscol,MPI_MAX);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(etoc,MPI_DOUBLE,ewts,wtscol,MPI_MAX);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
    } else {
      /* first step takes the column weights to the row weights */
      ierr = PetscSFComputeDegreeBegin(etoc,&degrees);CHKERRQ(ierr);
      ierr = PetscSFComputeDegreeEnd(etoc,&degrees);CHKERRQ(ierr);
      nentries=0;
      for(i=0;i<ncols;i++) {
          nentries += degrees[i];
      }
      ewts = mis->wtsleafcol;
      idx=0;
      for(i=0;i<ncols;i++) {
        for (j=0;j<degrees[i];j++) {
          ewts[idx] = wtscol[i];
          idx++;
        }
        wtsrow[i]=0.;
      }
      if (idx != nentries) SETERRQ2(PetscObjectComm((PetscObject)mc),PETSC_ERR_NOT_CONVERGED,"Bad number of entries %d vs %d",idx,nentries);
      ierr = PetscLogEventBegin(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
      ierr = PetscSFReduceBegin(etor,MPI_DOUBLE,ewts,wtsrow,MPI_MAX);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(etor,MPI_DOUBLE,ewts,wtsrow,MPI_MAX);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MISSpreadState_Private"
PetscErrorCode MISSpreadState_Private(MatColoring mc,PetscInt *statein,PetscInt *stateout)
{
  MC_MIS         *mis = (MC_MIS *)mc->data;
  PetscInt       nrows,ncols,nentries,idx,dist=mc->dist;
  PetscInt       i,j,k;
  const PetscInt *degrees;
  PetscInt       *estate;
  PetscInt       *staterow=mis->staterow,*statecol=mis->statecol;
  PetscSF        etoc=mis->etoc,etor=mis->etor;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSFGetGraph(etoc,&ncols,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(etor,&nrows,NULL,NULL,NULL);CHKERRQ(ierr);
  for (i=0;i<ncols;i++) {
    statecol[i] = statein[i];
  }
  for (k=0;k<dist;k++) {
    if (k%2 == 1) {
      ierr = PetscLogEventBegin(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
      ierr = PetscSFComputeDegreeBegin(etor,&degrees);CHKERRQ(ierr);
      ierr = PetscSFComputeDegreeEnd(etor,&degrees);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
      nentries=0;
      for(i=0;i<nrows;i++) {
        nentries += degrees[i];
      }
      estate = mis->stateleafrow;
      idx=0;
      for(i=0;i<nrows;i++) {
        for (j=0;j<degrees[i];j++) {
          estate[idx] = staterow[i];
          idx++;
        }
        statecol[i]=0;
      }
      if (idx != nentries) SETERRQ2(PetscObjectComm((PetscObject)mc),PETSC_ERR_NOT_CONVERGED,"Bad number of entries %d vs %d",idx,nentries);
      ierr = PetscLogEventBegin(Mat_Coloring_Comm,etoc,0,0,0);CHKERRQ(ierr);
      ierr = PetscSFReduceBegin(etoc,MPIU_INT,estate,statecol,MPIU_MAX);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(etoc,MPIU_INT,estate,statecol,MPIU_MAX);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(Mat_Coloring_Comm,etoc,0,0,0);CHKERRQ(ierr);
    } else {
      ierr = PetscLogEventBegin(Mat_Coloring_Comm,etoc,0,0,0);CHKERRQ(ierr);
      ierr = PetscSFComputeDegreeBegin(etoc,&degrees);CHKERRQ(ierr);
      ierr = PetscSFComputeDegreeEnd(etoc,&degrees);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(Mat_Coloring_Comm,etoc,0,0,0);CHKERRQ(ierr);
      nentries=0;
      for(i=0;i<ncols;i++) {
        nentries += degrees[i];
      }
      idx=0;
      estate = mis->stateleafcol;
      for(i=0;i<ncols;i++) {
        for (j=0;j<degrees[i];j++) {
          estate[idx] = statecol[i];
          idx++;
        }
        staterow[i]=0;
      }
      if (idx != nentries) SETERRQ2(PetscObjectComm((PetscObject)mc),PETSC_ERR_NOT_CONVERGED,"Bad number of entries %d vs %d",idx,nentries);
      ierr = PetscLogEventBegin(Mat_Coloring_Comm,etor,0,0,0);CHKERRQ(ierr);
      ierr = PetscSFReduceBegin(etor,MPIU_INT,estate,staterow,MPI_MAX);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(etor,MPIU_INT,estate,staterow,MPI_MAX);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(Mat_Coloring_Comm,etor,0,0,0);CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MISCompute_Private"
PetscErrorCode MISCompute_Private(MatColoring mc,ISColoringValue curcolor,PetscInt *nadded_global)
{
  MC_MIS          *mis = (MC_MIS *)mc->data;
  PetscInt        nr,nc,i;
  PetscInt        *state=mis->state,*spreadstate=mis->statespread;
  PetscErrorCode  ierr;
  PetscReal       *wts=mis->wts,*wtsinit=mis->wtsinit,*wtsspread=mis->wtsspread;
  PetscInt        nadded,misadded,misadded_global;
  PetscSF         etoc=mis->etoc,etor=mis->etor;
  ISColoringValue *colors = mis->color;

  PetscFunctionBegin;
  nadded=0;
  misadded_global=1;
  ierr = PetscSFGetGraph(etor,&nr,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(etoc,&nc,NULL,NULL,NULL);CHKERRQ(ierr);
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
    ierr = MISGreatestWeight_Private(mc,wtsinit,wtsspread);CHKERRQ(ierr);
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
    ierr = MPI_Allreduce(&misadded,&misadded_global,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)mc));CHKERRQ(ierr);
    ierr = MISSpreadState_Private(mc,state,spreadstate);CHKERRQ(ierr);
    for (i=0;i<nc;i++) {
      /* eliminated */
      if (colors[i] == IS_COLORING_MAX && state[i] == 0 && spreadstate[i] == 1) {
        wtsinit[i] = 0.;
      }
    }
  }
  *nadded_global=0;
  ierr = MPI_Allreduce(&nadded,nadded_global,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)mc));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringApply_MIS"
PETSC_EXTERN PetscErrorCode MatColoringApply_MIS(MatColoring mc,ISColoring *iscoloring)
{
  MC_MIS          *mis=(MC_MIS*)mc->data;
  PetscErrorCode  ierr;
  ISColoringValue curcolor,finalcolor;
  ISColoringValue *color;
  PetscInt        i,nadded,nadded_total,ncolstotal,ncols;

  PetscFunctionBegin;
  nadded=1;
  nadded_total=0;
  ierr = MatGetSize(mc->mat,NULL,&ncolstotal);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mc->mat,NULL,&ncols);CHKERRQ(ierr);
  ierr = MISInitialize_Private(mc);CHKERRQ(ierr);
  color = mis->color;
  ierr = MISCreateWeights_Private(mc);CHKERRQ(ierr);
  curcolor=0;
  for (i=0;(i<mc->maxcolors || mc->maxcolors == 0) && (nadded_total < ncolstotal);i++) {
    ierr = MISCompute_Private(mc,curcolor,&nadded);CHKERRQ(ierr);
    nadded_total += nadded;
    if (!nadded && nadded_total != ncolstotal) {SETERRQ(PetscObjectComm((PetscObject)mc),PETSC_ERR_NOT_CONVERGED,"MatColoringApply_MIS made no progress");}
    curcolor++;
  }
  finalcolor = curcolor;
  for (i=0;i<ncols;i++) {
    /* set up a dummy color if the coloring has been truncated */
    if (color[i] == IS_COLORING_MAX) {
      color[i] = curcolor;
      finalcolor = curcolor+1;
    }
  }
  ierr = ISColoringCreate(PetscObjectComm((PetscObject)mc),finalcolor,ncols,color,iscoloring);CHKERRQ(ierr);
  ierr = MISTearDown_Private(mc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringCreate_MIS"
/*MC
  MATCOLORINGMIS - Maximal Independent Set based Matrix Coloring

   Level: beginner

   Notes: This algorithm uses a Luby-type method to create a series of independent sets that may be combined into a
   maximal independent set.  This is repeated on the induced subgraph of uncolored vertices until every column of the
   matrix is assigned a color.  This algorithm supports arbitrary distance.  If the maximum number of colors is set to
   one, it will create a maximal independent set.

.seealso: MatColoringCreate(), MatColoring, MatColoringSetType()
M*/
PETSC_EXTERN PetscErrorCode MatColoringCreate_MIS(MatColoring mc)
{
  MC_MIS         *mis;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr                    = PetscNewLog(mc,MC_MIS,&mis);CHKERRQ(ierr);
  mc->data                = mis;
  mc->ops->apply          = MatColoringApply_MIS;
  mc->ops->view           = NULL;
  mc->ops->destroy        = MatColoringDestroy_MIS;
  mc->ops->setfromoptions = NULL;
  PetscFunctionReturn(0);
}
