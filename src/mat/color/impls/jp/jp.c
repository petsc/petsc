#include <petsc-private/matimpl.h>      /*I "petscmat.h"  I*/
#include <petscsf.h>



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
  ISColoringValue maxcolor;
  PetscInt        statesize;
  PetscInt        stateradix;
  PetscInt        *state;
  PetscInt        *statecol;
  PetscInt        *staterow;
  PetscInt        *stateleafcol;
  PetscInt        *stateleafrow;
  PetscInt        *statespread;

  ISColoringValue *color;
  ISColoringValue *mincolor;
} MC_JP;

#undef __FUNCT__
#define __FUNCT__ "JPCreateWeights_Private"
PetscErrorCode JPCreateWeights_Private(MatColoring mc)
{
  MC_JP          *jp = (MC_JP *)mc->data;
  PetscErrorCode ierr;
  PetscInt       i,ncols;
  PetscRandom    rand;
  PetscReal      *wts = jp->wts;
  PetscReal      r;
  const PetscInt *coldegrees;
  PetscSF        etoc=jp->etoc;

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
#define __FUNCT__ "JPBipartiteSF_Private"
PetscErrorCode JPBipartiteSF_Private(Mat m,PetscSF *etoc,PetscSF *etor)
{
  PetscErrorCode    ierr;
  PetscInt          nentries,ncolentries,idx;
  PetscInt          i,j,rs,re,cs,ce,cn;
  PetscInt          *rowleaf,*colleaf,*rowdata;
  PetscInt          ncol;
  const PetscScalar *vcol;
  const PetscInt    *icol;
  const PetscInt    *coldegrees;

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
  ierr = PetscMalloc(sizeof(PetscInt)*nentries,&rowleaf);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*nentries,&rowdata);CHKERRQ(ierr);
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
  if (idx != nentries) SETERRQ2(PetscObjectComm((PetscObject)m),PETSC_ERR_NOT_CONVERGED,"Bad number of entries %d vs %d",idx,nentries);
  ierr = PetscSFCreate(PetscObjectComm((PetscObject)m),etoc);CHKERRQ(ierr);
  ierr = PetscSFCreate(PetscObjectComm((PetscObject)m),etor);CHKERRQ(ierr);

  ierr = PetscSFSetGraphLayout(*etoc,m->cmap,nentries,NULL,PETSC_COPY_VALUES,rowleaf);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(*etoc);CHKERRQ(ierr);

  /* determine the number of entries in the column matrix */
  ierr = PetscLogEventBegin(Mat_Coloring_Comm,*etoc,0,0,0);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeBegin(*etoc,&coldegrees);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(*etoc,&coldegrees);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(Mat_Coloring_Comm,*etoc,0,0,0);CHKERRQ(ierr);
  ncolentries=0;
  for (i=0;i<cn;i++) {
    ncolentries += coldegrees[i];
  }
  ierr = PetscMalloc(sizeof(PetscInt)*ncolentries,&colleaf);CHKERRQ(ierr);

  /* create the one going the other way by building the leaf set */
  ierr = PetscLogEventBegin(Mat_Coloring_Comm,*etoc,0,0,0);CHKERRQ(ierr);
  ierr = PetscSFGatherBegin(*etoc,MPI_INT,rowdata,colleaf);CHKERRQ(ierr);
  ierr = PetscSFGatherEnd(*etoc,MPI_INT,rowdata,colleaf);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(Mat_Coloring_Comm,*etoc,0,0,0);CHKERRQ(ierr);

  /* this one takes mat entries in *columns* to rows -- you never have to actually be able to order the leaf entries. */
  ierr = PetscSFSetGraphLayout(*etor,m->rmap,ncolentries,NULL,PETSC_COPY_VALUES,colleaf);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(*etor);CHKERRQ(ierr);

  ierr = PetscFree(rowdata);CHKERRQ(ierr);
  ierr = PetscFree(rowleaf);CHKERRQ(ierr);
  ierr = PetscFree(colleaf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "JPInitialize_Private"
PetscErrorCode JPInitialize_Private(MatColoring mc)
{
  MC_JP          *jp = (MC_JP *)mc->data;
  PetscInt       i,croot,cleaf,rroot,rleaf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = JPBipartiteSF_Private(mc->mat,&jp->etoc,&jp->etor);CHKERRQ(ierr);
  jp->statesize = 1;
  jp->stateradix = (8*sizeof(PetscInt)-1);
  ierr = PetscSFGetGraph(jp->etoc,&croot,&cleaf,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(jp->etor,&rroot,&rleaf,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc7(croot,PetscReal,&jp->wts,
                      croot,PetscReal,&jp->wtsinit,
                      croot,PetscReal,&jp->wtscol,
                      rroot,PetscReal,&jp->wtsrow,
                      croot,PetscReal,&jp->wtsspread,
                      cleaf,PetscReal,&jp->wtsleafcol,
                      rleaf,PetscReal,&jp->wtsleafrow);CHKERRQ(ierr);
  ierr = PetscMalloc6(croot*jp->statesize,PetscInt,&jp->state,
                      croot*jp->statesize,PetscInt,&jp->statecol,
                      rroot*jp->statesize,PetscInt,&jp->staterow,
                      croot*jp->statesize,PetscInt,&jp->statespread,
                      cleaf*jp->statesize,PetscInt,&jp->stateleafcol,
                      rleaf*jp->statesize,PetscInt,&jp->stateleafrow);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(ISColoringValue)*croot,&jp->color);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(ISColoringValue)*croot,&jp->mincolor);CHKERRQ(ierr);
  for (i=0;i<croot;i++) {
    jp->color[i] = IS_COLORING_MAX;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringDestroy_JP"
PetscErrorCode MatColoringDestroy_JP(MatColoring mc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(mc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "JPTearDown_Private"
PetscErrorCode JPTearDown_Private(MatColoring mc)
{
  MC_JP          *jp = (MC_JP *)mc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSFDestroy(&jp->etoc);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&jp->etor);CHKERRQ(ierr);
  ierr = PetscFree7(jp->wts,
                    jp->wtsinit,
                    jp->wtscol,
                    jp->wtsrow,
                    jp->wtsspread,
                    jp->wtsleafcol,
                    jp->wtsleafrow);CHKERRQ(ierr);
  ierr = PetscFree6(jp->state,
                    jp->statecol,
                    jp->staterow,
                    jp->statespread,
                    jp->stateleafcol,
                    jp->stateleafrow);CHKERRQ(ierr);
  ierr = PetscFree(jp->mincolor);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "JPGreatestWeight_Private"
PetscErrorCode JPGreatestWeight_Private(MatColoring mc,PetscReal *wtsin,PetscReal *maxwts)
{
  MC_JP         *jp = (MC_JP *)mc->data;
  PetscInt       nrows,ncols,nentries,idx,dist=mc->dist;
  PetscInt       i,j,k;
  const PetscInt *degrees;
  PetscReal      *ewts,*wtsrow=jp->wtsrow,*wtscol=jp->wtscol;
  PetscSF        etoc=jp->etoc,etor=jp->etor;
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
      ewts = jp->wtsleafrow;
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
      ewts = jp->wtsleafcol;
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
#define __FUNCT__ "JPMinColor_Private"
PetscErrorCode JPMinColor_Private(MatColoring mc,ISColoringValue *colors,ISColoringValue *mincolor)
{
  MC_JP          *jp = (MC_JP *)mc->data;
  PetscInt       nrows,ncols,nleafcols,nleafrows,nentries,idx,dist=mc->dist;
  PetscInt       i,j,k,l,r;
  const PetscInt *degrees;
  PetscInt       *estate,*mask,mskvalue,*staterow,*statecol;
  PetscSF        etoc=jp->etoc,etor=jp->etor;
  ISColoringValue curmin;
  PetscErrorCode ierr;
  PetscBool      minfound;

  PetscFunctionBegin;
  ierr = PetscSFGetGraph(etoc,&ncols,&nleafcols,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(etor,&nrows,&nleafrows,NULL,NULL);CHKERRQ(ierr);
  /* reallocate so that we can create new size bitmasks */
  if (jp->statesize*jp->stateradix <= jp->maxcolor+1) {
    ierr = PetscFree6(jp->state,
                      jp->statecol,
                      jp->staterow,
                      jp->statespread,
                      jp->stateleafcol,
                      jp->stateleafrow);CHKERRQ(ierr);
    jp->statesize++;
    ierr = PetscMalloc6(ncols*jp->statesize,PetscInt,&jp->state,
                        ncols*jp->statesize,PetscInt,&jp->statecol,
                        nrows*jp->statesize,PetscInt,&jp->staterow,
                        ncols*jp->statesize,PetscInt,&jp->statespread,
                        nleafcols*jp->statesize,PetscInt,&jp->stateleafcol,
                        nleafrows*jp->statesize,PetscInt,&jp->stateleafrow);CHKERRQ(ierr);
  }
  statecol = jp->statecol;
  staterow = jp->staterow;

  /* set up the bitmask */
  for (i=0;i<ncols;i++) {
    if (colors[i] != IS_COLORING_MAX) {
      r = colors[i] / jp->stateradix;
      for (j=0;j<jp->statesize;j++) {
        if (j == r) {
          statecol[i+j*ncols] = 1;
          for (l=0;l < colors[i] % jp->stateradix;l++) {
            statecol[i+j*ncols] *= 2;
          }
        } else {
          statecol[i+j*ncols] = 0;
        }
      }
    } else {
      for (j=0;j<jp->statesize;j++) {
        statecol[i+j*ncols] = 0;
      }
    }
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
      estate = jp->stateleafrow;
      for (i=0;i<jp->statesize;i++) {
        idx=0;
        for(j=0;j<nrows;j++) {
          for (l=0;l<degrees[j];l++) {
            estate[idx] = staterow[j+i*nrows];
            idx++;
          }
        }
        for (j=0;j<ncols;j++) {
          statecol[j+i*ncols]=0;
        }
        if (idx != nentries) SETERRQ2(PetscObjectComm((PetscObject)mc),PETSC_ERR_NOT_CONVERGED,"Bad number of entries %d vs %d",idx,nentries);
        ierr = PetscLogEventBegin(Mat_Coloring_Comm,etoc,0,0,0);CHKERRQ(ierr);
        ierr = PetscSFReduceBegin(etoc,MPIU_INT,estate,&statecol[i*ncols],MPI_BOR);CHKERRQ(ierr);
        ierr = PetscSFReduceEnd(etoc,MPIU_INT,estate,&statecol[i*ncols],MPI_BOR);CHKERRQ(ierr);
        ierr = PetscLogEventEnd(Mat_Coloring_Comm,etoc,0,0,0);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscLogEventBegin(Mat_Coloring_Comm,etor,0,0,0);CHKERRQ(ierr);
      ierr = PetscSFComputeDegreeBegin(etoc,&degrees);CHKERRQ(ierr);
      ierr = PetscSFComputeDegreeEnd(etoc,&degrees);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(Mat_Coloring_Comm,etor,0,0,0);CHKERRQ(ierr);
      nentries=0;
      for(i=0;i<ncols;i++) {
        nentries += degrees[i];
      }
      estate = jp->stateleafcol;
      for (i=0;i<jp->statesize;i++) {
        idx=0;
        for(j=0;j<ncols;j++) {
          for (l=0;l<degrees[j];l++) {
            estate[idx] = statecol[j+i*ncols];
            idx++;
          }
        }
        for (j=0;j<nrows;j++) {
          staterow[j+i*nrows]=0;
        }
        if (idx != nentries) SETERRQ2(PetscObjectComm((PetscObject)mc),PETSC_ERR_NOT_CONVERGED,"Bad number of entries %d vs %d",idx,nentries);
        ierr = PetscLogEventBegin(Mat_Coloring_Comm,etoc,0,0,0);CHKERRQ(ierr);
        ierr = PetscSFReduceBegin(etor,MPIU_INT,estate,&staterow[i*ncols],MPI_BOR);CHKERRQ(ierr);
        ierr = PetscSFReduceEnd(etor,MPIU_INT,estate,&staterow[i*ncols],MPI_BOR);CHKERRQ(ierr);
        ierr = PetscLogEventEnd(Mat_Coloring_Comm,etoc,0,0,0);CHKERRQ(ierr);
      }
    }
  }
  if (mc->dist % 2 == 1) {
    mask = staterow;
  } else {
    mask = statecol;
  }
  /* reconstruct */
  for (i=0;i<ncols;i++) {
    curmin = 0;
    minfound=PETSC_FALSE;
    for (j=0;j<jp->statesize && !minfound;j++) {
      mskvalue = mask[i+j*ncols];
      for (k=0;k<jp->stateradix;k++) {
        if (mskvalue % 2 == 0) {
          mincolor[i] = curmin;
          minfound=PETSC_TRUE;
          break;
        }
        curmin++;
        mskvalue /= 2;
      }
    }
    if (!minfound) mincolor[i] = (ISColoringValue)jp->stateradix*jp->statesize;
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringApply_JP"
PETSC_EXTERN PetscErrorCode MatColoringApply_JP(MatColoring mc,ISColoring *iscoloring)
{
  MC_JP           *jp=(MC_JP*)mc->data;
  PetscErrorCode  ierr;
  PetscInt        i,nadded,nadded_total,ncolstotal,ncols;
  PetscInt        nr,nc;
  PetscInt        maxcolor_local,maxcolor_global;

  PetscFunctionBegin;
  ierr = JPInitialize_Private(mc);CHKERRQ(ierr);
  ierr = JPCreateWeights_Private(mc);CHKERRQ(ierr);
  ierr = MatGetSize(mc->mat,NULL,&ncolstotal);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mc->mat,NULL,&ncols);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(jp->etor,&nr,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(jp->etoc,&nc,NULL,NULL,NULL);CHKERRQ(ierr);
  for (i=0;i<nc;i++) {
    if (jp->color[i] == IS_COLORING_MAX) {
      jp->wtsinit[i] = jp->wts[i];
    } else {
      jp->wtsinit[i] = 0.;
    }
    jp->state[i]=0;
    jp->wtsspread[i]=0.;
    jp->statespread[i]=0;
  }
  nadded=0;
  nadded_total=0;
  while (nadded_total < ncolstotal) {
    ierr = JPGreatestWeight_Private(mc,jp->wtsinit,jp->wtsspread);CHKERRQ(ierr);
    ierr = JPMinColor_Private(mc,jp->color,jp->mincolor);CHKERRQ(ierr);
    for (i=0;i<nc;i++) {
      if (jp->wtsinit[i] >= jp->wtsspread[i] && jp->wtsinit[i] > 0.) {
        /* pick this one */
        if (mc->maxcolors > jp->mincolor[i] || mc->maxcolors==0) {
          jp->color[i] = jp->mincolor[i];
        } else {
          jp->color[i] = mc->maxcolors;
        }
        if (jp->color[i] > jp->maxcolor) jp->maxcolor = jp->color[i];
        jp->wtsinit[i] = 0.;
        nadded++;
      }
    }
    ierr = MPI_Allreduce(&nadded,&nadded_total,1,MPI_INT,MPI_SUM,PetscObjectComm((PetscObject)mc));CHKERRQ(ierr);
    maxcolor_local = (PetscInt)jp->maxcolor;
    maxcolor_global = 0;
    ierr = MPI_Allreduce(&maxcolor_local,&maxcolor_global,1,MPI_INT,MPI_MAX,PetscObjectComm((PetscObject)mc));CHKERRQ(ierr);
    jp->maxcolor = maxcolor_global;
  }
  ierr = ISColoringCreate(PetscObjectComm((PetscObject)mc),jp->maxcolor+1,ncols,jp->color,iscoloring);CHKERRQ(ierr);
  ierr = JPTearDown_Private(mc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringCreate_JP"
/*MC
  MATCOLORINGJP - Maximal Independent Set based Matrix Coloring

   Level: beginner

   Notes: This algorithm uses a Luby-type method to create a series of independent sets that may be combined into a
   maximal independent set.  This is repeated on the induced subgraph of uncolored vertices until every column of the
   matrix is assigned a color.  This algorithm supports arbitrary distance.  If the maximum number of colors is set to
   one, it will create a maximal independent set.

.seealso: MatColoringCreate(), MatColoring, MatColoringSetType()
M*/
PETSC_EXTERN PetscErrorCode MatColoringCreate_JP(MatColoring mc)
{
  MC_JP          *jp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr                    = PetscNewLog(mc,MC_JP,&jp);CHKERRQ(ierr);
  mc->data                = jp;
  mc->ops->apply          = MatColoringApply_JP;
  mc->ops->view           = NULL;
  mc->ops->destroy        = MatColoringDestroy_JP;
  mc->ops->setfromoptions = NULL;
  PetscFunctionReturn(0);
}
