
#include <../src/mat/impls/aij/mpi/mpiaij.h>     /*I "petscmat.h"  I*/
#include <petscsf.h>

typedef struct {
  PetscSF    sf;
  PetscReal *dwts,*owts;
  PetscInt  *dmask,*omask,*cmask;
  PetscBool local;
} MC_JP;

static PetscErrorCode MatColoringDestroy_JP(MatColoring mc)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(mc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatColoringSetFromOptions_JP(PetscOptionItems *PetscOptionsObject,MatColoring mc)
{
  MC_JP          *jp = (MC_JP*)mc->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"JP options");
  PetscCall(PetscOptionsBool("-mat_coloring_jp_local","Do an initial coloring of local columns","",jp->local,&jp->local,NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode MCJPGreatestWeight_Private(MatColoring mc,const PetscReal *weights,PetscReal *maxweights)
{
  MC_JP          *jp = (MC_JP*)mc->data;
  Mat            G=mc->mat,dG,oG;
  PetscBool      isSeq,isMPI;
  Mat_MPIAIJ     *aij;
  Mat_SeqAIJ     *daij,*oaij;
  PetscInt       *di,*oi,*dj,*oj;
  PetscSF        sf=jp->sf;
  PetscLayout    layout;
  PetscInt       dn,on;
  PetscInt       i,j,l;
  PetscReal      *dwts=jp->dwts,*owts=jp->owts;
  PetscInt       ncols;
  const PetscInt *cols;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)G,MATSEQAIJ,&isSeq));
  PetscCall(PetscObjectTypeCompare((PetscObject)G,MATMPIAIJ,&isMPI));
  PetscCheck(isSeq || isMPI,PetscObjectComm((PetscObject)G),PETSC_ERR_ARG_WRONGSTATE,"MatColoringDegrees requires an MPI/SEQAIJ Matrix");

  /* get the inner matrix structure */
  oG = NULL;
  oi = NULL;
  oj = NULL;
  if (isMPI) {
    aij = (Mat_MPIAIJ*)G->data;
    dG = aij->A;
    oG = aij->B;
    daij = (Mat_SeqAIJ*)dG->data;
    oaij = (Mat_SeqAIJ*)oG->data;
    di = daij->i;
    dj = daij->j;
    oi = oaij->i;
    oj = oaij->j;
    PetscCall(MatGetSize(oG,&dn,&on));
    if (!sf) {
      PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)mc),&sf));
      PetscCall(MatGetLayouts(G,&layout,NULL));
      PetscCall(PetscSFSetGraphLayout(sf,layout,on,NULL,PETSC_COPY_VALUES,aij->garray));
      jp->sf = sf;
    }
  } else {
    dG = G;
    PetscCall(MatGetSize(dG,NULL,&dn));
    daij = (Mat_SeqAIJ*)dG->data;
    di = daij->i;
    dj = daij->j;
  }
  /* set up the distance-zero weights */
  if (!dwts) {
    PetscCall(PetscMalloc1(dn,&dwts));
    jp->dwts = dwts;
    if (oG) {
      PetscCall(PetscMalloc1(on,&owts));
      jp->owts = owts;
    }
  }
  for (i=0;i<dn;i++) {
    maxweights[i] = weights[i];
    dwts[i] = maxweights[i];
  }
  /* get the off-diagonal weights */
  if (oG) {
    PetscCall(PetscLogEventBegin(MATCOLORING_Comm,mc,0,0,0));
    PetscCall(PetscSFBcastBegin(sf,MPIU_REAL,dwts,owts,MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf,MPIU_REAL,dwts,owts,MPI_REPLACE));
    PetscCall(PetscLogEventEnd(MATCOLORING_Comm,mc,0,0,0));
  }
  /* check for the maximum out to the distance of the coloring */
  for (l=0;l<mc->dist;l++) {
    /* check for on-diagonal greater weights */
    for (i=0;i<dn;i++) {
      ncols = di[i+1]-di[i];
      cols = &(dj[di[i]]);
      for (j=0;j<ncols;j++) {
        if (dwts[cols[j]] > maxweights[i]) maxweights[i] = dwts[cols[j]];
      }
      /* check for off-diagonal greater weights */
      if (oG) {
        ncols = oi[i+1]-oi[i];
        cols = &(oj[oi[i]]);
        for (j=0;j<ncols;j++) {
          if (owts[cols[j]] > maxweights[i]) maxweights[i] = owts[cols[j]];
        }
      }
    }
    if (l < mc->dist-1) {
      for (i=0;i<dn;i++) {
        dwts[i] = maxweights[i];
      }
      if (oG) {
        PetscCall(PetscLogEventBegin(MATCOLORING_Comm,mc,0,0,0));
        PetscCall(PetscSFBcastBegin(sf,MPIU_REAL,dwts,owts,MPI_REPLACE));
        PetscCall(PetscSFBcastEnd(sf,MPIU_REAL,dwts,owts,MPI_REPLACE));
        PetscCall(PetscLogEventEnd(MATCOLORING_Comm,mc,0,0,0));
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MCJPInitialLocalColor_Private(MatColoring mc,PetscInt *lperm,ISColoringValue *colors)
{
  PetscInt       j,i,s,e,n,bidx,cidx,idx,dist,distance=mc->dist;
  Mat            G=mc->mat,dG,oG;
  PetscInt       *seen;
  PetscInt       *idxbuf;
  PetscBool      *boundary;
  PetscInt       *distbuf;
  PetscInt      *colormask;
  PetscInt       ncols;
  const PetscInt *cols;
  PetscBool      isSeq,isMPI;
  Mat_MPIAIJ     *aij;
  Mat_SeqAIJ     *daij,*oaij;
  PetscInt       *di,*dj,dn;
  PetscInt       *oi;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(MATCOLORING_Local,mc,0,0,0));
  PetscCall(MatGetOwnershipRange(G,&s,&e));
  n=e-s;
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)G,MATSEQAIJ,&isSeq));
  PetscCall(PetscObjectTypeCompare((PetscObject)G,MATMPIAIJ,&isMPI));
  PetscCheck(isSeq || isMPI,PetscObjectComm((PetscObject)G),PETSC_ERR_ARG_WRONGSTATE,"MatColoringDegrees requires an MPI/SEQAIJ Matrix");

  /* get the inner matrix structure */
  oG = NULL;
  oi = NULL;
  if (isMPI) {
    aij = (Mat_MPIAIJ*)G->data;
    dG = aij->A;
    oG = aij->B;
    daij = (Mat_SeqAIJ*)dG->data;
    oaij = (Mat_SeqAIJ*)oG->data;
    di = daij->i;
    dj = daij->j;
    oi = oaij->i;
    PetscCall(MatGetSize(oG,&dn,NULL));
  } else {
    dG = G;
    PetscCall(MatGetSize(dG,NULL,&dn));
    daij = (Mat_SeqAIJ*)dG->data;
    di = daij->i;
    dj = daij->j;
  }
  PetscCall(PetscMalloc5(n,&colormask,n,&seen,n,&idxbuf,n,&distbuf,n,&boundary));
  for (i=0;i<dn;i++) {
    seen[i]=-1;
    colormask[i] = -1;
    boundary[i] = PETSC_FALSE;
  }
  /* pass one -- figure out which ones are off-boundary in the distance-n sense */
  if (oG) {
    for (i=0;i<dn;i++) {
      bidx=-1;
      /* nonempty off-diagonal, so this one is on the boundary */
      if (oi[i]!=oi[i+1]) {
        boundary[i] = PETSC_TRUE;
        continue;
      }
      ncols = di[i+1]-di[i];
      cols = &(dj[di[i]]);
      for (j=0;j<ncols;j++) {
        bidx++;
        seen[cols[j]] = i;
        distbuf[bidx] = 1;
        idxbuf[bidx] = cols[j];
      }
      while (bidx >= 0) {
        idx = idxbuf[bidx];
        dist = distbuf[bidx];
        bidx--;
        if (dist < distance) {
          if (oi[idx+1]!=oi[idx]) {
            boundary[i] = PETSC_TRUE;
            break;
          }
          ncols = di[idx+1]-di[idx];
          cols = &(dj[di[idx]]);
          for (j=0;j<ncols;j++) {
            if (seen[cols[j]] != i) {
              bidx++;
              seen[cols[j]] = i;
              idxbuf[bidx] = cols[j];
              distbuf[bidx] = dist+1;
            }
          }
        }
      }
    }
    for (i=0;i<dn;i++) {
      seen[i]=-1;
    }
  }
  /* pass two -- color it by looking at nearby vertices and building a mask */
  for (i=0;i<dn;i++) {
    cidx = lperm[i];
    if (!boundary[cidx]) {
      bidx=-1;
      ncols = di[cidx+1]-di[cidx];
      cols = &(dj[di[cidx]]);
      for (j=0;j<ncols;j++) {
        bidx++;
        seen[cols[j]] = cidx;
        distbuf[bidx] = 1;
        idxbuf[bidx] = cols[j];
      }
      while (bidx >= 0) {
        idx = idxbuf[bidx];
        dist = distbuf[bidx];
        bidx--;
        /* mask this color */
        if (colors[idx] < IS_COLORING_MAX) {
          colormask[colors[idx]] = cidx;
        }
        if (dist < distance) {
          ncols = di[idx+1]-di[idx];
          cols = &(dj[di[idx]]);
          for (j=0;j<ncols;j++) {
            if (seen[cols[j]] != cidx) {
              bidx++;
              seen[cols[j]] = cidx;
              idxbuf[bidx] = cols[j];
              distbuf[bidx] = dist+1;
            }
          }
        }
      }
      /* find the lowest untaken color */
      for (j=0;j<n;j++) {
        if (colormask[j] != cidx || j >= mc->maxcolors) {
          colors[cidx] = j;
          break;
        }
      }
    }
  }
  PetscCall(PetscFree5(colormask,seen,idxbuf,distbuf,boundary));
  PetscCall(PetscLogEventEnd(MATCOLORING_Local,mc,0,0,0));
  PetscFunctionReturn(0);
}

static PetscErrorCode MCJPMinColor_Private(MatColoring mc,ISColoringValue maxcolor,const ISColoringValue *colors,ISColoringValue *mincolors)
{
  MC_JP          *jp = (MC_JP*)mc->data;
  Mat            G=mc->mat,dG,oG;
  PetscBool      isSeq,isMPI;
  Mat_MPIAIJ     *aij;
  Mat_SeqAIJ     *daij,*oaij;
  PetscInt       *di,*oi,*dj,*oj;
  PetscSF        sf=jp->sf;
  PetscLayout    layout;
  PetscInt       maskrounds,maskbase,maskradix;
  PetscInt       dn,on;
  PetscInt       i,j,l,k;
  PetscInt       *dmask=jp->dmask,*omask=jp->omask,*cmask=jp->cmask,curmask;
  PetscInt       ncols;
  const PetscInt *cols;

  PetscFunctionBegin;
  maskradix = sizeof(PetscInt)*8;
  maskrounds = 1 + maxcolor / (maskradix);
  maskbase = 0;
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)G,MATSEQAIJ,&isSeq));
  PetscCall(PetscObjectTypeCompare((PetscObject)G,MATMPIAIJ,&isMPI));
  PetscCheck(isSeq || isMPI,PetscObjectComm((PetscObject)G),PETSC_ERR_ARG_WRONGSTATE,"MatColoringDegrees requires an MPI/SEQAIJ Matrix");

  /* get the inner matrix structure */
  oG = NULL;
  oi = NULL;
  oj = NULL;
  if (isMPI) {
    aij = (Mat_MPIAIJ*)G->data;
    dG = aij->A;
    oG = aij->B;
    daij = (Mat_SeqAIJ*)dG->data;
    oaij = (Mat_SeqAIJ*)oG->data;
    di = daij->i;
    dj = daij->j;
    oi = oaij->i;
    oj = oaij->j;
    PetscCall(MatGetSize(oG,&dn,&on));
    if (!sf) {
      PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)mc),&sf));
      PetscCall(MatGetLayouts(G,&layout,NULL));
      PetscCall(PetscSFSetGraphLayout(sf,layout,on,NULL,PETSC_COPY_VALUES,aij->garray));
      jp->sf = sf;
    }
  } else {
    dG = G;
    PetscCall(MatGetSize(dG,NULL,&dn));
    daij = (Mat_SeqAIJ*)dG->data;
    di = daij->i;
    dj = daij->j;
  }
  for (i=0;i<dn;i++) {
    mincolors[i] = IS_COLORING_MAX;
  }
  /* set up the distance-zero mask */
  if (!dmask) {
    PetscCall(PetscMalloc1(dn,&dmask));
    PetscCall(PetscMalloc1(dn,&cmask));
    jp->cmask = cmask;
    jp->dmask = dmask;
    if (oG) {
      PetscCall(PetscMalloc1(on,&omask));
      jp->omask = omask;
    }
  }
  /* the number of colors may be more than the number of bits in a PetscInt; take multiple rounds */
  for (k=0;k<maskrounds;k++) {
    for (i=0;i<dn;i++) {
      cmask[i] = 0;
      if (colors[i] < maskbase+maskradix && colors[i] >= maskbase)
        cmask[i] = 1 << (colors[i]-maskbase);
      dmask[i] = cmask[i];
    }
    if (oG) {
      PetscCall(PetscLogEventBegin(MATCOLORING_Comm,mc,0,0,0));
      PetscCall(PetscSFBcastBegin(sf,MPIU_INT,dmask,omask,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(sf,MPIU_INT,dmask,omask,MPI_REPLACE));
      PetscCall(PetscLogEventEnd(MATCOLORING_Comm,mc,0,0,0));
    }
    /* fill in the mask out to the distance of the coloring */
    for (l=0;l<mc->dist;l++) {
      /* fill in the on-and-off diagonal mask */
      for (i=0;i<dn;i++) {
        ncols = di[i+1]-di[i];
        cols = &(dj[di[i]]);
        for (j=0;j<ncols;j++) {
          cmask[i] = cmask[i] | dmask[cols[j]];
        }
        if (oG) {
          ncols = oi[i+1]-oi[i];
          cols = &(oj[oi[i]]);
          for (j=0;j<ncols;j++) {
            cmask[i] = cmask[i] | omask[cols[j]];
          }
        }
      }
      for (i=0;i<dn;i++) {
        dmask[i]=cmask[i];
      }
      if (l < mc->dist-1) {
        if (oG) {
          PetscCall(PetscLogEventBegin(MATCOLORING_Comm,mc,0,0,0));
          PetscCall(PetscSFBcastBegin(sf,MPIU_INT,dmask,omask,MPI_REPLACE));
          PetscCall(PetscSFBcastEnd(sf,MPIU_INT,dmask,omask,MPI_REPLACE));
          PetscCall(PetscLogEventEnd(MATCOLORING_Comm,mc,0,0,0));
        }
      }
    }
    /* read through the mask to see if we've discovered an acceptable color for any vertices in this round */
    for (i=0;i<dn;i++) {
      if (mincolors[i] == IS_COLORING_MAX) {
        curmask = dmask[i];
        for (j=0;j<maskradix;j++) {
          if (curmask % 2 == 0) {
            mincolors[i] = j+maskbase;
            break;
          }
          curmask = curmask >> 1;
        }
      }
    }
    /* do the next maskradix colors */
    maskbase += maskradix;
  }
  for (i=0;i<dn;i++) {
    if (mincolors[i] == IS_COLORING_MAX) {
      mincolors[i] = maxcolor+1;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatColoringApply_JP(MatColoring mc,ISColoring *iscoloring)
{
  MC_JP          *jp = (MC_JP*)mc->data;
  PetscInt        i,nadded,nadded_total,nadded_total_old,ntotal,n,round;
  PetscInt        maxcolor_local=0,maxcolor_global = 0,*lperm;
  PetscMPIInt     rank;
  PetscReal       *weights,*maxweights;
  ISColoringValue  *color,*mincolor;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)mc),&rank));
  PetscCall(PetscLogEventBegin(MATCOLORING_Weights,mc,0,0,0));
  PetscCall(MatColoringCreateWeights(mc,&weights,&lperm));
  PetscCall(PetscLogEventEnd(MATCOLORING_Weights,mc,0,0,0));
  PetscCall(MatGetSize(mc->mat,NULL,&ntotal));
  PetscCall(MatGetLocalSize(mc->mat,NULL,&n));
  PetscCall(PetscMalloc1(n,&maxweights));
  PetscCall(PetscMalloc1(n,&color));
  PetscCall(PetscMalloc1(n,&mincolor));
  for (i=0;i<n;i++) {
    color[i] = IS_COLORING_MAX;
    mincolor[i] = 0;
  }
  nadded=0;
  nadded_total=0;
  nadded_total_old=0;
  /* compute purely local vertices */
  if (jp->local) {
    PetscCall(MCJPInitialLocalColor_Private(mc,lperm,color));
    for (i=0;i<n;i++) {
      if (color[i] < IS_COLORING_MAX) {
        nadded++;
        weights[i] = -1;
        if (color[i] > maxcolor_local) maxcolor_local = color[i];
      }
    }
    PetscCall(MPIU_Allreduce(&nadded,&nadded_total,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)mc)));
    PetscCall(MPIU_Allreduce(&maxcolor_local,&maxcolor_global,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)mc)));
  }
  round = 0;
  while (nadded_total < ntotal) {
    PetscCall(MCJPMinColor_Private(mc,(ISColoringValue)maxcolor_global,color,mincolor));
    PetscCall(MCJPGreatestWeight_Private(mc,weights,maxweights));
    for (i=0;i<n;i++) {
      /* choose locally maximal vertices; weights less than zero are omitted from the graph */
      if (weights[i] >= maxweights[i] && weights[i] >= 0.) {
        /* assign the minimum possible color */
        if (mc->maxcolors > mincolor[i]) {
          color[i] = mincolor[i];
        } else {
          color[i] = mc->maxcolors;
        }
        if (color[i] > maxcolor_local) maxcolor_local = color[i];
        weights[i] = -1.;
        nadded++;
      }
    }
    PetscCall(MPIU_Allreduce(&maxcolor_local,&maxcolor_global,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)mc)));
    PetscCall(MPIU_Allreduce(&nadded,&nadded_total,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)mc)));
    PetscCheck(nadded_total != nadded_total_old,PetscObjectComm((PetscObject)mc),PETSC_ERR_NOT_CONVERGED,"JP didn't make progress");
    nadded_total_old = nadded_total;
    round++;
  }
  PetscCall(PetscLogEventBegin(MATCOLORING_ISCreate,mc,0,0,0));
  PetscCall(ISColoringCreate(PetscObjectComm((PetscObject)mc),maxcolor_global+1,n,color,PETSC_OWN_POINTER,iscoloring));
  PetscCall(PetscLogEventEnd(MATCOLORING_ISCreate,mc,0,0,0));
  PetscCall(PetscFree(jp->dwts));
  PetscCall(PetscFree(jp->dmask));
  PetscCall(PetscFree(jp->cmask));
  PetscCall(PetscFree(jp->owts));
  PetscCall(PetscFree(jp->omask));
  PetscCall(PetscFree(weights));
  PetscCall(PetscFree(lperm));
  PetscCall(PetscFree(maxweights));
  PetscCall(PetscFree(mincolor));
  PetscCall(PetscSFDestroy(&jp->sf));
  PetscFunctionReturn(0);
}

/*MC
  MATCOLORINGJP - Parallel Jones-Plassmann Coloring

   Level: beginner

   Notes:
    This method uses a parallel Luby-style coloring with weights to choose an independent set of processor
   boundary vertices at each stage that may be assigned colors independently.

   Supports both distance one and distance two colorings.

   References:
.  * - M. Jones and P. Plassmann, "A parallel graph coloring heuristic," SIAM Journal on Scientific Computing, vol. 14, no. 3,
   pp. 654-669, 1993.

.seealso: `MatColoringCreate()`, `MatColoring`, `MatColoringSetType()`
M*/
PETSC_EXTERN PetscErrorCode MatColoringCreate_JP(MatColoring mc)
{
  MC_JP          *jp;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(mc,&jp));
  jp->sf                  = NULL;
  jp->dmask               = NULL;
  jp->omask               = NULL;
  jp->cmask               = NULL;
  jp->dwts                = NULL;
  jp->owts                = NULL;
  jp->local               = PETSC_TRUE;
  mc->data                = jp;
  mc->ops->apply          = MatColoringApply_JP;
  mc->ops->view           = NULL;
  mc->ops->destroy        = MatColoringDestroy_JP;
  mc->ops->setfromoptions = MatColoringSetFromOptions_JP;
  PetscFunctionReturn(0);
}
