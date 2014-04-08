#include <petsc-private/matimpl.h>      /*I "petscmat.h"  I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petscsf.h>

typedef struct {
  PetscSF sf;
} MC_JP;

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
#define __FUNCT__ "MCJPGreatestWeight_Private"
PetscErrorCode MCJPGreatestWeight_Private(MatColoring mc,const PetscReal *weights,PetscReal *maxweights)
{
  MC_JP          *jp = (MC_JP*)mc->data;
  PetscErrorCode ierr;
  Mat            G=mc->mat,dG,oG;
  PetscBool      isSeq,isMPI;
  Mat_MPIAIJ     *aij;
  Mat_SeqAIJ     *daij,*oaij;
  PetscInt       *di,*oi,*dj,*oj;
  PetscSF        sf=jp->sf;
  PetscLayout    layout;
  PetscInt       dn,on;
  PetscInt       i,j,l;
  PetscReal      *dwts,*owts;
  PetscInt       ncols;
  const PetscInt *cols;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)G,MATSEQAIJ,&isSeq);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)G,MATMPIAIJ,&isMPI);CHKERRQ(ierr);
  if (!isSeq && !isMPI) {
    SETERRQ(PetscObjectComm((PetscObject)G),PETSC_ERR_ARG_WRONGSTATE,"MatColoringDegrees requires an MPI/SEQAIJ Matrix");
  }
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
    ierr = MatGetSize(oG,&dn,&on);CHKERRQ(ierr);
    if (!sf) {
      ierr = PetscSFCreate(PetscObjectComm((PetscObject)mc),&sf);CHKERRQ(ierr);
      ierr = MatGetLayouts(G,&layout,NULL);CHKERRQ(ierr);
      ierr = PetscSFSetGraphLayout(sf,layout,on,NULL,PETSC_COPY_VALUES,aij->garray);CHKERRQ(ierr);
      jp->sf = sf;
    }
  } else {
    dG = G;
    oG = NULL;
    ierr = MatGetSize(dG,NULL,&dn);CHKERRQ(ierr);
    daij = (Mat_SeqAIJ*)dG->data;
    di = daij->i;
    dj = daij->j;
    sf = NULL;
  }
  /* set up the distance-zero weights */
  ierr = PetscMalloc1(dn,&dwts);CHKERRQ(ierr);
  if (oG) {
    ierr = PetscMalloc1(on,&owts);CHKERRQ(ierr);
  }
  for (i=0;i<dn;i++) {
    maxweights[i] = weights[i];
    dwts[i] = maxweights[i];
  }
  /* get the off-diagonal weights */
  if (oG) {
    ierr = PetscSFBcastBegin(sf,MPIU_REAL,dwts,owts);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf,MPIU_REAL,dwts,owts);CHKERRQ(ierr);
  }
  for (l=0;l<mc->dist;l++) {
    /* check for on-diagonal greater weights */
    for (i=0;i<dn;i++) {
      ncols = di[i+1]-di[i];
      cols = &(dj[di[i]]);
      for (j=0;j<ncols;j++) {
        if (dwts[cols[j]] > maxweights[i]) maxweights[i] = dwts[cols[j]];
      }
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
        ierr = PetscSFBcastBegin(sf,MPIU_REAL,dwts,owts);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(sf,MPIU_REAL,dwts,owts);CHKERRQ(ierr);
      }
    }
  }
 ierr = PetscFree(dwts);CHKERRQ(ierr);
  if (oG) {
    ierr = PetscFree(owts);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MCJPMinColor_Private"
PetscErrorCode MCJPMinColor_Private(MatColoring mc,ISColoringValue maxcolor,const ISColoringValue *colors,ISColoringValue *mincolors)
{
  MC_JP          *jp = (MC_JP*)mc->data;
  PetscErrorCode ierr;
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
  PetscInt       *dmask,*omask,*cmask,curmask;
  PetscInt       ncols;
  const PetscInt *cols;

  PetscFunctionBegin;
  maskradix = sizeof(PetscInt)*8;
  maskrounds = 1 + maxcolor / (maskradix);
  maskbase = 0;
  ierr = PetscObjectTypeCompare((PetscObject)G,MATSEQAIJ,&isSeq);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)G,MATMPIAIJ,&isMPI);CHKERRQ(ierr);
  if (!isSeq && !isMPI) {
    SETERRQ(PetscObjectComm((PetscObject)G),PETSC_ERR_ARG_WRONGSTATE,"MatColoringDegrees requires an MPI/SEQAIJ Matrix");
  }
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
    ierr = MatGetSize(oG,&dn,&on);CHKERRQ(ierr);
    if (!sf) {
      ierr = PetscSFCreate(PetscObjectComm((PetscObject)mc),&sf);CHKERRQ(ierr);
      ierr = MatGetLayouts(G,&layout,NULL);CHKERRQ(ierr);
      ierr = PetscSFSetGraphLayout(sf,layout,on,NULL,PETSC_COPY_VALUES,aij->garray);CHKERRQ(ierr);
      jp->sf = sf;
    }
  } else {
    dG = G;
    oG = NULL;
    ierr = MatGetSize(dG,NULL,&dn);CHKERRQ(ierr);
    daij = (Mat_SeqAIJ*)dG->data;
    di = daij->i;
    dj = daij->j;
    sf = NULL;
  }
  for (i=0;i<dn;i++) {
    mincolors[i] = IS_COLORING_MAX;
  }
  /* set up the distance-zero weights */
  ierr = PetscMalloc1(dn,&dmask);CHKERRQ(ierr);
  ierr = PetscMalloc1(dn,&cmask);CHKERRQ(ierr);
  if (oG) {
    ierr = PetscMalloc1(on,&omask);CHKERRQ(ierr);
  }
  /* get the off-diagonal weights */
  for (k=0;k<maskrounds;k++) {
    for (i=0;i<dn;i++) {
      cmask[i] = 0;
      if (colors[i] < maskbase+maskradix && colors[i] >= maskbase)
        cmask[i] = 1 << (colors[i]-maskbase);
      dmask[i] = cmask[i];
    }
    if (oG) {
      ierr = PetscSFBcastBegin(sf,MPIU_INT,dmask,omask);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sf,MPIU_INT,dmask,omask);CHKERRQ(ierr);
    }
    for (l=0;l<mc->dist;l++) {
      /* check for on-diagonal greater weights */
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
          ierr = PetscSFBcastBegin(sf,MPIU_INT,dmask,omask);CHKERRQ(ierr);
          ierr = PetscSFBcastEnd(sf,MPIU_INT,dmask,omask);CHKERRQ(ierr);
        }
      }
    }
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
    maskbase += maskradix;
  }
  for (i=0;i<dn;i++) {
    if (mincolors[i] == IS_COLORING_MAX) {
      mincolors[i] = maxcolor+1;
    }
  }
 ierr = PetscFree(dmask);CHKERRQ(ierr);
 ierr = PetscFree(cmask);CHKERRQ(ierr);
  if (oG) {
    ierr = PetscFree(omask);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringApply_JP"
PETSC_EXTERN PetscErrorCode MatColoringApply_JP(MatColoring mc,ISColoring *iscoloring)
{
  PetscErrorCode  ierr;
  MC_JP          *jp = (MC_JP*)mc->data;
  PetscInt        i,nadded,nadded_total,nadded_total_old,ntotal,n;
  PetscInt        maxcolor_local=0,maxcolor_global;
  PetscMPIInt     rank;
  PetscReal       *weights,*maxweights;
  ISColoringValue  *color,*mincolor;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)mc),&rank);CHKERRQ(ierr);
  ierr = MatColoringCreateWeights(mc,&weights,NULL);CHKERRQ(ierr);
  ierr = MatGetSize(mc->mat,NULL,&ntotal);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mc->mat,NULL,&n);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&maxweights);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&color);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&mincolor);CHKERRQ(ierr);
  /* ierr = MCJPGreatestWeight_Private(mc,weights,maxweights);CHKERRQ(ierr); */
  for (i=0;i<n;i++) {
    color[i] = IS_COLORING_MAX;
    mincolor[i] = 0;
  }
  nadded=0;
  nadded_total=0;
  nadded_total_old=0;
  while (nadded_total < ntotal) {
    ierr = MCJPGreatestWeight_Private(mc,weights,maxweights);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
      if (weights[i] >= maxweights[i] && weights[i] > 0.) {
        /* pick this one */
        if (mc->maxcolors > mincolor[i]) {
          color[i] = mincolor[i];
        } else {
          color[i] = mc->maxcolors;
        }
        if (color[i] > maxcolor_local) maxcolor_local = color[i];
        weights[i] = 0.;
        nadded++;
      }
    }
    ierr = MPI_Allreduce(&nadded,&nadded_total,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)mc));CHKERRQ(ierr);
    if (nadded_total == nadded_total_old) {SETERRQ(PetscObjectComm((PetscObject)mc),PETSC_ERR_NOT_CONVERGED,"JP didn't make progress");}
    nadded_total_old = nadded_total;
    maxcolor_global = 0;
    ierr = MPI_Allreduce(&maxcolor_local,&maxcolor_global,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)mc));CHKERRQ(ierr);
    ierr = MCJPMinColor_Private(mc,maxcolor_global,color,mincolor);CHKERRQ(ierr);
  }

  ierr = MPI_Allreduce(&maxcolor_local,&maxcolor_global,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)mc));CHKERRQ(ierr);
  ierr = PetscLogEventBegin(Mat_Coloring_ISCreate,mc,0,0,0);CHKERRQ(ierr);
  ierr = ISColoringCreate(PetscObjectComm((PetscObject)mc),maxcolor_global+1,n,color,iscoloring);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(Mat_Coloring_ISCreate,mc,0,0,0);CHKERRQ(ierr);
  ierr = PetscFree(weights);CHKERRQ(ierr);
  ierr = PetscFree(maxweights);CHKERRQ(ierr);
  ierr = PetscFree(mincolor);CHKERRQ(ierr);
  if (jp->sf) {ierr = PetscSFDestroy(&jp->sf);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringCreate_JP"
/*MC
  MATCOLORINGJP - Parallel Jones-Plassmann Coloring

   Level: beginner

   Notes: This method uses a parallel Luby-style coloring with with weights to choose an independent set of processor
   boundary vertices at each stage that may be assigned colors independently.

   References:
   M. Jones and P. Plassmann, “A parallel graph coloring heuristic,” SIAM Journal on Scientific Computing, vol. 14, no. 3,
   pp. 654–669, 1993.

.seealso: MatColoringCreate(), MatColoring, MatColoringSetType()
M*/
PETSC_EXTERN PetscErrorCode MatColoringCreate_JP(MatColoring mc)
{
  MC_JP          *jp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr                    = PetscNewLog(mc,&jp);CHKERRQ(ierr);
  jp->sf                  = NULL;
  mc->data                = jp;
  mc->ops->apply          = MatColoringApply_JP;
  mc->ops->view           = NULL;
  mc->ops->destroy        = MatColoringDestroy_JP;
  mc->ops->setfromoptions = NULL;
  PetscFunctionReturn(0);
}
