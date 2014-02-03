#include <petsc-private/matimpl.h>      /*I "petscmat.h"  I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

typedef struct {
  PetscReal dummy;
} MC_Greedy;

#undef __FUNCT__
#define __FUNCT__ "GreedyCreateWeights_Private"
PetscErrorCode GreedyCreateWeights_Private(MatColoring mc,PetscReal **wts,PetscInt **perm)
{
  PetscRandom    rand;
  PetscReal      r;
  PetscInt       i,s,e,n,ncols;
  Mat            G=mc->mat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetOwnershipRange(G,&s,&e);CHKERRQ(ierr);
  n=e-s;
  ierr = PetscMalloc1(n,wts);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,perm);CHKERRQ(ierr);
  /* each weight should be the degree plus a random perturbation */
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)mc),&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  for (i=s;i<e;i++) {
    ierr = MatGetRow(G,i,&ncols,NULL,NULL);CHKERRQ(ierr);
    ierr = PetscRandomGetValueReal(rand,&r);CHKERRQ(ierr);
    (*wts)[i-s] = ncols + PetscAbsReal(r);
    (*perm)[i-s] = i-s;
    ierr = MatRestoreRow(G,i,&ncols,NULL,NULL);CHKERRQ(ierr);
  }
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = PetscSortRealWithPermutation(n,*wts,*perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringDestroy_Greedy"
PetscErrorCode MatColoringDestroy_Greedy(MatColoring mc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(mc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GreedyColoringLocalDistanceOne_Private"
PETSC_EXTERN PetscErrorCode GreedyColoringLocalDistanceOne_Private(MatColoring mc,PetscReal *wts,PetscInt *lperm,ISColoringValue *colors)
{
  PetscInt        i,j,k,s,e,n,nd,nd_global,n_global,idx,ncols,maxcolors,nneighbors,maxneighbors,*mask;
  PetscErrorCode  ierr;
  Mat             m=mc->mat;
  Mat_MPIAIJ      *aij = (Mat_MPIAIJ*)m->data;
  Mat             md=NULL,mo=NULL;
  PetscBool       isMPIAIJ,isSEQAIJ;
  ISColoringValue pcol,ncol;
  const PetscInt  *cidx;
  Vec             wtvec,owtvec,colvec,ocolvec;
  VecScatter      oscatter;
  PetscScalar     *wtar,*owtar,*colar,*ocolar;

  PetscFunctionBegin;
  ierr = MatGetSize(m,&n_global,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(m,&s,&e);CHKERRQ(ierr);
  n=e-s;
  maxneighbors=n;
  nd_global = 0;
  /* get the matrix communication structures */
  ierr = PetscObjectTypeCompare((PetscObject)m, MATMPIAIJ, &isMPIAIJ); CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)m, MATSEQAIJ, &isSEQAIJ); CHKERRQ(ierr);
  if (isMPIAIJ) {
    md=aij->A;
    mo=aij->B;
  } else if (isSEQAIJ) {
    /* no off-processor nodes */
    md=m;
    mo=NULL;
  } else {
    SETERRQ(PetscObjectComm((PetscObject)mc),PETSC_ERR_ARG_WRONG,"Matrix must be AIJ for greedy coloring");
  }
  /* create the vectors and communication structures if necessary */
  ierr = MatGetVecs(m,&wtvec,NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(wtvec,&colvec);CHKERRQ(ierr);
  owtvec=NULL;
  ocolvec=NULL;
  oscatter=NULL;
  if (mo) {
    ierr = VecDuplicate(aij->lvec,&owtvec);CHKERRQ(ierr);
    ierr = VecDuplicate(aij->lvec,&ocolvec);CHKERRQ(ierr);
    oscatter=aij->Mvctx;
  }
  ierr = VecSet(colvec,(PetscScalar)IS_COLORING_MAX);CHKERRQ(ierr);
  if (ocolvec) {ierr = VecSet(ocolvec,(PetscScalar)IS_COLORING_MAX);CHKERRQ(ierr);}

  maxcolors=IS_COLORING_MAX;
  if (mc->maxcolors) maxcolors=mc->maxcolors;
  ierr = PetscMalloc1(maxneighbors,&mask);CHKERRQ(ierr);

  /* transfer neighbor weights */
  ierr = VecGetArray(wtvec,&wtar);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    wtar[i]=wts[i];
  }
  ierr = VecRestoreArray(wtvec,&wtar);CHKERRQ(ierr);
  if (mo) {
    ierr = PetscLogEventBegin(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
    ierr = VecScatterBegin(oscatter,wtvec,owtvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(oscatter,wtvec,owtvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
  }

  while (nd_global < n_global) {
    nd=n;
    ierr = VecGetArray(colvec,&colar);CHKERRQ(ierr);
    if (mo) {
      ierr = VecGetArray(ocolvec,&ocolar);CHKERRQ(ierr);
    }
    /* assign lowest possible color to each local vertex */
    ierr = PetscLogEventBegin(Mat_Coloring_Local,mc,0,0,0);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
      idx=n-lperm[i]-1;
      if ((ISColoringValue)PetscRealPart(colar[idx]) == IS_COLORING_MAX) {
        ierr = MatGetRow(md,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
        nneighbors=0;
        for (j=0;j<ncols;j++) {
          if ((PetscInt)PetscRealPart(colar[cidx[j]]) != IS_COLORING_MAX) {
            mask[nneighbors] = (PetscInt)PetscRealPart(colar[cidx[j]]);
            nneighbors++;
            if (nneighbors>=maxneighbors) {
              PetscInt *newmask;
              ierr = PetscMalloc1(maxneighbors*2,&newmask);CHKERRQ(ierr);
              for(k=0;k<maxneighbors;k++) {
                newmask[k]=mask[k];
              }
              ierr = PetscFree(mask);CHKERRQ(ierr);
              mask=newmask;
              maxneighbors*=2;
            }
          }
        }
        ierr = MatRestoreRow(md,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
        if (mo) {
          ierr = MatGetRow(mo,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
          for (j=0;j<ncols;j++) {
            if ((PetscInt)PetscRealPart(ocolar[cidx[j]]) != IS_COLORING_MAX) {
              mask[nneighbors] = (PetscInt)PetscRealPart(ocolar[cidx[j]]);
              nneighbors++;
              if (nneighbors>=maxneighbors) {
                PetscInt *newmask;
                ierr = PetscMalloc1(maxneighbors*2,&newmask);CHKERRQ(ierr);
                for(k=0;k<maxneighbors;k++) {
                  newmask[k]=mask[k];
                }
                ierr = PetscFree(mask);CHKERRQ(ierr);
                mask=newmask;
                maxneighbors*=2;
              }
            }
          }
          ierr = MatRestoreRow(mo,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
        }
        ierr = PetscSortInt(nneighbors,mask);CHKERRQ(ierr);
        /* assign this one the lowest color possible by seeing if there's a gap in the sequence of sorted neighbor colors */
        pcol=0;
        for (j=0;j<nneighbors;j++) {
          ncol=mask[j];
          if (ncol-pcol > 0) {
            break;
          } else {
            pcol=ncol+1;
          }
        }
        if (pcol > maxcolors) pcol=maxcolors;
        colar[idx]=pcol;
      }
    }
    ierr = PetscLogEventEnd(Mat_Coloring_Local,mc,0,0,0);CHKERRQ(ierr);
    ierr = VecRestoreArray(colvec,&colar);CHKERRQ(ierr);
    if (mo) {
      ierr = VecRestoreArray(ocolvec,&ocolar);CHKERRQ(ierr);
    }
    if (mo) {
      /* transfer neighbor colors */
      ierr = PetscLogEventBegin(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
      ierr = VecScatterBegin(oscatter,colvec,ocolvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(oscatter,colvec,ocolvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
      /* check for conflicts -- this is merely checking if any adjacent off-processor rows have the same color and marking the ones that are lower weight locally for changing */
      ierr = VecGetArray(wtvec,&wtar);CHKERRQ(ierr);
      ierr = VecGetArray(owtvec,&owtar);CHKERRQ(ierr);
      ierr = VecGetArray(colvec,&colar);CHKERRQ(ierr);
      ierr = VecGetArray(ocolvec,&ocolar);CHKERRQ(ierr);
      for (i=0;i<n;i++) {
        ierr = MatGetRow(mo,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
        for (j=0;j<ncols;j++) {
          /* in the case of conflicts, the highest weight one stays and the others go */
          if ((PetscRealPart(ocolar[cidx[j]]) == PetscRealPart(colar[i])) && (PetscRealPart(owtar[cidx[j]]) > PetscRealPart(wtar[i]))) {
            colar[i]=IS_COLORING_MAX;
            nd--;
          }
        }
        ierr = MatRestoreRow(mo,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
      }
      ierr = VecRestoreArray(wtvec,&wtar);CHKERRQ(ierr);
      ierr = VecRestoreArray(owtvec,&owtar);CHKERRQ(ierr);
      ierr = VecRestoreArray(colvec,&colar);CHKERRQ(ierr);
      ierr = VecRestoreArray(ocolvec,&ocolar);CHKERRQ(ierr);
      nd_global=0;
    }
    ierr = MPI_Allreduce(&nd,&nd_global,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)mc));CHKERRQ(ierr);
  }
  ierr = VecGetArray(colvec,&colar);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    colors[i]=(PetscInt)PetscRealPart(colar[i]);
  }
  ierr = VecRestoreArray(colvec,&colar);CHKERRQ(ierr);
  ierr = PetscFree(mask);CHKERRQ(ierr);
  ierr = VecDestroy(&wtvec);CHKERRQ(ierr);
  ierr = VecDestroy(&colvec);CHKERRQ(ierr);
  if (mo) {
    ierr = VecDestroy(&owtvec);CHKERRQ(ierr);
    ierr = VecDestroy(&ocolvec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "GreedyColoringLocalDistanceTwo_Private"
PETSC_EXTERN PetscErrorCode GreedyColoringLocalDistanceTwo_Private(MatColoring mc,PetscReal *wts,PetscInt *lperm,ISColoringValue *colors)
{
  PetscInt        i,j,k,l,s,e,n,nd,nd_global,n_global,idx,ncols,maxcolors,mcol,mcol_global,nmask,nd1cols,*mask,*d1cols,*bad,maxmask,ccol;
  PetscErrorCode  ierr;
  Mat             m=mc->mat;
  Mat_MPIAIJ      *aij = (Mat_MPIAIJ*)m->data;
  Mat             md=NULL,mo=NULL;
  /* Mat             mtd=NULL,mto=NULL; */
  PetscBool       isMPIAIJ,isSEQAIJ;
  ISColoringValue pcol,ncol;
  const PetscInt  *cidx;
  Vec             wtvec,owtvec,colvec,ocolvec,confvec,oconfvec;
  VecScatter      oscatter;
  PetscScalar     *wtar,*owtar,*colar,*ocolar,*confar,*oconfar;
  PetscReal       *colorweights;

  PetscFunctionBegin;
  ierr = MatGetSize(m,&n_global,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(m,&s,&e);CHKERRQ(ierr);
  n=e-s;
  nd_global = 0;
  /* get the matrix communication structures */
  ierr = PetscObjectTypeCompare((PetscObject)m, MATMPIAIJ, &isMPIAIJ); CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)m, MATSEQAIJ, &isSEQAIJ); CHKERRQ(ierr);
  if (isMPIAIJ) {
    md=aij->A;
    mo=aij->B;
  } else if (isSEQAIJ) {
    /* no off-processor nodes */
    md=m;
  } else {
    SETERRQ(PetscObjectComm((PetscObject)mc),PETSC_ERR_ARG_WRONG,"Matrix must be AIJ for greedy coloring");
  }
  /* create the vectors and communication structures if necessary */
  ierr = MatGetVecs(m,&wtvec,NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(wtvec,&colvec);CHKERRQ(ierr);
  ierr = VecDuplicate(wtvec,&confvec);CHKERRQ(ierr);
  owtvec=NULL;
  ocolvec=NULL;
  oscatter=NULL;
  oconfvec=NULL;
  if (mo) {
    ierr = VecDuplicate(aij->lvec,&owtvec);CHKERRQ(ierr);
    ierr = VecDuplicate(aij->lvec,&ocolvec);CHKERRQ(ierr);
    ierr = VecDuplicate(aij->lvec,&oconfvec);CHKERRQ(ierr);
    oscatter=aij->Mvctx;
  }
  ierr = VecSet(colvec,(PetscScalar)IS_COLORING_MAX);CHKERRQ(ierr);
  if (ocolvec) {ierr = VecSet(ocolvec,(PetscScalar)IS_COLORING_MAX);CHKERRQ(ierr);}

  maxcolors=IS_COLORING_MAX;
  if (mc->maxcolors) maxcolors=mc->maxcolors;
  maxmask=n;
  ierr = PetscMalloc1(n,&d1cols);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxmask,&mask);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&bad);CHKERRQ(ierr);

  /* transfer neighbor weights */
  ierr = VecGetArray(wtvec,&wtar);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    wtar[i]=wts[i];
  }
  ierr = VecRestoreArray(wtvec,&wtar);CHKERRQ(ierr);
  if (mo) {
    ierr = PetscLogEventBegin(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
    ierr = VecScatterBegin(oscatter,wtvec,owtvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(oscatter,wtvec,owtvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
  }
  /* values below "bad" are verboten for a given column due to being nixed at the parallel level */
  for (i=0;i<n;i++) {
    bad[i]=-1;
  }
  mcol=0;
  while (nd_global < n_global) {
    nd=n;
    ierr = VecGetArray(colvec,&colar);CHKERRQ(ierr);
    if (mo) {
      ierr = VecGetArray(ocolvec,&ocolar);CHKERRQ(ierr);
    }
    /* assign lowest possible color to each local vertex */
    mcol_global=0;
    ierr = PetscLogEventBegin(Mat_Coloring_Local,mc,0,0,0);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
      idx=n-lperm[i]-1;
      if ((PetscInt)PetscRealPart(colar[idx]) == IS_COLORING_MAX) {
        nmask=0;
        nd1cols=0;
        /* diagonal distance-one rows */
        ierr = MatGetRow(md,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
        for (j=0;j<ncols;j++) {
          d1cols[nd1cols] = cidx[j];
          nd1cols++;
          if ((PetscInt)PetscRealPart(colar[cidx[j]]) != IS_COLORING_MAX) {
            mask[nmask] = (PetscInt)PetscRealPart(colar[cidx[j]]);
            nmask++;
            if (nmask>=maxmask) {
              PetscInt *newmask;
              ierr = PetscMalloc1(maxmask*2,&newmask);CHKERRQ(ierr);
              for(k=0;k<maxmask;k++) {
                newmask[k]=mask[k];
              }
              ierr = PetscFree(mask);CHKERRQ(ierr);
              mask=newmask;
              maxmask*=2;
            }
          }
        }
        ierr = MatRestoreRow(md,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
        /* off-diagonal distance-one rows */
        if (mo) {
          ierr = MatGetRow(mo,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
          for (j=0;j<ncols;j++) {
            if ((PetscInt)PetscRealPart(ocolar[cidx[j]]) != IS_COLORING_MAX) {
              mask[nmask] = (PetscInt)PetscRealPart(ocolar[cidx[j]]);
              nmask++;
              if (nmask>=maxmask) {
                PetscInt *newmask;
                ierr = PetscMalloc1(maxmask*2,&newmask);CHKERRQ(ierr);
                for(k=0;k<maxmask;k++) {
                  newmask[k]=mask[k];
                }
                ierr = PetscFree(mask);CHKERRQ(ierr);
                mask=newmask;
                maxmask*=2;
              }
            }
          }
          ierr = MatRestoreRow(mo,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
        }
        /* diagonal distance-two rows */
        for (j=0;j<nd1cols;j++) {
          ierr = MatGetRow (md,d1cols[j],&ncols,&cidx,NULL);CHKERRQ(ierr);
          for (l=0;l<ncols;l++) {
            if ((PetscInt)PetscRealPart(colar[cidx[l]]) != IS_COLORING_MAX) {
              mask[nmask] = (PetscInt)PetscRealPart(colar[cidx[l]]);
              nmask++;
              if (nmask>=maxmask) {
                PetscInt *newmask;
                ierr = PetscMalloc1(maxmask*2,&newmask);CHKERRQ(ierr);
                for(k=0;k<maxmask;k++) {
                  newmask[k]=mask[k];
                }
                ierr = PetscFree(mask);CHKERRQ(ierr);
                mask=newmask;
                maxmask*=2;
              }
            }
          }
          ierr = MatRestoreRow(md,d1cols[j],&ncols,&cidx,NULL);CHKERRQ(ierr);
        }
        /* off-diagonal distance-two rows */
        if (mo) {
          for (j=0;j<nd1cols;j++) {
            ierr = MatGetRow (mo,d1cols[j],&ncols,&cidx,NULL);CHKERRQ(ierr);
            for (l=0;l<ncols;l++) {
              if ((PetscInt)PetscRealPart(ocolar[cidx[l]]) != IS_COLORING_MAX) {
                mask[nmask] = (PetscInt)PetscRealPart(ocolar[cidx[l]]);
                nmask++;
                if (nmask>=maxmask) {
                  PetscInt *newmask;
                  ierr = PetscMalloc1(maxmask*2,&newmask);CHKERRQ(ierr);
                  for(k=0;k<maxmask;k++) {
                    newmask[k]=mask[k];
                  }
                  ierr = PetscFree(mask);CHKERRQ(ierr);
                  mask=newmask;
                  maxmask*=2;
                }
              }
            }
            ierr = MatRestoreRow(mo,d1cols[j],&ncols,&cidx,NULL);CHKERRQ(ierr);
          }
        }
        ierr = PetscSortInt(nmask,mask);CHKERRQ(ierr);
        /* assign this one the lowest color possible by seeing if there's a gap in the sequence of sorted neighbor colors */
        pcol=0;
        for (j=0;j<nmask;j++) {
          ncol=mask[j];
          if (ncol-pcol > 0 && pcol>bad[idx]) {
            break;
          } else {
            pcol=ncol+1;
          }
        }
        if (pcol<=bad[idx]) pcol=bad[idx]+1;
        if (pcol > maxcolors) pcol=maxcolors;
        colar[idx]=pcol;
        if (pcol > mcol) mcol = pcol;
      }
    }
    ierr = PetscLogEventEnd(Mat_Coloring_Local,mc,0,0,0);CHKERRQ(ierr);
    ierr = VecRestoreArray(colvec,&colar);CHKERRQ(ierr);
    if (mo) {
      ierr = VecRestoreArray(ocolvec,&ocolar);CHKERRQ(ierr);
    }
    if (mo) {
      /* transfer neighbor colors */
      ierr = PetscLogEventBegin(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
      ierr = VecScatterBegin(oscatter,colvec,ocolvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(oscatter,colvec,ocolvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
      /* find the maximum color assigned locally and allocate a mask */
      ierr = MPI_Allreduce(&mcol,&mcol_global,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)mc));CHKERRQ(ierr);
      ierr = PetscMalloc1(mcol_global+1,&colorweights);CHKERRQ(ierr);
      /* check for conflicts */
      ierr = VecGetArray(wtvec,&wtar);CHKERRQ(ierr);
      ierr = VecGetArray(owtvec,&owtar);CHKERRQ(ierr);
      ierr = VecGetArray(colvec,&colar);CHKERRQ(ierr);
      ierr = VecGetArray(ocolvec,&ocolar);CHKERRQ(ierr);
      ierr = VecSet(confvec,0);CHKERRQ(ierr);
      ierr = VecSet(oconfvec,0);CHKERRQ(ierr);
      ierr = VecGetArray(confvec,&confar);CHKERRQ(ierr);
      ierr = VecGetArray(oconfvec,&oconfar);CHKERRQ(ierr);
      for (i=0;i<n;i++) {
        ierr = MatGetRow(mo,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
        if (ncols > 0) {
          /* fill in the mask */
          for (j=0;j<mcol_global+1;j++) {
            colorweights[j]=0;
          }
          colorweights[(PetscInt)PetscRealPart(colar[i])]=PetscRealPart(wtar[i]);
          /* fill in the off-diagonal part of the mask */
          for (j=0;j<ncols;j++) {
            ccol=(PetscInt)PetscRealPart(ocolar[cidx[j]]);
            if (colorweights[ccol] < PetscRealPart(owtar[cidx[j]])) {
              colorweights[ccol] = PetscRealPart(owtar[cidx[j]]);
            }
          }
          ierr = MatRestoreRow(mo,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
          /* fill in the on-diagonal part of the mask */
          ierr = MatGetRow(md,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
          for (j=0;j<ncols;j++) {
            ccol=(PetscInt)PetscRealPart(colar[cidx[j]]);
            if (ccol < maxcolors) {
              if (colorweights[ccol] < PetscRealPart(wtar[cidx[j]])) {
                colorweights[ccol] = PetscRealPart(wtar[cidx[j]]);
              }
            }
          }
          ierr = MatRestoreRow(md,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
          /* go back through and set up on and off-diagonal conflict vectors */
          ierr = MatGetRow(md,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
          for (j=0;j<ncols;j++) {
            ccol=(PetscInt)PetscRealPart(colar[cidx[j]]);
            if (ccol < maxcolors) {
              if (colorweights[ccol] > PetscRealPart(wtar[cidx[j]])) {
                confar[cidx[j]]+=1;
              }
            }
          }
          ierr = MatRestoreRow(md,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
          ierr = MatGetRow(mo,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
          for (j=0;j<ncols;j++) {
            ccol=(PetscInt)PetscRealPart(ocolar[cidx[j]]);
            if (ccol < maxcolors) {
              if (colorweights[ccol] > PetscRealPart(owtar[cidx[j]])) {
                oconfar[cidx[j]]+=1;
              }
            }
          }
          ierr = MatRestoreRow(mo,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
        } else {
          ierr = MatRestoreRow(mo,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
        }
      }
      ierr = VecRestoreArray(wtvec,&wtar);CHKERRQ(ierr);
      ierr = VecRestoreArray(owtvec,&owtar);CHKERRQ(ierr);
      ierr = VecRestoreArray(colvec,&colar);CHKERRQ(ierr);
      ierr = VecRestoreArray(ocolvec,&ocolar);CHKERRQ(ierr);
      ierr = VecRestoreArray(confvec,&confar);CHKERRQ(ierr);
      ierr = VecRestoreArray(oconfvec,&oconfar);CHKERRQ(ierr);
      nd_global=0;
      ierr = PetscFree(colorweights);CHKERRQ(ierr);

      ierr = PetscLogEventBegin(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
      ierr = VecScatterBegin(oscatter,oconfvec,confvec,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(oscatter,oconfvec,confvec,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
      /* go through and unset local colors that have conflicts */
      ierr = VecGetArray(confvec,&confar);CHKERRQ(ierr);
      ierr = VecGetArray(colvec,&colar);CHKERRQ(ierr);
      for (i=0;i<n;i++) {
        if (PetscRealPart(confar[i]) > 0) {
          bad[i]=(PetscInt)PetscRealPart(colar[i]);
          colar[i] = IS_COLORING_MAX;
          nd--;
        }
      }
      ierr = VecRestoreArray(confvec,&confar);CHKERRQ(ierr);
      ierr = VecRestoreArray(colvec,&colar);CHKERRQ(ierr);
    }
    ierr = MPI_Allreduce(&nd,&nd_global,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)mc));CHKERRQ(ierr);
  }
  ierr = VecGetArray(colvec,&colar);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    colors[i]=(PetscInt)PetscRealPart(colar[i]);
  }
  ierr = VecRestoreArray(colvec,&colar);CHKERRQ(ierr);
  ierr = PetscFree(mask);CHKERRQ(ierr);
  ierr = PetscFree(d1cols);CHKERRQ(ierr);
  ierr = PetscFree(bad);CHKERRQ(ierr);
  ierr = VecDestroy(&wtvec);CHKERRQ(ierr);
  ierr = VecDestroy(&colvec);CHKERRQ(ierr);
  ierr = VecDestroy(&confvec);CHKERRQ(ierr);
  if (mo) {
    ierr = VecDestroy(&owtvec);CHKERRQ(ierr);
    ierr = VecDestroy(&ocolvec);CHKERRQ(ierr);
    ierr = VecDestroy(&oconfvec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringApply_Greedy"
PETSC_EXTERN PetscErrorCode MatColoringApply_Greedy(MatColoring mc,ISColoring *iscoloring)
{
  /* MC_Greedy       *gr=(MC_Greedy*)mc->data; */
  PetscErrorCode  ierr;
  PetscInt        finalcolor,finalcolor_global;
  ISColoringValue *colors;
  PetscInt        ncolstotal,ncols;
  PetscReal       *wts;
  PetscInt        i,*lperm;

  PetscFunctionBegin;
  ierr = MatGetSize(mc->mat,NULL,&ncolstotal);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mc->mat,NULL,&ncols);CHKERRQ(ierr);
  ierr = GreedyCreateWeights_Private(mc,&wts,&lperm);CHKERRQ(ierr);
  ierr = PetscMalloc1(ncols,&colors);CHKERRQ(ierr);
  if (mc->dist == 1) {
    ierr = GreedyColoringLocalDistanceOne_Private(mc,wts,lperm,colors);CHKERRQ(ierr);
  } else if (mc->dist == 2) {
    ierr = GreedyColoringLocalDistanceTwo_Private(mc,wts,lperm,colors);CHKERRQ(ierr);
  } else {
    SETERRQ(PetscObjectComm((PetscObject)mc),PETSC_ERR_ARG_OUTOFRANGE,"Only distance 1 and distance 2 supported by MatColoringGreedy");
  }
  finalcolor=0;
  for (i=0;i<ncols;i++) {
    if (colors[i] > finalcolor) finalcolor=colors[i];
  }
  finalcolor_global=0;
  ierr = MPI_Allreduce(&finalcolor,&finalcolor_global,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)mc));CHKERRQ(ierr);
  ierr = PetscLogEventBegin(Mat_Coloring_ISCreate,mc,0,0,0);CHKERRQ(ierr);
  ierr = ISColoringCreate(PetscObjectComm((PetscObject)mc),finalcolor_global+1,ncols,colors,iscoloring);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(Mat_Coloring_ISCreate,mc,0,0,0);CHKERRQ(ierr);
  ierr = PetscFree(wts);CHKERRQ(ierr);
  ierr = PetscFree(lperm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringCreate_Greedy"
/*MC
  MATCOLORINGGREEDY - Greedy-with-conflict correction based Matrix Coloring for distance 1 and 2.

   Level: beginner

   References:

   Bozdag et al. "A Parallel Distance-2 Graph Coloring Algorithm for Distributed Memory Computers"
   HPCC'05 Proceedings of the First international conference on High Performance Computing and Communications
   Pages 796--806

.seealso: MatColoringCreate(), MatColoring, MatColoringSetType()
M*/
PETSC_EXTERN PetscErrorCode MatColoringCreate_Greedy(MatColoring mc)
{
  MC_Greedy      *gr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr                    = PetscNewLog(mc,&gr);CHKERRQ(ierr);
  mc->data                = gr;
  mc->ops->apply          = MatColoringApply_Greedy;
  mc->ops->view           = NULL;
  mc->ops->destroy        = MatColoringDestroy_Greedy;
  mc->ops->setfromoptions = NULL;
  PetscFunctionReturn(0);
}
