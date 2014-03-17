#include <petsc-private/matimpl.h>      /*I "petscmat.h"  I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petscsf.h>

typedef struct {
  PetscReal dummy;
} MC_Greedy;

#undef __FUNCT__
#define __FUNCT__ "GreedyDiscoverDegrees_Private"
PetscErrorCode GreedyDiscoverDegrees_Private(MatColoring mc,PetscReal **degrees)
{
  Mat            G=mc->mat;
  PetscInt       i,j,k,s,e,n,ncols;
  PetscErrorCode ierr;
  PetscReal      *dar,*odar;

  PetscFunctionBegin;
  ierr = MatGetOwnershipRange(G,&s,&e);CHKERRQ(ierr);
  n=e-s;
  ierr = PetscMalloc1(n,degrees);CHKERRQ(ierr);
  if (mc->dist==1) {
    for (i=s;i<e;i++) {
      ierr = MatGetRow(G,i,&ncols,NULL,NULL);CHKERRQ(ierr);
      (*degrees)[i-s] = ncols;
      ierr = MatRestoreRow(G,i,&ncols,NULL,NULL);CHKERRQ(ierr);
    }
  } else if (mc->dist==2) {
    PetscSF        sf;
    PetscLayout    layout;
    PetscBool      isMPIAIJ,isSEQAIJ;
    Mat            gd,go;
    PetscInt       nd1cols,*d1cols,*seen,*oseen,no;
    const PetscInt *cidx;
    ierr = PetscObjectTypeCompare((PetscObject)G,MATMPIAIJ,&isMPIAIJ); CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)G,MATSEQAIJ,&isSEQAIJ); CHKERRQ(ierr);
    go=NULL;
    gd=NULL;
    ierr = PetscMalloc1(n,&seen);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
      seen[i]=-1;
    }
    if (isMPIAIJ) {
      Mat_MPIAIJ *aij = (Mat_MPIAIJ*)G->data;
      gd=aij->A;
      go=aij->B;
      ierr = VecGetSize(aij->lvec,&no);CHKERRQ(ierr);
      ierr = PetscMalloc1(no,&oseen);CHKERRQ(ierr);
      for (i=0;i<no;i++) {
        oseen[i]=-1;
      }
      ierr = PetscSFCreate(PetscObjectComm((PetscObject)G),&sf);CHKERRQ(ierr);
      ierr = MatGetLayouts(G,&layout,NULL);CHKERRQ(ierr);
      ierr = PetscSFSetGraphLayout(sf,layout,no,NULL,PETSC_COPY_VALUES,aij->garray);CHKERRQ(ierr);
    } else if (isSEQAIJ) {
      gd=G;
    } else {
      SETERRQ(PetscObjectComm((PetscObject)mc),PETSC_ERR_ARG_WRONG,"Matrix must be AIJ for greedy coloring");
    }
    ierr = PetscMalloc1(n,&d1cols);CHKERRQ(ierr);
    ierr = PetscMalloc1(n,&dar);CHKERRQ(ierr);
    /* the degree is going to be the locally computed on and off-diagonal distance one and on-diagonal distance two with remotely computed off-processor distance-two */
    for (i=0;i<n;i++) {
      dar[i] = 0.;
    }
    if (go) {
      ierr = PetscMalloc1(no,&odar);CHKERRQ(ierr);
      for (i=0;i<no;i++) {
        odar[i] = 0.;
      }
    }
    for (i=0;i<n;i++) {
      /* on-processor distance one */
      ierr = MatGetRow(gd,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
      nd1cols=ncols;
      for (j=0;j<ncols;j++) {
        d1cols[j]=cidx[j];
        seen[cidx[j]]=i;
      }
      dar[i]+=ncols;
      ierr = MatRestoreRow(gd,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
      /* off-processor distance one */
      if (go) {
        ierr = MatGetRow(go,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
        for (j=0;j<ncols;j++) {
          oseen[cidx[j]]=i;
        }
        ierr = MatRestoreRow(go,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
      }
      for (j=0;j<nd1cols;j++) {
        /* on-processor distance two */
        ierr = MatGetRow(gd,d1cols[j],&ncols,&cidx,NULL);CHKERRQ(ierr);
        for (k=0;k<ncols;k++) {
          if (seen[cidx[k]] != i) {
            dar[i]+=1;
            seen[cidx[k]]=i;
          }
        }
        ierr = MatRestoreRow(gd,d1cols[j],&ncols,&cidx,NULL);CHKERRQ(ierr);
        if (go) {
          ierr = MatGetRow(go,d1cols[j],&ncols,&cidx,NULL);CHKERRQ(ierr);
          for (k=0;k<ncols;k++) {
            if (oseen[cidx[k]] != i) {
              odar[cidx[k]]+=1;
              dar[i]+=1;
              oseen[cidx[k]]=i;
            }
          }
          ierr = MatRestoreRow(go,d1cols[j],&ncols,&cidx,NULL);CHKERRQ(ierr);
        }
      }
    }
    if (go) {
      ierr = PetscSFReduceBegin(sf,MPIU_REAL,odar,dar,MPIU_SUM);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(sf,MPIU_REAL,odar,dar,MPIU_SUM);CHKERRQ(ierr);
    }
    for (i=0;i<n;i++) {
      (*degrees)[i]=dar[i];
    }
    ierr = PetscFree(dar);CHKERRQ(ierr);
    ierr = PetscFree(d1cols);CHKERRQ(ierr);
    ierr = PetscFree(seen);CHKERRQ(ierr);
    if (go) {
      ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
      ierr = PetscFree(odar);CHKERRQ(ierr);
      ierr = PetscFree(oseen);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GreedyCreateWeights_Private"
PetscErrorCode GreedyCreateWeights_Private(MatColoring mc,PetscReal **wts,PetscInt **perm)
{
  PetscRandom    rand;
  PetscReal      r;
  PetscInt       i,s,e,n;
  Mat            G=mc->mat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetOwnershipRange(G,&s,&e);CHKERRQ(ierr);
  n=e-s;
  ierr = PetscMalloc1(n,perm);CHKERRQ(ierr);
  /* each weight should be the degree plus a random perturbation */
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)mc),&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = GreedyDiscoverDegrees_Private(mc,wts);CHKERRQ(ierr);
  for (i=s;i<e;i++) {
    ierr = PetscRandomGetValueReal(rand,&r);CHKERRQ(ierr);
    (*wts)[i-s] += PetscAbsReal(r);
    (*perm)[i-s] = i-s;
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
  PetscInt        i,j,k,s,e,n,no,nd,nd_global,n_global,idx,ncols,maxcolors,masksize,ccol,*mask;
  PetscErrorCode  ierr;
  Mat             m=mc->mat;
  Mat_MPIAIJ      *aij = (Mat_MPIAIJ*)m->data;
  Mat             md=NULL,mo=NULL;
  PetscBool       isMPIAIJ,isSEQAIJ;
  ISColoringValue pcol;
  const PetscInt  *cidx;
  PetscInt        *lcolors,*ocolors;
  PetscReal       *owts=NULL;
  PetscSF         sf;
  PetscLayout     layout;

  PetscFunctionBegin;
  ierr = MatGetSize(m,&n_global,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(m,&s,&e);CHKERRQ(ierr);
  n=e-s;
  masksize=20;
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
    no=0;
  } else {
    SETERRQ(PetscObjectComm((PetscObject)mc),PETSC_ERR_ARG_WRONG,"Matrix must be AIJ for greedy coloring");
  }
  if (mo) {
    ierr = VecGetSize(aij->lvec,&no);CHKERRQ(ierr);
    ierr = PetscMalloc2(no,&ocolors,no,&owts);CHKERRQ(ierr);
    for(i=0;i<no;i++) {
      ocolors[i]=IS_COLORING_MAX;
    }
  }

  maxcolors=IS_COLORING_MAX;
  if (mc->maxcolors) maxcolors=mc->maxcolors;
  ierr = PetscMalloc1(masksize,&mask);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&lcolors);CHKERRQ(ierr);
  for(i=0;i<n;i++) {
    lcolors[i]=IS_COLORING_MAX;
  }
  for (i=0;i<masksize;i++) {
    mask[i]=-1;
  }
  if (mo) {
    /* transfer neighbor weights */
    ierr = PetscSFCreate(PetscObjectComm((PetscObject)m),&sf);CHKERRQ(ierr);
    ierr = MatGetLayouts(m,&layout,NULL);CHKERRQ(ierr);
    ierr = PetscSFSetGraphLayout(sf,layout,no,NULL,PETSC_COPY_VALUES,aij->garray);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(sf,MPIU_REAL,wts,owts);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf,MPIU_REAL,wts,owts);CHKERRQ(ierr);
  }
  while (nd_global < n_global) {
    nd=n;
    /* assign lowest possible color to each local vertex */
    ierr = PetscLogEventBegin(Mat_Coloring_Local,mc,0,0,0);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
      idx=n-lperm[i]-1;
      if (lcolors[idx] == IS_COLORING_MAX) {
        ierr = MatGetRow(md,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
        for (j=0;j<ncols;j++) {
          if (lcolors[cidx[j]] != IS_COLORING_MAX) {
            ccol=lcolors[cidx[j]];
            if (ccol>=masksize) {
              PetscInt *newmask;
              ierr = PetscMalloc1(masksize*2,&newmask);CHKERRQ(ierr);
              for(k=0;k<2*masksize;k++) {
                newmask[k]=-1;
              }
              for(k=0;k<masksize;k++) {
                newmask[k]=mask[k];
              }
              ierr = PetscFree(mask);CHKERRQ(ierr);
              mask=newmask;
              masksize*=2;
            }
            mask[ccol]=idx;
          }
        }
        ierr = MatRestoreRow(md,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
        if (mo) {
          ierr = MatGetRow(mo,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
          for (j=0;j<ncols;j++) {
            if (ocolors[cidx[j]] != IS_COLORING_MAX) {
              ccol=ocolors[cidx[j]];
              if (ccol>=masksize) {
                PetscInt *newmask;
                ierr = PetscMalloc1(masksize*2,&newmask);CHKERRQ(ierr);
                for(k=0;k<2*masksize;k++) {
                  newmask[k]=-1;
                }
                for(k=0;k<masksize;k++) {
                  newmask[k]=mask[k];
                }
                ierr = PetscFree(mask);CHKERRQ(ierr);
                mask=newmask;
                masksize*=2;
              }
              mask[ccol]=idx;
            }
          }
          ierr = MatRestoreRow(mo,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
        }
        for (j=0;j<masksize;j++) {
          if (mask[j]!=idx) {
            break;
          }
        }
        pcol=j;
        if (pcol>maxcolors)pcol=maxcolors;
        lcolors[idx]=pcol;
      }
    }
    ierr = PetscLogEventEnd(Mat_Coloring_Local,mc,0,0,0);CHKERRQ(ierr);
    if (mo) {
      /* transfer neighbor colors */
      ierr = PetscLogEventBegin(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(sf,MPIU_INT,lcolors,ocolors);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sf,MPIU_INT,lcolors,ocolors);CHKERRQ(ierr);
      /* check for conflicts -- this is merely checking if any adjacent off-processor rows have the same color and marking the ones that are lower weight locally for changing */
      for (i=0;i<n;i++) {
        ierr = MatGetRow(mo,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
        for (j=0;j<ncols;j++) {
          /* in the case of conflicts, the highest weight one stays and the others go */
          if ((ocolors[cidx[j]] == lcolors[i]) && (owts[cidx[j]] > wts[i])) {
            lcolors[i]=IS_COLORING_MAX;
            nd--;
          }
        }
        ierr = MatRestoreRow(mo,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
      }
      nd_global=0;
    }
    ierr = MPI_Allreduce(&nd,&nd_global,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)mc));CHKERRQ(ierr);
  }
  for (i=0;i<n;i++) {
    colors[i] = (ISColoringValue)lcolors[i];
  }
  ierr = PetscFree(mask);CHKERRQ(ierr);
  ierr = PetscFree(lcolors);CHKERRQ(ierr);
  if (mo) {
    ierr = PetscFree2(ocolors,owts);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "GreedyColoringLocalDistanceTwo_Private"
PETSC_EXTERN PetscErrorCode GreedyColoringLocalDistanceTwo_Private(MatColoring mc,PetscReal *wts,PetscInt *lperm,ISColoringValue *colors)
{
  PetscInt        i,j,k,l,s,e,n,nd,nd_global,n_global,idx,ncols,maxcolors,mcol,mcol_global,nd1cols,*mask,masksize,*d1cols,*bad,*badnext,nbad,badsize,ccol,no,cbad;
  PetscErrorCode  ierr;
  Mat             m=mc->mat;
  Mat_MPIAIJ      *aij = (Mat_MPIAIJ*)m->data;
  Mat             md=NULL,mo=NULL;
  /* Mat             mtd=NULL,mto=NULL; */
  PetscBool       isMPIAIJ,isSEQAIJ;
  ISColoringValue pcol,*ocolors,*badidx;
  const PetscInt  *cidx;
  Vec             wvec,owvec;
  VecScatter      oscatter;
  PetscScalar     *war,*owar;
  PetscReal       *owts,*colorweights;
  PetscBool       *oconf,*conf;

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
  ierr = MatGetVecs(m,&wvec,NULL);CHKERRQ(ierr);
  no=0;
  owvec=NULL;
  if (mo) {
    owvec=aij->lvec;
    oscatter=aij->Mvctx;
    ierr = VecGetLocalSize(owvec,&no);CHKERRQ(ierr);
  }
  maxcolors=IS_COLORING_MAX;
  if (mc->maxcolors) maxcolors=mc->maxcolors;
  masksize=n;
  ierr = PetscMalloc1(n,&d1cols);CHKERRQ(ierr);
  ierr = PetscMalloc1(masksize,&mask);CHKERRQ(ierr);
  for(i=0;i<masksize;i++) {
    mask[i]=-1;
  }
  ierr = PetscMalloc1(n,&conf);CHKERRQ(ierr);
  nbad=0;
  badsize=n;
  ierr = PetscMalloc1(n,&bad);CHKERRQ(ierr);
  ierr = PetscMalloc2(badsize,&badidx,badsize,&badnext);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    colors[i]=IS_COLORING_MAX;
    bad[i]=-1;
  }
  for (i=0;i<badsize;i++) {
    badnext[i]=-1;
  }
  if (mo) {
    ierr = VecGetArray(wvec,&war);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
      war[i]=wts[i];
    }
    ierr = VecRestoreArray(wvec,&war);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
    ierr = VecScatterBegin(oscatter,wvec,owvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(oscatter,wvec,owvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
    ierr = VecGetArray(owvec,&owar);CHKERRQ(ierr);
    ierr = PetscMalloc3(no,&ocolors,no,&owts,no,&oconf);CHKERRQ(ierr);
    for (i=0;i<no;i++) {
      owts[i]=PetscRealPart(owar[i]);
      ocolors[i]=IS_COLORING_MAX;
    }
    ierr = VecRestoreArray(owvec,&owar);CHKERRQ(ierr);
  }
  mcol=0;
  while (nd_global < n_global) {
    nd=n;
    /* assign lowest possible color to each local vertex */
    mcol_global=0;
    ierr = PetscLogEventBegin(Mat_Coloring_Local,mc,0,0,0);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
      idx=n-lperm[i]-1;
      if (colors[idx] == IS_COLORING_MAX) {
        /* entries in bad */
        cbad=bad[idx];
        while (cbad>=0) {
          ccol=badidx[cbad];
          if (ccol>=masksize) {
            PetscInt *newmask;
            ierr = PetscMalloc1(masksize*2,&newmask);CHKERRQ(ierr);
            for(k=0;k<2*masksize;k++) {
              newmask[k]=-1;
            }
            for(k=0;k<masksize;k++) {
              newmask[k]=mask[k];
            }
            ierr = PetscFree(mask);CHKERRQ(ierr);
            mask=newmask;
            masksize*=2;
          }
          mask[ccol]=idx;
          cbad=badnext[cbad];
        }
        /* diagonal distance-one rows */
        nd1cols=0;
        ierr = MatGetRow(md,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
        for (j=0;j<ncols;j++) {
          d1cols[nd1cols] = cidx[j];
          nd1cols++;
          ccol=colors[cidx[j]];
          if (ccol != IS_COLORING_MAX) {
            if (ccol>=masksize) {
              PetscInt *newmask;
              ierr = PetscMalloc1(masksize*2,&newmask);CHKERRQ(ierr);
              for(k=0;k<2*masksize;k++) {
                newmask[k]=-1;
              }
              for(k=0;k<masksize;k++) {
                newmask[k]=mask[k];
              }
              ierr = PetscFree(mask);CHKERRQ(ierr);
              mask=newmask;
              masksize*=2;
            }
            mask[ccol]=idx;
          }
        }
        ierr = MatRestoreRow(md,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
        /* off-diagonal distance-one rows */
        if (mo) {
          ierr = MatGetRow(mo,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
          for (j=0;j<ncols;j++) {
            ccol=colors[cidx[j]];
            if (ccol != IS_COLORING_MAX) {
              if (ccol>=masksize) {
                PetscInt *newmask;
                ierr = PetscMalloc1(masksize*2,&newmask);CHKERRQ(ierr);
                for(k=0;k<2*masksize;k++) {
                  newmask[k]=-1;
                }
                for(k=0;k<masksize;k++) {
                  newmask[k]=mask[k];
                }
                ierr = PetscFree(mask);CHKERRQ(ierr);
                mask=newmask;
                masksize*=2;
              }
              mask[ccol]=idx;
            }
          }
          ierr = MatRestoreRow(mo,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
        }
        /* diagonal distance-two rows */
        for (j=0;j<nd1cols;j++) {
          ierr = MatGetRow (md,d1cols[j],&ncols,&cidx,NULL);CHKERRQ(ierr);
          for (l=0;l<ncols;l++) {
            ccol=colors[cidx[l]];
            if (ccol != IS_COLORING_MAX) {
              if (ccol>=masksize) {
                PetscInt *newmask;
                ierr = PetscMalloc1(masksize*2,&newmask);CHKERRQ(ierr);
                for(k=0;k<2*masksize;k++) {
                  newmask[k]=-1;
                }
                for(k=0;k<masksize;k++) {
                  newmask[k]=mask[k];
                }
                ierr = PetscFree(mask);CHKERRQ(ierr);
                mask=newmask;
                masksize*=2;
              }
              mask[ccol]=idx;
            }
          }
          ierr = MatRestoreRow(md,d1cols[j],&ncols,&cidx,NULL);CHKERRQ(ierr);
        }
        /* off-diagonal distance-two rows */
        if (mo) {
          for (j=0;j<nd1cols;j++) {
            ierr = MatGetRow (mo,d1cols[j],&ncols,&cidx,NULL);CHKERRQ(ierr);
            for (l=0;l<ncols;l++) {
              ccol=ocolors[cidx[l]];
              if (ccol != IS_COLORING_MAX) {
                if (ccol>=masksize) {
                  PetscInt *newmask;
                  ierr = PetscMalloc1(masksize*2,&newmask);CHKERRQ(ierr);
                  for(k=0;k<2*masksize;k++) {
                    newmask[k]=-1;
                  }
                  for(k=0;k<masksize;k++) {
                    newmask[k]=mask[k];
                  }
                  ierr = PetscFree(mask);CHKERRQ(ierr);
                  mask=newmask;
                  masksize*=2;
                }
                mask[ccol]=idx;
              }
            }
            ierr = MatRestoreRow(mo,d1cols[j],&ncols,&cidx,NULL);CHKERRQ(ierr);
          }
        }
        /* assign this one the lowest color possible by seeing if there's a gap in the sequence of sorted neighbor colors */
        pcol=0;
        for (j=0;j<masksize;j++) {
          if (mask[j]!=idx) {
            break;
          }
        }
        pcol=j;
        if (pcol>maxcolors) pcol=maxcolors;
        colors[idx]=pcol;
        if (pcol>mcol) mcol=pcol;
      }
    }
    ierr = PetscLogEventEnd(Mat_Coloring_Local,mc,0,0,0);CHKERRQ(ierr);
    if (mo) {
      /* transfer neighbor colors */
      ierr = VecGetArray(wvec,&war);CHKERRQ(ierr);
      for (i=0;i<n;i++) {
        war[i]=colors[i];
      }
      ierr = VecRestoreArray(wvec,&war);CHKERRQ(ierr);
      ierr = PetscLogEventBegin(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
      ierr = VecScatterBegin(oscatter,wvec,owvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(oscatter,wvec,owvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
      ierr = VecGetArray(owvec,&owar);CHKERRQ(ierr);
      for (i=0;i<no;i++) {
        ocolors[i]=PetscRealPart(owar[i]);
      }
      ierr = VecRestoreArray(owvec,&owar);CHKERRQ(ierr);
      /* find the maximum color assigned locally and allocate a mask */
      ierr = MPI_Allreduce(&mcol,&mcol_global,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)mc));CHKERRQ(ierr);
      ierr = PetscMalloc1(mcol_global+1,&colorweights);CHKERRQ(ierr);
      /* check for conflicts */
      for (i=0;i<n;i++) {
        conf[i]=PETSC_FALSE;
      }
      for (i=0;i<no;i++) {
        oconf[i]=PETSC_FALSE;
      }
      for (i=0;i<n;i++) {
        ierr = MatGetRow(mo,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
        if (ncols > 0) {
          /* fill in the mask */
          for (j=0;j<mcol_global+1;j++) {
            colorweights[j]=0;
          }
          colorweights[colors[i]]=wts[i];
          /* fill in the off-diagonal part of the mask */
          for (j=0;j<ncols;j++) {
            ccol=ocolors[cidx[j]];
            if (ccol < IS_COLORING_MAX) {
              if (colorweights[ccol] < owts[cidx[j]]) {
                colorweights[ccol] = owts[cidx[j]];
              }
            }
          }
          ierr = MatRestoreRow(mo,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
          /* fill in the on-diagonal part of the mask */
          ierr = MatGetRow(md,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
          for (j=0;j<ncols;j++) {
            ccol=colors[cidx[j]];
            if (ccol < IS_COLORING_MAX) {
              if (colorweights[ccol] < wts[cidx[j]]) {
                colorweights[ccol] = wts[cidx[j]];
              }
            }
          }
          ierr = MatRestoreRow(md,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
          /* go back through and set up on and off-diagonal conflict vectors */
          ierr = MatGetRow(md,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
          for (j=0;j<ncols;j++) {
            ccol=colors[cidx[j]];
            if (ccol < IS_COLORING_MAX) {
              if (colorweights[ccol] > wts[cidx[j]]) {
                conf[cidx[j]]=PETSC_TRUE;
              }
            }
          }
          ierr = MatRestoreRow(md,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
          ierr = MatGetRow(mo,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
          for (j=0;j<ncols;j++) {
            ccol=ocolors[cidx[j]];
            if (ccol < IS_COLORING_MAX) {
              if (colorweights[ccol] > owts[cidx[j]]) {
                oconf[cidx[j]]=PETSC_TRUE;
              }
            }
          }
          ierr = MatRestoreRow(mo,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
        } else {
          ierr = MatRestoreRow(mo,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
        }
      }
      nd_global=0;
      ierr = PetscFree(colorweights);CHKERRQ(ierr);
      ierr = VecGetArray(wvec,&war);CHKERRQ(ierr);
      for (i=0;i<n;i++) {
        war[i]=conf[i]?1.:0.;
      }
      ierr = VecRestoreArray(wvec,&war);CHKERRQ(ierr);
      ierr = VecGetArray(owvec,&owar);CHKERRQ(ierr);
      for (i=0;i<no;i++) {
        owar[i]=oconf[i]?1.:0.;
      }
      ierr = VecRestoreArray(owvec,&owar);CHKERRQ(ierr);
      ierr = PetscLogEventBegin(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
      ierr = VecScatterBegin(oscatter,owvec,wvec,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(oscatter,owvec,wvec,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(Mat_Coloring_Comm,mc,0,0,0);CHKERRQ(ierr);
      /* go through and unset local colors that have conflicts */
      ierr = VecGetArray(wvec,&war);CHKERRQ(ierr);
      for (i=0;i<n;i++) {
        if (PetscRealPart(war[i])>0) {
          /* push this color onto the bad stack */
          badidx[nbad]=colors[i];
          badnext[nbad]=bad[i];
          bad[i]=nbad;
          nbad++;
          if (nbad>=badsize) {
            PetscInt *newbadnext;
            ISColoringValue *newbadidx;
            ierr = PetscMalloc2(badsize*2,&newbadidx,badsize*2,&newbadnext);CHKERRQ(ierr);
            for(k=0;k<2*badsize;k++) {
              newbadnext[k]=-1;
            }
            for(k=0;k<badsize;k++) {
              newbadidx[k]=badidx[k];
              newbadnext[k]=badnext[k];
            }
            ierr = PetscFree2(badidx,badnext);CHKERRQ(ierr);
            badidx=newbadidx;
            badnext=newbadnext;
            badsize*=2;
          }
          colors[i] = IS_COLORING_MAX;
          nd--;
        }
      }
      ierr = VecRestoreArray(wvec,&war);CHKERRQ(ierr);
    }
    ierr = MPI_Allreduce(&nd,&nd_global,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)mc));CHKERRQ(ierr);
  }
  ierr = PetscFree(mask);CHKERRQ(ierr);
  ierr = PetscFree(d1cols);CHKERRQ(ierr);
  ierr = PetscFree(bad);CHKERRQ(ierr);
  ierr = PetscFree2(badnext,badidx);CHKERRQ(ierr);
  ierr = PetscFree(conf);CHKERRQ(ierr);
  if (mo) {
    ierr = PetscFree3(ocolors,owts,oconf);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&wvec);CHKERRQ(ierr);
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

   Notes:

   These algorithms proceed in two phases -- local coloring and conflict resolution.  The local coloring
   tentatively colors all vertices at the distance required given what's known of the global coloring.  Then,
   the updated colors are transferred to different processors at distance one.  In the distance one case, each
   vertex with nonlocal neighbors is then checked to see if it conforms, with the vertex being
   marked for recoloring if its lower weight than its same colored neighbor.  In the distance two case,
   each boundary vertex's immediate star is checked for validity of the coloring.  Lower-weight conflict
   vertices are marked, and then the conflicts are gathered back on owning processors.  In both cases
   this is done until each column has received a valid color.

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
