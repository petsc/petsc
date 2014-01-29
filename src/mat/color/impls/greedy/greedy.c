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
  PetscInt        i,j,s,e,n,nd,nd_global,n_global,idx,ncols,maxcols,maxcolors;
  PetscErrorCode  ierr;
  Mat             m=mc->mat;
  Mat_MPIAIJ      *aij = (Mat_MPIAIJ*)m->data;
  Mat             md=NULL,mo=NULL;
  PetscBool       isMPIAIJ,isSEQAIJ;
  /* MC_Greedy       *gr=(MC_Greedy*)mc->data; */
  ISColoringValue *mask;
  const PetscInt  *cidx;
  Vec             wtvec,owtvec,colvec,ocolvec,confvec,oconfvec;
  VecScatter      oscatter;
  PetscScalar     *wtar,*owtar,*colar,*ocolar;

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
    mo=NULL;
  } else {
    SETERRQ(PetscObjectComm((PetscObject)mc),PETSC_ERR_ARG_WRONG,"Matrix must be AIJ for greedy coloring");
  }
  /* create the vectors and communication structures if necessary */
  ierr = MatGetVecs(m,&wtvec,NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(wtvec,&colvec);CHKERRQ(ierr);
  ierr = VecDuplicate(wtvec,&confvec);CHKERRQ(ierr);
  owtvec=NULL;
  ocolvec=NULL;
  oconfvec=NULL;
  oscatter=NULL;
  if (mo) {
    ierr = VecDuplicate(aij->lvec,&owtvec);CHKERRQ(ierr);
    ierr = VecDuplicate(aij->lvec,&ocolvec);CHKERRQ(ierr);
    ierr = VecDuplicate(aij->lvec,&oconfvec);CHKERRQ(ierr);
    oscatter=aij->Mvctx;
  }
  /* get the maximum possible number of colors and allocate the mask */
  for (i=s;i<e;i++) {
    ierr = MatGetRow(m,i,&ncols,NULL,NULL);CHKERRQ(ierr);
    if (ncols>maxcols) maxcols=ncols;
    ierr = MatRestoreRow(m,i,&ncols,NULL,NULL);CHKERRQ(ierr);
  }
  maxcols++;
if ((maxcols > mc->maxcolors) && mc->maxcolors) maxcols=mc->maxcolors;
  /* setup the initial colors for on and off processors in the vectors, for ease of use */
  ierr = VecSet(colvec,(PetscScalar)IS_COLORING_MAX);CHKERRQ(ierr);
  if (ocolvec) {ierr = VecSet(ocolvec,(PetscScalar)IS_COLORING_MAX);CHKERRQ(ierr);}

  maxcolors=0;
  ierr = MPI_Allreduce(&maxcols,&maxcolors,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)mc));CHKERRQ(ierr);
  ierr = PetscMalloc1(maxcolors,&mask);CHKERRQ(ierr);

  /* transfer neighbor weights */
  ierr = VecGetArray(wtvec,&wtar);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    wtar[i]=wts[i];
  }
  ierr = VecRestoreArray(wtvec,&wtar);CHKERRQ(ierr);
  if (mo) {
    ierr = VecScatterBegin(oscatter,wtvec,owtvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(oscatter,wtvec,owtvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }

  while (nd_global < n_global) {
    nd=n;
    ierr = VecGetArray(colvec,&colar);CHKERRQ(ierr);
    if (mo) {
      ierr = VecGetArray(ocolvec,&ocolar);CHKERRQ(ierr);
    }
    /* assign lowest possible color to each local vertex */
    for (i=0;i<n;i++) {
      idx=n-lperm[i]-1;
      if (colar[idx] == IS_COLORING_MAX) {
        for (j=0;j<maxcolors;j++) mask[j]=0;
        ierr = MatGetRow(md,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
        for (j=0;j<ncols;j++) {
          if ((PetscInt)colar[cidx[j]] != IS_COLORING_MAX) {
            mask[(PetscInt)colar[cidx[j]]] = 1;
          }
        }
        ierr = MatRestoreRow(md,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
        if (mo) {
          ierr = MatGetRow(mo,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
          for (j=0;j<ncols;j++) {
            if ((PetscInt)ocolar[cidx[j]] != IS_COLORING_MAX) {
              mask[(PetscInt)ocolar[cidx[j]]] = 1;
            }
          }
          ierr = MatRestoreRow(mo,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
        }
        /* assign this one the lowest color possible */
        for (j=0;j<maxcolors;j++) {
          if (!mask[j]) {
            break;
          }
        }
        colar[idx]=j;
      }
    }
    ierr = VecRestoreArray(colvec,&colar);CHKERRQ(ierr);
    if (mo) {
      ierr = VecRestoreArray(ocolvec,&ocolar);CHKERRQ(ierr);
    }
    if (mo) {
      /* transfer neighbor colors */
      ierr = VecScatterBegin(oscatter,colvec,ocolvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(oscatter,colvec,ocolvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      /* check for conflicts -- this is merely checking if any adjacent off-processor rows have the same color and marking the ones that are lower weight locally for changing */
      ierr = VecGetArray(wtvec,&wtar);CHKERRQ(ierr);
      ierr = VecGetArray(owtvec,&owtar);CHKERRQ(ierr);
      ierr = VecGetArray(colvec,&colar);CHKERRQ(ierr);
      ierr = VecGetArray(ocolvec,&ocolar);CHKERRQ(ierr);
      for (i=0;i<n;i++) {
        ierr = MatGetRow(mo,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
        for (j=0;j<ncols;j++) {
          /* in the case of conflicts, the highest weight one stays and the others go */
          if ((ocolar[cidx[j]] == colar[i]) && (owtar[cidx[j]] > wtar[i])) {
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
    colors[i]=(PetscInt)colar[i];
  }
  ierr = VecRestoreArray(colvec,&colar);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GreedyColoringLocalDistanceTwo_Private"
PETSC_EXTERN PetscErrorCode GreedyColoringLocalDistanceTwo_Private(MatColoring mc,ISColoring *colors)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringApply_Greedy"
PETSC_EXTERN PetscErrorCode MatColoringApply_Greedy(MatColoring mc,ISColoring *iscoloring)
{
  /* MC_Greedy       *gr=(MC_Greedy*)mc->data; */
  PetscErrorCode  ierr;
  ISColoringValue curcolor;
  PetscInt        finalcolor,finalcolor_global;
  ISColoringValue *colors;
  PetscInt        nadded,nadded_total,ncolstotal,ncols;
  PetscReal       *wts;
  PetscInt        i,*lperm;

  PetscFunctionBegin;
  nadded=1;
  nadded_total=0;
  ierr = MatGetSize(mc->mat,NULL,&ncolstotal);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mc->mat,NULL,&ncols);CHKERRQ(ierr);
  ierr = GreedyCreateWeights_Private(mc,&wts,&lperm);CHKERRQ(ierr);
  ierr = PetscMalloc1(ncols,&colors);CHKERRQ(ierr);
  curcolor=0;
  if (mc->dist == 1) {
    ierr = GreedyColoringLocalDistanceOne_Private(mc,wts,lperm,colors);CHKERRQ(ierr);
  } else if (mc->dist == 2) {
  } else {
    SETERRQ(PetscObjectComm((PetscObject)mc),PETSC_ERR_ARG_OUTOFRANGE,"Only distance 1 and distance 2 supported by MatColoringGreedy");
  }
  finalcolor=0;
  for (i=0;i<ncols;i++) {
    if (colors[i] > finalcolor) finalcolor=colors[i];
  }
  finalcolor_global=0;
  ierr = MPI_Allreduce(&finalcolor,&finalcolor_global,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)mc));CHKERRQ(ierr);
  ierr = ISColoringCreate(PetscObjectComm((PetscObject)mc),finalcolor_global+1,ncols,colors,iscoloring);CHKERRQ(ierr);
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
