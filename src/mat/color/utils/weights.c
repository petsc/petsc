#include <petsc-private/matimpl.h>      /*I "petscmat.h"  I*/
#include <../src/mat/impls/aij/seq/aij.h>

#undef __FUNCT__
#define __FUNCT__ "MatColoringCreateLexicalWeights"
PetscErrorCode MatColoringCreateLexicalWeights(MatColoring mc,PetscReal *weights)
{
  PetscErrorCode ierr;
  PetscInt       i,s,e;
  Mat            G=mc->mat;

  PetscFunctionBegin;
  ierr = MatGetOwnershipRange(G,&s,&e);CHKERRQ(ierr);
  for (i=s;i<e;i++) {
    weights[i-s] = i;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringCreateRandomWeights"
PetscErrorCode MatColoringCreateRandomWeights(MatColoring mc,PetscReal *weights)
{
  PetscErrorCode ierr;
  PetscInt       i,s,e;
  PetscRandom    rand;
  PetscReal      r;
  Mat            G = mc->mat;

  PetscFunctionBegin;
  /* each weight should be the degree plus a random perturbation */
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)mc),&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(G,&s,&e);CHKERRQ(ierr);
  for (i=s;i<e;i++) {
    ierr = PetscRandomGetValueReal(rand,&r);CHKERRQ(ierr);
    weights[i-s] = PetscAbsReal(r);
  }
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringGetDegrees"
PetscErrorCode MatColoringGetDegrees(Mat G,PetscInt distance,PetscInt *degrees)
{
  PetscInt       j,i,s,e,n,ln,lm,degree,bidx,idx,dist;
  Mat            lG,*lGs;
  IS             ris;
  PetscErrorCode ierr;
  PetscInt       *seen;
  const PetscInt *gidx;
  PetscInt       *idxbuf;
  PetscInt       *distbuf;
  PetscInt       ncols;
  PetscInt       *coltorow;
  const PetscInt *cols;
  PetscBool      isSEQAIJ;
  Mat_SeqAIJ     *aij;
  PetscInt       *Gi,*Gj;

  PetscFunctionBegin;
  ierr = MatGetOwnershipRange(G,&s,&e);CHKERRQ(ierr);
  n=e-s;
  ierr = ISCreateStride(PetscObjectComm((PetscObject)G),n,s,1,&ris);CHKERRQ(ierr);
  ierr = MatIncreaseOverlap(G,1,&ris,distance);CHKERRQ(ierr);
  ierr = ISSort(ris);CHKERRQ(ierr);
  ierr = MatGetSubMatrices(G,1,&ris,&ris,MAT_INITIAL_MATRIX,&lGs);CHKERRQ(ierr);
  lG = lGs[0];
  ierr = PetscObjectTypeCompare((PetscObject)lG,MATSEQAIJ,&isSEQAIJ);CHKERRQ(ierr);
  if (!isSEQAIJ) {
    SETERRQ(PetscObjectComm((PetscObject)G),PETSC_ERR_ARG_WRONGSTATE,"MatColoringDegrees requires an MPI/SEQAIJ Matrix");
  }
  ierr = MatGetSize(lG,&ln,&lm);CHKERRQ(ierr);
  aij = (Mat_SeqAIJ*)lG->data;
  Gi = aij->i;
  Gj = aij->j;
  ierr = PetscMalloc4(lm,&seen,lm,&idxbuf,lm,&distbuf,lm,&coltorow);CHKERRQ(ierr);
  for (i=0;i<ln;i++) {
    seen[i]=-1;
  }
  ierr = ISGetIndices(ris,&gidx);CHKERRQ(ierr);
  for (i=0;i<ln;i++) {
    if (gidx[i] >= e || gidx[i] < s) continue;
    bidx=-1;
    ncols = Gi[i+1]-Gi[i];
    cols = &(Gj[Gi[i]]);
    degree = 0;
    /* place the distance-one neighbors on the queue */
    for (j=0;j<ncols;j++) {
      bidx++;
      seen[cols[j]] = i;
      distbuf[bidx] = 1;
      idxbuf[bidx] = cols[j];
    }
    while (bidx >= 0) {
      /* pop */
      idx = idxbuf[bidx];
      dist = distbuf[bidx];
      bidx--;
      degree++;
      if (dist < distance) {
        ncols = Gi[idx+1]-Gi[idx];
        cols = &(Gj[Gi[idx]]);
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
    degrees[gidx[i]-s] = degree;
  }
  ierr = ISRestoreIndices(ris,&gidx);CHKERRQ(ierr);
  ierr = ISDestroy(&ris);CHKERRQ(ierr);
  ierr = PetscFree4(seen,idxbuf,distbuf,coltorow);CHKERRQ(ierr);
  ierr = MatDestroyMatrices(1,&lGs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringCreateLargestFirstWeights"
PetscErrorCode MatColoringCreateLargestFirstWeights(MatColoring mc,PetscReal *weights)
{
  PetscErrorCode ierr;
  PetscInt       i,s,e,n,ncols;
  PetscRandom    rand;
  PetscReal      r;
  PetscInt       *degrees;
  Mat            G = mc->mat;

  PetscFunctionBegin;
  /* each weight should be the degree plus a random perturbation */
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)mc),&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(G,&s,&e);CHKERRQ(ierr);
  n=e-s;
  ierr = PetscMalloc1(n,&degrees);CHKERRQ(ierr);
  ierr = MatColoringGetDegrees(G,mc->dist,degrees);CHKERRQ(ierr);
  for (i=s;i<e;i++) {
    ierr = MatGetRow(G,i,&ncols,NULL,NULL);CHKERRQ(ierr);
    ierr = PetscRandomGetValueReal(rand,&r);CHKERRQ(ierr);
    weights[i-s] = ncols + PetscAbsReal(r);
    ierr = MatRestoreRow(G,i,&ncols,NULL,NULL);CHKERRQ(ierr);
  }
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = PetscFree(degrees);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringCreateSmallestLastWeights"
PetscErrorCode MatColoringCreateSmallestLastWeights(MatColoring mc,PetscReal *weights)
{
  PetscInt       *degrees,*degb,*llprev,*llnext;
  PetscInt       j,i,s,e,n,nin,ln,lm,degree,maxdegree=0,bidx,idx,dist,distance=mc->dist;
  Mat            lG,*lGs;
  IS             ris;
  PetscErrorCode ierr;
  PetscInt       *seen;
  const PetscInt *gidx;
  PetscInt       *idxbuf;
  PetscInt       *distbuf;
  PetscInt       ncols,nxt,prv,cur;
  PetscInt       *coltorow;
  const PetscInt *cols;
  PetscBool      isSEQAIJ;
  Mat_SeqAIJ     *aij;
  PetscInt       *Gi,*Gj;
  Mat            G = mc->mat;
  PetscReal      *lweights,r;
  PetscRandom    rand;

  PetscFunctionBegin;
  ierr = MatGetOwnershipRange(G,&s,&e);CHKERRQ(ierr);
  n=e-s;
  ierr = ISCreateStride(PetscObjectComm((PetscObject)G),n,s,1,&ris);CHKERRQ(ierr);
  ierr = MatIncreaseOverlap(G,1,&ris,distance);CHKERRQ(ierr);
  ierr = ISSort(ris);CHKERRQ(ierr);
  ierr = MatGetSubMatrices(G,1,&ris,&ris,MAT_INITIAL_MATRIX,&lGs);CHKERRQ(ierr);
  lG = lGs[0];
  ierr = PetscObjectTypeCompare((PetscObject)lG,MATSEQAIJ,&isSEQAIJ);CHKERRQ(ierr);
  if (!isSEQAIJ) {
    SETERRQ(PetscObjectComm((PetscObject)G),PETSC_ERR_ARG_WRONGSTATE,"MatColoringDegrees requires an MPI/SEQAIJ Matrix");
  }
  ierr = MatGetSize(lG,&ln,&lm);CHKERRQ(ierr);
  aij = (Mat_SeqAIJ*)lG->data;
  Gi = aij->i;
  Gj = aij->j;
  ierr = PetscMalloc4(lm,&seen,lm,&idxbuf,lm,&distbuf,lm,&coltorow);CHKERRQ(ierr);
  ierr = PetscMalloc1(lm,&degrees);CHKERRQ(ierr);
  ierr = PetscMalloc1(lm,&lweights);CHKERRQ(ierr);
  for (i=0;i<ln;i++) {
    seen[i]=-1;
    lweights[i] = 1.;
  }
  ierr = ISGetIndices(ris,&gidx);CHKERRQ(ierr);
  for (i=0;i<ln;i++) {
    bidx=-1;
    ncols = Gi[i+1]-Gi[i];
    cols = &(Gj[Gi[i]]);
    degree = 0;
    /* place the distance-one neighbors on the queue */
    for (j=0;j<ncols;j++) {
      bidx++;
      seen[cols[j]] = i;
      distbuf[bidx] = 1;
      idxbuf[bidx] = cols[j];
    }
    while (bidx >= 0) {
      /* pop */
      idx = idxbuf[bidx];
      dist = distbuf[bidx];
      bidx--;
      degree++;
      if (dist < distance) {
        ncols = Gi[idx+1]-Gi[idx];
        cols = &(Gj[Gi[idx]]);
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
    degrees[i] = degree;
    if (degree > maxdegree) maxdegree = degree;
  }
  /* bucket by degree */
  ierr = PetscMalloc1(maxdegree+1,&degb);CHKERRQ(ierr);
  ierr = PetscMalloc2(ln,&llnext,ln,&llprev);CHKERRQ(ierr);
  for (i=0;i<maxdegree+1;i++) {
    degb[i] = -1;
  }
  for (i=0;i<ln;i++) {
    llnext[i] = -1;
    llprev[i] = -1;
    seen[i] = -1;
  }
  for (i=0;i<ln;i++) {
    llnext[i] = degb[degrees[i]];
    if (degb[degrees[i]] > 0) llprev[degb[degrees[i]]] = i;
    degb[degrees[i]] = i;
  }
  /* remove the lowest degree one */
  i=0;
  nin=0;
  while (i != maxdegree+1) {
    for (i=0;i<maxdegree+1; i++) {
      if (degb[i] > 0) {
        cur = degb[i];
        nin++;
        degrees[cur] = -1;
        degb[i] = llnext[cur];
        bidx=-1;
        ncols = Gi[cur+1]-Gi[cur];
        cols = &(Gj[Gi[cur]]);
        /* place the distance-one neighbors on the queue */
        for (j=0;j<ncols;j++) {
          if (cols[j] != cur) {
            bidx++;
            seen[cols[j]] = i;
            distbuf[bidx] = 1;
            idxbuf[bidx] = cols[j];
          }
        }
        while (bidx >= 0) {
          /* pop */
          idx = idxbuf[bidx];
          nxt=llnext[idx];
          prv=llprev[idx];
          dist = distbuf[bidx];
          bidx--;
          if (idx != cur && degrees[idx] > 0) {
            /* change up the degree of the neighbors still in the graph */
            if (lweights[idx] <= lweights[cur]) lweights[idx] = lweights[cur]+1;
            if (nxt > 0) {
              llprev[nxt] = prv;
            }
            if (prv > 0) {
              llnext[prv] = nxt;
            } else {
              degb[degrees[idx]] = nxt;
            }
            degrees[idx]--;
            llnext[idx] = llnext[degb[degrees[idx]]];
            degb[degrees[idx]] = idx;
            if (dist < distance) {
              ncols = Gi[idx+1]-Gi[idx];
              cols = &(Gj[Gi[idx]]);
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
        break;
      }
    }
  }
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)mc),&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  for (i=0;i<lm;i++) {
    if (gidx[i] >= s && gidx[i] < e) {
    ierr = PetscRandomGetValueReal(rand,&r);CHKERRQ(ierr);
      weights[gidx[i]-s] = lweights[i] + r;
    }
  }
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = PetscFree(degb);CHKERRQ(ierr);
  ierr = PetscFree2(llnext,llprev);CHKERRQ(ierr);
  ierr = PetscFree(degrees);CHKERRQ(ierr);
  ierr = PetscFree(lweights);CHKERRQ(ierr);
  ierr = ISRestoreIndices(ris,&gidx);CHKERRQ(ierr);
  ierr = ISDestroy(&ris);CHKERRQ(ierr);
  ierr = PetscFree4(seen,idxbuf,distbuf,coltorow);CHKERRQ(ierr);
  ierr = MatDestroyMatrices(1,&lGs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringCreateWeights"
PetscErrorCode MatColoringCreateWeights(MatColoring mc,PetscReal **weights,PetscInt **lperm)
{
  PetscErrorCode ierr;
  PetscInt       i,s,e,n;
  PetscReal      *wts;

  PetscFunctionBegin;
  /* create weights of the specified type */
  ierr = MatGetOwnershipRange(mc->mat,&s,&e);CHKERRQ(ierr);
  n=e-s;
  ierr = PetscMalloc1(n,&wts);CHKERRQ(ierr);
  switch(mc->weight_type) {
  case MAT_COLORING_WEIGHT_RANDOM:
    ierr = MatColoringCreateRandomWeights(mc,wts);CHKERRQ(ierr);
    break;
  case MAT_COLORING_WEIGHT_LEXICAL:
    ierr = MatColoringCreateLexicalWeights(mc,wts);CHKERRQ(ierr);
    break;
  case MAT_COLORING_WEIGHT_LF:
    ierr = MatColoringCreateLargestFirstWeights(mc,wts);CHKERRQ(ierr);
    break;
  case MAT_COLORING_WEIGHT_SL:
    ierr = MatColoringCreateSmallestLastWeights(mc,wts);CHKERRQ(ierr);
    break;
  }
  if (lperm) {
    ierr = PetscMalloc1(n,lperm);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
      (*lperm)[i] = n-1-i;
    }
    ierr = PetscSortRealWithPermutation(n,wts,*lperm);CHKERRQ(ierr);
  }
  if (weights) *weights = wts;
  PetscFunctionReturn(0);
}
