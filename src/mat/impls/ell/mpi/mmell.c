
/*
   Support for the parallel ELL matrix vector multiply
*/
#include <../src/mat/impls/ell/mpi/mpiell.h>
#include <petsc/private/isimpl.h>    /* needed because accesses data structure of ISLocalToGlobalMapping directly */

#undef __FUNCT__
#define __FUNCT__ "MatSetUpMultiply_MPIELL"
PetscErrorCode MatSetUpMultiply_MPIELL(Mat mat)
{
  Mat_MPIELL     *ell = (Mat_MPIELL*)mat->data;
  Mat_SeqELL     *B   = (Mat_SeqELL*)(ell->B->data);
  PetscErrorCode ierr;
  PetscInt       i,*bcolidx = B->colidx,ec = 0,*garray;
  IS             from,to;
  Vec            gvec;
#if defined(PETSC_USE_CTABLE)
  PetscTable         gid1_lid1;
  PetscTablePosition tpos;
  PetscInt           gid,lid;
#else
  PetscInt N = mat->cmap->N,*indices;
#endif

  PetscFunctionBegin;
  /* ec counts the number of columns that contain nonzeros */
#if defined(PETSC_USE_CTABLE)
  /* use a table */
  ierr = PetscTableCreate(ell->B->rmap->n,mat->cmap->N+1,&gid1_lid1);CHKERRQ(ierr);
  for (i=0; i<B->sliidx[ell->B->rmap->n/8]; i++) { /* loop over all elements */
    if (B->bt[i>>3] & (char)(1<<(i&0x07))) { /* check the mask bit */
      PetscInt data,gid1 = bcolidx[i] + 1;
      ierr = PetscTableFind(gid1_lid1,gid1,&data);CHKERRQ(ierr);
      if (!data) {
        /* one based table */
        ierr = PetscTableAdd(gid1_lid1,gid1,++ec,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }

  /* form array of columns we need */
  ierr = PetscMalloc1(ec+1,&garray);CHKERRQ(ierr);
  ierr = PetscTableGetHeadPosition(gid1_lid1,&tpos);CHKERRQ(ierr);
  while (tpos) {
    ierr = PetscTableGetNext(gid1_lid1,&tpos,&gid,&lid);CHKERRQ(ierr);
    gid--;
    lid--;
    garray[lid] = gid;
  }
  ierr = PetscSortInt(ec,garray);CHKERRQ(ierr); /* sort, and rebuild */
  ierr = PetscTableRemoveAll(gid1_lid1);CHKERRQ(ierr);
  for (i=0; i<ec; i++) {
    ierr = PetscTableAdd(gid1_lid1,garray[i]+1,i+1,INSERT_VALUES);CHKERRQ(ierr);
  }
  /* compact out the extra columns in B */
  for (i=0; i<B->sliidx[ell->B->rmap->n/8]; i++) {
    if (B->bt[i>>3] & (char)(1<<(i&0x07))) {
      PetscInt gid1 = bcolidx[i] + 1;
      ierr = PetscTableFind(gid1_lid1,gid1,&lid);CHKERRQ(ierr);
      lid--;
      bcolidx[i] = lid;
    }
  }
  ell->B->cmap->n = ell->B->cmap->N = ec;
  ell->B->cmap->bs = 1;

  ierr = PetscLayoutSetUp((ell->B->cmap));CHKERRQ(ierr);
  ierr = PetscTableDestroy(&gid1_lid1);CHKERRQ(ierr);
#else
  /* Make an array as long as the number of columns */
  /* mark those columns that are in ell->B */
  ierr = PetscCalloc1(N+1,&indices);CHKERRQ(ierr);

  for (i=0; i<B->sliidx[ell->B->rmap->n/8]; i++) {
    if ((B->bt[i>>3] & (char)(1<<(i&0x07))) && !indices[bcolidx[i]]) ec++;
    indices[bcolidx[i]] = 1;
  }

  /* form array of columns we need */
  ierr = PetscMalloc1(ec+1,&garray);CHKERRQ(ierr);
  ec   = 0;
  for (i=0; i<N; i++) {
    if (indices[i]) garray[ec++] = i;
  }

  /* make indices now point into garray */
  for (i=0; i<ec; i++) {
    indices[garray[i]] = i;
  }

  /* compact out the extra columns in B */
  for (i=0; i<B->sliidx[ell->B->rmap->n/8]; i++) {
    if (B->bt[i>>3] & (char)(1<<(i&0x07))) bcolidx[i] = indices[bcolidx[i]];
  }
  ell->B->cmap->n = ell->B->cmap->N = ec; /* number of columns that are not all zeros */
  ell->B->cmap->bs = 1;

  ierr = PetscLayoutSetUp((ell->B->cmap));CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);
#endif
  /* create local vector that is used to scatter into */
  ierr = VecCreateSeq(PETSC_COMM_SELF,ec,&ell->lvec);CHKERRQ(ierr);

  /* create two temporary Index sets for build scatter gather */
  ierr = ISCreateGeneral(((PetscObject)mat)->comm,ec,garray,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);

  ierr = ISCreateStride(PETSC_COMM_SELF,ec,0,1,&to);CHKERRQ(ierr);

  /* create temporary global vector to generate scatter context */
  /* This does not allocate the array's memory so is efficient */
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)mat),1,mat->cmap->n,mat->cmap->N,NULL,&gvec);CHKERRQ(ierr);

  /* generate the scatter context */
  ierr = VecScatterCreate(gvec,from,ell->lvec,to,&ell->Mvctx);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)ell->Mvctx);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)ell->lvec);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)from);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)to);CHKERRQ(ierr);

  ell->garray = garray;

  ierr = PetscLogObjectMemory((PetscObject)mat,(ec+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = VecDestroy(&gvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
