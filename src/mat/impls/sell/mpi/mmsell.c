/*
   Support for the parallel SELL matrix vector multiply
*/
#include <../src/mat/impls/sell/mpi/mpisell.h>
#include <petsc/private/isimpl.h>    /* needed because accesses data structure of ISLocalToGlobalMapping directly */

/*
   Takes the local part of an already assembled MPISELL matrix
   and disassembles it. This is to allow new nonzeros into the matrix
   that require more communication in the matrix vector multiply.
   Thus certain data-structures must be rebuilt.

   Kind of slow! But that's what application programmers get when
   they are sloppy.
*/
PetscErrorCode MatDisAssemble_MPISELL(Mat A)
{
  Mat_MPISELL    *sell=(Mat_MPISELL*)A->data;
  Mat            B=sell->B,Bnew;
  Mat_SeqSELL    *Bsell=(Mat_SeqSELL*)B->data;
  PetscInt       i,j,totalslices,N=A->cmap->N,ec,row;
  PetscBool      isnonzero;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* free stuff related to matrix-vec multiply */
  ierr = VecGetSize(sell->lvec,&ec);CHKERRQ(ierr); /* needed for PetscLogObjectMemory below */
  ierr = VecDestroy(&sell->lvec);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&sell->Mvctx);CHKERRQ(ierr);
  if (sell->colmap) {
#if defined(PETSC_USE_CTABLE)
    ierr = PetscTableDestroy(&sell->colmap);CHKERRQ(ierr);
#else
    ierr = PetscFree(sell->colmap);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)A,-sell->B->cmap->n*sizeof(PetscInt));CHKERRQ(ierr);
#endif
  }

  /* make sure that B is assembled so we can access its values */
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* invent new B and copy stuff over */
  ierr = MatCreate(PETSC_COMM_SELF,&Bnew);CHKERRQ(ierr);
  ierr = MatSetSizes(Bnew,B->rmap->n,N,B->rmap->n,N);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(Bnew,A,A);CHKERRQ(ierr);
  ierr = MatSetType(Bnew,((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = MatSeqSELLSetPreallocation(Bnew,0,Bsell->rlen);CHKERRQ(ierr);
  if (Bsell->nonew >= 0) { /* Inherit insertion error options (if positive). */
    ((Mat_SeqSELL*)Bnew->data)->nonew = Bsell->nonew;
  }

  /*
   Ensure that B's nonzerostate is monotonically increasing.
   Or should this follow the MatSetValues() loop to preserve B's nonzerstate across a MatDisAssemble() call?
   */
  Bnew->nonzerostate = B->nonzerostate;

  totalslices = B->rmap->n/8+((B->rmap->n & 0x07)?1:0); /* floor(n/8) */
  for (i=0; i<totalslices; i++) { /* loop over slices */
    for (j=Bsell->sliidx[i],row=0; j<Bsell->sliidx[i+1]; j++,row=((row+1)&0x07)) {
      isnonzero = (PetscBool)((j-Bsell->sliidx[i])/8 < Bsell->rlen[8*i+row]);
      if (isnonzero) {
        ierr = MatSetValue(Bnew,8*i+row,sell->garray[Bsell->colidx[j]],Bsell->val[j],B->insertmode);CHKERRQ(ierr);
      }
    }
  }

  ierr = PetscFree(sell->garray);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)A,-ec*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)Bnew);CHKERRQ(ierr);

  sell->B          = Bnew;
  A->was_assembled = PETSC_FALSE;
  A->assembled     = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUpMultiply_MPISELL(Mat mat)
{
  Mat_MPISELL    *sell=(Mat_MPISELL*)mat->data;
  Mat_SeqSELL    *B=(Mat_SeqSELL*)(sell->B->data);
  PetscErrorCode ierr;
  PetscInt       i,j,*bcolidx=B->colidx,ec=0,*garray,totalslices;
  IS             from,to;
  Vec            gvec;
  PetscBool      isnonzero;
#if defined(PETSC_USE_CTABLE)
  PetscTable         gid1_lid1;
  PetscTablePosition tpos;
  PetscInt           gid,lid;
#else
  PetscInt N = mat->cmap->N,*indices;
#endif

  PetscFunctionBegin;
  totalslices = sell->B->rmap->n/8+((sell->B->rmap->n & 0x07)?1:0); /* floor(n/8) */

  /* ec counts the number of columns that contain nonzeros */
#if defined(PETSC_USE_CTABLE)
  /* use a table */
  ierr = PetscTableCreate(sell->B->rmap->n,mat->cmap->N+1,&gid1_lid1);CHKERRQ(ierr);
  for (i=0; i<totalslices; i++) { /* loop over slices */
    for (j=B->sliidx[i]; j<B->sliidx[i+1]; j++) {
      isnonzero = (PetscBool)((j-B->sliidx[i])/8 < B->rlen[(i<<3)+(j&0x07)]);
      if (isnonzero) { /* check the mask bit */
        PetscInt data,gid1 = bcolidx[j] + 1;
        ierr = PetscTableFind(gid1_lid1,gid1,&data);CHKERRQ(ierr);
        if (!data) {
          /* one based table */
          ierr = PetscTableAdd(gid1_lid1,gid1,++ec,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }

  /* form array of columns we need */
  ierr = PetscMalloc1(ec,&garray);CHKERRQ(ierr);
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
  for (i=0; i<totalslices; i++) { /* loop over slices */
    for (j=B->sliidx[i]; j<B->sliidx[i+1]; j++) {
      isnonzero = (PetscBool)((j-B->sliidx[i])/8 < B->rlen[(i<<3)+(j&0x07)]);
      if (isnonzero) {
        PetscInt gid1 = bcolidx[j] + 1;
        ierr = PetscTableFind(gid1_lid1,gid1,&lid);CHKERRQ(ierr);
        lid--;
        bcolidx[j] = lid;
      }
    }
  }
  ierr = PetscLayoutDestroy(&sell->B->cmap);CHKERRQ(ierr);
  ierr = PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)sell->B),ec,ec,1,&sell->B->cmap);CHKERRQ(ierr);
  ierr = PetscTableDestroy(&gid1_lid1);CHKERRQ(ierr);
#else
  /* Make an array as long as the number of columns */
  ierr = PetscCalloc1(N,&indices);CHKERRQ(ierr);
  /* mark those columns that are in sell->B */
  for (i=0; i<totalslices; i++) { /* loop over slices */
    for (j=B->sliidx[i]; j<B->sliidx[i+1]; j++) {
      isnonzero = (PetscBool)((j-B->sliidx[i])/8 < B->rlen[(i<<3)+(j&0x07)]);
      if (isnonzero) {
        if (!indices[bcolidx[j]]) ec++;
        indices[bcolidx[j]] = 1;
      }
    }
  }

  /* form array of columns we need */
  ierr = PetscMalloc1(ec,&garray);CHKERRQ(ierr);
  ec   = 0;
  for (i=0; i<N; i++) {
    if (indices[i]) garray[ec++] = i;
  }

  /* make indices now point into garray */
  for (i=0; i<ec; i++) {
    indices[garray[i]] = i;
  }

  /* compact out the extra columns in B */
  for (i=0; i<totalslices; i++) { /* loop over slices */
    for (j=B->sliidx[i]; j<B->sliidx[i+1]; j++) {
      isnonzero = (PetscBool)((j-B->sliidx[i])/8 < B->rlen[(i<<3)+(j&0x07)]);
      if (isnonzero) bcolidx[j] = indices[bcolidx[j]];
    }
  }
  ierr = PetscLayoutDestroy(&sell->B->cmap);CHKERRQ(ierr);
  ierr = PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)sell->B),ec,ec,1,&sell->B->cmap);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);
#endif
  /* create local vector that is used to scatter into */
  ierr = VecCreateSeq(PETSC_COMM_SELF,ec,&sell->lvec);CHKERRQ(ierr);
  /* create two temporary Index sets for build scatter gather */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,ec,garray,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,ec,0,1,&to);CHKERRQ(ierr);

  /* create temporary global vector to generate scatter context */
  /* This does not allocate the array's memory so is efficient */
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)mat),1,mat->cmap->n,mat->cmap->N,NULL,&gvec);CHKERRQ(ierr);

  /* generate the scatter context */
  ierr = VecScatterCreate(gvec,from,sell->lvec,to,&sell->Mvctx);CHKERRQ(ierr);
  ierr = VecScatterViewFromOptions(sell->Mvctx,(PetscObject)mat,"-matmult_vecscatter_view");CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)sell->Mvctx);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)sell->lvec);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)from);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)to);CHKERRQ(ierr);

  sell->garray = garray;

  ierr = PetscLogObjectMemory((PetscObject)mat,ec*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = VecDestroy(&gvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*      ugly stuff added for Glenn someday we should fix this up */
static PetscInt *auglyrmapd = NULL,*auglyrmapo = NULL; /* mapping from the local ordering to the "diagonal" and "off-diagonal" parts of the local matrix */
static Vec auglydd          = NULL,auglyoo     = NULL; /* work vectors used to scale the two parts of the local matrix */

PetscErrorCode MatMPISELLDiagonalScaleLocalSetUp(Mat inA,Vec scale)
{
  Mat_MPISELL    *ina=(Mat_MPISELL*)inA->data; /*access private part of matrix */
  PetscErrorCode ierr;
  PetscInt       i,n,nt,cstart,cend,no,*garray=ina->garray,*lindices;
  PetscInt       *r_rmapd,*r_rmapo;

  PetscFunctionBegin;
  ierr = MatGetOwnershipRange(inA,&cstart,&cend);CHKERRQ(ierr);
  ierr = MatGetSize(ina->A,NULL,&n);CHKERRQ(ierr);
  ierr = PetscCalloc1(inA->rmap->mapping->n+1,&r_rmapd);CHKERRQ(ierr);
  nt   = 0;
  for (i=0; i<inA->rmap->mapping->n; i++) {
    if (inA->rmap->mapping->indices[i] >= cstart && inA->rmap->mapping->indices[i] < cend) {
      nt++;
      r_rmapd[i] = inA->rmap->mapping->indices[i] + 1;
    }
  }
  if (nt != n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Hmm nt %D n %D",nt,n);
  ierr = PetscMalloc1(n+1,&auglyrmapd);CHKERRQ(ierr);
  for (i=0; i<inA->rmap->mapping->n; i++) {
    if (r_rmapd[i]) {
      auglyrmapd[(r_rmapd[i]-1)-cstart] = i;
    }
  }
  ierr = PetscFree(r_rmapd);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&auglydd);CHKERRQ(ierr);
  ierr = PetscCalloc1(inA->cmap->N+1,&lindices);CHKERRQ(ierr);
  for (i=0; i<ina->B->cmap->n; i++) {
    lindices[garray[i]] = i+1;
  }
  no   = inA->rmap->mapping->n - nt;
  ierr = PetscCalloc1(inA->rmap->mapping->n+1,&r_rmapo);CHKERRQ(ierr);
  nt   = 0;
  for (i=0; i<inA->rmap->mapping->n; i++) {
    if (lindices[inA->rmap->mapping->indices[i]]) {
      nt++;
      r_rmapo[i] = lindices[inA->rmap->mapping->indices[i]];
    }
  }
  if (nt > no) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Hmm nt %D no %D",nt,n);
  ierr = PetscFree(lindices);CHKERRQ(ierr);
  ierr = PetscMalloc1(nt+1,&auglyrmapo);CHKERRQ(ierr);
  for (i=0; i<inA->rmap->mapping->n; i++) {
    if (r_rmapo[i]) {
      auglyrmapo[(r_rmapo[i]-1)] = i;
    }
  }
  ierr = PetscFree(r_rmapo);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,nt,&auglyoo);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDiagonalScaleLocal_MPISELL(Mat A,Vec scale)
{
  Mat_MPISELL       *a=(Mat_MPISELL*)A->data; /*access private part of matrix */
  PetscErrorCode    ierr;
  PetscInt          n,i;
  PetscScalar       *d,*o;
  const PetscScalar *s;

  PetscFunctionBegin;
  if (!auglyrmapd) {
    ierr = MatMPISELLDiagonalScaleLocalSetUp(A,scale);CHKERRQ(ierr);
  }
  ierr = VecGetArrayRead(scale,&s);CHKERRQ(ierr);
  ierr = VecGetLocalSize(auglydd,&n);CHKERRQ(ierr);
  ierr = VecGetArray(auglydd,&d);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    d[i] = s[auglyrmapd[i]]; /* copy "diagonal" (true local) portion of scale into dd vector */
  }
  ierr = VecRestoreArray(auglydd,&d);CHKERRQ(ierr);
  /* column scale "diagonal" portion of local matrix */
  ierr = MatDiagonalScale(a->A,NULL,auglydd);CHKERRQ(ierr);
  ierr = VecGetLocalSize(auglyoo,&n);CHKERRQ(ierr);
  ierr = VecGetArray(auglyoo,&o);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    o[i] = s[auglyrmapo[i]]; /* copy "off-diagonal" portion of scale into oo vector */
  }
  ierr = VecRestoreArrayRead(scale,&s);CHKERRQ(ierr);
  ierr = VecRestoreArray(auglyoo,&o);CHKERRQ(ierr);
  /* column scale "off-diagonal" portion of local matrix */
  ierr = MatDiagonalScale(a->B,NULL,auglyoo);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
