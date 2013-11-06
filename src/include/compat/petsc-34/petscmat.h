#define MatNullSpaceRemove(nsp,vec) MatNullSpaceRemove(nsp,vec,NULL)

static
#undef  __FUNCT__
#define __FUNCT__ "MatGetBlockSize_Compat"
PetscErrorCode MatGetBlockSize_Compat(Mat A,PetscInt *bs)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidIntPointer(bs,2);
  ierr = PetscLayoutGetBlockSize(A->rmap,bs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define MatGetBlockSize MatGetBlockSize_Compat

static
#undef  __FUNCT__
#define __FUNCT__ "MatGetBlockSizes_Compat"
PetscErrorCode MatGetBlockSizes_Compat(Mat A,PetscInt *rbs, PetscInt *cbs)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidIntPointer(rbs,2);
  PetscValidIntPointer(cbs,3);
  ierr = PetscLayoutGetBlockSize(A->rmap,rbs);CHKERRQ(ierr);
  ierr = PetscLayoutGetBlockSize(A->cmap,cbs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define MatGetBlockSizes MatGetBlockSizes_Compat

static
#undef  __FUNCT__
#define __FUNCT__ "MatSeqSBAIJSetPreallocationCSR_SeqSBAIJ"
PetscErrorCode MatSeqSBAIJSetPreallocationCSR_SeqSBAIJ(Mat B,PetscInt bs,const PetscInt ii[],const PetscInt jj[], const PetscScalar V[])
{
  PetscInt       i,j,m,nz,nz_max=0,*nnz;
  PetscScalar    *values=0;
  PetscBool      roworiented = PETSC_FALSE;/*((Mat_SeqSBAIJ*)B->data)->roworiented;*/
  PetscErrorCode (*MatSetValuesBlocked_SeqSBAIJ)(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode) = B->ops->setvaluesblocked;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (bs < 1) SETERRQ1(PetscObjectComm((PetscObject)B),PETSC_ERR_ARG_OUTOFRANGE,"Invalid block size specified, must be positive but it is %D",bs);
  ierr   = PetscLayoutSetBlockSize(B->rmap,bs);CHKERRQ(ierr);
  ierr   = PetscLayoutSetBlockSize(B->cmap,bs);CHKERRQ(ierr);
  ierr   = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr   = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  ierr   = PetscLayoutGetBlockSize(B->rmap,&bs);CHKERRQ(ierr);
  m      = B->rmap->n/bs;

  if (ii[0]) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"ii[0] must be 0 but it is %D",ii[0]);
  ierr = PetscMalloc((m+1)*sizeof(PetscInt),&nnz);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    nz = ii[i+1] - ii[i];
    if (nz < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row %D has a negative number of columns %D",i,nz);
    nz_max = PetscMax(nz_max,nz);
    nnz[i] = nz;
  }
  ierr = MatSeqSBAIJSetPreallocation(B,bs,0,nnz);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);

  values = (PetscScalar*)V;
  if (!values) {
    ierr = PetscMalloc(bs*bs*nz_max*sizeof(PetscScalar),&values);CHKERRQ(ierr);
    ierr = PetscMemzero(values,bs*bs*nz_max*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  for (i=0; i<m; i++) {
    PetscInt          ncols  = ii[i+1] - ii[i];
    const PetscInt    *icols = jj + ii[i];
    if (!roworiented || bs == 1) {
      const PetscScalar *svals = values + (V ? (bs*bs*ii[i]) : 0);
      ierr = MatSetValuesBlocked_SeqSBAIJ(B,1,&i,ncols,icols,svals,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      for (j=0; j<ncols; j++) {
        const PetscScalar *svals = values + (V ? (bs*bs*(ii[i]+j)) : 0);
        ierr = MatSetValuesBlocked_SeqSBAIJ(B,1,&i,1,&icols[j],svals,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  if (!V) { ierr = PetscFree(values);CHKERRQ(ierr); }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static
#undef  __FUNCT__
#define __FUNCT__ "MatSeqSBAIJSetPreallocationCSR"
PetscErrorCode MatSeqSBAIJSetPreallocationCSR(Mat B,PetscInt bs,const PetscInt ii[],const PetscInt jj[], const PetscScalar V[])
{
  void (*f)(void) = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidType(B,1);
  PetscValidLogicalCollectiveInt(B,bs,2);
  ierr = PetscObjectQueryFunction((PetscObject)B,"MatSeqSBAIJSetPreallocation_C",&f);CHKERRQ(ierr);
  if (!f) PetscFunctionReturn(0);
  ierr = MatSeqSBAIJSetPreallocationCSR_SeqSBAIJ(B,bs,ii,jj,V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
