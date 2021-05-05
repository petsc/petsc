#include <petscvec_kokkos.hpp>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/mat/impls/aij/seq/kokkos/aijkokkosimpl.hpp>


PetscErrorCode MatAssemblyEnd_MPIAIJKokkos(Mat A,MatAssemblyType mode)
{
  PetscErrorCode   ierr;
  Mat_MPIAIJ       *mpiaij = (Mat_MPIAIJ*)A->data;
  Mat_SeqAIJKokkos *aijkok = mpiaij->A->spptr ? static_cast<Mat_SeqAIJKokkos*>(mpiaij->A->spptr) : NULL;

  PetscFunctionBegin;
  ierr = MatAssemblyEnd_MPIAIJ(A,mode);CHKERRQ(ierr);
  if (!A->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = VecSetType(mpiaij->lvec,VECSEQKOKKOS);CHKERRQ(ierr);
  }
  if (aijkok && aijkok->device_mat_d.data()) {
    A->offloadmask = PETSC_OFFLOAD_GPU; // in GPU mode, no going back. MatSetValues checks this
  }

  PetscFunctionReturn(0);
}

PetscErrorCode MatMPIAIJSetPreallocation_MPIAIJKokkos(Mat mat,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  PetscErrorCode ierr;
  Mat_MPIAIJ     *mpiaij = (Mat_MPIAIJ*)mat->data;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(mat->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(mat->cmap);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  if (d_nnz) {
    PetscInt i;
    for (i=0; i<mat->rmap->n; i++) {
      if (d_nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"d_nnz cannot be less than 0: local row %D value %D",i,d_nnz[i]);
    }
  }
  if (o_nnz) {
    PetscInt i;
    for (i=0; i<mat->rmap->n; i++) {
      if (o_nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"o_nnz cannot be less than 0: local row %D value %D",i,o_nnz[i]);
    }
  }
#endif
#if defined(PETSC_USE_CTABLE)
  ierr = PetscTableDestroy(&mpiaij->colmap);CHKERRQ(ierr);
#else
  ierr = PetscFree(mpiaij->colmap);CHKERRQ(ierr);
#endif
  ierr = PetscFree(mpiaij->garray);CHKERRQ(ierr);
  ierr = VecDestroy(&mpiaij->lvec);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&mpiaij->Mvctx);CHKERRQ(ierr);
  /* Because the B will have been resized we simply destroy it and create a new one each time */
  ierr = MatDestroy(&mpiaij->B);CHKERRQ(ierr);

  if (!mpiaij->A) {
    ierr = MatCreate(PETSC_COMM_SELF,&mpiaij->A);CHKERRQ(ierr);
    ierr = MatSetSizes(mpiaij->A,mat->rmap->n,mat->cmap->n,mat->rmap->n,mat->cmap->n);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)mpiaij->A);CHKERRQ(ierr);
  }
  if (!mpiaij->B) {
    PetscMPIInt size;
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size);CHKERRMPI(ierr);
    ierr = MatCreate(PETSC_COMM_SELF,&mpiaij->B);CHKERRQ(ierr);
    ierr = MatSetSizes(mpiaij->B,mat->rmap->n,size > 1 ? mat->cmap->N : 0,mat->rmap->n,size > 1 ? mat->cmap->N : 0);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)mpiaij->B);CHKERRQ(ierr);
  }
  ierr = MatSetType(mpiaij->A,MATSEQAIJKOKKOS);CHKERRQ(ierr);
  ierr = MatSetType(mpiaij->B,MATSEQAIJKOKKOS);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(mpiaij->A,d_nz,d_nnz);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(mpiaij->B,o_nz,o_nnz);CHKERRQ(ierr);
  mat->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_MPIAIJKokkos(Mat mat,Vec xx,Vec yy)
{
  Mat_MPIAIJ     *mpiaij = (Mat_MPIAIJ*)mat->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != mat->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of mat (%D) and xx (%D)",mat->cmap->n,nt);
  ierr = VecScatterBegin(mpiaij->Mvctx,xx,mpiaij->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*mpiaij->A->ops->mult)(mpiaij->A,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterEnd(mpiaij->Mvctx,xx,mpiaij->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*mpiaij->B->ops->multadd)(mpiaij->B,mpiaij->lvec,yy,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_MPIAIJKokkos(Mat mat,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIAIJ     *mpiaij = (Mat_MPIAIJ*)mat->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != mat->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of mat (%D) and xx (%D)",mat->cmap->n,nt);
  ierr = VecScatterBegin(mpiaij->Mvctx,xx,mpiaij->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*mpiaij->A->ops->multadd)(mpiaij->A,xx,yy,zz);CHKERRQ(ierr);
  ierr = VecScatterEnd(mpiaij->Mvctx,xx,mpiaij->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*mpiaij->B->ops->multadd)(mpiaij->B,mpiaij->lvec,zz,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_MPIAIJKokkos(Mat mat,Vec xx,Vec yy)
{
  Mat_MPIAIJ     *mpiaij = (Mat_MPIAIJ*)mat->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != mat->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of mat (%D) and xx (%D)",mat->rmap->n,nt);
  ierr = (*mpiaij->B->ops->multtranspose)(mpiaij->B,xx,mpiaij->lvec);CHKERRQ(ierr);
  ierr = (*mpiaij->A->ops->multtranspose)(mpiaij->A,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterBegin(mpiaij->Mvctx,mpiaij->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(mpiaij->Mvctx,mpiaij->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIAIJKokkos(Mat A, MatType mtype, MatReuse reuse, Mat* newmat)
{
  PetscErrorCode     ierr;
  Mat                B;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,newmat);CHKERRQ(ierr);
  } else if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatCopy(A,*newmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  B = *newmat;

  B->boundtocpu = PETSC_FALSE;
  ierr = PetscFree(B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECKOKKOS,&B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATMPIAIJKOKKOS);CHKERRQ(ierr);

  B->ops->assemblyend           = MatAssemblyEnd_MPIAIJKokkos;
  B->ops->mult                  = MatMult_MPIAIJKokkos;
  B->ops->multadd               = MatMultAdd_MPIAIJKokkos;
  B->ops->multtranspose         = MatMultTranspose_MPIAIJKokkos;
  // Needs an efficient implementation of the COO preallocation routines
  //B->ops->productsetfromoptions = MatProductSetFromOptions_MPIAIJBACKEND;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMPIAIJSetPreallocation_C",MatMPIAIJSetPreallocation_MPIAIJKokkos);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJKokkos(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
  ierr = MatCreate_MPIAIJ(A);CHKERRQ(ierr);
  ierr = MatConvert_MPIAIJ_MPIAIJKokkos(A,MATMPIAIJKOKKOS,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatCreateAIJKokkos - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format).  This matrix will ultimately pushed down
   to Kokkos for calculations. For good matrix
   assembly performance the user should preallocate the matrix storage by setting
   the parameter nz (or the array nnz).  By setting these parameters accurately,
   performance during matrix assembly can be increased by more than a factor of 50.

   Collective

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nz - number of nonzeros per row (same for all rows)
-  nnz - array containing the number of nonzeros in the various rows
         (possibly different for each row) or NULL

   Output Parameter:
.  A - the matrix

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradigm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqAIJSetPreallocation]

   Notes:
   If nnz is given then nz is ignored

   The AIJ format (also called the Yale sparse matrix format or
   compressed row storage), is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=NULL for PETSc to control dynamic memory
   allocation.  For large problems you MUST preallocate memory or you
   will get TERRIBLE performance, see the users' manual chapter on matrices.

   By default, this format uses inodes (identical nodes) when possible, to
   improve numerical efficiency of matrix-vector products and solves. We
   search for consecutive rows with the same nonzero structure, thereby
   reusing matrix information to achieve increased efficiency.

   Level: intermediate

.seealso: MatCreate(), MatCreateAIJ(), MatSetValues(), MatSeqAIJSetColumnIndices(), MatCreateSeqAIJWithArrays(), MatCreateAIJ(), MATMPIAIJKOKKOS, MATAIJKokkos
@*/
PetscErrorCode  MatCreateAIJKokkos(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[],Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size > 1) {
    ierr = MatSetType(*A,MATMPIAIJKOKKOS);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(*A,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*A,MATSEQAIJKOKKOS);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(*A,d_nz,d_nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

// get GPU pointer to stripped down Mat. For both Seq and MPI Mat.
PetscErrorCode MatKokkosGetDeviceMatWrite(Mat A, PetscSplitCSRDataStructure **B)
{
  #if defined(PETSC_USE_CTABLE)
  SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Device metadata does not support ctable (--with-ctable=0)");
  #else
  PetscMPIInt                size,rank;
  MPI_Comm                   comm;
  PetscErrorCode             ierr;
  PetscSplitCSRDataStructure *d_mat=NULL;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  if (size == 1) {
    ierr   = MatSeqAIJKokkosGetDeviceMat(A,&d_mat);CHKERRQ(ierr);
  } else {
    Mat_MPIAIJ  *aij = (Mat_MPIAIJ*)A->data;
    ierr   = MatSeqAIJKokkosGetDeviceMat(aij->A,&d_mat);CHKERRQ(ierr);
  }
  // act like MatSetValues because not called on host
  if (A->assembled) {
    if (A->was_assembled) {
      ierr = PetscInfo(A,"Assemble more than once already\n");CHKERRQ(ierr);
    }
    A->was_assembled = PETSC_TRUE; // this is done (lazy) in MatAssemble but we are not calling it anymore - done in AIJ AssemblyEnd, need here?
  } else {
    ierr = PetscInfo1(A,"Warning !assemble ??? assembled=%D\n",A->assembled);CHKERRQ(ierr);
    // SETERRQ(comm,PETSC_ERR_SUP,"Need assemble matrix");
  }
  if (!d_mat) {
    PetscSplitCSRDataStructure  h_mat; /* host container */
    Mat_SeqAIJKokkos            *aijkokA;
    Mat_SeqAIJ                  *jaca;
    PetscInt                    n = A->rmap->n, nnz;
    Mat                         Amat;
    // create and copy
    ierr = PetscInfo(A,"Create device matrix in Kokkos\n");CHKERRQ(ierr);
    if (size == 1) {
      Amat = A;
      jaca = (Mat_SeqAIJ*)A->data;
      h_mat.rstart = 0; h_mat.rend = A->rmap->n;
      h_mat.cstart = 0; h_mat.cend = A->cmap->n;
      h_mat.offdiag.i = h_mat.offdiag.j = NULL;
      h_mat.offdiag.a = NULL;
      aijkokA = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
      aijkokA->i_uncompressed_d = NULL;
      aijkokA->colmap_d = NULL;
    } else {
      Mat_MPIAIJ       *aij = (Mat_MPIAIJ*)A->data;
      Mat_SeqAIJ       *jacb = (Mat_SeqAIJ*)aij->B->data;
      PetscInt         ii;
      Mat_SeqAIJKokkos *aijkokB;
      Amat = aij->A;
      aijkokA = static_cast<Mat_SeqAIJKokkos*>(aij->A->spptr);
      aijkokB = static_cast<Mat_SeqAIJKokkos*>(aij->B->spptr);
      aijkokA->i_uncompressed_d = NULL;
      aijkokA->colmap_d = NULL;
      jaca = (Mat_SeqAIJ*)aij->A->data;
      if (!aij->garray) SETERRQ(comm,PETSC_ERR_PLIB,"MPIAIJ Matrix was assembled but is missing garray");
      if (aij->B->rmap->n != aij->A->rmap->n) SETERRQ(comm,PETSC_ERR_SUP,"Only support aij->B->rmap->n == aij->A->rmap->n");
      // create colmap - this is ussually done (lazy) in MatSetValues
      aij->donotstash = PETSC_TRUE;
      aij->A->nooffprocentries = aij->B->nooffprocentries = A->nooffprocentries = PETSC_TRUE;
      jaca->nonew = jacb->nonew = PETSC_TRUE; // no more dissassembly
      ierr = PetscCalloc1(A->cmap->N+1,&aij->colmap);CHKERRQ(ierr);
      aij->colmap[A->cmap->N] = -9;
      ierr = PetscLogObjectMemory((PetscObject)A,(A->cmap->N+1)*sizeof(PetscInt));CHKERRQ(ierr);
      for (ii=0; ii<aij->B->cmap->n; ii++) aij->colmap[aij->garray[ii]] = ii+1;
      if (aij->colmap[A->cmap->N] != -9) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"aij->colmap[A->cmap->N] != -9");
      // allocate B copy data
      h_mat.rstart = A->rmap->rstart; h_mat.rend = A->rmap->rend;
      h_mat.cstart = A->cmap->rstart; h_mat.cend = A->cmap->rend;
      nnz = jacb->i[n];
      if (jacb->compressedrow.use) {
        const Kokkos::View<PetscInt*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_i_k (jacb->i,n+1);
        aijkokB->i_uncompressed_d = new Kokkos::View<PetscInt*>(Kokkos::create_mirror(DeviceMemorySpace(),h_i_k));
        Kokkos::deep_copy (*aijkokB->i_uncompressed_d, h_i_k);
        h_mat.offdiag.i = aijkokB->i_uncompressed_d->data();
        //err = cudaMalloc((void **)&h_mat.offdiag.i,               (n+1)*sizeof(int));CHKERRCUDA(err); // kernel input
        //err = cudaMemcpy(          h_mat.offdiag.i,    jacb->i,   (n+1)*sizeof(int), cudaMemcpyHostToDevice);CHKERRCUDA(err);
      } else {
         h_mat.offdiag.i = (PetscInt*)aijkokB->i_d.data();
      }
      h_mat.offdiag.j = (PetscInt*)aijkokB->j_d.data();
      h_mat.offdiag.a = aijkokB->a_d.data();
      {
        Kokkos::View<PetscInt*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_colmap_k (aij->colmap,A->cmap->N);
        aijkokB->colmap_d = new Kokkos::View<PetscInt*>(Kokkos::create_mirror(DeviceMemorySpace(),h_colmap_k));
        Kokkos::deep_copy (*aijkokB->colmap_d, h_colmap_k);
        h_mat.colmap = aijkokB->colmap_d->data();
      }
      //err = cudaMalloc((void **)&h_mat.colmap,                  (A->cmap->N+1)*sizeof(PetscInt));CHKERRCUDA(err); // kernel output
      //err = cudaMemcpy(          h_mat.colmap,    aij->colmap,  (A->cmap->N+1)*sizeof(PetscInt), cudaMemcpyHostToDevice);CHKERRCUDA(err);
      h_mat.offdiag.ignorezeroentries = jacb->ignorezeroentries;
      h_mat.offdiag.n = n;
    }
    // allocate A copy data
    nnz = jaca->i[n];
    h_mat.diag.n = n;
    h_mat.diag.ignorezeroentries = jaca->ignorezeroentries;
    ierr = MPI_Comm_rank(comm,&h_mat.rank);CHKERRMPI(ierr);
    if (jaca->compressedrow.use) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"A does not suppport compressed row (todo)");
    } else {
      h_mat.diag.i = (PetscInt*)aijkokA->i_d.data();
    }
    //h_mat.diag.j = aj;
    //h_mat.diag.a = aa;
    h_mat.diag.j = (PetscInt*)aijkokA->j_d.data();
    h_mat.diag.a = aijkokA->a_d.data();
    // copy pointers and metdata to device
    ierr = MatSeqAIJKokkosSetDeviceMat(Amat,&h_mat);CHKERRQ(ierr);
    ierr = MatSeqAIJKokkosGetDeviceMat(Amat,&d_mat);CHKERRQ(ierr);
    ierr = PetscInfo2(A,"Create device Mat n=%D nnz=%D\n",h_mat.diag.n, nnz);CHKERRQ(ierr);
  }
  *B = d_mat; // return it, set it in Mat, and set it up
  A->assembled = PETSC_FALSE; // ready to write with matsetvalues - this done (lazy) in normal MatSetValues
  PetscFunctionReturn(0);
  #endif
}
