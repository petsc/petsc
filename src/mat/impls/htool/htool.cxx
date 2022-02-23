#include <../src/mat/impls/htool/htool.hpp> /*I "petscmat.h" I*/
#include <petscblaslapack.h>
#include <set>

#define ALEN(a) (sizeof(a)/sizeof((a)[0]))
const char *const MatHtoolCompressorTypes[] = { "sympartialACA", "fullACA", "SVD" };
const char *const MatHtoolClusteringTypes[] = { "PCARegular", "PCAGeometric", "BoundingBox1Regular", "BoundingBox1Geometric" };
const char HtoolCitation[] = "@article{marchand2020two,\n"
"  Author = {Marchand, Pierre and Claeys, Xavier and Jolivet, Pierre and Nataf, Fr\\'ed\\'eric and Tournier, Pierre-Henri},\n"
"  Title = {Two-level preconditioning for $h$-version boundary element approximation of hypersingular operator with {GenEO}},\n"
"  Year = {2020},\n"
"  Publisher = {Elsevier},\n"
"  Journal = {Numerische Mathematik},\n"
"  Volume = {146},\n"
"  Pages = {597--628},\n"
"  Url = {https://github.com/htool-ddm/htool}\n"
"}\n";
static PetscBool HtoolCite = PETSC_FALSE;

static PetscErrorCode MatGetDiagonal_Htool(Mat A,Vec v)
{
  Mat_Htool      *a = (Mat_Htool*)A->data;
  PetscScalar    *x;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatHasCongruentLayouts(A,&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Only congruent layouts supported");
  ierr = VecGetArrayWrite(v,&x);CHKERRQ(ierr);
  a->hmatrix->copy_local_diagonal(x);
  ierr = VecRestoreArrayWrite(v,&x);CHKERRQ(ierr);
  ierr = VecScale(v,a->s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonalBlock_Htool(Mat A,Mat *b)
{
  Mat_Htool      *a = (Mat_Htool*)A->data;
  Mat            B;
  PetscScalar    *ptr;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatHasCongruentLayouts(A,&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Only congruent layouts supported");
  ierr = PetscObjectQuery((PetscObject)A,"DiagonalBlock",(PetscObject*)&B);CHKERRQ(ierr); /* same logic as in MatGetDiagonalBlock_MPIDense() */
  if (!B) {
    ierr = MatCreateDense(PETSC_COMM_SELF,A->rmap->n,A->rmap->n,A->rmap->n,A->rmap->n,NULL,&B);CHKERRQ(ierr);
    ierr = MatDenseGetArrayWrite(B,&ptr);CHKERRQ(ierr);
    a->hmatrix->copy_local_diagonal_block(ptr);
    ierr = MatDenseRestoreArrayWrite(B,&ptr);CHKERRQ(ierr);
    ierr = MatPropagateSymmetryOptions(A,B);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatScale(B,a->s);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)A,"DiagonalBlock",(PetscObject)B);CHKERRQ(ierr);
    *b   = B;
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  } else *b = B;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_Htool(Mat A,Vec x,Vec y)
{
  Mat_Htool         *a = (Mat_Htool*)A->data;
  const PetscScalar *in;
  PetscScalar       *out;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(x,&in);CHKERRQ(ierr);
  ierr = VecGetArrayWrite(y,&out);CHKERRQ(ierr);
  a->hmatrix->mvprod_local_to_local(in,out);
  ierr = VecRestoreArrayRead(x,&in);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(y,&out);CHKERRQ(ierr);
  ierr = VecScale(y,a->s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* naive implementation of MatMultAdd() needed for FEM-BEM coupling via MATNEST */
static PetscErrorCode MatMultAdd_Htool(Mat A,Vec v1,Vec v2,Vec v3)
{
  Mat_Htool         *a = (Mat_Htool*)A->data;
  Vec               tmp;
  const PetscScalar scale = a->s;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(v2,&tmp);CHKERRQ(ierr);
  ierr = VecCopy(v2,v3);CHKERRQ(ierr); /* no-op in MatMultAdd(bA->m[i][j],bx[j],by[i],by[i]) since VecCopy() checks for x == y */
  a->s = 1.0; /* set s to 1.0 since VecAXPY() may be used to scale the MatMult() output Vec */
  ierr = MatMult_Htool(A,v1,tmp);CHKERRQ(ierr);
  ierr = VecAXPY(v3,scale,tmp);CHKERRQ(ierr);
  ierr = VecDestroy(&tmp);CHKERRQ(ierr);
  a->s = scale; /* set s back to its original value */
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_Htool(Mat A,Vec x,Vec y)
{
  Mat_Htool         *a = (Mat_Htool*)A->data;
  const PetscScalar *in;
  PetscScalar       *out;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(x,&in);CHKERRQ(ierr);
  ierr = VecGetArrayWrite(y,&out);CHKERRQ(ierr);
  a->hmatrix->mvprod_transp_local_to_local(in,out);
  ierr = VecRestoreArrayRead(x,&in);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(y,&out);CHKERRQ(ierr);
  ierr = VecScale(y,a->s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatIncreaseOverlap_Htool(Mat A,PetscInt is_max,IS is[],PetscInt ov)
{
  std::set<PetscInt> set;
  const PetscInt     *idx;
  PetscInt           *oidx,size,bs[2];
  PetscMPIInt        csize;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MatGetBlockSizes(A,bs,bs+1);CHKERRQ(ierr);
  if (bs[0] != bs[1]) bs[0] = 1;
  for (PetscInt i=0; i<is_max; ++i) {
    /* basic implementation that adds indices by shifting an IS by -ov, -ov+1..., -1, 1..., ov-1, ov */
    /* needed to avoid subdomain matrices to replicate A since it is dense                           */
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)is[i]),&csize);CHKERRMPI(ierr);
    PetscCheckFalse(csize != 1,PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported parallel IS");
    ierr = ISGetSize(is[i],&size);CHKERRQ(ierr);
    ierr = ISGetIndices(is[i],&idx);CHKERRQ(ierr);
    for (PetscInt j=0; j<size; ++j) {
      set.insert(idx[j]);
      for (PetscInt k=1; k<=ov; ++k) {               /* for each layer of overlap      */
        if (idx[j] - k >= 0) set.insert(idx[j] - k); /* do not insert negative indices */
        if (idx[j] + k < A->rmap->N && idx[j] + k < A->cmap->N) set.insert(idx[j] + k); /* do not insert indices greater than the dimension of A */
      }
    }
    ierr = ISRestoreIndices(is[i],&idx);CHKERRQ(ierr);
    ierr = ISDestroy(is+i);CHKERRQ(ierr);
    if (bs[0] > 1) {
      for (std::set<PetscInt>::iterator it=set.cbegin(); it!=set.cend(); it++) {
        std::vector<PetscInt> block(bs[0]);
        std::iota(block.begin(),block.end(),(*it/bs[0])*bs[0]);
        set.insert(block.cbegin(),block.cend());
      }
    }
    size = set.size(); /* size with overlap */
    ierr = PetscMalloc1(size,&oidx);CHKERRQ(ierr);
    for (const PetscInt j : set) *oidx++ = j;
    oidx -= size;
    ierr = ISCreateGeneral(PETSC_COMM_SELF,size,oidx,PETSC_OWN_POINTER,is+i);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatrices_Htool(Mat A,PetscInt n,const IS irow[],const IS icol[],MatReuse scall,Mat *submat[])
{
  Mat_Htool         *a = (Mat_Htool*)A->data;
  Mat               D,B,BT;
  const PetscScalar *copy;
  PetscScalar       *ptr;
  const PetscInt    *idxr,*idxc,*it;
  PetscInt          nrow,m,i;
  PetscBool         flg;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (scall != MAT_REUSE_MATRIX) {
    ierr = PetscCalloc1(n,submat);CHKERRQ(ierr);
  }
  for (i=0; i<n; ++i) {
    ierr = ISGetLocalSize(irow[i],&nrow);CHKERRQ(ierr);
    ierr = ISGetLocalSize(icol[i],&m);CHKERRQ(ierr);
    ierr = ISGetIndices(irow[i],&idxr);CHKERRQ(ierr);
    ierr = ISGetIndices(icol[i],&idxc);CHKERRQ(ierr);
    if (scall != MAT_REUSE_MATRIX) {
      ierr = MatCreateDense(PETSC_COMM_SELF,nrow,m,nrow,m,NULL,(*submat)+i);CHKERRQ(ierr);
    }
    ierr = MatDenseGetArrayWrite((*submat)[i],&ptr);CHKERRQ(ierr);
    if (irow[i] == icol[i]) { /* same row and column IS? */
      ierr = MatHasCongruentLayouts(A,&flg);CHKERRQ(ierr);
      if (flg) {
        ierr = ISSorted(irow[i],&flg);CHKERRQ(ierr);
        if (flg) { /* sorted IS? */
          it = std::lower_bound(idxr,idxr+nrow,A->rmap->rstart);
          if (it != idxr+nrow && *it == A->rmap->rstart) { /* rmap->rstart in IS? */
            if (std::distance(idxr,it) + A->rmap->n <= nrow) { /* long enough IS to store the local diagonal block? */
              for (PetscInt j=0; j<A->rmap->n && flg; ++j) if (PetscUnlikely(it[j] != A->rmap->rstart+j)) flg = PETSC_FALSE;
              if (flg) { /* complete local diagonal block in IS? */
                /* fast extraction when the local diagonal block is part of the submatrix, e.g., for PCASM or PCHPDDM
                 *      [   B   C   E   ]
                 *  A = [   B   D   E   ]
                 *      [   B   F   E   ]
                 */
                m = std::distance(idxr,it); /* shift of the coefficient (0,0) of block D from above */
                ierr = MatGetDiagonalBlock_Htool(A,&D);CHKERRQ(ierr);
                ierr = MatDenseGetArrayRead(D,&copy);CHKERRQ(ierr);
                for (PetscInt k=0; k<A->rmap->n; ++k) {
                  ierr = PetscArraycpy(ptr+(m+k)*nrow+m,copy+k*A->rmap->n,A->rmap->n);CHKERRQ(ierr); /* block D from above */
                }
                ierr = MatDenseRestoreArrayRead(D,&copy);CHKERRQ(ierr);
                if (m) {
                  a->wrapper->copy_submatrix(nrow,m,idxr,idxc,ptr); /* vertical block B from above */
                  /* entry-wise assembly may be costly, so transpose already-computed entries when possible */
                  if (A->symmetric || A->hermitian) {
                    ierr = MatCreateDense(PETSC_COMM_SELF,A->rmap->n,m,A->rmap->n,m,ptr+m,&B);CHKERRQ(ierr);
                    ierr = MatDenseSetLDA(B,nrow);CHKERRQ(ierr);
                    ierr = MatCreateDense(PETSC_COMM_SELF,m,A->rmap->n,m,A->rmap->n,ptr+m*nrow,&BT);CHKERRQ(ierr);
                    ierr = MatDenseSetLDA(BT,nrow);CHKERRQ(ierr);
                    if (A->hermitian && PetscDefined(USE_COMPLEX)) {
                      ierr = MatHermitianTranspose(B,MAT_REUSE_MATRIX,&BT);CHKERRQ(ierr);
                    } else {
                      ierr = MatTranspose(B,MAT_REUSE_MATRIX,&BT);CHKERRQ(ierr);
                    }
                    ierr = MatDestroy(&B);CHKERRQ(ierr);
                    ierr = MatDestroy(&BT);CHKERRQ(ierr);
                  } else {
                    for (PetscInt k=0; k<A->rmap->n; ++k) { /* block C from above */
                      a->wrapper->copy_submatrix(m,1,idxr,idxc+m+k,ptr+(m+k)*nrow);
                    }
                  }
                }
                if (m+A->rmap->n != nrow) {
                  a->wrapper->copy_submatrix(nrow,std::distance(it+A->rmap->n,idxr+nrow),idxr,idxc+m+A->rmap->n,ptr+(m+A->rmap->n)*nrow); /* vertical block E from above */
                  /* entry-wise assembly may be costly, so transpose already-computed entries when possible */
                  if (A->symmetric || A->hermitian) {
                    ierr = MatCreateDense(PETSC_COMM_SELF,A->rmap->n,nrow-(m+A->rmap->n),A->rmap->n,nrow-(m+A->rmap->n),ptr+(m+A->rmap->n)*nrow+m,&B);CHKERRQ(ierr);
                    ierr = MatDenseSetLDA(B,nrow);CHKERRQ(ierr);
                    ierr = MatCreateDense(PETSC_COMM_SELF,nrow-(m+A->rmap->n),A->rmap->n,nrow-(m+A->rmap->n),A->rmap->n,ptr+m*nrow+m+A->rmap->n,&BT);CHKERRQ(ierr);
                    ierr = MatDenseSetLDA(BT,nrow);CHKERRQ(ierr);
                    if (A->hermitian && PetscDefined(USE_COMPLEX)) {
                      ierr = MatHermitianTranspose(B,MAT_REUSE_MATRIX,&BT);CHKERRQ(ierr);
                    } else {
                      ierr = MatTranspose(B,MAT_REUSE_MATRIX,&BT);CHKERRQ(ierr);
                    }
                    ierr = MatDestroy(&B);CHKERRQ(ierr);
                    ierr = MatDestroy(&BT);CHKERRQ(ierr);
                  } else {
                    for (PetscInt k=0; k<A->rmap->n; ++k) { /* block F from above */
                      a->wrapper->copy_submatrix(std::distance(it+A->rmap->n,idxr+nrow),1,it+A->rmap->n,idxc+m+k,ptr+(m+k)*nrow+m+A->rmap->n);
                    }
                  }
                }
              } /* complete local diagonal block not in IS */
            } else flg = PETSC_FALSE; /* IS not long enough to store the local diagonal block */
          } else flg = PETSC_FALSE; /* rmap->rstart not in IS */
        } /* unsorted IS */
      }
    } else flg = PETSC_FALSE; /* different row and column IS */
    if (!flg) a->wrapper->copy_submatrix(nrow,m,idxr,idxc,ptr); /* reassemble everything */
    ierr = ISRestoreIndices(irow[i],&idxr);CHKERRQ(ierr);
    ierr = ISRestoreIndices(icol[i],&idxc);CHKERRQ(ierr);
    ierr = MatDenseRestoreArrayWrite((*submat)[i],&ptr);CHKERRQ(ierr);
    ierr = MatAssemblyBegin((*submat)[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd((*submat)[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatScale((*submat)[i],a->s);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_Htool(Mat A)
{
  Mat_Htool               *a = (Mat_Htool*)A->data;
  PetscContainer          container;
  MatHtoolKernelTranspose *kernelt;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  ierr = PetscObjectChangeTypeName((PetscObject)A,NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_htool_seqdense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_htool_mpidense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_htool_seqdense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_htool_mpidense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatHtoolGetHierarchicalMat_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatHtoolSetKernel_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatHtoolGetPermutationSource_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatHtoolGetPermutationTarget_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatHtoolUsePermutation_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)A,"KernelTranspose",(PetscObject*)&container);CHKERRQ(ierr);
  if (container) { /* created in MatTranspose_Htool() */
    ierr = PetscContainerGetPointer(container,(void**)&kernelt);CHKERRQ(ierr);
    ierr = MatDestroy(&kernelt->A);CHKERRQ(ierr);
    ierr = PetscFree(kernelt);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)A,"KernelTranspose",NULL);CHKERRQ(ierr);
  }
  if (a->gcoords_source != a->gcoords_target) {
    ierr = PetscFree(a->gcoords_source);CHKERRQ(ierr);
  }
  ierr = PetscFree(a->gcoords_target);CHKERRQ(ierr);
  ierr = PetscFree2(a->work_source,a->work_target);CHKERRQ(ierr);
  delete a->wrapper;
  delete a->hmatrix;
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_Htool(Mat A,PetscViewer pv)
{
  Mat_Htool      *a = (Mat_Htool*)A->data;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)pv,PETSCVIEWERASCII,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerASCIIPrintf(pv,"symmetry: %c\n",a->hmatrix->get_symmetry_type());CHKERRQ(ierr);
    if (PetscAbsScalar(a->s-1.0) > PETSC_MACHINE_EPSILON) {
#if defined(PETSC_USE_COMPLEX)
      ierr = PetscViewerASCIIPrintf(pv,"scaling: %g+%gi\n",(double)PetscRealPart(a->s),(double)PetscImaginaryPart(a->s));CHKERRQ(ierr);
#else
      ierr = PetscViewerASCIIPrintf(pv,"scaling: %g\n",(double)a->s);CHKERRQ(ierr);
#endif
    }
    ierr = PetscViewerASCIIPrintf(pv,"minimum cluster size: %" PetscInt_FMT "\n",a->bs[0]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"maximum block size: %" PetscInt_FMT "\n",a->bs[1]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"epsilon: %g\n",(double)a->epsilon);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"eta: %g\n",(double)a->eta);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"minimum target depth: %" PetscInt_FMT "\n",a->depth[0]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"minimum source depth: %" PetscInt_FMT "\n",a->depth[1]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"compressor: %s\n",MatHtoolCompressorTypes[a->compressor]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"clustering: %s\n",MatHtoolClusteringTypes[a->clustering]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"compression ratio: %s\n",a->hmatrix->get_infos("Compression_ratio").c_str());CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"space saving: %s\n",a->hmatrix->get_infos("Space_saving").c_str());CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"number of dense (resp. low rank) matrices: %s (resp. %s)\n",a->hmatrix->get_infos("Number_of_dmat").c_str(),a->hmatrix->get_infos("Number_of_lrmat").c_str());CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"(minimum, mean, maximum) dense block sizes: (%s, %s, %s)\n",a->hmatrix->get_infos("Dense_block_size_min").c_str(),a->hmatrix->get_infos("Dense_block_size_mean").c_str(),a->hmatrix->get_infos("Dense_block_size_max").c_str());CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"(minimum, mean, maximum) low rank block sizes: (%s, %s, %s)\n",a->hmatrix->get_infos("Low_rank_block_size_min").c_str(),a->hmatrix->get_infos("Low_rank_block_size_mean").c_str(),a->hmatrix->get_infos("Low_rank_block_size_max").c_str());CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"(minimum, mean, maximum) ranks: (%s, %s, %s)\n",a->hmatrix->get_infos("Rank_min").c_str(),a->hmatrix->get_infos("Rank_mean").c_str(),a->hmatrix->get_infos("Rank_max").c_str());CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_Htool(Mat A,PetscScalar s)
{
  Mat_Htool *a = (Mat_Htool*)A->data;

  PetscFunctionBegin;
  a->s *= s;
  PetscFunctionReturn(0);
}

/* naive implementation of MatGetRow() needed for MatConvert_Nest_AIJ() */
static PetscErrorCode MatGetRow_Htool(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_Htool      *a = (Mat_Htool*)A->data;
  PetscInt       *idxc;
  PetscBLASInt   one = 1,bn;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (nz) *nz = A->cmap->N;
  if (idx || v) { /* even if !idx, need to set idxc for htool::copy_submatrix() */
    ierr = PetscMalloc1(A->cmap->N,&idxc);CHKERRQ(ierr);
    for (PetscInt i=0; i<A->cmap->N; ++i) idxc[i] = i;
  }
  if (idx) *idx = idxc;
  if (v) {
    ierr = PetscMalloc1(A->cmap->N,v);CHKERRQ(ierr);
    if (a->wrapper) a->wrapper->copy_submatrix(1,A->cmap->N,&row,idxc,*v);
    else reinterpret_cast<htool::VirtualGenerator<PetscScalar>*>(a->kernelctx)->copy_submatrix(1,A->cmap->N,&row,idxc,*v);
    ierr = PetscBLASIntCast(A->cmap->N,&bn);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASscal",BLASscal_(&bn,&a->s,*v,&one));
  }
  if (!idx) {
    ierr = PetscFree(idxc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatRestoreRow_Htool(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (nz) *nz = 0;
  if (idx) {
    ierr = PetscFree(*idx);CHKERRQ(ierr);
  }
  if (v) {
    ierr = PetscFree(*v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetFromOptions_Htool(PetscOptionItems *PetscOptionsObject,Mat A)
{
  Mat_Htool      *a = (Mat_Htool*)A->data;
  PetscInt       n;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Htool options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_htool_min_cluster_size","Minimal leaf size in cluster tree",NULL,a->bs[0],a->bs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_htool_max_block_size","Maximal number of coefficients in a dense block",NULL,a->bs[1],a->bs + 1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_htool_epsilon","Relative error in Frobenius norm when approximating a block",NULL,a->epsilon,&a->epsilon,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_htool_eta","Admissibility condition tolerance",NULL,a->eta,&a->eta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_htool_min_target_depth","Minimal cluster tree depth associated with the rows",NULL,a->depth[0],a->depth,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_htool_min_source_depth","Minimal cluster tree depth associated with the columns",NULL,a->depth[1],a->depth + 1,NULL);CHKERRQ(ierr);
  n = 0;
  ierr = PetscOptionsEList("-mat_htool_compressor","Type of compression","MatHtoolCompressorType",MatHtoolCompressorTypes,ALEN(MatHtoolCompressorTypes),MatHtoolCompressorTypes[MAT_HTOOL_COMPRESSOR_SYMPARTIAL_ACA],&n,&flg);CHKERRQ(ierr);
  if (flg) a->compressor = MatHtoolCompressorType(n);
  n = 0;
  ierr = PetscOptionsEList("-mat_htool_clustering","Type of clustering","MatHtoolClusteringType",MatHtoolClusteringTypes,ALEN(MatHtoolClusteringTypes),MatHtoolClusteringTypes[MAT_HTOOL_CLUSTERING_PCA_REGULAR],&n,&flg);CHKERRQ(ierr);
  if (flg) a->clustering = MatHtoolClusteringType(n);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyEnd_Htool(Mat A,MatAssemblyType type)
{
  Mat_Htool                                                    *a = (Mat_Htool*)A->data;
  const PetscInt                                               *ranges;
  PetscInt                                                     *offset;
  PetscMPIInt                                                  size;
  char                                                         S = PetscDefined(USE_COMPLEX) && A->hermitian ? 'H' : (A->symmetric ? 'S' : 'N'),uplo = S == 'N' ? 'N' : 'U';
  htool::VirtualGenerator<PetscScalar>                         *generator = nullptr;
  std::shared_ptr<htool::VirtualCluster>                       t,s = nullptr;
  std::shared_ptr<htool::VirtualLowRankGenerator<PetscScalar>> compressor = nullptr;
  PetscErrorCode                                               ierr;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(HtoolCitation,&HtoolCite);CHKERRQ(ierr);
  delete a->wrapper;
  delete a->hmatrix;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);
  ierr = PetscMalloc1(2*size,&offset);CHKERRQ(ierr);
  ierr = MatGetOwnershipRanges(A,&ranges);CHKERRQ(ierr);
  for (PetscInt i=0; i<size; ++i) {
    offset[2*i] = ranges[i];
    offset[2*i+1] = ranges[i+1] - ranges[i];
  }
  switch (a->clustering) {
  case MAT_HTOOL_CLUSTERING_PCA_GEOMETRIC:
    t = std::make_shared<htool::Cluster<htool::PCA<htool::SplittingTypes::GeometricSplitting>>>(a->dim);
    break;
  case MAT_HTOOL_CLUSTERING_BOUNDING_BOX_1_GEOMETRIC:
    t = std::make_shared<htool::Cluster<htool::BoundingBox1<htool::SplittingTypes::GeometricSplitting>>>(a->dim);
    break;
  case MAT_HTOOL_CLUSTERING_BOUNDING_BOX_1_REGULAR:
    t = std::make_shared<htool::Cluster<htool::BoundingBox1<htool::SplittingTypes::RegularSplitting>>>(a->dim);
    break;
  default:
    t = std::make_shared<htool::Cluster<htool::PCA<htool::SplittingTypes::RegularSplitting>>>(a->dim);
  }
  t->set_minclustersize(a->bs[0]);
  t->build(A->rmap->N,a->gcoords_target,offset);
  if (a->kernel) a->wrapper = new WrapperHtool(A->rmap->N,A->cmap->N,a->dim,a->kernel,a->kernelctx);
  else {
    a->wrapper = NULL;
    generator = reinterpret_cast<htool::VirtualGenerator<PetscScalar>*>(a->kernelctx);
  }
  if (a->gcoords_target != a->gcoords_source) {
    ierr = MatGetOwnershipRangesColumn(A,&ranges);CHKERRQ(ierr);
    for (PetscInt i=0; i<size; ++i) {
      offset[2*i] = ranges[i];
      offset[2*i+1] = ranges[i+1] - ranges[i];
    }
    switch (a->clustering) {
    case MAT_HTOOL_CLUSTERING_PCA_GEOMETRIC:
      s = std::make_shared<htool::Cluster<htool::PCA<htool::SplittingTypes::GeometricSplitting>>>(a->dim);
      break;
    case MAT_HTOOL_CLUSTERING_BOUNDING_BOX_1_GEOMETRIC:
      s = std::make_shared<htool::Cluster<htool::BoundingBox1<htool::SplittingTypes::GeometricSplitting>>>(a->dim);
      break;
    case MAT_HTOOL_CLUSTERING_BOUNDING_BOX_1_REGULAR:
      s = std::make_shared<htool::Cluster<htool::BoundingBox1<htool::SplittingTypes::RegularSplitting>>>(a->dim);
      break;
    default:
      s = std::make_shared<htool::Cluster<htool::PCA<htool::SplittingTypes::RegularSplitting>>>(a->dim);
    }
    s->set_minclustersize(a->bs[0]);
    s->build(A->cmap->N,a->gcoords_source,offset);
    S = uplo = 'N';
  }
  ierr = PetscFree(offset);CHKERRQ(ierr);
  switch (a->compressor) {
  case MAT_HTOOL_COMPRESSOR_FULL_ACA:
    compressor = std::make_shared<htool::fullACA<PetscScalar>>();
    break;
  case MAT_HTOOL_COMPRESSOR_SVD:
    compressor = std::make_shared<htool::SVD<PetscScalar>>();
    break;
  default:
    compressor = std::make_shared<htool::sympartialACA<PetscScalar>>();
  }
  a->hmatrix = dynamic_cast<htool::VirtualHMatrix<PetscScalar>*>(new htool::HMatrix<PetscScalar>(t,s ? s : t,a->epsilon,a->eta,S,uplo));
  a->hmatrix->set_compression(compressor);
  a->hmatrix->set_maxblocksize(a->bs[1]);
  a->hmatrix->set_mintargetdepth(a->depth[0]);
  a->hmatrix->set_minsourcedepth(a->depth[1]);
  if (s) a->hmatrix->build(a->wrapper ? *a->wrapper : *generator,a->gcoords_target,a->gcoords_source);
  else   a->hmatrix->build(a->wrapper ? *a->wrapper : *generator,a->gcoords_target);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductNumeric_Htool(Mat C)
{
  Mat_Product       *product = C->product;
  Mat_Htool         *a = (Mat_Htool*)product->A->data;
  const PetscScalar *in;
  PetscScalar       *out;
  PetscInt          N,lda;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  ierr = MatGetSize(C,NULL,&N);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(C,&lda);CHKERRQ(ierr);
  PetscCheckFalse(lda != C->rmap->n,PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported leading dimension (%" PetscInt_FMT " != %" PetscInt_FMT ")",lda,C->rmap->n);
  ierr = MatDenseGetArrayRead(product->B,&in);CHKERRQ(ierr);
  ierr = MatDenseGetArrayWrite(C,&out);CHKERRQ(ierr);
  switch (product->type) {
  case MATPRODUCT_AB:
    a->hmatrix->mvprod_local_to_local(in,out,N);
    break;
  case MATPRODUCT_AtB:
    a->hmatrix->mvprod_transp_local_to_local(in,out,N);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatProductType %s is not supported",MatProductTypes[product->type]);
  }
  ierr = MatDenseRestoreArrayWrite(C,&out);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(product->B,&in);CHKERRQ(ierr);
  ierr = MatScale(C,a->s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSymbolic_Htool(Mat C)
{
  Mat_Product    *product = C->product;
  Mat            A,B;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  A = product->A;
  B = product->B;
  ierr = PetscObjectTypeCompareAny((PetscObject)B,&flg,MATSEQDENSE,MATMPIDENSE,"");CHKERRQ(ierr);
  PetscCheckFalse(!flg,PetscObjectComm((PetscObject)B),PETSC_ERR_SUP,"MatProduct_AB not supported for %s",((PetscObject)product->B)->type_name);
  switch (product->type) {
  case MATPRODUCT_AB:
    if (C->rmap->n == PETSC_DECIDE || C->cmap->n == PETSC_DECIDE || C->rmap->N == PETSC_DECIDE || C->cmap->N == PETSC_DECIDE) {
      ierr = MatSetSizes(C,A->rmap->n,B->cmap->n,A->rmap->N,B->cmap->N);CHKERRQ(ierr);
    }
    break;
  case MATPRODUCT_AtB:
    if (C->rmap->n == PETSC_DECIDE || C->cmap->n == PETSC_DECIDE || C->rmap->N == PETSC_DECIDE || C->cmap->N == PETSC_DECIDE) {
      ierr = MatSetSizes(C,A->cmap->n,B->cmap->n,A->cmap->N,B->cmap->N);CHKERRQ(ierr);
    }
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_SUP,"ProductType %s is not supported",MatProductTypes[product->type]);
  }
  ierr = MatSetType(C,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  ierr = MatSetOption(C,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  C->ops->productsymbolic = NULL;
  C->ops->productnumeric = MatProductNumeric_Htool;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_Htool(Mat C)
{
  PetscFunctionBegin;
  MatCheckProduct(C,1);
  if (C->product->type == MATPRODUCT_AB || C->product->type == MATPRODUCT_AtB) C->ops->productsymbolic = MatProductSymbolic_Htool;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatHtoolGetHierarchicalMat_Htool(Mat A,const htool::VirtualHMatrix<PetscScalar> **hmatrix)
{
  Mat_Htool *a = (Mat_Htool*)A->data;

  PetscFunctionBegin;
  *hmatrix = a->hmatrix;
  PetscFunctionReturn(0);
}

/*@C
     MatHtoolGetHierarchicalMat - Retrieves the opaque pointer to a Htool virtual matrix stored in a MATHTOOL.

   Input Parameter:
.     A - hierarchical matrix

   Output Parameter:
.     hmatrix - opaque pointer to a Htool virtual matrix

   Level: advanced

.seealso:  MATHTOOL
@*/
PETSC_EXTERN PetscErrorCode MatHtoolGetHierarchicalMat(Mat A,const htool::VirtualHMatrix<PetscScalar> **hmatrix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(hmatrix,2);
  ierr = PetscTryMethod(A,"MatHtoolGetHierarchicalMat_C",(Mat,const htool::VirtualHMatrix<PetscScalar>**),(A,hmatrix));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatHtoolSetKernel_Htool(Mat A,MatHtoolKernel kernel,void *kernelctx)
{
  Mat_Htool *a = (Mat_Htool*)A->data;

  PetscFunctionBegin;
  a->kernel    = kernel;
  a->kernelctx = kernelctx;
  delete a->wrapper;
  if (a->kernel) a->wrapper = new WrapperHtool(A->rmap->N,A->cmap->N,a->dim,a->kernel,a->kernelctx);
  PetscFunctionReturn(0);
}

/*@C
     MatHtoolSetKernel - Sets the kernel and context used for the assembly of a MATHTOOL.

   Input Parameters:
+     A - hierarchical matrix
.     kernel - computational kernel (or NULL)
-     kernelctx - kernel context (if kernel is NULL, the pointer must be of type htool::VirtualGenerator<PetscScalar>*)

   Level: advanced

.seealso:  MATHTOOL, MatCreateHtoolFromKernel()
@*/
PETSC_EXTERN PetscErrorCode MatHtoolSetKernel(Mat A,MatHtoolKernel kernel,void *kernelctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  if (!kernelctx) PetscValidFunction(kernel,2);
  if (!kernel)    PetscValidPointer(kernelctx,3);
  ierr = PetscTryMethod(A,"MatHtoolSetKernel_C",(Mat,MatHtoolKernel,void*),(A,kernel,kernelctx));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatHtoolGetPermutationSource_Htool(Mat A,IS* is)
{
  Mat_Htool             *a = (Mat_Htool*)A->data;
  std::vector<PetscInt> source;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  source = a->hmatrix->get_source_cluster()->get_local_perm();
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)A),source.size(),source.data(),PETSC_COPY_VALUES,is);CHKERRQ(ierr);
  ierr = ISSetPermutation(*is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
     MatHtoolGetPermutationSource - Gets the permutation associated to the source cluster.

   Input Parameter:
.     A - hierarchical matrix

   Output Parameter:
.     is - permutation

   Level: advanced

.seealso:  MATHTOOL, MatHtoolGetPermutationTarget(), MatHtoolUsePermutation()
@*/
PETSC_EXTERN PetscErrorCode MatHtoolGetPermutationSource(Mat A,IS* is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  if (!is) PetscValidPointer(is,2);
  ierr = PetscTryMethod(A,"MatHtoolGetPermutationSource_C",(Mat,IS*),(A,is));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatHtoolGetPermutationTarget_Htool(Mat A,IS* is)
{
  Mat_Htool             *a = (Mat_Htool*)A->data;
  std::vector<PetscInt> target;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  target = a->hmatrix->get_target_cluster()->get_local_perm();
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)A),target.size(),target.data(),PETSC_COPY_VALUES,is);CHKERRQ(ierr);
  ierr = ISSetPermutation(*is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
     MatHtoolGetPermutationTarget - Gets the permutation associated to the target cluster.

   Input Parameter:
.     A - hierarchical matrix

   Output Parameter:
.     is - permutation

   Level: advanced

.seealso:  MATHTOOL, MatHtoolGetPermutationSource(), MatHtoolUsePermutation()
@*/
PETSC_EXTERN PetscErrorCode MatHtoolGetPermutationTarget(Mat A,IS* is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  if (!is) PetscValidPointer(is,2);
  ierr = PetscTryMethod(A,"MatHtoolGetPermutationTarget_C",(Mat,IS*),(A,is));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatHtoolUsePermutation_Htool(Mat A,PetscBool use)
{
  Mat_Htool *a = (Mat_Htool*)A->data;

  PetscFunctionBegin;
  a->hmatrix->set_use_permutation(use);
  PetscFunctionReturn(0);
}

/*@C
     MatHtoolUsePermutation - Sets whether MATHTOOL should permute input (resp. output) vectors following its internal source (resp. target) permutation.

   Input Parameters:
+     A - hierarchical matrix
-     use - Boolean value

   Level: advanced

.seealso:  MATHTOOL, MatHtoolGetPermutationSource(), MatHtoolGetPermutationTarget()
@*/
PETSC_EXTERN PetscErrorCode MatHtoolUsePermutation(Mat A,PetscBool use)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidLogicalCollectiveBool(A,use,2);
  ierr = PetscTryMethod(A,"MatHtoolUsePermutation_C",(Mat,PetscBool),(A,use));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatConvert_Htool_Dense(Mat A,MatType newtype,MatReuse reuse,Mat *B)
{
  Mat            C;
  Mat_Htool      *a = (Mat_Htool*)A->data;
  PetscInt       lda;
  PetscScalar    *array;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) {
    C = *B;
    PetscCheckFalse(C->rmap->n != A->rmap->n || C->cmap->N != A->cmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible dimensions");
    ierr = MatDenseGetLDA(C,&lda);CHKERRQ(ierr);
    PetscCheckFalse(lda != C->rmap->n,PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported leading dimension (%" PetscInt_FMT " != %" PetscInt_FMT ")",lda,C->rmap->n);
  } else {
    ierr = MatCreate(PetscObjectComm((PetscObject)A),&C);CHKERRQ(ierr);
    ierr = MatSetSizes(C,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
    ierr = MatSetType(C,MATDENSE);CHKERRQ(ierr);
    ierr = MatSetUp(C);CHKERRQ(ierr);
    ierr = MatSetOption(C,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatDenseGetArrayWrite(C,&array);CHKERRQ(ierr);
  a->hmatrix->copy_local_dense_perm(array);
  ierr = MatDenseRestoreArrayWrite(C,&array);CHKERRQ(ierr);
  ierr = MatScale(C,a->s);CHKERRQ(ierr);
  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&C);CHKERRQ(ierr);
  } else *B = C;
  PetscFunctionReturn(0);
}

static PetscErrorCode GenEntriesTranspose(PetscInt sdim,PetscInt M,PetscInt N,const PetscInt *rows,const PetscInt *cols,PetscScalar *ptr,void *ctx)
{
  MatHtoolKernelTranspose *generator = (MatHtoolKernelTranspose*)ctx;
  PetscScalar             *tmp;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  generator->kernel(sdim,N,M,cols,rows,ptr,generator->kernelctx);
  ierr = PetscMalloc1(M*N,&tmp);CHKERRQ(ierr);
  ierr = PetscArraycpy(tmp,ptr,M*N);CHKERRQ(ierr);
  for (PetscInt i=0; i<M; ++i) {
    for (PetscInt j=0; j<N; ++j) ptr[i+j*M] = tmp[j+i*N];
  }
  ierr = PetscFree(tmp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* naive implementation which keeps a reference to the original Mat */
static PetscErrorCode MatTranspose_Htool(Mat A,MatReuse reuse,Mat *B)
{
  Mat                     C;
  Mat_Htool               *a = (Mat_Htool*)A->data,*c;
  PetscInt                M = A->rmap->N,N = A->cmap->N,m = A->rmap->n,n = A->cmap->n;
  PetscContainer          container;
  MatHtoolKernelTranspose *kernelt;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscCheckFalse(reuse == MAT_INPLACE_MATRIX,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MatTranspose() with MAT_INPLACE_MATRIX not supported");
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatCreate(PetscObjectComm((PetscObject)A),&C);CHKERRQ(ierr);
    ierr = MatSetSizes(C,n,m,N,M);CHKERRQ(ierr);
    ierr = MatSetType(C,((PetscObject)A)->type_name);CHKERRQ(ierr);
    ierr = MatSetUp(C);CHKERRQ(ierr);
    ierr = PetscContainerCreate(PetscObjectComm((PetscObject)C),&container);CHKERRQ(ierr);
    ierr = PetscNew(&kernelt);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(container,kernelt);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)C,"KernelTranspose",(PetscObject)container);CHKERRQ(ierr);
  } else {
    C = *B;
    ierr = PetscObjectQuery((PetscObject)C,"KernelTranspose",(PetscObject*)&container);CHKERRQ(ierr);
    PetscCheckFalse(!container,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call MatTranspose() with MAT_INITIAL_MATRIX first");
    ierr = PetscContainerGetPointer(container,(void**)&kernelt);CHKERRQ(ierr);
  }
  c                  = (Mat_Htool*)C->data;
  c->dim             = a->dim;
  c->s               = a->s;
  c->kernel          = GenEntriesTranspose;
  if (kernelt->A != A) {
    ierr = MatDestroy(&kernelt->A);CHKERRQ(ierr);
    kernelt->A       = A;
    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  }
  kernelt->kernel    = a->kernel;
  kernelt->kernelctx = a->kernelctx;
  c->kernelctx       = kernelt;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = PetscMalloc1(N*c->dim,&c->gcoords_target);CHKERRQ(ierr);
    ierr = PetscArraycpy(c->gcoords_target,a->gcoords_source,N*c->dim);CHKERRQ(ierr);
    if (a->gcoords_target != a->gcoords_source) {
      ierr = PetscMalloc1(M*c->dim,&c->gcoords_source);CHKERRQ(ierr);
      ierr = PetscArraycpy(c->gcoords_source,a->gcoords_target,M*c->dim);CHKERRQ(ierr);
    } else c->gcoords_source = c->gcoords_target;
    ierr = PetscCalloc2(M,&c->work_source,N,&c->work_target);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX) *B = C;
  PetscFunctionReturn(0);
}

/*@C
     MatCreateHtoolFromKernel - Creates a MATHTOOL from a user-supplied kernel.

   Input Parameters:
+     comm - MPI communicator
.     m - number of local rows (or PETSC_DECIDE to have calculated if M is given)
.     n - number of local columns (or PETSC_DECIDE to have calculated if N is given)
.     M - number of global rows (or PETSC_DETERMINE to have calculated if m is given)
.     N - number of global columns (or PETSC_DETERMINE to have calculated if n is given)
.     spacedim - dimension of the space coordinates
.     coords_target - coordinates of the target
.     coords_source - coordinates of the source
.     kernel - computational kernel (or NULL)
-     kernelctx - kernel context (if kernel is NULL, the pointer must be of type htool::VirtualGenerator<PetscScalar>*)

   Output Parameter:
.     B - matrix

   Options Database Keys:
+     -mat_htool_min_cluster_size <PetscInt> - minimal leaf size in cluster tree
.     -mat_htool_max_block_size <PetscInt> - maximal number of coefficients in a dense block
.     -mat_htool_epsilon <PetscReal> - relative error in Frobenius norm when approximating a block
.     -mat_htool_eta <PetscReal> - admissibility condition tolerance
.     -mat_htool_min_target_depth <PetscInt> - minimal cluster tree depth associated with the rows
.     -mat_htool_min_source_depth <PetscInt> - minimal cluster tree depth associated with the columns
.     -mat_htool_compressor <sympartialACA, fullACA, SVD> - type of compression
-     -mat_htool_clustering <PCARegular, PCAGeometric, BounbingBox1Regular, BoundingBox1Geometric> - type of clustering

   Level: intermediate

.seealso:  MatCreate(), MATHTOOL, PCSetCoordinates(), MatHtoolSetKernel(), MatHtoolCompressorType, MATH2OPUS, MatCreateH2OpusFromKernel()
@*/
PetscErrorCode MatCreateHtoolFromKernel(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt spacedim,const PetscReal coords_target[],const PetscReal coords_source[],MatHtoolKernel kernel,void *kernelctx,Mat *B)
{
  Mat            A;
  Mat_Htool      *a;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  PetscValidLogicalCollectiveInt(A,spacedim,6);
  PetscValidRealPointer(coords_target,7);
  PetscValidRealPointer(coords_source,8);
  if (!kernelctx) PetscValidFunction(kernel,9);
  if (!kernel)    PetscValidPointer(kernelctx,10);
  ierr = MatSetSizes(A,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(A,MATHTOOL);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  a            = (Mat_Htool*)A->data;
  a->dim       = spacedim;
  a->s         = 1.0;
  a->kernel    = kernel;
  a->kernelctx = kernelctx;
  ierr = PetscCalloc1(A->rmap->N*spacedim,&a->gcoords_target);CHKERRQ(ierr);
  ierr = PetscArraycpy(a->gcoords_target+A->rmap->rstart*spacedim,coords_target,A->rmap->n*spacedim);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(MPI_IN_PLACE,a->gcoords_target,A->rmap->N*spacedim,MPIU_REAL,MPI_SUM,PetscObjectComm((PetscObject)A));CHKERRMPI(ierr); /* global target coordinates */
  if (coords_target != coords_source) {
    ierr = PetscCalloc1(A->cmap->N*spacedim,&a->gcoords_source);CHKERRQ(ierr);
    ierr = PetscArraycpy(a->gcoords_source+A->cmap->rstart*spacedim,coords_source,A->cmap->n*spacedim);CHKERRQ(ierr);
    ierr = MPIU_Allreduce(MPI_IN_PLACE,a->gcoords_source,A->cmap->N*spacedim,MPIU_REAL,MPI_SUM,PetscObjectComm((PetscObject)A));CHKERRMPI(ierr); /* global source coordinates */
  } else a->gcoords_source = a->gcoords_target;
  ierr = PetscCalloc2(A->cmap->N,&a->work_source,A->rmap->N,&a->work_target);CHKERRQ(ierr);
  *B = A;
  PetscFunctionReturn(0);
}

/*MC
     MATHTOOL = "htool" - A matrix type for hierarchical matrices using the Htool package.

  Use ./configure --download-htool to install PETSc to use Htool.

   Options Database Keys:
.     -mat_type htool - matrix type to "htool" during a call to MatSetFromOptions()

   Level: beginner

.seealso: MATH2OPUS, MATDENSE, MatCreateHtoolFromKernel(), MatHtoolSetKernel()
M*/
PETSC_EXTERN PetscErrorCode MatCreate_Htool(Mat A)
{
  Mat_Htool      *a;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(A,&a);CHKERRQ(ierr);
  A->data = (void*)a;
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATHTOOL);CHKERRQ(ierr);
  ierr = PetscMemzero(A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);
  A->ops->getdiagonal       = MatGetDiagonal_Htool;
  A->ops->getdiagonalblock  = MatGetDiagonalBlock_Htool;
  A->ops->mult              = MatMult_Htool;
  A->ops->multadd           = MatMultAdd_Htool;
  A->ops->multtranspose     = MatMultTranspose_Htool;
  if (!PetscDefined(USE_COMPLEX)) A->ops->multhermitiantranspose = MatMultTranspose_Htool;
  A->ops->increaseoverlap   = MatIncreaseOverlap_Htool;
  A->ops->createsubmatrices = MatCreateSubMatrices_Htool;
  A->ops->transpose         = MatTranspose_Htool;
  A->ops->destroy           = MatDestroy_Htool;
  A->ops->view              = MatView_Htool;
  A->ops->setfromoptions    = MatSetFromOptions_Htool;
  A->ops->scale             = MatScale_Htool;
  A->ops->getrow            = MatGetRow_Htool;
  A->ops->restorerow        = MatRestoreRow_Htool;
  A->ops->assemblyend       = MatAssemblyEnd_Htool;
  a->dim                    = 0;
  a->gcoords_target         = NULL;
  a->gcoords_source         = NULL;
  a->s                      = 1.0;
  a->bs[0]                  = 10;
  a->bs[1]                  = 1000000;
  a->epsilon                = PetscSqrtReal(PETSC_SMALL);
  a->eta                    = 10.0;
  a->depth[0]               = 0;
  a->depth[1]               = 0;
  a->compressor             = MAT_HTOOL_COMPRESSOR_SYMPARTIAL_ACA;
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_htool_seqdense_C",MatProductSetFromOptions_Htool);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_htool_mpidense_C",MatProductSetFromOptions_Htool);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_htool_seqdense_C",MatConvert_Htool_Dense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_htool_mpidense_C",MatConvert_Htool_Dense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatHtoolGetHierarchicalMat_C",MatHtoolGetHierarchicalMat_Htool);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatHtoolSetKernel_C",MatHtoolSetKernel_Htool);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatHtoolGetPermutationSource_C",MatHtoolGetPermutationSource_Htool);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatHtoolGetPermutationTarget_C",MatHtoolGetPermutationTarget_Htool);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatHtoolUsePermutation_C",MatHtoolUsePermutation_Htool);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
