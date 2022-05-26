#include <petscconf.h>
#include <petscsys.h>
#include <petsctime.h>
#include <petscdevice.h>
#include <../src/mat/impls/hypre/svalue.h>
#include <HYPRE.h>
#include <HYPRE_utilities.h>
#include <_hypre_parcsr_ls.h>
#include <_hypre_sstruct_ls.h>
#include <petscdevice.h>
#include <petscpkg_version.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/sfimpl.h>
#include <petscsystypes.h>
#include <petscerror.h>
#include <unistd.h>
#include <sys/time.h>
#include <petscconf.h>
#include <../src/mat/impls/aij/seq/aij.h>          /*I "petscmat.h" I*/
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <../src/vec/vec/impls/dvecimpl.h>
#include <petsc/private/vecimpl.h>
#include <sys/time.h>
#undef VecType
#include <../src/mat/impls/aij/seq/seqcusparse/cusparsematimpl.h>
#include <thrust/adjacent_difference.h>
#include <thrust/async/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <../src/mat/impls/hypre/svalue.h>

#define THRUSTINTARRAY32_k thrust::device_vector<int>
#define THRUSTINTARRAY_k thrust::device_vector<PetscInt>
#define THRUSTARRAY_k thrust::device_vector<PetscCount>
#define THRUSTINTARRAY_k_h thrust::host_vector<PetscInt>
#define THRUSTARRAY_k_h thrust::host_vector<PetscCount>

struct IJCompare
{
  __host__ __device__
  inline bool operator() (const thrust::tuple<PetscInt, PetscInt> &t1, const thrust::tuple<PetscInt, PetscInt> &t2)
  {
    if (t1.get<0>() < t2.get<0>()) return true;
    if (t1.get<0>() == t2.get<0>()) return t1.get<1>() < t2.get<1>();
    return false;
  }
};

struct is_less_than_zero
{
  __host__ __device__
  bool operator()(int x)
  {
    return x < 0;
  }
};

struct IJEqual
{
  __host__ __device__
  inline bool operator() (const thrust::tuple<PetscInt, PetscInt> &t1, const thrust::tuple<PetscInt, PetscInt> &t2)
  {
    if (t1.get<0>() != t2.get<0>() || t1.get<1>() != t2.get<1>()) return false;
    return true;
  }
};

struct IJDiff
{
  __host__ __device__
  inline PetscInt operator() (const PetscInt &t1, const PetscInt &t2)
  {
    return t1 == t2 ? 0 : 1;
  }
};

struct IJDValue
{
  __host__ __device__
  inline PetscInt operator() (const PetscInt &t1, const PetscInt &t2)
  {
    return t2 - t1;
  }
};


struct IJSum
{
  __host__ __device__
  inline PetscInt operator() (const PetscInt &t1, const PetscInt &t2)
  {
    return t1||t2;
  }
};
#include <../src/mat/impls/hypre/mhypre.h>

static PetscErrorCode MatHYPRE_IJMatrixPreallocate_test(PetscInt n_d,const PetscInt *nnz_d, HYPRE_IJMatrix ij)
{
  HYPRE_Int      *nnz_o=NULL;

  PetscFunctionBegin;
  //PetscCall(PetscCalloc1(n_d,&nnz_o));

#if PETSC_PKG_HYPRE_VERSION_GE(2,16,0)
  { /* If we don't do this, the columns of the matrix will be all zeros! */
    hypre_AuxParCSRMatrix *aux_matrix;
    aux_matrix = (hypre_AuxParCSRMatrix*)hypre_IJMatrixTranslator(ij);
    hypre_AuxParCSRMatrixDestroy(aux_matrix);
    hypre_IJMatrixTranslator(ij) = NULL;
    PetscStackCallStandard(HYPRE_IJMatrixSetDiagOffdSizes,ij,nnz_d,nnz_o);
    /* it seems they partially fixed it in 2.19.0 */
#if PETSC_PKG_HYPRE_VERSION_LT(2,19,0)
    aux_matrix = (hypre_AuxParCSRMatrix*)hypre_IJMatrixTranslator(ij);
    hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 1;
#endif
  }
#else
  PetscStackCallStandard(HYPRE_IJMatrixSetDiagOffdSizes,ij,nnz_d,nnz_o);
#endif
  //PetscCall(PetscFree(nnz_o));

  PetscFunctionReturn(0);
}

__global__ static void MarkDiagnoalD(PetscCount n, PetscInt *Ai, PetscInt *Aj, PetscInt* diag)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) {
    diag[i] = Ai[i+1];
    for (int j=Ai[i]; j<Ai[i+1]; j++) {
      if (Aj[j] == i) {
        diag[i] = j;
        break;
      }
    }
  }
}

static PetscErrorCode MarkDiagonal_SeqAIJ(Mat A)
{
  Mat_HYPRE      *hmat = (Mat_HYPRE*)(A->data);
  Mat_SeqAIJ_info     *a = (Mat_SeqAIJ_info*)hmat->saij;
  PetscInt       m = A->rmap->n;

  PetscFunctionBegin;
  PetscStackCall("hypre_TAlloc",hmat->diag = hypre_TAlloc(PetscInt,A->rmap->n,HYPRE_MEMORY_DEVICE));
  MarkDiagnoalD<<<(m+255)/256,256,0>>>(m, a->i, a->j, hmat->diag);

  PetscFunctionReturn(0);
}


PETSC_INTERN PetscErrorCode MatSetPreallocationCOO_SeqAIJ_hypre(Mat mat, PetscCount coo_n, const PetscInt coo_i[], const PetscInt coo_j[])
{
  MPI_Comm                  comm;
  PetscInt                  M,N;
  //PetscInt                  *Ai; /* Change to PetscCount once we use it for row pointers */
  Mat_HYPRE                 *hmat = (Mat_HYPRE*)(mat->data);
  Mat_SeqAIJ_info                *seqaij = (Mat_SeqAIJ_info*)(hmat->saij);
  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)mat,&comm));
  PetscCall(MatGetSize(mat,&M,&N));
  seqaij->singlemalloc = PETSC_FALSE;
  seqaij->free_a       = PETSC_TRUE;
  seqaij->free_ij      = PETSC_TRUE;
  THRUSTINTARRAY_k ilen(mat->rmap->n);
  THRUSTINTARRAY_k d_i(coo_n);
  THRUSTINTARRAY_k d_j(coo_n);
  THRUSTINTARRAY_k *cooPerm   = new THRUSTINTARRAY_k(coo_n);
  THRUSTINTARRAY_k *cooPerm_a = new THRUSTINTARRAY_k(coo_n+1);
  //sleep(1);
  d_i.assign(coo_i,coo_i+coo_n);
  d_j.assign(coo_j,coo_j+coo_n);
  thrust::replace_if(thrust::device, d_i.begin(), d_i.end(), d_j.begin(), is_less_than_zero(), -1);
  auto fkey = thrust::make_zip_iterator(thrust::make_tuple(d_i.begin(),d_j.begin()));
  auto ekey = thrust::make_zip_iterator(thrust::make_tuple(d_i.end(),d_j.end()));
  thrust::sequence(thrust::device, cooPerm->begin(), cooPerm->end(), 0);
  thrust::sort_by_key(fkey, ekey, cooPerm->begin(), IJCompare()); /* sort by row, then by col */
  *cooPerm_a = d_i; /* copy the sorted array */
  THRUSTINTARRAY_k w = d_j;
  THRUSTINTARRAY_k ii(mat->rmap->n+1);
  auto nekey = thrust::unique(fkey, ekey, IJEqual()); /* unique (d_i, d_j) */
  adjacent_difference(cooPerm_a->begin(),cooPerm_a->end(),cooPerm_a->begin(),IJDiff()); /* cooPerm_a: [1,1,3,3,4,4] => [1,0,1,0,1,0]*/
  adjacent_difference(w.begin(),w.end(),w.begin(),IJDiff());                                              /* w:         [2,2,2,3,5,6] => [2,0,0,1,1,1]*/
  (*cooPerm_a)[0] = 0; /* clear the first entry, though accessing an entry on device implies a cudaMemcpy */
  w[0] = 0;
  thrust::transform(cooPerm_a->begin(),cooPerm_a->end(),w.begin(),cooPerm_a->begin(),IJSum()); /* cooPerm_a =          [0,0,1,1,1,1]*/
  cooPerm_a->push_back(1);
  thrust::inclusive_scan(cooPerm_a->begin(),cooPerm_a->end(),cooPerm_a->begin(),thrust::plus<PetscInt>()); /*cooPerm_a=[0,0,1,2,3,4]*/
  thrust::counting_iterator<PetscInt> search_begin(0);
  thrust::upper_bound(d_i.begin(), nekey.get_iterator_tuple().get<0>(), /* binary search entries of [0,1,2,3,4,5,6) in ordered array d_i = [1,3,3,4,4], supposing A->rmap->n = 6. */
                      search_begin, search_begin + mat->rmap->n,  /* return in ii[] the index of last position in d_i[] where value could be inserted without violating the ordering */
                      ii.begin()+1); /* ii = [0,1,1,3,5,5]. A leading 0 will be added later */
  thrust::transform(ii.begin(),ii.end()-1,ii.begin()+1,ilen.begin(),IJDValue()); /* cooPerm_a =          [0,0,1,1,1,1]*/

#if 1
    ii[0] = 0;
    PetscCallCUDA(cudaMalloc(&seqaij->i,(mat->rmap->n+1)*sizeof(PetscInt)));
    PetscCallCUDA(cudaMemcpy(seqaij->i,ii.data().get(),(mat->rmap->n+1)*sizeof(PetscInt),cudaMemcpyHostToDevice));
    //seqaij->i = ii.data().get();
    if (!seqaij->ilen) PetscCall(PetscMalloc1(mat->rmap->n,&seqaij->ilen));

    seqaij->nz = seqaij->maxnz = ii[mat->rmap->n];
    seqaij->rmax = 0;
    seqaij->j= d_j.data().get();
    PetscCallCUDA(cudaMemcpy(seqaij->ilen,ilen.data().get(),(mat->rmap->n)*sizeof(PetscInt),cudaMemcpyDeviceToHost));
    mat->preallocated = PETSC_TRUE;
#else
  PetscCall(MatSetSeqAIJWithArrays_private(PETSC_COMM_SELF,M,N,Ai,Aj,Aa,rtype,mat));
#endif
  /* Record COO fields */
  seqaij->coo_n        = coo_n;
  seqaij->Atot         = coo_n; 

  seqaij->jmap         = cooPerm_a->data().get(); /* of length nnz+1 */
  seqaij->perm         = cooPerm->data().get();

  PetscBool    sameint = (PetscBool)(sizeof(PetscInt) == sizeof(HYPRE_Int));

    /* Copy the sparsity pattern from cooMat to hypre IJMatrix hmat->ij */
    PetscCall(MatSetOption(mat,MAT_SORTED_FULL,PETSC_TRUE));
    PetscCall(MatSetOption(mat,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE));
    PetscStackCallStandard(HYPRE_IJMatrixCreate,hmat->comm,mat->rmap->rstart,mat->rmap->rend-1,mat->cmap->rstart,mat->cmap->rend-1,&hmat->ij);
    PetscStackCallStandard(HYPRE_IJMatrixSetObjectType,hmat->ij,HYPRE_PARCSR);
   PetscCall(MarkDiagonal_SeqAIJ(mat)); /* We need updated diagonal positions */

    PetscCall(MatHYPRE_IJMatrixPreallocate_test(mat->rmap->n,seqaij->ilen,hmat->ij));
  hypre_ParCSRMatrix    *par_matrix;
  hypre_AuxParCSRMatrix *aux_matrix;
  hypre_CSRMatrix       *hdiag;
  
#if PETSC_PKG_HYPRE_VERSION_LT(2,19,0)
      PetscStackCallStandard(HYPRE_IJMatrixInitialize,hmat->ij);
#else
      PetscStackCallStandard(HYPRE_IJMatrixInitialize_v2,hmat->ij,HYPRE_MEMORY_HOST);
#endif
     PetscStackCallStandard(HYPRE_IJMatrixGetObject,hmat->ij,(void**)&par_matrix);
     hdiag = hypre_ParCSRMatrixDiag(par_matrix);
     /*
          this is the Hack part where we monkey directly with the hypre datastructures
     */
     
    if (sameint) {
       PetscCallCUDA(cudaMemcpy(hdiag->i,ii.data().get(),(mat->rmap->n+1)*sizeof(PetscInt),cudaMemcpyDeviceToHost));
       PetscCallCUDA(cudaMemcpy(hdiag->j,d_j.data().get(),seqaij->nz*sizeof(PetscInt),cudaMemcpyDeviceToHost));
     } 
     
     aux_matrix = (hypre_AuxParCSRMatrix*)hypre_IJMatrixTranslator(hmat->ij);
     hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0;

    mat->preallocated = PETSC_TRUE;

  PetscFunctionReturn(0);
}

__global__ static void setvalueD(PetscCount n, PetscInt *jmap, PetscInt *perm, MatScalar* aa, const PetscScalar *kv)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) {
    for (PetscCount k=jmap[i]; k<jmap[i+1]; k++) 
      aa[i] = kv[perm[k]];
  }
}

__global__ static void setdiagonalD(PetscCount n,PetscInt *Ai, MatScalar *Aa, PetscInt* Adiag)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  PetscScalar tmp;
  if (i < n) {
      if (Adiag[i] >= Ai[i] && Adiag[i] < Ai[i+1]) { /* The diagonal element exists */
        tmp          = Aa[Ai[i]];
        Aa[Ai[i]]    = Aa[Adiag[i]];
        Aa[Adiag[i]] = tmp;
      }
  }
}

PETSC_INTERN PetscErrorCode setdiagonal(Mat mat, PetscInt *diag){
  Mat_HYPRE      *hmat = (Mat_HYPRE*)mat->data;
  Mat_SeqAIJ_info     *aseq = hmat->saij;
  PetscFunctionBegin;
  setdiagonalD<<<(mat->rmap->n+255)/256,256,0>>>(mat->rmap->n, aseq->i, aseq->a, diag);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode setCOOValue(Mat mat, const PetscScalar v[], InsertMode imode){
  Mat_HYPRE      *hmat = (Mat_HYPRE*)mat->data;
  Mat_SeqAIJ_info     *aseq = hmat->saij;
  PetscCount     Annz = aseq->nz;
  PetscScalar    *kv;
  PetscMemType   memtype;
  PetscFunctionBegin;
  PetscCall(PetscGetMemType(v,&memtype));
  if (PetscMemTypeHost(memtype)) { /* If user gave v[] in host, we might need to copy it to device if any */
    PetscCallCUDA(cudaMalloc(&kv,Annz*sizeof(PetscScalar)));
    PetscCallCUDA(cudaMemcpy(kv,v,Annz*sizeof(PetscScalar),cudaMemcpyHostToDevice));
    setvalueD<<<(aseq->nz+255)/256,256,0>>>(Annz, aseq->jmap, aseq->perm, aseq->a, kv);
  } else {
    setvalueD<<<(aseq->nz+255)/256,256,0>>>(Annz, aseq->jmap, aseq->perm, aseq->a, v);
  }
  if (PetscMemTypeHost(memtype)) 
    PetscCallCUDA(cudaFree(kv));
  PetscFunctionReturn(0);
}

