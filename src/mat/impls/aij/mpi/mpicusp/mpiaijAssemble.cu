#include "petscconf.h"
PETSC_CUDA_EXTERN_C_BEGIN
#include "../src/mat/impls/aij/seq/aij.h"          /*I "petscmat.h" I*/
#include "petscbt.h"
#include "../src/vec/vec/impls/dvecimpl.h"
#include "private/vecimpl.h"
PETSC_CUDA_EXTERN_C_END
#undef VecType
#include "../src/mat/impls/aij/seq/seqcusp/cuspmatimpl.h"

#include <thrust/reduce.h>
#include <thrust/inner_product.h>

#include <cusp/array1d.h>
#include <cusp/print.h>
#include <cusp/coo_matrix.h>
#include <cusp/detail/device/reduce_by_key.h>

#include <cusp/io/matrix_market.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/partition.h>
#include <thrust/remove.h>

// this example illustrates how to make repeated access to a range of values
// examples:
//   repeated_range([0, 1, 2, 3], 1) -> [0, 1, 2, 3]
//   repeated_range([0, 1, 2, 3], 2) -> [0, 0, 1, 1, 2, 2, 3, 3]
//   repeated_range([0, 1, 2, 3], 3) -> [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
//   ...

template <typename Iterator>
class repeated_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct repeat_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type repeats;

        repeat_functor(difference_type repeats)
            : repeats(repeats) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        {
            return i / repeats;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<repeat_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the repeated_range iterator
    typedef PermutationIterator iterator;

    // construct repeated_range for the range [first,last)
    repeated_range(Iterator first, Iterator last, difference_type repeats)
        : first(first), last(last), repeats(repeats) {}

    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), repeat_functor(repeats)));
    }

    iterator end(void) const
    {
        return begin() + repeats * (last - first);
    }

    protected:
    difference_type repeats;
    Iterator first;
    Iterator last;

};

// this example illustrates how to repeat blocks in a range multiple times
// examples:
//   tiled_range([0, 1, 2, 3], 2)    -> [0, 1, 2, 3, 0, 1, 2, 3]
//   tiled_range([0, 1, 2, 3], 4, 2) -> [0, 1, 2, 3, 0, 1, 2, 3]
//   tiled_range([0, 1, 2, 3], 2, 2) -> [0, 1, 0, 1, 2, 3, 2, 3]
//   tiled_range([0, 1, 2, 3], 2, 3) -> [0, 1, 0, 1 0, 1, 2, 3, 2, 3, 2, 3]
//   ...

template <typename Iterator>
class tiled_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct tile_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type repeats;
        difference_type tile_size;

        tile_functor(difference_type repeats, difference_type tile_size)
            : tile_size(tile_size), repeats(repeats) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        {
            return tile_size * (i / (tile_size * repeats)) + i % tile_size;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<tile_functor, CountingIterator>   TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the tiled_range iterator
    typedef PermutationIterator iterator;

    // construct repeated_range for the range [first,last)
    tiled_range(Iterator first, Iterator last, difference_type repeats)
        : first(first), last(last), repeats(repeats), tile_size(last - first) {}

    tiled_range(Iterator first, Iterator last, difference_type repeats, difference_type tile_size)
        : first(first), last(last), repeats(repeats), tile_size(tile_size)
    {
      // ASSERT((last - first) % tile_size == 0)
    }

    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), tile_functor(repeats, tile_size)));
    }

    iterator end(void) const
    {
        return begin() + repeats * (last - first);
    }

    protected:
    difference_type repeats;
    difference_type tile_size;
    Iterator first;
    Iterator last;
};

typedef cusp::device_memory memSpace;
typedef int   IndexType;
typedef float ValueType;
typedef cusp::array1d<IndexType, memSpace> IndexArray;
typedef cusp::array1d<ValueType, memSpace> ValueArray;
typedef cusp::array1d<IndexType, cusp::host_memory> IndexHostArray;
typedef IndexArray::iterator IndexArrayIterator;
typedef ValueArray::iterator ValueArrayIterator;

struct is_diag
{
  IndexType first, last;

  is_diag(IndexType first, IndexType last) : first(first), last(last) {};

  template <typename Tuple>
  __host__ __device__
  bool operator()(Tuple t) {
    // Check column
    const IndexType row = thrust::get<0>(t);
    const IndexType col = thrust::get<1>(t);
    return (row >= first) && (row < last) && (col >= first) && (col < last);
  }
};

struct is_nonlocal
{
  IndexType first, last;

  is_nonlocal(IndexType first, IndexType last) : first(first), last(last) {};

  template <typename Tuple>
  __host__ __device__
  bool operator()(Tuple t) {
    // Check column
    const IndexType row = thrust::get<0>(t);
    return (row < first) || (row >= last);
  }
};

EXTERN_C_BEGIN
// Ne: Number of elements
// Nl: Number of dof per element
// Nr: Number of matrix rows (dof)
#undef __FUNCT__
#define __FUNCT__ "MatMPIAIJSetValuesBatch"
PetscErrorCode MatMPIAIJSetValuesBatch(Mat J, PetscInt Ne, PetscInt Nl, PetscInt *elemRows, PetscScalar *elemMats)
{
  MPI_Comm        comm = ((PetscObject) J)->comm;
  size_t          N    = Ne * Nl;
  size_t          No   = Ne * Nl*Nl;
  const PetscInt *rowRanges;
  PetscInt       *offDiagRows;
  PetscInt        numNonlocalRows, numSendEntries, numRecvEntries;
  PetscInt        Nr, Nc, firstRow, lastRow, firstCol;
  PetscMPIInt     numProcs, rank;
  PetscErrorCode  ierr;

  // copy elemRows and elemMat to device
  IndexArray d_elemRows(elemRows, elemRows + N);
  ValueArray d_elemMats(elemMats, elemMats + No);

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  // get matrix information
  ierr = MatGetLocalSize(J, &Nr, PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(J, &firstRow, &lastRow);CHKERRQ(ierr);
  ierr = MatGetOwnershipRanges(J, &rowRanges);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(J, &firstCol, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscInfo3(J, "Making matrix of size %d (rows %d -- %d)\n", Nr, firstRow, lastRow);CHKERRQ(ierr);

  // repeat elemRows entries Nl times
  ierr = PetscInfo(J, "Making row indices\n");CHKERRQ(ierr);
  repeated_range<IndexArrayIterator> rowInd(d_elemRows.begin(), d_elemRows.end(), Nl);

  // tile rows of elemRows Nl times
  ierr = PetscInfo(J, "Making column indices\n");CHKERRQ(ierr);
  tiled_range<IndexArrayIterator> colInd(d_elemRows.begin(), d_elemRows.end(), Nl, Nl);

  // Find number of nonlocal rows
  // TODO: Ask Nathan how to do this on GPU
  ierr = PetscMalloc(N * sizeof(PetscInt), &offDiagRows);CHKERRQ(ierr);
  numNonlocalRows = 0;
  for(PetscInt i = 0; i < N; ++i) {
    const PetscInt row = elemRows[i];
    if ((row < firstRow) || (row >= lastRow)) {
      offDiagRows[numNonlocalRows++] = row;
    }
  }
  numSendEntries  = numNonlocalRows*Nl;
  ierr = PetscInfo2(J, "Nonlocal rows %d total entries %d\n", numNonlocalRows, No);CHKERRQ(ierr);
  // Convert rows to procs
  // TODO: Ask Nathan how to do this on GPU
  //   Count up entries going to each proc (convert rows to procs and sum)
  //   send sizes of off-proc entries (could send diag and offdiag sizes)
  PetscInt      *procSendSizes, *procRecvSizes;
  ierr = PetscMalloc2(numProcs, PetscInt, &procSendSizes, numProcs, PetscInt, &procRecvSizes);CHKERRQ(ierr);
  ierr = PetscMemzero(procSendSizes, numProcs * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(procRecvSizes, numProcs * sizeof(PetscInt));CHKERRQ(ierr);
  for(PetscInt i = 0; i < numNonlocalRows; ++i) {
    const IndexType row = offDiagRows[i];
    for(IndexType p = 0; p < numProcs; ++p) {
      if ((row >= rowRanges[p]) && (row < rowRanges[p+1])) {
        procSendSizes[p] += Nl;
        break;
      }
    }
  }
  ierr = MPI_Alltoall(procSendSizes, 1, MPIU_INT, procRecvSizes, 1, MPIU_INT, comm);CHKERRQ(ierr);
  numRecvEntries = 0;
  for(PetscInt p = 0; p < numProcs; ++p) {
    numRecvEntries += procRecvSizes[p];
  }
  ierr = PetscPrintf(PETSC_COMM_SELF, "[%d]Send entries %d Recv Entries %d\n", rank, numSendEntries, numRecvEntries);CHKERRQ(ierr);
  // Allocate storage for "fat" COO representation of matrix
  ierr = PetscInfo2(J, "Making COO matrices, diag entries %d, nondiag entries %d\n", No-numSendEntries+numRecvEntries, numSendEntries*2);CHKERRQ(ierr);
  cusp::coo_matrix<IndexType,ValueType, memSpace> diagCOO(Nr, Nr, No-numSendEntries+numRecvEntries); // TODO: Currently oversized
  IndexArray nondiagonalRows(numSendEntries*2); // TODO: Currently oversized
  IndexArray nondiagonalCols(numSendEntries*2); // TODO: Currently oversized
  ValueArray nondiagonalVals(numSendEntries*2); // TODO: Currently oversized
  IndexArray nonlocalRows(numSendEntries);
  IndexArray nonlocalCols(numSendEntries);
  ValueArray nonlocalVals(numSendEntries);
  // partition entries into diagonal and off-diagonal+off-proc
  ierr = PetscInfo(J, "Splitting into diagonal and off-diagonal+off-proc\n");CHKERRQ(ierr);
  thrust::fill(diagCOO.row_indices.begin(), diagCOO.row_indices.end(), -1);
  thrust::fill(nondiagonalRows.begin(),     nondiagonalRows.end(),     -1);
  thrust::partition_copy(thrust::make_zip_iterator(thrust::make_tuple(rowInd.begin(), colInd.begin(), d_elemMats.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(rowInd.end(),   colInd.end(),   d_elemMats.end())),
                         thrust::make_zip_iterator(thrust::make_tuple(diagCOO.row_indices.begin(),    diagCOO.column_indices.begin(), diagCOO.values.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(nondiagonalRows.begin(),        nondiagonalCols.begin(),        nondiagonalVals.begin())),
                         is_diag(firstRow, lastRow));
  // Current size without off-proc entries
  PetscInt diagonalSize    = (diagCOO.row_indices.end() - diagCOO.row_indices.begin()) - thrust::count(diagCOO.row_indices.begin(), diagCOO.row_indices.end(), -1);
  PetscInt nondiagonalSize = No - diagonalSize;
  ierr = PetscPrintf(PETSC_COMM_SELF, "[%d]Diagonal size %d Nondiagonal size %d\n", rank, diagonalSize, nondiagonalSize);CHKERRQ(ierr);
  cusp::print(diagCOO);
  cusp::print(nondiagonalRows);
  // partition again into off-diagonal and off-proc
  ierr = PetscInfo(J, "Splitting into off-diagonal and off-proc\n");CHKERRQ(ierr);
  thrust::stable_partition(thrust::make_zip_iterator(thrust::make_tuple(nondiagonalRows.begin(), nondiagonalCols.begin(), nondiagonalVals.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(nondiagonalRows.end(),   nondiagonalCols.end(),   nondiagonalVals.end())),
                    is_nonlocal(firstRow, lastRow));
  PetscInt nonlocalSize    = numSendEntries;
  PetscInt offdiagonalSize = nondiagonalSize - nonlocalSize;
  ierr = PetscPrintf(PETSC_COMM_SELF, "[%d]Nonlocal size %d Offdiagonal size %d\n", rank, nonlocalSize, offdiagonalSize);CHKERRQ(ierr);
  cusp::print(nondiagonalRows);
  // send off-proc entries (pack this up later)
  PetscInt    *procSendDispls, *procRecvDispls;
  PetscInt    *sendRows, *recvRows;
  PetscInt    *sendCols, *recvCols;
  PetscScalar *sendVals, *recvVals;
  ierr = PetscMalloc2(numProcs, PetscInt, &procSendDispls, numProcs, PetscInt, &procRecvDispls);CHKERRQ(ierr);
  ierr = PetscMalloc3(numSendEntries, PetscInt, &sendRows, numSendEntries, PetscInt, &sendCols, numSendEntries, PetscScalar, &sendVals);CHKERRQ(ierr);
  ierr = PetscMalloc3(numRecvEntries, PetscInt, &recvRows, numRecvEntries, PetscInt, &recvCols, numRecvEntries, PetscScalar, &recvVals);CHKERRQ(ierr);
  procSendDispls[0] = procRecvDispls[0] = 0;
  for(PetscInt p = 1; p < numProcs; ++p) {
    procSendDispls[p] = procSendDispls[p-1] + procSendSizes[p-1];
    procRecvDispls[p] = procRecvDispls[p-1] + procRecvSizes[p-1];
  }
  thrust::copy(nondiagonalRows.begin(), nondiagonalRows.begin()+nonlocalSize, sendRows);
  thrust::copy(nondiagonalCols.begin(), nondiagonalCols.begin()+nonlocalSize, sendCols);
  thrust::copy(nondiagonalVals.begin(), nondiagonalVals.begin()+nonlocalSize, sendVals);
  ierr = MPI_Alltoallv(sendRows, procSendSizes, procSendDispls, MPIU_INT,    recvRows, procRecvSizes, procRecvDispls, MPIU_INT,    comm);CHKERRQ(ierr);
  ierr = MPI_Alltoallv(sendCols, procSendSizes, procSendDispls, MPIU_INT,    recvCols, procRecvSizes, procRecvDispls, MPIU_INT,    comm);CHKERRQ(ierr);
  ierr = MPI_Alltoallv(sendVals, procSendSizes, procSendDispls, MPIU_SCALAR, recvVals, procRecvSizes, procRecvDispls, MPIU_SCALAR, comm);CHKERRQ(ierr);
  ierr = PetscFree2(procSendDispls, procRecvDispls);CHKERRQ(ierr);
  ierr = PetscFree3(sendRows, sendCols, sendVals);CHKERRQ(ierr);
  // Create off-diagonal matrix
  cusp::coo_matrix<IndexType,ValueType, memSpace> offdiagCOO(Nr, Nr, offdiagonalSize+numRecvEntries);
  // partition again into diagonal and off-diagonal
  IndexArray d_recvRows(recvRows, recvRows+numRecvEntries);
  IndexArray d_recvCols(recvCols, recvCols+numRecvEntries);
  ValueArray d_recvVals(recvVals, recvVals+numRecvEntries);
  thrust::copy(nondiagonalRows.end()-offdiagonalSize, nondiagonalRows.end(), offdiagCOO.row_indices.begin());
  thrust::copy(nondiagonalCols.end()-offdiagonalSize, nondiagonalCols.end(), offdiagCOO.column_indices.begin());
  thrust::copy(nondiagonalVals.end()-offdiagonalSize, nondiagonalVals.end(), offdiagCOO.values.begin());
  thrust::fill(diagCOO.row_indices.begin()+diagonalSize,       diagCOO.row_indices.end(),    -1);
  thrust::fill(offdiagCOO.row_indices.begin()+offdiagonalSize, offdiagCOO.row_indices.end(), -1);
  thrust::partition_copy(thrust::make_zip_iterator(thrust::make_tuple(d_recvRows.begin(), d_recvCols.begin(), d_recvVals.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(d_recvRows.end(),   d_recvCols.end(),   d_recvVals.end())),
                         thrust::make_zip_iterator(thrust::make_tuple(diagCOO.row_indices.begin()+diagonalSize, diagCOO.column_indices.begin()+diagonalSize, diagCOO.values.begin()+diagonalSize)),
                         thrust::make_zip_iterator(thrust::make_tuple(offdiagCOO.row_indices.begin()+offdiagonalSize, offdiagCOO.column_indices.begin()+offdiagonalSize, offdiagCOO.values.begin()+offdiagonalSize)),
                         is_diag(firstRow, lastRow));
  ierr = PetscFree3(recvRows, recvCols, recvVals);CHKERRQ(ierr);
  diagonalSize    = (diagCOO.row_indices.end()    - diagCOO.row_indices.begin())    - thrust::count(diagCOO.row_indices.begin(),    diagCOO.row_indices.end(),    -1);
  offdiagonalSize = (offdiagCOO.row_indices.end() - offdiagCOO.row_indices.begin()) - thrust::count(offdiagCOO.row_indices.begin(), offdiagCOO.row_indices.end(), -1);

  // sort COO format by (i,j), this is the most costly step
  ierr = PetscInfo(J, "Sorting rows and columns\n");CHKERRQ(ierr);
  diagCOO.sort_by_row_and_column();
  offdiagCOO.sort_by_row_and_column();
  PetscInt diagonalOffset    = (diagCOO.row_indices.end()    - diagCOO.row_indices.begin())    - diagonalSize;
  PetscInt offdiagonalOffset = (offdiagCOO.row_indices.end() - offdiagCOO.row_indices.begin()) - offdiagonalSize;

  // print the "fat" COO representation
  if (PetscLogPrintInfo) {
    cusp::print(diagCOO);
    cusp::print(offdiagCOO);
  }

  // Figure out the number of unique off-diagonal columns
  Nc = thrust::inner_product(offdiagCOO.column_indices.begin()+offdiagonalOffset,
                             offdiagCOO.column_indices.end()   - 1,
                             offdiagCOO.column_indices.begin()+offdiagonalOffset + 1,
                             size_t(1), thrust::plus<size_t>(), thrust::not_equal_to<IndexType>());

  // compute number of unique (i,j) entries
  //   this counts the number of changes as we move along the (i,j) list
  ierr = PetscInfo(J, "Computing number of unique entries\n");CHKERRQ(ierr);
  size_t num_diag_entries = thrust::inner_product
    (thrust::make_zip_iterator(thrust::make_tuple(diagCOO.row_indices.begin(), diagCOO.column_indices.begin())) + diagonalOffset,
     thrust::make_zip_iterator(thrust::make_tuple(diagCOO.row_indices.end(),   diagCOO.column_indices.end())) - 1,
     thrust::make_zip_iterator(thrust::make_tuple(diagCOO.row_indices.begin(), diagCOO.column_indices.begin())) + diagonalOffset + 1,
     size_t(1),
     thrust::plus<size_t>(),
     thrust::not_equal_to< thrust::tuple<IndexType,IndexType> >());
  size_t num_offdiag_entries = thrust::inner_product
    (thrust::make_zip_iterator(thrust::make_tuple(offdiagCOO.row_indices.begin(), offdiagCOO.column_indices.begin())) + offdiagonalOffset,
     thrust::make_zip_iterator(thrust::make_tuple(offdiagCOO.row_indices.end(),   offdiagCOO.column_indices.end())) - 1,
     thrust::make_zip_iterator(thrust::make_tuple(offdiagCOO.row_indices.begin(), offdiagCOO.column_indices.begin())) + offdiagonalOffset + 1,
     size_t(1),
     thrust::plus<size_t>(),
     thrust::not_equal_to< thrust::tuple<IndexType,IndexType> >());

  // allocate COO storage for final matrix
  ierr = PetscInfo(J, "Allocating compressed matrices\n");CHKERRQ(ierr);
  cusp::coo_matrix<IndexType, ValueType, memSpace> A(Nr, Nr, num_diag_entries);
  cusp::coo_matrix<IndexType, ValueType, memSpace> B(Nr, Nc, num_offdiag_entries);

  // sum values with the same (i,j) index
  // XXX thrust::reduce_by_key is unoptimized right now, so we provide a SpMV-based one in cusp::detail
  //     the Cusp one is 2x faster, but still not optimal
  // This could possibly be done in-place
  ierr = PetscInfo(J, "Compressing matrices\n");CHKERRQ(ierr);
  cusp::detail::device::reduce_by_key
    (thrust::make_zip_iterator(thrust::make_tuple(diagCOO.row_indices.begin(), diagCOO.column_indices.begin())) + diagonalOffset,
     thrust::make_zip_iterator(thrust::make_tuple(diagCOO.row_indices.end(),   diagCOO.column_indices.end())),
     diagCOO.values.begin() + diagonalOffset,
     thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin())),
     A.values.begin(),
     thrust::equal_to< thrust::tuple<IndexType,IndexType> >(),
     thrust::plus<ValueType>());
  cusp::detail::device::reduce_by_key
    (thrust::make_zip_iterator(thrust::make_tuple(offdiagCOO.row_indices.begin(), offdiagCOO.column_indices.begin())) + offdiagonalOffset,
     thrust::make_zip_iterator(thrust::make_tuple(offdiagCOO.row_indices.end(),   offdiagCOO.column_indices.end())),
     offdiagCOO.values.begin() + offdiagonalOffset,
     thrust::make_zip_iterator(thrust::make_tuple(B.row_indices.begin(), B.column_indices.begin())),
     B.values.begin(),
     thrust::equal_to< thrust::tuple<IndexType,IndexType> >(),
     thrust::plus<ValueType>());

  // Convert row and column numbers
  if (firstRow) {
    thrust::transform(A.row_indices.begin(), A.row_indices.end(), thrust::make_constant_iterator(firstRow), A.row_indices.begin(), thrust::minus<IndexType>());
    thrust::transform(B.row_indices.begin(), B.row_indices.end(), thrust::make_constant_iterator(firstRow), B.row_indices.begin(), thrust::minus<IndexType>());
  }
  if (firstCol) {
    thrust::transform(A.column_indices.begin(), A.column_indices.end(), thrust::make_constant_iterator(firstCol), A.column_indices.begin(), thrust::minus<IndexType>());
  }
  //   TODO: Get better code from Nathan
  IndexArray d_colmap(Nc);
  thrust::unique_copy(B.column_indices.begin(), B.column_indices.end(), d_colmap.begin());
  IndexHostArray colmap(d_colmap.begin(), d_colmap.end());
  IndexType      newCol = 0;
  for(IndexHostArray::iterator c_iter = colmap.begin(); c_iter != colmap.end(); ++c_iter, ++newCol) {
    thrust::replace(B.column_indices.begin(), B.column_indices.end(), *c_iter, newCol);
  }

  // print the final matrix
  if (PetscLogPrintInfo) {
    cusp::print(A);
    cusp::print(B);
  }

  ierr = PetscInfo(J, "Converting to PETSc matrix\n");CHKERRQ(ierr);
  ierr = MatSetType(J, MATMPIAIJCUSP);CHKERRQ(ierr);
  //cusp::csr_matrix<PetscInt,PetscScalar,cusp::device_memory> Jgpu;
  CUSPMATRIX *Jgpu = new CUSPMATRIX;
  CUSPMATRIX *Kgpu = new CUSPMATRIX;
  cusp::convert(A, *Jgpu);
  cusp::convert(B, *Kgpu);
  if (PetscLogPrintInfo) {
    cusp::print(*Jgpu);
    cusp::print(*Kgpu);
  }
#if 0
  ierr = PetscInfo(J, "Copying to CPU matrix");CHKERRQ(ierr);
  ierr = MatCUSPCopyFromGPU(J, Jgpu, Kgpu);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
EXTERN_C_END
