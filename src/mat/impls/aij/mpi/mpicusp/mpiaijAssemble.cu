#include "petscconf.h"
PETSC_CUDA_EXTERN_C_BEGIN
#include "../src/mat/impls/aij/seq/aij.h"          /*I "petscmat.h" I*/
#include "../src/mat/impls/aij/mpi/mpiaij.h"
#include "petscbt.h"
#include "../src/vec/vec/impls/dvecimpl.h"
#include "petsc-private/vecimpl.h"
PETSC_CUDA_EXTERN_C_END
#undef VecType
#include "../src/mat/impls/aij/seq/seqcusp/cuspmatimpl.h"

#include <thrust/reduce.h>
#include <thrust/inner_product.h>

#include <cusp/array1d.h>
#include <cusp/print.h>
#include <cusp/coo_matrix.h>

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
typedef PetscScalar ValueType;
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
    IndexType row = thrust::get<0>(t);
    IndexType col = thrust::get<1>(t);
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
    IndexType row = thrust::get<0>(t);
    return (row < first) || (row >= last);
  }
};

/*@C
  MatMPIAIJSetValuesBatch - Set multiple blocks of values into a matrix

  Not collective

  Input Parameters:
+ J  - the assembled Mat object
. Ne -  the number of blocks (elements)
. Nl -  the block size (number of dof per element)
. elemRows - List of block row indices, in bunches of length Nl
- elemMats - List of block values, in bunches of Nl*Nl

  Level: advanced

.seealso MatSetValues()
@*/
#undef __FUNCT__
#define __FUNCT__ "MatSetValuesBatch_MPIAIJCUSP"
PetscErrorCode MatSetValuesBatch_MPIAIJCUSP(Mat J, PetscInt Ne, PetscInt Nl, PetscInt *elemRows, const PetscScalar *elemMats)
{
  // Assumptions:
  //   1) Each elemMat is square, of size Nl x Nl
  //
  //      This means that any nonlocal entry (i,j) where i is owned by another process is matched to
  //        a) an offdiagonal entry (j,i) if j is diagonal, or
  //        b) another nonlocal entry (j,i) if j is offdiagonal
  //
  //      No - numSendEntries: The number of on-process  diagonal+offdiagonal entries
  //      numRecvEntries:      The number of off-process diagonal+offdiagonal entries
  //
  //  Glossary:
  //     diagonal: (i,j) such that i,j in [firstRow, lastRow)
  //  offdiagonal: (i,j) such that i in [firstRow, lastRow), and j not in [firstRow, lastRow)
  //     nonlocal: (i,j) such that i not in [firstRow, lastRow)
  //  nondiagonal: (i,j) such that i not in [firstRow, lastRow), or j not in [firstRow, lastRow)
  //   on-process: entries provided by elemMats
  //  off-process: entries received from other processes
  MPI_Comm        comm = ((PetscObject) J)->comm;
  Mat_MPIAIJ     *j    = (Mat_MPIAIJ *) J->data;
  size_t          N    = Ne * Nl;    // Length of elemRows (dimension of unassembled space)
  size_t          No   = Ne * Nl*Nl; // Length of elemMats (total number of values)
  PetscInt        Nr;                // Size of J          (dimension of assembled space)
  PetscInt        firstRow, lastRow, firstCol;
  const PetscInt *rowRanges;
  PetscInt        numNonlocalRows;   // Number of rows in elemRows not owned by this process
  PetscInt        numSendEntries;    // Number of (i,j,v) entries sent to other processes
  PetscInt        numRecvEntries;    // Number of (i,j,v) entries received from other processes
  PetscInt        Nc;
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
  ierr = PetscInfo3(J, "Assembling matrix of size %d (rows %d -- %d)\n", Nr, firstRow, lastRow);CHKERRQ(ierr);

  // repeat elemRows entries Nl times
  ierr = PetscInfo(J, "Making row indices\n");CHKERRQ(ierr);
  repeated_range<IndexArrayIterator> rowInd(d_elemRows.begin(), d_elemRows.end(), Nl);

  // tile rows of elemRows Nl times
  ierr = PetscInfo(J, "Making column indices\n");CHKERRQ(ierr);
  tiled_range<IndexArrayIterator> colInd(d_elemRows.begin(), d_elemRows.end(), Nl, Nl);

  // Find number of nonlocal rows, convert nonlocal rows to procs, and send sizes of off-proc entries (could send diag and offdiag sizes)
  // TODO: Ask Nathan how to do this on GPU
  ierr = PetscLogEventBegin(MAT_SetValuesBatchI,0,0,0,0);CHKERRQ(ierr);
  PetscMPIInt *procSendSizes, *procRecvSizes;
  ierr = PetscMalloc2(numProcs, PetscMPIInt, &procSendSizes, numProcs, PetscMPIInt, &procRecvSizes);CHKERRQ(ierr);
  ierr = PetscMemzero(procSendSizes, numProcs * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(procRecvSizes, numProcs * sizeof(PetscInt));CHKERRQ(ierr);
  numNonlocalRows = 0;
  for (size_t i = 0; i < N; ++i) {
    const PetscInt row = elemRows[i];

    if ((row < firstRow) || (row >= lastRow)) {
      numNonlocalRows++;
      for (IndexType p = 0; p < numProcs; ++p) {
        if ((row >= rowRanges[p]) && (row < rowRanges[p+1])) {
          procSendSizes[p] += Nl;
          break;
        }
      }
    }
  }
  numSendEntries = numNonlocalRows*Nl;
  ierr = PetscInfo2(J, "Nonlocal rows %d total entries %d\n", numNonlocalRows, No);CHKERRQ(ierr);
  ierr = MPI_Alltoall(procSendSizes, 1, MPIU_INT, procRecvSizes, 1, MPIU_INT, comm);CHKERRQ(ierr);
  numRecvEntries = 0;
  for (PetscInt p = 0; p < numProcs; ++p) {
    numRecvEntries += procRecvSizes[p];
  }
  ierr = PetscInfo2(j->A, "Send entries %d Recv Entries %d\n", numSendEntries, numRecvEntries);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_SetValuesBatchI,0,0,0,0);CHKERRQ(ierr);
  // Allocate storage for "fat" COO representation of matrix
  ierr = PetscLogEventBegin(MAT_SetValuesBatchII,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscInfo2(j->A, "Making COO matrices, diag entries %d, nondiag entries %d\n", No-numSendEntries+numRecvEntries, numSendEntries*2);CHKERRQ(ierr);
  cusp::coo_matrix<IndexType,ValueType, memSpace> diagCOO(Nr, Nr, No-numSendEntries+numRecvEntries); // ALLOC: This is oversized because I also count offdiagonal entries
  IndexArray nondiagonalRows(numSendEntries+numSendEntries); // ALLOC: This is oversized because numSendEntries > on-process offdiagonal entries
  IndexArray nondiagonalCols(numSendEntries+numSendEntries); // ALLOC: This is oversized because numSendEntries > on-process offdiagonal entries
  ValueArray nondiagonalVals(numSendEntries+numSendEntries); // ALLOC: This is oversized because numSendEntries > on-process offdiagonal entries
  IndexArray nonlocalRows(numSendEntries);
  IndexArray nonlocalCols(numSendEntries);
  ValueArray nonlocalVals(numSendEntries);
  // partition on-process entries into diagonal and off-diagonal+nonlocal
  ierr = PetscInfo(J, "Splitting on-process entries into diagonal and off-diagonal+nonlocal\n");CHKERRQ(ierr);
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
  ierr = PetscInfo2(j->A, "Diagonal size %d Nondiagonal size %d\n", diagonalSize, nondiagonalSize);CHKERRQ(ierr);
  ///cusp::print(diagCOO);
  ///cusp::print(nondiagonalRows);
  // partition on-process entries again into off-diagonal and nonlocal
  ierr = PetscInfo(J, "Splitting on-process entries into off-diagonal and nonlocal\n");CHKERRQ(ierr);
  thrust::stable_partition(thrust::make_zip_iterator(thrust::make_tuple(nondiagonalRows.begin(), nondiagonalCols.begin(), nondiagonalVals.begin())),
                           thrust::make_zip_iterator(thrust::make_tuple(nondiagonalRows.end(),   nondiagonalCols.end(),   nondiagonalVals.end())),
                           is_nonlocal(firstRow, lastRow));
  PetscInt nonlocalSize    = numSendEntries;
  PetscInt offdiagonalSize = nondiagonalSize - nonlocalSize;
  ierr = PetscInfo2(j->A, "Nonlocal size %d Offdiagonal size %d\n", nonlocalSize, offdiagonalSize);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_SetValuesBatchII,0,0,0,0);CHKERRQ(ierr);
  ///cusp::print(nondiagonalRows);
  // send off-proc entries (pack this up later)
  ierr = PetscLogEventBegin(MAT_SetValuesBatchIII,0,0,0,0);CHKERRQ(ierr);
  PetscMPIInt *procSendDispls, *procRecvDispls;
  PetscInt    *sendRows, *recvRows;
  PetscInt    *sendCols, *recvCols;
  PetscScalar *sendVals, *recvVals;
  ierr = PetscMalloc2(numProcs, PetscMPIInt, &procSendDispls, numProcs, PetscMPIInt, &procRecvDispls);CHKERRQ(ierr);
  ierr = PetscMalloc3(numSendEntries, PetscInt, &sendRows, numSendEntries, PetscInt, &sendCols, numSendEntries, PetscScalar, &sendVals);CHKERRQ(ierr);
  ierr = PetscMalloc3(numRecvEntries, PetscInt, &recvRows, numRecvEntries, PetscInt, &recvCols, numRecvEntries, PetscScalar, &recvVals);CHKERRQ(ierr);
  procSendDispls[0] = procRecvDispls[0] = 0;
  for (PetscInt p = 1; p < numProcs; ++p) {
    procSendDispls[p] = procSendDispls[p-1] + procSendSizes[p-1];
    procRecvDispls[p] = procRecvDispls[p-1] + procRecvSizes[p-1];
  }
#if 0
  thrust::copy(nondiagonalRows.begin(), nondiagonalRows.begin()+nonlocalSize, sendRows);
  thrust::copy(nondiagonalCols.begin(), nondiagonalCols.begin()+nonlocalSize, sendCols);
  thrust::copy(nondiagonalVals.begin(), nondiagonalVals.begin()+nonlocalSize, sendVals);
#else
  thrust::copy(thrust::make_zip_iterator(thrust::make_tuple(nondiagonalRows.begin(), nondiagonalCols.begin(), nondiagonalVals.begin())),
               thrust::make_zip_iterator(thrust::make_tuple(nondiagonalRows.begin(), nondiagonalCols.begin(), nondiagonalVals.begin()))+nonlocalSize,
               thrust::make_zip_iterator(thrust::make_tuple(sendRows,                sendCols,                sendVals)));
#endif
  //   could pack into a struct and unpack
  ierr = MPI_Alltoallv(sendRows, procSendSizes, procSendDispls, MPIU_INT,    recvRows, procRecvSizes, procRecvDispls, MPIU_INT,    comm);CHKERRQ(ierr);
  ierr = MPI_Alltoallv(sendCols, procSendSizes, procSendDispls, MPIU_INT,    recvCols, procRecvSizes, procRecvDispls, MPIU_INT,    comm);CHKERRQ(ierr);
  ierr = MPI_Alltoallv(sendVals, procSendSizes, procSendDispls, MPIU_SCALAR, recvVals, procRecvSizes, procRecvDispls, MPIU_SCALAR, comm);CHKERRQ(ierr);
  ierr = PetscFree2(procSendSizes, procRecvSizes);CHKERRQ(ierr);
  ierr = PetscFree2(procSendDispls, procRecvDispls);CHKERRQ(ierr);
  ierr = PetscFree3(sendRows, sendCols, sendVals);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_SetValuesBatchIII,0,0,0,0);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_SetValuesBatchIV,0,0,0,0);CHKERRQ(ierr);
  // Create off-diagonal matrix
  cusp::coo_matrix<IndexType,ValueType, memSpace> offdiagCOO(Nr, Nr, offdiagonalSize+numRecvEntries); // ALLOC: This is oversizes because we count diagonal entries in numRecvEntries
  // partition again into diagonal and off-diagonal
  IndexArray d_recvRows(recvRows, recvRows+numRecvEntries);
  IndexArray d_recvCols(recvCols, recvCols+numRecvEntries);
  ValueArray d_recvVals(recvVals, recvVals+numRecvEntries);
#if 0
  thrust::copy(nondiagonalRows.end()-offdiagonalSize, nondiagonalRows.end(), offdiagCOO.row_indices.begin());
  thrust::copy(nondiagonalCols.end()-offdiagonalSize, nondiagonalCols.end(), offdiagCOO.column_indices.begin());
  thrust::copy(nondiagonalVals.end()-offdiagonalSize, nondiagonalVals.end(), offdiagCOO.values.begin());
#else
  thrust::copy(thrust::make_zip_iterator(thrust::make_tuple(nondiagonalRows.end(),          nondiagonalCols.end(),             nondiagonalVals.end()))-offdiagonalSize,
               thrust::make_zip_iterator(thrust::make_tuple(nondiagonalRows.end(),          nondiagonalCols.end(),             nondiagonalVals.end())),
               thrust::make_zip_iterator(thrust::make_tuple(offdiagCOO.row_indices.begin(), offdiagCOO.column_indices.begin(), offdiagCOO.values.begin())));
#endif
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
#if !defined(PETSC_USE_COMPLEX)
    cusp::print(diagCOO);
    cusp::print(offdiagCOO);
#endif
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
  thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(diagCOO.row_indices.begin(), diagCOO.column_indices.begin())) + diagonalOffset,
     thrust::make_zip_iterator(thrust::make_tuple(diagCOO.row_indices.end(),   diagCOO.column_indices.end())),
     diagCOO.values.begin() + diagonalOffset,
     thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin())),
     A.values.begin(),
     thrust::equal_to< thrust::tuple<IndexType,IndexType> >(),
     thrust::plus<ValueType>());

  thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(offdiagCOO.row_indices.begin(), offdiagCOO.column_indices.begin())) + offdiagonalOffset,
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
#if 0 // This is done by MatSetUpMultiply_MPIAIJ()
  //   TODO: Get better code from Nathan
  IndexArray d_colmap(Nc);
  thrust::unique_copy(B.column_indices.begin(), B.column_indices.end(), d_colmap.begin());
  IndexHostArray colmap(d_colmap.begin(), d_colmap.end());
  IndexType      newCol = 0;
  for (IndexHostArray::iterator c_iter = colmap.begin(); c_iter != colmap.end(); ++c_iter, ++newCol) {
    thrust::replace(B.column_indices.begin(), B.column_indices.end(), *c_iter, newCol);
  }
#endif

  // print the final matrix
  if (PetscLogPrintInfo) {
#if !defined(PETSC_USE_COMPLEX)
    cusp::print(A);
    cusp::print(B);
#endif
  }

  ierr = PetscInfo(J, "Converting to PETSc matrix\n");CHKERRQ(ierr);
  ierr = MatSetType(J, MATMPIAIJCUSP);CHKERRQ(ierr);
  CUSPMATRIX *Agpu = new CUSPMATRIX;
  CUSPMATRIX *Bgpu = new CUSPMATRIX;
  cusp::convert(A, *Agpu);
  cusp::convert(B, *Bgpu);
  if (PetscLogPrintInfo) {
#if !defined(PETSC_USE_COMPLEX)
    cusp::print(*Agpu);
    cusp::print(*Bgpu);
#endif
  }
  {
    ierr = PetscInfo(J, "Copying to CPU matrix");CHKERRQ(ierr);
    ierr = MatCUSPCopyFromGPU(j->A, Agpu);CHKERRQ(ierr);
    ierr = MatCUSPCopyFromGPU(j->B, Bgpu);CHKERRQ(ierr);
#if 0 // This is done by MatSetUpMultiply_MPIAIJ()
    // Create the column map
    ierr = PetscFree(j->garray);CHKERRQ(ierr);
    ierr = PetscMalloc(Nc * sizeof(PetscInt), &j->garray);CHKERRQ(ierr);
    PetscInt c = 0;
    for (IndexHostArray::iterator c_iter = colmap.begin(); c_iter != colmap.end(); ++c_iter, ++c) {
      j->garray[c] = *c_iter;
    }
#endif
  }
  ierr = PetscLogEventEnd(MAT_SetValuesBatchIV,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
