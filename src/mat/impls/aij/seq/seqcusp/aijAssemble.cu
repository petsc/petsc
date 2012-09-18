#include "petscconf.h"
PETSC_CUDA_EXTERN_C_BEGIN
#include "../src/mat/impls/aij/seq/aij.h"          /*I "petscmat.h" I*/
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
typedef IndexArray::iterator IndexArrayIterator;
typedef ValueArray::iterator ValueArrayIterator;

// Ne: Number of elements
// Nl: Number of dof per element
#undef __FUNCT__
#define __FUNCT__ "MatSetValuesBatch_SeqAIJCUSP"
PetscErrorCode MatSetValuesBatch_SeqAIJCUSP(Mat J, PetscInt Ne, PetscInt Nl, PetscInt *elemRows, const PetscScalar *elemMats)
{
  size_t   N  = Ne * Nl;
  size_t   No = Ne * Nl*Nl;
  PetscInt Nr; // Number of rows
  PetscErrorCode ierr;

  // copy elemRows and elemMat to device
  IndexArray d_elemRows(elemRows, elemRows + N);
  ValueArray d_elemMats(elemMats, elemMats + No);

  PetscFunctionBegin;
  ierr = MatGetSize(J, &Nr, PETSC_NULL);CHKERRQ(ierr);
  // allocate storage for "fat" COO representation of matrix
  ierr = PetscInfo1(J, "Making COO matrix of size %d\n", Nr);CHKERRQ(ierr);
  cusp::coo_matrix<IndexType,ValueType, memSpace> COO(Nr, Nr, No);

  // repeat elemRows entries Nl times
  ierr = PetscInfo(J, "Making row indices\n");CHKERRQ(ierr);
  repeated_range<IndexArrayIterator> rowInd(d_elemRows.begin(), d_elemRows.end(), Nl);
  thrust::copy(rowInd.begin(), rowInd.end(), COO.row_indices.begin());

  // tile rows of elemRows Nl times
  ierr = PetscInfo(J, "Making column indices\n");CHKERRQ(ierr);
  tiled_range<IndexArrayIterator> colInd(d_elemRows.begin(), d_elemRows.end(), Nl, Nl);
  thrust::copy(colInd.begin(), colInd.end(), COO.column_indices.begin());

  // copy values from elemMats into COO structure (could be avoided)
  thrust::copy(d_elemMats.begin(), d_elemMats.end(), COO.values.begin());

  // For MPIAIJ, split this into two COO matrices, and return both
  //   Need the column map

  // print the "fat" COO representation
#if !defined(PETSC_USE_COMPLEX)
  if (PetscLogPrintInfo) {cusp::print(COO);}
#endif
  // sort COO format by (i,j), this is the most costly step
  ierr = PetscInfo(J, "Sorting rows and columns\n");CHKERRQ(ierr);
#if 1
  COO.sort_by_row_and_column();
#else
  {
    ierr = PetscInfo(J, "  Making permutation\n");CHKERRQ(ierr);
    IndexArray permutation(No);
    thrust::sequence(permutation.begin(), permutation.end());

    // compute permutation and sort by (I,J)
    {
        ierr = PetscInfo(J, "  Sorting columns\n");CHKERRQ(ierr);
        IndexArray temp(No);
        thrust::copy(COO.column_indices.begin(), COO.column_indices.end(), temp.begin());
        thrust::stable_sort_by_key(temp.begin(), temp.end(), permutation.begin());
        ierr = PetscInfo(J, "    Sorted columns\n");CHKERRQ(ierr);
        if (PetscLogPrintInfo) {
          for (IndexArrayIterator t_iter = temp.begin(), p_iter = permutation.begin(); t_iter != temp.end(); ++t_iter, ++p_iter) {
            ierr = PetscInfo2(J, "%d(%d)\n", *t_iter, *p_iter);CHKERRQ(ierr);
          }
        }

        ierr = PetscInfo(J, "  Copying rows\n");CHKERRQ(ierr);
        //cusp::copy(COO.row_indices, temp);
        thrust::copy(COO.row_indices.begin(), COO.row_indices.end(), temp.begin());
        ierr = PetscInfo(J, "  Gathering rows\n");CHKERRQ(ierr);
        thrust::gather(permutation.begin(), permutation.end(), temp.begin(), COO.row_indices.begin());
        ierr = PetscInfo(J, "  Sorting rows\n");CHKERRQ(ierr);
        thrust::stable_sort_by_key(COO.row_indices.begin(), COO.row_indices.end(), permutation.begin());

        ierr = PetscInfo(J, "  Gathering columns\n");CHKERRQ(ierr);
        cusp::copy(COO.column_indices, temp);
        thrust::gather(permutation.begin(), permutation.end(), temp.begin(), COO.column_indices.begin());
    }

    // use permutation to reorder the values
    {
        ierr = PetscInfo(J, "  Sorting values\n");CHKERRQ(ierr);
        ValueArray temp(COO.values);
        cusp::copy(COO.values, temp);
        thrust::gather(permutation.begin(), permutation.end(), temp.begin(), COO.values.begin());
    }
  }
#endif

  // print the "fat" COO representation
#if !defined(PETSC_USE_COMPLEX)
  if (PetscLogPrintInfo) {cusp::print(COO);}
#endif
  // compute number of unique (i,j) entries
  //   this counts the number of changes as we move along the (i,j) list
  ierr = PetscInfo(J, "Computing number of unique entries\n");CHKERRQ(ierr);
  size_t num_entries = thrust::inner_product
    (thrust::make_zip_iterator(thrust::make_tuple(COO.row_indices.begin(), COO.column_indices.begin())),
     thrust::make_zip_iterator(thrust::make_tuple(COO.row_indices.end (),  COO.column_indices.end()))   - 1,
     thrust::make_zip_iterator(thrust::make_tuple(COO.row_indices.begin(), COO.column_indices.begin())) + 1,
     size_t(1),
     thrust::plus<size_t>(),
     thrust::not_equal_to< thrust::tuple<IndexType,IndexType> >());

  // allocate COO storage for final matrix
  ierr = PetscInfo(J, "Allocating compressed matrix\n");CHKERRQ(ierr);
  cusp::coo_matrix<IndexType, ValueType, memSpace> A(Nr, Nr, num_entries);

  // sum values with the same (i,j) index
  // XXX thrust::reduce_by_key is unoptimized right now, so we provide a SpMV-based one in cusp::detail
  //     the Cusp one is 2x faster, but still not optimal
  // This could possibly be done in-place
  ierr = PetscInfo(J, "Compressing matrix\n");CHKERRQ(ierr);
  thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(COO.row_indices.begin(), COO.column_indices.begin())),
     thrust::make_zip_iterator(thrust::make_tuple(COO.row_indices.end(),   COO.column_indices.end())),
     COO.values.begin(),
     thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin())),
     A.values.begin(),
     thrust::equal_to< thrust::tuple<IndexType,IndexType> >(),
     thrust::plus<ValueType>());

  // print the final matrix
#if !defined(PETSC_USE_COMPLEX)
  if (PetscLogPrintInfo) {cusp::print(A);}
#endif
  //std::cout << "Writing matrix" << std::endl;
  //cusp::io::write_matrix_market_file(A, "A.mtx");

  ierr = PetscInfo(J, "Converting to PETSc matrix\n");CHKERRQ(ierr);
  ierr = MatSetType(J, MATSEQAIJCUSP);CHKERRQ(ierr);
  //cusp::csr_matrix<PetscInt,PetscScalar,cusp::device_memory> Jgpu;
  CUSPMATRIX *Jgpu = new CUSPMATRIX;
  cusp::convert(A, *Jgpu);
#if !defined(PETSC_USE_COMPLEX)
  if (PetscLogPrintInfo) {cusp::print(*Jgpu);}
#endif
  ierr = PetscInfo(J, "Copying to CPU matrix\n");CHKERRQ(ierr);
  ierr = MatCUSPCopyFromGPU(J, Jgpu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
