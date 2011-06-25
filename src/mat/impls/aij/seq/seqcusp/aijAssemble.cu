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
typedef IndexArray::iterator IndexArrayIterator;
typedef ValueArray::iterator ValueArrayIterator;

EXTERN_C_BEGIN
// Ne: Number of elements
// Nl: Number of dof per element
// Nr: Number of matrix rows (dof)
#undef __FUNCT__
#define __FUNCT__ "MatSeqAIJSetValuesBatch"
PetscErrorCode MatSeqAIJSetValuesBatch(Mat J, PetscInt Ne, PetscInt Nl, PetscInt Nr, PetscInt *elemRows, PetscScalar *elemMats)
{
  size_t N  = Ne * Nl;
  size_t No = Ne * Nl*Nl;

  // copy elemRows and elemMat to device
  IndexArray d_elemRows(elemRows, elemRows + N);
  ValueArray d_elemMats(elemMats, elemMats + No);

  // allocate storage for "fat" COO representation of matrix
  std::cout << "Making COO matrix" << std::endl;
  cusp::coo_matrix<IndexType,ValueType, memSpace> COO(Nr, Nr, No);

  // repeat elemRows entries Nl times
  std::cout << "Making row indices" << std::endl;
  repeated_range<IndexArrayIterator> rowInd(d_elemRows.begin(), d_elemRows.end(), Nl);
  thrust::copy(rowInd.begin(), rowInd.end(), COO.row_indices.begin());

  // tile rows of elemRows Nl times
  std::cout << "Making column indices" << std::endl;
  tiled_range<IndexArrayIterator> colInd(d_elemRows.begin(), d_elemRows.end(), Nl, Nl);
  thrust::copy(colInd.begin(), colInd.end(), COO.column_indices.begin());

  // copy values from elemMats into COO structure (could be avoided)
  thrust::copy(d_elemMats.begin(), d_elemMats.end(), COO.values.begin());

  // print the "fat" COO representation
  cusp::print(COO);

  // sort COO format by (i,j), this is the most costly step
  std::cout << "Sorting rows and columns" << std::endl;
  //COO.sort_by_row_and_column();
  ///cusp::detail::sort_by_row_and_column(COO.row_indices, COO.column_indices, COO.values);
  {
    std::cout << "  Making permutation" << std::endl;
    IndexArray permutation(No);
    thrust::sequence(permutation.begin(), permutation.end());

    // compute permutation and sort by (I,J)
    {
        std::cout << "  Sorting columns" << std::endl;
        IndexArray temp(No);
        thrust::copy(COO.column_indices.begin(), COO.column_indices.end(), temp.begin());
        thrust::stable_sort_by_key(temp.begin(), temp.end(), permutation.begin());
        std::cout << "    Sorted columns" << std::endl;
        for(IndexArrayIterator t_iter = temp.begin(), p_iter = permutation.begin(); t_iter != temp.end(); ++t_iter, ++p_iter) {
          std::cout << *t_iter << "("<<*p_iter<<")" << std::endl;
        }

        std::cout << "  Copying rows" << std::endl;
        //cusp::copy(COO.row_indices, temp);
        thrust::copy(COO.row_indices.begin(), COO.row_indices.end(), temp.begin());
        std::cout << "  Gathering rows" << std::endl;
        thrust::gather(permutation.begin(), permutation.end(), temp.begin(), COO.row_indices.begin());
        std::cout << "  Sorting rows" << std::endl;
        thrust::stable_sort_by_key(COO.row_indices.begin(), COO.row_indices.end(), permutation.begin());

        std::cout << "  Gathering columns" << std::endl;
        cusp::copy(COO.column_indices, temp);
        thrust::gather(permutation.begin(), permutation.end(), temp.begin(), COO.column_indices.begin());
    }

    // use permutation to reorder the values
    {
        std::cout << "  Sorting values" << std::endl;
        ValueArray temp(COO.values);
        cusp::copy(COO.values, temp);
        thrust::gather(permutation.begin(), permutation.end(), temp.begin(), COO.values.begin());
    }
  }

  // print the "fat" COO representation
  cusp::print(COO);

  // compute number of unique (i,j) entries
  //   this counts the number of changes as we move along the (i,j) list
  std::cout << "Computing number of unique entries" << std::endl;
  size_t num_entries = thrust::inner_product
    (thrust::make_zip_iterator(thrust::make_tuple(COO.row_indices.begin(), COO.column_indices.begin())),
     thrust::make_zip_iterator(thrust::make_tuple(COO.row_indices.end (),  COO.column_indices.end()))   - 1,
     thrust::make_zip_iterator(thrust::make_tuple(COO.row_indices.begin(), COO.column_indices.begin())) + 1,
     size_t(1),
     thrust::plus<size_t>(),
     thrust::not_equal_to< thrust::tuple<IndexType,IndexType> >());

  // allocate COO storage for final matrix
  std::cout << "Allocating compressed matrix" << std::endl;
  cusp::coo_matrix<IndexType, ValueType, memSpace> A(Nr, Nr, num_entries);

  // sum values with the same (i,j) index
  // XXX thrust::reduce_by_key is unoptimized right now, so we provide a SpMV-based one in cusp::detail
  //     the Cusp one is 2x faster, but still not optimal
  // This could possibly be done in-place
  std::cout << "Compressing matrix" << std::endl;
#if 0
  cusp::detail::device::reduce_by_key
    (thrust::make_zip_iterator(thrust::make_tuple(COO.row_indices.begin(), COO.column_indices.begin())),
     thrust::make_zip_iterator(thrust::make_tuple(COO.row_indices.end(),   COO.column_indices.end())),
     COO.values.begin(),
     thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin())),
     A.values.begin(),
     thrust::equal_to< thrust::tuple<IndexType,IndexType> >(),
     thrust::plus<ValueType>());
#else
  thrust::reduce_by_key
    (thrust::make_zip_iterator(thrust::make_tuple(COO.row_indices.begin(), COO.column_indices.begin())),
     thrust::make_zip_iterator(thrust::make_tuple(COO.row_indices.end(),   COO.column_indices.end())),
     COO.values.begin(),
     thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin())),
     A.values.begin(),
     thrust::equal_to< thrust::tuple<IndexType,IndexType> >(),
     thrust::plus<ValueType>());
#endif

  // print the final matrix
  cusp::print(A);

  std::cout << "Writing matrix" << std::endl;
  cusp::io::write_matrix_market_file(A, "A.mtx");

  PetscErrorCode ierr;

  ierr = MatSetType(J, MATSEQAIJCUSP);CHKERRQ(ierr);
  //cusp::csr_matrix<PetscInt,PetscScalar,cusp::device_memory> Jgpu;
  CUSPMATRIX *Jgpu = new CUSPMATRIX;
  cusp::convert(A, *Jgpu);
  cusp::print(*Jgpu);
  ierr = MatCUSPCopyFromGPU(J, Jgpu);CHKERRQ(ierr);
  return 0;
}
EXTERN_C_END
