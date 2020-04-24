The Various Matrix Classes
==========================

PETSc provides a variety of matrix implementations, since no single
matrix format is appropriate for all problems. This section first
discusses various matrix blocking strategies and then describes the
assortment of matrix types in PETSc.

Matrix Blocking Strategies
--------------------------

In today’s computers, the time to perform an arithmetic operation is
dominated by the time to move the data into position, not the time to
compute the arithmetic result. For example, the time to perform a
multiplication operation may be one clock cycle, while the time to move
the floating-point number from memory to the arithmetic unit may take 10
or more cycles. In order to help manage this difference in time scales,
most processors have at least three levels of memory: registers, cache,
and random access memory. (In addition, some processors have external
caches, and the complications of paging introduce another level to the
hierarchy.)

Thus, to achieve high performance, a code should first move data into
cache and from there move it into registers and use it repeatedly while
it remains in the cache or registers before returning it to main memory.
If a floating-point number is reused 50 times while it is in registers,
then the “hit” of 10 clock cycles to bring it into the register is not
important. But if the floating-point number is used only once, the “hit”
of 10 clock cycles becomes noticeable, resulting in disappointing flop
rates.

Unfortunately, the compiler controls the use of the registers, and the
hardware controls the use of the cache. Since the user has essentially
no direct control, code must be written in such a way that the compiler
and hardware cache system can perform well. Good-quality code is then
said to respect the memory hierarchy.

The standard approach to improving the hardware utilization is to use
blocking. That is, rather than working with individual elements in the
matrices, you employ blocks of elements. Since the use of implicit
methods in PDE-based simulations leads to matrices with a naturally
blocked structure (with a block size equal to the number of degrees of
freedom per cell), blocking is advantageous. The PETSc sparse matrix
representations use a variety of techniques for blocking, including the
following:

-  Storing the matrices using a generic sparse matrix format, but
   storing additional information about adjacent rows with identical
   nonzero structure (so-called I-nodes); this I-node information is
   used in the key computational routines to improve performance (the
   default for the ``MATSEQAIJ`` and ``MATMPIAIJ`` formats).

-  Storing the matrices using a fixed (problem dependent) block size
   (via the ``MATSEQBAIJ`` and ``MATMPIBAIJ`` formats).

The advantage of the first approach is that it is a minimal change from
a standard sparse matrix format and brings a large percentage of the
improvement obtained via blocking. Using a fixed block size gives the
best performance, since the code can be hardwired with that particular
size (for example, in some problems the size may be 3, in others 5, and
so on), so that the compiler will then optimize for that size, removing
the overhead of small loops entirely.

The following table presents the floating-point performance for a basic
matrix-vector product using three approaches: a basic compressed row
storage format (using the PETSc runtime options
``-mat_seqaij -mat_nounroll)``; the same compressed row format using
I-nodes (with the option ``-mat_seqaij``); and a fixed block size code,
with a block size of 3 for these problems (using the option
``-mat_seqbaij``). The rates were computed on one node of an older IBM
Power processor based system, using two test matrices. The first matrix
(ARCO1), courtesy of Rick Dean of Arco, arises in multiphase flow
simulation; it has 1,501 degrees of freedom, 26,131 matrix nonzeros, a
natural block size of 3, and a small number of well terms. The second
matrix (CFD), arises in a three-dimensional Euler flow simulation and
has 15,360 degrees of freedom, 496,000 nonzeros, and a natural block
size of 5. In addition to displaying the flop rates for matrix-vector
products, we display them for triangular solves obtained from an ILU(0)
factorization.

+------------+------------+-------+-----------------+------------------+
| Problem    | Block size | Basic | I-node version  | Fixed block size |
+============+============+=======+=================+==================+
| Matrix-Vector Product (Mflop/sec)                                    |
+------------+------------+-------+-----------------+------------------+
| Multiphase | 3          | 27    | 43              | 70               |
+------------+------------+-------+-----------------+------------------+
| Euler      | 5          | 28    | 58              | 90               |
+------------+------------+-------+-----------------+------------------+
| Triangular Solves from ILU(0) (Mflop/sec)                            |
+------------+------------+-------+-----------------+------------------+
| Multiphase | 3          | 22    | 31              | 49               |
+------------+------------+-------+-----------------+------------------+
| Euler      | 5          | 22    | 39              | 65               |
+------------+------------+-------+-----------------+------------------+

These examples demonstrate that careful implementations of the basic
sequential kernels in PETSc can dramatically improve overall floating
point performance, and users can immediately benefit from such
enhancements without altering a single line of their application codes.
Note that the speeds of the I-node and fixed block operations are
several times that of the basic sparse implementations.

Assorted Matrix Types
---------------------

PETSc offers a variety of both sparse and dense matrix types.

Sequential AIJ Sparse Matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default matrix representation within PETSc is the general sparse AIJ
format (also called the Yale sparse matrix format or compressed sparse
row format, CSR).

Parallel AIJ Sparse Matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The AIJ sparse matrix type, is the default parallel matrix format;
additional implementation details are given in :cite:`petsc-efficient`.

Sequential Block AIJ Sparse Matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The sequential and parallel block AIJ formats, which are extensions of
the AIJ formats described above, are intended especially for use with
multiclass PDEs. The block variants store matrix elements by fixed-sized
dense ``nb`` :math:`\times` ``nb`` blocks. The stored row and column
indices begin at zero.

The routine for creating a sequential block AIJ matrix with ``m`` rows,
``n`` columns, and a block size of ``nb`` is

::

    MatCreateSeqBAIJ(MPI_Comm comm,int nb,int m,int n,int nz,int *nnz,Mat *A)

The arguments ``nz`` and ``nnz`` can be used to preallocate matrix
memory by indicating the number of *block* nonzeros per row. For good
performance during matrix assembly, preallocation is crucial; however,
you can set ``nz=0`` and ``nnz=NULL`` for PETSc to dynamically allocate
matrix memory as needed. The PETSc users manual discusses preallocation
for the AIJ format; extension to the block AIJ format is
straightforward.

Note that the routine ``MatSetValuesBlocked()`` can be used for more
efficient matrix assembly when using the block AIJ format.

Parallel Block AIJ Sparse Matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parallel block AIJ matrices with block size nb can be created with the
command ``MatCreateBAIJ()``

::

    MatCreateBAIJ(MPI_Comm comm,int nb,int m,int n,int M,int N,int d_nz,int *d_nnz,int o_nz,int *o_nnz,Mat *A);

``A`` is the newly created matrix, while the arguments ``m``, ``n``,
``M``, and ``N`` indicate the number of local rows and columns and the
number of global rows and columns, respectively. Either the local or
global parameters can be replaced with ``PETSC_DECIDE``, so that PETSc
will determine them. The matrix is stored with a fixed number of rows on
each processor, given by ``m``, or determined by PETSc if ``m`` is
``PETSC_DECIDE``.

If ``PETSC_DECIDE`` is not used for ``m`` and ``n`` then you must ensure
that they are chosen to be compatible with the vectors. To do so, you
first consider the product :math:`y = A x`. The ``m`` that used in
``MatCreateBAIJ()`` must match the local size used in the
``VecCreateMPI()`` for ``y``. The ``n`` used must match that used as the
local size in ``VecCreateMPI()`` for ``x``.

You must set ``d_nz=0``, ``o_nz=0``, ``d_nnz=NULL``, and ``o_nnz=NULL`` for
PETSc to control dynamic allocation of matrix memory space. Analogous to
``nz`` and ``nnz`` for the routine ``MatCreateSeqBAIJ()``, these
arguments optionally specify block nonzero information for the diagonal
(``d_nz`` and ``d_nnz``) and off-diagonal (``o_nz`` and ``o_nnz``) parts of
the matrix. For a square global matrix, we define each processor’s
diagonal portion to be its local rows and the corresponding columns (a
square submatrix); each processor’s off-diagonal portion encompasses the
remainder of the local matrix (a rectangular submatrix). The PETSc users
manual gives an example of preallocation for the parallel AIJ matrix
format; extension to the block parallel AIJ case is straightforward.

Sequential Dense Matrices
~~~~~~~~~~~~~~~~~~~~~~~~~

PETSc provides both sequential and parallel dense matrix formats, where
each processor stores its entries in a column-major array in the usual
Fortran style.

Parallel Dense Matrices
~~~~~~~~~~~~~~~~~~~~~~~

The parallel dense matrices are partitioned by rows across the
processors, so that each local rectangular submatrix is stored in the
dense format described above.

References
----------

.. bibliography:: ../../tex/petsc.bib
   :filter: docname in docnames

.. bibliography:: ../../tex/petscapp.bib
   :filter: docname in docnames
