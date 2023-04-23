.. _doc_matrix:

==========================================
Summary of Matrix Types Available In PETSc
==========================================

.. list-table::
   :widths: auto
   :align: center
   :header-rows: 1

   * - Format
     - Matrix Types
     - Constructor
     - External Packages
     - Details
   * - CSR
     - ``MATAIJ``
     - ``MatCreateAIJ()``
     -
     - Compressed sparse row
   * -
     - ``MATAIJMKL``
     - ``MatCreateMPIAIJMKL()``
     - Intel's MKL, OpenOne API
     - OpenMP support
   * -
     - ``MATAIJSELL``
     - ``MatCreateMPIAIJSELL()``
     -
     - SIMD acceleration
   * -
     - ``MATAIJPERM``
     - ``MatCreateMPIAIJPERM()``
     -
     - Vectorized version
   * -
     - ``MATAIJCUSPARSE``
     - ``MatCreateAIJCUSPARSE()``
     - NVIDIA's CuSparse library
     - NVIDIA GPU acceleration
   * -
     - ``MATAIJKOKKOS``
     - ``MatCreateAIJKokkos()``
     - Kokkos
     - GPU acceleration
   * - Constant row length
     - ``MATAIJCRL``
     - ``MatCreateMPIAIJCRL()``
     -
     - Vectorized version
   * - Multiple applications of single ``MATAIJ``
     - ``MATMAIJ``
     - ``MatCreateMAIJ()``
     -
     - Commonly used for identical interpolations on each component of a multi-component vector
   * - Kronecker product of sparse matrix $A$; $ I \otimes S + A \otimes T $
     - ``MATKAIJ``
     - ``MatCreateKAIJ()``
     -
     -
   * - Sliced Ellpack
     - ``MATSELL``
     - ``MatCreateSELL()``
     -
     - SIMD and GPU acceleration
   * - Block CSR
     - ``MATBAIJ``
     - ``MatCreateBAIJ()``
     -
     - Block compressed sparse row
   * - Symmetric Block CSR
     - ``MATSBAIJ``
     - ``MatCreateSBAIJ()``
     -
     - Upper triangular compressed sparse row
   * - Dense
     - ``MATDENSE``
     - ``MatCreateDense()``
     -
     - Row oriented storage
   * -
     - ``MATELEMENTAL``
     - ``MatCreateElemental()``
     - Elemental by Jack Poulson
     - Block cyclic storage
   * -
     - ``MATSCALAPACK``
     - ``MatCreateScaLAPACK()()``
     - ScaLAPACK
     - Block cyclic storage
   * -
     - ``MATDENSECUDA``
     - ``MatCreateDenseCUDA()``
     -
     - NVIDIA GPU Acceleration
   * - ``MatMult()`` via finite differencing
     - ``MATMFFD``
     - ``MatCreateMFFD()``
     -
     - Provides only matrix-vector products
   * - User provided operations
     - ``MATSHELL``
     - ``MatCreateShell()``
     -
     -
   * - Low rank updates
     - ``MATLMVM``, ``MATLMVMDFP``, ``MATLMVMBFGS``, ``MATLMVMSR1``, ...
     - ``MatCreateLMVM()``
     -
     -  limited-memory BFGS style matrices
   * -
     - ``MATLRC``
     - ``MatCreateLRC()``
     -
     -  $A + UCV^T$
   * - FFT
     - ``MATFFTW``
     - ``MatCreateFFT()``
     - FFTW
     -
   * -
     - ``MATSEQCUFFT``
     - ``MatCreateSeqCUFFT()``
     - NVIDIA's CuFFT
     - NVIDIA GPUs
   * - Hierarchical
     - ``MATHTOOL``
     - ``MatCreateHtoolFromKernel()``
     - Htool
     -
   * -
     - ``MATH2OPUS``
     - ``MatCreateH2OpusFromMat()``
     - H_2 matrices
     -
   * - Transpose, virtual
     - ``MATTRANSPOSEVIRTUAL``
     - ``MatCreateTranspose()``
     -
     -
   * - Hermitian Transpose, virtual
     - ``MATHERMITIANTRANSPOSEVIRTUAL``
     - ``MatCreateHermitianTranspose()``
     -
     -
   * - Normal, A'\*A, virtual
     - ``MATNORMAL``
     - ``MatCreateNormal()``
     -
     -
   * - Hermitian Normal, A'\*A, virtual
     - ``MATNORMALHERMITIAN``
     - ``MatCreateNormalHermitian()``
     -
     -
   * - Schur complement
     - ``MATSCHURCOMPLEMENT``
     - ``MatCreateSchurComplement()``, ``MatGetSchurComplement()``
     -
     -
   * - Sub-matrix, virtual
     - ``MATSUBMATRIX``
     - ``MatCreateSubMatrixVirtual()``
     -
     - Provides ``MatMult()`` and similar operations
   * -
     - ``MATLOCALREF``
     - ``MatCreateLocalRef()``
     -
     - For use in matrix assembly
   * - Nested matrix
     - ``MATNEST``
     - ``MatCreateNest()``
     -
     -
   * - Scatter operator
     - ``MATSCATTER``
     - ``MatCreateScatter()``
     -
     -
   * - Centering operator
     - ``MATCENTERING``
     - ``MatCreateCentering()``
     -
     -  I - (1/N) * ones*ones'
   * - Block matrix
     - ``MATBLOCKMAT``
     - ``MatCreateBlockMat()``
     -
     -


