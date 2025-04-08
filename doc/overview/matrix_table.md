(doc_matrix)=

# Summary of Matrix Types Available In PETSc

```{eval-rst}
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
     - Intel oneAPI MKL
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
     - NVIDIA cuSPARSE library
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
   * - Kronecker product of sparse matrix :math:`A`; :math:`I \otimes S + A \otimes T`
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
     - ``MatCreateScaLAPACK()``
     - ScaLAPACK
     - Block cyclic storage
   * -
     - ``MATDENSECUDA``
     - ``MatCreateDenseCUDA()``
     -
     - NVIDIA GPU Acceleration
   * - ``MatMult()`` via finite differencing of a function
     - ``MATMFFD``, see :any:`sec_nlmatrixfree`
     - ``MatCreateMFFD()``, see also ``MatCreateSNESMF()``
     -
     - Provides only matrix-vector products
   * - User-provided operations
     - ``MATSHELL``, see also :any:`sec_matrixfree`
     - ``MatCreateShell()``
     -
     -
   * - Low-rank updates
     - ``MATLMVM``, ``MATLMVMDFP``, ``MATLMVMBFGS``, ``MATLMVMSR1``, ...
     - ``MatCreateLMVM()``
     -
     -  limited-memory BFGS style matrices
   * -
     - ``MATLRC``
     - ``MatCreateLRC()``
     -
     - :math:`A + UCV^T`
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
     - :math:`\mathcal H^2` matrices
     -
   * - Transpose, :math:`A^T`, virtual
     - ``MATTRANSPOSEVIRTUAL``
     - ``MatCreateTranspose()``
     -
     -
   * - Hermitian Transpose, :math:`A^H`, virtual
     - ``MATHERMITIANTRANSPOSEVIRTUAL``
     - ``MatCreateHermitianTranspose()``
     -
     -
   * - Normal, :math:`A^TA`, virtual
     - ``MATNORMAL``
     - ``MatCreateNormal()``
     -
     -
   * - Hermitian Normal, :math:`A^HA`, virtual
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
     - :math:`I - \frac{1}{N}e e^T`, :math:`e=[1,\dots,1]^T`
   * - Block matrix
     - ``MATBLOCKMAT``
     - ``MatCreateBlockMat()``
     -
     -

```
