.. _ch_blas-lapack:

The Use of BLAS and LAPACK in PETSc and external libraries
----------------------------------------------------------


#. BLAS 1 operations - vector operations such as VecNorm(), VecAXPY(), and VecScale(). Depending on the
   simulation the size of the vectors may be from hundreds of entries to many millions.

#. BLAS 2 operations - dense matrix with vector operations, generally the dense matrices are very small.

#. Eigenvalue and SVD computations, generally for very small matrices

#. External packages such as MUMPS and SuperLU_DIST use BLAS 3 operations (and possibly BLAS 1 and 2). The
   dense matrices may be of modest size, going up to thousands of rows and columns.

For most PETSc simulations (that is not using certain external packages) using an optimized set of BLAS/LAPACK routines
only provides a modest improvement in performance. For some external packages using optimized BLAS/LAPACK can make a
dramatic improvement in performance.

32 or 64 bit BLAS/LAPACK integers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BLAS/LAPACK libraries may use 32 bit integers or 64 bit integers. PETSc configure and compile handles this automatically
so long at the arguments to the BLAS/LAPACK routines are set to the type PetscBLASInt.  The routine PetscBLASIntCast(PetscInt,PetscBLASInt*) casts
a PetscInt to the BLAS/LAPACK size. If the BLAS/LAPACK size is not large enough it generates an error. For the vast majority of
simulations even very large ones 64 bit BLAS/LAPACK integers are not needed, even if 64 bit PETSc integers are used, The configure
option ``-with-64-bit-blas-indices`` attempts to locate and use a 64 bit integer BLAS/LAPACK library.

Shared memory BLAS/LAPACK parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some BLAS/LAPACK libraries make use of shared memory parallelism within the function calls, generally using OpenMP, or possibly PThreads.
If this feature is on, it is in addition to the MPI based parallelism that PETSc is using. Thus it can result in over-subscription of hardware resources. For example,
if a system has 16 cores and PETSc is run with an MPI size of 16 then each core is assigned an MPI process. But if the BLAS/LAPACK is running with
OpenMP and 4 threads per process this results 64 threads competing to use 16 cores which generally will perform poorly.

If one elects to use both MPI parallelism and OpenMP BLAS/LAPACK parallelism one should insure they do not over subscribe the hardware
resources. Since PETSc does not natively using OpenMP this means that phases of the computation that do not use BLAS/LAPACK will be under-subscribed,
thus under-utilizing the system. For PETSc simulations which do not us external packages there is generally no benefit to using parallel
BLAS/LAPACK. The environmental variable ``OMP_NUM_THREADS`` can be used to set the number of threads used by parallel BLAS/LAPACK. The additional
environmental variables ``OMP_PROC_BIND`` and ``OMP_PLACES`` may also need to be set appropriate for the system to obtain good parallel performance with
BLAS/LAPACK. The configure option ``-with-openmp`` will trigger PETSc to try to locate and use a parallel BLAS/LAPACK library.


Certain external packages such as MUMPS may benefit from using parallel BLAS/LAPACK operations. See the manual page MATSOLVERMUMPS for details on
how one can restrict the number of MPI processes while running MUMPS to utilize parallel BLAS/LAPACK.

.. _ch_blas-lapack_avail-libs:

Available BLAS/LAPACK libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most systems (besides Microsoft Windows) come with pre-installed BLAS/LAPACK which are satisfactory for many PETSc simulations.

The freely available Intel MKL mathematics libraries provide BLAS/LAPACK that are generally better performing than the system provided libraries
and are generally fine for most users.

For systems that do not provide BLAS/LAPACK, such as Microsoft Windows, PETSc provides the Fortran reference version
``--download-fblaslapack`` and a f2c generated C version ``--download-f2cblaslapack`` (which also supports 128 bit real number computations).
These libraries are generally low performing but useful to get started with PETSc easily.

PETSc also provides access to OpenBLAS via the ``--download-openblas`` configure option. OpenBLAS uses some highly optimized operations but falls back on reference
routines for many other operations. See the OpenBLAS manual for more information. The configure option ``--download-openblas`` provides a full BLAS/LAPACK implementation.

BLIS does not bundle LAPACK with it so PETSc's configure attempts to locate a compatible system LAPACK library to use if ``--download-blis`` is
selected. One can use ``--download-f2cblaslapack --download-blis`` to build netlib LAPACK with BLIS. This is recommended as a portable high-performance option.


