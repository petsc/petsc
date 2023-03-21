.. _chapter_fortran:

PETSc for Fortran Users
-----------------------

Most of the functionality of PETSc can be obtained by people who program
purely in Fortran.

.. _sec_fortran_includes:

Modules and Include Files
~~~~~~~~~~~~~~~~~~~~~~~~~

To use PETSc with Fortran you must use both PETSc include files and modules.
At the beginning of every function and module definition you need something like

.. code-block:: fortran

  #include "petsc/finclude/petscXXX.h"
           use petscXXX

You can declare PETSc object variables using either of the following:

.. code-block:: fortran

    XXX variablename


.. code-block:: fortran

    type(tXXX) variablename

For example,

.. code-block:: fortran

  #include "petsc/finclude/petscvec.h"
           use petscvec

      Vec b
      type(tVec) x

PETSc types like ``PetscInt`` and ``PetscReal`` are simply aliases for basic Fortran types and cannot be written as ``type(tPetscInt)``


The Fortran include files for PETSc are located in the directory
``$PETSC_DIR/include/petsc/finclude`` and should be used via
statements such as the following:

.. code-block:: fortran

   #include <petsc/finclude/petscXXX.h>

for example,

.. code-block:: fortran

   #include <petsc/finclude/petscksp.h>

You must also use the appropriate Fortran module which is done with

.. code-block:: fortran

   use petscXXX

for example,

.. code-block:: fortran

   use petscksp

The few differences between the C and Fortran PETSc interfaces
are due to Fortran syntax differences. All Fortran routines have the
same names as the corresponding C versions, and PETSc command line
options are fully supported. The routine arguments follow the usual
Fortran conventions; the user need not worry about passing pointers or
values. The calling sequences for the Fortran version are in most cases
identical to the C version, except for the error checking variable
discussed in :any:`sec_fortran_errors` and a few routines
listed in :any:`sec_fortran_exceptions`.

When passing floating point numbers into PETSc Fortran subroutines, always
make sure you have them marked as double precision (e.g., pass in ``10.d0``
instead of ``10.0`` or declare them as PETSc variables, e.g.
``PetscScalar one = 1.0``). Otherwise, the compiler interprets the input as a single
precision number, which can cause crashes or other mysterious problems.
Make sure to declare all variables (do not use the implicit feature of
Fortran). In fact, we **highly** recommend using the **implicit none**
option at the beginning of each Fortran subroutine you write.

.. _sec_fortran_errors:

Error Checking
^^^^^^^^^^^^^^

In the Fortran version, each PETSc routine has as its final argument an
integer error variable. The error code is set to
be nonzero if an error has been detected; otherwise, it is zero. For
example, the Fortran and C variants of ``KSPSolve()`` are given,
respectively, below, where ``ierr`` denotes the ``PetscErrorCode`` error variable:

.. code-block:: fortran

   call KSPSolve(ksp,b,x,ierr) ! Fortran
   ierr = KSPSolve(ksp,b,x);   // C

For proper error handling one should not use the above syntax instead one should use

.. code-block:: fortran

   PetscCall(KSPSolve(ksp,b,x,ierr))   ! Fortran subroutines
   PetscCallA(KSPSolve(ksp,b,x,ierr))  ! Fortran main program
   PetscCall(KSPSolve(ksp,b,x))        // C




Calling Fortran Routines from C (and C Routines from Fortran)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Different compilers have different methods of naming Fortran routines
called from C (or C routines called from Fortran). Most Fortran
compilers change all the capital letters in Fortran routines to
all lowercase. With some compilers, the Fortran compiler appends an underscore
to the end of each Fortran routine name; for example, the Fortran
routine ``Dabsc()`` would be called from C with ``dabsc_()``. Other
compilers change all the letters in Fortran routine names to capitals.

PETSc provides two macros (defined in C/C++) to help write portable code
that mixes C/C++ and Fortran. They are ``PETSC_HAVE_FORTRAN_UNDERSCORE``
and ``PETSC_HAVE_FORTRAN_CAPS`` , which will be defined in the file
``$PETSC_DIR/$PETSC_ARCH/include/petscconf.h`` based on the compilers
conventions. The macros are used,
for example, as follows:

.. code-block:: fortran

   #if defined(PETSC_HAVE_FORTRAN_CAPS)
   #define dabsc_ DABSC
   #elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
   #define dabsc_ dabsc
   #endif
   .....
   dabsc_( &n,x,y); /* call the Fortran function */

Passing Null Pointers
^^^^^^^^^^^^^^^^^^^^^

Many PETSc C functions have the option of passing a NULL
argument (for example, the fifth argument of ``MatCreateSeqAIJ()``).
From Fortran, users *must* pass ``PETSC_NULL_XXX`` to indicate a null
argument (where ``XXX`` is ``INTEGER``, ``DOUBLE``, ``CHARACTER``,
``SCALAR``, ``VEC``, ``MAT``, etc depending on the argument type); passing a literal 0 from
Fortran in this case will crash the code.  For example, when no options prefix is desired
in the routine ``PetscOptionsGetInt()``, one must use the following
command in Fortran:

.. code-block:: fortran

   call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,PETSC_NULL_CHARACTER,'-name',N,flg,ierr)

.. _sec_fortvecd:

Duplicating Multiple Vectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Fortran interface to ``VecDuplicateVecs()`` differs slightly from
the C/C++ variant because Fortran does not allow conventional arrays to
be returned in routine arguments. To create ``n`` vectors of the same
format as an existing vector, the user must declare a vector array,
``v_new`` of size ``n``. Then, after ``VecDuplicateVecs()`` has been
called, ``v_new`` will contain (pointers to) the new PETSc vector
objects. When finished with the vectors, the user should destroy them by
calling ``VecDestroyVecs()``. For example, the following code fragment
duplicates ``v_old`` to form two new vectors, ``v_new(1)`` and
``v_new(2)``.

.. code-block:: fortran

   Vec          v_old, v_new(2)
   PetscInt     ierr
   PetscScalar  alpha
   ....
   call VecDuplicateVecs(v_old,2,v_new,ierr)
   alpha = 4.3
   call VecSet(v_new(1),alpha,ierr)
   alpha = 6.0
   call VecSet(v_new(2),alpha,ierr)
   ....
   call VecDestroyVecs(2,v_new,ierr)

Matrix, Vector and IS Indices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All matrices, vectors and ``IS`` in PETSc use zero-based indexing,
regardless of whether C or Fortran is being used. The interface
routines, such as ``MatSetValues()`` and ``VecSetValues()``, always use
zero indexing. See :any:`sec_matoptions` for further
details.

Setting Routines
^^^^^^^^^^^^^^^^

When a function pointer is passed as an argument to a PETSc function,
such as the test in ``KSPSetConvergenceTest()``, it is assumed that this
pointer references a routine written in the same language as the PETSc
interface function that was called. For instance, if
``KSPSetConvergenceTest()`` is called from C, the test argument is
assumed to be a C function. Likewise, if it is called from Fortran, the
test is assumed to be written in Fortran.

.. _sec_fortcompile:

Compiling and Linking Fortran Programs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See :any:`sec_writing_application_codes`.

.. _sec_fortran_exceptions:

Routines with Different Fortran Interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following Fortran routines differ slightly from their C
counterparts; see the manual pages and previous discussion in this
chapter for details:

.. code-block:: fortran

   PetscInitialize()
   PetscOptionsGetString()

The following functions are not supported in Fortran:

.. code-block:: fortran

   PetscFClose(), PetscFOpen(), PetscFPrintf(), PetscPrintf()
   PetscPopErrorHandler(), PetscPushErrorHandler()
   PetscInfo()
   PetscSetDebugger()
   VecGetArrays(), VecRestoreArrays()
   PetscViewerASCIIGetPointer(), PetscViewerBinaryGetDescriptor()
   PetscViewerStringOpen(), PetscViewerStringSPrintf()
   PetscOptionsGetStringArray()

.. _sec_fortranarrays:

Routines that Return Fortran Allocatable Arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PETSc includes support for direct use of Fortran allocatable arrays.

.. list-table::
   :header-rows: 1

   * - C-API
     - Fortran-API
   * - ``ISGetIndices()``
     - ``ISGetIndicesF90()``
   * - ``ISRestoreIndices()``
     - ``ISRestoreIndicesF90()``
   * - ``ISLocalToGlobalMappingGetIndices()``
     - ``ISLocalToGlobalMappingGetIndicesF90()``
   * - ``ISLocalToGlobalMappingRestoreIndices()``
     - ``ISLocalToGlobalMappingRestoreIndicesF90()``
   * - ``VecGetArray()``
     - ``VecGetArrayF90()``
   * - ``VecRestoreArray()``
     - ``VecRestoreArrayF90()``
   * - ``VecGetArrayRead()``
     - ``VecGetArrayReadF90()``
   * - ``VecRestoreArrayRead()``
     - ``VecRestoreArrayReadF90()``
   * - ``VecDuplicateVecs()``
     - ``VecDuplicateVecsF90()``
   * - ``VecDestroyVecs()``
     - ``VecDestroyVecsF90()``
   * - ``DMDAVecGetArray()``
     - ``DMDAVecGetArrayF90()``
   * - ``DMDAVecRestoreArray()``
     - ``DMDAVecRestoreArrayF90()``
   * - ``DMDAVecGetArrayRead()``
     - ``DMDAVecGetArrayReadF90()``
   * - ``DMDAVecRestoreArrayRead()``
     - ``DMDAVecRestoreArrayReadF90()``
   * - ``DMDAVecGetArrayWrite()``
     - ``DMDAVecGetArrayWriteF90()``
   * - ``DMDAVecRestoreArrayWrite()``
     - ``DMDAVecRestoreArrayWriteF90()``
   * - ``MatGetRowIJ()``
     - ``MatGetRowIJF90()``
   * - ``MatRestoreRowIJ()``
     - ``MatRestoreRowIJF90()``
   * - ``MatSeqAIJGetArray()``
     - ``MatSeqAIJGetArrayF90()``
   * - ``MatSeqAIJRestoreArray()``
     - ``MatSeqAIJRestoreArrayF90()``
   * - ``MatMPIAIJGetSeqAIJ()``
     - ``MatMPIAIJGetSeqAIJF90()``
   * - ``MatDenseGetArray()``
     - ``MatDenseGetArrayF90()``
   * - ``MatDenseRestoreArray()``
     - ``MatDenseRestoreArrayF90()``

The array arguments to these Fortran functions should be declared with forms such as

.. code-block:: fortran

   PetscScalar, pointer :: x(:)
   PetscInt, pointer :: idx(:)

See the manual pages for details and pointers to example programs.

.. _sec_fortran-examples:

Sample Fortran Programs
~~~~~~~~~~~~~~~~~~~~~~~

Sample programs that illustrate the PETSc interface for Fortran are
given below, corresponding to
`Vec Test ex19f <PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/vec/vec/tests/ex19f.F.html>`__,
`Vec Tutorial ex4f <PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/vec/vec/tutorials/ex4f.F.html>`__,
`Draw Test ex5f <PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/sys/classes/draw/tests/ex5f.F.html>`__,
and
`SNES Tutorial ex1f <PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/snes/tutorials/ex1f.F90.html>`__,
respectively. We also refer Fortran programmers to the C examples listed
throughout the manual, since PETSc usage within the two languages
differs only slightly.


.. admonition:: Listing: ``src/vec/vec/tests/ex19f.F90``
   :name: vec-test-ex19f

   .. literalinclude:: /../src/vec/vec/tests/ex19f.F90
      :language: fortran
      :end-at: end

.. _listing_vec_ex4f:

.. admonition:: Listing: ``src/vec/vec/tutorials/ex4f.F90``
   :name: vec-ex4f

   .. literalinclude:: /../src/vec/vec/tutorials/ex4f.F90
      :language: fortran
      :end-before: !/*TEST

.. admonition:: Listing: ``src/sys/classes/draw/tests/ex5f.F90``
   :name: draw-test-ex5f

   .. literalinclude:: /../src/sys/classes/draw/tests/ex5f.F90
      :language: fortran
      :end-at: end

.. admonition:: Listing: ``src/snes/tutorials/ex1f.F90``
   :name: snes-ex1f

   .. literalinclude:: /../src/snes/tutorials/ex1f.F90
      :language: fortran
      :end-before: !/*TEST


