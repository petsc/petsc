.. _chapter_fortran:

PETSc for Fortran Users
-----------------------

Most of the functionality of PETSc can be obtained by people who program
purely in Fortran.

Synopsis
~~~~~~~~

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

C vs. Fortran Interfaces
~~~~~~~~~~~~~~~~~~~~~~~~

Only a few differences exist between the C and Fortran PETSc interfaces,
are due to Fortran syntax differences. All Fortran routines have the
same names as the corresponding C versions, and PETSc command line
options are fully supported. The routine arguments follow the usual
Fortran conventions; the user need not worry about passing pointers or
values. The calling sequences for the Fortran version are in most cases
identical to the C version, except for the error checking variable
discussed in :any:`sec_fortran_errors` and a few routines
listed in :any:`sec_fortran_exceptions`.

.. _sec_fortran_includes:

Fortran Include Files
^^^^^^^^^^^^^^^^^^^^^

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

.. _sec_fortran_errors:

Error Checking
^^^^^^^^^^^^^^

In the Fortran version, each PETSc routine has as its final argument an
integer error variable, in contrast to the C convention of providing the
error variable as the routine’s return value. The error code is set to
be nonzero if an error has been detected; otherwise, it is zero. For
example, the Fortran and C variants of ``KSPSolve()`` are given,
respectively, below, where ``ierr`` denotes the error variable:

.. code-block:: fortran

   call KSPSolve(ksp,b,x,ierr) ! Fortran
   ierr = KSPSolve(ksp,b,x);   /* C */

Fortran programmers can check these error codes with ``PetscCall(ierr)``,
which terminates all processes when an error is encountered. Likewise,
one can set error codes within Fortran programs by using
``SETERRQ(comm,p,' ')``, which again terminates all processes upon
detection of an error. Note that complete error tracebacks with
``PetscCall()`` and ``SETERRQ()``, as described in
:any:`sec_simple` for C routines, are *not* directly supported for
Fortran routines; however, Fortran programmers can easily use the error
codes in writing their own tracebacks. For example, one could use code
such as the following:

.. code-block:: fortran

   call KSPSolve(ksp,b,x,ierr)
   if (ierr .ne. 0) then
      print*, 'Error in routine ...'
      return
   end if

Calling Fortran Routines from C (and C Routines from Fortran)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Different machines have different methods of naming Fortran routines
called from C (or C routines called from Fortran). Most Fortran
compilers change all the capital letters in Fortran routines to
lowercase. On some machines, the Fortran compiler appends an underscore
to the end of each Fortran routine name; for example, the Fortran
routine ``Dabsc()`` would be called from C with ``dabsc_()``. Other
machines change all the letters in Fortran routine names to capitals.

PETSc provides two macros (defined in C/C++) to help write portable code
that mixes C/C++ and Fortran. They are ``PETSC_HAVE_FORTRAN_UNDERSCORE``
and ``PETSC_HAVE_FORTRAN_CAPS`` , which are defined in the file
``$PETSC_DIR/$PETSC_ARCH/include/petscconf.h``. The macros are used,
for example, as follows:

.. code-block:: fortran

   #if defined(PETSC_HAVE_FORTRAN_CAPS)
   #define dabsc_ DMDABSC
   #elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
   #define dabsc_ dabsc
   #endif
   .....
   dabsc_( &n,x,y); /* call the Fortran function */

Passing Null Pointers
^^^^^^^^^^^^^^^^^^^^^

In several PETSc C functions, one has the option of passing a NULL (0)
argument (for example, the fifth argument of ``MatCreateSeqAIJ()``).
From Fortran, users *must* pass ``PETSC_NULL_XXX`` to indicate a null
argument (where ``XXX`` is ``INTEGER``, ``DOUBLE``, ``CHARACTER``, or
``SCALAR`` depending on the type of argument required); passing 0 from
Fortran will crash the code. Note that the C convention of passing NULL
(or 0) *cannot* be used. For example, when no options prefix is desired
in the routine ``PetscOptionsGetInt()``, one must use the following
command in Fortran:

.. code-block:: fortran

   call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,PETSC_NULL_CHARACTER,'-name',N,flg,ierr)

This Fortran requirement is inconsistent with C, where the user can
employ ``NULL`` for all null arguments.

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

   PetscInitialize(char *filename,int ierr)
   PetscError(MPI_COMM,int err,char *message,int ierr)
   VecGetArray(), MatDenseGetArray()
   ISGetIndices(),
   VecDuplicateVecs(), VecDestroyVecs()
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

PETSc includes some support for direct use of Fortran90 pointers.
Current routines include:

.. code-block:: fortran

   VecGetArrayF90(), VecRestoreArrayF90()
   VecGetArrayReadF90(), VecRestoreArrayReadF90()
   VecDuplicateVecsF90(), VecDestroyVecsF90()
   DMDAVecGetArrayF90(), DMDAVecGetArrayReadF90(), ISLocalToGlobalMappingGetIndicesF90()
   MatDenseGetArrayF90(), MatDenseRestoreArrayF90()
   ISGetIndicesF90(), ISRestoreIndicesF90()

See the manual pages for details and pointers to example programs.

.. _sec_fortran-examples:

Sample Fortran Programs
~~~~~~~~~~~~~~~~~~~~~~~

Sample programs that illustrate the PETSc interface for Fortran are
given below, corresponding to
`Vec Test ex19f <../../../src/vec/vec/tests/ex19f.F.html>`__,
`Vec Tutorial ex4f <../../../src/vec/vec/tutorials/ex4f.F.html>`__,
`Draw Test ex5f <../../../src/sys/classes/draw/tests/ex5f.F.html>`__,
and
`SNES Tutorial ex1f <../../../src/snes/tutorials/ex1f.F90.html>`__,
respectively. We also refer Fortran programmers to the C examples listed
throughout the manual, since PETSc usage within the two languages
differs only slightly.


.. admonition:: Listing: ``src/vec/vec/tests/ex19f.F``
   :name: vec-test-ex19f

   .. literalinclude:: /../src/vec/vec/tests/ex19f.F
      :language: fortran

.. _listing_vec_ex4f:

.. admonition:: Listing: ``src/vec/vec/tutorials/ex4f.F``
   :name: vec-ex4f

   .. literalinclude:: /../src/vec/vec/tutorials/ex4f.F
      :language: fortran

.. admonition:: Listing: ``src/sys/classes/draw/tests/ex5f.F``
   :name: draw-test-ex5f

   .. literalinclude:: /../src/sys/classes/draw/tests/ex5f.F
      :language: fortran

.. admonition:: Listing: ``src/snes/tutorials/ex1f.F90``
   :name: snes-ex1f

   .. literalinclude:: /../src/snes/tutorials/ex1f.F90
      :language: fortran

.. _sec_fortranarrays:

Array Arguments
^^^^^^^^^^^^^^^

This material is no longer relevant since one should use
``VecGetArrayF90()`` and the other routines that utilize Fortran
pointers, instead of the code below, but it is included for historical
reasons and because many of the Fortran examples still utilize the old
approach.

Since Fortran 77 does not allow arrays to be returned in routine
arguments, all PETSc routines that return arrays, such as
``VecGetArray()``, ``MatDenseGetArray()``, and ``ISGetIndices()``, are
defined slightly differently in Fortran than in C. Instead of returning
the array itself, these routines accept as input a user-specified array
of dimension one and return an integer index to the actual array used
for data storage within PETSc. The Fortran interface for several
routines is as follows:

.. code-block:: fortran

   PetscScalar    xx_v(1), aa_v(1)
   PetscErrorCode ierr
   PetscInt       ss_v(1), dd_v(1), nloc
   PetscOffset    ss_i, xx_i, aa_i, dd_i
   Vec            x
   Mat            A
   IS             s
   DM             d

   call VecGetArray(x,xx_v,xx_i,ierr)
   call MatDenseGetArray(A,aa_v,aa_i,ierr)
   call ISGetIndices(s,ss_v,ss_i,ierr)

To access array elements directly, both the user-specified array and the
integer index *must* then be used together. For example, the following
Fortran program fragment illustrates directly setting the values of a
vector array instead of using ``VecSetValues()``. Note the (optional)
use of the preprocessor ``#define`` statement to enable array
manipulations in the conventional Fortran manner.

.. code-block:: fortran

   #define xx_a(ib)  xx_v(xx_i + (ib))

      double precision xx_v(1)
      PetscOffset      xx_i
      PetscErrorCode   ierr
      PetscInt         i, n
      Vec              x
      call VecGetArray(x,xx_v,xx_i,ierr)
      call VecGetLocalSize(x,n,ierr)
      do 10, i=1,n
        xx_a(i) = 3*i + 1
   10 continue
      call VecRestoreArray(x,xx_v,xx_i,ierr)

:ref:`The Vec ex4f Tutorial listed above <listing_vec_ex4f>` contains an example of
using ``VecGetArray()`` within a Fortran routine.

Since in this case the array is accessed directly from Fortran, indexing
begins with 1, not 0 (unless the array is declared as ``xx_v(0:1)``).
This is different from the use of ``VecSetValues()`` where, indexing
always starts with 0.

*Note*: If using ``VecGetArray()``, ``MatDenseGetArray()``, or
``ISGetIndices()``, from Fortran, the user *must not* compile the
Fortran code with options to check for “array entries out of bounds”
(e.g., on the IBM RS/6000 this is done with the ``-C`` compiler option,
so never use the ``-C`` option with this).
