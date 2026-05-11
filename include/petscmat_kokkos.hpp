#pragma once

#include <petscconf.h>

/* SUBMANSEC = Mat */

#if defined(PETSC_HAVE_KOKKOS)

  #include <Kokkos_Core.hpp>
  #include <petscmat.h>

/*@C
   MatCreateSeqAIJKokkosWithKokkosViews - Creates a MATSEQAIJKOKKOS matrix with Kokkos views of the aij data

   Synopsis:
   #include <petscmat_kokkos.hpp>
   PetscErrorCode MatCreateSeqAIJKokkosWithKokkosViews  (MPI_Comm comm, PetscInt m, PetscInt n, Kokkos::View<PetscInt *>& i_d, Kokkos::View<PetscInt *>& j_d, Kokkos::View<PetscScalar *>& a_d, Mat *A);

   Logically Collective, No Fortran Support

   Input Parameter:
+  comm  - the MPI communicator
-  m     - row size
-  n     - the column size
-  i     - the Kokkos view of row data (in Kokkos::DefaultExecutionSpace)
-  j     - the Kokkos view of the column data (in Kokkos::DefaultExecutionSpace)
-  a     - the Kokkos view of the values (in Kokkos::DefaultExecutionSpace)

   Output Parameter:
.  A  - the `MATSEQAIJKOKKOS` matrix

   Level: intermediate

   Notes:
   Creates a Mat given the csr data input as Kokkos views. This routine allows a Mat
   to be built without involving the host. Don't modify entries in the views after this routine.
   There should be no outstanding asynchronous operations on the views (ie this routine does not call fence()
   before using the views)

.seealso:
@*/
PetscErrorCode MatCreateSeqAIJKokkosWithKokkosViews(MPI_Comm, PetscInt, PetscInt, Kokkos::View<PetscInt *> &, Kokkos::View<PetscInt *> &, Kokkos::View<PetscScalar *> &, Mat *);

/*@C
   MatSeqAIJGetKokkosView - Returns a Kokkos View of the matrix nonzero values on the device, with up-to-date data.

   Not Collective, No Fortran Support

   Input Parameter:
.  A - the matrix in type of `MATSEQAIJKOKKOS`

   Output Parameter:
.  kv - the Kokkos View over the matrix nonzero values on the device

   Level: intermediate

   Notes:
   If the matrix is not of type `MATSEQAIJKOKKOS`, an error will be raised.

   Passing in a const View enables read-only access; passing in a non-const View enables read-write access. In the
   read-write case, the matching `MatSeqAIJRestoreKokkosView()` marks the device side as modified so subsequent
   operations see the new values.

   One must return the View with a matching `MatSeqAIJRestoreKokkosView()` after finishing using the View.

.seealso: `MatSeqAIJRestoreKokkosView()`, `MatSeqAIJGetKokkosViewWrite()`, `MatSeqAIJGetCSRAndMemType()`, `MatCreateSeqAIJKokkosWithKokkosViews()`, `VecGetKokkosView()`
@*/
PetscErrorCode MatSeqAIJGetKokkosView(Mat, Kokkos::View<const PetscScalar *> *);
PetscErrorCode MatSeqAIJGetKokkosView(Mat, Kokkos::View<PetscScalar *> *);

/*@C
   MatSeqAIJRestoreKokkosView - Returns a Kokkos View obtained with `MatSeqAIJGetKokkosView()`.

   Not Collective, No Fortran Support

   Input Parameters:
+  A  - the matrix in type of `MATSEQAIJKOKKOS`
-  kv - the Kokkos View previously obtained with `MatSeqAIJGetKokkosView()`

   Level: intermediate

   Notes:
   If the matrix is not of type `MATSEQAIJKOKKOS`, an error will be raised.

   When the matching `MatSeqAIJGetKokkosView()` returned a non-const View, the device side of the matrix is marked
   modified so subsequent PETSc operations see the new values. The const overload is a no-op.

.seealso: `MatSeqAIJGetKokkosView()`, `MatSeqAIJRestoreKokkosViewWrite()`, `MatSeqAIJGetKokkosViewWrite()`
@*/
PetscErrorCode MatSeqAIJRestoreKokkosView(Mat, Kokkos::View<const PetscScalar *> *);
PetscErrorCode MatSeqAIJRestoreKokkosView(Mat, Kokkos::View<PetscScalar *> *);

/*@C
   MatSeqAIJGetKokkosViewWrite - Returns a Kokkos View of the matrix nonzero values on the device for write-only access.

   Not Collective, No Fortran Support

   Input Parameter:
.  A - the matrix in type of `MATSEQAIJKOKKOS`

   Output Parameter:
.  kv - the Kokkos View over the matrix nonzero values on the device

   Level: intermediate

   Notes:
   If the matrix is not of type `MATSEQAIJKOKKOS`, an error will be raised.

   This routine does not synchronize the device with the host first. The caller is expected to overwrite all entries
   in the View; reading from it may return stale or garbage data.

   One must return the View with a matching `MatSeqAIJRestoreKokkosViewWrite()` after finishing using the View.

.seealso: `MatSeqAIJRestoreKokkosViewWrite()`, `MatSeqAIJGetKokkosView()`, `MatSeqAIJRestoreKokkosView()`
@*/
PetscErrorCode MatSeqAIJGetKokkosViewWrite(Mat, Kokkos::View<PetscScalar *> *);

/*@C
   MatSeqAIJRestoreKokkosViewWrite - Returns a Kokkos View obtained with `MatSeqAIJGetKokkosViewWrite()`.

   Not Collective, No Fortran Support

   Input Parameters:
+  A  - the matrix in type of `MATSEQAIJKOKKOS`
-  kv - the Kokkos View previously obtained with `MatSeqAIJGetKokkosViewWrite()`

   Level: intermediate

   Notes:
   If the matrix is not of type `MATSEQAIJKOKKOS`, an error will be raised.

   The device side of the matrix is marked modified so subsequent PETSc operations see the new values.

.seealso: `MatSeqAIJGetKokkosViewWrite()`, `MatSeqAIJGetKokkosView()`, `MatSeqAIJRestoreKokkosView()`
@*/
PetscErrorCode MatSeqAIJRestoreKokkosViewWrite(Mat, Kokkos::View<PetscScalar *> *);

#endif
