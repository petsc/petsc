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

#endif
