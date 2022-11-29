#if !defined(MPIAIJKOEKOS_HPP_)
  #define MPIAIJKOEKOS_HPP_

  #include <petscsf.h>
  #include <../src/mat/impls/aij/mpi/mpiaij.h> /*I "petscmat.h" I*/
  #include <../src/mat/impls/aij/seq/kokkos/aijkok.hpp>

struct Mat_MPIAIJKokkos {
  /* MatSetValuesCOO() related stuff on device */
  PetscCountKokkosView Ajmap1_d, Aperm1_d;           /* Local entries to diag */
  PetscCountKokkosView Bjmap1_d, Bperm1_d;           /* Local entries to offdiag */
  PetscCountKokkosView Aimap2_d, Ajmap2_d, Aperm2_d; /* Remote entries to diag */
  PetscCountKokkosView Bimap2_d, Bjmap2_d, Bperm2_d; /* Remote entries to offdiag */
  PetscCountKokkosView Cperm1_d;                     /* Permutation to fill send buffer. 'C' for communication */
  MatScalarKokkosView  sendbuf_d, recvbuf_d;         /* Buffers for remote values in MatSetValuesCOO() */

  Mat_MPIAIJKokkos(const Mat_MPIAIJ *mpiaij) :
    Ajmap1_d(Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(mpiaij->Ajmap1, mpiaij->Annz + 1))),
    Aperm1_d(Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(mpiaij->Aperm1, mpiaij->Atot1))),

    Bjmap1_d(Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(mpiaij->Bjmap1, mpiaij->Bnnz + 1))),
    Bperm1_d(Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(mpiaij->Bperm1, mpiaij->Btot1))),

    Aimap2_d(Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(mpiaij->Aimap2, mpiaij->Annz2))),
    Ajmap2_d(Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(mpiaij->Ajmap2, mpiaij->Annz2 + 1))),
    Aperm2_d(Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(mpiaij->Aperm2, mpiaij->Atot2))),

    Bimap2_d(Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(mpiaij->Bimap2, mpiaij->Bnnz2))),
    Bjmap2_d(Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(mpiaij->Bjmap2, mpiaij->Bnnz2 + 1))),
    Bperm2_d(Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(mpiaij->Bperm2, mpiaij->Btot2))),

    Cperm1_d(Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(mpiaij->Cperm1, mpiaij->sendlen))),

    sendbuf_d(Kokkos::create_mirror_view(DefaultMemorySpace(), MatScalarKokkosViewHost(mpiaij->sendbuf, mpiaij->sendlen))),
    recvbuf_d(Kokkos::create_mirror_view(DefaultMemorySpace(), MatScalarKokkosViewHost(mpiaij->recvbuf, mpiaij->recvlen)))
  {
  }
};

#endif
