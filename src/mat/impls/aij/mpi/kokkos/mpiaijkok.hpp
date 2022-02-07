#if !defined(MPIAIJKOEKOS_HPP_)
#define MPIAIJKOEKOS_HPP_

#include <petscsf.h>
#include <../src/mat/impls/aij/seq/kokkos/aijkok.hpp>

struct Mat_MPIAIJKokkos {
  /* MatSetValuesCOO() related stuff */
  PetscCount          coo_n; /* Number of COOs used in MatSetPreallocationCOO)() */
  PetscSF             coo_sf; /* SF to send/recv remote values in MatSetValuesCOO() */
  PetscCount          Annz1,Annz2,Bnnz1,Bnnz2; /* See comments in MatSetPreallocationCOO_MPIAIJKokkos() */
  MatRowMapKokkosView Aimap1_d,Ajmap1_d,Aperm1_d; /* Local entries to diag */
  MatRowMapKokkosView Bimap1_d,Bjmap1_d,Bperm1_d; /* Local entries to offdiag */
  MatRowMapKokkosView Aimap2_d,Ajmap2_d,Aperm2_d; /* Remote entries to diag */
  MatRowMapKokkosView Bimap2_d,Bjmap2_d,Bperm2_d; /* Remote entries to offdiag */
  MatRowMapKokkosView Cperm1_d; /* Permutation to fill send buffer. 'C' for communication */
  MatScalarKokkosView sendbuf_d,recvbuf_d; /* Buffers for remote values in MatSetValuesCOO() */

  Mat_MPIAIJKokkos(PetscCount n,PetscSF sf,PetscInt nroots,PetscInt nleaves,PetscCount Annz1,PetscCount Annz2,PetscCount Bnnz1,PetscCount Bnnz2,
                   MatRowMapKokkosViewHost& Aimap1_h,MatRowMapKokkosViewHost& Aimap2_h,MatRowMapKokkosViewHost& Bimap1_h,MatRowMapKokkosViewHost& Bimap2_h,
                   MatRowMapKokkosViewHost& Ajmap1_h,MatRowMapKokkosViewHost& Ajmap2_h,MatRowMapKokkosViewHost& Bjmap1_h,MatRowMapKokkosViewHost& Bjmap2_h,
                   MatRowMapKokkosViewHost& Aperm1_h,MatRowMapKokkosViewHost& Aperm2_h,MatRowMapKokkosViewHost& Bperm1_h,MatRowMapKokkosViewHost& Bperm2_h,MatRowMapKokkosViewHost& Cperm1_h)
    : coo_n(n),coo_sf(sf),Annz1(Annz1),Annz2(Annz2),Bnnz1(Bnnz1),Bnnz2(Bnnz2),
      Aimap1_d(Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),Aimap1_h)),
      Ajmap1_d(Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),Ajmap1_h)),
      Aperm1_d(Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),Aperm1_h)),
      Bimap1_d(Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),Bimap1_h)),
      Bjmap1_d(Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),Bjmap1_h)),
      Bperm1_d(Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),Bperm1_h)),
      Aimap2_d(Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),Aimap2_h)),
      Ajmap2_d(Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),Ajmap2_h)),
      Aperm2_d(Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),Aperm2_h)),
      Bimap2_d(Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),Bimap2_h)),
      Bjmap2_d(Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),Bjmap2_h)),
      Bperm2_d(Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),Bperm2_h)),
      Cperm1_d(Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),Cperm1_h)),
      sendbuf_d("sendbuf",nleaves),recvbuf_d("recvbuf",nroots)
  {
    PetscObjectReference((PetscObject)sf);
  }

  ~Mat_MPIAIJKokkos() {
    PetscSFDestroy(&coo_sf);
    /* Kokkos views are auto refcounted and destroyed */
  }
};

#endif
