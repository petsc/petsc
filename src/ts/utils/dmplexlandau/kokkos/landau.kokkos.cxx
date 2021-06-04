/*
  Implements the Kokkos kernel
*/
#include <petscconf.h>
#include <petscvec_kokkos.hpp>
#include <petsc/private/dmpleximpl.h>   /*I   "petscdmplex.h"   I*/
#include <petsclandau.h>
#include <petscts.h>

#include <Kokkos_Core.hpp>
#include <cstdio>
typedef Kokkos::TeamPolicy<>::member_type team_member;
#define PETSC_DEVICE_FUNC_DECL KOKKOS_INLINE_FUNCTION
#include "../land_tensors.h"
#define atomicAdd(e, f) Kokkos::atomic_fetch_add(e, f)
#include <petscaijdevice.h>
#undef atomicAdd

namespace landau_inner_red {  // namespace helps with name resolution in reduction identity
  template< class ScalarType >
  struct array_type {
    ScalarType gg2[LANDAU_DIM];
    ScalarType gg3[LANDAU_DIM][LANDAU_DIM];

    KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
    array_type() {
      for (int j = 0; j < LANDAU_DIM; j++) {
        gg2[j] = 0;
        for (int k = 0; k < LANDAU_DIM; k++) {
          gg3[j][k] = 0;
        }
      }
    }
    KOKKOS_INLINE_FUNCTION   // Copy Constructor
    array_type(const array_type & rhs) {
      for (int j = 0; j < LANDAU_DIM; j++) {
        gg2[j] = rhs.gg2[j];
        for (int k = 0; k < LANDAU_DIM; k++) {
          gg3[j][k] = rhs.gg3[j][k];
        }
      }
    }
    KOKKOS_INLINE_FUNCTION   // add operator
    array_type& operator += (const array_type& src)
    {
      for (int j = 0; j < LANDAU_DIM; j++) {
        gg2[j] += src.gg2[j];
        for (int k = 0; k < LANDAU_DIM; k++) {
          gg3[j][k] += src.gg3[j][k];
        }
      }
      return *this;
    }
    KOKKOS_INLINE_FUNCTION   // volatile add operator
    void operator += (const volatile array_type& src) volatile
    {
      for (int j = 0; j < LANDAU_DIM; j++) {
        gg2[j] += src.gg2[j];
        for (int k = 0; k < LANDAU_DIM; k++) {
          gg3[j][k] += src.gg3[j][k];
        }
      }
    }
  };
  typedef array_type<PetscReal> TensorValueType;  // used to simplify code below
}

namespace Kokkos { //reduction identity must be defined in Kokkos namespace
  template<>
  struct reduction_identity< landau_inner_red::TensorValueType > {
    KOKKOS_FORCEINLINE_FUNCTION static landau_inner_red::TensorValueType sum() {
      return landau_inner_red::TensorValueType();
    }
  };
}

extern "C"  {
  PetscErrorCode LandauKokkosCreateMatMaps(P4estVertexMaps *maps, pointInterpolationP4est (*pointMaps)[LANDAU_MAX_Q_FACE], PetscInt Nf, PetscInt Nq)
  {
    P4estVertexMaps   h_maps;  /* host container */
    const Kokkos::View<pointInterpolationP4est*[LANDAU_MAX_Q_FACE], Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >    h_points ((pointInterpolationP4est*)pointMaps, maps->num_reduced);
    const Kokkos::View< LandauIdx*[LANDAU_MAX_SPECIES][LANDAU_MAX_NQ], Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_gidx ((LandauIdx*)maps->gIdx, maps->num_elements);
    Kokkos::View<pointInterpolationP4est*[LANDAU_MAX_Q_FACE], Kokkos::LayoutRight>   *d_points = new Kokkos::View<pointInterpolationP4est*[LANDAU_MAX_Q_FACE], Kokkos::LayoutRight>("points", maps->num_reduced);
    Kokkos::View<LandauIdx*[LANDAU_MAX_SPECIES][LANDAU_MAX_NQ], Kokkos::LayoutRight> *d_gidx = new Kokkos::View<LandauIdx*[LANDAU_MAX_SPECIES][LANDAU_MAX_NQ], Kokkos::LayoutRight>("gIdx", maps->num_elements);

    PetscFunctionBegin;
    Kokkos::deep_copy (*d_gidx, h_gidx);
    Kokkos::deep_copy (*d_points, h_points);
    h_maps.num_elements = maps->num_elements;
    h_maps.num_face = maps->num_face;
    h_maps.num_reduced = maps->num_reduced;
    h_maps.deviceType = maps->deviceType;
    h_maps.Nf = Nf;
    h_maps.Nq = Nq;
    h_maps.c_maps = (pointInterpolationP4est (*)[LANDAU_MAX_Q_FACE]) d_points->data();
    maps->vp1 = (void*)d_points;
    h_maps.gIdx = (LandauIdx (*)[LANDAU_MAX_SPECIES][LANDAU_MAX_NQ]) d_gidx->data();
    maps->vp2 = (void*)d_gidx;
    {
      Kokkos::View<P4estVertexMaps, Kokkos::HostSpace> h_maps_k(&h_maps);
      Kokkos::View<P4estVertexMaps>                    *d_maps_k = new Kokkos::View<P4estVertexMaps>(Kokkos::create_mirror(Kokkos::DefaultExecutionSpace::memory_space(),h_maps_k));
      Kokkos::deep_copy (*d_maps_k, h_maps_k);
      maps->data = d_maps_k->data();
      maps->vp3 = (void*)d_maps_k;
    }
    PetscFunctionReturn(0);
  }
  PetscErrorCode LandauKokkosDestroyMatMaps(P4estVertexMaps *maps)
  {
    Kokkos::View<pointInterpolationP4est*[LANDAU_MAX_Q_FACE], Kokkos::LayoutRight>  *a = (Kokkos::View<pointInterpolationP4est*[LANDAU_MAX_Q_FACE], Kokkos::LayoutRight>*)maps->vp1;
    Kokkos::View<LandauIdx*[LANDAU_MAX_SPECIES][LANDAU_MAX_NQ], Kokkos::LayoutRight>*b = (Kokkos::View<LandauIdx*[LANDAU_MAX_SPECIES][LANDAU_MAX_NQ], Kokkos::LayoutRight>*)maps->vp2;
    Kokkos::View<P4estVertexMaps*>                                                  *c = (Kokkos::View<P4estVertexMaps*>*)maps->vp3;
    delete a;  delete b;  delete c;
    PetscFunctionReturn(0);
  }

  PetscErrorCode LandauKokkosStaticDataSet(DM plex, const PetscInt Nq, PetscReal nu_alpha[], PetscReal nu_beta[], PetscReal a_invMass[], PetscReal a_invJ[], PetscReal a_mass_w[],
                                           PetscReal a_x[], PetscReal a_y[], PetscReal a_z[], PetscReal a_w[], LandauGeomData *SData_d)
  {
    PetscReal       *BB,*DD;
    PetscErrorCode  ierr;
    PetscTabulation *Tf;
    LandauCtx       *ctx;
    PetscInt        *Nbf,dim,Nf,Nb,nip,cStart,cEnd;
    PetscDS         prob;

    PetscFunctionBegin;
    ierr = DMGetApplicationContext(plex, &ctx);CHKERRQ(ierr);
    if (!ctx) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
    ierr = DMGetDimension(plex, &dim);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
    nip = (cEnd - cStart)*Nq;
    ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
    ierr = PetscDSGetDimensions(prob, &Nbf);CHKERRQ(ierr); Nb = Nbf[0];
    if (Nq != Nb) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Nq != Nb. %D  %D",Nq,Nb);
    if (LANDAU_DIM != dim) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "dim %D != LANDAU_DIM %d",dim,LANDAU_DIM);
    ierr = PetscDSGetTabulation(prob, &Tf);CHKERRQ(ierr);
    BB   = Tf[0]->T[0]; DD = Tf[0]->T[1];
    ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
    {
      const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_alpha (nu_alpha, Nf);
      auto alpha = new Kokkos::View<PetscReal*, Kokkos::LayoutLeft> ("alpha", Nf);
      SData_d->alpha = static_cast<void*>(alpha);
      const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_beta (nu_beta, Nf);
      auto beta = new Kokkos::View<PetscReal*, Kokkos::LayoutLeft> ("beta", Nf);
      SData_d->beta = static_cast<void*>(beta);
      const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_invMass (a_invMass,Nf);
      auto invMass = new Kokkos::View<PetscReal*, Kokkos::LayoutLeft> ("invMass", Nf);
      SData_d->invMass = static_cast<void*>(invMass);
      const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_BB (BB,Nq*Nb);
      auto B = new Kokkos::View<PetscReal*, Kokkos::LayoutLeft> ("B", Nq*Nb);
      SData_d->B = static_cast<void*>(B);
      const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_DD (DD,Nq*Nb*dim);
      auto D = new Kokkos::View<PetscReal*, Kokkos::LayoutLeft> ("D", Nq*Nb*dim);
      SData_d->D = static_cast<void*>(D);
      const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_mass_w (a_mass_w, nip);
      auto mass_w = new Kokkos::View<PetscReal*, Kokkos::LayoutLeft> ("mass_w", nip);
      SData_d->mass_w = static_cast<void*>(mass_w);
      const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_invJ (a_invJ, nip*dim*dim);
      auto invJ = new Kokkos::View<PetscReal*, Kokkos::LayoutLeft> ("invJ", nip*dim*dim);
      SData_d->invJ = static_cast<void*>(invJ);
      const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_x (a_x, nip);
      auto x = new Kokkos::View<PetscReal*, Kokkos::LayoutLeft> ("x", nip);
      SData_d->x = static_cast<void*>(x);
      const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_y (a_y, nip);
      auto y = new Kokkos::View<PetscReal*, Kokkos::LayoutLeft> ("y", nip);
      SData_d->y = static_cast<void*>(y);
      const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_w (a_w, nip);
      auto w = new Kokkos::View<PetscReal*, Kokkos::LayoutLeft> ("w", nip);
      SData_d->w = static_cast<void*>(w);
      const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_z (a_z , dim==3 ? nip : 0);
      auto z = new Kokkos::View<PetscReal*, Kokkos::LayoutLeft> ("z", dim==3 ? nip : 0);
      SData_d->z = static_cast<void*>(z);

      Kokkos::deep_copy (*mass_w, h_mass_w);
      Kokkos::deep_copy (*alpha, h_alpha);
      Kokkos::deep_copy (*beta, h_beta);
      Kokkos::deep_copy (*invMass, h_invMass);
      Kokkos::deep_copy (*B, h_BB);
      Kokkos::deep_copy (*D, h_DD);
      Kokkos::deep_copy (*invJ, h_invJ);
      Kokkos::deep_copy (*x, h_x);
      Kokkos::deep_copy (*y, h_y);
      Kokkos::deep_copy (*z, h_z);
      Kokkos::deep_copy (*w, h_w);

      auto Eq_m = new Kokkos::View<PetscReal*, Kokkos::LayoutLeft> ("Eq_m",Nf);
      SData_d->Eq_m = static_cast<void*>(Eq_m);
      auto IPf = new Kokkos::View<PetscScalar*, Kokkos::LayoutLeft> ("IPf",nip*Nf); // Nq==Nb
      SData_d->IPf = static_cast<void*>(IPf);
    }
    PetscFunctionReturn(0);
  }

  PetscErrorCode LandauKokkosStaticDataClear(LandauGeomData *SData_d)
  {
    PetscFunctionBegin;
    if (SData_d->alpha) {
      auto alpha = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->alpha);
      delete alpha;
      auto beta = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->beta);
      delete beta;
      auto invMass = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->invMass);
      delete invMass;
      auto B = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->B);
      delete B;
      auto D = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->D);
      delete D;
      auto mass_w = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->mass_w);
      delete mass_w;
      auto invJ = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->invJ);
      delete invJ;
      auto x = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->x);
      delete x;
      auto y = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->y);
      delete y;
      auto z = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->z);
      delete z;
      auto w = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->w);
      delete w;
      auto Eq_m = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->Eq_m);
      delete Eq_m;
      if (SData_d->IPf) {
        auto IPf = static_cast<Kokkos::View<PetscScalar*, Kokkos::LayoutLeft>*>(SData_d->IPf);
        delete IPf;
      }
    }
    PetscFunctionReturn(0);
  }

  PetscErrorCode LandauKokkosJacobian(DM plex, const PetscInt Nq, PetscReal a_Eq_m[], PetscScalar a_IPf[],  const PetscInt N, const PetscScalar a_xarray[], LandauGeomData *SData_d,
                                      const PetscInt num_sub_blocks, PetscReal shift, const PetscLogEvent events[], Mat JacP)
  {
    using scr_mem_t = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using g2_scr_t = Kokkos::View<PetscReal***, Kokkos::LayoutRight, scr_mem_t>;
    using g3_scr_t = Kokkos::View<PetscReal****, Kokkos::LayoutRight, scr_mem_t>;
    PetscErrorCode    ierr;
    PetscInt          *Nbf,Nb,cStart,cEnd,Nf,dim,numCells,totDim,global_elem_mat_sz,nip;
    PetscDS           prob;
    LandauCtx         *ctx;
    PetscReal         *d_Eq_m=NULL;
    PetscScalar       *d_IPf=NULL;
    P4estVertexMaps   *d_maps=NULL;
    PetscSplitCSRDataStructure *d_mat=NULL;
    const int         conc = Kokkos::DefaultExecutionSpace().concurrency(), openmp = !!(conc < 1000), team_size = (openmp==0) ? Nq : 1;
    int               scr_bytes;
    auto              d_alpha_k = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->alpha); //static data
    const PetscReal   *d_alpha = d_alpha_k->data();
    auto              d_beta_k = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->beta);
    const PetscReal   *d_beta = d_beta_k->data();
    auto              d_invMass_k = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->invMass);
    const PetscReal   *d_invMass = d_invMass_k->data();
    auto              d_B_k = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->B);
    const PetscReal   *d_BB = d_B_k->data();
    auto              d_D_k = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->D);
    const PetscReal   *d_DD = d_D_k->data();
    auto              d_mass_w_k = *static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->mass_w);
    auto              d_invJ_k = *static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->invJ); // use Kokkos vector in kernels
    auto              d_x_k = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->x); //static data
    const PetscReal   *d_x = d_x_k->data();
    auto              d_y_k = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->y); //static data
    const PetscReal   *d_y = d_y_k->data();
    auto              d_z_k = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->z); //static data
    const PetscReal   *d_z = d_z_k->data();
    auto              d_w_k = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->w); //static data
    const PetscReal   *d_w = d_w_k->data();
    // dynamic data pointers
    auto              d_Eq_m_k = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->Eq_m); //static data
    auto              d_IPf_k  = static_cast<Kokkos::View<PetscScalar*, Kokkos::LayoutLeft>*>(SData_d->IPf); //static data

    PetscFunctionBegin;
    ierr = PetscLogEventBegin(events[3],0,0,0,0);CHKERRQ(ierr);
    ierr = DMGetApplicationContext(plex, &ctx);CHKERRQ(ierr);
    if (!ctx) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
    ierr = DMGetDimension(plex, &dim);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
    numCells = cEnd - cStart;
    nip = numCells*Nq;
    ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
    ierr = PetscDSGetDimensions(prob, &Nbf);CHKERRQ(ierr); Nb = Nbf[0];
    if (Nq != Nb) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Nq != Nb. %D  %D",Nq,Nb);
    if (LANDAU_DIM != dim) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "dim %D != LANDAU_DIM %d",dim,LANDAU_DIM);
    scr_bytes = 2*(g2_scr_t::shmem_size(dim,Nf,Nq) + g3_scr_t::shmem_size(dim,dim,Nf,Nq));
    ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
    if (ctx->gpu_assembly) {
      PetscContainer container;
      ierr = PetscObjectQuery((PetscObject) JacP, "assembly_maps", (PetscObject *) &container);CHKERRQ(ierr);
      if (container) { // not here first call
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
        P4estVertexMaps   *h_maps=NULL;
        ierr = PetscContainerGetPointer(container, (void **) &h_maps);CHKERRQ(ierr);
        if (h_maps->data) {
          d_maps = h_maps->data;
        } else {
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "GPU assembly but no metadata in container");
        }
        // this does the setup the first time called
        ierr = MatKokkosGetDeviceMatWrite(JacP,&d_mat);CHKERRQ(ierr);
        global_elem_mat_sz = 0;
#else
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "GPU assembly w/o kokkos kernels -- should not be here");
#endif
      } else { // kernel output - first call assembled on device
        global_elem_mat_sz = numCells;
      }
    } else {
      global_elem_mat_sz = numCells; // no device assembly
    }
    ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
    ierr = PetscLogEventEnd(events[3],0,0,0,0);CHKERRQ(ierr);
    {
#if defined(LANDAU_LAYOUT_LEFT)
      Kokkos::View<PetscReal***, Kokkos::LayoutLeft >                               d_fdf_k("df", dim+1, Nf, (a_IPf || a_xarray) ? nip : 0);
#else
      Kokkos::View<PetscReal***, Kokkos::LayoutRight >                              d_fdf_k("df", dim+1, Nf, (a_IPf || a_xarray) ? nip : 0);
#endif
      const Kokkos::View<PetscScalar*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >h_IPf_k  (a_IPf,  (a_IPf || a_xarray) ? nip*Nf : 0);
      const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >  h_Eq_m_k (a_Eq_m, (a_IPf || a_xarray) ? Nf : 0);
      Kokkos::View<PetscScalar**, Kokkos::LayoutRight>                                                                 d_elem_mats("element matrices", global_elem_mat_sz, totDim*totDim);
      // copy dynamic data to device
      if (a_IPf || a_xarray) {  // form f and df
        static int cc=0;
        ierr = PetscLogEventBegin(events[1],0,0,0,0);CHKERRQ(ierr);
        Kokkos::deep_copy (*d_Eq_m_k, h_Eq_m_k);
        d_Eq_m = d_Eq_m_k->data();
        if (a_IPf) {
          Kokkos::deep_copy (*d_IPf_k, h_IPf_k);
          d_IPf  = d_IPf_k->data();
        } else {
          d_IPf = (PetscScalar*)a_xarray;
        }
        ierr = PetscLogEventEnd(events[1],0,0,0,0);CHKERRQ(ierr);
#define KOKKOS_SHARED_LEVEL 1
        if (cc++ == 0) {
          ierr = PetscInfo4(plex, "shared memory size: %d bytes in level %d conc=%D team size=%D\n",scr_bytes,KOKKOS_SHARED_LEVEL,conc,team_size);CHKERRQ(ierr);
        }
        // get f and df
        ierr = PetscLogEventBegin(events[8],0,0,0,0);CHKERRQ(ierr);
        ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
        Kokkos::parallel_for("f, df", Kokkos::TeamPolicy<>(numCells, team_size, /* Kokkos::AUTO */ 16), KOKKOS_LAMBDA (const team_member team) {
            const PetscInt    elem = team.league_rank();
            const PetscScalar *coef;
            PetscScalar       coef_buff[LANDAU_MAX_SPECIES*LANDAU_MAX_NQ];
            // un pack IPData
            if (!d_maps) {
              coef = &d_IPf[elem*Nb*Nf];
            } else {
              coef = coef_buff;
              for (int f = 0; f < Nf; ++f) {
                LandauIdx *const Idxs = &d_maps->gIdx[elem][f][0];
                for (int b = 0; b < Nb; ++b) {
                  PetscInt idx = Idxs[b];
                  if (idx >= 0) {
                    coef_buff[f*Nb+b] = d_IPf[idx];
                  } else {
                    idx = -idx - 1;
                    coef_buff[f*Nb+b] = 0;
                    for (int q = 0; q < d_maps->num_face; q++) {
                      PetscInt    id = d_maps->c_maps[idx][q].gid;
                      PetscScalar scale = d_maps->c_maps[idx][q].scale;
                      coef_buff[f*Nb+b] += scale*d_IPf[id];
                    }
                  }
                }
              }
            }
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,Nq), [=] (int myQi) {
                const PetscInt          ipidx = myQi + elem * Nq;
                const PetscReal *const  invJj = &d_invJ_k(ipidx*dim*dim);
                const PetscReal         *Bq = &d_BB[myQi*Nb], *Dq = &d_DD[myQi*Nb*dim];
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,0,(int)Nf), [=] (int f) {
                    PetscInt     b, e, d;
                    PetscReal  refSpaceDer[LANDAU_DIM];
                    d_fdf_k(0,f,ipidx) = 0.0;
                    for (d = 0; d < LANDAU_DIM; ++d) refSpaceDer[d] = 0.0;
                    for (b = 0; b < Nb; ++b) {
                      const PetscInt    cidx = b;
                      d_fdf_k(0,f,ipidx) += Bq[cidx]*PetscRealPart(coef[f*Nb+cidx]);
                      for (d = 0; d < dim; ++d) refSpaceDer[d] += Dq[cidx*dim+d]*PetscRealPart(coef[f*Nb+cidx]);
                    }
                    for (d = 0; d < dim; ++d) {
                      for (e = 0, d_fdf_k(d+1,f,ipidx) = 0.0; e < dim; ++e) {
                        d_fdf_k(d+1,f,ipidx) += invJj[e*dim+d]*refSpaceDer[e];
                      }
                    }
                  }); // Nf
              }); // Nq
          }); // elems
        Kokkos::fence();
#if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_HIP)
        ierr = PetscLogGpuFlops(nip*(PetscLogDouble)(2*Nb*(1+dim)));CHKERRQ(ierr);
#else
        ierr = PetscLogFlops(nip*(PetscLogDouble)(2*Nb*(1+dim)));CHKERRQ(ierr);
#endif
        ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
        ierr = PetscLogEventEnd(events[8],0,0,0,0);CHKERRQ(ierr);
      }
      ierr = PetscLogEventBegin(events[4],0,0,0,0);CHKERRQ(ierr);
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_HIP)
      ierr = PetscLogGpuFlops(nip*(PetscLogDouble)(!(a_IPf || a_xarray) ? (nip*(11*Nf+ 4*dim*dim) + 6*Nf*dim*dim*dim + 10*Nf*dim*dim + 4*Nf*dim + Nb*Nf*Nb*Nq*dim*dim*5) : Nb*Nf*Nb*Nq*4));CHKERRQ(ierr);
      if (ctx->deviceType == LANDAU_CPU) PetscInfo(plex, "Warning: Landau selected CPU but no support for Kokkos using CPU\n");
#else
      ierr = PetscLogFlops(nip*(PetscLogDouble)(!(a_IPf || a_xarray) ? (nip*(11*Nf+ 4*dim*dim) + 6*Nf*dim*dim*dim + 10*Nf*dim*dim + 4*Nf*dim + Nb*Nf*Nb*Nq*dim*dim*5) : Nb*Nf*Nb*Nq*4));CHKERRQ(ierr);
#endif
      Kokkos::parallel_for("Landau elements", Kokkos::TeamPolicy<>(numCells, team_size, /*Kokkos::AUTO*/ 16).set_scratch_size(KOKKOS_SHARED_LEVEL, Kokkos::PerTeam(scr_bytes)), KOKKOS_LAMBDA (const team_member team) {
          const PetscInt  elem = team.league_rank();
          g2_scr_t        g2(team.team_scratch(KOKKOS_SHARED_LEVEL),dim,Nf,Nq); // we don't use these for mass matrix
          g3_scr_t        g3(team.team_scratch(KOKKOS_SHARED_LEVEL),dim,dim,Nf,Nq);
          if (d_IPf) {
            g2_scr_t        gg2(team.team_scratch(KOKKOS_SHARED_LEVEL),dim,Nf,Nq);
            g3_scr_t        gg3(team.team_scratch(KOKKOS_SHARED_LEVEL),dim,dim,Nf,Nq);
            // get g2[] & g3[]
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,Nq), [=] (int myQi) {
                using Kokkos::parallel_reduce;
                const PetscInt                    jpidx = myQi + elem * Nq;
                const PetscReal* const            invJj = &d_invJ_k(jpidx*dim*dim);
                const PetscReal                   vj[3] = {d_x[jpidx], d_y[jpidx], d_z ? d_z[jpidx] : 0}, wj = d_w[jpidx];
                landau_inner_red::TensorValueType gg_temp; // reduce on part of gg2 and g33 for IP jpidx
                Kokkos::parallel_reduce(Kokkos::ThreadVectorRange (team, (int)nip), [=] (const int& ipidx, landau_inner_red::TensorValueType & ggg) {
                    const PetscReal wi = d_w[ipidx], x = d_x[ipidx], y = d_y[ipidx];
                    PetscReal       temp1[3] = {0, 0, 0}, temp2 = 0;
                    PetscInt        fieldA,d2,d3;
#if LANDAU_DIM==2
                    PetscReal Ud[2][2], Uk[2][2];
                    LandauTensor2D(vj, x, y, Ud, Uk, (ipidx==jpidx) ? 0. : 1.);
#else
                    PetscReal U[3][3], z = d_z[ipidx];
                    LandauTensor3D(vj, x, y, z, U, (ipidx==jpidx) ? 0. : 1.);
#endif
                    for (fieldA = 0; fieldA < Nf; ++fieldA) {
                      temp1[0] += d_fdf_k(1,fieldA,ipidx)*d_beta[fieldA]*d_invMass[fieldA];
                      temp1[1] += d_fdf_k(2,fieldA,ipidx)*d_beta[fieldA]*d_invMass[fieldA];
#if LANDAU_DIM==3
                      temp1[2] += d_fdf_k(3,fieldA,ipidx)*d_beta[fieldA]*d_invMass[fieldA];
#endif
                      temp2    += d_fdf_k(0,fieldA,ipidx)*d_beta[fieldA];
                    }
                    temp1[0] *= wi;
                    temp1[1] *= wi;
#if LANDAU_DIM==3
                    temp1[2] *= wi;
#endif
                    temp2    *= wi;
#if LANDAU_DIM==2
                    for (d2 = 0; d2 < 2; d2++) {
                      for (d3 = 0; d3 < 2; ++d3) {
                        /* K = U * grad(f): g2=e: i,A */
                        ggg.gg2[d2] += Uk[d2][d3]*temp1[d3];
                        /* D = -U * (I \kron (fx)): g3=f: i,j,A */
                        ggg.gg3[d2][d3] += Ud[d2][d3]*temp2;
                      }
                    }
#else
                    for (d2 = 0; d2 < 3; ++d2) {
                      for (d3 = 0; d3 < 3; ++d3) {
                        /* K = U * grad(f): g2 = e: i,A */
                        ggg.gg2[d2] += U[d2][d3]*temp1[d3];
                        /* D = -U * (I \kron (fx)): g3 = f: i,j,A */
                        ggg.gg3[d2][d3] += U[d2][d3]*temp2;
                      }
                    }
#endif
                  }, Kokkos::Sum<landau_inner_red::TensorValueType>(gg_temp));
                // add alpha and put in gg2/3
                Kokkos::parallel_for(Kokkos::ThreadVectorRange (team, (int)Nf), [&] (const int& fieldA) {
                    PetscInt d2,d3;
                    for (d2 = 0; d2 < dim; d2++) {
                      gg2(d2,fieldA,myQi) = gg_temp.gg2[d2]*d_alpha[fieldA];
                      for (d3 = 0; d3 < dim; d3++) {
                        gg3(d2,d3,fieldA,myQi) = -gg_temp.gg3[d2][d3]*d_alpha[fieldA]*d_invMass[fieldA];
                      }
                    }
                  });
                /* add electric field term once per IP */
                Kokkos::parallel_for(Kokkos::ThreadVectorRange (team, (int)Nf), [&] (const int& fieldA) {
                    gg2(dim-1,fieldA,myQi) += d_Eq_m[fieldA];
                  });
                Kokkos::parallel_for(Kokkos::ThreadVectorRange (team, (int)Nf), [=] (const int& fieldA) {
                    int d,d2,d3,dp;
                    /* Jacobian transform - g2, g3 - per thread (2D) */
                    for (d = 0; d < dim; ++d) {
                      g2(d,fieldA,myQi) = 0;
                      for (d2 = 0; d2 < dim; ++d2) {
                        g2(d,fieldA,myQi) += invJj[d*dim+d2]*gg2(d2,fieldA,myQi);
                        g3(d,d2,fieldA,myQi) = 0;
                        for (d3 = 0; d3 < dim; ++d3) {
                          for (dp = 0; dp < dim; ++dp) {
                            g3(d,d2,fieldA,myQi) += invJj[d*dim + d3]*gg3(d3,dp,fieldA,myQi)*invJj[d2*dim + dp];
                          }
                        }
                        g3(d,d2,fieldA,myQi) *= wj;
                      }
                      g2(d,fieldA,myQi) *= wj;
                    }
                  });
              }); // Nq
            team.team_barrier();
          } // Jacobian
          /* assemble */
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,Nb), [=] (int blk_i) {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,0,(int)Nf), [=] (int fieldA) {
                  int blk_j,qj,d,d2;
                  const PetscInt i = fieldA*Nb + blk_i; /* Element matrix row */
                  for (blk_j = 0; blk_j < Nb; ++blk_j) {
                    const PetscInt j    = fieldA*Nb + blk_j; /* Element matrix column */
                    const PetscInt fOff = i*totDim + j;
                    PetscScalar t = global_elem_mat_sz ? d_elem_mats(elem,fOff) : 0;
                    for (qj = 0 ; qj < Nq ; qj++) { // look at others integration points
                      const PetscReal *BJq = &d_BB[qj*Nb], *DIq = &d_DD[qj*Nb*dim];
                      if (d_IPf) {
                        for (d = 0; d < dim; ++d) {
                          t += DIq[blk_i*dim+d]*g2(d,fieldA,qj)*BJq[blk_j];
                          for (d2 = 0; d2 < dim; ++d2) {
                            t += DIq[blk_i*dim + d]*g3(d,d2,fieldA,qj)*DIq[blk_j*dim + d2];
                          }
                        }
                      } else {
                        const PetscInt jpidx = qj + elem * Nq;
                        t += BJq[blk_i] * d_mass_w_k(jpidx) * shift * BJq[blk_j];
                      }
                    }
                    if (global_elem_mat_sz) d_elem_mats(elem,fOff) = t; // can set this because local element matrix[fOff]
                    else {
                      PetscErrorCode         ierr = 0;
                      PetscScalar            vals[LANDAU_MAX_Q_FACE*LANDAU_MAX_Q_FACE],row_scale[LANDAU_MAX_Q_FACE],col_scale[LANDAU_MAX_Q_FACE];
                      PetscInt               q,idx,nr,nc,rows0[LANDAU_MAX_Q_FACE],cols0[LANDAU_MAX_Q_FACE],rows[LANDAU_MAX_Q_FACE],cols[LANDAU_MAX_Q_FACE];
                      const LandauIdx *const Idxs = &d_maps->gIdx[elem][fieldA][0];
                      idx = Idxs[blk_i];
                      if (idx >= 0) {
                        nr = 1;
                        rows0[0] = idx;
                        row_scale[0] = 1.;
                      } else {
                        idx = -idx - 1;
                        nr = d_maps->num_face;
                        for (q = 0; q < d_maps->num_face; q++) {
                          rows0[q]     = d_maps->c_maps[idx][q].gid;
                          row_scale[q] = d_maps->c_maps[idx][q].scale;
                        }
                      }
                      idx = Idxs[blk_j];
                      if (idx >= 0) {
                        nc = 1;
                        cols0[0] = idx;
                        col_scale[0] = 1.;
                      } else {
                        idx = -idx - 1;
                        nc = d_maps->num_face;
                        for (q = 0; q < d_maps->num_face; q++) {
                          cols0[q]     = d_maps->c_maps[idx][q].gid;
                          col_scale[q] = d_maps->c_maps[idx][q].scale;
                        }
                      }
                      for (q = 0; q < nr; q++) rows[q] = rows0[q];
                      for (q = 0; q < nc; q++) cols[q] = cols0[q];
                      for (q = 0; q < nr; q++) {
                        for (d = 0; d < nc; d++) {
                          vals[q*nc + d] = row_scale[q]*col_scale[d]*t;
                        }
                      }
                      MatSetValuesDevice(d_mat,nr,rows,nc,cols,vals,ADD_VALUES,&ierr);
                      if (ierr) return;
                    }
                  }
                });
            });
        });
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      Kokkos::fence();
      ierr = PetscLogEventEnd(events[4],0,0,0,0);CHKERRQ(ierr);
      if (global_elem_mat_sz) {
        PetscSection      section, globalSection;
        Kokkos::View<PetscScalar**, Kokkos::LayoutRight>::HostMirror h_elem_mats = Kokkos::create_mirror_view(d_elem_mats);
        ierr = PetscLogEventBegin(events[5],0,0,0,0);CHKERRQ(ierr);
        ierr = DMGetLocalSection(plex, &section);CHKERRQ(ierr);
        ierr = DMGetGlobalSection(plex, &globalSection);CHKERRQ(ierr);
        Kokkos::deep_copy (h_elem_mats, d_elem_mats);
        ierr = PetscLogEventEnd(events[5],0,0,0,0);CHKERRQ(ierr);
        ierr = PetscLogEventBegin(events[6],0,0,0,0);CHKERRQ(ierr);
        PetscInt ej;
        for (ej = cStart ; ej < cEnd; ++ej) {
          const PetscScalar *elMat = &h_elem_mats(ej-cStart,0);
          ierr = DMPlexMatSetClosure(plex, section, globalSection, JacP, ej, elMat, ADD_VALUES);CHKERRQ(ierr);
          if (ej==-1) {
            int d,f;
            PetscPrintf(PETSC_COMM_SELF,"Kokkos Element matrix %d/%d\n",1,(int)numCells);
            for (d = 0; d < totDim; ++d) {
              for (f = 0; f < totDim; ++f) PetscPrintf(PETSC_COMM_SELF," %12.5e",  PetscRealPart(elMat[d*totDim + f]));
              PetscPrintf(PETSC_COMM_SELF,"\n");
            }
            exit(14);
          }
        }
        ierr = PetscLogEventEnd(events[6],0,0,0,0);CHKERRQ(ierr);
        // transition to use of maps for VecGetClosure
        if (ctx->gpu_assembly) {
          auto IPf = static_cast<Kokkos::View<PetscScalar*, Kokkos::LayoutLeft>*>(SData_d->IPf);
          delete IPf;
          SData_d->IPf = NULL;
          if (!(a_IPf || a_xarray)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "transition without Jacobian");
        }
      }
    }
    PetscFunctionReturn(0);
  }
} // extern "C"
