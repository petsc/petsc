/*
   Implements the Kokkos kernel
*/
#define PETSC_SKIP_CXX_COMPLEX_FIX
#include <petscconf.h>
#include <petsc/private/dmpleximpl.h>   /*I   "petscdmplex.h"   I*/
#include <petsclandau.h>
#include <petscts.h>
#include <petscveckokkos.hpp>

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
      for (int j = 0; j < LANDAU_DIM; j++){
        gg2[j] = 0;
        for (int k = 0; k < LANDAU_DIM; k++){
          gg3[j][k] = 0;
        }
      }
    }
    KOKKOS_INLINE_FUNCTION   // Copy Constructor
    array_type(const array_type & rhs) {
      for (int j = 0; j < LANDAU_DIM; j++){
        gg2[j] = rhs.gg2[j];
        for (int k = 0; k < LANDAU_DIM; k++){
          gg3[j][k] = rhs.gg3[j][k];
        }
      }
    }
    KOKKOS_INLINE_FUNCTION   // add operator
    array_type& operator += (const array_type& src) {
      for (int j = 0; j < LANDAU_DIM; j++){
        gg2[j] += src.gg2[j];
        for (int k = 0; k < LANDAU_DIM; k++){
          gg3[j][k] += src.gg3[j][k];
        }
      }
      return *this;
    }
    KOKKOS_INLINE_FUNCTION   // volatile add operator
    void operator += (const volatile array_type& src) volatile {
      for (int j = 0; j < LANDAU_DIM; j++){
        gg2[j] += src.gg2[j];
        for (int k = 0; k < LANDAU_DIM; k++){
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
    Kokkos::View<P4estVertexMaps>                    *d_maps_k = new Kokkos::View<P4estVertexMaps>(Kokkos::create_mirror(DeviceMemorySpace(),h_maps_k));
    Kokkos::deep_copy (*d_maps_k, h_maps_k);
    maps->data = d_maps_k->data();
    maps->vp3 = (void*)d_maps_k;
    // debug
    // PetscErrorCode    ierr;
    // PetscInt          ej,q,s;
    // ierr = PetscPrintf(PETSC_COMM_SELF,"=== c_maps size --> %D. Kokkos::View<P4estVertexMaps*>=%p maps->data=%p P4estVertexMaps *maps=%p\n",maps->num_reduced, d_maps_k, maps->data, maps);CHKERRQ(ierr);
    // for (ej = 0; ej < maps->num_reduced; ++ej) {
    //   ierr = PetscPrintf(PETSC_COMM_SELF,"\t\tconstrant %3D) ",ej);
    //   for (q = 0; q < maps->num_face; ++q) {
    //     ierr = PetscPrintf(PETSC_COMM_SELF,"\t %3D - %13.5e ; ", maps->data->c_maps[ej][q].gid, maps->data->c_maps[ej][q].scale);CHKERRQ(ierr);
    //   }
    //   ierr = PetscPrintf(PETSC_COMM_SELF,"\n");
    // }
    // ierr = PetscPrintf(PETSC_COMM_SELF,"================= gIDx \n");
    // for (ej = 0; ej < maps->num_elements; ++ej) {
    //   for (q = 0; q < Nq; ++q) {
    //     for (s = 0; s < Nf; ++s) {
    //       ierr = PetscPrintf(PETSC_COMM_SELF," %3D ", maps->data->gIdx[ej][s][q]);
    //     }
    //     ierr = PetscPrintf(PETSC_COMM_SELF," ; ");
    //   }
    //   ierr = PetscPrintf(PETSC_COMM_SELF,"\n");
    // }
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

PetscErrorCode LandauKokkosJacobian(DM plex, const PetscInt Nq, PetscReal nu_alpha[], PetscReal nu_beta[], PetscReal invMass[], PetscReal Eq_m[],
                                    const LandauIPData *const IPData, PetscReal a_invJ[], const PetscInt num_sub_blocks, const PetscLogEvent events[], Mat JacP)
{
  PetscErrorCode    ierr;
  PetscInt          *Nbf,Nb,cStart,cEnd,Nf,dim,numCells,totDim,ipdatasz,global_elem_mat_sz;
  PetscTabulation   *Tf;
  PetscDS           prob;
  PetscSection      section, globalSection;
  PetscLogDouble    flops;
  PetscReal         *BB,*DD;
  LandauCtx         *ctx;
  P4estVertexMaps   *d_maps=NULL;
  PetscSplitCSRDataStructure *d_mat=NULL;

  PetscFunctionBegin;
  ierr = DMGetApplicationContext(plex, &ctx);CHKERRQ(ierr);
  if (!ctx) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  ierr = DMGetDimension(plex, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(prob, &Nbf);CHKERRQ(ierr); Nb = Nbf[0];
  if (Nq != Nb) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Nq != Nb. %D  %D",Nq,Nb);
  if (LANDAU_DIM != dim) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "dim %D != LANDAU_DIM %d",dim,LANDAU_DIM);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(prob, &Tf);CHKERRQ(ierr);

  if (ctx->gpu_assembly) {
    PetscContainer container;
    ierr = PetscObjectQuery((PetscObject) JacP, "assembly_maps", (PetscObject *) &container);CHKERRQ(ierr);
    if (container) { // not here first call
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
      P4estVertexMaps   *h_maps=NULL;
      ierr = PetscContainerGetPointer(container, (void **) &h_maps);CHKERRQ(ierr);
      ierr = PetscInfo2(JacP, "Have container maps=%p maps->data=%p\n", h_maps, h_maps ? h_maps->data : NULL);CHKERRQ(ierr);
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
  BB   = Tf[0]->T[0]; DD = Tf[0]->T[1];
  ierr = DMGetLocalSection(plex, &section);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(plex, &globalSection);CHKERRQ(ierr);
  flops = (PetscLogDouble)numCells*Nq*(5*dim*dim*Nf*Nf + 165);
  ipdatasz = LandauGetIPDataSize(IPData);
  ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
  {
    using scr_mem_t = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using g2_scr_t = Kokkos::View<PetscReal***, Kokkos::LayoutRight, scr_mem_t>;
    using g3_scr_t = Kokkos::View<PetscReal****, Kokkos::LayoutRight, scr_mem_t>;

    const int scr_bytes = 2*(g2_scr_t::shmem_size(dim,Nf,Nq) + g3_scr_t::shmem_size(dim,dim,Nf,Nq));
    int   conc, team_size;
    ierr = PetscLogEventBegin(events[3],0,0,0,0);CHKERRQ(ierr);
    const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_alpha (nu_alpha, Nf);
    Kokkos::View<PetscReal*, Kokkos::LayoutLeft> d_alpha ("nu_alpha", Nf);
    const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_beta (nu_beta, Nf);
    Kokkos::View<PetscReal*, Kokkos::LayoutLeft> d_beta ("nu_beta", Nf);
    const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_invMass (invMass,Nf);
    Kokkos::View<PetscReal*, Kokkos::LayoutLeft> d_invMass ("invMass", Nf);
    const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_Eq_m (Eq_m,Nf);
    Kokkos::View<PetscReal*, Kokkos::LayoutLeft> d_Eq_m ("Eq_m", Nf);
    const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_BB (BB,Nq*Nb);
    Kokkos::View<PetscReal*, Kokkos::LayoutLeft> d_BB ("BB", Nq*Nb);
    const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_DD (DD,Nq*Nb*dim);
    Kokkos::View<PetscReal*, Kokkos::LayoutLeft> d_DD ("DD", Nq*Nb*dim);
    const Kokkos::View<LandauIPReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_ipdata_raw (IPData->w,ipdatasz);
    Kokkos::View<LandauIPReal*, Kokkos::LayoutLeft> d_ipdata_raw ("ipdata", ipdatasz);
    const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_invJ (a_invJ,IPData->nip_*dim*dim);
    Kokkos::View<PetscReal*, Kokkos::LayoutLeft> d_invJ ("invJ", IPData->nip_*dim*dim);
    Kokkos::View<PetscScalar**, Kokkos::LayoutRight> d_elem_mats("element matrices", global_elem_mat_sz, totDim*totDim);
    Kokkos::View<PetscScalar**, Kokkos::LayoutRight> d_f("element matrices", Nf, IPData->nip_);
    Kokkos::View<PetscScalar***, Kokkos::LayoutRight> d_df("element matrices", dim, Nf, IPData->nip_);

    Kokkos::deep_copy (d_ipdata_raw, h_ipdata_raw);
    Kokkos::deep_copy (d_alpha, h_alpha);
    Kokkos::deep_copy (d_beta, h_beta);
    Kokkos::deep_copy (d_invMass, h_invMass);
    Kokkos::deep_copy (d_Eq_m, h_Eq_m);
    Kokkos::deep_copy (d_BB, h_BB);
    Kokkos::deep_copy (d_DD, h_DD);
    Kokkos::deep_copy (d_invJ, h_invJ);

    ierr = PetscLogEventEnd(events[3],0,0,0,0);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(events[8],0,0,0,0);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_VIENNACL)
    ierr = PetscLogGpuFlops(flops*IPData->nip_);CHKERRQ(ierr);
    if (ctx->deviceType == LANDAU_CPU) PetscInfo(plex, "Warning: Landau selected CPU but no support for Kokkos using GPU\n");
#else
    ierr = PetscLogFlops(flops*IPData->nip_);CHKERRQ(ierr);
#endif
#define KOKKOS_SHARED_LEVEL 1
    //PetscInfo2(plex, "shared memory size: %d bytes in level %d\n",scr_bytes,KOKKOS_SHARED_LEVEL);
    conc = Kokkos::DefaultExecutionSpace().concurrency(), team_size = conc > Nq ? Nq : 1;
    // get f and df
    Kokkos::parallel_for("Landau_elements", Kokkos::TeamPolicy<>(numCells, team_size, num_sub_blocks).set_scratch_size(KOKKOS_SHARED_LEVEL, Kokkos::PerTeam(scr_bytes)), KOKKOS_LAMBDA (const team_member team) {
        const PetscInt  myelem = team.league_rank();
        // un pack IPData
        LandauIPReal *IPData_coefs = &d_ipdata_raw[IPData->nip_*(dim+1)];
        LandauIPReal *coef = &IPData_coefs[myelem*Nb*Nf];
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,Nq), [=] (int myQi) {
            const PetscInt          ipidx = myQi + myelem * Nq;
            const PetscReal *const  invJj = &d_invJ(ipidx*dim*dim);
            const PetscReal         *Bq = &d_BB[myQi*Nb], *Dq = &d_DD[myQi*Nb*dim];
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,0,(int)Nf), [=] (int f) {
                PetscInt     b, e, d;
                PetscScalar  refSpaceDer[LANDAU_DIM];
                d_f(f,ipidx) = 0.0;
                for (d = 0; d < LANDAU_DIM; ++d) refSpaceDer[d] = 0.0;
                for (b = 0; b < Nb; ++b) {
                  const PetscInt    cidx = b;
                  d_f(f,ipidx) += Bq[cidx]*coef[f*Nb+cidx];
                  for (d = 0; d < dim; ++d) refSpaceDer[d] += Dq[cidx*dim+d]*coef[f*Nb+cidx];
                }
                for (d = 0; d < dim; ++d) {
                  for (e = 0, d_df(d,f,ipidx) = 0.0; e < dim; ++e) {
                    d_df(d,f,ipidx) += invJj[e*dim+d]*refSpaceDer[e];
                  }
                }
              }); // Nf
          }); // Nq
      }); // elems
    ierr = PetscLogEventEnd(events[8],0,0,0,0);CHKERRQ(ierr);
    Kokkos::fence();
    ierr = PetscLogEventBegin(events[4],0,0,0,0);CHKERRQ(ierr);
    Kokkos::parallel_for("Landau_elements", Kokkos::TeamPolicy<>(numCells, team_size, num_sub_blocks).set_scratch_size(KOKKOS_SHARED_LEVEL, Kokkos::PerTeam(scr_bytes)), KOKKOS_LAMBDA (const team_member team) {
        const PetscInt  myelem = team.league_rank();
        g2_scr_t        g2(team.team_scratch(KOKKOS_SHARED_LEVEL),dim,Nf,Nq);
        g3_scr_t        g3(team.team_scratch(KOKKOS_SHARED_LEVEL),dim,dim,Nf,Nq);
        g2_scr_t        gg2(team.team_scratch(KOKKOS_SHARED_LEVEL),dim,Nf,Nq);
        g3_scr_t        gg3(team.team_scratch(KOKKOS_SHARED_LEVEL),dim,dim,Nf,Nq);
        LandauIPData    d_IPData;
        // un pack IPData
        d_IPData.w   = &d_ipdata_raw[0];
        d_IPData.x   = &d_ipdata_raw[1*IPData->nip_];
        d_IPData.y   = &d_ipdata_raw[2*IPData->nip_];
        if (dim==2) d_IPData.z = NULL;
        else        d_IPData.z = &d_ipdata_raw[3*IPData->nip_];
        // get g2[] & g3[]
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,Nq), [=] (int myQi) {
            using Kokkos::parallel_reduce;
            const PetscInt                    jpidx = myQi + myelem * Nq;
            const PetscReal* const            invJj = &d_invJ(jpidx*dim*dim);
            const PetscReal                   vj[3] = {d_IPData.x[jpidx], d_IPData.y[jpidx], d_IPData.z ? d_IPData.z[jpidx] : 0}, wj = d_IPData.w[jpidx];
            landau_inner_red::TensorValueType gg_temp; // reduce on part of gg2 and g33 for IP jpidx
            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange (team, (int)IPData->nip_), [=] (const int& ipidx, landau_inner_red::TensorValueType & ggg) {
                const PetscReal wi = d_IPData.w[ipidx], x = d_IPData.x[ipidx], y = d_IPData.y[ipidx];
                PetscReal       temp1[3] = {0, 0, 0}, temp2 = 0;
                PetscInt        fieldA,d2,d3;
#if LANDAU_DIM==2
                PetscReal Ud[2][2], Uk[2][2];
                LandauTensor2D(vj, x, y, Ud, Uk, (ipidx==jpidx) ? 0. : 1.);
#else
                PetscReal U[3][3], z = d_IPData.z[ipidx];
                LandauTensor3D(vj, x, y, z, U, (ipidx==jpidx) ? 0. : 1.);
#endif
                for (fieldA = 0; fieldA < Nf; ++fieldA) {
                  temp1[0] += d_df(0,fieldA,ipidx)*d_beta[fieldA]*d_invMass[fieldA];
                  temp1[1] += d_df(1,fieldA,ipidx)*d_beta[fieldA]*d_invMass[fieldA];
#if LANDAU_DIM==3
                  temp1[2] += d_df(2,fieldA,ipidx)*d_beta[fieldA]*d_invMass[fieldA];
#endif
                  temp2    += d_f(fieldA,ipidx)*d_beta[fieldA];
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
            //if (myelem==0) printf("\t:%d.%d) temp gg3=%e %e %e %e\n",myelem,myQi,gg_temp.gg3[0][0],gg_temp.gg3[1][0],gg_temp.gg3[0][1],gg_temp.gg3[1][1]);
            // add alpha and put in gg2/3
            Kokkos::parallel_for(Kokkos::ThreadVectorRange (team, (int)Nf), [&] (const int& fieldA) {
                PetscInt d2,d3;
                for (d2 = 0; d2 < dim; d2++) {
                  gg2(d2,fieldA,myQi) = gg_temp.gg2[d2]*d_alpha[fieldA];
                  //if (myelem==0 && fieldA==1) printf("\t\t:%d.%d) gg2[%d]=%e (+= %e)\n",myelem,myQi,d2,gg2(d2,fieldA,myQi),gg_temp.gg2[d2]*d_alpha[fieldA]);
                  //gg2[d2][myQi][fieldA] += gg_temp.gg2[d2]*d_alpha[fieldA];
                  for (d3 = 0; d3 < dim; d3++) {
                    //gg3[d2][d3][myQi][fieldA] -= gg_temp.gg3[d2][d3]*d_alpha[fieldA]*s_invMass[fieldA];
                    gg3(d2,d3,fieldA,myQi) = -gg_temp.gg3[d2][d3]*d_alpha[fieldA]*d_invMass[fieldA];
                    //if (myelem==0 && fieldA==1) printf("\t\t\t:%d.%d) gg3[%d][%d]=%e\n",myelem,myQi,d2,d3,gg3(d2,d3,fieldA,myQi));
                  }
                }
              });

            /* add electric field term once per IP */
            Kokkos::parallel_for(Kokkos::ThreadVectorRange (team, (int)Nf), [&] (const int& fieldA) {
                //gg.gg2[fieldA][dim-1] += d_Eq_m[fieldA];
                gg2(dim-1,fieldA,myQi) += d_Eq_m[fieldA];
              });
            // Kokkos::single(Kokkos::PerThread(team), [&]() {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange (team, (int)Nf), [=] (const int& fieldA) {
                int d,d2,d3,dp;
                //printf("%d %d %d gg2[][1]=%18.10e\n",myelem,myQi,fieldA,gg.gg2[fieldA][dim-1]);
                /* Jacobian transform - g2, g3 - per thread (2D) */
                for (d = 0; d < dim; ++d) {
                  g2(d,fieldA,myQi) = 0;
                  for (d2 = 0; d2 < dim; ++d2) {
                    g2(d,fieldA,myQi) += invJj[d*dim+d2]*gg2(d2,fieldA,myQi);
                    //if (myelem==0 && myQi==0) printf("\t:g2[%d][%d][%d]=%e. %e %e\n",(int)myQi,(int)fieldA,(int)d,g2(fieldA,myQi,d),invJj[d*dim+d2],gg.gg2[fieldA][d2]);
                    g3(d,d2,fieldA,myQi) = 0;
                    for (d3 = 0; d3 < dim; ++d3) {
                      for (dp = 0; dp < dim; ++dp) {
                        g3(d,d2,fieldA,myQi) += invJj[d*dim + d3]*gg3(d3,dp,fieldA,myQi)*invJj[d2*dim + dp];
                        //printf("\t%d %d %d %d %d %d %d g3=%g wj=%g g3 = %g * %g * %g\n",myelem,myQi,fieldA,d,d2,d3,dp,g3(fieldA,myQi,d,d2),wj,invJj[d*dim + d3],gg.gg3[fieldA][d3][dp],invJj[d2*dim + dp]);
                      }
                    }
                    g3(d,d2,fieldA,myQi) *= wj;
                  }
                  g2(d,fieldA,myQi) *= wj;
                }
              });
          }); // Nq
        team.team_barrier();
        /* assemble - on the diagonal (I,I) */
        //Kokkos::single(Kokkos::PerTeam(team), [&]() {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,Nb), [=] (int blk_i) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,0,(int)Nf), [=] (int fieldA) {
                int blk_j,qj,d,d2;
                const PetscInt i = fieldA*Nb + blk_i; /* Element matrix row */
                for (blk_j = 0; blk_j < Nb; ++blk_j) {
                  const PetscInt j    = fieldA*Nb + blk_j; /* Element matrix column */
                  const PetscInt fOff = i*totDim + j;
                  PetscScalar t = global_elem_mat_sz ? d_elem_mats(myelem,fOff) : 0;
                  for (qj = 0 ; qj < Nq ; qj++) { // look at others integration points
                    const PetscReal *BJq = &d_BB[qj*Nb], *DIq = &d_DD[qj*Nb*dim];
                    for (d = 0; d < dim; ++d) {
                      t += DIq[blk_i*dim+d]*g2(d,fieldA,qj)*BJq[blk_j];
                      //printf("\tmat[%d %d %d %d %d]=%g D[%d]=%g g2[%d][%d][%d]=%g B=%g\n",myelem,fOff,fieldA,qj,d,d_elem_mats(myelem,fOff),blk_i*dim+d,DIq[blk_i*dim+d],fieldA,qj,d,g2(fieldA,qj,d),BJq[blk_j]);
                      for (d2 = 0; d2 < dim; ++d2) {
                        t += DIq[blk_i*dim + d]*g3(d,d2,fieldA,qj)*DIq[blk_j*dim + d2];
                      }
                    }
                  }
                  if (global_elem_mat_sz) d_elem_mats(myelem,fOff) = t; // can set this because local element matrix[fOff]
                  else {
                    PetscErrorCode         ierr = 0;
                    PetscScalar            vals[LANDAU_MAX_Q*LANDAU_MAX_Q],row_scale[LANDAU_MAX_Q],col_scale[LANDAU_MAX_Q];
                    PetscInt               q,idx,nr,nc,rows0[LANDAU_MAX_Q],cols0[LANDAU_MAX_Q],rows[LANDAU_MAX_Q],cols[LANDAU_MAX_Q];
                    const LandauIdx *const Idxs = &d_maps->gIdx[myelem][fieldA][0];
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
    Kokkos::fence();
    ierr = PetscLogEventEnd(events[4],0,0,0,0);CHKERRQ(ierr);
    if (global_elem_mat_sz) {
      Kokkos::View<PetscScalar**, Kokkos::LayoutRight>::HostMirror h_elem_mats = Kokkos::create_mirror_view(d_elem_mats);
      ierr = PetscLogEventBegin(events[5],0,0,0,0);CHKERRQ(ierr);
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
          for (d = 0; d < totDim; ++d){
            for (f = 0; f < totDim; ++f) PetscPrintf(PETSC_COMM_SELF," %12.5e",  PetscRealPart(elMat[d*totDim + f]));
            PetscPrintf(PETSC_COMM_SELF,"\n");
          }
        }
      }
      ierr = PetscLogEventEnd(events[6],0,0,0,0);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
} // extern "C"
