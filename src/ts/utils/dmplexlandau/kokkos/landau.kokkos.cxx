/*
   Implements the Kokkos kernel
*/

#define PETSC_SKIP_CXX_COMPLEX_FIX
#include <petscconf.h>
#include <petsc/private/dmpleximpl.h>   /*I   "petscdmplex.h"   I*/
#include <petsclandau.h>
#include <petscts.h>
#include <Kokkos_Core.hpp>
#include <cstdio>
typedef Kokkos::TeamPolicy<>::member_type team_member;
#define PETSC_DEVICE_FUNC_DECL KOKKOS_INLINE_FUNCTION
#include "../land_tensors.h"

namespace landau_inner_red {  // namespace helps with name resolution in reduction identity
  template< class ScalarType, int Nf >
  struct array_type {
    ScalarType gg2[Nf][LANDAU_DIM];
    ScalarType gg3[Nf][LANDAU_DIM][LANDAU_DIM];

    KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
    array_type() {
      for (int i = 0; i < Nf; i++){
        for (int j = 0; j < LANDAU_DIM; j++){
          gg2[i][j] = 0;
          for (int k = 0; k < LANDAU_DIM; k++){
            gg3[i][j][k] = 0;
          }
        }
      }
    }
    KOKKOS_INLINE_FUNCTION   // Copy Constructor
    array_type(const array_type & rhs) {
      for (int i = 0; i < Nf; i++){
        for (int j = 0; j < LANDAU_DIM; j++){
          gg2[i][j] = rhs.gg2[i][j];
          for (int k = 0; k < LANDAU_DIM; k++){
            gg3[i][j][k] = rhs.gg3[i][j][k];
          }
        }
      }
    }
    KOKKOS_INLINE_FUNCTION   // add operator
    array_type& operator += (const array_type& src) {
      for (int i = 0; i < Nf; i++){
        for (int j = 0; j < LANDAU_DIM; j++){
          gg2[i][j] += src.gg2[i][j];
          for (int k = 0; k < LANDAU_DIM; k++){
            gg3[i][j][k] += src.gg3[i][j][k];
          }
        }
      }
      return *this;
    }
    KOKKOS_INLINE_FUNCTION   // volatile add operator
    void operator += (const volatile array_type& src) volatile {
      for (int i = 0; i < Nf; i++){
        for (int j = 0; j < LANDAU_DIM; j++){
          gg2[i][j] += src.gg2[i][j];
          for (int k = 0; k < LANDAU_DIM; k++){
            gg3[i][j][k] += src.gg3[i][j][k];
          }
        }
      }
    }
  };
  typedef array_type<PetscReal,LANDAU_MAX_SPECIES> ValueType;  // used to simplify code below
}

namespace Kokkos { //reduction identity must be defined in Kokkos namespace
  template<>
  struct reduction_identity< landau_inner_red::ValueType > {
    KOKKOS_FORCEINLINE_FUNCTION static landau_inner_red::ValueType sum() {
      return landau_inner_red::ValueType();
    }
  };
}

extern "C"  {
PetscErrorCode LandauKokkosJacobian(DM plex, const PetscInt Nq, PetscReal nu_alpha[], PetscReal nu_beta[],
                                   PetscReal invMass[], PetscReal Eq_m[], PetscReal * const IPDataGlobal,
                                   PetscReal wiGlobal[], PetscReal invJ[], const PetscInt num_sub_blocks, const PetscLogEvent events[], PetscBool quarter3DDomain,
                                   Mat JacP)
{
  PetscErrorCode    ierr;
  PetscInt          *Nbf,Nb,cStart,cEnd,Nf,dim,numCells,totDim,nip;
  PetscTabulation   *Tf;
  PetscDS           prob;
  PetscSection      section, globalSection;
  PetscLogDouble    flops;
  PetscReal         *BB,*DD;
  LandauCtx           *ctx;

  PetscFunctionBegin;
  ierr = DMGetApplicationContext(plex, &ctx);CHKERRQ(ierr);
  if (!ctx) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  ierr = DMGetDimension(plex, &dim);CHKERRQ(ierr);
  if (dim!=LANDAU_DIM) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "LANDAU_DIM != dim");
  ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  nip = numCells*Nq;
  ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(prob, &Nbf);CHKERRQ(ierr); Nb = Nbf[0];
  if (Nq != Nb) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Nq != Nb. %D  %D",Nq,Nb);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(prob, &Tf);CHKERRQ(ierr);
  BB   = Tf[0]->T[0]; DD = Tf[0]->T[1];
  ierr = DMGetLocalSection(plex, &section);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(plex, &globalSection);CHKERRQ(ierr);
  flops = (PetscLogDouble)numCells*Nq*(5*dim*dim*Nf*Nf + 165);
  if (!Kokkos::is_initialized()){
    int argc = 1;
    char string[1][32], *argv[1] = {string[0]};
    ierr = PetscStrcpy(string[0],"landau");CHKERRQ(ierr);
    Kokkos::initialize(argc, argv);
  }
#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
#else
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "no KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA");
#endif
  {
    using scr_mem_t = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using g2_scr_t = Kokkos::View<PetscReal***, Kokkos::LayoutRight, scr_mem_t>;
    using g3_scr_t = Kokkos::View<PetscReal****, Kokkos::LayoutRight, scr_mem_t>;
    const int scr_bytes = g2_scr_t::shmem_size(Nf,Nq,dim) +  g3_scr_t::shmem_size(Nf,Nq,dim,dim);
    ierr = PetscLogEventBegin(events[3],0,0,0,0);CHKERRQ(ierr);
    Kokkos::View<PetscScalar**, Kokkos::LayoutRight> d_elem_mats("element matrices", numCells, totDim*totDim);
    Kokkos::View<PetscScalar**, Kokkos::LayoutRight>::HostMirror h_elem_mats = Kokkos::create_mirror_view(d_elem_mats);
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
    const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_wiGlobal (wiGlobal,nip);
    Kokkos::View<PetscReal*, Kokkos::LayoutLeft> d_wiGlobal ("wiGlobal", nip);
    const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_ipdata (IPDataGlobal,nip*(dim + Nf*(dim+1)));
    Kokkos::View<PetscReal*, Kokkos::LayoutLeft> d_ipdata ("ipdata", nip*(dim + Nf*(dim+1)));
    const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_invJ (invJ,nip*dim*dim);
    Kokkos::View<PetscReal*, Kokkos::LayoutLeft> d_invJ ("invJ", nip*dim*dim);

    Kokkos::deep_copy (d_alpha, h_alpha);
    Kokkos::deep_copy (d_beta, h_beta);
    Kokkos::deep_copy (d_invMass, h_invMass);
    Kokkos::deep_copy (d_Eq_m, h_Eq_m);
    Kokkos::deep_copy (d_BB, h_BB);
    Kokkos::deep_copy (d_DD, h_DD);
    Kokkos::deep_copy (d_wiGlobal, h_wiGlobal);
    Kokkos::deep_copy (d_ipdata, h_ipdata);
    Kokkos::deep_copy (d_invJ, h_invJ);
    ierr = PetscLogEventEnd(events[3],0,0,0,0);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(events[4],0,0,0,0);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_VIENNACL)
    ierr = PetscLogGpuFlops(flops*nip);CHKERRQ(ierr);
    if (ctx->deviceType == LANDAU_CPU) PetscInfo(plex, "Warning: Landau selected CPU but no support for Kokkos using GPU\n");
#else
    ierr = PetscLogFlops(flops*nip);CHKERRQ(ierr);
#endif
#define KOKKOS_SHARED_LEVEL 1
    //PetscInfo2(plex, "shared memory size: %D kB in level %d\n",Nf*Nq*dim*(dim+1)*sizeof(PetscReal)/1024,KOKKOS_SHARED_LEVEL);
    int conc = Kokkos::DefaultExecutionSpace().concurrency(), team_size = conc > Nq ? Nq : 1;
    Kokkos::parallel_for("Landau_elements", Kokkos::TeamPolicy<>(numCells, team_size, num_sub_blocks).set_scratch_size(KOKKOS_SHARED_LEVEL, Kokkos::PerTeam(scr_bytes)), KOKKOS_LAMBDA (const team_member team) {
        const PetscInt  myelem = team.league_rank();
        g2_scr_t        g2(team.team_scratch(KOKKOS_SHARED_LEVEL),Nf,Nq,dim);
        g3_scr_t        g3(team.team_scratch(KOKKOS_SHARED_LEVEL),Nf,Nq,dim,dim);
        // get g2[] & g3[]
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,Nq), [=] (int myQi) {
            using Kokkos::parallel_reduce;
            const PetscInt              jpidx = myQi + myelem * Nq;
            const PetscReal     * const invJj = &d_invJ(jpidx*dim*dim);
            const PetscInt              ipdata_sz = (dim + Nf*(1+dim));
            const LandauPointData * const fplpt_j = (LandauPointData*)(&d_ipdata(jpidx*ipdata_sz));
            const PetscReal     * const vj = fplpt_j->crd, wj = d_wiGlobal[jpidx];
            // reduce on g22 and g33 for IP jpidx
            landau_inner_red::ValueType gg;
            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange (team, (int)nip), [=] (const int& ipidx, landau_inner_red::ValueType & ggg) {
                const LandauPointData * const fplpt = (LandauPointData*)(&d_ipdata(ipidx*ipdata_sz));
                const LandauFDF * const       fdf = &fplpt->fdf[0];
                const PetscReal             wi = d_wiGlobal[ipidx];
                PetscInt                    fieldA,fieldB,d2,d3;
#if LANDAU_DIM==2
                PetscReal                   Ud[2][2], Uk[2][2];
                LandauTensor2D(vj, fplpt->crd[0], fplpt->crd[1], Ud, Uk, (ipidx==jpidx) ? 0. : 1.);
                for (fieldA = 0; fieldA < Nf; ++fieldA) {
                  for (fieldB = 0; fieldB < Nf; ++fieldB) {
                    for (d2 = 0; d2 < 2; ++d2) {
                      for (d3 = 0; d3 < 2; ++d3) {
                        /* K = U * grad(f): g2=e: i,A */
                        ggg.gg2[fieldA][d2] += d_alpha[fieldA]*d_beta[fieldB] * d_invMass[fieldB] * Uk[d2][d3] * fdf[fieldB].df[d3] * wi;
                        /* D = -U * (I \kron (fx)): g3=f: i,j,A */
                        ggg.gg3[fieldA][d2][d3] -= d_alpha[fieldA]*d_beta[fieldB] * d_invMass[fieldA] * Ud[d2][d3] * fdf[fieldB].f * wi;
                      }
                    }
                  }
                }
#else
                PetscReal                   U[3][3];
                LandauTensor3D(vj, fplpt->crd[0], fplpt->crd[1], fplpt->crd[2], U, (ipidx==jpidx) ? 0. : 1.);
                for (fieldA = 0; fieldA < Nf; ++fieldA) {
                  for (fieldB = 0; fieldB < Nf; ++fieldB) {
                    for (d2 = 0; d2 < 3; ++d2) {
                      for (d3 = 0; d3 < 3; ++d3) {
                        /* K = U * grad(f): g2 = e: i,A */
                        ggg.gg2[fieldA][d2] += d_alpha[fieldA]*d_beta[fieldB] * d_invMass[fieldB] * U[d2][d3] * fplpt->fdf[fieldB].df[d3] * wi;
                        /* D = -U * (I \kron (fx)): g3 = f: i,j,A */
                        ggg.gg3[fieldA][d2][d3] -= d_alpha[fieldA]*d_beta[fieldB] * d_invMass[fieldA] * U[d2][d3] * fplpt->fdf[fieldB].f * wi;
                      }
                    }
                  }
                }
#endif
              }, Kokkos::Sum<landau_inner_red::ValueType>(gg));
            Kokkos::parallel_for(Kokkos::ThreadVectorRange (team, (int)Nf), [&] (const int& fieldA) {
                gg.gg2[fieldA][dim-1] += d_Eq_m[fieldA];
              });
            //kkos::single(Kokkos::PerThread(team), [&]() {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange (team, (int)Nf), [=] (const int& fieldA) {
                int d,d2,d3,dp;
                //printf("%d %d %d gg2[][1]=%18.10e\n",myelem,myQi,fieldA,gg.gg2[fieldA][dim-1]);
                /* Jacobian transform - g2, g3 - per thread (2D) */
                for (d = 0; d < dim; ++d) {
                  g2(fieldA,myQi,d) = 0;
                  for (d2 = 0; d2 < dim; ++d2) {
                    g2(fieldA,myQi,d) += invJj[d*dim+d2]*gg.gg2[fieldA][d2];
                    //printf("\t\t%d %d %d %d %d g2[%d][%d][%d]=%g\n",myelem,myQi,fieldA,d,d2,fieldA,myQi,d,g2(fieldA,myQi,d));
                    g3(fieldA,myQi,d,d2) = 0;
                    for (d3 = 0; d3 < dim; ++d3) {
                      for (dp = 0; dp < dim; ++dp) {
                        g3(fieldA,myQi,d,d2) += invJj[d*dim + d3]*gg.gg3[fieldA][d3][dp]*invJj[d2*dim + dp];
                        //printf("\t%d %d %d %d %d %d %d g3=%g wj=%g g3 = %g * %g * %g\n",myelem,myQi,fieldA,d,d2,d3,dp,g3(fieldA,myQi,d,d2),wj,invJj[d*dim + d3],gg.gg3[fieldA][d3][dp],invJj[d2*dim + dp]);
                      }
                    }
                    g3(fieldA,myQi,d,d2) *= wj;
                  }
                  g2(fieldA,myQi,d) *= wj;
                }
              });
          });
        team.team_barrier();
        /* assemble - on the diagonal (I,I) */
        //Kokkos::single(Kokkos::PerTeam(team), [&]() {
        { // int fieldA,blk_i;
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,Nb), [=] (int blk_i) {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,0,(int)Nf), [=] (int fieldA) {
                  //for (fieldA = 0; fieldA < Nf; ++fieldA) {
                  //for (blk_i = 0; blk_i < Nb; ++blk_i) {
                  int blk_j,qj,d,d2;
                  const PetscInt i = fieldA*Nb + blk_i; /* Element matrix row */
                  for (blk_j = 0; blk_j < Nb; ++blk_j) {
                    const PetscInt j    = fieldA*Nb + blk_j; /* Element matrix column */
                    const PetscInt fOff = i*totDim + j;
                    for (qj = 0 ; qj < Nq ; qj++) { // look at others integration points
                      const PetscReal *BJq = &d_BB[qj*Nb], *DIq = &d_DD[qj*Nb*dim];
                      for (d = 0; d < dim; ++d) {
                        d_elem_mats(myelem,fOff) += DIq[blk_i*dim+d]*g2(fieldA,qj,d)*BJq[blk_j];
                        //printf("\tmat[%d %d %d %d %d]=%g D[%d]=%g g2[%d][%d][%d]=%g B=%g\n",myelem,fOff,fieldA,qj,d,d_elem_mats(myelem,fOff),blk_i*dim+d,DIq[blk_i*dim+d],fieldA,qj,d,g2(fieldA,qj,d),BJq[blk_j]);
                        for (d2 = 0; d2 < dim; ++d2) {
                          d_elem_mats(myelem,fOff) += DIq[blk_i*dim + d]*g3(fieldA,qj,d,d2)*DIq[blk_j*dim + d2];
                        }
                      }
                    }
                  }
                });
            });
        }
      });
    Kokkos::fence();
    ierr = PetscLogEventEnd(events[4],0,0,0,0);CHKERRQ(ierr);

    ierr = PetscLogEventBegin(events[5],0,0,0,0);CHKERRQ(ierr);
    Kokkos::deep_copy (h_elem_mats, d_elem_mats);
    ierr = PetscLogEventEnd(events[5],0,0,0,0);CHKERRQ(ierr);

    ierr = PetscLogEventBegin(events[6],0,0,0,0);CHKERRQ(ierr);
#if defined(PETSC_HAVE_OPENMP)
    {
      PetscContainer container = NULL;
      ierr = PetscObjectQuery((PetscObject)JacP,"coloring",(PetscObject*)&container);CHKERRQ(ierr);
      if (!container) {
        ierr = PetscLogEventBegin(events[8],0,0,0,0);CHKERRQ(ierr);
        ierr = LandauCreateColoring(JacP, plex, &container);CHKERRQ(ierr);
        ierr = PetscLogEventEnd(events[8],0,0,0,0);CHKERRQ(ierr);
      }
      ierr = LandauAssembleOpenMP(cStart, cEnd, totDim, plex, section, globalSection, JacP, &h_elem_mats(0,0), container);CHKERRQ(ierr);
    }
#else
    {
      PetscInt ej;
      for (ej = cStart ; ej < cEnd; ++ej) {
        const PetscScalar *elMat = &h_elem_mats(ej-cStart,0);
        ierr = DMPlexMatSetClosure(plex, section, globalSection, JacP, ej, elMat, ADD_VALUES);CHKERRQ(ierr);
        if (ej==-1) {
          int d,f;
          printf("Kokkos Element matrix %d/%d\n",1,(int)numCells);
          for (d = 0; d < totDim; ++d){
            for (f = 0; f < totDim; ++f) printf(" %17.9e",  PetscRealPart(elMat[d*totDim + f]));
            printf("\n");
          }
        }
      }
    }
#endif
    ierr = PetscLogEventEnd(events[6],0,0,0,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
} // extern "C"
