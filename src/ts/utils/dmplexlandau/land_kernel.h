#include "./land_tensors.h"

/* landau_inner_integral() */
/* Compute g2 and g3 for element, assemble into eleme matrix */
PETSC_DEVICE_FUNC_DECL void
landau_inner_integral(const PetscInt myQi, const PetscInt qi_inc, const PetscInt mySubBlk, const PetscInt nSubBlks, const PetscInt ip_start, const PetscInt ip_end, const PetscInt ip_stride, /* decomposition args, not discretization */
                       const PetscInt jpidx, const PetscInt Nf, const PetscInt dim, const PetscReal * const IPDataGlobal, const PetscReal wiGlobal[], const PetscReal invJj[],
                       const PetscReal nu_alpha[], const PetscReal nu_beta[], const PetscReal invMass[], const PetscReal Eq_m[], PetscBool quarter3DDomain,
                       const PetscInt Nq, const PetscInt Nb, const PetscInt qj_start, const PetscInt qj_end, const PetscReal * const BB, const PetscReal * const DD, PetscScalar *elemMat, /* discretization args; local output */
                       PetscReal g2[/* LANDAU_MAX_NQ */][LANDAU_MAX_SUB_THREAD_BLOCKS][LANDAU_MAX_SPECIES][LANDAU_DIM], PetscReal g3[/* LANDAU_MAX_NQ */][LANDAU_MAX_SUB_THREAD_BLOCKS][LANDAU_MAX_SPECIES][LANDAU_DIM][LANDAU_DIM] /* shared memory buffers */
                       , PetscInt myelem)
{
  PetscReal                   gg2[LANDAU_MAX_SPECIES][LANDAU_DIM],gg3[LANDAU_MAX_SPECIES][LANDAU_DIM][LANDAU_DIM];
  const PetscInt              ipdata_sz = (dim + Nf*(1+dim));
  PetscInt                    d,f,d2,dp,d3,fieldB,ipidx,fieldA;
  const LandauPointData * const fplpt_j = (LandauPointData*)(IPDataGlobal + jpidx*ipdata_sz);
  const PetscReal * const     vj = fplpt_j->crd, wj = wiGlobal[jpidx];
  // create g2 & g3
  for (d=0;d<dim;d++) { // clear accumulation data D & K
    for (f=0;f<Nf;f++) {
      gg2[f][d] = 0;
      for (d2=0;d2<dim;d2++) gg3[f][d][d2] = 0;
    }
  }
  for (ipidx = ip_start; ipidx < ip_end; ipidx += ip_stride) {
    const LandauPointData * const fplpt = (LandauPointData*)(IPDataGlobal + ipidx*ipdata_sz);
    const LandauFDF * const       fdf = &fplpt->fdf[0];
    const PetscReal             wi = wiGlobal[ipidx];
#if LANDAU_DIM==2
    PetscReal       Ud[2][2], Uk[2][2];
    LandauTensor2D(vj, fplpt->crd[0], fplpt->crd[1], Ud, Uk, (ipidx==jpidx) ? 0. : 1.);
    for (fieldA = 0; fieldA < Nf; ++fieldA) {
      for (fieldB = 0; fieldB < Nf; ++fieldB) {
        for (d2 = 0; d2 < 2; ++d2) {
          for (d3 = 0; d3 < 2; ++d3) {
            /* K = U * grad(f): g2=e: i,A */
            gg2[fieldA][d2] += nu_alpha[fieldA]*nu_beta[fieldB] * invMass[fieldB] * Uk[d2][d3] * fdf[fieldB].df[d3] * wi;
            /* D = -U * (I \kron (fx)): g3=f: i,j,A */
            gg3[fieldA][d2][d3] -= nu_alpha[fieldA]*nu_beta[fieldB] * invMass[fieldA] * Ud[d2][d3] * fdf[fieldB].f * wi;
          }
        }
      }
    }
#else
    PetscReal U[3][3];
    if (!quarter3DDomain) {
      LandauTensor3D(vj, fplpt->crd[0], fplpt->crd[1], fplpt->crd[2], U, (ipidx==jpidx) ? 0. : 1.);
      for (fieldA = 0; fieldA < Nf; ++fieldA) {
        for (fieldB = 0; fieldB < Nf; ++fieldB) {
          for (d2 = 0; d2 < 3; ++d2) {
            for (d3 = 0; d3 < 3; ++d3) {
              /* K = U * grad(f): g2 = e: i,A */
              gg2[fieldA][d2] += nu_alpha[fieldA]*nu_beta[fieldB] * invMass[fieldB] * U[d2][d3] * fplpt->fdf[fieldB].df[d3] * wi;
              /* D = -U * (I \kron (fx)): g3 = f: i,j,A */
              gg3[fieldA][d2][d3] -= nu_alpha[fieldA]*nu_beta[fieldB] * invMass[fieldA] * U[d2][d3] * fplpt->fdf[fieldB].f * wi;
            }
          }
        }
      }
    } else {
      PetscReal lxx[] = {fplpt->crd[0], fplpt->crd[1]}, R[2][2] = {{-1,1},{1,-1}};
      PetscReal ldf[3*LANDAU_MAX_SPECIES];
      for (fieldB = 0; fieldB < Nf; ++fieldB) for (d3 = 0; d3 < 3; ++d3) ldf[d3 + fieldB*3] = fplpt->fdf[fieldB].df[d3] * wi * invMass[fieldB];
      for (dp=0;dp<4;dp++) {
        LandauTensor3D(vj, lxx[0], lxx[1], fplpt->z, U, (ipidx==jpidx) ? 0. : 1.);
        for (fieldA = 0; fieldA < Nf; ++fieldA) {
          for (fieldB = 0; fieldB < Nf; ++fieldB) {
            for (d2 = 0; d2 < 3; ++d2) {
              for (d3 = 0; d3 < 3; ++d3) {
                /* K = U * grad(f): g2 = e: i,A */
                gg2[fieldA][d2] += nu_alpha[fieldA]*nu_beta[fieldB] * U[d2][d3] * ldf[d3 + fieldB*3];
                /* D = -U * (I \kron (fx)): g3 = f: i,j,A */
                gg3[fieldA][d2][d3] -= nu_alpha[fieldA]*nu_beta[fieldB] * invMass[fieldA] * U[d2][d3] * f[fieldB] * wi;
              }
            }
          }
        }
        for (d3 = 0; d3 < 2; ++d3) {
          lxx[d3] *= R[d3][dp%2];
          for (fieldB = 0; fieldB < Nf; ++fieldB) {
            ldf[d3 + fieldB*3] *= R[d3][dp%2];
          }
        }
      }
    }
#endif
  } /* IPs */
  /* add electric field term once per IP */
  if (mySubBlk==0) {
    for (fieldA = 0; fieldA < Nf; ++fieldA) {
      gg2[fieldA][dim-1] += Eq_m[fieldA];
    }
  }
  //intf("%d %d gg2[1][1]=%g\n",myelem,qj_start,gg2[1][dim-1]);
  /* Jacobian transform - g2 */
  for (fieldA = 0; fieldA < Nf; ++fieldA) {
    for (d = 0; d < dim; ++d) {
      g2[myQi][mySubBlk][fieldA][d] = 0.0;
      for (d2 = 0; d2 < dim; ++d2) {
        g2[myQi][mySubBlk][fieldA][d] += invJj[d*dim+d2]*gg2[fieldA][d2];
        //printf("\t:g2[%d][%d][%d]=%g\n",fieldA,qj_start,d,g2[myQi][mySubBlk][fieldA][d]);
        g3[myQi][mySubBlk][fieldA][d][d2] = 0.0;
        for (d3 = 0; d3 < dim; ++d3) {
          for (dp = 0; dp < dim; ++dp) {
            g3[myQi][mySubBlk][fieldA][d][d2] += invJj[d*dim + d3]*gg3[fieldA][d3][dp]*invJj[d2*dim + dp];
            //printf("\t\t\t:%d %d %d %d g3=%g\n",qj_start,fieldA,d,d2,g3[myQi][mySubBlk][fieldA][d][d2]);
          }
        }
        g3[myQi][mySubBlk][fieldA][d][d2] *= wj;
      }
      g2[myQi][mySubBlk][fieldA][d] *= wj;
    }
  }
  // Synchronize (ensure all the data is available) and sum g2 & g3
  PETSC_THREAD_SYNC;
  if (mySubBlk==0) { /* on one thread, sum up g2 & g3 (noop with one subblock) -- could parallelize! */
    for (fieldA = 0; fieldA < Nf; ++fieldA) {
      for (d = 0; d < dim; ++d) {
        for (d3 = 1; d3 < nSubBlks; ++d3) {
          g2[myQi][0][fieldA][d] += g2[myQi][d3][fieldA][d];
          for (dp = 0; dp < dim; ++dp) {
            g3[myQi][0][fieldA][d][dp] += g3[myQi][d3][fieldA][d][dp];
          }
        }
      }
    }
  }

  /* FE matrix construction */
  PETSC_THREAD_SYNC;   // Synchronize (ensure all the data is available) and sum IP matrices
  {
    PetscInt  fieldA,d,f,qj,qj_0,d2,g,totDim=Nb*Nf;
    /* assemble - on the diagonal (I,I) */
    for (fieldA = mySubBlk; fieldA < Nf ; fieldA += nSubBlks) {
      for (f = myQi; f < Nb ; f += qi_inc) { /* vectorizing here, maybe */
        const PetscInt i = fieldA*Nb + f; /* Element matrix row */
        for (g = 0; g < Nb; ++g) {
          const PetscInt j    = fieldA*Nb + g; /* Element matrix column */
          const PetscInt fOff = i*totDim + j;
          for (qj = qj_start, qj_0 = 0 ; qj < qj_end ; qj++, qj_0++) {
            const PetscReal *BJq = &BB[qj*Nb], *DIq = &DD[qj*Nb*dim];
            for (d = 0; d < dim; ++d) {
              elemMat[fOff] += DIq[f*dim+d]*g2[qj_0][0][fieldA][d]*BJq[g];
              //intf("\tmat[%d %d %d %d %d]=%g D[%d]=%g g2[%d][%d][%d]=%g B=%g\n", print, fOff,fieldA,qj,d, elemMat[fOff],f*dim+d,DIq[f*dim+d],fieldA,qj,d,g2[qj_0][0][fieldA][d],BJq[g]);
              for (d2 = 0; d2 < dim; ++d2) {
                elemMat[fOff] += DIq[f*dim + d]*g3[qj_0][0][fieldA][d][d2]*DIq[g*dim + d2];
              }
            }
          }
        }
      }
    }
  }
}
