/*
   Implements the Landau kernel
*/
#include <petscconf.h>
#include <petsc/private/dmpleximpl.h>   /*I  "dmpleximpl.h"   I*/
#include <petsclandau.h>
#define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1
#include <../src/mat/impls/aij/seq/aij.h>
#include <petscmat.h>
#include <petsccublas.h>

// hack to avoid configure problems in CI. Delete when resolved
#if !defined (PETSC_HAVE_CUDA_ATOMIC)
#define atomicAdd(e, f) (*e) += f
#endif
#define PETSC_DEVICE_FUNC_DECL __device__
#include "../land_tensors.h"
#include <petscaijdevice.h>

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err));        \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)
// Macro to catch CUDA errors in kernel launches
#define CHECK_LAUNCH_ERROR()                                          \
do {                                                                  \
    /* Check synchronous errors, i.e. pre-launch */                   \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err));        \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
    err = cudaDeviceSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err));       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

PETSC_EXTERN PetscErrorCode LandauCUDACreateMatMaps(P4estVertexMaps *maps, pointInterpolationP4est (*points)[LANDAU_MAX_Q_FACE], PetscInt Nf, PetscInt Nq)
{
  P4estVertexMaps h_maps;
  PetscFunctionBegin;
  h_maps.num_elements =maps->num_elements;
  h_maps.num_face = maps->num_face;
  h_maps.num_reduced = maps->num_reduced;
  h_maps.deviceType = maps->deviceType;
  h_maps.Nf = Nf;
  h_maps.Nq = Nq;
  CUDA_SAFE_CALL(cudaMalloc((void **)&h_maps.c_maps,               maps->num_reduced  * sizeof *points));
  CUDA_SAFE_CALL(cudaMemcpy(          h_maps.c_maps, maps->c_maps, maps->num_reduced  * sizeof *points, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&h_maps.gIdx,                 maps->num_elements * sizeof *maps->gIdx));
  CUDA_SAFE_CALL(cudaMemcpy(          h_maps.gIdx, maps->gIdx,     maps->num_elements * sizeof *maps->gIdx, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&maps->data, sizeof(P4estVertexMaps)));
  CUDA_SAFE_CALL(cudaMemcpy(          maps->data,   &h_maps, sizeof(P4estVertexMaps), cudaMemcpyHostToDevice));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode LandauCUDADestroyMatMaps(P4estVertexMaps *pMaps)
{
  P4estVertexMaps *d_maps = pMaps->data, h_maps;
  PetscFunctionBegin;
  CUDA_SAFE_CALL(cudaMemcpy(&h_maps, d_maps, sizeof(P4estVertexMaps), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(h_maps.c_maps));
  CUDA_SAFE_CALL(cudaFree(h_maps.gIdx));
  CUDA_SAFE_CALL(cudaFree(d_maps));
  PetscFunctionReturn(0);
}

// The GPU Landau kernel
//
__global__
void __launch_bounds__(256,1) landau_form_fdf(const PetscInt nip, const PetscInt dim, const PetscInt Nf, const PetscInt Nb, const PetscReal invJ_a[],
                   const PetscReal * const BB, const PetscReal * const DD, LandauIPReal *IPDataRaw, LandauIPReal d_f[], LandauIPReal d_dfdx[], LandauIPReal d_dfdy[], 
#if LANDAU_DIM==3
                   LandauIPReal d_dfdz[],
#endif
                   PetscErrorCode *ierr) // output
{
  const PetscInt  Nq = blockDim.y, myelem = blockIdx.x;
  const PetscInt  myQi = threadIdx.y;
  const PetscInt  jpidx = myQi + myelem * Nq;
  const PetscReal *invJ = &invJ_a[jpidx*dim*dim];
  const PetscReal *Bq = &BB[myQi*Nb], *Dq = &DD[myQi*Nb*dim];
  // un pack IPData
  LandauIPReal    *IPData_coefs = &IPDataRaw[nip*(dim+1)];
  LandauIPReal    *coef = &IPData_coefs[myelem*Nb*Nf];
  PetscInt        f,d,b,e;
  PetscScalar     u_x[LANDAU_MAX_SPECIES][LANDAU_DIM];
  *ierr = 0;
  /* get f and df */
  for (f = 0; f < Nf; ++f) {
    PetscScalar refSpaceDer[LANDAU_DIM];
    d_f[jpidx + f*nip] = 0.0;
    for (d = 0; d < LANDAU_DIM; ++d) refSpaceDer[d] = 0.0;
    for (b = 0; b < Nb; ++b) {
      const PetscInt    cidx = b;
      d_f[jpidx + f*nip] += Bq[cidx]*coef[f*Nb+cidx];
      for (d = 0; d < dim; ++d) refSpaceDer[d] += Dq[cidx*dim+d]*coef[f*Nb+cidx];
    }
    for (d = 0; d < dim; ++d) {
      for (e = 0, u_x[f][d] = 0.0; e < dim; ++e) {
	u_x[f][d] += invJ[e*dim+d]*refSpaceDer[e];
      }
    }
  }
  for (f=0;f<Nf;f++) {
    d_dfdx[jpidx + f*nip] = PetscRealPart(u_x[f][0]);
    d_dfdy[jpidx + f*nip] = PetscRealPart(u_x[f][1]);
#if LANDAU_DIM==3
    d_dfdz[jpidx + f*nip] = PetscRealPart(u_x[f][2]);
#endif
  }
}

__device__ void
landau_inner_integral_v2(const PetscInt myQi, const PetscInt jpidx, PetscInt nip, const PetscInt Nq, const PetscInt Nf, const PetscInt Nb,
			 const PetscInt dim, LandauIPReal *IPDataRaw, const PetscReal invJj[], const PetscReal nu_alpha[],
			 const PetscReal nu_beta[], const PetscReal invMass[], const PetscReal Eq_m[],
			 const PetscReal * const BB, const PetscReal * const DD,
			 PetscScalar *elemMat, P4estVertexMaps *d_maps, PetscSplitCSRDataStructure *d_mat, // output
			 PetscScalar fieldMats[][LANDAU_MAX_NQ], // all these arrays are in shared memory
			 PetscReal g2[][LANDAU_MAX_NQ][LANDAU_MAX_SPECIES],
			 PetscReal g3[][LANDAU_DIM][LANDAU_MAX_NQ][LANDAU_MAX_SPECIES],
			 PetscReal gg2[][LANDAU_MAX_NQ][LANDAU_MAX_SPECIES],
			 PetscReal gg3[][LANDAU_DIM][LANDAU_MAX_NQ][LANDAU_MAX_SPECIES],
			 PetscReal s_nu_alpha[],
			 PetscReal s_nu_beta[],
			 PetscReal s_invMass[],
			 PetscReal s_f[],
			 PetscReal s_dfx[],
			 PetscReal s_dfy[],
			 LandauIPReal d_f[], LandauIPReal d_dfdx[], LandauIPReal d_dfdy[], // global memory
#if LANDAU_DIM==3
			 PetscReal s_dfz[], LandauIPReal d_dfdz[],
#endif
			 PetscInt myelem, PetscErrorCode *ierr)
{
  int           delta,d,f,g,d2,dp,d3,fieldA,ipidx_b,nip_pad = nip; // vectorization padding not supported;
  PetscReal     gg2_temp[LANDAU_DIM], gg3_temp[LANDAU_DIM][LANDAU_DIM];
  LandauIPData  IPData;

  *ierr = 0;
  // create g2 & g3
  for (f=threadIdx.x; f<Nf; f+=blockDim.x) {
    for (d=0;d<dim;d++) { // clear accumulation data D & K
      gg2[d][myQi][f] = 0;
      for (d2=0;d2<dim;d2++) gg3[d][d2][myQi][f] = 0;
    }
  }
  if (threadIdx.y == 0) {
    for (int i = threadIdx.x; i < Nf; i += blockDim.x) {
      s_nu_alpha[i] = nu_alpha[i];
      s_nu_beta[i] = nu_beta[i];
      s_invMass[i] = invMass[i];
    }
  }
  for (d2 = 0; d2 < dim; d2++) {
    gg2_temp[d2] = 0;
    for (d3 = 0; d3 < dim; d3++) {
      gg3_temp[d2][d3] = 0;
    }
  }
  __syncthreads();
  // un pack IPData
  IPData.w   = IPDataRaw;
  IPData.x   = IPDataRaw + 1*nip_pad;
  IPData.y   = IPDataRaw + 2*nip_pad;
  IPData.z   = IPDataRaw + 3*nip_pad;

  const PetscReal vj[3] = {IPData.x[jpidx], IPData.y[jpidx], IPData.z ? IPData.z[jpidx] : 0}, wj = IPData.w[jpidx];
  for (ipidx_b = 0; ipidx_b < nip; ipidx_b += blockDim.x) {
    int ipidx = ipidx_b + threadIdx.x;

    __syncthreads();
    if (ipidx < nip) {
      for (fieldA = threadIdx.y; fieldA < Nf; fieldA += blockDim.y) {
        s_f  [fieldA*blockDim.x+threadIdx.x] =    d_f[ipidx + fieldA*nip_pad];
        s_dfx[fieldA*blockDim.x+threadIdx.x] = d_dfdx[ipidx + fieldA*nip_pad];
        s_dfy[fieldA*blockDim.x+threadIdx.x] = d_dfdy[ipidx + fieldA*nip_pad];
#if LANDAU_DIM==3
        s_dfz[fieldA*blockDim.x+threadIdx.x] = d_dfdz[ipidx + fieldA*nip_pad];
#endif
      }
    }
    __syncthreads();
    if (ipidx < nip) {
      const PetscReal wi = IPData.w[ipidx], x = IPData.x[ipidx], y = IPData.y[ipidx];
      PetscReal       temp1[3] = {0, 0, 0}, temp2 = 0;
#if LANDAU_DIM==2
      PetscReal Ud[2][2], Uk[2][2];
      LandauTensor2D(vj, x, y, Ud, Uk, (ipidx==jpidx) ? 0. : 1.);
#else
      PetscReal U[3][3], z = IPData.z[ipidx];
      LandauTensor3D(vj, x, y, z, U, (ipidx==jpidx) ? 0. : 1.);
#endif
      for (fieldA = 0; fieldA < Nf; fieldA++) {
        temp1[0] += s_dfx[fieldA*blockDim.x+threadIdx.x]*s_nu_beta[fieldA]*s_invMass[fieldA];
        temp1[1] += s_dfy[fieldA*blockDim.x+threadIdx.x]*s_nu_beta[fieldA]*s_invMass[fieldA];
#if LANDAU_DIM==3
        temp1[2] += s_dfz[fieldA*blockDim.x+threadIdx.x]*s_nu_beta[fieldA]*s_invMass[fieldA];
#endif
        temp2    += s_f  [fieldA*blockDim.x+threadIdx.x]*s_nu_beta[fieldA];
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
          gg2_temp[d2] += Uk[d2][d3]*temp1[d3];
          /* D = -U * (I \kron (fx)): g3=f: i,j,A */
          gg3_temp[d2][d3] += Ud[d2][d3]*temp2;
        }
      }
      #else
      for (d2 = 0; d2 < 3; ++d2) {
        for (d3 = 0; d3 < 3; ++d3) {
          /* K = U * grad(f): g2 = e: i,A */
          gg2_temp[d2] += U[d2][d3]*temp1[d3];
          /* D = -U * (I \kron (fx)): g3 = f: i,j,A */
          gg3_temp[d2][d3] += U[d2][d3]*temp2;
        }
      }
      #endif
    }
  } /* IPs */
 

  /* reduce gg temp sums across threads */
  for (delta = blockDim.x/2; delta > 0; delta /= 2) {
    for (d2 = 0; d2 < dim; d2++) {
      gg2_temp[d2] += __shfl_xor_sync(0xffffffff, gg2_temp[d2], delta, blockDim.x);
      for (d3 = 0; d3 < dim; d3++) {
        gg3_temp[d2][d3] += __shfl_xor_sync(0xffffffff, gg3_temp[d2][d3], delta, blockDim.x);
      }
    }
  }

  // add alpha and put in gg2/3
  for (fieldA = threadIdx.x; fieldA < Nf; fieldA += blockDim.x) {
    for (d2 = 0; d2 < dim; d2++) {
      gg2[d2][myQi][fieldA] += gg2_temp[d2]*s_nu_alpha[fieldA];
      for (d3 = 0; d3 < dim; d3++) {
        gg3[d2][d3][myQi][fieldA] -= gg3_temp[d2][d3]*s_nu_alpha[fieldA]*s_invMass[fieldA];
      }
    }
  }
  __syncthreads();

  /* add electric field term once per IP */
  for (fieldA = threadIdx.x; fieldA < Nf; fieldA += blockDim.x) {
    gg2[dim-1][myQi][fieldA] += Eq_m[fieldA];
  }
  __syncthreads();
  //intf("%d %d gg2[1][1]=%g\n",myelem,qj_start,gg2[1][dim-1]);
  /* Jacobian transform - g2 */
  for (fieldA = threadIdx.x; fieldA < Nf; fieldA += blockDim.x) {
    for (d = 0; d < dim; ++d) {
      g2[d][myQi][fieldA] = 0.0;
      for (d2 = 0; d2 < dim; ++d2) {
        g2[d][myQi][fieldA] += invJj[d*dim+d2]*gg2[d2][myQi][fieldA];
        g3[d][d2][myQi][fieldA] = 0.0;
        for (d3 = 0; d3 < dim; ++d3) {
          for (dp = 0; dp < dim; ++dp) {
            g3[d][d2][myQi][fieldA] += invJj[d*dim + d3]*gg3[d3][dp][myQi][fieldA]*invJj[d2*dim + dp];
          }
        }
        g3[d][d2][myQi][fieldA] *= wj;
      }
      g2[d][myQi][fieldA] *= wj;
    }
  }
  /* FE matrix construction */
  __syncthreads();  // Synchronize (ensure all the data is available) and sum IP matrices
  {
    int fieldA,d,qj,d2,q,idx,totDim=Nb*Nf;
    /* assemble */
    for (fieldA = 0; fieldA < Nf; fieldA++) {
      if (fieldMats) {
	for (f = threadIdx.y; f < Nb ; f += blockDim.y) {
	  for (g = threadIdx.x; g < Nb; g += blockDim.x) {
	    fieldMats[f][g] = 0;
	  }
	}
      }
      for (f = threadIdx.y; f < Nb ; f += blockDim.y) {
        const PetscInt i = fieldA*Nb + f; /* Element matrix row */
        for (g = threadIdx.x; g < Nb; g += blockDim.x) {
          const PetscInt j    = fieldA*Nb + g; /* Element matrix column */
          const PetscInt fOff = i*totDim + j;
          PetscScalar t = elemMat ? elemMat[fOff] : fieldMats[f][g];
          for (qj = 0 ; qj < Nq ; qj++) {
            const PetscReal *BJq = &BB[qj*Nb], *DIq = &DD[qj*Nb*dim];
            for (d = 0; d < dim; ++d) {
              t += DIq[f*dim+d]*g2[d][qj][fieldA]*BJq[g];
              for (d2 = 0; d2 < dim; ++d2) {
                t += DIq[f*dim + d]*g3[d][d2][qj][fieldA]*DIq[g*dim + d2];
              }
            }
          }
	  if (elemMat) elemMat[fOff] = t;
	  else fieldMats[f][g] = t;
	}
      }
      if (fieldMats) {
	PetscScalar            vals[LANDAU_MAX_Q*LANDAU_MAX_Q];
	PetscReal              row_scale[LANDAU_MAX_Q],col_scale[LANDAU_MAX_Q];
	PetscInt               nr,nc,rows0[LANDAU_MAX_Q],cols0[LANDAU_MAX_Q],rows[LANDAU_MAX_Q],cols[LANDAU_MAX_Q];
	const LandauIdx *const Idxs = &d_maps->gIdx[myelem][fieldA][0];
	for (f = threadIdx.y; f < Nb ; f += blockDim.y) {
	  idx = Idxs[f];
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
	  for (g = threadIdx.x; g < Nb; g += blockDim.x) {
	    idx = Idxs[g];
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
		vals[q*nc + d] = row_scale[q]*col_scale[d]*fieldMats[f][g];
	      }
	    }
	    MatSetValuesDevice(d_mat,nr,rows,nc,cols,vals,ADD_VALUES,ierr);
	    if (*ierr) return;
	  }
	}
      }
    }
  }
}

//
// The GPU Landau kernel
//
__global__
void __launch_bounds__(256,1) landau_kernel_v2(const PetscInt nip, const PetscInt dim, const PetscInt totDim, const PetscInt Nf, const PetscInt Nb, const PetscReal invJj[],
  const PetscReal nu_alpha[], const PetscReal nu_beta[], const PetscReal invMass[], const PetscReal Eq_m[],
  const PetscReal * const BB, const PetscReal * const DD, LandauIPReal *IPDataRaw,
  PetscScalar elemMats_out[], P4estVertexMaps *d_maps, PetscSplitCSRDataStructure *d_mat, LandauIPReal d_f[], LandauIPReal d_dfdx[], LandauIPReal d_dfdy[], 
#if LANDAU_DIM==3
					       LandauIPReal d_dfdz[],
#endif
					       PetscErrorCode *ierr)
{
  const PetscInt  Nq = blockDim.y, myelem = blockIdx.x;
  extern __shared__ PetscReal smem[];
  int size = 0;
  PetscReal (*g2)[LANDAU_DIM][LANDAU_MAX_NQ][LANDAU_MAX_SPECIES]              =
    (PetscReal (*)[LANDAU_DIM][LANDAU_MAX_NQ][LANDAU_MAX_SPECIES])             &smem[size];
  size += LANDAU_MAX_NQ*LANDAU_MAX_SPECIES*LANDAU_DIM;
  PetscReal (*g3)[LANDAU_DIM][LANDAU_DIM][LANDAU_MAX_NQ][LANDAU_MAX_SPECIES]  =
    (PetscReal (*)[LANDAU_DIM][LANDAU_DIM][LANDAU_MAX_NQ][LANDAU_MAX_SPECIES]) &smem[size];
  size += LANDAU_DIM*LANDAU_DIM*LANDAU_MAX_NQ*LANDAU_MAX_SPECIES;
  PetscReal (*gg2)[LANDAU_DIM][LANDAU_MAX_NQ][LANDAU_MAX_SPECIES]             =
    (PetscReal (*)[LANDAU_DIM][LANDAU_MAX_NQ][LANDAU_MAX_SPECIES])             &smem[size];
  size += LANDAU_MAX_NQ*LANDAU_MAX_SPECIES*LANDAU_DIM;
  PetscReal (*gg3)[LANDAU_DIM][LANDAU_DIM][LANDAU_MAX_NQ][LANDAU_MAX_SPECIES] =
    (PetscReal (*)[LANDAU_DIM][LANDAU_DIM][LANDAU_MAX_NQ][LANDAU_MAX_SPECIES]) &smem[size];
  size += LANDAU_DIM*LANDAU_DIM*LANDAU_MAX_NQ*LANDAU_MAX_SPECIES;
  PetscReal *s_nu_alpha = &smem[size];
  size += LANDAU_MAX_SPECIES;
  PetscReal *s_nu_beta  = &smem[size];
  size += LANDAU_MAX_SPECIES;
  PetscReal *s_invMass  = &smem[size];
  size += LANDAU_MAX_SPECIES;
  PetscReal *s_f        = &smem[size];
  size += blockDim.x*LANDAU_MAX_SPECIES;
  PetscReal *s_dfx      = &smem[size];
  size += blockDim.x*LANDAU_MAX_SPECIES;
  PetscReal *s_dfy      = &smem[size];
  size += blockDim.x*LANDAU_MAX_SPECIES;
#if LANDAU_DIM==3
  PetscReal *s_dfz      = &smem[size];
  size += blockDim.x*LANDAU_MAX_SPECIES;
#endif
  PetscScalar (*fieldMats)[LANDAU_MAX_NQ][LANDAU_MAX_NQ] = d_maps ?
    (PetscScalar (*)[LANDAU_MAX_NQ][LANDAU_MAX_NQ]) &smem[size] : NULL;
  if (d_maps) size += LANDAU_MAX_NQ*LANDAU_MAX_NQ;
  const PetscInt  myQi = threadIdx.y;
  const PetscInt  jpidx = myQi + myelem * Nq;
  //const PetscInt  subblocksz = nip/nSubBlks + !!(nip%nSubBlks), ip_start = mySubBlk*subblocksz, ip_end = (mySubBlk+1)*subblocksz > nip ? nip : (mySubBlk+1)*subblocksz; /* this could be wrong with very few global IPs */
  PetscScalar     *elemMat  = elemMats_out ? &elemMats_out[myelem*totDim*totDim] : NULL; /* my output */
  int tid = threadIdx.x + threadIdx.y*blockDim.x;

  if (elemMat) for (int i = tid; i < totDim*totDim; i += blockDim.x*blockDim.y) elemMat[i] = 0;
  __syncthreads();

  landau_inner_integral_v2(myQi, jpidx, nip, Nq, Nf, Nb, dim, IPDataRaw, &invJj[jpidx*dim*dim], nu_alpha, nu_beta, invMass, Eq_m, BB, DD,
			   elemMat, d_maps, d_mat, *fieldMats, *g2, *g3, *gg2, *gg3, s_nu_alpha, s_nu_beta, s_invMass, s_f, s_dfx, s_dfy, d_f, d_dfdx, d_dfdy,
#if LANDAU_DIM==3
			   s_dfz, d_dfdz,
#endif
			   myelem, ierr); /* compact */
}

PetscErrorCode LandauCUDAJacobian(DM plex, const PetscInt Nq, const PetscReal nu_alpha[],const PetscReal nu_beta[], const PetscReal invMass[], const PetscReal Eq_m[],
				  const LandauIPData *const IPData, const PetscReal invJj[], const PetscLogEvent events[], Mat JacP)
{
  PetscErrorCode    ierr,*d_ierr;
  //cudaError_t       cerr;
  PetscInt          ii,ej,*Nbf,Nb,nip_dim2,cStart,cEnd,Nf,dim,numGCells,totDim,nip,szf=sizeof(LandauIPReal),ipdatasz;
  PetscReal         *d_BB,*d_DD,*d_invJj,*d_nu_alpha,*d_nu_beta,*d_invMass,*d_Eq_m;
  PetscScalar       *d_elemMats=NULL;
  LandauIPReal       *d_f, *d_dfdx, *d_dfdy;
#if LANDAU_DIM==3
  PetscScalar       *d_dfdz;
#endif
  PetscLogDouble    flops;
  PetscTabulation   *Tf;
  PetscDS           prob;
  PetscSection      section, globalSection;
  LandauIPReal      *d_IPDataRaw;
  LandauCtx         *ctx;
  PetscSplitCSRDataStructure *d_mat=NULL;
  P4estVertexMaps            *h_maps, *d_maps=NULL;
  int               nnn = 256/Nq;

  PetscFunctionBegin;
  while (nnn & nnn - 1) nnn = nnn & nnn - 1;
  if (nnn>16) nnn = 16;
  ierr = PetscLogEventBegin(events[3],0,0,0,0);CHKERRQ(ierr);
  ierr = DMGetDimension(plex, &dim);CHKERRQ(ierr);
  if (dim!=LANDAU_DIM) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "LANDAU_DIM %D != dim %d",LANDAU_DIM,dim);
  ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
  numGCells = cEnd - cStart;
  nip  = numGCells*Nq; /* length of inner global iteration */
  ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(prob, &Nbf);CHKERRQ(ierr); Nb = Nbf[0];
  if (Nq != Nb) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Nq != Nb. %D  %D",Nq,Nb);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(prob, &Tf);CHKERRQ(ierr);
  ierr = DMGetLocalSection(plex, &section);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(plex, &globalSection);CHKERRQ(ierr);
  ipdatasz = LandauGetIPDataSize(IPData);
  // create data
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_IPDataRaw,ipdatasz*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_nu_alpha, Nf*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_nu_beta,  Nf*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_invMass,  Nf*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_Eq_m,     Nf*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(d_IPDataRaw, IPData->w, ipdatasz*szf, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_nu_alpha, nu_alpha, Nf*szf,       cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_nu_beta,  nu_beta,  Nf*szf,       cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_invMass,  invMass,  Nf*szf,       cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_Eq_m,     Eq_m,     Nf*szf,       cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_BB,              Nq*Nb*szf));     // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          d_BB, Tf[0]->T[0], Nq*Nb*szf,   cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_DD,              Nq*Nb*dim*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          d_DD, Tf[0]->T[1], Nq*Nb*dim*szf,   cudaMemcpyHostToDevice));
  // f and df
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_f,    nip*Nf*szf));     // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_dfdx, nip*Nf*szf));     // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_dfdy, nip*Nf*szf));     // kernel input
#if LANDAU_DIM==3
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_dfdz, nip*Nf*szf));     // kernel input
#endif
  // collect geometry
  flops = (PetscLogDouble)numGCells*(PetscLogDouble)Nq*(PetscLogDouble)(5.*dim*dim*Nf*Nf + 165.);
  nip_dim2 = Nq*numGCells*dim*dim;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_invJj, nip_dim2*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(d_invJj, invJj, nip_dim2*szf,       cudaMemcpyHostToDevice));
  //cerr = WaitForCUDA();CHKERRCUDA(cerr);
  ierr = PetscLogEventEnd(events[3],0,0,0,0);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(events[4],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(flops*nip);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(plex, &ctx);CHKERRQ(ierr);
  if (!ctx) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  if (ctx->gpu_assembly) {
    PetscContainer container;
    ierr = PetscObjectQuery((PetscObject) JacP, "assembly_maps", (PetscObject *) &container);CHKERRQ(ierr);
    if (container) { // not here first call
      ierr = PetscContainerGetPointer(container, (void **) &h_maps);CHKERRQ(ierr);
      if (h_maps->data) {
        d_maps = h_maps->data;
	if (!d_maps) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "GPU assembly but no metadata");
      } else {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "GPU assembly but no metadata in container");
      }
      // this does the setup the first time called
      ierr = MatCUSPARSEGetDeviceMatWrite(JacP,&d_mat);CHKERRQ(ierr);
    } else {
      CUDA_SAFE_CALL(cudaMalloc((void **)&d_elemMats, totDim*totDim*numGCells*sizeof(PetscScalar))); // kernel output - first call is on CPU
    }
  } else {
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_elemMats, totDim*totDim*numGCells*sizeof(PetscScalar))); // kernel output - no GPU assembly
  }
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_ierr, sizeof(ierr))); // kernel input
  { // form f and df
    dim3 dimBlock(nnn,Nq);
    ierr = PetscLogEventBegin(events[8],0,0,0,0);CHKERRQ(ierr);
    ii = 0;
    // PetscPrintf(PETSC_COMM_SELF, "numGCells=%d dim.x=%d Nq=%d nThreads=%d, %d kB shared mem\n",numGCells,n,Nq,Nq*n,ii*szf/1024);
    landau_form_fdf<<<numGCells,dimBlock,ii*szf>>>( nip, dim, Nf, Nb, d_invJj, d_BB, d_DD, d_IPDataRaw, d_f, d_dfdx, d_dfdy,
#if LANDAU_DIM==3
						    d_dfdz,
#endif
						    d_ierr);
    CHECK_LAUNCH_ERROR();
    CUDA_SAFE_CALL(cudaMemcpy(&ierr, d_ierr, sizeof(ierr), cudaMemcpyDeviceToHost));
    CHKERRQ(ierr);
    ierr = PetscLogEventEnd(events[8],0,0,0,0);CHKERRQ(ierr);
  }
  {
    dim3 dimBlock(nnn,Nq);
    ii = 2*LANDAU_MAX_NQ*LANDAU_MAX_SPECIES*LANDAU_DIM*(1+LANDAU_DIM) + 3*LANDAU_MAX_SPECIES + (1+LANDAU_DIM)*dimBlock.x*LANDAU_MAX_SPECIES;
    ii += (LANDAU_MAX_NQ*LANDAU_MAX_NQ)*LANDAU_MAX_SPECIES;
    if (ii*szf >= 49152) {
      CUDA_SAFE_CALL(cudaFuncSetAttribute(landau_kernel_v2,
					  cudaFuncAttributeMaxDynamicSharedMemorySize,
					  98304));
    }
    // PetscPrintf(PETSC_COMM_SELF, "numGCells=%d dim.x=%d Nq=%d nThreads=%d, %d kB shared mem\n",numGCells,n,Nq,Nq*n,ii*szf/1024);
    landau_kernel_v2<<<numGCells,dimBlock,ii*szf>>>(nip,dim,totDim,Nf,Nb,d_invJj,d_nu_alpha,d_nu_beta,d_invMass,d_Eq_m,
						    d_BB, d_DD, d_IPDataRaw, d_elemMats, d_maps, d_mat, d_f, d_dfdx, d_dfdy,
#if LANDAU_DIM==3
						    d_dfdz,
#endif
						    d_ierr);
    CHECK_LAUNCH_ERROR();
    CUDA_SAFE_CALL(cudaMemcpy(&ierr, d_ierr, sizeof(ierr), cudaMemcpyDeviceToHost));
    CHKERRQ(ierr);
  }
  CUDA_SAFE_CALL(cudaFree(d_ierr));
  //cerr = WaitForCUDA();CHKERRCUDA(cerr);
  ierr = PetscLogEventEnd(events[4],0,0,0,0);CHKERRQ(ierr);
  // delete device data
  ierr = PetscLogEventBegin(events[5],0,0,0,0);CHKERRQ(ierr);
  CUDA_SAFE_CALL(cudaFree(d_IPDataRaw));
  CUDA_SAFE_CALL(cudaFree(d_invJj));
  CUDA_SAFE_CALL(cudaFree(d_nu_alpha));
  CUDA_SAFE_CALL(cudaFree(d_nu_beta));
  CUDA_SAFE_CALL(cudaFree(d_invMass));
  CUDA_SAFE_CALL(cudaFree(d_Eq_m));
  CUDA_SAFE_CALL(cudaFree(d_BB));
  CUDA_SAFE_CALL(cudaFree(d_DD));
  CUDA_SAFE_CALL(cudaFree(d_f));
  CUDA_SAFE_CALL(cudaFree(d_dfdx));
  CUDA_SAFE_CALL(cudaFree(d_dfdy));
#if LANDAU_DIM==3
  CUDA_SAFE_CALL(cudaFree(d_dfdz));
#endif
  //cerr = WaitForCUDA();CHKERRCUDA(cerr);
  ierr = PetscLogEventEnd(events[5],0,0,0,0);CHKERRQ(ierr);
  // First time assembly even with GPU assembly
  if (d_elemMats) {
    PetscScalar *elemMats=NULL,*elMat;
    ierr = PetscLogEventBegin(events[5],0,0,0,0);CHKERRQ(ierr);
    ierr = PetscMalloc1(totDim*totDim*numGCells,&elemMats);CHKERRQ(ierr);
    CUDA_SAFE_CALL(cudaMemcpy(elemMats, d_elemMats, totDim*totDim*numGCells*sizeof(PetscScalar), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(d_elemMats));
    ierr = PetscLogEventEnd(events[5],0,0,0,0);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(events[6],0,0,0,0);CHKERRQ(ierr);
    for (ej = cStart, elMat = elemMats ; ej < cEnd; ++ej, elMat += totDim*totDim) {
      ierr = DMPlexMatSetClosure(plex, section, globalSection, JacP, ej, elMat, ADD_VALUES);CHKERRQ(ierr);
      if (ej==-1) {
        int d,f;
        PetscPrintf(PETSC_COMM_SELF,"GPU Element matrix\n");
        for (d = 0; d < totDim; ++d){
          for (f = 0; f < totDim; ++f) PetscPrintf(PETSC_COMM_SELF," %12.5e",  PetscRealPart(elMat[d*totDim + f]));
          PetscPrintf(PETSC_COMM_SELF,"\n");
        }
      }
    }
    ierr = PetscFree(elemMats);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(events[6],0,0,0,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
