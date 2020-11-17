/*
   Implements the Landau kernel
*/
#include <petscconf.h>
#include <petsc/private/dmpleximpl.h>   /*I  "dmpleximpl.h"   I*/
#include <petsclandau.h>
#include <../src/mat/impls/aij/seq/aij.h>
#include <petsc/private/kernels/petscaxpy.h>

#define PETSC_THREAD_SYNC __syncthreads()
#define PETSC_DEVICE_FUNC_DECL __device__
#include "../land_tensors.h"

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
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
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
    err = cudaDeviceSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

__device__ void
landau_inner_integral_v2(const PetscInt myQi, const PetscInt jpidx, PetscInt nip, const PetscInt Nq, const PetscInt Nf, const PetscInt Nb,
			 const PetscInt dim, LandauIPReal *IPDataRaw, const PetscReal invJj[], const PetscReal nu_alpha[],
			 const PetscReal nu_beta[], const PetscReal invMass[], const PetscReal Eq_m[],
			 const PetscReal * const BB, const PetscReal * const DD,
			 PetscScalar *elemMat, // output
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
#if LANDAU_DIM==3
                         PetscReal s_dfz[],
#endif
                         PetscInt myelem)
{
  PetscInt                      d,f,d2,dp,d3,fieldA;
  PetscReal                     gg2_temp[LANDAU_DIM], gg3_temp[LANDAU_DIM][LANDAU_DIM];
  LandauIPData                  IPData;
  PetscInt                      nip_pad = nip; // vectorization padding not supported

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
  // pack IPData
  IPData.w_data   = IPDataRaw;
  IPData.x   = IPDataRaw + 1*nip_pad;
  IPData.y   = IPDataRaw + 2*nip_pad;
  IPData.z   = IPDataRaw + 3*nip_pad;
  IPData.f   = IPDataRaw + nip_pad*((dim+1) + 0);
  IPData.dfx = IPDataRaw + nip_pad*((dim+1) + 1*Nf);
  IPData.dfy = IPDataRaw + nip_pad*((dim+1) + 2*Nf);
  if (dim==2) IPData.z = IPData.dfz = NULL;
  else IPData.dfz = IPDataRaw + nip_pad*((dim+1) + 3*Nf);

  const PetscReal vj[3] = {IPData.x[jpidx], IPData.y[jpidx], IPData.z ? IPData.z[jpidx] : 0}, wj = IPData.w_data[jpidx];
  for (int ipidx_b = 0; ipidx_b < nip; ipidx_b += blockDim.x) {
    int ipidx = ipidx_b + threadIdx.x;

    __syncthreads();
    if (ipidx < nip) {
      for (fieldA = threadIdx.y; fieldA < Nf; fieldA += blockDim.y) {
        s_f  [fieldA*blockDim.x+threadIdx.x] = IPData.f  [ipidx + fieldA*nip_pad];
        s_dfx[fieldA*blockDim.x+threadIdx.x] = IPData.dfx[ipidx + fieldA*nip_pad];
        s_dfy[fieldA*blockDim.x+threadIdx.x] = IPData.dfy[ipidx + fieldA*nip_pad];
#if LANDAU_DIM==3
        s_dfz[fieldA*blockDim.x+threadIdx.x] = IPData.dfz[ipidx + fieldA*nip_pad];
#endif
      }
    }
    __syncthreads();
    if (ipidx < nip) {
      const PetscReal wi = IPData.w_data[ipidx], x = IPData.x[ipidx], y = IPData.y[ipidx];
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
  for (int delta = blockDim.x/2; delta > 0; delta /= 2) {
    for (d2 = 0; d2 < dim; d2++) {
      gg2_temp[d2] += __shfl_down_sync(0xffffffff, gg2_temp[d2], delta, blockDim.x);
      for (d3 = 0; d3 < dim; d3++) {
        gg3_temp[d2][d3] += __shfl_down_sync(0xffffffff, gg3_temp[d2][d3], delta, blockDim.x);
      }
    }
  }

  /* broadcast the reduction results to all threads */
  for (d2 = 0; d2 < dim; d2++) {
    gg2_temp[d2] = __shfl_sync(0xffffffff, gg2_temp[d2], 0, blockDim.x);
    for (d3 = 0; d3 < dim; d3++) {
      gg3_temp[d2][d3] = __shfl_sync(0xffffffff, gg3_temp[d2][d3], 0, blockDim.x);
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
  PetscInt  fieldA,d,f,qj,d2,g,totDim=Nb*Nf;
  /* assemble - on the diagonal (I,I) */
  for (fieldA = 0; fieldA < Nf; fieldA++) {
    for (f = threadIdx.y; f < Nb ; f += blockDim.y) {
      const PetscInt i = fieldA*Nb + f; /* Element matrix row */
      for (g = threadIdx.x; g < Nb; g += blockDim.x) {
        const PetscInt j    = fieldA*Nb + g; /* Element matrix column */
        const PetscInt fOff = i*totDim + j;
        PetscReal t = PetscRealPart(elemMat[fOff]);
        for (qj = 0 ; qj < Nq ; qj++) {
          const PetscReal *BJq = &BB[qj*Nb], *DIq = &DD[qj*Nb*dim];
          for (d = 0; d < dim; ++d) {
            t += DIq[f*dim+d]*g2[d][qj][fieldA]*BJq[g];
            for (d2 = 0; d2 < dim; ++d2) {
              t += DIq[f*dim + d]*g3[d][d2][qj][fieldA]*DIq[g*dim + d2];
            }
          }
        }
        elemMat[fOff] = t;
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
					       const PetscReal * const BB, const PetscReal * const DD, LandauIPReal *IPDataRaw, PetscScalar elemMats_out[])
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
  const PetscInt  myQi = threadIdx.y;
  const PetscInt  jpidx = myQi + myelem * Nq;
  //const PetscInt  subblocksz = nip/nSubBlks + !!(nip%nSubBlks), ip_start = mySubBlk*subblocksz, ip_end = (mySubBlk+1)*subblocksz > nip ? nip : (mySubBlk+1)*subblocksz; /* this could be wrong with very few global IPs */
  PetscScalar     *elemMat  = &elemMats_out[myelem*totDim*totDim]; /* my output */
  int tid = threadIdx.x + threadIdx.y*blockDim.x;
  for (int i = tid; i < totDim*totDim; i += blockDim.x*blockDim.y) elemMat[i] = 0;
  __syncthreads();

  landau_inner_integral_v2(myQi, jpidx, nip, Nq, Nf, Nb, dim, IPDataRaw, &invJj[jpidx*dim*dim], nu_alpha, nu_beta, invMass, Eq_m, BB, DD, elemMat, *g2, *g3,
    *gg2, *gg3, s_nu_alpha, s_nu_beta, s_invMass, s_f, s_dfx, s_dfy,
#if LANDAU_DIM==3
    s_dfz,
#endif
    myelem); /* compact */
}

PetscErrorCode LandauCUDAJacobian(DM plex, const PetscInt Nq, const PetscReal nu_alpha[],const PetscReal nu_beta[], const PetscReal invMass[], const PetscReal Eq_m[],
				  const LandauIPData *const IPData, const PetscReal invJj[], const PetscInt num_sub_blocks, const PetscLogEvent events[], Mat JacP)
{
  PetscErrorCode    ierr;
  PetscInt          ii,ej,*Nbf,Nb,nip_dim2,cStart,cEnd,Nf,dim,numGCells,totDim,nip,szf=sizeof(LandauIPReal),ipdatasz;
  PetscReal         *d_BB,*d_DD,*d_invJj,*d_nu_alpha,*d_nu_beta,*d_invMass,*d_Eq_m;
  PetscScalar       *elemMats,*d_elemMats;
  PetscLogDouble    flops;
  PetscTabulation   *Tf;
  PetscDS           prob;
  PetscSection      section, globalSection;
  LandauIPReal      *d_IPDataRaw;

  PetscFunctionBegin;
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
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_IPDataRaw,ipdatasz*szf )); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_nu_alpha, Nf*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_nu_beta,  Nf*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_invMass,  Nf*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_Eq_m,     Nf*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(d_IPDataRaw, IPData->w_data, ipdatasz*szf, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_nu_alpha, nu_alpha, Nf*szf,       cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_nu_beta,  nu_beta,  Nf*szf,       cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_invMass,  invMass,  Nf*szf,       cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_Eq_m,     Eq_m,     Nf*szf,       cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_BB,              Nq*Nb*szf));     // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          d_BB, Tf[0]->T[0], Nq*Nb*szf,   cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_DD,              Nq*Nb*dim*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          d_DD, Tf[0]->T[1], Nq*Nb*dim*szf,   cudaMemcpyHostToDevice));
  // collect geometry
  flops = (PetscLogDouble)numGCells*(PetscLogDouble)Nq*(PetscLogDouble)(5.*dim*dim*Nf*Nf + 165.);
  nip_dim2 = Nq*numGCells*dim*dim;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_invJj, nip_dim2*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(d_invJj, invJj, nip_dim2*szf,       cudaMemcpyHostToDevice));
  ierr = PetscLogEventEnd(events[3],0,0,0,0);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(events[4],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(flops*nip);CHKERRQ(ierr);
  {
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_elemMats, totDim*totDim*numGCells*sizeof(PetscScalar))); // kernel output
    {
      int n = 256/Nq;
      while (n & n - 1) n = n & n - 1;
      dim3 dimBlock(n,Nq);
      ii = 2*LANDAU_MAX_NQ*LANDAU_MAX_SPECIES*LANDAU_DIM*(1+LANDAU_DIM) +
        3*LANDAU_MAX_SPECIES + (1+LANDAU_DIM)*dimBlock.x*LANDAU_MAX_SPECIES;
      if (ii*szf >= 49152) {
        CUDA_SAFE_CALL(cudaFuncSetAttribute(landau_kernel_v2,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            98304));
      }
      // PetscPrintf(PETSC_COMM_SELF, "numGCells=%d dim.x=%d Nq=%d nThreads=%d, %d kB shared mem\n",numGCells,n,Nq,Nq*n,ii*szf/1024);
      landau_kernel_v2<<<numGCells,dimBlock,ii*szf>>>( nip,dim,totDim,Nf,Nb,d_invJj,d_nu_alpha,d_nu_beta,d_invMass,d_Eq_m,
						       d_BB, d_DD, d_IPDataRaw, d_elemMats);
      CHECK_LAUNCH_ERROR();
    }
  }
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
  ierr = PetscMalloc1(totDim*totDim*numGCells,&elemMats);CHKERRQ(ierr);
  CUDA_SAFE_CALL(cudaMemcpy(elemMats, d_elemMats, totDim*totDim*numGCells*sizeof(PetscScalar), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_elemMats));
  ierr = PetscLogEventEnd(events[5],0,0,0,0);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(events[6],0,0,0,0);CHKERRQ(ierr);
  {
    PetscScalar *elMat;
    for (ej = cStart, elMat = elemMats ; ej < cEnd; ++ej, elMat += totDim*totDim) {
      ierr = DMPlexMatSetClosure(plex, section, globalSection, JacP, ej, elMat, ADD_VALUES);CHKERRQ(ierr);
      if (ej==-1) {
	int d,f;
	PetscPrintf(PETSC_COMM_SELF,"GPU Element matrix\n");
	for (d = 0; d < totDim; ++d){
	  for (f = 0; f < totDim; ++f) PetscPrintf(PETSC_COMM_SELF," %12.5e", (double)PetscRealPart(elMat[d*totDim + f]));
	  PetscPrintf(PETSC_COMM_SELF,"\n");
	}
      }
    }
  }
  ierr = PetscFree(elemMats);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(events[6],0,0,0,0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
