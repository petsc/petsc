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
#include "../land_kernel.h"

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

#define LANDAU_USE_SHARED_GPU_MEM
//j
// The GPU Landau kernel
//
__global__
void landau_kernel(const PetscInt nip, const PetscInt dim, const PetscInt totDim, const PetscInt Nf, const PetscInt Nb, const PetscReal invJj[],
		 const PetscReal nu_alpha[], const PetscReal nu_beta[], const PetscReal invMass[], const PetscReal Eq_m[],
		 const PetscReal * const BB, const PetscReal * const DD, const PetscReal * const IPDataGlobal, const PetscReal wiGlobal[],
#if !defined(LANDAU_USE_SHARED_GPU_MEM)
		 PetscReal *g2arr, PetscReal *g3arr,
#endif
		 PetscBool quarter3DDomain, PetscScalar elemMats_out[])
{
  const PetscInt  Nq = blockDim.x, myelem = blockIdx.x;
#if defined(LANDAU_USE_SHARED_GPU_MEM)
  extern __shared__ PetscReal g2_g3_qi[]; // Nq * { [NSubBlocks][Nf][dim] ; [NSubBlocks][Nf][dim][dim] }
  PetscReal       (*g2)[LANDAU_MAX_NQ][LANDAU_MAX_SUB_THREAD_BLOCKS][LANDAU_MAX_SPECIES][LANDAU_DIM]         = (PetscReal (*)[LANDAU_MAX_NQ][LANDAU_MAX_SUB_THREAD_BLOCKS][LANDAU_MAX_SPECIES][LANDAU_DIM])         &g2_g3_qi[0];
  PetscReal       (*g3)[LANDAU_MAX_NQ][LANDAU_MAX_SUB_THREAD_BLOCKS][LANDAU_MAX_SPECIES][LANDAU_DIM][LANDAU_DIM] = (PetscReal (*)[LANDAU_MAX_NQ][LANDAU_MAX_SUB_THREAD_BLOCKS][LANDAU_MAX_SPECIES][LANDAU_DIM][LANDAU_DIM]) &g2_g3_qi[LANDAU_MAX_SUB_THREAD_BLOCKS*LANDAU_MAX_NQ*LANDAU_MAX_SPECIES*LANDAU_DIM];
#else
  PetscReal       (*g2)[LANDAU_MAX_NQ][LANDAU_MAX_SUB_THREAD_BLOCKS][LANDAU_MAX_SPECIES][LANDAU_DIM]         = (PetscReal (*)[LANDAU_MAX_NQ][LANDAU_MAX_SUB_THREAD_BLOCKS][LANDAU_MAX_SPECIES][LANDAU_DIM])         &g2arr[myelem*LANDAU_MAX_SUB_THREAD_BLOCKS*LANDAU_MAX_NQ*LANDAU_MAX_SPECIES*LANDAU_DIM       ];
  PetscReal       (*g3)[LANDAU_MAX_NQ][LANDAU_MAX_SUB_THREAD_BLOCKS][LANDAU_MAX_SPECIES][LANDAU_DIM][LANDAU_DIM] = (PetscReal (*)[LANDAU_MAX_NQ][LANDAU_MAX_SUB_THREAD_BLOCKS][LANDAU_MAX_SPECIES][LANDAU_DIM][LANDAU_DIM]) &g3arr[myelem*LANDAU_MAX_SUB_THREAD_BLOCKS*LANDAU_MAX_NQ*LANDAU_MAX_SPECIES*LANDAU_DIM*LANDAU_DIM];
#endif
  const PetscInt  myQi = threadIdx.x, mySubBlk = threadIdx.y, nSubBlks = blockDim.y;
  const PetscInt  jpidx = myQi + myelem * Nq;
  const PetscInt  subblocksz = nip/nSubBlks + !!(nip%nSubBlks), ip_start = mySubBlk*subblocksz, ip_end = (mySubBlk+1)*subblocksz > nip ? nip : (mySubBlk+1)*subblocksz; /* this could be wrong with very few global IPs */
  PetscScalar     *elemMat  = &elemMats_out[myelem*totDim*totDim]; /* my output */

  if (threadIdx.x==0 && threadIdx.y==0) {
    memset(elemMat, 0, totDim*totDim*sizeof(PetscScalar));
  }
  __syncthreads();
  landau_inner_integral(myQi, Nq, mySubBlk, nSubBlks, ip_start, ip_end, 1,        jpidx, Nf, dim, IPDataGlobal, wiGlobal, &invJj[jpidx*dim*dim], nu_alpha, nu_beta, invMass, Eq_m, quarter3DDomain, Nq, Nb, 0, Nq, BB, DD, elemMat, *g2, *g3, myelem); /* compact */
  // landau_inner_integral(myQi, Nq, mySubBlk, nSubBlks, mySubBlk,    nip, nSubBlks, jpidx, Nf, dim, IPDataGlobal, wiGlobal, &invJj[jpidx*dim*dim], nu_alpha, nu_beta, invMass, Eq_m, quarter3DDomain, Nq, Nb, 0, Nq, BB, DD, elemMat, *g2, *g3, myelem); /* spread */
}
static PetscErrorCode LandauAssembleCuda(PetscInt cStart, PetscInt cEnd, PetscInt totDim, DM plex, PetscSection section, PetscSection globalSection, Mat JacP, PetscScalar elemMats[], PetscContainer container, const PetscLogEvent events[]);
__global__ void assemble_kernel(const PetscInt nidx_arr[], PetscInt *idx_arr[], PetscScalar *el_mats[], const ISColoringValue colors[], Mat_SeqAIJ mats[]);
PetscErrorCode LandauCUDAJacobian( DM plex, const PetscInt Nq, const PetscReal nu_alpha[],const PetscReal nu_beta[],
				 const PetscReal invMass[], const PetscReal Eq_m[], const PetscReal * const IPDataGlobal,
				 const PetscReal wiGlobal[], const PetscReal invJj[], const PetscInt num_sub_blocks, const PetscLogEvent events[], PetscBool quarter3DDomain,
				 Mat JacP)
{
  PetscErrorCode    ierr;
  PetscInt          ii,ej,*Nbf,Nb,nip_dim2,cStart,cEnd,Nf,dim,numGCells,totDim,nip,szf=sizeof(PetscReal);
  PetscReal         *d_BB,*d_DD,*d_invJj,*d_wiGlobal,*d_nu_alpha,*d_nu_beta,*d_invMass,*d_Eq_m;
  PetscScalar       *elemMats,*d_elemMats;
  PetscLogDouble    flops;
  PetscTabulation   *Tf;
  PetscDS           prob;
  PetscSection      section, globalSection;
  PetscReal        *d_IPDataGlobal;
  PetscBool         cuda_assemble = PETSC_FALSE;
  PetscFunctionBegin;

  ierr = PetscLogEventBegin(events[3],0,0,0,0);CHKERRQ(ierr);
  ierr = DMGetDimension(plex, &dim);CHKERRQ(ierr);
  if (dim!=LANDAU_DIM) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "LANDAU_DIM != dim");
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
  // create data
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_IPDataGlobal, nip*(dim + Nf*(dim+1))*szf )); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_nu_alpha, Nf*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_nu_beta,  Nf*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_invMass,  Nf*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_Eq_m,     Nf*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(d_IPDataGlobal, IPDataGlobal, nip*(dim + Nf*(dim+1))*szf, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_nu_alpha, nu_alpha, Nf*szf,                             cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_nu_beta,  nu_beta,  Nf*szf,                             cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_invMass,  invMass,  Nf*szf,                             cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_Eq_m,     Eq_m,     Nf*szf,                             cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_BB,              Nq*Nb*szf));     // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          d_BB, Tf[0]->T[0], Nq*Nb*szf,   cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_DD,              Nq*Nb*dim*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          d_DD, Tf[0]->T[1], Nq*Nb*dim*szf,   cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_wiGlobal,           Nq*numGCells*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          d_wiGlobal, wiGlobal, Nq*numGCells*szf,   cudaMemcpyHostToDevice));
  // collect geometry
  flops = (PetscLogDouble)numGCells*(PetscLogDouble)Nq*(PetscLogDouble)(5.*dim*dim*Nf*Nf + 165.);
  nip_dim2 = Nq*numGCells*dim*dim;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_invJj, nip_dim2*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(d_invJj, invJj, nip_dim2*szf,       cudaMemcpyHostToDevice));
  ierr = PetscLogEventEnd(events[3],0,0,0,0);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(events[4],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(flops*nip);CHKERRQ(ierr);
  {
    dim3 dimBlock(Nq,num_sub_blocks);
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_elemMats, totDim*totDim*numGCells*sizeof(PetscScalar))); // kernel output
    ii = LANDAU_MAX_NQ*LANDAU_MAX_SPECIES*LANDAU_DIM*(1+LANDAU_DIM)*LANDAU_MAX_SUB_THREAD_BLOCKS;
#if defined(LANDAU_USE_SHARED_GPU_MEM)
    // PetscPrintf(PETSC_COMM_SELF,"Call land_kernel with %D kB shared memory\n",ii*8/1024);
    landau_kernel<<<numGCells,dimBlock,ii*szf>>>( nip,dim,totDim,Nf,Nb,d_invJj,d_nu_alpha,d_nu_beta,d_invMass,d_Eq_m,
						d_BB, d_DD, d_IPDataGlobal, d_wiGlobal, quarter3DDomain, d_elemMats);
    CHECK_LAUNCH_ERROR();
#else
    PetscReal  *d_g2g3;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_g2g3, ii*szf*numGCells)); // kernel input
    PetscReal  *g2 = &d_g2g3[0];
    PetscReal  *g3 = &d_g2g3[LANDAU_MAX_SUB_THREAD_BLOCKS*LANDAU_MAX_NQ*LANDAU_MAX_SPECIES*LANDAU_DIM*numGCells];
    landau_kernel<<<numGCells,dimBlock>>>(nip,dim,totDim,Nf,Nb,d_invJj,d_nu_alpha,d_nu_beta,d_invMass,d_Eq_m,
					d_BB, d_DD, d_IPDataGlobal, d_wiGlobal, g2, g3, quarter3DDomain, d_elemMats);
    CHECK_LAUNCH_ERROR();
    CUDA_SAFE_CALL (cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaFree(d_g2g3));
#endif
  }
  ierr = PetscLogEventEnd(events[4],0,0,0,0);CHKERRQ(ierr);
  // delete device data
  ierr = PetscLogEventBegin(events[5],0,0,0,0);CHKERRQ(ierr);
  CUDA_SAFE_CALL(cudaFree(d_IPDataGlobal));
  CUDA_SAFE_CALL(cudaFree(d_invJj));
  CUDA_SAFE_CALL(cudaFree(d_wiGlobal));
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
  if (!cuda_assemble) {
    PetscScalar *elMat;
    for (ej = cStart, elMat = elemMats ; ej < cEnd; ++ej, elMat += totDim*totDim) {
      ierr = DMPlexMatSetClosure(plex, section, globalSection, JacP, ej, elMat, ADD_VALUES);CHKERRQ(ierr);
      if (ej==-1) {
	int d,f;
	printf("GPU Element matrix\n");
	for (d = 0; d < totDim; ++d){
	  for (f = 0; f < totDim; ++f) printf(" %17.10e",  PetscRealPart(elMat[d*totDim + f]));
	  printf("\n");
	}
	exit(12);
      }
    }
  } else {
    PetscContainer container = NULL;
    ierr = PetscObjectQuery((PetscObject)JacP,"coloring",(PetscObject*)&container);CHKERRQ(ierr);
    if (!container) {
      ierr = PetscLogEventBegin(events[8],0,0,0,0);CHKERRQ(ierr);
      ierr = LandauCreateColoring(JacP, plex, &container);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(events[8],0,0,0,0);CHKERRQ(ierr);
    }
    ierr = LandauAssembleCuda(cStart, cEnd, totDim, plex, section, globalSection, JacP, elemMats, container, events);CHKERRQ(ierr);
  }
  ierr = PetscFree(elemMats);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(events[6],0,0,0,0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

__global__
void assemble_kernel(const PetscInt nidx_arr[], PetscInt *idx_arr[], PetscScalar *el_mats[], const ISColoringValue colors[], Mat_SeqAIJ mats[])
{
  const PetscInt     myelem = (gridDim.x==1) ? threadIdx.x : blockIdx.x;
  Mat_SeqAIJ         a = mats[colors[myelem]]; /* copy to GPU */
  const PetscScalar *v = el_mats[myelem];
  const PetscInt    *in = idx_arr[myelem], *im = idx_arr[myelem], n = nidx_arr[myelem], m = nidx_arr[myelem];
  /* mat set values */
  PetscInt          *rp,k,low,high,t,row,nrow,i,col,l;
  PetscInt          *ai = a.i,*ailen = a.ilen;
  PetscInt          *aj = a.j,lastcol = -1;
  MatScalar         *ap=NULL,value=0.0,*aa = a.a;
  for (k=0; k<m; k++) { /* loop over added rows */
    row = im[k];
    if (row < 0) continue;
    rp   = aj + ai[row];
    ap = aa + ai[row];
    nrow = ailen[row];
    low  = 0;
    high = nrow;
    for (l=0; l<n; l++) { /* loop over added columns */
      /* if (in[l] < 0) { */
      /* 	printf("\t\tin[l] < 0 ?????\n"); */
      /* 	continue; */
      /* } */
      while (l<n && (value = v[l + k*n]) == 0.0) l++;
      if (l==n) break;
      col = in[l];
      if (col <= lastcol) low = 0;
      else high = nrow;
      lastcol = col;
      while (high-low > 5) {
        t = (low+high)/2;
        if (rp[t] > col) high = t;
        else low = t;
      }
      for (i=low; i<high; i++) {
        // if (rp[i] > col) break;
        if (rp[i] == col) {
	  ap[i] += value;
	  low = i + 1;
          goto noinsert;
        }
      }
      printf("\t\t\t ERROR in assemble_kernel\n");
    noinsert:;
    }
  }
}

static PetscErrorCode LandauAssembleCuda(PetscInt cStart, PetscInt cEnd, PetscInt totDim, DM plex, PetscSection section, PetscSection globalSection, Mat JacP, PetscScalar elemMats[], PetscContainer container, const PetscLogEvent events[])
{
  PetscErrorCode    ierr;
#define LANDAU_MAX_COLORS 16
#define LANDAU_MAX_ELEMS 512
  Mat_SeqAIJ             h_mats[LANDAU_MAX_COLORS], *jaca = (Mat_SeqAIJ *)JacP->data, *d_mats;
  const PetscInt         nelems = cEnd - cStart, nnz = jaca->i[JacP->rmap->n], N = JacP->rmap->n;  /* serial */
  const ISColoringValue *colors;
  ISColoringValue       *d_colors,colour;
  PetscInt              *h_idx_arr[LANDAU_MAX_ELEMS], h_nidx_arr[LANDAU_MAX_ELEMS], *d_nidx_arr, **d_idx_arr,nc,ej,j,cell;
  PetscScalar           *h_new_el_mats[LANDAU_MAX_ELEMS], *val_buf, **d_new_el_mats;
  ISColoring             iscoloring;
  ierr = PetscContainerGetPointer(container,(void**)&iscoloring);CHKERRQ(ierr);
  /* get colors */
  ierr = ISColoringGetColors(iscoloring, &j, &nc, &colors);CHKERRQ(ierr);
  if (nelems>LANDAU_MAX_ELEMS) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "too many elements. %D > %D",nelems,LANDAU_MAX_ELEMS);
  if (nc>LANDAU_MAX_COLORS) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "too many colors. %D > %D",nc,LANDAU_MAX_COLORS);
  /* colors for kernel */
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_colors,         nelems*sizeof(ISColoringValue))); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          d_colors, colors, nelems*sizeof(ISColoringValue), cudaMemcpyHostToDevice));
  /* get indices and element matrices */
  for (cell = cStart, ej = 0 ; cell < cEnd; ++cell, ++ej) {
    PetscInt numindices,*indices;
    PetscScalar *elMat = &elemMats[ej*totDim*totDim];
    PetscScalar *valuesOrig = elMat;
    ierr = DMPlexGetClosureIndices(plex, section, globalSection, cell, PETSC_TRUE, &numindices, &indices, NULL, (PetscScalar **) &elMat);CHKERRQ(ierr);
    h_nidx_arr[ej] = numindices;
    CUDA_SAFE_CALL(cudaMalloc((void **)&h_idx_arr[ej],            numindices*sizeof(PetscInt))); // kernel input
    CUDA_SAFE_CALL(cudaMemcpy(          h_idx_arr[ej],   indices, numindices*sizeof(PetscInt), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **)&h_new_el_mats[ej],        numindices*numindices*sizeof(PetscScalar))); // kernel input
    CUDA_SAFE_CALL(cudaMemcpy(          h_new_el_mats[ej], elMat, numindices*numindices*sizeof(PetscScalar), cudaMemcpyHostToDevice));
    ierr = DMPlexRestoreClosureIndices(plex, section, globalSection, cell, PETSC_TRUE, &numindices, &indices, NULL, (PetscScalar **) &elMat);CHKERRQ(ierr);
    if (elMat != valuesOrig) {ierr = DMRestoreWorkArray(plex, numindices*numindices, MPIU_SCALAR, &elMat);CHKERRQ(ierr);}
  }
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_nidx_arr,                  nelems*sizeof(PetscInt))); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          d_nidx_arr,    h_nidx_arr,   nelems*sizeof(PetscInt), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_idx_arr,                   nelems*sizeof(PetscInt*))); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          d_idx_arr,     h_idx_arr,    nelems*sizeof(PetscInt*), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_new_el_mats,               nelems*sizeof(PetscScalar*))); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          d_new_el_mats, h_new_el_mats,nelems*sizeof(PetscScalar*), cudaMemcpyHostToDevice));
  /* make matrix buffers */
  for (colour=0; colour<nc; colour++) {
    Mat_SeqAIJ *a = &h_mats[colour];
    /* create on GPU and copy to GPU */
    CUDA_SAFE_CALL(cudaMalloc((void **)&a->i,               (N+1)*sizeof(PetscInt))); // kernel input
    CUDA_SAFE_CALL(cudaMemcpy(          a->i,    jaca->i,   (N+1)*sizeof(PetscInt), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **)&a->ilen,            (N)*sizeof(PetscInt))); // kernel input
    CUDA_SAFE_CALL(cudaMemcpy(          a->ilen, jaca->ilen,(N)*sizeof(PetscInt), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **)&a->j,               (nnz)*sizeof(PetscInt))); // kernel input
    CUDA_SAFE_CALL(cudaMemcpy(          a->j,    jaca->j,   (nnz)*sizeof(PetscInt), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **)&a->a,               (nnz)*sizeof(PetscScalar))); // kernel output
    CUDA_SAFE_CALL(cudaMemset(          a->a, 0,            (nnz)*sizeof(PetscScalar)));
  }
  CUDA_SAFE_CALL(cudaMalloc(&d_mats,         nc*sizeof(Mat_SeqAIJ))); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy( d_mats, h_mats, nc*sizeof(Mat_SeqAIJ), cudaMemcpyHostToDevice));
  /* do it */
  assemble_kernel<<<nelems,1>>>(d_nidx_arr, d_idx_arr, d_new_el_mats, d_colors, d_mats);
  CHECK_LAUNCH_ERROR();
  /* cleanup */
  CUDA_SAFE_CALL(cudaFree(d_colors));
  CUDA_SAFE_CALL(cudaFree(d_nidx_arr));
  for (ej = cStart ; ej < nelems; ++ej) {
    CUDA_SAFE_CALL(cudaFree(h_idx_arr[ej]));
    CUDA_SAFE_CALL(cudaFree(h_new_el_mats[ej]));
  }
  CUDA_SAFE_CALL(cudaFree(d_idx_arr));
  CUDA_SAFE_CALL(cudaFree(d_new_el_mats));
  /* copy & add Mat data back to CPU to JacP */

  ierr = PetscLogEventBegin(events[2],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscMalloc1(nnz,&val_buf);CHKERRQ(ierr);
  ierr = PetscMemzero(jaca->a,nnz*sizeof(PetscScalar));CHKERRQ(ierr);
  for (colour=0; colour<nc; colour++) {
    Mat_SeqAIJ *a = &h_mats[colour];
    CUDA_SAFE_CALL(cudaMemcpy(val_buf, a->a, (nnz)*sizeof(PetscScalar), cudaMemcpyDeviceToHost));
    PetscKernelAXPY(jaca->a,1.0,val_buf,nnz);
  }
  ierr = PetscFree(val_buf);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(events[2],0,0,0,0);CHKERRQ(ierr);

  for (colour=0; colour<nc; colour++) {
    Mat_SeqAIJ *a = &h_mats[colour];
    /* destroy mat */
    CUDA_SAFE_CALL(cudaFree(a->i));
    CUDA_SAFE_CALL(cudaFree(a->ilen));
    CUDA_SAFE_CALL(cudaFree(a->j));
    CUDA_SAFE_CALL(cudaFree(a->a));
  }
  CUDA_SAFE_CALL(cudaFree(d_mats));
  PetscFunctionReturn(0);
}
