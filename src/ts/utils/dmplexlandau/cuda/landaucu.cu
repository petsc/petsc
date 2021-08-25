/*
  Implements the Landau kernel
*/
#include <petscconf.h>
#include <petsc/private/dmpleximpl.h>   /*I  "dmpleximpl.h"   I*/
#include <petsclandau.h>
#define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1
#include <../src/mat/impls/aij/seq/aij.h>
#include <petscmat.h>
#include <petscdevice.h>

#include "../land_tensors.h"
#include <petscaijdevice.h>

#define CHECK_LAUNCH_ERROR()                                                             \
do {                                                                                     \
  /* Check synchronous errors, i.e. pre-launch */                                        \
  cudaError_t err = cudaGetLastError();                                                  \
  if (cudaSuccess != err) {                                                              \
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cuda error: %s",cudaGetErrorString(err)); \
  }                                                                                      \
  /* Check asynchronous errors, i.e. kernel failed (ULF) */                              \
  err = cudaDeviceSynchronize();                                                         \
  if (cudaSuccess != err) {                                                              \
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cuda error: %s",cudaGetErrorString(err)); \
  }                                                                                      \
 } while (0)

PETSC_EXTERN PetscErrorCode LandauCUDACreateMatMaps(P4estVertexMaps maps[], pointInterpolationP4est (*pointMaps)[LANDAU_MAX_Q_FACE],
                                                    PetscInt Nf[], PetscInt Nq, PetscInt grid)
{
  P4estVertexMaps h_maps;
  cudaError_t     cerr;
  PetscFunctionBegin;
  h_maps.num_elements = maps[grid].num_elements;
  h_maps.num_face = maps[grid].num_face;
  h_maps.num_reduced = maps[grid].num_reduced;
  h_maps.deviceType = maps[grid].deviceType;
  h_maps.Nf = Nf[grid];
  h_maps.numgrids = maps[grid].numgrids;
  h_maps.Nq = Nq;
  cerr = cudaMalloc((void **)&h_maps.c_maps,                    maps[grid].num_reduced  * sizeof *pointMaps);CHKERRCUDA(cerr);
  cerr = cudaMemcpy(          h_maps.c_maps, maps[grid].c_maps, maps[grid].num_reduced  * sizeof *pointMaps, cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
  cerr = cudaMalloc((void **)&h_maps.gIdx,                 maps[grid].num_elements * sizeof *maps[grid].gIdx);CHKERRCUDA(cerr);
  cerr = cudaMemcpy(          h_maps.gIdx, maps[grid].gIdx,maps[grid].num_elements * sizeof *maps[grid].gIdx, cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
  cerr = cudaMalloc((void **)&maps[grid].d_self,            sizeof(P4estVertexMaps));CHKERRCUDA(cerr);
  cerr = cudaMemcpy(          maps[grid].d_self,   &h_maps, sizeof(P4estVertexMaps), cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode LandauCUDADestroyMatMaps(P4estVertexMaps maps[], PetscInt num_grids)
{
  cudaError_t     cerr;
  PetscFunctionBegin;
  for (PetscInt grid=0;grid<num_grids;grid++) {
    P4estVertexMaps *d_maps = maps[grid].d_self, h_maps;
    cerr = cudaMemcpy(&h_maps, d_maps, sizeof(P4estVertexMaps), cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
    cerr = cudaFree(h_maps.c_maps);CHKERRCUDA(cerr);
    cerr = cudaFree(h_maps.gIdx);CHKERRCUDA(cerr);
    cerr = cudaFree(d_maps);CHKERRCUDA(cerr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode LandauCUDAStaticDataSet(DM plex, const PetscInt Nq, const PetscInt num_grids, PetscInt a_numCells[], PetscInt a_species_offset[], PetscInt a_mat_offset[],
                                       PetscReal nu_alpha[], PetscReal nu_beta[], PetscReal a_invMass[], PetscReal a_invJ[],
                                       PetscReal a_x[], PetscReal a_y[], PetscReal a_z[], PetscReal a_w[], LandauStaticData *SData_d)
{
  PetscErrorCode  ierr;
  PetscTabulation *Tf;
  PetscReal       *BB,*DD;
  PetscInt        dim,Nb=Nq,szf=sizeof(PetscReal),szs=sizeof(PetscScalar),szi=sizeof(PetscInt);
  PetscInt        h_ip_offset[LANDAU_MAX_GRIDS+1],h_ipf_offset[LANDAU_MAX_GRIDS+1],h_elem_offset[LANDAU_MAX_GRIDS+1],nip,IPfdf_sz,Nf;
  PetscDS         prob;
  cudaError_t     cerr;

  PetscFunctionBegin;
  ierr = DMGetDimension(plex, &dim);CHKERRQ(ierr);
  ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
  if (LANDAU_DIM != dim) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "dim %D != LANDAU_DIM %d",dim,LANDAU_DIM);
  ierr = PetscDSGetTabulation(prob, &Tf);CHKERRQ(ierr);
  BB   = Tf[0]->T[0]; DD = Tf[0]->T[1];
  Nf = h_ip_offset[0] = h_ipf_offset[0] = h_elem_offset[0] = 0;
  nip = 0;
  IPfdf_sz = 0;
  for (PetscInt grid=0 ; grid<num_grids ; grid++) {
    PetscInt nfloc = a_species_offset[grid+1] - a_species_offset[grid];
    h_elem_offset[grid+1] = h_elem_offset[grid] + a_numCells[grid];
    nip += a_numCells[grid]*Nq;
    h_ip_offset[grid+1] = nip;
    IPfdf_sz += Nq*nfloc*a_numCells[grid];
    h_ipf_offset[grid+1] = IPfdf_sz;
  }
  Nf = a_species_offset[num_grids];
  {
    cerr = cudaMalloc((void **)&SData_d->B,      Nq*Nb*szf);CHKERRCUDA(cerr);     // kernel input
    cerr = cudaMemcpy(          SData_d->B, BB,  Nq*Nb*szf,   cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMalloc((void **)&SData_d->D,      Nq*Nb*dim*szf);CHKERRCUDA(cerr); // kernel input
    cerr = cudaMemcpy(          SData_d->D, DD,  Nq*Nb*dim*szf,   cudaMemcpyHostToDevice);CHKERRCUDA(cerr);

    cerr = cudaMalloc((void **)&SData_d->alpha, Nf*szf);CHKERRCUDA(cerr); // kernel input
    cerr = cudaMalloc((void **)&SData_d->beta,  Nf*szf);CHKERRCUDA(cerr); // kernel input
    cerr = cudaMalloc((void **)&SData_d->invMass,  Nf*szf);CHKERRCUDA(cerr); // kernel input

    cerr = cudaMemcpy(SData_d->alpha,  nu_alpha, Nf*szf, cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(SData_d->beta,   nu_beta,  Nf*szf, cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(SData_d->invMass,a_invMass,Nf*szf, cudaMemcpyHostToDevice);CHKERRCUDA(cerr);

    // collect geometry
    cerr = cudaMalloc((void **)&SData_d->invJ,   nip*dim*dim*szf);CHKERRCUDA(cerr); // kernel input
    cerr = cudaMemcpy(SData_d->invJ,   a_invJ,   nip*dim*dim*szf, cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMalloc((void **)&SData_d->x,      nip*szf);CHKERRCUDA(cerr);     // kernel input
    cerr = cudaMemcpy(          SData_d->x, a_x, nip*szf,   cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMalloc((void **)&SData_d->y,      nip*szf);CHKERRCUDA(cerr); // kernel input
    cerr = cudaMemcpy(          SData_d->y, a_y, nip*szf,   cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
#if LANDAU_DIM==3
    cerr = cudaMalloc((void **)&SData_d->z,      nip*szf);CHKERRCUDA(cerr); // kernel input
    cerr = cudaMemcpy(          SData_d->z, a_z, nip*szf,   cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
#endif
    cerr = cudaMalloc((void **)&SData_d->w,      nip*szf);CHKERRCUDA(cerr); // kernel input
    cerr = cudaMemcpy(          SData_d->w, a_w, nip*szf,   cudaMemcpyHostToDevice);CHKERRCUDA(cerr);

    cerr = cudaMalloc((void **)&SData_d->NCells,              num_grids*szi);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(          SData_d->NCells, a_numCells,  num_grids*szi,   cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMalloc((void **)&SData_d->species_offset,                   (num_grids+1)*szi);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(          SData_d->species_offset, a_species_offset, (num_grids+1)*szi,   cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMalloc((void **)&SData_d->mat_offset,                      (num_grids+1)*szi);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(          SData_d->mat_offset, a_mat_offset,       (num_grids+1)*szi,   cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMalloc((void **)&SData_d->ip_offset,                       (num_grids+1)*szi);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(          SData_d->ip_offset, h_ip_offset,          (num_grids+1)*szi,   cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMalloc((void **)&SData_d->ipf_offset,                     (num_grids+1)*szi);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(          SData_d->ipf_offset, h_ipf_offset,     (num_grids+1)*szi,   cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMalloc((void **)&SData_d->elem_offset,                     (num_grids+1)*szi);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(          SData_d->elem_offset, h_elem_offset,     (num_grids+1)*szi,   cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    // allocate space for dynamic data once
    cerr = cudaMalloc((void **)&SData_d->Eq_m,       Nf*szf);CHKERRCUDA(cerr);
    cerr = cudaMalloc((void **)&SData_d->f,      nip*Nf*szs);CHKERRCUDA(cerr);
    cerr = cudaMalloc((void **)&SData_d->dfdx,   nip*Nf*szs);CHKERRCUDA(cerr);
    cerr = cudaMalloc((void **)&SData_d->dfdy,   nip*Nf*szs);CHKERRCUDA(cerr);
#if LANDAU_DIM==3
    cerr = cudaMalloc((void **)&SData_d->dfdz,   nip*Nf*szs);CHKERRCUDA(cerr);     // kernel input
#endif
    cerr = cudaMalloc((void**)&SData_d->maps, num_grids*sizeof(P4estVertexMaps*));CHKERRCUDA(cerr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode LandauCUDAStaticDataClear(LandauStaticData *SData_d)
{
  cudaError_t     cerr;

  PetscFunctionBegin;
  if (SData_d->alpha) {
    cerr = cudaFree(SData_d->alpha);CHKERRCUDA(cerr);
    SData_d->alpha = NULL;
    cerr = cudaFree(SData_d->beta);CHKERRCUDA(cerr);
    cerr = cudaFree(SData_d->invMass);CHKERRCUDA(cerr);
    cerr = cudaFree(SData_d->B);CHKERRCUDA(cerr);
    cerr = cudaFree(SData_d->D);CHKERRCUDA(cerr);
    cerr = cudaFree(SData_d->invJ);CHKERRCUDA(cerr);
#if LANDAU_DIM==3
    cerr = cudaFree(SData_d->z);CHKERRCUDA(cerr);
#endif
    cerr = cudaFree(SData_d->x);CHKERRCUDA(cerr);
    cerr = cudaFree(SData_d->y);CHKERRCUDA(cerr);
    cerr = cudaFree(SData_d->w);CHKERRCUDA(cerr);
    // dynamic data
    cerr = cudaFree(SData_d->Eq_m);CHKERRCUDA(cerr);
    cerr = cudaFree(SData_d->f);CHKERRCUDA(cerr);
    cerr = cudaFree(SData_d->dfdx);CHKERRCUDA(cerr);
    cerr = cudaFree(SData_d->dfdy);CHKERRCUDA(cerr);
#if LANDAU_DIM==3
    cerr = cudaFree(SData_d->dfdz);CHKERRCUDA(cerr);
#endif
    cerr = cudaFree(SData_d->NCells);CHKERRCUDA(cerr);
    cerr = cudaFree(SData_d->species_offset);CHKERRCUDA(cerr);
    cerr = cudaFree(SData_d->mat_offset);CHKERRCUDA(cerr);
    cerr = cudaFree(SData_d->ip_offset);CHKERRCUDA(cerr);
    cerr = cudaFree(SData_d->ipf_offset);CHKERRCUDA(cerr);
    cerr = cudaFree(SData_d->elem_offset);CHKERRCUDA(cerr);
    cerr = cudaFree(SData_d->maps);CHKERRCUDA(cerr);
  }
  PetscFunctionReturn(0);
}

// The GPU Landau kernel
//
__global__
void landau_form_fdf(const PetscInt dim, const PetscInt Nb, const PetscReal d_invJ[],
                     const PetscReal * const BB, const PetscReal * const DD, PetscScalar *d_vertex_f, P4estVertexMaps *d_maps[],
                     PetscReal d_f[], PetscReal d_dfdx[], PetscReal d_dfdy[],
#if LANDAU_DIM==3
                     PetscReal d_dfdz[],
#endif
                     const PetscInt d_numCells[], const PetscInt d_species_offset[], const PetscInt d_mat_offset[], const PetscInt d_ip_offset[], const PetscInt d_ipf_offset[], const PetscInt d_elem_offset[]) // output
{
  const PetscInt    Nq = blockDim.y, g_cell = blockIdx.x;
  const PetscInt    myQi = threadIdx.y;
  const PetscReal   *Bq = &BB[myQi*Nb], *Dq = &DD[myQi*Nb*dim];
  PetscInt          grid = 0, f,d,b,e,q;
  while (g_cell >= d_elem_offset[grid+1]) grid++; // yuck search for grid
  {
    const PetscInt         moffset = d_mat_offset[grid], nip_loc = d_numCells[grid]*Nq, Nfloc = d_species_offset[grid+1] - d_species_offset[grid], elem = g_cell -  d_elem_offset[grid];
    const PetscInt         IP_idx = d_ip_offset[grid], IPf_vertex_idx = d_ipf_offset[grid];
    const PetscInt         ipidx_g = myQi + elem*Nq, ipidx = IP_idx + ipidx_g;
    const PetscReal *const invJj = &d_invJ[ipidx*dim*dim];
    PetscReal              u_x[LANDAU_MAX_SPECIES][LANDAU_DIM];
    const PetscScalar      *coef;
    PetscScalar            coef_buff[LANDAU_MAX_SPECIES*LANDAU_MAX_NQ];
    if (!d_maps) {
      coef = &d_vertex_f[elem*Nb*Nfloc + IPf_vertex_idx]; // closure and IP indexing are the same
    } else {
      coef = coef_buff;
      for (f = 0; f < Nfloc ; ++f) {
        LandauIdx *const Idxs = &d_maps[grid]->gIdx[elem][f][0];
        for (b = 0; b < Nb; ++b) {
          PetscInt idx = Idxs[b];
          if (idx >= 0) {
            coef_buff[f*Nb+b] = d_vertex_f[idx+moffset];
          } else {
            idx = -idx - 1;
            coef_buff[f*Nb+b] = 0;
            for (q = 0; q < d_maps[grid]->num_face; q++) {
              PetscInt  id    = d_maps[grid]->c_maps[idx][q].gid;
              PetscReal scale = d_maps[grid]->c_maps[idx][q].scale;
              coef_buff[f*Nb+b] += scale*d_vertex_f[id+moffset];
            }
          }
        }
      }
    }

    /* get f and df */
    for (f = threadIdx.x; f < Nfloc; f += blockDim.x) {
      PetscReal      refSpaceDer[LANDAU_DIM];
      const PetscInt idx = IPf_vertex_idx + f*nip_loc + ipidx_g;
      d_f[idx] = 0.0;
      for (d = 0; d < LANDAU_DIM; ++d) refSpaceDer[d] = 0.0;
      for (b = 0; b < Nb; ++b) {
        const PetscInt    cidx = b;
        d_f[idx] += Bq[cidx]*PetscRealPart(coef[f*Nb+cidx]);
        for (d = 0; d < dim; ++d) refSpaceDer[d] += Dq[cidx*dim+d]*PetscRealPart(coef[f*Nb+cidx]);
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0, u_x[f][d] = 0.0; e < dim; ++e) {
          u_x[f][d] += invJj[e*dim+d]*refSpaceDer[e];
        }
      }
    }
    for (f = threadIdx.x; f < Nfloc ; f += blockDim.x) {
      const PetscInt idx = IPf_vertex_idx + f*nip_loc + ipidx_g;
      d_dfdx[idx] = u_x[f][0];
      d_dfdy[idx] = u_x[f][1];
#if LANDAU_DIM==3
      d_dfdz[idx] = u_x[f][2];
#endif
    }
  }
}

__device__ void
jac_kernel(const PetscInt myQi, const PetscInt jpidx, PetscInt nip_global, const PetscInt Nq, const PetscInt grid,
           const PetscInt dim,  const PetscReal xx[], const PetscReal yy[], const PetscReal ww[], const PetscReal invJj[],
           const PetscInt Nftot, const PetscReal nu_alpha[], const PetscReal nu_beta[], const PetscReal invMass[], const PetscReal Eq_m[],
           const PetscReal * const BB, const PetscReal * const DD, PetscScalar *elemMat, P4estVertexMaps *d_maps[], PetscSplitCSRDataStructure d_mat, // output
           PetscScalar s_fieldMats[][LANDAU_MAX_NQ], // all these arrays are in shared memory
           PetscReal s_scale[][LANDAU_MAX_Q_FACE],
           PetscInt  s_idx[][LANDAU_MAX_Q_FACE],
           PetscReal s_g2[][LANDAU_MAX_NQ][LANDAU_MAX_SPECIES],
           PetscReal s_g3[][LANDAU_DIM][LANDAU_MAX_NQ][LANDAU_MAX_SPECIES],
           PetscReal s_gg2[][LANDAU_MAX_NQ][LANDAU_MAX_SPECIES],
           PetscReal s_gg3[][LANDAU_DIM][LANDAU_MAX_NQ][LANDAU_MAX_SPECIES],
           PetscReal s_nu_alpha[],
           PetscReal s_nu_beta[],
           PetscReal s_invMass[],
           PetscReal s_f[],
           PetscReal s_dfx[],
           PetscReal s_dfy[],
           PetscReal d_f[], PetscReal d_dfdx[], PetscReal d_dfdy[], // global memory
#if LANDAU_DIM==3
           const PetscReal zz[], PetscReal s_dfz[], PetscReal d_dfdz[],
#endif
           const PetscInt d_numCells[], const PetscInt d_species_offset[], const PetscInt d_mat_offset[], const PetscInt d_ip_offset[], const PetscInt d_ipf_offset[], const PetscInt d_elem_offset[])
{
  int             delta,d,f,g,d2,dp,d3,fieldA,ipidx_b;
  PetscReal       gg2_temp[LANDAU_DIM], gg3_temp[LANDAU_DIM][LANDAU_DIM];
#if LANDAU_DIM==2
  const PetscReal vj[3] = {xx[jpidx], yy[jpidx]};
#else
  const PetscReal vj[3] = {xx[jpidx], yy[jpidx], zz[jpidx]};
#endif
  const PetscInt  moffset = d_mat_offset[grid], Nfloc = d_species_offset[grid+1]-d_species_offset[grid], g_cell = blockIdx.x, elem = g_cell - d_elem_offset[grid];
  const PetscInt  f_off = d_species_offset[grid], Nb=Nq;

  // create g2 & g3
  for (f=threadIdx.x; f<Nfloc; f+=blockDim.x) {
    for (d=0;d<dim;d++) { // clear accumulation data D & K
      s_gg2[d][myQi][f] = 0;
      for (d2=0;d2<dim;d2++) s_gg3[d][d2][myQi][f] = 0;
    }
  }
  for (d2 = 0; d2 < dim; d2++) {
    gg2_temp[d2] = 0;
    for (d3 = 0; d3 < dim; d3++) {
      gg3_temp[d2][d3] = 0;
    }
  }
  if (threadIdx.y == 0) {
    // copy species into shared memory
    for (fieldA = threadIdx.x; fieldA < Nftot; fieldA += blockDim.x) {
      s_nu_alpha[fieldA] = nu_alpha[fieldA];
      s_nu_beta[fieldA] = nu_beta[fieldA];
      s_invMass[fieldA] = invMass[fieldA];
    }
  }
  __syncthreads();
  // inner integral, collect gg2/3
  for (ipidx_b = 0; ipidx_b < nip_global; ipidx_b += blockDim.x) {
    const PetscInt ipidx = ipidx_b + threadIdx.x;
    PetscInt       f_off_r,grid_r,Nfloc_r,nip_loc_r,ipidx_g,fieldB,IPf_idx_r;
    __syncthreads();
    if (ipidx < nip_global) {
      grid_r = 0;
      while (ipidx >= d_ip_offset[grid_r+1]) grid_r++; // yuck search for grid
      f_off_r = d_species_offset[grid_r];
      ipidx_g = ipidx - d_ip_offset[grid_r];
      nip_loc_r = d_numCells[grid_r]*Nq;
      Nfloc_r = d_species_offset[grid_r+1] - d_species_offset[grid_r];
      IPf_idx_r = d_ipf_offset[grid_r];
      for (fieldB = threadIdx.y; fieldB < Nfloc_r ; fieldB += blockDim.y) {
        const PetscInt idx = IPf_idx_r + fieldB*nip_loc_r + ipidx_g;
        s_f  [fieldB*blockDim.x + threadIdx.x] =    d_f[idx]; // all vector threads get copy of data (Peng: why?)
        s_dfx[fieldB*blockDim.x + threadIdx.x] = d_dfdx[idx];
        s_dfy[fieldB*blockDim.x + threadIdx.x] = d_dfdy[idx];
#if LANDAU_DIM==3
        s_dfz[fieldB*blockDim.x + threadIdx.x] = d_dfdz[idx];
#endif
      }
    }
    __syncthreads();
    if (ipidx < nip_global) {
      const PetscReal wi = ww[ipidx], x = xx[ipidx], y = yy[ipidx];
      PetscReal       temp1[3] = {0, 0, 0}, temp2 = 0;
#if LANDAU_DIM==2
      PetscReal Ud[2][2], Uk[2][2], mask = (PetscAbs(vj[0]-x) < 100*PETSC_SQRT_MACHINE_EPSILON && PetscAbs(vj[1]-y) < 100*PETSC_SQRT_MACHINE_EPSILON) ? 0. : 1.;
      LandauTensor2D(vj, x, y, Ud, Uk, mask);
#else
      PetscReal U[3][3], z = zz[ipidx], mask = (PetscAbs(vj[0]-x) < 100*PETSC_SQRT_MACHINE_EPSILON && PetscAbs(vj[1]-y) < 100*PETSC_SQRT_MACHINE_EPSILON && PetscAbs(vj[2]-z) < 100*PETSC_SQRT_MACHINE_EPSILON) ? 0. : 1.;
      LandauTensor3D(vj, x, y, z, U, mask);
#endif
      for (fieldB = 0; fieldB < Nfloc_r; fieldB++) {
        temp1[0] += s_dfx[fieldB*blockDim.x + threadIdx.x]*s_nu_beta[fieldB + f_off_r]*s_invMass[fieldB + f_off_r];
        temp1[1] += s_dfy[fieldB*blockDim.x + threadIdx.x]*s_nu_beta[fieldB + f_off_r]*s_invMass[fieldB + f_off_r];
#if LANDAU_DIM==3
        temp1[2] += s_dfz[fieldB*blockDim.x + threadIdx.x]*s_nu_beta[fieldB + f_off_r]*s_invMass[fieldB + f_off_r];
#endif
        temp2    += s_f  [fieldB*blockDim.x + threadIdx.x]*s_nu_beta[fieldB + f_off_r];
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
  for (fieldA = threadIdx.x; fieldA < Nfloc; fieldA += blockDim.x) {
    for (d2 = 0; d2 < dim; d2++) {
      s_gg2[d2][myQi][fieldA] += gg2_temp[d2]*s_nu_alpha[fieldA+f_off];
      for (d3 = 0; d3 < dim; d3++) {
        s_gg3[d2][d3][myQi][fieldA] -= gg3_temp[d2][d3]*s_nu_alpha[fieldA+f_off]*s_invMass[fieldA+f_off];
      }
    }
  }
  __syncthreads();
  /* add electric field term once per IP */
  for (fieldA = threadIdx.x ; fieldA < Nfloc ; fieldA += blockDim.x) {
    s_gg2[dim-1][myQi][fieldA] += Eq_m[fieldA+f_off];
  }
  __syncthreads();
  /* Jacobian transform - g2 */
  for (fieldA = threadIdx.x ; fieldA < Nfloc ; fieldA += blockDim.x) {
    PetscReal wj = ww[jpidx];
    for (d = 0; d < dim; ++d) {
      s_g2[d][myQi][fieldA] = 0.0;
      for (d2 = 0; d2 < dim; ++d2) {
        s_g2[d][myQi][fieldA] += invJj[d*dim+d2]*s_gg2[d2][myQi][fieldA];
        s_g3[d][d2][myQi][fieldA] = 0.0;
        for (d3 = 0; d3 < dim; ++d3) {
          for (dp = 0; dp < dim; ++dp) {
            s_g3[d][d2][myQi][fieldA] += invJj[d*dim + d3]*s_gg3[d3][dp][myQi][fieldA]*invJj[d2*dim + dp];
          }
        }
        s_g3[d][d2][myQi][fieldA] *= wj;
      }
      s_g2[d][myQi][fieldA] *= wj;
    }
  }
  __syncthreads();  // Synchronize (ensure all the data is available) and sum IP matrices
  /* FE matrix construction */
  {
    int fieldA,d,qj,d2,q,idx,totDim=Nb*Nfloc;
    /* assemble */
    for (fieldA = 0; fieldA < Nfloc; fieldA++) {
      for (f = threadIdx.y; f < Nb ; f += blockDim.y) {
        for (g = threadIdx.x; g < Nb; g += blockDim.x) {
          PetscScalar t = 0;
          for (qj = 0 ; qj < Nq ; qj++) {
            const PetscReal *BJq = &BB[qj*Nb], *DIq = &DD[qj*Nb*dim];
            for (d = 0; d < dim; ++d) {
              t += DIq[f*dim+d]*s_g2[d][qj][fieldA]*BJq[g];
              for (d2 = 0; d2 < dim; ++d2) {
                t += DIq[f*dim + d]*s_g3[d][d2][qj][fieldA]*DIq[g*dim + d2];
              }
            }
          }
          if (elemMat) {
            const PetscInt fOff = (fieldA*Nb + f)*totDim + fieldA*Nb + g;
            elemMat[fOff] += t; // ????
          } else s_fieldMats[f][g] = t;
        }
      }
      if (s_fieldMats) {
        PetscScalar vals[LANDAU_MAX_Q_FACE*LANDAU_MAX_Q_FACE];
        PetscInt    nr,nc;
        const LandauIdx *const Idxs = &d_maps[grid]->gIdx[elem][fieldA][0];
        __syncthreads();
        if (threadIdx.y == 0) {
          for (f = threadIdx.x; f < Nb ; f += blockDim.x) {
            idx = Idxs[f];
            if (idx >= 0) {
              s_idx[f][0] = idx + moffset;
              s_scale[f][0] = 1.;
            } else {
              idx = -idx - 1;
              for (q = 0; q < d_maps[grid]->num_face; q++) {
                s_idx[f][q]   = d_maps[grid]->c_maps[idx][q].gid + moffset;
                s_scale[f][q] = d_maps[grid]->c_maps[idx][q].scale;
              }
            }
          }
        }
        __syncthreads();
        for (f = threadIdx.y; f < Nb ; f += blockDim.y) {
          idx = Idxs[f];
          if (idx >= 0) {
            nr = 1;
          } else {
            nr = d_maps[grid]->num_face;
          }
          for (g = threadIdx.x; g < Nb; g += blockDim.x) {
            idx = Idxs[g];
            if (idx >= 0) {
              nc = 1;
            } else {
              nc = d_maps[grid]->num_face;
            }
            for (q = 0; q < nr; q++) {
              for (d = 0; d < nc; d++) {
                vals[q*nc + d] = s_scale[f][q]*s_scale[g][d]*s_fieldMats[f][g];
              }
            }
            MatSetValuesDevice(d_mat,nr,s_idx[f],nc,s_idx[g],vals,ADD_VALUES);
          }
        }
        __syncthreads();
      }
    }
  }
}

//
// The CUDA Landau kernel
//
__global__
void __launch_bounds__(256,4)
  landau_jacobian(const PetscInt nip_global, const PetscInt dim, const PetscInt Nb, const PetscReal invJj[],
                  const PetscInt Nftot, const PetscReal nu_alpha[], const PetscReal nu_beta[], const PetscReal invMass[], const PetscReal Eq_m[],
                  const PetscReal * const BB, const PetscReal * const DD, const PetscReal xx[], const PetscReal yy[], const PetscReal ww[],
                  PetscScalar d_elem_mats[], P4estVertexMaps *d_maps[], PetscSplitCSRDataStructure d_mat, PetscReal d_f[], PetscReal d_dfdx[], PetscReal d_dfdy[],
#if LANDAU_DIM==3
                  const PetscReal zz[], PetscReal d_dfdz[],
#endif
                  const PetscInt d_numCells[], const PetscInt d_species_offset[], const PetscInt d_mat_offset[], const PetscInt d_ip_offset[], const PetscInt d_ipf_offset[], const PetscInt d_elem_offset[])
{
  extern __shared__ PetscReal smem[];
  int size = 0;
  PetscReal (*s_g2)[LANDAU_DIM][LANDAU_MAX_NQ][LANDAU_MAX_SPECIES]              =
    (PetscReal (*)[LANDAU_DIM][LANDAU_MAX_NQ][LANDAU_MAX_SPECIES])             &smem[size];
  size += LANDAU_MAX_NQ*LANDAU_MAX_SPECIES*LANDAU_DIM;
  PetscReal (*s_g3)[LANDAU_DIM][LANDAU_DIM][LANDAU_MAX_NQ][LANDAU_MAX_SPECIES]  =
    (PetscReal (*)[LANDAU_DIM][LANDAU_DIM][LANDAU_MAX_NQ][LANDAU_MAX_SPECIES]) &smem[size];
  size += LANDAU_DIM*LANDAU_DIM*LANDAU_MAX_NQ*LANDAU_MAX_SPECIES;
  PetscReal (*s_gg2)[LANDAU_DIM][LANDAU_MAX_NQ][LANDAU_MAX_SPECIES]             =
    (PetscReal (*)[LANDAU_DIM][LANDAU_MAX_NQ][LANDAU_MAX_SPECIES])             &smem[size];
  size += LANDAU_MAX_NQ*LANDAU_MAX_SPECIES*LANDAU_DIM;
  PetscReal (*s_gg3)[LANDAU_DIM][LANDAU_DIM][LANDAU_MAX_NQ][LANDAU_MAX_SPECIES] =
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
  PetscScalar (*s_fieldMats)[LANDAU_MAX_NQ][LANDAU_MAX_NQ];
  PetscReal   (*s_scale)[LANDAU_MAX_NQ][LANDAU_MAX_Q_FACE];
  PetscInt    (*s_idx)[LANDAU_MAX_NQ][LANDAU_MAX_Q_FACE];
  PetscInt    Nq = blockDim.y, g_cell = blockIdx.x, grid = 0;
  PetscScalar *elemMat = NULL; /* my output */

  while (g_cell >= d_elem_offset[grid+1]) grid++; // yuck search for grid
  {
    const PetscInt   Nfloc = d_species_offset[grid+1]-d_species_offset[grid], totDim = Nfloc*Nq, elem = g_cell - d_elem_offset[grid];
    const PetscInt   IP_idx = d_ip_offset[grid];
    const PetscInt   myQi = threadIdx.y;
    const PetscInt   jpidx = IP_idx + myQi + elem * Nq;
    const PetscReal  *invJ = &invJj[jpidx*dim*dim];
    if (d_elem_mats) {
      elemMat = d_elem_mats; // start a beginning
      for (PetscInt grid2=0 ; grid2<grid ; grid2++) {
        PetscInt Nfloc2 = d_species_offset[grid2+1] - d_species_offset[grid2], totDim2 = Nfloc2*Nb;
        elemMat += d_numCells[grid2]*totDim2*totDim2; // jump past grids, could be in an offset
      }
      elemMat += elem*totDim*totDim; // index into local matrix & zero out
      for (int i = threadIdx.x + threadIdx.y*blockDim.x; i < totDim*totDim; i += blockDim.x*blockDim.y) elemMat[i] = 0;
    }
    __syncthreads();
    if (d_maps) {
      // reuse the space for fieldMats
      s_fieldMats = (PetscScalar (*)[LANDAU_MAX_NQ][LANDAU_MAX_NQ]) &smem[size];
      size += LANDAU_MAX_NQ*LANDAU_MAX_NQ;
      s_scale =  (PetscReal (*)[LANDAU_MAX_NQ][LANDAU_MAX_Q_FACE]) &smem[size];
      size += LANDAU_MAX_NQ*LANDAU_MAX_Q_FACE;
      s_idx = (PetscInt (*)[LANDAU_MAX_NQ][LANDAU_MAX_Q_FACE]) &smem[size];
      size += LANDAU_MAX_NQ*LANDAU_MAX_Q_FACE; // this is too big, idx is an integer
    } else {
      s_fieldMats = NULL;
    }
    __syncthreads();
    jac_kernel(myQi, jpidx, nip_global, Nq, grid, dim, xx, yy, ww,
               invJ, Nftot, nu_alpha, nu_beta, invMass, Eq_m, BB, DD,
               elemMat, d_maps, d_mat,
               *s_fieldMats, *s_scale, *s_idx,
               *s_g2, *s_g3, *s_gg2, *s_gg3,
               s_nu_alpha, s_nu_beta, s_invMass,
               s_f, s_dfx, s_dfy, d_f, d_dfdx, d_dfdy,
#if LANDAU_DIM==3
               zz, s_dfz, d_dfdz,
#endif
               d_numCells, d_species_offset, d_mat_offset, d_ip_offset, d_ipf_offset, d_elem_offset);
  }
}

__global__
void __launch_bounds__(256,4) landau_mass(const PetscInt dim, const PetscInt Nb, const PetscReal d_w[], const PetscReal * const BB, const PetscReal * const DD,
                                          PetscScalar d_elem_mats[], P4estVertexMaps *d_maps[], PetscSplitCSRDataStructure d_mat, PetscReal shift,
                                          const PetscInt d_numCells[], const PetscInt d_species_offset[], const PetscInt d_mat_offset[], const PetscInt d_ip_offset[], const PetscInt d_ipf_offset[], const PetscInt d_elem_offset[])
{
  const PetscInt         Nq = blockDim.y, g_cell = blockIdx.x;
  __shared__ PetscScalar s_fieldMats[LANDAU_MAX_NQ][LANDAU_MAX_NQ];
  __shared__ PetscInt    s_idx[LANDAU_MAX_NQ][LANDAU_MAX_Q_FACE];
  __shared__ PetscReal   s_scale[LANDAU_MAX_NQ][LANDAU_MAX_Q_FACE];
  int                    tid = threadIdx.x + threadIdx.y*blockDim.x;
  PetscScalar            *elemMat = NULL; /* my output */
  int                    fieldA,d,qj,q,idx,f,g, grid = 0;

  while (g_cell >= d_elem_offset[grid+1]) grid++; // yuck search for grid
  {
    const PetscInt moffset = d_mat_offset[grid], Nfloc = d_species_offset[grid+1]-d_species_offset[grid], totDim = Nfloc*Nq, elem = g_cell-d_elem_offset[grid];
    const PetscInt IP_idx = d_ip_offset[grid];
    if (d_elem_mats) {
      elemMat = d_elem_mats; // start a beginning
      for (PetscInt grid2=0 ; grid2<grid ; grid2++) {
        PetscInt Nfloc2 = d_species_offset[grid2+1] - d_species_offset[grid2], totDim2 = Nfloc2*Nb;
        elemMat += d_numCells[grid2]*totDim2*totDim2; // jump past grids,could be in an offset
      }
      elemMat += elem*totDim*totDim;
      for (int i = tid; i < totDim*totDim; i += blockDim.x*blockDim.y) elemMat[i] = 0;
    }
    __syncthreads();
    /* FE mass matrix construction */
    for (fieldA = 0; fieldA < Nfloc; fieldA++) {
      PetscScalar            vals[LANDAU_MAX_Q_FACE*LANDAU_MAX_Q_FACE];
      PetscInt               nr,nc;
      for (f = threadIdx.y; f < Nb ; f += blockDim.y) {
        for (g = threadIdx.x; g < Nb; g += blockDim.x) {
          PetscScalar t = 0;
          for (qj = 0 ; qj < Nq ; qj++) {
            const PetscReal *BJq = &BB[qj*Nb];
            const PetscInt jpidx = IP_idx + qj + elem * Nq;
            if (dim==2) {
              t += BJq[f] * d_w[jpidx]*shift * BJq[g] * 2. * PETSC_PI;
            } else {
              t += BJq[f] * d_w[jpidx]*shift * BJq[g];
            }
          }
          if (elemMat) {
            const PetscInt fOff = (fieldA*Nb + f)*totDim + fieldA*Nb + g;
            elemMat[fOff] += t; // ????
          } else s_fieldMats[f][g] = t;
        }
      }
      if (!elemMat) {
        const LandauIdx *const Idxs = &d_maps[grid]->gIdx[elem][fieldA][0];
        __syncthreads();
        if (threadIdx.y == 0) {
          for (f = threadIdx.x; f < Nb ; f += blockDim.x) {
            idx = Idxs[f];
            if (idx >= 0) {
              s_idx[f][0] = idx + moffset;
              s_scale[f][0] = 1.;
            } else {
              idx = -idx - 1;
              for (q = 0; q < d_maps[grid]->num_face; q++) {
                s_idx[f][q]   = d_maps[grid]->c_maps[idx][q].gid + moffset;
                s_scale[f][q] = d_maps[grid]->c_maps[idx][q].scale;
              }
            }
          }
        }
        __syncthreads();
        for (f = threadIdx.y; f < Nb ; f += blockDim.y) {
          idx = Idxs[f];
          if (idx >= 0) {
            nr = 1;
          } else {
            nr = d_maps[grid]->num_face;
          }
          for (g = threadIdx.x; g < Nb; g += blockDim.x) {
            idx = Idxs[g];
            if (idx >= 0) {
              nc = 1;
            } else {
              nc = d_maps[grid]->num_face;
            }
            for (q = 0; q < nr; q++) {
              for (d = 0; d < nc; d++) {
                vals[q*nc + d] = s_scale[f][q]*s_scale[g][d]*s_fieldMats[f][g];
              }
            }
            MatSetValuesDevice(d_mat,nr,s_idx[f],nc,s_idx[g],vals,ADD_VALUES);
          }
        }
      }
      __syncthreads();
    }
  }
}

PetscErrorCode LandauCUDAJacobian(DM plex[], const PetscInt Nq, const PetscInt num_grids, const PetscInt a_numCells[], PetscReal a_Eq_m[], PetscScalar a_elem_closure[],
                                  const PetscInt N, const PetscScalar a_xarray[], const LandauStaticData *SData_d, const PetscInt num_sub_blocks, const PetscReal shift,
                                  const PetscLogEvent events[], const PetscInt a_mat_offset[], const PetscInt a_species_offset[], Mat subJ[], Mat JacP)
{
  PetscErrorCode    ierr;
  cudaError_t       cerr;
  PetscInt          Nb=Nq,dim,nip_global,num_cells_tot,elem_mat_size_tot;
  PetscInt          *d_numCells, *d_species_offset, *d_mat_offset, *d_ip_offset, *d_ipf_offset, *d_elem_offset;
  PetscInt          szf=sizeof(PetscReal),szs=sizeof(PetscScalar),Nftot=a_species_offset[num_grids];
  PetscReal         *d_BB=NULL,*d_DD=NULL,*d_invJj=NULL,*d_nu_alpha=NULL,*d_nu_beta=NULL,*d_invMass=NULL,*d_Eq_m=NULL,*d_x=NULL,*d_y=NULL,*d_w=NULL;
  PetscScalar       *d_elem_mats=NULL,*d_vertex_f=NULL;
  PetscReal         *d_f=NULL,*d_dfdx=NULL,*d_dfdy=NULL;
#if LANDAU_DIM==3
  PetscReal         *d_dfdz=NULL, *d_z = NULL;
#endif
  LandauCtx         *ctx;
  PetscSplitCSRDataStructure d_mat=NULL;
  P4estVertexMaps   **d_maps,*maps[LANDAU_MAX_GRIDS];
  int               nnn = 256/Nq; // machine dependent
  PetscContainer    container;
  PetscFunctionBegin;
  ierr = PetscLogEventBegin(events[3],0,0,0,0);CHKERRQ(ierr);
  while (nnn & nnn - 1) nnn = nnn & nnn - 1;
  if (nnn>4) nnn = 4; // 16 debug
  ierr = DMGetApplicationContext(plex[0], &ctx);CHKERRQ(ierr);
  if (!ctx) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  ierr = DMGetDimension(plex[0], &dim);CHKERRQ(ierr);
  if (dim!=LANDAU_DIM) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "LANDAU_DIM %D != dim %d",LANDAU_DIM,dim);
  if (ctx->gpu_assembly) {
    ierr = PetscObjectQuery((PetscObject) JacP, "assembly_maps", (PetscObject *) &container);CHKERRQ(ierr);
    if (container) { // not here first call
      static int init = 0; // hack. just do every time, or put in setup (but that is in base class code), or add init_maps flag
      if (!init++) {
        P4estVertexMaps   *h_maps=NULL;
        ierr = PetscContainerGetPointer(container, (void **) &h_maps);CHKERRQ(ierr);
        for (PetscInt grid=0 ; grid<num_grids ; grid++) {
          if (h_maps[grid].d_self) {
            maps[grid] = h_maps[grid].d_self;
          } else {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "GPU assembly but no metadata in container");
          }
        }
        cerr = cudaMemcpy(SData_d->maps, maps, num_grids*sizeof(P4estVertexMaps*), cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
      }
      d_maps = (P4estVertexMaps**)SData_d->maps;
      // this does the setup the first time called
      ierr = MatCUSPARSEGetDeviceMatWrite(JacP,&d_mat);CHKERRQ(ierr);
    } else {
      d_maps = NULL;
    }
  } else {
    container = NULL;
    d_maps = NULL;
  }
  ierr = PetscLogEventEnd(events[3],0,0,0,0);CHKERRQ(ierr);
  {
    PetscInt elem_mat_size = 0;
    nip_global = num_cells_tot = 0;
    for (PetscInt grid=0 ; grid<num_grids ; grid++) {
      PetscInt Nfloc = a_species_offset[grid+1] - a_species_offset[grid], totDim = Nfloc*Nb;
      nip_global += a_numCells[grid]*Nq;
      num_cells_tot += a_numCells[grid]; // is in d_elem_offset
      elem_mat_size += a_numCells[grid]*totDim*totDim; // could be in an offset
    }
    elem_mat_size_tot = d_maps ? 0 : elem_mat_size;
  }
  if (elem_mat_size_tot) {
    cerr = cudaMalloc((void **)&d_elem_mats, elem_mat_size_tot*szs);CHKERRCUDA(cerr); // kernel output - first call is on CPU
  } else d_elem_mats = NULL;
  // create data
  d_BB = (PetscReal*)SData_d->B;
  d_DD = (PetscReal*)SData_d->D;
  if (a_elem_closure || a_xarray) {  // form f and df
    ierr = PetscLogEventBegin(events[1],0,0,0,0);CHKERRQ(ierr);
    cerr = cudaMemcpy(SData_d->Eq_m, a_Eq_m,   Nftot*szf, cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    d_invJj = (PetscReal*)SData_d->invJ;
    d_nu_alpha = (PetscReal*)SData_d->alpha;
    d_nu_beta = (PetscReal*)SData_d->beta;
    d_invMass = (PetscReal*)SData_d->invMass;
    d_x = (PetscReal*)SData_d->x;
    d_y = (PetscReal*)SData_d->y;
    d_w = (PetscReal*)SData_d->w;
    d_Eq_m = (PetscReal*)SData_d->Eq_m;
    d_dfdx = (PetscReal*)SData_d->dfdx;
    d_dfdy = (PetscReal*)SData_d->dfdy;
#if LANDAU_DIM==3
    d_dfdz = (PetscReal*)SData_d->dfdz;
    d_z = (PetscReal*)SData_d->z;
#endif
    d_f    = (PetscReal*)SData_d->f;
    // get a d_vertex_f
    if (a_elem_closure) {
      PetscInt closure_sz = 0; // argh, don't have this on the host!!!
      for (PetscInt grid=0 ; grid<num_grids ; grid++) {
        PetscInt nfloc = a_species_offset[grid+1] - a_species_offset[grid];
        closure_sz += Nq*nfloc*a_numCells[grid];
      }
      cerr = cudaMalloc((void **)&d_vertex_f, closure_sz * sizeof(*a_elem_closure));CHKERRCUDA(cerr);
      cerr = cudaMemcpy(d_vertex_f, a_elem_closure, closure_sz*sizeof(*a_elem_closure), cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    } else {
      d_vertex_f = (PetscScalar*)a_xarray;
    }
    ierr = PetscLogEventEnd(events[1],0,0,0,0);CHKERRQ(ierr);
  } else {
    d_w = (PetscReal*)SData_d->w;
  }
  //
  d_numCells = (PetscInt*)SData_d->NCells; // redundant -- remove
  d_species_offset = (PetscInt*)SData_d->species_offset;
  d_mat_offset = (PetscInt*)SData_d->mat_offset;
  d_ip_offset = (PetscInt*)SData_d->ip_offset;
  d_ipf_offset = (PetscInt*)SData_d->ipf_offset;
  d_elem_offset = (PetscInt*)SData_d->elem_offset;
  if (a_elem_closure || a_xarray) {  // form f and df
    dim3 dimBlockFDF(nnn>Nftot ? Nftot : nnn, Nq), dimBlock((nip_global>nnn) ? nnn : nip_global , Nq);
    ierr = PetscLogEventBegin(events[8],0,0,0,0);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = PetscInfo2(plex[0], "Form F and dF/dx vectors: nip_global=%D num_grids=%D\n",nip_global,num_grids);CHKERRQ(ierr);
    landau_form_fdf<<<num_cells_tot,dimBlockFDF>>>(dim, Nb, d_invJj, d_BB, d_DD, d_vertex_f, d_maps, d_f, d_dfdx, d_dfdy,
#if LANDAU_DIM==3
                                                   d_dfdz,
#endif
                                                   d_numCells, d_species_offset, d_mat_offset, d_ip_offset, d_ipf_offset, d_elem_offset);
    CHECK_LAUNCH_ERROR();
    ierr = PetscLogGpuFlops(nip_global*(PetscLogDouble)(2*Nb*(1+dim)));CHKERRQ(ierr);
    if (a_elem_closure) {
      cerr = cudaFree(d_vertex_f);CHKERRCUDA(cerr);
      d_vertex_f = NULL;
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogEventEnd(events[8],0,0,0,0);CHKERRQ(ierr);
    // Jacobian
    ierr = PetscLogEventBegin(events[4],0,0,0,0);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(nip_global*(PetscLogDouble)(a_elem_closure ? (nip_global*(11*Nftot+ 4*dim*dim) + 6*Nftot*dim*dim*dim + 10*Nftot*dim*dim + 4*Nftot*dim + Nb*Nftot*Nb*Nq*dim*dim*5) : Nb*Nftot*Nb*Nq*4));CHKERRQ(ierr);
    PetscInt ii = 2*LANDAU_MAX_NQ*LANDAU_MAX_SPECIES*LANDAU_DIM*(1+LANDAU_DIM) + 3*LANDAU_MAX_SPECIES + (1+LANDAU_DIM)*dimBlock.x*LANDAU_MAX_SPECIES + LANDAU_MAX_NQ*LANDAU_MAX_NQ + 2*LANDAU_MAX_NQ*LANDAU_MAX_Q_FACE;
    if (ii*szf >= 49152) {
      cerr = cudaFuncSetAttribute(landau_jacobian,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  98304);CHKERRCUDA(cerr);
    }
    ierr = PetscInfo3(plex[0], "Jacobian shared memory size: %D bytes, d_elem_mats=%p d_maps=%p\n",ii,d_elem_mats,d_maps);CHKERRQ(ierr);
    landau_jacobian<<<num_cells_tot,dimBlock,ii*szf>>>(nip_global, dim, Nb, d_invJj, Nftot, d_nu_alpha, d_nu_beta, d_invMass, d_Eq_m,
                                                     d_BB, d_DD, d_x, d_y, d_w,
                                                     d_elem_mats, d_maps, d_mat, d_f, d_dfdx, d_dfdy,
#if LANDAU_DIM==3
                                                     d_z, d_dfdz,
#endif
                                                     d_numCells, d_species_offset, d_mat_offset, d_ip_offset, d_ipf_offset, d_elem_offset);
    CHECK_LAUNCH_ERROR(); // has sync
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogEventEnd(events[4],0,0,0,0);CHKERRQ(ierr);
  } else { // mass
    dim3 dimBlock(nnn,Nq);
    ierr = PetscInfo4(plex[0], "Mass d_maps = %p. Nq=%d, vector size %d num_cells_tot=%d\n",d_maps,Nq,nnn,num_cells_tot);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(events[4],0,0,0,0);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    landau_mass<<<num_cells_tot,dimBlock>>>(dim, Nb, d_w, d_BB, d_DD, d_elem_mats,
                                            d_maps, d_mat, shift, d_numCells, d_species_offset, d_mat_offset, d_ip_offset, d_ipf_offset, d_elem_offset);
    CHECK_LAUNCH_ERROR(); // has sync
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogEventEnd(events[4],0,0,0,0);CHKERRQ(ierr);
  }
  // First time assembly with or without GPU assembly
  if (d_elem_mats) {
    for (PetscInt grid=0, elem_mats_idx = 0 ; grid<num_grids ; grid++) {  // elem_mats_idx += totDim*totDim*a_numCells[grid];
      const PetscInt Nfloc = a_species_offset[grid+1]-a_species_offset[grid], totDim = Nfloc*Nq;
      PetscScalar   *elemMats = NULL, *elMat;
      PetscSection  section, globalSection;
      PetscInt      cStart,cEnd,ej;
      ierr = PetscLogEventBegin(events[5],0,0,0,0);CHKERRQ(ierr);
      ierr = DMPlexGetHeightStratum(plex[grid],0,&cStart,&cEnd);CHKERRQ(ierr);
      ierr = DMGetLocalSection(plex[grid], &section);CHKERRQ(ierr);
      ierr = DMGetGlobalSection(plex[grid], &globalSection);CHKERRQ(ierr);
      ierr = PetscMalloc1(totDim*totDim*a_numCells[grid],&elemMats);CHKERRQ(ierr);
      cerr = cudaMemcpy(elemMats, &d_elem_mats[elem_mats_idx], totDim*totDim*a_numCells[grid]*sizeof(*elemMats), cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
      ierr = PetscLogEventEnd(events[5],0,0,0,0);CHKERRQ(ierr);
      ierr = PetscLogEventBegin(events[6],0,0,0,0);CHKERRQ(ierr);
      for (ej = cStart, elMat = elemMats ; ej < cEnd; ++ej, elMat += totDim*totDim) {
        ierr = DMPlexMatSetClosure(plex[grid], section, globalSection, subJ[grid], ej, elMat, ADD_VALUES);CHKERRQ(ierr);
        if (ej==-1) {
          int d,f;
          PetscPrintf(PETSC_COMM_SELF,"GPU Element matrix\n");
          for (d = 0; d < totDim; ++d) {
            for (f = 0; f < totDim; ++f) PetscPrintf(PETSC_COMM_SELF," %12.5e",  PetscRealPart(elMat[d*totDim + f]));
            PetscPrintf(PETSC_COMM_SELF,"\n");
          }
          //exit(14);
        }
      }
      ierr = PetscFree(elemMats);CHKERRQ(ierr);
      elem_mats_idx += totDim*totDim*a_numCells[grid]; // this can be a stored offset?
      //
      if (!container) {   // move nest matrix to global JacP
        PetscInt          moffset = a_mat_offset[grid], nloc, nzl, colbuf[1024], row;
        const PetscInt    *cols;
        const PetscScalar *vals;
        Mat               B = subJ[grid];
        ierr = MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatGetSize(B, &nloc, NULL);CHKERRQ(ierr);
        if (nloc != a_mat_offset[grid+1] - moffset) SETERRQ2(PetscObjectComm((PetscObject) B), PETSC_ERR_PLIB, "nloc %D != mat_offset[grid+1] - moffset = %D",nloc, a_mat_offset[grid+1] - moffset);
        for (int i=0 ; i<nloc ; i++) {
          ierr = MatGetRow(B,i,&nzl,&cols,&vals);CHKERRQ(ierr);
          if (nzl>1024) SETERRQ1(PetscObjectComm((PetscObject) B), PETSC_ERR_PLIB, "Row too big: %D",nzl);
          for (int j=0; j<nzl; j++) colbuf[j] = cols[j] + moffset;
          row = i + moffset;
          ierr = MatSetValues(JacP,1,&row,nzl,colbuf,vals,ADD_VALUES);CHKERRQ(ierr);
          ierr = MatRestoreRow(B,i,&nzl,&cols,&vals);CHKERRQ(ierr);
        }
        ierr = MatDestroy(&subJ[grid]);CHKERRQ(ierr);
      } else exit(34);
      ierr = PetscLogEventEnd(events[6],0,0,0,0);CHKERRQ(ierr);
    } // grids
    cerr = cudaFree(d_elem_mats);CHKERRCUDA(cerr);
    if (ctx->gpu_assembly) {
      // transition to use of maps for VecGetClosure
      if (!(a_elem_closure || a_xarray)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "transition without Jacobian");
    }
  }

  PetscFunctionReturn(0);
}
