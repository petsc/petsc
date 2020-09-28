#include <petsc/private/dmpleximpl.h>   /*I "petscdmplex.h" I*/
#include <petsclandau.h>                /*I "petsclandau.h"   I*/
#include <petscts.h>
#include <petscdmforest.h>

/* Landau collision operator */
#define PETSC_THREAD_SYNC
#define PETSC_DEVICE_FUNC_DECL static
#include "land_kernel.h"

#define LANDAU_VL  1
static PetscErrorCode LandauPointDataCreate(PetscReal **IPData, PetscInt dim, PetscInt nip, PetscInt Ns)
{
  PetscErrorCode  ierr, d, s, jj, nip_pad = LANDAU_VL*(nip/LANDAU_VL + !!(nip%LANDAU_VL)), pnt_sz = (dim + Ns*(1+dim));
  PetscReal       *pdata;
  PetscFunctionBegin;
  ierr = PetscMalloc(nip_pad*pnt_sz*sizeof(PetscReal),IPData);CHKERRQ(ierr);
  /* pad with zeros in case we vectorize into this */
  for (jj=nip, pdata = *IPData + nip*pnt_sz; jj < nip_pad; jj++, pdata += pnt_sz){
    LandauPointData *fplpt = (LandauPointData*)pdata; /* [dim + NS*(1+dim)] */
    for (d=0;d<dim;d++) fplpt->crd[d] = -1;
    for (s=0;s<Ns;s++) {
      fplpt->fdf[s].f = 0;
      for (d=0;d<dim;d++) fplpt->fdf[s].df[d] = 0;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode LandauPointDataDestroy(PetscReal *IPData)
{
  PetscErrorCode   ierr;
  PetscFunctionBegin;
  ierr = PetscFree(IPData);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
/*
  LandauFormJacobian_Internal - Evaluates Jacobian matrix.

  Input Parameters:
  .  globX - input vector
  .  actx - optional user-defined context
  .  dim - dimension

  Output Parameters:
  .  J0acP - Jacobian matrix filled, not created
*/
PetscErrorCode LandauFormJacobian_Internal(Vec a_X, Mat JacP, const PetscInt dim, void *a_ctx)
{
  LandauCtx         *ctx = (LandauCtx*)a_ctx;
  PetscErrorCode    ierr;
  PetscInt          cStart, cEnd, elemMatSize;
  DM                plex = NULL;
  PetscDS           prob;
  PetscSection      section,globsection;
  PetscScalar       *elemMat;
  PetscInt          numCells,totDim,ej,Nq,*Nbf,*Ncf,Nb,Ncx,Nf,d,f,fieldA,Nip,nip_pad,ipdata_sz;
  PetscQuadrature   quad;
  PetscTabulation   *Tf;
  PetscReal         *wiGlob, nu_alpha[LANDAU_MAX_SPECIES], nu_beta[LANDAU_MAX_SPECIES];
  const PetscReal   *quadWeights;
  PetscReal         *IPData,*invJ,*invJ_a;
  PetscReal         invMass[LANDAU_MAX_SPECIES],Eq_m[LANDAU_MAX_SPECIES],m_0=ctx->m_0; /* normalize mass -- not needed! */
  PetscLogDouble    flops;
  Vec               locX;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(a_X,VEC_CLASSID,1);
  PetscValidHeaderSpecific(JacP,MAT_CLASSID,2);
  PetscValidPointer(ctx,4);

  ierr = PetscLogEventBegin(ctx->events[1],0,0,0,0);CHKERRQ(ierr);
  ierr = DMConvert(ctx->dmv, DMPLEX, &plex);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(plex, &locX);CHKERRQ(ierr);
  ierr = VecZeroEntries(locX);CHKERRQ(ierr); /* zero BCs so don't set */
  ierr = DMGlobalToLocalBegin(plex, a_X, INSERT_VALUES, locX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (plex, a_X, INSERT_VALUES, locX);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(plex, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMGetLocalSection(plex, &section);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(plex, &globsection);CHKERRQ(ierr);
  ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(prob, &Tf);CHKERRQ(ierr); // Bf, &Df
  ierr = PetscDSGetDimensions(prob, &Nbf);CHKERRQ(ierr); Nb = Nbf[0]; /* number of vertices*S */
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);         if (Nf!=ctx->num_species) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Nf %D != S",Nf);
  ierr = PetscDSGetComponents(prob, &Ncf);CHKERRQ(ierr); Ncx = Ncf[0]; if (Ncx!=1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Nc %D != 1",Ncx);
  for (fieldA=0;fieldA<Nf;fieldA++) {
    invMass[fieldA] = m_0/ctx->masses[fieldA];
    Eq_m[fieldA] = -ctx->Ez * ctx->t_0 * ctx->charges[fieldA] / (ctx->v_0 * ctx->masses[fieldA]); /* normalize dimensionless */
    if (dim==2) Eq_m[fieldA] *=  2 * PETSC_PI; /* add the 2pi term that is not in Landau */
    nu_alpha[fieldA] = PetscSqr(ctx->charges[fieldA]/m_0)*m_0/ctx->masses[fieldA];
    nu_beta[fieldA] = PetscSqr(ctx->charges[fieldA]/ctx->epsilon0)*ctx->lnLam / (8*PETSC_PI) * ctx->t_0*ctx->n_0/PetscPowReal(ctx->v_0,3);
  }
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  ierr = PetscFEGetQuadrature(ctx->fe[0], &quad);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, &quadWeights);CHKERRQ(ierr);
  if (Nb!=Nq) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Nb!=Nq %D %D over integration or simplices? Tf[0]->Nb=%D dim=%D",Nb,Nq,Tf[0]->Nb,dim);
  if (Nq >LANDAU_MAX_NQ) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Order too high. Nq = %D > LANDAU_MAX_NQ (%D)",Nq,LANDAU_MAX_NQ);
  Nip = numCells*Nq;
  nip_pad = LANDAU_VL*(Nip/LANDAU_VL + !!(Nip%LANDAU_VL));
  flops = (PetscLogDouble)numCells*(PetscLogDouble)Nq*(PetscLogDouble)(5*dim*dim*Nf*Nf + 165);
  ierr = MatZeroEntries(JacP);CHKERRQ(ierr);
  elemMatSize = totDim*totDim;
  {
    static int         cc = 0;
    PetscScalar        uu[LANDAU_MAX_SPECIES],u_x[LANDAU_MAX_SPECIES*LANDAU_DIM];
    /* collect f data */
    if (ctx->verbose > 2 || (ctx->verbose > 0 && cc++ == 0)) {
      PetscInt N,Nloc;
      ierr = MatGetSize(JacP,&N,NULL);CHKERRQ(ierr);
      ierr = VecGetSize(locX,&Nloc);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"[%D]%s: %D IPs, %D cells, %s elements, totDim=%D, Nb=%D, Nq=%D, elemMatSize=%D, dim=%D, Tab: Nb=%D Nf=%D Np=%D cdim=%D N=%D N_loc=%D\n",
                         0,"FormLandau",Nq*numCells,numCells,ctx->simplex ? "SIMPLEX" : "TENSOR", totDim, Nb, Nq, elemMatSize, dim, Tf[0]->Nb, Nf, Tf[0]->Np, Tf[0]->cdim, N, Nloc);CHKERRQ(ierr);
    }
    ierr = LandauPointDataCreate(&IPData, dim, Nq*numCells, Nf);CHKERRQ(ierr);
    ipdata_sz = (dim + Nf*(1+dim));
    ierr = PetscMalloc3(elemMatSize,&elemMat,nip_pad,&wiGlob,nip_pad*dim*dim,&invJ_a);CHKERRQ(ierr);
    /* cache geometry and x, f and df/dx at IPs */
    for (ej = 0, invJ = invJ_a ; ej < numCells; ++ej, invJ += Nq*dim*dim) {
      PetscReal    vj[LANDAU_MAX_NQ*LANDAU_DIM],detJj[LANDAU_MAX_NQ], Jdummy[LANDAU_MAX_NQ*LANDAU_DIM*LANDAU_DIM];
      PetscInt     qj,f;
      PetscScalar *coef = NULL;
      ierr = DMPlexComputeCellGeometryFEM(plex, cStart+ej, quad, vj, Jdummy, invJ, detJj);CHKERRQ(ierr);
      ierr = DMPlexVecGetClosure(plex, section, locX, cStart+ej, NULL, &coef);CHKERRQ(ierr);
      /* create point data for cell i for Landau tensor: x, f(x), grad f(x) */
      for (qj = 0; qj < Nq; ++qj) {
        PetscInt       gidx = (ej*Nq + qj);
        LandauPointData  *pnt_data = (LandauPointData*)(IPData + gidx*ipdata_sz);
        PetscScalar    refSpaceDer[LANDAU_DIM];
        PetscInt       dOffset = 0, fOffset = 0;
        for (d = 0; d < dim; ++d) pnt_data->crd[d] = vj[qj * dim + d]; /* coordinate */
        wiGlob[gidx] = detJj[qj] * quadWeights[qj];
        if (dim==2) wiGlob[gidx] *= pnt_data->crd[0];  /* cylindrical coordinate, w/o 2pi */
        /* get u & du (EvaluateFieldJets) */
        for (f = 0; f < Nf; ++f) {
          const PetscReal *Bq = &Tf[f]->T[0][qj*Nb];
          const PetscReal *Dq = &Tf[f]->T[1][qj*Nb*dim];
          PetscInt         b, e;
          uu[fOffset] = 0.0;
          for (d = 0; d < LANDAU_DIM; ++d) refSpaceDer[d] = 0.0;
          for (b = 0; b < Nb; ++b) {
            const PetscInt    cidx = b;
            uu[fOffset] += Bq[cidx]*coef[dOffset+cidx];
            for (d = 0; d < dim; ++d) refSpaceDer[d] += Dq[cidx*dim+d]*coef[dOffset+cidx];
          }
          for (d = 0; d < dim; ++d) {
            for (e = 0, u_x[fOffset*dim+d] = 0.0; e < dim; ++e) { // should add directly to point data here!!!
              u_x[fOffset*dim+d] += invJ[qj * dim * dim + e*dim+d]*refSpaceDer[e];
            }
          }
          fOffset += 1;
          dOffset += Nb;
        }
        /* copy to IPDataLocal */
        for (f=0;f<Nf;f++) {
          pnt_data->fdf[f].f = PetscRealPart(uu[f]);
          for (d = 0; d < dim; ++d) pnt_data->fdf[f].df[d] = PetscRealPart(u_x[f*dim+d]);
        }
      } /* q */
      ierr = DMPlexVecRestoreClosure(plex, section, locX, cStart+ej, NULL, &coef);CHKERRQ(ierr);
    } /* e */
  }
  ierr = DMRestoreLocalVector(plex, &locX);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ctx->events[1],0,0,0,0);CHKERRQ(ierr);

  /* outer element loop j is like a regular assembly loop */
  if (ctx->deviceType == LANDAU_CUDA) {
#if defined(PETSC_HAVE_CUDA)
    ierr = LandauCUDAJacobian(plex,Nq,nu_alpha,nu_beta,invMass,Eq_m,IPData,wiGlob,invJ_a,ctx->subThreadBlockSize,ctx->events,ctx->quarter3DDomain,JacP);CHKERRQ(ierr);
#else
    SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"-landau_device_type %s not built","cuda");
#endif
  } else if (ctx->deviceType == LANDAU_KOKKOS) {
#if defined(PETSC_HAVE_KOKKOS)
    ierr = LandauKokkosJacobian(plex,Nq,nu_alpha,nu_beta,invMass,Eq_m,IPData,wiGlob,invJ_a,ctx->subThreadBlockSize,ctx->events,ctx->quarter3DDomain,JacP);CHKERRQ(ierr);
#else
    SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"-landau_device_type %s not built","kokkos");
#endif
  } else { /* CPU version */
    for (ej = cStart, invJ = invJ_a; ej < cEnd; ++ej, invJ += Nq*dim*dim) {
      PetscInt     qj;
      ierr = PetscLogEventBegin(ctx->events[3],0,0,0,0);CHKERRQ(ierr);
      ierr = PetscMemzero(elemMat, totDim *totDim * sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = PetscLogEventEnd(ctx->events[3],0,0,0,0);CHKERRQ(ierr);
      for (qj = 0; qj < Nq; ++qj) {
        PetscReal       g2[1][LANDAU_MAX_SUB_THREAD_BLOCKS][LANDAU_MAX_SPECIES][LANDAU_DIM], g3[1][LANDAU_MAX_SUB_THREAD_BLOCKS][LANDAU_MAX_SPECIES][LANDAU_DIM][LANDAU_DIM];
        const PetscInt  nip = numCells*Nq, jpidx = Nq*(ej-cStart) + qj, one = 1, zero = 0; /* length of inner global interation, outer integration point */

        ierr = PetscLogEventBegin(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
        ierr = PetscLogFlops(flops);CHKERRQ(ierr);
        landau_inner_integral(zero, one, zero, one, zero, nip, 1, jpidx, Nf, dim, IPData, wiGlob, &invJ[qj*dim*dim], nu_alpha, nu_beta, invMass, Eq_m, ctx->quarter3DDomain, Nq, Nb, qj, qj+1, Tf[0]->T[0], Tf[0]->T[1], elemMat, g2, g3, ej);
        ierr = PetscLogEventEnd(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
      } /* qj loop */
      /* assemble matrix */
      ierr = PetscLogEventBegin(ctx->events[6],0,0,0,0);CHKERRQ(ierr);
      ierr = DMPlexMatSetClosure(plex, section, globsection, JacP, ej, elemMat, ADD_VALUES);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(ctx->events[6],0,0,0,0);CHKERRQ(ierr);

      if (ej==-1) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "CPU Element matrix\n");CHKERRQ(ierr);
        for (d = 0; d < totDim; ++d){
          for (f = 0; f < totDim; ++f) PetscPrintf(PETSC_COMM_SELF," %17.9e",  PetscRealPart(elemMat[d*totDim + f]));
           PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);
        }
      }
    } /* ej cells loop, not cuda */
  }
  // PetscSleep(2); exit(13);

  /* assemble matrix or vector */
  ierr = PetscLogEventBegin(ctx->events[7],0,0,0,0);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatScale(JacP, -1.0);CHKERRQ(ierr); /* The code reflect the papers: du/dt = C, whereas PETSc use the form G(u) = du/dt - C(u) = 0 */
  ierr = PetscLogEventEnd(ctx->events[7],0,0,0,0);CHKERRQ(ierr);

  /* clean up */
  ierr = PetscFree3(elemMat,wiGlob,invJ_a);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  ierr = LandauPointDataDestroy(IPData);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(LANDAU_ADD_BCS)
static void zero_bc(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar uexact[])
{
  uexact[0] = 0;
}
#endif

#define MATVEC2(__a,__x,__p) {int i,j; for (i=0.; i<2; i++) {__p[i] = 0; for (j=0.; j<2; j++) __p[i] += __a[i][j]*__x[j]; }}
static void CircleInflate(PetscReal r1, PetscReal r2, PetscReal r0, PetscInt num_sections, PetscReal x, PetscReal y,
                          PetscReal *outX, PetscReal *outY)
{
  PetscReal rr = PetscSqrtReal(x*x + y*y), outfact, efact;
  if (rr < r1 + PETSC_SQRT_MACHINE_EPSILON) {
    *outX = x; *outY = y;
  } else {
    const PetscReal xy[2] = {x,y}, sinphi=y/rr, cosphi=x/rr;
    PetscReal       cth,sth,xyprime[2],Rth[2][2],rotcos,newrr;
    if (num_sections==2) {
      rotcos = 0.70710678118654;
      outfact = 1.5; efact = 2.5;
      /* rotate normalized vector into [-pi/4,pi/4) */
      if (sinphi >= 0.) {         /* top cell, -pi/2 */
        cth = 0.707106781186548; sth = -0.707106781186548;
      } else {                    /* bottom cell -pi/8 */
        cth = 0.707106781186548; sth = .707106781186548;
      }
    } else if (num_sections==3) {
      rotcos = 0.86602540378443;
      outfact = 1.5; efact = 2.5;
      /* rotate normalized vector into [-pi/6,pi/6) */
      if (sinphi >= 0.5) {         /* top cell, -pi/3 */
        cth = 0.5; sth = -0.866025403784439;
      } else if (sinphi >= -.5) {  /* mid cell 0 */
        cth = 1.; sth = .0;
      } else { /* bottom cell +pi/3 */
        cth = 0.5; sth = 0.866025403784439;
      }
    } else if (num_sections==4) {
      rotcos = 0.9238795325112;
      outfact = 1.5; efact = 3;
      /* rotate normalized vector into [-pi/8,pi/8) */
      if (sinphi >= 0.707106781186548) {         /* top cell, -3pi/8 */
        cth = 0.38268343236509; sth = -0.923879532511287;
      } else if (sinphi >= 0.) {                 /* mid top cell -pi/8 */
        cth = 0.923879532511287; sth = -.38268343236509;
      } else if (sinphi >= -0.707106781186548) { /* mid bottom cell + pi/8 */
        cth = 0.923879532511287; sth = 0.38268343236509;
      } else {                                   /* bottom cell + 3pi/8 */
        cth = 0.38268343236509; sth = .923879532511287;
      }
    } else {
      cth = 0.; sth = 0.; rotcos = 0; efact = 0;
    }
    Rth[0][0] = cth; Rth[0][1] =-sth;
    Rth[1][0] = sth; Rth[1][1] = cth;
    MATVEC2(Rth,xy,xyprime);
    if (num_sections==2) {
      newrr = xyprime[0]/rotcos;
    } else {
      PetscReal newcosphi=xyprime[0]/rr, rin = r1, rout = rr - rin;
      PetscReal routmax = r0*rotcos/newcosphi - rin, nroutmax = r0 - rin, routfrac = rout/routmax;
      newrr = rin + routfrac*nroutmax;
    }
    *outX = cosphi*newrr; *outY = sinphi*newrr;
    /* grade */
    PetscReal fact,tt,rs,re, rr = PetscSqrtReal(PetscSqr(*outX) + PetscSqr(*outY));
    if (rr > r2) { rs = r2; re = r0; fact = outfact;} /* outer zone */
    else {         rs = r1; re = r2; fact = efact;} /* electron zone */
    tt = (rs + PetscPowReal((rr - rs)/(re - rs),fact) * (re-rs)) / rr;
    *outX *= tt;
    *outY *= tt;
  }
}

static PetscErrorCode GeometryDMLandau(DM base, PetscInt point, PetscInt dim, const PetscReal abc[], PetscReal xyz[], void *a_ctx)
{
  LandauCtx   *ctx = (LandauCtx*)a_ctx;
  PetscReal   r = abc[0], z = abc[1];
  if (ctx->inflate) {
    PetscReal absR, absZ;
    absR = PetscAbsReal(r);
    absZ = PetscAbsReal(z);
    CircleInflate(ctx->i_radius,ctx->e_radius,ctx->radius,ctx->num_sections,absR,absZ,&absR,&absZ);
    r = (r > 0) ? absR : -absR;
    z = (z > 0) ? absZ : -absZ;
  }
  xyz[0] = r;
  xyz[1] = z;
  if (dim==3) xyz[2] = abc[2];

  PetscFunctionReturn(0);
}

static PetscErrorCode ErrorIndicator_Simple(PetscInt dim, PetscReal volume, PetscReal x[], PetscInt Nc, const PetscInt Nf[], const PetscScalar u[], const PetscScalar u_x[], PetscReal *error, void *actx)
{
  PetscReal err = 0.0;
  PetscInt  f = *(PetscInt*)actx, j;
  PetscFunctionBegin;
  for (j = 0; j < dim; ++j) {
    err += PetscSqr(PetscRealPart(u_x[f*dim+j]));
  }
  err = PetscRealPart(u[f]); /* just use rho */
  *error = volume * err; /* * (ctx->axisymmetric ? 2.*PETSC_PI * r : 1); */
  PetscFunctionReturn(0);
}

static PetscErrorCode LandauDMCreateVMesh(MPI_Comm comm, const PetscInt dim, const char prefix[], LandauCtx *ctx, DM *dm)
{
  PetscErrorCode ierr;
  PetscReal      radius = ctx->radius;
  size_t         len;
  char           fname[128] = ""; /* we can add a file if we want */

  PetscFunctionBegin;
  /* create DM */
  ierr = PetscStrlen(fname, &len);CHKERRQ(ierr);
  if (len) {
    PetscInt dim2;
    ierr = DMPlexCreateFromFile(comm, fname, ctx->interpolate, dm);CHKERRQ(ierr);
    ierr = DMGetDimension(*dm, &dim2);CHKERRQ(ierr);
  } else {    /* p4est, quads */
    /* Create plex mesh of Landau domain */
    if (!ctx->sphere) {
      PetscInt       cells[] = {2,2,2};
      PetscReal      lo[] = {-radius,-radius,-radius}, hi[] = {radius,radius,radius};
      DMBoundaryType periodicity[3] = {DM_BOUNDARY_NONE, dim==2 ? DM_BOUNDARY_NONE : DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
      if (dim==2) { lo[0] = 0; cells[0] = 1; }
      else if (ctx->quarter3DDomain) { lo[0] = lo[1] = 0; cells[0] = cells[1] = 2; }
      ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, cells, lo, hi, periodicity, PETSC_TRUE, dm);CHKERRQ(ierr);
      ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr); /* needed for periodic */
      if (dim==3) {ierr = PetscObjectSetName((PetscObject) *dm, "cube");CHKERRQ(ierr);}
      else {ierr = PetscObjectSetName((PetscObject) *dm, "half-plane");CHKERRQ(ierr);}
    } else if (dim==2) {
      PetscInt       numCells,cells[16][4],i,j;
      PetscInt       numVerts;
      PetscReal      inner_radius1 = ctx->i_radius, inner_radius2 = ctx->e_radius;
      PetscReal      *flatCoords = NULL;
      PetscInt       *flatCells = NULL, *pcell;
      if (ctx->num_sections==2) {
#if 1
        numCells = 5;
        numVerts = 10;
        int cells2[][4] = { {0,1,4,3},
                            {1,2,5,4},
                            {3,4,7,6},
                            {4,5,8,7},
                            {6,7,8,9} };
        for (i = 0; i < numCells; i++) for (j = 0; j < 4; j++) cells[i][j] = cells2[i][j];
        ierr = PetscMalloc2(numVerts * 2, &flatCoords, numCells * 4, &flatCells);CHKERRQ(ierr);
        {
          PetscReal (*coords)[2] = (PetscReal (*) [2]) flatCoords;
          for (j = 0; j < numVerts-1; j++) {
            PetscReal z, r, theta = -PETSC_PI/2 + (j%3) * PETSC_PI/2;
            PetscReal rad = (j >= 6) ? inner_radius1 : (j >= 3) ? inner_radius2 : ctx->radius;
            z = rad * PetscSinReal(theta);
            coords[j][1] = z;
            r = rad * PetscCosReal(theta);
            coords[j][0] = r;
          }
          coords[numVerts-1][0] = coords[numVerts-1][1] = 0;
        }
#else
        numCells = 4;
        numVerts = 8;
        static int     cells2[][4] = {{0,1,2,3},
                                     {4,5,1,0},
                                     {5,6,2,1},
                                     {6,7,3,2}};
        for (i = 0; i < numCells; i++) for (j = 0; j < 4; j++) cells[i][j] = cells2[i][j];
        ierr = PetscMalloc2(numVerts * 2, &flatCoords, numCells * 4, &flatCells);CHKERRQ(ierr);
        {
          PetscReal (*coords)[2] = (PetscReal (*) [2]) flatCoords;
          PetscInt j;
          for (j = 0; j < 8; j++) {
            PetscReal z, r;
            PetscReal theta = -PETSC_PI/2 + (j%4) * PETSC_PI/3.;
            PetscReal rad = ctx->radius * ((j < 4) ? 0.5 : 1.0);
            z = rad * PetscSinReal(theta);
            coords[j][1] = z;
            r = rad * PetscCosReal(theta);
            coords[j][0] = r;
          }
        }
#endif
      } else if (ctx->num_sections==3) {
        numCells = 7;
        numVerts = 12;
        int cells2[][4] = { {0,1,5,4},
                            {1,2,6,5},
                            {2,3,7,6},
                            {4,5,9,8},
                            {5,6,10,9},
                            {6,7,11,10},
                            {8,9,10,11} };
        for (i = 0; i < numCells; i++) for (j = 0; j < 4; j++) cells[i][j] = cells2[i][j];
        ierr = PetscMalloc2(numVerts * 2, &flatCoords, numCells * 4, &flatCells);CHKERRQ(ierr);
        {
          PetscReal (*coords)[2] = (PetscReal (*) [2]) flatCoords;
          for (j = 0; j < numVerts; j++) {
            PetscReal z, r, theta = -PETSC_PI/2 + (j%4) * PETSC_PI/3;
            PetscReal rad = (j >= 8) ? inner_radius1 : (j >= 4) ? inner_radius2 : ctx->radius;
            z = rad * PetscSinReal(theta);
            coords[j][1] = z;
            r = rad * PetscCosReal(theta);
            coords[j][0] = r;
          }
        }
      } else if (ctx->num_sections==4) {
        numCells = 10;
        numVerts = 16;
        int cells2[][4] = { {0,1,6,5},
                            {1,2,7,6},
                            {2,3,8,7},
                            {3,4,9,8},
                            {5,6,11,10},
                            {6,7,12,11},
                            {7,8,13,12},
                            {8,9,14,13},
                            {10,11,12,15},
                            {12,13,14,15}};
        for (i = 0; i < numCells; i++) for (j = 0; j < 4; j++) cells[i][j] = cells2[i][j];
        ierr = PetscMalloc2(numVerts * 2, &flatCoords, numCells * 4, &flatCells);CHKERRQ(ierr);
        {
          PetscReal (*coords)[2] = (PetscReal (*) [2]) flatCoords;
          for (j = 0; j < numVerts-1; j++) {
            PetscReal z, r, theta = -PETSC_PI/2 + (j%5) * PETSC_PI/4;
            PetscReal rad = (j >= 10) ? inner_radius1 : (j >= 5) ? inner_radius2 : ctx->radius;
            z = rad * PetscSinReal(theta);
            coords[j][1] = z;
            r = rad * PetscCosReal(theta);
            coords[j][0] = r;
          }
          coords[numVerts-1][0] = coords[numVerts-1][1] = 0;
        }
      } else {
        numCells = 0;
        numVerts = 0;
      }
      for (j = 0, pcell = flatCells; j < numCells; j++, pcell += 4) {
        pcell[0] = cells[j][0]; pcell[1] = cells[j][1];
        pcell[2] = cells[j][2]; pcell[3] = cells[j][3];
      }
      ierr = DMPlexCreateFromCellListPetsc(comm,2,numCells,numVerts,4,ctx->interpolate,flatCells,2,flatCoords,dm);CHKERRQ(ierr);
      ierr = PetscFree2(flatCoords,flatCells);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) *dm, "semi-circle");CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Velocity space meshes does not support cubed sphere");
  }
  ierr = PetscObjectSetOptionsPrefix((PetscObject)*dm,prefix);CHKERRQ(ierr);

  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr); /* Plex refine */

  { /* p4est? */
    char      convType[256];
    PetscBool flg;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, prefix, "Mesh conversion options", "DMPLEX");CHKERRQ(ierr);
    ierr = PetscOptionsFList("-dm_landau_type","Convert DMPlex to another format (should not be Plex!)","ex6f.c",DMList,DMPLEX,convType,256,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();
    if (flg) {
      DM dmforest;
      ierr = DMConvert(*dm,convType,&dmforest);CHKERRQ(ierr);
      if (dmforest) {
        PetscBool isForest;
        ierr = PetscObjectSetOptionsPrefix((PetscObject)dmforest,prefix);CHKERRQ(ierr);
        ierr = DMIsForest(dmforest,&isForest);CHKERRQ(ierr);
        if (isForest) {
          if (ctx->sphere && ctx->inflate) {
            ierr = DMForestSetBaseCoordinateMapping(dmforest,GeometryDMLandau,ctx);CHKERRQ(ierr);
          }
          ierr = DMDestroy(dm);CHKERRQ(ierr);
          *dm = dmforest;
          ctx->errorIndicator = ErrorIndicator_Simple; /* flag for Forest */
        } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Converted to non Forest?");
      } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Convert failed?");
    }
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDS(DM dm, PetscInt dim, LandauCtx *ctx)
{
  PetscErrorCode  ierr;
  PetscInt        ii;
  PetscFunctionBegin;
  for (ii=0;ii<ctx->num_species;ii++) {
    char     buf[256];
    if (ii==0) ierr = PetscSNPrintf(buf, 256, "e");
    else {ierr = PetscSNPrintf(buf, 256, "i%D", ii);CHKERRQ(ierr);}
    /* Setup Discretization - FEM */
    ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, ctx->simplex, NULL, PETSC_DECIDE, &ctx->fe[ii]);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) ctx->fe[ii], buf);CHKERRQ(ierr);
    ierr = DMSetField(dm, ii, NULL, (PetscObject) ctx->fe[ii]);CHKERRQ(ierr);
  }
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  if (1) {
    PetscInt        ii;
    PetscSection    section;
    ierr = DMGetSection(dm, &section);CHKERRQ(ierr);
    for (ii=0;ii<ctx->num_species;ii++){
      char buf[256];
      if (ii==0) ierr = PetscSNPrintf(buf, 256, "se");
      else ierr = PetscSNPrintf(buf, 256, "si%D", ii);
      ierr = PetscSectionSetComponentName(section, ii, 0, buf);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* Define a Maxwellian function for testing out the operator. */

 /* Using cartesian velocity space coordinates, the particle */
 /* density, [1/m^3], is defined according to */

 /* $$ n=\int_{R^3} dv^3 \left(\frac{m}{2\pi T}\right)^{3/2}\exp [- mv^2/(2T)] $$ */

 /* Using some constant, c, we normalize the velocity vector into a */
 /* dimensionless variable according to v=c*x. Thus the density, $n$, becomes */

 /* $$ n=\int_{R^3} dx^3 \left(\frac{mc^2}{2\pi T}\right)^{3/2}\exp [- mc^2/(2T)*x^2] $$ */

 /* Defining $\theta=2T/mc^2$, we thus find that the probability density */
 /* for finding the particle within the interval in a box dx^3 around x is */

 /* f(x;\theta)=\left(\frac{1}{\pi\theta}\right)^{3/2} \exp [ -x^2/\theta ] */

typedef struct {
  LandauCtx   *ctx;
  PetscReal kT_m;
  PetscReal n;
  PetscReal shift;
} MaxwellianCtx;

static PetscErrorCode maxwellian(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf_dummy, PetscScalar *u, void *actx)
{
  MaxwellianCtx *mctx = (MaxwellianCtx*)actx;
  LandauCtx     *ctx = mctx->ctx;
  PetscInt      i;
  PetscReal     v2 = 0, theta = 2*mctx->kT_m/(ctx->v_0*ctx->v_0); /* theta = 2kT/mc^2 */
  PetscFunctionBegin;
  /* compute the exponents, v^2 */
  for (i = 0; i < dim; ++i) v2 += x[i]*x[i];
  /* evaluate the Maxwellian */
  u[0] = mctx->n*PetscPowReal(PETSC_PI*theta,-1.5)*(PetscExpReal(-v2/theta));
  if (mctx->shift!=0.) {
    v2 = 0;
    for (i = 0; i < dim-1; ++i) v2 += x[i]*x[i];
    v2 += (x[dim-1]-mctx->shift)*(x[dim-1]-mctx->shift);
    /* evaluate the shifted Maxwellian */
    u[0] += mctx->n*PetscPowReal(PETSC_PI*theta,-1.5)*(PetscExpReal(-v2/theta));
  }
  PetscFunctionReturn(0);
}

/*@
 LandauAddMaxwellians - Add a Maxwellian distribution to a state

 Collective on X

 Input Parameters:
 .   dm - The mesh
 +   time - Current time
 -   temps - Temperatures of each species
 .   ns - Number density of each species
 +   actx - Landau context

 Output Parameter:
 .   X  - The state

 Level: beginner

.keywords: mesh
.seealso: LandauCreateVelocitySpace()
@*/
PetscErrorCode LandauAddMaxwellians(DM dm, Vec X, PetscReal time, PetscReal temps[], PetscReal ns[], void *actx)
{
  LandauCtx      *ctx = (LandauCtx*)actx;
  PetscErrorCode (*initu[LANDAU_MAX_SPECIES])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar [], void *);
  PetscErrorCode ierr,ii;
  PetscInt       dim;
  MaxwellianCtx  *mctxs[LANDAU_MAX_SPECIES], data[LANDAU_MAX_SPECIES];

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (!ctx) { ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr); }
  for (ii=0;ii<ctx->num_species;ii++) {
    mctxs[ii] = &data[ii];
    data[ii].ctx = ctx;
    data[ii].kT_m = ctx->k*temps[ii]/ctx->masses[ii]; /* kT/m */
    data[ii].n = ns[ii];
    initu[ii] = maxwellian;
    data[ii].shift = 0;
  }
  data[0].shift = ctx->electronShift;
  /* need to make ADD_ALL_VALUES work - TODO */
  ierr = DMProjectFunction(dm, time, initu, (void**)mctxs, INSERT_ALL_VALUES, X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 LandauSetInitialCondition - Addes Maxwellians with context

Collective on X

 Input Parameters:
 .   dm - The mesh
 +   actx - Landau context with T and n

 Output Parameter:
 .   X  - The state

 Level: beginner

.keywords: mesh
.seealso: LandauCreateVelocitySpace(), LandauAddMaxwellians()
*/
static PetscErrorCode LandauSetInitialCondition(DM dm, Vec X, void *actx)
{
  LandauCtx        *ctx = (LandauCtx*)actx;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!ctx) { ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr); }
  ierr = VecZeroEntries(X);CHKERRQ(ierr);
  ierr = LandauAddMaxwellians(dm, X, 0.0, ctx->thermal_temps, ctx->n, ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode adaptToleranceFEM(PetscFE fem, Vec sol, PetscReal refineTol[], PetscReal coarsenTol[], PetscInt type, LandauCtx *ctx, DM *newDM)
{
  DM               dm, plex, adaptedDM = NULL;
  PetscDS          prob;
  PetscBool        isForest;
  PetscQuadrature  quad;
  PetscInt         Nq, *Nb, cStart, cEnd, c, dim, qj, k;
  DMLabel          adaptLabel = NULL;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(sol, &dm);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMIsForest(dm, &isForest);CHKERRQ(ierr);
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMLabelCreate(PETSC_COMM_SELF,"adapt",&adaptLabel);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
  if (Nq >LANDAU_MAX_NQ) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Order too high. Nq = %D > LANDAU_MAX_NQ (%D)",Nq,LANDAU_MAX_NQ);
  ierr = PetscDSGetDimensions(prob, &Nb);CHKERRQ(ierr);
  if (type==4) {
    for (c = cStart; c < cEnd; c++) {
      ierr = DMLabelSetValue(adaptLabel, c, DM_ADAPT_REFINE);CHKERRQ(ierr);
    }
    ierr = PetscInfo1(sol, "Phase:%s: Uniform refinement\n","adaptToleranceFEM");
  } else if (type==2) {
    PetscInt  rCellIdx[8], eCellIdx[64], iCellIdx[64], eMaxIdx = -1, iMaxIdx = -1, nr = 0, nrmax = (dim==3 && !ctx->quarter3DDomain) ? 8 : 2;
    PetscReal minRad = PETSC_INFINITY, r, eMinRad = PETSC_INFINITY, iMinRad = PETSC_INFINITY;
    for (c = 0; c < 64; c++) { eCellIdx[c] = iCellIdx[c] = -1; }
    for (c = cStart; c < cEnd; c++) {
      PetscReal    tt, v0[LANDAU_MAX_NQ*3], detJ[LANDAU_MAX_NQ];
      ierr = DMPlexComputeCellGeometryFEM(plex, c, quad, v0, NULL, NULL, detJ);CHKERRQ(ierr);
      for (qj = 0; qj < Nq; ++qj) {
        tt = PetscSqr(v0[dim*qj+0]) + PetscSqr(v0[dim*qj+1]) + PetscSqr(((dim==3) ? v0[dim*qj+2] : 0));
        r = PetscSqrtReal(tt);
        if (r < minRad - PETSC_SQRT_MACHINE_EPSILON*10.) {
          minRad = r;
          nr = 0;
          rCellIdx[nr++]= c;
          ierr = PetscInfo4(sol, "\t\tPhase: adaptToleranceFEM Found first inner r=%e, cell %D, qp %D/%D\n", r, c, qj+1, Nq);CHKERRQ(ierr);
        } else if ((r-minRad) < PETSC_SQRT_MACHINE_EPSILON*100. && nr < nrmax) {
          for (k=0;k<nr;k++) if (c == rCellIdx[k]) break;
          if (k==nr) {
            rCellIdx[nr++]= c;
            ierr = PetscInfo5(sol, "\t\t\tPhase: adaptToleranceFEM Found another inner r=%e, cell %D, qp %D/%D, d=%e\n", r, c, qj+1, Nq, r-minRad);CHKERRQ(ierr);
          }
        }
        if (ctx->sphere) {
          if ((tt=r-ctx->e_radius) > 0) {
            PetscInfo2(sol, "\t\t\t %D cell r=%g\n",c,tt);
            if (tt < eMinRad - PETSC_SQRT_MACHINE_EPSILON*100.) {
              eMinRad = tt;
              eMaxIdx = 0;
              eCellIdx[eMaxIdx++] = c;
            } else if (eMaxIdx > 0 && (tt-eMinRad) <= PETSC_SQRT_MACHINE_EPSILON && c != eCellIdx[eMaxIdx-1]) {
              eCellIdx[eMaxIdx++] = c;
            }
          }
          if ((tt=r-ctx->i_radius) > 0) {
            if (tt < iMinRad - 1.e-5) {
              iMinRad = tt;
              iMaxIdx = 0;
              iCellIdx[iMaxIdx++] = c;
            } else if (iMaxIdx > 0 && (tt-iMinRad) <= PETSC_SQRT_MACHINE_EPSILON && c != iCellIdx[iMaxIdx-1]) {
              iCellIdx[iMaxIdx++] = c;
            }
          }
        }
      }
    }
    for (k=0;k<nr;k++) {
      ierr = DMLabelSetValue(adaptLabel, rCellIdx[k], DM_ADAPT_REFINE);CHKERRQ(ierr);
    }
    if (ctx->sphere) {
      for (c = 0; c < eMaxIdx; c++) {
        ierr = DMLabelSetValue(adaptLabel, eCellIdx[c], DM_ADAPT_REFINE);CHKERRQ(ierr);
        ierr = PetscInfo3(sol, "\t\tPhase:%s: refine sphere e cell %D r=%g\n","adaptToleranceFEM",eCellIdx[c],eMinRad);
      }
      for (c = 0; c < iMaxIdx; c++) {
        ierr = DMLabelSetValue(adaptLabel, iCellIdx[c], DM_ADAPT_REFINE);CHKERRQ(ierr);
        ierr = PetscInfo3(sol, "\t\tPhase:%s: refine sphere i cell %D r=%g\n","adaptToleranceFEM",iCellIdx[c],iMinRad);
      }
    }
    ierr = PetscInfo4(sol, "Phase:%s: Adaptive refine origin cells %D,%D r=%g\n","adaptToleranceFEM",rCellIdx[0],rCellIdx[1],minRad);
  } else if (type==0 || type==1 || type==3) { /* refine along r=0 axis */
    PetscScalar  *coef = NULL;
    Vec          coords;
    PetscInt     csize,Nv,d,nz;
    DM           cdm;
    PetscSection cs;
    ierr = DMGetCoordinatesLocal(dm, &coords);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
    ierr = DMGetLocalSection(cdm, &cs);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; c++) {
      PetscInt doit = 0, outside = 0;
      ierr = DMPlexVecGetClosure(cdm, cs, coords, c, &csize, &coef);CHKERRQ(ierr);
      Nv = csize/dim;
      for (nz = d = 0; d < Nv; d++) {
        PetscReal z = PetscRealPart(coef[d*dim + (dim-1)]), x = PetscSqr(PetscRealPart(coef[d*dim + 0])) + ((dim==3) ? PetscSqr(PetscRealPart(coef[d*dim + 1])) : 0);
        x = PetscSqrtReal(x);
        if (x < PETSC_MACHINE_EPSILON*10. && PetscAbsReal(z)<PETSC_MACHINE_EPSILON*10.) doit = 1;             /* refine origin */
        else if (type==0 && (z < -PETSC_MACHINE_EPSILON*10. || z > ctx->re_radius+PETSC_MACHINE_EPSILON*10.)) outside++;   /* first pass don't refine bottom */
        else if (type==1 && (z > ctx->vperp0_radius1 || z < -ctx->vperp0_radius1)) outside++; /* don't refine outside electron refine radius */
        else if (type==3 && (z > ctx->vperp0_radius2 || z < -ctx->vperp0_radius2)) outside++; /* don't refine outside ion refine radius */
        if (x < PETSC_MACHINE_EPSILON*10.) nz++;
      }
      ierr = DMPlexVecRestoreClosure(cdm, cs, coords, c, &csize, &coef);CHKERRQ(ierr);
      if (doit || (outside<Nv && nz)) {
        ierr = DMLabelSetValue(adaptLabel, c, DM_ADAPT_REFINE);CHKERRQ(ierr);
      }
    }
    ierr = PetscInfo1(sol, "Phase:%s: RE refinement\n","adaptToleranceFEM");
  }
  /* ierr = VecDestroy(&locX);CHKERRQ(ierr); */
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  ierr = DMAdaptLabel(dm, adaptLabel, &adaptedDM);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&adaptLabel);CHKERRQ(ierr);
  *newDM = adaptedDM;
  if (adaptedDM) {
    if (isForest) {
      ierr = DMForestSetAdaptivityForest(adaptedDM,NULL);CHKERRQ(ierr);
    }
    ierr = DMConvert(adaptedDM, DMPLEX, &plex);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
    ierr = PetscInfo2(sol, "\tPhase: adaptToleranceFEM: %D cells, %d total quadrature points\n",cEnd-cStart,Nq*(cEnd-cStart));CHKERRQ(ierr);
    ierr = DMDestroy(&plex);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode adapt(DM *dm, LandauCtx *ctx, Vec *uu)
{
  PetscErrorCode  ierr;
  PetscInt        type, limits[5] = {ctx->numRERefine,ctx->nZRefine1,ctx->maxRefIts,ctx->nZRefine2,ctx->postAMRRefine};
  PetscInt        adaptIter;

  PetscFunctionBegin;
  for (type=0;type<5;type++) {
    for (adaptIter = 0; adaptIter<limits[type];adaptIter++) {
      DM  dmNew = NULL;
      ierr = adaptToleranceFEM(ctx->fe[0], *uu, ctx->refineTol, ctx->coarsenTol, type, ctx, &dmNew);CHKERRQ(ierr);
      if (!dmNew) {
        exit(113);
        break;
      } else {
        ierr = DMDestroy(dm);CHKERRQ(ierr);
        ierr = VecDestroy(uu);CHKERRQ(ierr);
        ierr = DMCreateGlobalVector(dmNew,uu);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject) *uu, "u");CHKERRQ(ierr);
        ierr = LandauSetInitialCondition(dmNew, *uu, ctx);CHKERRQ(ierr);
        *dm = dmNew;
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ProcessOptions(LandauCtx *ctx, const char prefix[])
{
  PetscErrorCode    ierr;
  PetscBool         flg, sph_flg;
  PetscInt          ii,nt,nm,nc;
  DM                dummy;

  PetscFunctionBegin;
  ierr = DMCreate(PETSC_COMM_WORLD,&dummy);CHKERRQ(ierr);
  /* get options - initialize context */
  ctx->normJ = 0;
  ctx->verbose = 1;
  ctx->interpolate = PETSC_TRUE;
  ctx->simplex = PETSC_FALSE;
  ctx->sphere = PETSC_FALSE;
  ctx->inflate = PETSC_FALSE;
  ctx->electronShift = 0;
  ctx->errorIndicator = NULL;
  ctx->radius = 5.; /* electron thermal radius (velocity) */
  ctx->re_radius = 0.;
  ctx->vperp0_radius1 = 0;
  ctx->vperp0_radius2 = 0;
  ctx->e_radius = .1;
  ctx->i_radius = .01;
  ctx->maxRefIts = 5;
  ctx->postAMRRefine = 0;
  ctx->nZRefine1 = 0;
  ctx->nZRefine2 = 0;
  ctx->numRERefine = 0;
  ctx->num_sections = 3; /* 2, 3 or 4 */
  /* species - [0] electrons, [1] one ion species eg, duetarium, [2] heavy impurity ion, ... */
  ctx->charges[0] = -1;  /* electron charge (MKS) */
  ctx->masses[0] = 1/1835.5; /* temporary value in proton mass */
  ctx->n[0] = 1;
  ctx->thermal_temps[0] = 1;
  /* constants, etc. */
  ctx->epsilon0 = 8.8542e-12; /* permittivity of free space (MKS) F/m */
  ctx->k = 1.38064852e-23; /* Boltzmann constant (MKS) J/K */
  ctx->lnLam = 10;         /* cross section ratio large - small angle collisions */
  ctx->n_0 = 1.e20;        /* typical plasma n, but could set it to 1 */
  ctx->Ez = 0;
  ctx->v_0 = 1; /* in electron thermal velocity */
  ctx->subThreadBlockSize = 1; /* for device and maybe OMP */
  ctx->quarter3DDomain = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, prefix, "Options for Fokker-Plank-Landau collision operator", "none");CHKERRQ(ierr);
  {
    char opstring[256];
#if defined(PETSC_HAVE_KOKKOS)
    ctx->deviceType = LANDAU_KOKKOS;
    ierr = PetscStrcpy(opstring,"kokkos");CHKERRQ(ierr);
#if !defined(PETSC_HAVE_CUDA)
    ctx->subThreadBlockSize = 0;
#endif
#elif defined(PETSC_HAVE_CUDA)
    ctx->deviceType = LANDAU_CUDA;
    ierr = PetscStrcpy(opstring,"cuda");CHKERRQ(ierr);
#else
    ctx->deviceType = LANDAU_CPU;
    ierr = PetscStrcpy(opstring,"cpu");CHKERRQ(ierr);
    ctx->subThreadBlockSize = 0;
#endif
    ierr = PetscOptionsString("-dm_landau_device_type","Use kernels on 'cpu', 'cuda', or 'kokkos'","plexland.c",opstring,opstring,256,NULL);CHKERRQ(ierr);
    ierr = PetscStrcmp("cpu",opstring,&flg);CHKERRQ(ierr);
    if (flg) {
      ctx->deviceType = LANDAU_CPU;
      ctx->subThreadBlockSize = 0;
    } else {
      ierr = PetscStrcmp("cuda",opstring,&flg);CHKERRQ(ierr);
      if (flg) ctx->deviceType = LANDAU_CUDA;
      else {
        ierr = PetscStrcmp("kokkos",opstring,&flg);CHKERRQ(ierr);
        if (flg) ctx->deviceType = LANDAU_KOKKOS;
        else SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"-dm_landau_device_type %s",opstring);
      }
    }
  }
  ierr = PetscOptionsReal("-dm_landau_electron_shift","Shift in thermal velocity of electrons","none",ctx->electronShift,&ctx->electronShift, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_landau_sphere", "use sphere/semi-circle domain instead of rectangle", "plexland.c", ctx->sphere, &ctx->sphere, &sph_flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_landau_inflate", "With sphere, inflate for curved edges (no AMR)", "plexland.c", ctx->inflate, &ctx->inflate, NULL);CHKERRQ(ierr);
  /* ierr = PetscOptionsBool("-dm_landau_quarter_3d_domain", "Use symmetry in 3D to model 1/4 of domain", "plexland.c", ctx->quarter3DDomain, &ctx->quarter3DDomain, NULL);CHKERRQ(ierr); */
  ierr = PetscOptionsInt("-dm_landau_amr_re_levels", "Number of levels to refine along v_perp=0, z>0", "plexland.c", ctx->numRERefine, &ctx->numRERefine, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dm_landau_amr_z_refine1",  "Number of levels to refine along v_perp=0", "plexland.c", ctx->nZRefine1, &ctx->nZRefine1, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dm_landau_amr_z_refine2",  "Number of levels to refine along v_perp=0", "plexland.c", ctx->nZRefine2, &ctx->nZRefine2, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dm_landau_amr_levels_max", "Number of AMR levels of refinement around origin after r=0 refinements", "plexland.c", ctx->maxRefIts, &ctx->maxRefIts, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dm_landau_amr_post_refine", "Number of levels to uniformly refine after AMR", "plexland.c", ctx->postAMRRefine, &ctx->postAMRRefine, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dm_landau_verbose", "", "plexland.c", ctx->verbose, &ctx->verbose, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dm_landau_re_radius","velocity range to refine on positive (z>0) r=0 axis for runaways","plexland.c",ctx->re_radius,&ctx->re_radius, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dm_landau_z_radius1","velocity range to refine r=0 axis (for electrons)","plexland.c",ctx->vperp0_radius1,&ctx->vperp0_radius1, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dm_landau_z_radius2","velocity range to refine r=0 axis (for ions) after origin AMR","plexland.c",ctx->vperp0_radius2,&ctx->vperp0_radius2, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dm_landau_Ez","Initial parallel electric field in unites of Conner-Hastie criticle field","plexland.c",ctx->Ez,&ctx->Ez, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dm_landau_n_0","Normalization constant for number density","plexland.c",ctx->n_0,&ctx->n_0, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dm_landau_ln_lambda","Cross section parameter","plexland.c",ctx->lnLam,&ctx->lnLam, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dm_landau_num_sections", "Number of tangential section in (2D) grid, 2, 3, of 4", "plexland.c", ctx->num_sections, &ctx->num_sections, NULL);CHKERRQ(ierr);
  ctx->simplex = PETSC_FALSE;
  /* get num species with tempurature*/
  {
    PetscReal arr[100];
    nt = 100;
    ierr = PetscOptionsRealArray("-dm_landau_thermal_temps", "Temperature of each species [e,i_0,i_1,...] in keV", "plexland.c", arr, &nt, &flg);CHKERRQ(ierr);
    if (flg && nt > LANDAU_MAX_SPECIES) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"-thermal_temps ,t1,t2,.. number of species %D > MAX %D",nt,LANDAU_MAX_SPECIES);
  }
  nt = LANDAU_MAX_SPECIES;
  for (ii=1;ii<LANDAU_MAX_SPECIES;ii++) {
    ctx->thermal_temps[ii] = 1.;
    ctx->charges[ii] = 1;
    ctx->masses[ii] = 1;
    ctx->n[ii] = (ii==1) ? 1 : 0;
  }
  ierr = PetscOptionsRealArray("-dm_landau_thermal_temps", "Temperature of each species [e,i_0,i_1,...] in keV (must be set to set number of species)", "plexland.c", ctx->thermal_temps, &nt, &flg);CHKERRQ(ierr);
  if (flg) {
    PetscInfo1(dummy, "num_species set to number of thermal temps provided (%D)\n",nt);
    ctx->num_species = nt;
  } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"-dm_landau_thermal_temps ,t1,t2,.. must be provided to set the number of species");
  for (ii=0;ii<ctx->num_species;ii++) ctx->thermal_temps[ii] *= 1.1604525e7; /* convert to Kelvin */
  nm = LANDAU_MAX_SPECIES-1;
  ierr = PetscOptionsRealArray("-dm_landau_ion_masses", "Mass of each species in units of proton mass [i_0=2,i_1=40...]", "plexland.c", &ctx->masses[1], &nm, &flg);CHKERRQ(ierr);
  if (flg && nm != ctx->num_species-1) {
    SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"num ion masses %D != num species %D",nm,ctx->num_species-1);
  }
  nm = LANDAU_MAX_SPECIES;
  ierr = PetscOptionsRealArray("-dm_landau_n", "Normalized (by -n_0) number density of each species", "plexland.c", ctx->n, &nm, &flg);CHKERRQ(ierr);
  if (flg && nm != ctx->num_species) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"wrong num n: %D != num species %D",nm,ctx->num_species);
  ctx->n_0 *= ctx->n[0]; /* normalized number density */
  for (ii=1;ii<ctx->num_species;ii++) ctx->n[ii] = ctx->n[ii]/ctx->n[0];
  ctx->n[0] = 1;
  for (ii=0;ii<LANDAU_MAX_SPECIES;ii++) ctx->masses[ii] *= 1.6720e-27; /* scale by proton mass kg */
  ctx->masses[0] = 9.10938356e-31; /* electron mass kg (should be about right already) */
  ctx->m_0 = ctx->masses[0]; /* arbitrary reference mass, electrons */
  ierr = PetscOptionsReal("-dm_landau_v_0","Velocity to normalize with in units of initial electrons thermal velocity (not recommended to change default)","plexland.c",ctx->v_0,&ctx->v_0, NULL);CHKERRQ(ierr);
  ctx->v_0 *= PetscSqrtReal(ctx->k*ctx->thermal_temps[0]/(ctx->masses[0])); /* electron mean velocity in 1D (need 3D form in computing T from FE integral) */
  nc = LANDAU_MAX_SPECIES-1;
  ierr = PetscOptionsRealArray("-dm_landau_ion_charges", "Charge of each species in units of proton charge [i_0=2,i_1=18,...]", "plexland.c", &ctx->charges[1], &nc, &flg);CHKERRQ(ierr);
  if (flg && nc != ctx->num_species-1) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"num charges %D != num species %D",nc,ctx->num_species-1);
  for (ii=0;ii<LANDAU_MAX_SPECIES;ii++) ctx->charges[ii] *= 1.6022e-19; /* electron/proton charge (MKS) */
  ctx->t_0 = 8*PETSC_PI*PetscSqr(ctx->epsilon0*ctx->m_0/PetscSqr(ctx->charges[0]))/ctx->lnLam/ctx->n_0*PetscPowReal(ctx->v_0,3); /* note, this t_0 makes nu[0,0]=1 */
  /* geometry */
  for (ii=0;ii<ctx->num_species;ii++) ctx->refineTol[ii]  = PETSC_MAX_REAL;
  for (ii=0;ii<ctx->num_species;ii++) ctx->coarsenTol[ii] = 0.;
  ii = LANDAU_MAX_SPECIES;
  ierr = PetscOptionsRealArray("-dm_landau_refine_tol","tolerance for refining cells in AMR","plexland.c",ctx->refineTol, &ii, &flg);CHKERRQ(ierr);
  if (flg && ii != ctx->num_species) ierr = PetscInfo2(dummy, "Phase: Warning, #refine_tol %D != num_species %D\n",ii,ctx->num_species);CHKERRQ(ierr);
  ii = LANDAU_MAX_SPECIES;
  ierr = PetscOptionsRealArray("-dm_landau_coarsen_tol","tolerance for coarsening cells in AMR","plexland.c",ctx->coarsenTol, &ii, &flg);CHKERRQ(ierr);
  if (flg && ii != ctx->num_species) ierr = PetscInfo2(dummy, "Phase: Warning, #coarsen_tol %D != num_species %D\n",ii,ctx->num_species);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dm_landau_domain_radius","Phase space size in units of electron thermal velocity","plexland.c",ctx->radius,&ctx->radius, &flg);CHKERRQ(ierr);
  if (flg && ctx->radius <= 0) { /* negative is ratio of c */
    if (ctx->radius == 0) ctx->radius = 0.75;
    else ctx->radius = -ctx->radius;
    ctx->radius = ctx->radius*299792458/ctx->v_0;
    ierr = PetscInfo1(dummy, "Change domain radius to %e\n",ctx->radius);CHKERRQ(ierr);
  }
  ierr = PetscOptionsReal("-dm_landau_i_radius","Ion thermal velocity, used for circular meshes","plexland.c",ctx->i_radius,&ctx->i_radius, &flg);CHKERRQ(ierr);
  if (flg && !sph_flg) ctx->sphere = PETSC_TRUE; /* you gave me an ion radius but did not set sphere, user error really */
  if (!flg) {
    ctx->i_radius = 1.5*PetscSqrtReal(8*ctx->k*ctx->thermal_temps[1]/ctx->masses[1]/PETSC_PI)/ctx->v_0; /* normalized radius with thermal velocity of first ion */
  }
  ierr = PetscOptionsReal("-dm_landau_e_radius","Electron thermal velocity, used for circular meshes","plexland.c",ctx->e_radius,&ctx->e_radius, &flg);CHKERRQ(ierr);
  if (flg && !sph_flg) ctx->sphere = PETSC_TRUE; /* you gave me an e radius but did not set sphere, user error really */
  if (!flg) {
    ctx->e_radius = 1.5*PetscSqrtReal(8*ctx->k*ctx->thermal_temps[0]/ctx->masses[0]/PETSC_PI)/ctx->v_0; /* normalized radius with thermal velocity of electrons */
  }
  if (ctx->sphere && (ctx->e_radius <= ctx->i_radius || ctx->radius <= ctx->e_radius)) SETERRQ3(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"bad radii: %g < %g < %g",ctx->i_radius,ctx->e_radius,ctx->radius);
  ierr = PetscOptionsInt("-dm_landau_sub_thread_block_size", "Number of threads in CUDA integration point subblock", "plexland.c", ctx->subThreadBlockSize, &ctx->subThreadBlockSize, NULL);CHKERRQ(ierr);
  if (ctx->subThreadBlockSize > LANDAU_MAX_SUB_THREAD_BLOCKS) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"num sub threads %D > MAX %D",ctx->subThreadBlockSize,LANDAU_MAX_SUB_THREAD_BLOCKS);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  for (ii=ctx->num_species;ii<LANDAU_MAX_SPECIES;ii++) ctx->masses[ii] = ctx->thermal_temps[ii]  = ctx->charges[ii] = 0;
  if (ctx->verbose > 0) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "masses:        e=%10.3e; ions in proton mass units:   %10.3e %10.3e ...\n",ctx->masses[0],ctx->masses[1]/1.6720e-27,ctx->num_species>2 ? ctx->masses[2]/1.6720e-27 : 0);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "charges:       e=%10.3e; charges in elementary units: %10.3e %10.3e\n", ctx->charges[0],-ctx->charges[1]/ctx->charges[0],ctx->num_species>2 ? -ctx->charges[2]/ctx->charges[0] : 0);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "thermal T (K): e=%10.3e i=%10.3e imp=%10.3e. v_0=%10.3e n_0=%10.3e t_0=%10.3e domain=%10.3e\n",ctx->thermal_temps[0],ctx->thermal_temps[1],ctx->num_species>2 ? ctx->thermal_temps[2] : 0,ctx->v_0,ctx->n_0,ctx->t_0,ctx->radius);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&dummy);CHKERRQ(ierr);
  {
    PetscMPIInt    rank;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
    /* PetscLogStage  setup_stage; */
    ierr = PetscLogEventRegister("Landau Operator", DM_CLASSID, &ctx->events[0]);CHKERRQ(ierr); /* 0 */
    ierr = PetscLogEventRegister(" Jac-vector", DM_CLASSID, &ctx->events[1]);CHKERRQ(ierr); /* 1 */
    ierr = PetscLogEventRegister(" Jac-kern-init", DM_CLASSID, &ctx->events[3]);CHKERRQ(ierr); /* 3 */
    ierr = PetscLogEventRegister(" Jac-kernel", DM_CLASSID, &ctx->events[4]);CHKERRQ(ierr); /* 4 */
    ierr = PetscLogEventRegister(" Jac-kernel-post", DM_CLASSID, &ctx->events[5]);CHKERRQ(ierr); /* 5 */
    ierr = PetscLogEventRegister(" Jac-assemble", DM_CLASSID, &ctx->events[6]);CHKERRQ(ierr); /* 6 */
    ierr = PetscLogEventRegister(" Jac-end", DM_CLASSID, &ctx->events[7]);CHKERRQ(ierr); /* 7 */
    ierr = PetscLogEventRegister("  Jac-geo-color", DM_CLASSID, &ctx->events[8]);CHKERRQ(ierr); /* 8 */
    ierr = PetscLogEventRegister("  Jac-cuda-sum", DM_CLASSID, &ctx->events[2]);CHKERRQ(ierr); /* 2 */
    ierr = PetscLogEventRegister("Landau Jacobian", DM_CLASSID, &ctx->events[9]);CHKERRQ(ierr); /* 9 */
    if (rank) { /* turn off output stuff for duplicate runs - do we need to add the prefix to all this? */
      ierr = PetscOptionsClearValue(NULL,"-snes_converged_reason");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-ksp_converged_reason");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-snes_monitor");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-ksp_monitor");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-ts_monitor");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-ts_adapt_monitor");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-dm_landau_amr_dm_view");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-dm_landau_amr_vec_view");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-dm_landau_pre_dm_view");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-dm_landau_pre_vec_view");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-info");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  LandauCreateVelocitySpace - Create a DMPlex velocity space mesh

  Collective on comm

  Input Parameters:
+   comm  - The MPI communicator
.   dim - velocity space dimension (2 for axisymmetric, 3 for full 3X + 3V solver)
-   prefix - prefix for options

  Output Parameter:
.   dm  - The DM object representing the mesh
+   X - A vector (user destroys)
-   J - Optional matrix (object destroys)

  Level: beginner

.keywords: mesh
.seealso: DMPlexCreate(), LandauDestroyVelocitySpace()
@*/
PetscErrorCode LandauCreateVelocitySpace(MPI_Comm comm, PetscInt dim, const char prefix[], Vec *X, Mat *J, DM *dm)
{
  PetscMPIInt    size;
  PetscErrorCode ierr;
  LandauCtx      *ctx;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (size!=1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Velocity space meshes should be serial (but should work in parallel)");
  if (dim!=2 && dim!=3) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Only 2D and 3D supported");
  ctx = (LandauCtx*)malloc(sizeof(LandauCtx));
  /* process options */
  ierr = ProcessOptions(ctx,prefix);CHKERRQ(ierr);
  /* Create Mesh */
  ierr = LandauDMCreateVMesh(comm, dim, prefix, ctx, dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm,NULL,"-dm_landau_pre_dm_view");CHKERRQ(ierr);
  ierr = DMSetApplicationContext(*dm, ctx);CHKERRQ(ierr);
  /* create FEM */
  ierr = SetupDS(*dm,dim,ctx);CHKERRQ(ierr);
  /* set initial state */
  ierr = DMCreateGlobalVector(*dm,X);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *X, "u");CHKERRQ(ierr);
  /* initial static refinement, no solve */
  ierr = LandauSetInitialCondition(*dm, *X, ctx);CHKERRQ(ierr);
  ierr = VecViewFromOptions(*X, NULL, "-dm_landau_pre_vec_view");CHKERRQ(ierr);
  /* forest refinement */
  if (ctx->errorIndicator) {
    /* AMR */
    ierr = adapt(dm,ctx,X);CHKERRQ(ierr);
    ierr = DMViewFromOptions(*dm,NULL,"-dm_landau_amr_dm_view");CHKERRQ(ierr);
    ierr = VecViewFromOptions(*X, NULL, "-dm_landau_amr_vec_view");CHKERRQ(ierr);
  }
  ierr = DMSetApplicationContext(*dm, ctx);CHKERRQ(ierr);
  ctx->dmv = *dm;
  ierr = DMCreateMatrix(ctx->dmv, &ctx->J);CHKERRQ(ierr);
  if (J) *J = ctx->J;
  PetscFunctionReturn(0);
}

/*@
  LandauDestroyVelocitySpace - Destroy a DMPlex velocity space mesh

  Collective on dm

  Input/Output Parameters:
  .   dm - the dm to destroy

  Level: beginner

.keywords: mesh
.seealso: LandauCreateVelocitySpace()
@*/
PetscErrorCode LandauDestroyVelocitySpace(DM *dm)
{
  PetscErrorCode ierr,ii;
  LandauCtx        *ctx;
  PetscContainer container = NULL;
  PetscFunctionBegin;
  ierr = DMGetApplicationContext(*dm, &ctx);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)ctx->J,"coloring", (PetscObject*)&container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&ctx->M);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->J);CHKERRQ(ierr);
  for (ii=0;ii<ctx->num_species;ii++) {
    ierr = PetscFEDestroy(&ctx->fe[ii]);CHKERRQ(ierr);
  }
  free(ctx);
  ierr = DMDestroy(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* < v, ru > */
static void f0_s_den(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscInt ii = (PetscInt)PetscRealPart(constants[0]);
  f0[0] = u[ii];
}

/* < v, ru > */
static void f0_s_mom(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscInt ii = (PetscInt)PetscRealPart(constants[0]), jj = (PetscInt)PetscRealPart(constants[1]);
  f0[0] = x[jj]*u[ii]; /* x momentum */
}

static void f0_s_v2(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscInt i, ii = (PetscInt)PetscRealPart(constants[0]);
  double tmp1 = 0.;
  for (i = 0; i < dim; ++i) tmp1 += x[i]*x[i];
  f0[0] = tmp1*u[ii];
}

/* < v, ru > */
static void f0_s_rden(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscInt ii = (PetscInt)PetscRealPart(constants[0]);
  f0[0] = 2.*PETSC_PI*x[0]*u[ii];
}

/* < v, ru > */
static void f0_s_rmom(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscInt ii = (PetscInt)PetscRealPart(constants[0]);
  f0[0] = 2.*PETSC_PI*x[0]*x[1]*u[ii];
}

static void f0_s_rv2(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscInt ii = (PetscInt)PetscRealPart(constants[0]);
  f0[0] =  2.*PETSC_PI*x[0]*(x[0]*x[0] + x[1]*x[1])*u[ii];
}

/*@
  LandauPrintNorms - collects moments and prints them

  Collective on dm

  Input Parameters:
+   X  - the state
-   stepi - current step to print

  Level: beginner

.keywords: mesh
.seealso: LandauCreateVelocitySpace()
@*/
PetscErrorCode LandauPrintNorms(Vec X, PetscInt stepi)
{
  PetscErrorCode ierr;
  LandauCtx      *ctx;
  PetscDS        prob;
  DM             plex,dm;
  PetscInt       cStart, cEnd, dim, ii;
  PetscScalar    xmomentumtot=0, ymomentumtot=0, zmomentumtot=0, energytot=0, densitytot=0, tt[LANDAU_MAX_SPECIES];
  PetscScalar    xmomentum[LANDAU_MAX_SPECIES],  ymomentum[LANDAU_MAX_SPECIES],  zmomentum[LANDAU_MAX_SPECIES], energy[LANDAU_MAX_SPECIES], density[LANDAU_MAX_SPECIES];

  PetscFunctionBegin;
  ierr = VecGetDM(X, &dm);CHKERRQ(ierr);
  if (!dm) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "no DM");
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr);
  if (!ctx) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  ierr = DMConvert(ctx->dmv, DMPLEX, &plex);CHKERRQ(ierr);
  ierr = DMCreateDS(plex);CHKERRQ(ierr);
  ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
  /* print momentum and energy */
  for (ii=0;ii<ctx->num_species;ii++) {
    PetscScalar user[2] = { (PetscScalar)ii, (PetscScalar)ctx->charges[ii]};
    ierr = PetscDSSetConstants(prob, 2, user);CHKERRQ(ierr);
    if (dim==2) { /* 2/3X + 3V (cylindrical coordinates) */
      ierr = PetscDSSetObjective(prob, 0, &f0_s_rden);CHKERRQ(ierr);
      ierr = DMPlexComputeIntegralFEM(plex,X,tt,ctx);CHKERRQ(ierr);
      density[ii] = tt[0]*ctx->n_0*ctx->charges[ii];
      ierr = PetscDSSetObjective(prob, 0, &f0_s_rmom);CHKERRQ(ierr);
      ierr = DMPlexComputeIntegralFEM(plex,X,tt,ctx);CHKERRQ(ierr);
      zmomentum[ii] = tt[0]*ctx->n_0*ctx->v_0*ctx->masses[ii];
      ierr = PetscDSSetObjective(prob, 0, &f0_s_rv2);CHKERRQ(ierr);
      ierr = DMPlexComputeIntegralFEM(plex,X,tt,ctx);CHKERRQ(ierr);
      energy[ii] = tt[0]*0.5*ctx->n_0*ctx->v_0*ctx->v_0*ctx->masses[ii];
      zmomentumtot += zmomentum[ii];
      energytot  += energy[ii];
      densitytot += density[ii];
      ierr = PetscPrintf(PETSC_COMM_WORLD, "%3D) species-%D: charge density= %20.13e z-momentum= %20.13e energy= %20.13e",stepi,ii,density[ii],zmomentum[ii],energy[ii]);CHKERRQ(ierr);
    } else { /* 2/3X + 3V */
      ierr = PetscDSSetObjective(prob, 0, &f0_s_den);CHKERRQ(ierr);
      ierr = DMPlexComputeIntegralFEM(plex,X,tt,ctx);CHKERRQ(ierr);
      density[ii] = tt[0]*ctx->n_0*ctx->charges[ii];
      ierr = PetscDSSetObjective(prob, 0, &f0_s_mom);CHKERRQ(ierr);
      user[1] = 0;
      ierr = DMPlexComputeIntegralFEM(plex,X,tt,ctx);CHKERRQ(ierr);
      xmomentum[ii]  = tt[0]*ctx->n_0*ctx->v_0*ctx->masses[ii];
      user[1] = 1;
      ierr = DMPlexComputeIntegralFEM(plex,X,tt,ctx);CHKERRQ(ierr);
      ymomentum[ii] = tt[0]*ctx->n_0*ctx->v_0*ctx->masses[ii];
      user[1] = 2;
      ierr = DMPlexComputeIntegralFEM(plex,X,tt,ctx);CHKERRQ(ierr);
      zmomentum[ii] = tt[0]*ctx->n_0*ctx->v_0*ctx->masses[ii];
      ierr = PetscDSSetObjective(prob, 0, &f0_s_v2);CHKERRQ(ierr);
      ierr = DMPlexComputeIntegralFEM(plex,X,tt,ctx);CHKERRQ(ierr);
      energy[ii]    = 0.5*tt[0]*ctx->n_0*ctx->v_0*ctx->v_0*ctx->masses[ii];
      ierr = PetscPrintf(PETSC_COMM_WORLD, "%3D) species %D: density=%20.13e, x-momentum=%20.13e, y-momentum=%20.13e, z-momentum=%20.13e, energy=%21.13e",
                         stepi,ii,density[ii],xmomentum[ii],ymomentum[ii],zmomentum[ii],energy[ii]);CHKERRQ(ierr);
      xmomentumtot += xmomentum[ii];
      ymomentumtot += ymomentum[ii];
      zmomentumtot += zmomentum[ii];
      energytot  += energy[ii];
      densitytot += density[ii];
    }
    if (ctx->num_species>1) PetscPrintf(PETSC_COMM_WORLD, "\n");
  }
  /* totals */
  ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  if (ctx->num_species>1) {
    if (dim==2) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "\t%3D) Total: charge density=%21.13e, momentum=%21.13e, energy=%21.13e (m_i[0]/m_e = %g, %D cells)",
                  stepi,densitytot,zmomentumtot,energytot,ctx->masses[1]/ctx->masses[0],cEnd-cStart);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "\t%3D) Total: charge density=%21.13e, x-momentum=%21.13e, y-momentum=%21.13e, z-momentum=%21.13e, energy=%21.13e (m_i[0]/m_e = %g, %D cells)",
                  stepi,densitytot,xmomentumtot,ymomentumtot,zmomentumtot,energytot,ctx->masses[1]/ctx->masses[0],cEnd-cStart);CHKERRQ(ierr);
    }
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD, " -- %D cells",cEnd-cStart);CHKERRQ(ierr);
  }
  if (ctx->verbose > 1) {ierr = PetscPrintf(PETSC_COMM_WORLD,", %D sub (vector) threads\n",ctx->subThreadBlockSize);CHKERRQ(ierr);}
  else {ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode destroy_coloring (void *is)
{
  ISColoring tmp = (ISColoring)is;
  return ISColoringDestroy(&tmp);
}

/*@
  LandauCreateColoring - create a coloring and add to matrix (Landau context used just for 'print' flag, should be in DMPlex)

  Collective on JacP

  Input Parameters:
+   JacP  - matrix to add coloring to
-   plex - The DM

  Output Parameter:
.   container  - Container with coloring

  Level: beginner

.keywords: mesh
.seealso: LandauCreateVelocitySpace()
@*/
PetscErrorCode LandauCreateColoring(Mat JacP, DM plex, PetscContainer *container)
{
  PetscErrorCode  ierr;
  PetscInt        dim,cell,i,ej,nc,Nv,totDim,numGCells,cStart,cEnd;
  ISColoring      iscoloring = NULL;
  Mat             G,Q;
  PetscScalar     ones[128];
  MatColoring     mc;
  IS             *is;
  PetscInt        csize,colour,j,k;
  const PetscInt *indices;
  PetscInt       numComp[1];
  PetscInt       numDof[4];
  PetscFE        fe;
  DM             colordm;
  PetscSection   csection, section, globalSection;
  PetscDS        prob;
  LandauCtx      *ctx;

  PetscFunctionBegin;
  ierr = DMGetApplicationContext(plex, &ctx);CHKERRQ(ierr);
  ierr = DMGetLocalSection(plex, &section);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(plex, &globalSection);CHKERRQ(ierr);
  ierr = DMGetDimension(plex, &dim);CHKERRQ(ierr);
  ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
  numGCells = cEnd - cStart;
  /* create cell centered DM */
  ierr = DMClone(plex, &colordm);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) plex), dim, 1, PETSC_FALSE, "color_", PETSC_DECIDE, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "color");CHKERRQ(ierr);
  ierr = DMSetField(colordm, 0, NULL, (PetscObject)fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  for (i = 0; i < (dim+1); ++i) numDof[i] = 0;
  numDof[dim] = 1;
  numComp[0] = 1;
  ierr = DMPlexCreateSection(colordm, NULL, numComp, numDof, 0, NULL, NULL, NULL, NULL, &csection);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(csection, 0, "color");CHKERRQ(ierr);
  ierr = DMSetLocalSection(colordm, csection);CHKERRQ(ierr);
  ierr = DMViewFromOptions(colordm,NULL,"-color_dm_view");CHKERRQ(ierr);
  /* get vertex to element map Q and colroing graph G */
  ierr = MatGetSize(JacP,NULL,&Nv);CHKERRQ(ierr);
  ierr = MatCreateAIJ(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,numGCells,Nv,totDim,NULL,0,NULL,&Q);CHKERRQ(ierr);
  for (i=0;i<128;i++) ones[i] = 1.0;
  for (cell = cStart, ej = 0 ; cell < cEnd; ++cell, ++ej) {
    PetscInt numindices,*indices;
    ierr = DMPlexGetClosureIndices(plex, section, globalSection, cell, PETSC_TRUE, &numindices, &indices, NULL, NULL);CHKERRQ(ierr);
    if (numindices>128) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "too many indices. %D > %D",numindices,128);
    ierr = MatSetValues(Q,1,&ej,numindices,indices,ones,ADD_VALUES);CHKERRQ(ierr);
    ierr = DMPlexRestoreClosureIndices(plex, section, globalSection, cell, PETSC_TRUE, &numindices, &indices, NULL, NULL);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Q, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Q, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatMatTransposeMult(Q,Q,MAT_INITIAL_MATRIX,4.0,&G);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) Q, "Q");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) G, "coloring graph");CHKERRQ(ierr);
  ierr = MatViewFromOptions(G,NULL,"-coloring_mat_view");CHKERRQ(ierr);
  ierr = MatViewFromOptions(Q,NULL,"-coloring_mat_view");CHKERRQ(ierr);
  ierr = MatDestroy(&Q);CHKERRQ(ierr);
  /* coloring */
  ierr = MatColoringCreate(G,&mc);CHKERRQ(ierr);
  ierr = MatColoringSetDistance(mc,1);CHKERRQ(ierr);
  ierr = MatColoringSetType(mc,MATCOLORINGJP);CHKERRQ(ierr);
  ierr = MatColoringSetFromOptions(mc);CHKERRQ(ierr);
  ierr = MatColoringApply(mc,&iscoloring);CHKERRQ(ierr);
  ierr = MatColoringDestroy(&mc);CHKERRQ(ierr);
  /* view */
  ierr = ISColoringViewFromOptions(iscoloring,NULL,"-coloring_is_view");CHKERRQ(ierr);
  ierr = ISColoringGetIS(iscoloring,PETSC_USE_POINTER,&nc,&is);CHKERRQ(ierr);
  if (ctx && ctx->verbose > 5) {
    PetscViewer    viewer;
    Vec            color_vec, eidx_vec;
    ierr = DMGetGlobalVector(colordm, &color_vec);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(colordm, &eidx_vec);CHKERRQ(ierr);
    for (colour=0; colour<nc; colour++) {
      ierr = ISGetLocalSize(is[colour],&csize);CHKERRQ(ierr);
      ierr = ISGetIndices(is[colour],&indices);CHKERRQ(ierr);
      for (j=0; j<csize; j++) {
        PetscScalar v = (PetscScalar)colour;
        k = indices[j];
        ierr = VecSetValues(color_vec,1,&k,&v,INSERT_VALUES);
        v = (PetscScalar)k;
        ierr = VecSetValues(eidx_vec,1,&k,&v,INSERT_VALUES);
      }
      ierr = ISRestoreIndices(is[colour],&indices);CHKERRQ(ierr);
    }
    /* view */
    ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD, "color.vtu", FILE_MODE_WRITE, &viewer);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) color_vec, "color");CHKERRQ(ierr);
    ierr = VecView(color_vec, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD, "eidx.vtu", FILE_MODE_WRITE, &viewer);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) eidx_vec, "element-idx");CHKERRQ(ierr);
    ierr = VecView(eidx_vec, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(colordm, &color_vec);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(colordm, &eidx_vec);CHKERRQ(ierr);
  }
  ierr = PetscSectionDestroy(&csection);CHKERRQ(ierr);
  ierr = DMDestroy(&colordm);CHKERRQ(ierr);
  ierr = ISColoringRestoreIS(iscoloring,PETSC_USE_POINTER,&is);CHKERRQ(ierr);
  ierr = MatDestroy(&G);CHKERRQ(ierr);
  /* stash coloring */
  ierr = PetscContainerCreate(PETSC_COMM_SELF, container);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(*container,(void*)iscoloring);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(*container, destroy_coloring);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)JacP,"coloring",(PetscObject)*container);CHKERRQ(ierr);
  if (ctx && ctx->verbose > 0) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Made coloring with %D colors\n", nc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
  }

PetscErrorCode LandauAssembleOpenMP(PetscInt cStart, PetscInt cEnd, PetscInt totDim, DM plex, PetscSection section, PetscSection globalSection, Mat JacP, PetscScalar elemMats[], PetscContainer container)
{
  PetscErrorCode  ierr;
  IS             *is;
  PetscInt        nc,colour,j;
  const PetscInt *clr_idxs;
  ISColoring      iscoloring;
  PetscFunctionBegin;
  ierr = PetscContainerGetPointer(container,(void**)&iscoloring);CHKERRQ(ierr);
  ierr = ISColoringGetIS(iscoloring,PETSC_USE_POINTER,&nc,&is);CHKERRQ(ierr);
  for (colour=0; colour<nc; colour++) {
    PetscInt    *idx_arr[1024]; /* need to make dynamic for general use */
    PetscScalar *new_el_mats[1024];
    PetscInt     idx_size[1024],csize;
    ierr = ISGetLocalSize(is[colour],&csize);CHKERRQ(ierr);
    if (csize>1024) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "too many elements in color. %D > %D",csize,1024);
    ierr = ISGetIndices(is[colour],&clr_idxs);CHKERRQ(ierr);
    /* get indices and mats */
    for (j=0; j<csize; j++) {
      PetscInt    cell = cStart + clr_idxs[j];
      PetscInt    numindices,*indices;
      PetscScalar *elMat = &elemMats[clr_idxs[j]*totDim*totDim];
      PetscScalar *valuesOrig = elMat;
      ierr = DMPlexGetClosureIndices(plex, section, globalSection, cell, PETSC_TRUE, &numindices, &indices, NULL, (PetscScalar **) &elMat);CHKERRQ(ierr);
      idx_size[j] = numindices;
      ierr = PetscMalloc2(numindices,&idx_arr[j],numindices*numindices,&new_el_mats[j]);CHKERRQ(ierr);
      ierr = PetscMemcpy(idx_arr[j],indices,numindices*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscMemcpy(new_el_mats[j],elMat,numindices*numindices*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = DMPlexRestoreClosureIndices(plex, section, globalSection, cell, PETSC_TRUE, &numindices, &indices, NULL, (PetscScalar **) &elMat);CHKERRQ(ierr);
      if (elMat != valuesOrig) {ierr = DMRestoreWorkArray(plex, numindices*numindices, MPIU_SCALAR, &elMat);}
    }
    /* assemble matrix - pragmas break CI ? */
    //#pragma omp parallel default(JacP,idx_size,idx_arr,new_el_mats,colour,clr_idxs)  private(j)
    //#pragma omp parallel for private(j)
    for (j=0; j<csize; j++) {
      PetscInt    numindices = idx_size[j], *indices = idx_arr[j];
      PetscScalar *elMat = new_el_mats[j];
      MatSetValues(JacP,numindices,indices,numindices,indices,elMat,ADD_VALUES);
    }
    /* free */
    ierr = ISRestoreIndices(is[colour],&clr_idxs);CHKERRQ(ierr);
    for (j=0; j<csize; j++) {
      ierr = PetscFree2(idx_arr[j],new_el_mats[j]);CHKERRQ(ierr);
    }
  }
  ierr = ISColoringRestoreIS(iscoloring,PETSC_USE_POINTER,&is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* < v, u > */
static void g0_1(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.;
}

/* < v, u > */
static void g0_r(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 2.*PETSC_PI*x[0];
}

/*@
  LandauCreateMassMatrix - Create mass matrix for Landau

  Collective on dm

  Input Parameters:
. dm     - the DM object

 Output Parameters:
. Amat - The mass matrix (optional), mass matrix is added to the DM context

  Level: beginner

.keywords: mesh
.seealso: LandauCreateVelocitySpace()
@*/
PetscErrorCode LandauCreateMassMatrix(DM dm, Mat *Amat)
{
  DM             massDM;
  PetscDS        prob;
  PetscInt       ii,dim,N1=1,N2;
  PetscErrorCode ierr;
  LandauCtx      *ctx;
  Mat            M;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (Amat) PetscValidPointer(Amat,3);
  ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr);
  if (!ctx) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMClone(dm, &massDM);CHKERRQ(ierr);
  ierr = DMCopyFields(dm, massDM);CHKERRQ(ierr);
  ierr = DMCreateDS(massDM);CHKERRQ(ierr);
  ierr = DMGetDS(massDM, &prob);CHKERRQ(ierr);
  for (ii=0;ii<ctx->num_species;ii++) {
    if (dim==3) {ierr = PetscDSSetJacobian(prob, ii, ii, g0_1, NULL, NULL, NULL);CHKERRQ(ierr);}
    else        {ierr = PetscDSSetJacobian(prob, ii, ii, g0_r, NULL, NULL, NULL);CHKERRQ(ierr);}
  }
  ierr = DMViewFromOptions(massDM,NULL,"-dm_landau_mass_dm_view");CHKERRQ(ierr);
  ierr = DMCreateMatrix(massDM, &M);CHKERRQ(ierr);
  {
    Vec locX;
    DM  plex;
    ierr = DMConvert(massDM, DMPLEX, &plex);CHKERRQ(ierr);
    ierr = DMGetLocalVector(massDM, &locX);CHKERRQ(ierr);
    /* Mass matrix is independent of the input, so no need to fill locX */
    ierr = DMPlexSNESComputeJacobianFEM(plex, locX, M, M, ctx);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(massDM, &locX);CHKERRQ(ierr);
    ierr = DMDestroy(&plex);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&massDM);CHKERRQ(ierr);
  ierr = MatGetSize(ctx->J, &N1, NULL);CHKERRQ(ierr);
  ierr = MatGetSize(M, &N2, NULL);CHKERRQ(ierr);
  if (N1 != N2) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Incorrect matrix sizes: |Jacobian| = %D, |Mass|=%D",N1,N2);
  ierr = MatViewFromOptions(M,NULL,"-dm_landau_mass_mat_view");CHKERRQ(ierr);
  ctx->M = M; /* this could be a noop, a = a */
  if (Amat) *Amat = M;
  PetscFunctionReturn(0);
}

/*@
  LandauIFunction - TS residual calculation

  Collective on ts

  Input Parameters:
+   TS  - The time stepping context
.   time_dummy - current time (not used)
-   X - Current state
+   X_t - Time derivative of current state
.   actx - Landau context

  Output Parameter:
.   F  - The residual

  Level: beginner

.keywords: mesh
.seealso: LandauCreateVelocitySpace(), LandauIJacobian()
@*/
PetscErrorCode LandauIFunction(TS ts,PetscReal time_dummy,Vec X,Vec X_t,Vec F,void *actx)
{
  PetscErrorCode ierr;
  LandauCtx      *ctx=(LandauCtx*)actx;
  PetscReal      unorm;
  PetscInt       dim;
  DM dm;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr);
  if (!ctx) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  ierr = VecNorm(X,NORM_2,&unorm);CHKERRQ(ierr);
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr); // remove in real application
  ierr = PetscLogEventBegin(ctx->events[0],0,0,0,0);CHKERRQ(ierr);
  ierr = DMGetDimension(ctx->dmv, &dim);CHKERRQ(ierr);
  if (ctx->normJ!=unorm) {
    ctx->normJ = unorm;
    ierr = LandauFormJacobian_Internal(X,ctx->J,dim,(void*)ctx);CHKERRQ(ierr);
    ctx->aux_bool = PETSC_TRUE; /* debug: set flag that we made a new Jacobian */
  } else ctx->aux_bool = PETSC_FALSE;
  /* mat vec for op */
  ierr = MatMult(ctx->J,X,F);CHKERRQ(ierr);CHKERRQ(ierr); /* C*f */
  /* add time term */
  if (X_t) {
    ierr = MatMultAdd(ctx->M,X_t,F,F);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(ctx->events[0],0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  LandauIJacobian - TS Jacobian construction

  Collective on ts

  Input Parameters:
+   TS  - The time stepping context
.   time_dummy - current time (not used)
-   X - Current state
+   U_tdummy - Time derivative of current state (not used)
.   shift - shift for du/dt term
-   actx - Landau context

  Output Parameter:
.   Amat  - Jacobian
+   Pmat  - same as Amat

  Level: beginner

.keywords: mesh
.seealso: LandauCreateVelocitySpace(), LandauIFunction()
@*/
PetscErrorCode LandauIJacobian(TS ts,PetscReal time_dummy,Vec X,Vec U_tdummy,PetscReal shift,Mat Amat,Mat Pmat,void *actx)
{
  PetscErrorCode ierr;
  LandauCtx      *ctx=(LandauCtx*)actx;
  PetscReal      unorm;
  PetscInt       dim;
  DM dm;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr);
  if (!ctx) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  if (Amat!=Pmat || Amat!=ctx->J) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Amat!=Pmat || Amat!=ctx->J");
  ierr = DMGetDimension(ctx->dmv, &dim);CHKERRQ(ierr);
  /* get collision Jacobian into A */
  ierr = PetscLogEventBegin(ctx->events[9],0,0,0,0);CHKERRQ(ierr);
  ierr = VecNorm(X,NORM_2,&unorm);CHKERRQ(ierr);
  if (ctx->normJ!=unorm) {
    ierr = LandauFormJacobian_Internal(X,ctx->J,dim,(void*)ctx);CHKERRQ(ierr);
    ctx->normJ = unorm;
    ctx->aux_bool = PETSC_TRUE; /* debug: set flag that we made a new Jacobian */
  } else ctx->aux_bool = PETSC_FALSE;
  /* add C */
  ierr = MatCopy(ctx->J,Pmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  /* add mass */
  ierr = MatAXPY(Pmat,shift,ctx->M,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ctx->events[9],0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
