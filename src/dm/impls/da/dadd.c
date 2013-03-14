#include <petsc-private/dmdaimpl.h>

#undef __FUNCT__
#define __FUNCT__ "DMDACreatePatchIS"
/*@
  DMDACreatePatchIS - Creates an index set corresponding to a patch of the DA.

  Not Collective

  Input Parameters:
+  da - the DMDA
.  lower - a matstencil with i, j and k corresponding to the lower corner of the patch
-  upper - a matstencil with i, j and k corresponding to the upper corner of the patch

  Output Parameters:
.  is - the IS corresponding to the patch

  Level: developer

.seealso: DMDACreateDomainDecomposition(), DMDACreateDomainDecompositionScatters()
@*/
PetscErrorCode DMDACreatePatchIS(DM da,MatStencil *lower,MatStencil *upper,IS *is)
{
  PetscInt       ms=0,ns=0,ps=0;
  PetscInt       me=1,ne=1,pe=1;
  PetscInt       mr=0,nr=0,pr=0;
  PetscInt       ii,jj,kk;
  PetscInt       si,sj,sk;
  PetscInt       i,j,k,l,idx;
  PetscInt       base;
  PetscInt       xm=1,ym=1,zm=1;
  const PetscInt *lx,*ly,*lz;
  PetscInt       ox,oy,oz;
  PetscInt       m,n,p,M,N,P,dof;
  PetscInt       nindices;
  PetscInt       *indices;
  DM_DA          *dd = (DM_DA*)da->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* need to get the sizes of the actual DM rather than the "global" space of a subdomain DM */
  M = dd->M;N = dd->N;P=dd->P;
  m = dd->m;n = dd->n;p=dd->p;
  dof = dd->w;
  ierr = DMDAGetOffset(da,&ox,&oy,&oz,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetOwnershipRanges(da,&lx,&ly,&lz);CHKERRQ(ierr);
  nindices = (upper->i - lower->i)*(upper->j - lower->j)*(upper->k - lower->k)*dof;
  ierr = PetscMalloc(sizeof(PetscInt)*nindices,&indices);CHKERRQ(ierr);
  /* start at index 0 on processor 0 */
  mr = 0;
  nr = 0;
  pr = 0;
  ms = 0;
  ns = 0;
  ps = 0;
  if (lx) me = lx[0];
  if (ly) ne = ly[0];
  if (lz) pe = lz[0];
  idx = 0;
  for (k=lower->k-oz;k<upper->k-oz;k++) {
    for (j=lower->j-oy;j < upper->j-oy;j++) {
      for (i=lower->i-ox;i < upper->i-ox;i++) {
        /* "actual" indices rather than ones outside of the domain */
        ii = i;
        jj = j;
        kk = k;
        if (ii < 0) ii = M + ii;
        if (jj < 0) jj = N + jj;
        if (kk < 0) kk = P + kk;
        if (ii > M-1) ii = ii - M;
        if (jj > N-1) jj = jj - N;
        if (kk > P-1) kk = kk - P;
        /* gone out of processor range on x axis */
        while(ii > me-1 || ii < ms) {
          if (mr == m-1) {
            ms = 0;
            me = lx[0];
            mr = 0;
          } else {
            mr++;
            ms = me;
            me += lx[mr];
          }
        }
        /* gone out of processor range on y axis */
        while(jj > ne-1 || jj < ns) {
          if (nr == n-1) {
            ns = 0;
            ne = ly[0];
            nr = 0;
          } else {
            nr++;
            ns = ne;
            ne += ly[nr];
          }
        }
        /* gone out of processor range on z axis */
        while(kk > pe-1 || kk < ps) {
          if (pr == p-1) {
            ps = 0;
            pe = lz[0];
            pr = 0;
          } else {
            pr++;
            ps = pe;
            pe += lz[pr];
          }
        }
        /* compute the vector base on owning processor */
        xm = me - ms;
        ym = ne - ns;
        zm = pe - ps;
        base = ms*ym*zm + ns*M + ps*M*N;
        /* compute the local coordinates on owning processor */
        si = ii - ms;
        sj = jj - ns;
        sk = kk - ps;
        for (l=0;l<dof;l++) {
          indices[idx] = l + dof*(base + si + xm*sj + xm*ym*sk);
          idx++;
        }
      }
    }
  }
  ISCreateGeneral(PETSC_COMM_SELF,idx,indices,PETSC_OWN_POINTER,is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDASubDomainDA_Private"
PetscErrorCode DMDASubDomainDA_Private(DM dm, DM *dddm)
{
  DM             da;
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  PetscReal      lmin[3],lmax[3];
  PetscInt       xsize,ysize,zsize;
  PetscInt       xo,yo,zo;
  PetscInt       xol,yol,zol;

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  ierr = DMDAGetOverlap(dm,&xol,&yol,&zol);CHKERRQ(ierr);

  ierr = DMDACreate(PETSC_COMM_SELF,&da);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(da,"sub_");CHKERRQ(ierr);
  ierr = DMDASetDim(da, info.dim);CHKERRQ(ierr);
  ierr = DMDASetDof(da, info.dof);CHKERRQ(ierr);

  ierr = DMDASetStencilType(da,info.st);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(da,info.sw);CHKERRQ(ierr);

  /* local with overlap */
  xsize = info.xm;
  ysize = info.ym;
  zsize = info.zm;
  xo    = info.xs;
  yo    = info.ys;
  zo    = info.zs;
  if (info.bx == DMDA_BOUNDARY_PERIODIC || (info.xs != 0)) {
    xsize += xol;
    xo    -= xol;
  }
  if (info.by == DMDA_BOUNDARY_PERIODIC || (info.ys != 0)) {
    ysize += yol;
    yo    -= yol;
  }
  if (info.bz == DMDA_BOUNDARY_PERIODIC || (info.zs != 0)) {
    zsize += zol;
    zo    -= zol;
  }

  if (info.bx == DMDA_BOUNDARY_PERIODIC || (info.xs+info.xm != info.mx)) xsize += xol;
  if (info.by == DMDA_BOUNDARY_PERIODIC || (info.ys+info.ym != info.my)) ysize += yol;
  if (info.bz == DMDA_BOUNDARY_PERIODIC || (info.zs+info.zm != info.mz)) zsize += zol;

  ierr = DMDASetSizes(da, xsize, ysize, zsize);CHKERRQ(ierr);
  ierr = DMDASetNumProcs(da, 1, 1, 1);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(da, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED);CHKERRQ(ierr);

  /* set up as a block instead */
  ierr = DMSetUp(da);CHKERRQ(ierr);

  /* todo - nonuniform coordinates */
  ierr = DMDAGetLocalBoundingBox(dm,lmin,lmax);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,lmin[0],lmax[0],lmin[1],lmax[1],lmin[2],lmax[2]);CHKERRQ(ierr);

  /* nonoverlapping region */
  ierr = DMDASetNonOverlappingRegion(da,info.xs,info.ys,info.zs,info.xm,info.ym,info.zm);CHKERRQ(ierr);

  /* this alters the behavior of DMDAGetInfo, DMDAGetLocalInfo, DMDAGetCorners, and DMDAGetGhostedCorners and should be used with care */
  ierr = DMDASetOffset(da,xo,yo,zo,info.mx,info.my,info.mz);CHKERRQ(ierr);
  *dddm = da;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateDomainDecompositionScatters_DA"
/*
 Fills the local vector problem on the subdomain from the global problem.

 Right now this assumes one subdomain per processor.

 */
PetscErrorCode DMCreateDomainDecompositionScatters_DA(DM dm,PetscInt nsubdms,DM *subdms,VecScatter **iscat,VecScatter **oscat, VecScatter **lscat)
{
  PetscErrorCode ierr;
  DMDALocalInfo  info,subinfo;
  DM             subdm;
  MatStencil     upper,lower;
  IS             idis,isis,odis,osis,gdis;
  Vec            svec,dvec,slvec;
  PetscInt       xm,ym,zm,xs,ys,zs;

  PetscFunctionBegin;
  if (nsubdms != 1) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Cannot have more than one subdomain per processor (yet)");

  /* allocate the arrays of scatters */
  if (iscat) {ierr = PetscMalloc(sizeof(VecScatter*),iscat);CHKERRQ(ierr);}
  if (oscat) {ierr = PetscMalloc(sizeof(VecScatter*),oscat);CHKERRQ(ierr);}
  if (lscat) {ierr = PetscMalloc(sizeof(VecScatter*),lscat);CHKERRQ(ierr);}

  ierr  = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  subdm = subdms[0];
  ierr  = DMDAGetLocalInfo(subdm,&subinfo);CHKERRQ(ierr);

  /* create the global and subdomain index sets for the inner domain */
  /* TODO - make this actually support multiple subdomains -- subdomain needs to provide where it's nonoverlapping portion belongs */
  ierr = DMDAGetNonOverlappingRegion(subdm,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  lower.i = xs;
  lower.j = ys;
  lower.k = zs;
  upper.i = xs+xm;
  upper.j = ys+ym;
  upper.k = zs+zm;
  ierr    = DMDACreatePatchIS(dm,&lower,&upper,&idis);CHKERRQ(ierr);
  ierr    = DMDACreatePatchIS(subdm,&lower,&upper,&isis);CHKERRQ(ierr);

  /* create the global and subdomain index sets for the outer subdomain */
  lower.i = subinfo.xs;
  lower.j = subinfo.ys;
  lower.k = subinfo.zs;
  upper.i = subinfo.xs+subinfo.xm;
  upper.j = subinfo.ys+subinfo.ym;
  upper.k = subinfo.zs+subinfo.zm;
  ierr    = DMDACreatePatchIS(dm,&lower,&upper,&odis);CHKERRQ(ierr);
  ierr    = DMDACreatePatchIS(subdm,&lower,&upper,&osis);CHKERRQ(ierr);

  /* global and subdomain ISes for the local indices of the subdomain */
  /* todo - make this not loop over at nonperiodic boundaries, which will be more involved */
  lower.i = subinfo.gxs;
  lower.j = subinfo.gys;
  lower.k = subinfo.gzs;
  upper.i = subinfo.gxs+subinfo.gxm;
  upper.j = subinfo.gys+subinfo.gym;
  upper.k = subinfo.gzs+subinfo.gzm;

  ierr = DMDACreatePatchIS(dm,&lower,&upper,&gdis);CHKERRQ(ierr);

  /* form the scatter */
  ierr = DMGetGlobalVector(dm,&dvec);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(subdm,&svec);CHKERRQ(ierr);
  ierr = DMGetLocalVector(subdm,&slvec);CHKERRQ(ierr);

  if (iscat) {ierr = VecScatterCreate(dvec,idis,svec,isis,&(*iscat)[0]);CHKERRQ(ierr);}
  if (oscat) {ierr = VecScatterCreate(dvec,odis,svec,osis,&(*oscat)[0]);CHKERRQ(ierr);}
  if (lscat) {ierr = VecScatterCreate(dvec,gdis,slvec,NULL,&(*lscat)[0]);CHKERRQ(ierr);}

  ierr = DMRestoreGlobalVector(dm,&dvec);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(subdm,&svec);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(subdm,&slvec);CHKERRQ(ierr);

  ierr = ISDestroy(&idis);CHKERRQ(ierr);
  ierr = ISDestroy(&isis);CHKERRQ(ierr);

  ierr = ISDestroy(&odis);CHKERRQ(ierr);
  ierr = ISDestroy(&osis);CHKERRQ(ierr);

  ierr = ISDestroy(&gdis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDASubDomainIS_Private"
/* We have that the interior regions are going to be the same, but the ghost regions might not match up

----------
----------
--++++++o=
--++++++o=
--++++++o=
--++++++o=
--ooooooo=
--========

Therefore, for each point in the overall, we must check if it's:

1. +: In the interior of the global dm; it lines up
2. o: In the overlap region -- for now the same as 1; no overlap
3. =: In the shared ghost region -- handled by DMCreateDomainDecompositionLocalScatter()
4. -: In the nonshared ghost region
 */

PetscErrorCode DMDASubDomainIS_Private(DM dm,DM subdm,IS *iis,IS *ois)
{
  PetscErrorCode ierr;
  DMDALocalInfo  info,subinfo;
  MatStencil     lower,upper;

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(subdm,&subinfo);CHKERRQ(ierr);

  /* create the inner IS */
  lower.i = info.xs;
  lower.j = info.ys;
  lower.k = info.zs;
  upper.i = info.xs+info.xm;
  upper.j = info.ys+info.ym;
  upper.k = info.zs+info.zm;

  ierr = DMDACreatePatchIS(dm,&lower,&upper,iis);CHKERRQ(ierr);

  /* create the outer IS */
  lower.i = subinfo.xs;
  lower.j = subinfo.ys;
  lower.k = subinfo.zs;
  upper.i = subinfo.xs+subinfo.xm;
  upper.j = subinfo.ys+subinfo.ym;
  upper.k = subinfo.zs+subinfo.zm;
  ierr    = DMDACreatePatchIS(dm,&lower,&upper,ois);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateDomainDecomposition_DA"
PetscErrorCode DMCreateDomainDecomposition_DA(DM dm,PetscInt *len,char ***names,IS **iis,IS **ois,DM **subdm)
{
  PetscErrorCode ierr;
  IS             iis0,ois0;
  DM             subdm0;

  PetscFunctionBegin;
  if (len) *len = 1;

  if (iis) {ierr = PetscMalloc(sizeof(IS*),iis);CHKERRQ(ierr);}
  if (ois) {ierr = PetscMalloc(sizeof(IS*),ois);CHKERRQ(ierr);}
  if (subdm) {ierr = PetscMalloc(sizeof(DM*),subdm);CHKERRQ(ierr);}
  if (names) {ierr = PetscMalloc(sizeof(char*),names);CHKERRQ(ierr);}
  ierr = DMDASubDomainDA_Private(dm,&subdm0);CHKERRQ(ierr);
  ierr = DMDASubDomainIS_Private(dm,subdm0,&iis0,&ois0);CHKERRQ(ierr);
  if (iis) (*iis)[0] = iis0;
  else {
    ierr = ISDestroy(&iis0);CHKERRQ(ierr);
  }
  if (ois) (*ois)[0] = ois0;
  else {
    ierr = ISDestroy(&ois0);CHKERRQ(ierr);
  }
  if (subdm) (*subdm)[0] = subdm0;
  if (names) (*names)[0] = 0;
  PetscFunctionReturn(0);
}
