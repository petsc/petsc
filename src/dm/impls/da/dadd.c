#include <petsc-private/daimpl.h>

#undef __FUNCT__
#define __FUNCT__ "DMDACreatePatchIS"

PetscErrorCode DMDACreatePatchIS(DM da,MatStencil *lower,MatStencil *upper,IS *is)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,idx;
  PetscInt       ii,jj,kk;
  Vec            v;
  PetscInt       n,pn,bs;
  PetscMPIInt    rank;
  PetscSF        sf,psf;
  PetscLayout    map;
  MPI_Comm       comm;
  PetscInt       *natidx,*globidx,*leafidx;
  PetscInt       *pnatidx,*pleafidx;
  PetscInt       base;
  PetscInt       ox,oy,oz;
  DM_DA          *dd;

  PetscFunctionBegin;
  comm = ((PetscObject)da)->comm;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  dd = (DM_DA *)da->data;

  /* construct the natural mapping */
  ierr = DMGetGlobalVector(da,&v);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(v,&base,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecGetBlockSize(v,&bs);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&v);CHKERRQ(ierr);

  /* construct the layout */
  ierr = PetscLayoutCreate(comm,&map);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(map,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(map,n);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(map);CHKERRQ(ierr);

  /* construct the list of natural indices on this process when PETSc ordering is considered */
  ierr = DMDAGetOffset(da,&ox,&oy,&oz,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*n,&natidx);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*n,&globidx);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*n,&leafidx);CHKERRQ(ierr);
  idx = 0;
  for (k=dd->zs;k<dd->ze;k++) {
    for (j=dd->ys;j<dd->ye;j++) {
      for (i=dd->xs;i<dd->xe;i++) {
        natidx[idx] = i + dd->w*(j*dd->M + k*dd->M*dd->N);
        globidx[idx] = base + idx;
        leafidx[idx] = 0;
        idx++;
      }
    }
  }

  if (idx != n) SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE, "for some reason the count is wrong.");

  /* construct the SF going from the natural indices to the local set of PETSc indices */
  ierr = PetscSFCreate(comm,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraphLayout(sf,map,n,PETSC_NULL,PETSC_OWN_POINTER,natidx);CHKERRQ(ierr);

  /* broadcast the global indices over to the corresponding natural indices */
  ierr = PetscSFGatherBegin(sf,MPIU_INT,globidx,leafidx);CHKERRQ(ierr);
  ierr = PetscSFGatherEnd(sf,MPIU_INT,globidx,leafidx);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);


  pn = dd->w*(upper->k - lower->k)*(upper->j - lower->j)*(upper->i - lower->i);
  ierr = PetscMalloc(sizeof(PetscInt)*pn,&pnatidx);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*pn,&pleafidx);CHKERRQ(ierr);
  idx = 0;
  for (k=lower->k-oz;k<upper->k-oz;k++) {
    for (j=lower->j-oy;j<upper->j-oy;j++) {
      for (i=dd->w*(lower->i-ox);i<dd->w*(upper->i-ox);i++) {
        ii = i % (dd->w*dd->M);
        jj = j % dd->N;
        kk = k % dd->P;
        if (ii < 0) ii = dd->w*dd->M + ii;
        if (jj < 0) jj = dd->N + jj;
        if (kk < 0) kk = dd->P + kk;
        pnatidx[idx] = ii + dd->w*(jj*dd->M + kk*dd->M*dd->N);
        idx++;
      }
    }
  }

  if (idx != pn) SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE, "for some reason the count is wrong");

  ierr = PetscSFCreate(comm,&psf);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(psf);CHKERRQ(ierr);
  ierr = PetscSFSetGraphLayout(psf,map,pn,PETSC_NULL,PETSC_OWN_POINTER,pnatidx);CHKERRQ(ierr);

  /* broadcast the global indices through to the patch */
  ierr = PetscSFBcastBegin(psf,MPIU_INT,leafidx,pleafidx);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(psf,MPIU_INT,leafidx,pleafidx);CHKERRQ(ierr);

  /* create the IS */
  ierr = ISCreateGeneral(comm,pn,pleafidx,PETSC_OWN_POINTER,is);CHKERRQ(ierr);

  ierr = PetscSFDestroy(&psf);CHKERRQ(ierr);

  ierr = PetscLayoutDestroy(&map);CHKERRQ(ierr);

  ierr = PetscFree(globidx);CHKERRQ(ierr);
  ierr = PetscFree(leafidx);CHKERRQ(ierr);
  ierr = PetscFree(natidx);CHKERRQ(ierr);
  ierr = PetscFree(pnatidx);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDASubDomainDA_Private"
PetscErrorCode DMDASubDomainDA_Private(DM dm, DM *dddm)
{
  DM               da;
  DM_DA            *dd;
  PetscErrorCode   ierr;
  DMDALocalInfo    info;
  PetscReal        lmin[3],lmax[3];
  PetscInt         xsize,ysize,zsize;
  PetscInt         xo,yo,zo;

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  ierr = DMDACreate(PETSC_COMM_SELF,&da);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(da,"sub_");CHKERRQ(ierr);

  ierr = DMDASetDim(da, info.dim);CHKERRQ(ierr);
  ierr = DMDASetDof(da, info.dof);CHKERRQ(ierr);

  ierr = DMDASetStencilType(da,info.st);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(da,info.sw);CHKERRQ(ierr);

  dd = (DM_DA *)dm->data;

  /* local with overlap */
  xsize = info.xm;
  ysize = info.ym;
  zsize = info.zm;
  xo    = info.xs;
  yo    = info.ys;
  zo    = info.zs;
  if (info.bx == DMDA_BOUNDARY_PERIODIC || (info.xs != 0)) {
    xsize += dd->overlap;
    xo -= dd->overlap;
  }
  if (info.by == DMDA_BOUNDARY_PERIODIC || (info.ys != 0)) {
    ysize += dd->overlap;
    yo    -= dd->overlap;
  }
  if (info.bz == DMDA_BOUNDARY_PERIODIC || (info.zs != 0)) {
    zsize += dd->overlap;
    zo    -= dd->overlap;
  }

  if (info.bx == DMDA_BOUNDARY_PERIODIC || (info.xs+info.xm != info.mx)) {
    xsize += dd->overlap;
  }
  if (info.by == DMDA_BOUNDARY_PERIODIC || (info.ys+info.ym != info.my)) {
    ysize += dd->overlap;
  }
  if (info.bz == DMDA_BOUNDARY_PERIODIC || (info.zs+info.zm != info.mz)) {
    zsize += dd->overlap;
  }
  ierr = DMDASetSizes(da, xsize, ysize, zsize);CHKERRQ(ierr);
  ierr = DMDASetNumProcs(da, 1, 1, 1);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(da, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED);CHKERRQ(ierr);

  /* set up as a block instead */
  ierr = DMSetUp(da);CHKERRQ(ierr);

  /* todo - nonuniform coordinates */
  ierr = DMDAGetLocalBoundingBox(dm,lmin,lmax);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,lmin[0],lmax[0],lmin[1],lmax[1],lmin[2],lmax[2]);CHKERRQ(ierr);

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
  PetscErrorCode   ierr;
  DMDALocalInfo    info,subinfo;
  DM               subdm;
  MatStencil       upper,lower;
  IS               idis,isis,odis,osis,gdis;
  Vec              svec,dvec,slvec;

  PetscFunctionBegin;
  if (nsubdms != 1) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Cannot have more than one subdomain per processor (yet)");

  /* allocate the arrays of scatters */
  if (iscat) {ierr = PetscMalloc(sizeof(VecScatter *),iscat);CHKERRQ(ierr);}
  if (oscat) {ierr = PetscMalloc(sizeof(VecScatter *),oscat);CHKERRQ(ierr);}
  if (lscat) {ierr = PetscMalloc(sizeof(VecScatter *),lscat);CHKERRQ(ierr);}

  ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  subdm = subdms[0];
  ierr = DMDAGetLocalInfo(subdm,&subinfo);CHKERRQ(ierr);

  /* create the global and subdomain index sets for the inner domain */
  /* TODO - make this actually support multiple subdomains -- subdomain needs to provide where it's nonoverlapping portion belongs */
  lower.i = info.xs;
  lower.j = info.ys;
  lower.k = info.zs;
  upper.i = info.xs+info.xm;
  upper.j = info.ys+info.ym;
  upper.k = info.zs+info.zm;
  ierr = DMDACreatePatchIS(dm,&lower,&upper,&idis);CHKERRQ(ierr);
  ierr = DMDACreatePatchIS(subdm,&lower,&upper,&isis);CHKERRQ(ierr);

  /* create the global and subdomain index sets for the outer subdomain */
  lower.i = subinfo.xs;
  lower.j = subinfo.ys;
  lower.k = subinfo.zs;
  upper.i = subinfo.xs+subinfo.xm;
  upper.j = subinfo.ys+subinfo.ym;
  upper.k = subinfo.zs+subinfo.zm;
  ierr = DMDACreatePatchIS(dm,&lower,&upper,&odis);CHKERRQ(ierr);
  ierr = DMDACreatePatchIS(subdm,&lower,&upper,&osis);CHKERRQ(ierr);

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
  if (lscat) {ierr = VecScatterCreate(dvec,gdis,slvec,PETSC_NULL,&(*lscat)[0]);CHKERRQ(ierr);}

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
  PetscErrorCode   ierr;
  DMDALocalInfo    info,subinfo;
  MatStencil       lower,upper;

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
  ierr = DMDACreatePatchIS(dm,&lower,&upper,ois);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateDomainDecomposition_DA"
PetscErrorCode DMCreateDomainDecomposition_DA(DM dm,PetscInt *len,char ***names,IS **iis,IS **ois,DM **subdm)
{
  PetscErrorCode ierr;
  IS             iis0,ois0;
  DM             subdm0;
  DM_DA          *dd = (DM_DA*)dm;

  PetscFunctionBegin;
  /* fix to enable PCASM default behavior as taking overlap from the matrix */
  if (!dd->decompositiondm) {
    if (len)*len=0;
    if (names)*names=0;
    if (iis)*iis=0;
    if (ois)*ois=0;
    if (subdm)*subdm=0;
    PetscFunctionReturn(0);
  }

  if (len)*len = 1;

  if (iis) {ierr = PetscMalloc(sizeof(IS *),iis);CHKERRQ(ierr);}
  if (ois) {ierr = PetscMalloc(sizeof(IS *),ois);CHKERRQ(ierr);}
  if (subdm) {ierr = PetscMalloc(sizeof(DM *),subdm);CHKERRQ(ierr);}
  if (names) {ierr = PetscMalloc(sizeof(char *),names);CHKERRQ(ierr);}
  ierr = DMDASubDomainDA_Private(dm,&subdm0);CHKERRQ(ierr);
  ierr = DMDASubDomainIS_Private(dm,subdm0,&iis0,&ois0);CHKERRQ(ierr);
  if (iis) {
    (*iis)[0] = iis0;
  } else {
    ierr = ISDestroy(&iis0);CHKERRQ(ierr);
  }
  if (ois) {
    (*ois)[0] = ois0;
  } else {
    ierr = ISDestroy(&ois0);CHKERRQ(ierr);
  }
  if (subdm) (*subdm)[0] = subdm0;
  if (names) (*names)[0] = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateDomainDecompositionDM_DA"
PetscErrorCode DMCreateDomainDecompositionDM_DA(DM dm,const char *name,DM *ddm)
{
  DM_DA          *dd = (DM_DA*)dm;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrcmp(name,"default",&flg);CHKERRQ(ierr);
  if (flg) {
    dd->decompositiondm = PETSC_TRUE;
    *ddm = dm;
  } else {
    dd->decompositiondm = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}
