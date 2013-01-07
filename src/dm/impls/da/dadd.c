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
  ierr = DMDAGetOffset(da,&ox,&oy,&oz);CHKERRQ(ierr);
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
PetscErrorCode DMDASubDomainDA_Private(DM dm, DM *dddm) {
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
  ierr = DMDASetOffset(da,xo,yo,zo);CHKERRQ(ierr);


  /* todo - nonuniform coordinates */
  ierr = DMDAGetLocalBoundingBox(dm,lmin,lmax);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,lmin[0],lmax[0],lmin[1],lmax[1],lmin[2],lmax[2]);CHKERRQ(ierr);

  dd = (DM_DA *)da->data;
  dd->Mo = info.mx;
  dd->No = info.my;
  dd->Po = info.mz;

  *dddm = da;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateDomainDecompositionScatters_DA"
/*
 Fills the local vector problem on the subdomain from the global problem.

 */
PetscErrorCode DMCreateDomainDecompositionScatters_DA(DM dm,PetscInt nsubdms,DM *subdms,VecScatter **iscat,VecScatter **oscat, VecScatter **lscat) {
  PetscErrorCode   ierr;
  DMDALocalInfo    dinfo,sinfo;
  IS               isis,idis,osis,odis,gsis,gdis;
  PetscInt         *ididx,*isidx,*odidx,*osidx,*gdidx,*gsidx,*idx_global,n_global,*idx_sub,n_sub;
  PetscInt         l,i,j,k,d,n_i,n_o,n_g,sl,dl,di,dj,dk,si,sj,sk;
  Vec              dvec,svec,slvec;
  DM               subdm;

  PetscFunctionBegin;

  /* allocate the arrays of scatters */
  if (iscat) {ierr = PetscMalloc(sizeof(VecScatter *),iscat);CHKERRQ(ierr);}
  if (oscat) {ierr = PetscMalloc(sizeof(VecScatter *),oscat);CHKERRQ(ierr);}
  if (lscat) {ierr = PetscMalloc(sizeof(VecScatter *),lscat);CHKERRQ(ierr);}

  ierr = DMDAGetLocalInfo(dm,&dinfo);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(dm,&n_global,&idx_global);CHKERRQ(ierr);
  for (l = 0;l < nsubdms;l++) {
    n_i = 0;
    n_o = 0;
    n_g = 0;
    subdm = subdms[l];
    ierr = DMDAGetLocalInfo(subdm,&sinfo);CHKERRQ(ierr);
    ierr = DMDAGetGlobalIndices(subdm,&n_sub,&idx_sub);CHKERRQ(ierr);
    /* count the three region sizes */
    for (k=sinfo.gzs;k<sinfo.gzs+sinfo.gzm;k++) {
      for (j=sinfo.gys;j<sinfo.gys+sinfo.gym;j++) {
        for (i=sinfo.gxs;i<sinfo.gxs+sinfo.gxm;i++) {
          for (d=0;d<sinfo.dof;d++) {
            if (k >= sinfo.zs           && j >= sinfo.ys         && i >= sinfo.xs &&
                k <  sinfo.zs+sinfo.zm  && j < sinfo.ys+sinfo.ym && i < sinfo.xs+sinfo.xm) {

              /* interior - subinterior overlap */
              if (k >= dinfo.zs            && j >= dinfo.ys          && i >= dinfo.xs &&
                  k <  dinfo.zs+dinfo.zm  && j < dinfo.ys+dinfo.ym && i < dinfo.xs+dinfo.xm) {
                n_i++;
              }
              /* ghost - subinterior overlap */
              if (k >= dinfo.gzs            && j >= dinfo.gys          && i >= dinfo.gxs &&
                  k <  dinfo.gzs+dinfo.gzm  && j < dinfo.gys+dinfo.gym && i < dinfo.gxs+dinfo.gxm) {
                n_o++;
              }
            }

            /* ghost - subghost overlap */
            if (k >= dinfo.gzs            && j >= dinfo.gys          && i >= dinfo.gxs &&
                k <  dinfo.gzs+dinfo.gzm  && j < dinfo.gys+dinfo.gym && i < dinfo.gxs+dinfo.gxm) {
              n_g++;
            }
          }
        }
      }
    }

    if (n_g == 0) SETERRQ(((PetscObject)subdm)->comm,PETSC_ERR_ARG_WRONGSTATE,"Processor-local domain and subdomain do not intersect!");

    /* local and subdomain local index set indices */
    ierr = PetscMalloc(sizeof(PetscInt)*n_i,&ididx);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscInt)*n_i,&isidx);CHKERRQ(ierr);

    ierr = PetscMalloc(sizeof(PetscInt)*n_o,&odidx);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscInt)*n_o,&osidx);CHKERRQ(ierr);

    ierr = PetscMalloc(sizeof(PetscInt)*n_g,&gdidx);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscInt)*n_g,&gsidx);CHKERRQ(ierr);

    n_i = 0; n_o = 0;n_g = 0;
    for (k=sinfo.gzs;k<sinfo.gzs+sinfo.gzm;k++) {
      for (j=sinfo.gys;j<sinfo.gys+sinfo.gym;j++) {
        for (i=sinfo.gxs;i<sinfo.gxs+sinfo.gxm;i++) {
          for (d=0;d<sinfo.dof;d++) {
            si = i - sinfo.gxs;
            sj = j - sinfo.gys;
            sk = k - sinfo.gzs;
            sl = d + sinfo.dof*(si + sinfo.gxm*(sj + sinfo.gym*sk));
            di = i - dinfo.gxs;
            dj = j - dinfo.gys;
            dk = k - dinfo.gzs;
            dl = d + dinfo.dof*(di + dinfo.gxm*(dj + dinfo.gym*dk));

            if (k >= sinfo.zs           && j >= sinfo.ys         && i >= sinfo.xs &&
                k <  sinfo.zs+sinfo.zm  && j < sinfo.ys+sinfo.ym && i < sinfo.xs+sinfo.xm) {

              /* interior - subinterior overlap */
              if (k >= dinfo.zs            && j >= dinfo.ys          && i >= dinfo.xs &&
                  k <  dinfo.zs+dinfo.zm  && j < dinfo.ys+dinfo.ym && i < dinfo.xs+dinfo.xm) {
                ididx[n_i] = idx_global[dl];
                isidx[n_i] = idx_sub[sl];
                n_i++;
              }
              /* ghost - subinterior overlap */
              if (k >= dinfo.gzs            && j >= dinfo.gys          && i >= dinfo.gxs &&
                  k <  dinfo.gzs+dinfo.gzm  && j < dinfo.gys+dinfo.gym && i < dinfo.gxs+dinfo.gxm) {
                odidx[n_o] = idx_global[dl];
                osidx[n_o] = idx_sub[sl];
                n_o++;
              }
            }

            /* ghost - subghost overlap */
            if (k >= dinfo.gzs            && j >= dinfo.gys          && i >= dinfo.gxs &&
                k <  dinfo.gzs+dinfo.gzm  && j < dinfo.gys+dinfo.gym && i < dinfo.gxs+dinfo.gxm) {
              gdidx[n_g] = idx_global[dl];
              gsidx[n_g] = sl;
              n_g++;
            }
          }
        }
      }
    }

    ierr = ISCreateGeneral(PETSC_COMM_SELF,n_i,ididx,PETSC_OWN_POINTER,&idis);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,n_i,isidx,PETSC_OWN_POINTER,&isis);CHKERRQ(ierr);

    ierr = ISCreateGeneral(PETSC_COMM_SELF,n_o,odidx,PETSC_OWN_POINTER,&odis);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,n_o,osidx,PETSC_OWN_POINTER,&osis);CHKERRQ(ierr);

    ierr = ISCreateGeneral(PETSC_COMM_SELF,n_g,gdidx,PETSC_OWN_POINTER,&gdis);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,n_g,gsidx,PETSC_OWN_POINTER,&gsis);CHKERRQ(ierr);

    /* form the scatter */
    ierr = DMGetGlobalVector(dm,&dvec);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(subdm,&svec);CHKERRQ(ierr);
    ierr = DMGetLocalVector(subdm,&slvec);CHKERRQ(ierr);

    if (iscat) {ierr = VecScatterCreate(dvec,idis,svec,isis,&(*iscat)[l]);CHKERRQ(ierr);}
    if (oscat) {ierr = VecScatterCreate(dvec,odis,svec,osis,&(*oscat)[l]);CHKERRQ(ierr);}
    if (lscat) {ierr = VecScatterCreate(dvec,gdis,slvec,gsis,&(*lscat)[l]);CHKERRQ(ierr);}

    ierr = DMRestoreGlobalVector(dm,&dvec);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(subdm,&svec);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(subdm,&slvec);CHKERRQ(ierr);

    ierr = ISDestroy(&idis);CHKERRQ(ierr);
    ierr = ISDestroy(&isis);CHKERRQ(ierr);

    ierr = ISDestroy(&odis);CHKERRQ(ierr);
    ierr = ISDestroy(&osis);CHKERRQ(ierr);

    ierr = ISDestroy(&gdis);CHKERRQ(ierr);
    ierr = ISDestroy(&gsis);CHKERRQ(ierr);
  }
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

PetscErrorCode DMDASubDomainIS_Private(DM dm,DM subdm,IS *iis,IS *ois) {
  PetscErrorCode   ierr;
  DMDALocalInfo    info,subinfo;
  PetscInt         *iiidx,*oiidx,*gidx,gindx;
  PetscInt         i,j,k,d,n,nsub,nover,llindx,lindx,li,lj,lk,gi,gj,gk;

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(subdm,&subinfo);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(dm,&n,&gidx);CHKERRQ(ierr);
  /* todo -- overlap */
  nsub = info.xm*info.ym*info.zm*info.dof;
  nover = subinfo.xm*subinfo.ym*subinfo.zm*subinfo.dof;
  /* iis is going to have size of the local problem's global part but have a lot of fill-in */
  ierr = PetscMalloc(sizeof(PetscInt)*nsub,&iiidx);CHKERRQ(ierr);
  /* ois is going to have size of the local problem's global part */
  ierr = PetscMalloc(sizeof(PetscInt)*nover,&oiidx);CHKERRQ(ierr);
  /* loop over the ghost region of the subdm and fill in the indices */
  for (k=subinfo.gzs;k<subinfo.gzs+subinfo.gzm;k++) {
    for (j=subinfo.gys;j<subinfo.gys+subinfo.gym;j++) {
      for (i=subinfo.gxs;i<subinfo.gxs+subinfo.gxm;i++) {
        for (d=0;d<subinfo.dof;d++) {
          li = i - subinfo.xs;
          lj = j - subinfo.ys;
          lk = k - subinfo.zs;
          lindx = d + subinfo.dof*(li + subinfo.xm*(lj + subinfo.ym*lk));
          li = i - info.xs;
          lj = j - info.ys;
          lk = k - info.zs;
          llindx = d + info.dof*(li + info.xm*(lj + info.ym*lk));
          gi = i - info.gxs;
          gj = j - info.gys;
          gk = k - info.gzs;
          gindx = d + info.dof*(gi + info.gxm*(gj + info.gym*gk));

          /* check if the current point is inside the interior region */
          if (k >= info.zs          && j >= info.ys          && i >= info.xs &&
              k <  info.zs+info.zm  && j < info.ys+info.ym   && i < info.xs+info.xm) {
            iiidx[llindx] = gidx[gindx];
            oiidx[lindx] = gidx[gindx];
            /* overlap region */
          } else if (k >= subinfo.zs             && j >= subinfo.ys                && i >= subinfo.xs &&
                     k <  subinfo.zs+subinfo.zm  && j < subinfo.ys+subinfo.ym   && i < subinfo.xs+subinfo.xm) {
            oiidx[lindx] = gidx[gindx];
          }
        }
      }
    }
  }

  /* create the index sets */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nsub,iiidx,PETSC_OWN_POINTER,iis);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nover,oiidx,PETSC_OWN_POINTER,ois);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateDomainDecomposition_DA"
PetscErrorCode DMCreateDomainDecomposition_DA(DM dm,PetscInt *len,char ***names,IS **iis,IS **ois,DM **subdm) {
  PetscErrorCode ierr;
  IS             iis0,ois0;
  DM             subdm0;
  PetscFunctionBegin;
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
