/* Routines to visualize DMDAs and fields through GLVis */

#include <petsc/private/dmdaimpl.h>
#include <petsc/private/glvisviewerimpl.h>

typedef struct {
  PetscBool ll;
} DMDAGhostedGLVisViewerCtx;

static PetscErrorCode DMDAGhostedDestroyGLVisViewerCtx_Private(void **vctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(*vctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  Vec xlocal;
} DMDAFieldGLVisViewerCtx;

static PetscErrorCode DMDAFieldDestroyGLVisViewerCtx_Private(void *vctx)
{
  DMDAFieldGLVisViewerCtx *ctx = (DMDAFieldGLVisViewerCtx*)vctx;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&ctx->xlocal);CHKERRQ(ierr);
  ierr = PetscFree(vctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   dactx->ll is false -> all but the last proc per dimension claim the ghosted node on the right
   dactx->ll is true -> all but the first proc per dimension claim the ghosted node on the left
*/
static PetscErrorCode DMDAGetNumElementsGhosted(DM da, PetscInt *nex, PetscInt *ney, PetscInt *nez)
{
  DMDAGhostedGLVisViewerCtx *dactx;
  PetscInt                  sx,sy,sz,ien,jen,ken;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  /* Appease -Wmaybe-uninitialized */
  if (nex) *nex = -1;
  if (ney) *ney = -1;
  if (nez) *nez = -1;
  ierr = DMDAGetCorners(da,&sx,&sy,&sz,&ien,&jen,&ken);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(da,(void**)&dactx);CHKERRQ(ierr);
  if (dactx->ll) {
    PetscInt dim;

    ierr = DMGetDimension(da,&dim);CHKERRQ(ierr);
    if (!sx) ien--;
    if (!sy && dim > 1) jen--;
    if (!sz && dim > 2) ken--;
  } else {
    PetscInt M,N,P;

    ierr = DMDAGetInfo(da,NULL,&M,&N,&P,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    if (sx+ien == M) ien--;
    if (sy+jen == N) jen--;
    if (sz+ken == P) ken--;
  }
  if (nex) *nex = ien;
  if (ney) *ney = jen;
  if (nez) *nez = ken;
  PetscFunctionReturn(0);
}

/* inherits number of vertices from DMDAGetNumElementsGhosted */
static PetscErrorCode DMDAGetNumVerticesGhosted(DM da, PetscInt *nvx, PetscInt *nvy, PetscInt *nvz)
{
  PetscInt       ien = 0,jen = 0,ken = 0,dim;
  PetscInt       tote;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(da,&dim);CHKERRQ(ierr);
  ierr = DMDAGetNumElementsGhosted(da,&ien,&jen,&ken);CHKERRQ(ierr);
  tote = ien * (dim > 1 ? jen : 1) * (dim > 2 ? ken : 1);
  if (tote) {
    ien = ien+1;
    jen = dim > 1 ? jen+1 : 1;
    ken = dim > 2 ? ken+1 : 1;
  }
  if (nvx) *nvx = ien;
  if (nvy) *nvy = jen;
  if (nvz) *nvz = ken;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDASampleGLVisFields_Private(PetscObject oX, PetscInt nf, PetscObject oXf[], void *vctx)
{
  DM                        da;
  DMDAFieldGLVisViewerCtx   *ctx = (DMDAFieldGLVisViewerCtx*)vctx;
  DMDAGhostedGLVisViewerCtx *dactx;
  const PetscScalar         *array;
  PetscScalar               **arrayf;
  PetscInt                  i,f,ii,ien,jen,ken,ie,je,ke,bs,*bss;
  PetscInt                  sx,sy,sz,gsx,gsy,gsz,ist,jst,kst,gm,gn,gp;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(ctx->xlocal,&da);CHKERRQ(ierr);
  if (!da) SETERRQ(PetscObjectComm(oX),PETSC_ERR_ARG_WRONG,"Vector not generated from a DMDA");
  ierr = DMGetApplicationContext(da,(void**)&dactx);CHKERRQ(ierr);
  ierr = VecGetBlockSize(ctx->xlocal,&bs);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,(Vec)oX,INSERT_VALUES,ctx->xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,(Vec)oX,INSERT_VALUES,ctx->xlocal);CHKERRQ(ierr);
  ierr = DMDAGetNumVerticesGhosted(da,&ien,&jen,&ken);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gsx,&gsy,&gsz,&gm,&gn,&gp);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&sx,&sy,&sz,NULL,NULL,NULL);CHKERRQ(ierr);
  if (dactx->ll) {
    kst = jst = ist = 0;
  } else {
    kst  = gsz != sz ? 1 : 0;
    jst  = gsy != sy ? 1 : 0;
    ist  = gsx != sx ? 1 : 0;
  }
  ierr = PetscMalloc2(nf,&arrayf,nf,&bss);CHKERRQ(ierr);
  ierr = VecGetArrayRead(ctx->xlocal,&array);CHKERRQ(ierr);
  for (f=0;f<nf;f++) {
    ierr = VecGetBlockSize((Vec)oXf[f],&bss[f]);CHKERRQ(ierr);
    ierr = VecGetArray((Vec)oXf[f],&arrayf[f]);CHKERRQ(ierr);
  }
  for (ke = kst, ii = 0; ke < kst + ken; ke++) {
    for (je = jst; je < jst + jen; je++) {
      for (ie = ist; ie < ist + ien; ie++) {
        PetscInt cf,b;
        i = ke * gm * gn + je * gm + ie;
        for (f=0,cf=0;f<nf;f++)
          for (b=0;b<bss[f];b++)
            arrayf[f][bss[f]*ii+b] = array[i*bs+cf++];
        ii++;
      }
    }
  }
  for (f=0;f<nf;f++) { ierr = VecRestoreArray((Vec)oXf[f],&arrayf[f]);CHKERRQ(ierr); }
  ierr = VecRestoreArrayRead(ctx->xlocal,&array);CHKERRQ(ierr);
  ierr = PetscFree2(arrayf,bss);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMSetUpGLVisViewer_DMDA(PetscObject oda, PetscViewer viewer)
{
  DM             da = (DM)oda,daview;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery(oda,"GLVisGraphicsDMDAGhosted",(PetscObject*)&daview);CHKERRQ(ierr);
  if (!daview) {
    DMDAGhostedGLVisViewerCtx *dactx;
    DM                        dacoord = NULL;
    Vec                       xcoor,xcoorl;
    PetscBool                 hashocoord = PETSC_FALSE;
    const PetscInt            *lx,*ly,*lz;
    PetscInt                  dim,M,N,P,m,n,p,dof,s,i;

    ierr = PetscNew(&dactx);CHKERRQ(ierr);
    dactx->ll = PETSC_TRUE; /* default to match elements layout obtained by DMDAGetElements */
    ierr = PetscOptionsBegin(PetscObjectComm(oda),oda->prefix,"GLVis PetscViewer DMDA Options","PetscViewer");CHKERRQ(ierr);
    ierr = PetscOptionsBool("-viewer_glvis_dm_da_ll","Left-looking subdomain view",NULL,dactx->ll,&dactx->ll,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    /* Create a properly ghosted DMDA to visualize the mesh and the fields associated with */
    ierr = DMDAGetInfo(da,&dim,&M,&N,&P,&m,&n,&p,&dof,&s,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = DMDAGetOwnershipRanges(da,&lx,&ly,&lz);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)da,"_glvis_mesh_coords",(PetscObject*)&xcoor);CHKERRQ(ierr);
    if (!xcoor) {
      ierr = DMGetCoordinates(da,&xcoor);CHKERRQ(ierr);
    } else {
      hashocoord = PETSC_TRUE;
    }
    ierr = PetscInfo(da,"Creating auxilary DMDA for managing GLVis graphics\n");CHKERRQ(ierr);
    switch (dim) {
    case 1:
      ierr = DMDACreate1d(PetscObjectComm((PetscObject)da),DM_BOUNDARY_NONE,M,dof,1,lx,&daview);CHKERRQ(ierr);
      if (!hashocoord) {
        ierr = DMDACreate1d(PetscObjectComm((PetscObject)da),DM_BOUNDARY_NONE,M,1,1,lx,&dacoord);CHKERRQ(ierr);
      }
      break;
    case 2:
      ierr = DMDACreate2d(PetscObjectComm((PetscObject)da),DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,N,m,n,dof,1,lx,ly,&daview);CHKERRQ(ierr);
      if (!hashocoord) {
        ierr = DMDACreate2d(PetscObjectComm((PetscObject)da),DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,N,m,n,2,1,lx,ly,&dacoord);CHKERRQ(ierr);
      }
      break;
    case 3:
      ierr = DMDACreate3d(PetscObjectComm((PetscObject)da),DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,N,P,m,n,p,dof,1,lx,ly,lz,&daview);CHKERRQ(ierr);
      if (!hashocoord) {
        ierr = DMDACreate3d(PetscObjectComm((PetscObject)da),DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,N,P,m,n,p,3,1,lx,ly,lz,&dacoord);CHKERRQ(ierr);
      }
      break;
    default:
      SETERRQ1(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Unsupported dimension %D",dim);
      break;
    }
    ierr = DMSetApplicationContext(daview,dactx);CHKERRQ(ierr);
    ierr = DMSetApplicationContextDestroy(daview,DMDAGhostedDestroyGLVisViewerCtx_Private);CHKERRQ(ierr);
    ierr = DMSetUp(daview);CHKERRQ(ierr);
    if (!xcoor) {
      ierr = DMDASetUniformCoordinates(daview,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
      ierr = DMGetCoordinates(daview,&xcoor);CHKERRQ(ierr);
    }
    if (dacoord) {
      ierr = DMSetUp(dacoord);CHKERRQ(ierr);
      ierr = DMCreateLocalVector(dacoord,&xcoorl);CHKERRQ(ierr);
      ierr = DMGlobalToLocalBegin(dacoord,xcoor,INSERT_VALUES,xcoorl);CHKERRQ(ierr);
      ierr = DMGlobalToLocalEnd(dacoord,xcoor,INSERT_VALUES,xcoorl);CHKERRQ(ierr);
      ierr = DMDestroy(&dacoord);CHKERRQ(ierr);
    } else {
      PetscInt   ien,jen,ken,nc,nl,cdof,deg;
      char       fecmesh[64];
      const char *name;
      PetscBool  flg;

      ierr = DMDAGetNumElementsGhosted(daview,&ien,&jen,&ken);CHKERRQ(ierr);
      nc   = ien*(jen>0 ? jen : 1)*(ken>0 ? ken : 1);

      ierr = VecGetLocalSize(xcoor,&nl);CHKERRQ(ierr);
      if (nc && nl % nc) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Incompatible local coordinate size %D and number of cells %D",nl,nc);
      ierr = VecDuplicate(xcoor,&xcoorl);CHKERRQ(ierr);
      ierr = VecCopy(xcoor,xcoorl);CHKERRQ(ierr);
      ierr = VecSetDM(xcoorl,NULL);CHKERRQ(ierr);
      ierr = PetscObjectGetName((PetscObject)xcoor,&name);CHKERRQ(ierr);
      ierr = PetscStrbeginswith(name,"FiniteElementCollection:",&flg);CHKERRQ(ierr);
      if (!flg) {
        deg = 0;
        if (nc && nl) {
          cdof = nl/(nc*dim);
          deg  = 1;
          while (1) {
            PetscInt degd = 1;
            for (i=0;i<dim;i++) degd *= (deg+1);
            if (degd > cdof) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cell dofs %D",cdof);
            if (degd == cdof) break;
            deg++;
          }
        }
        ierr = PetscSNPrintf(fecmesh,sizeof(fecmesh),"FiniteElementCollection: L2_T1_%DD_P%D",dim,deg);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)xcoorl,fecmesh);CHKERRQ(ierr);
      } else {
        ierr = PetscObjectSetName((PetscObject)xcoorl,name);CHKERRQ(ierr);
      }
    }

    /* xcoorl is composed with the ghosted DMDA, the ghosted coordinate DMDA (if present) is only available through this vector */
    ierr = PetscObjectCompose((PetscObject)daview,"GLVisGraphicsCoordsGhosted",(PetscObject)xcoorl);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)xcoorl);CHKERRQ(ierr);

    /* daview is composed with the original DMDA */
    ierr = PetscObjectCompose(oda,"GLVisGraphicsDMDAGhosted",(PetscObject)daview);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)daview);CHKERRQ(ierr);
  }

  /* customize the viewer if present */
  if (viewer) {
    DMDAFieldGLVisViewerCtx   *ctx;
    DMDAGhostedGLVisViewerCtx *dactx;
    char                      fec[64];
    Vec                       xlocal,*Ufield;
    const char                **dafieldname;
    char                      **fec_type,**fieldname;
    PetscInt                  *nlocal,*bss,*dims;
    PetscInt                  dim,M,N,P,dof,s,i,nf;
    PetscBool                 bsset;

    ierr = DMDAGetInfo(daview,&dim,NULL,NULL,NULL,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = DMGetApplicationContext(daview,(void**)&dactx);CHKERRQ(ierr);
    ierr = DMCreateLocalVector(daview,&xlocal);CHKERRQ(ierr);
    ierr = DMDAGetFieldNames(da,(const char * const **)&dafieldname);CHKERRQ(ierr);
    ierr = DMDAGetNumVerticesGhosted(daview,&M,&N,&P);CHKERRQ(ierr);
    ierr = PetscSNPrintf(fec,sizeof(fec),"FiniteElementCollection: H1_%DD_P1",dim);CHKERRQ(ierr);
    ierr = PetscMalloc6(dof,&fec_type,dof,&nlocal,dof,&bss,dof,&dims,dof,&fieldname,dof,&Ufield);CHKERRQ(ierr);
    for (i=0;i<dof;i++) bss[i] = 1;
    nf = dof;

    ierr = PetscOptionsBegin(PetscObjectComm(oda),oda->prefix,"GLVis PetscViewer DMDA Field options","PetscViewer");CHKERRQ(ierr);
    ierr = PetscOptionsIntArray("-viewer_glvis_dm_da_bs","Block sizes for subfields; enable vector representation",NULL,bss,&nf,&bsset);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (bsset) {
      PetscInt t;
      for (i=0,t=0;i<nf;i++) t += bss[i];
      if (t != dof) SETERRQ2(PetscObjectComm(oda),PETSC_ERR_USER,"Sum of block sizes %D should equal %D",t,dof);
    } else nf = dof;

    for (i=0,s=0;i<nf;i++) {
      ierr = PetscStrallocpy(fec,&fec_type[i]);CHKERRQ(ierr);
      if (bss[i] == 1) {
        ierr = PetscStrallocpy(dafieldname[s],&fieldname[i]);CHKERRQ(ierr);
      } else {
        PetscInt b;
        size_t tlen = 9; /* "Vector-" + end */
        for (b=0;b<bss[i];b++) {
          size_t len;
          ierr = PetscStrlen(dafieldname[s+b],&len);CHKERRQ(ierr);
          tlen += len + 1; /* field + "-" */
        }
        ierr = PetscMalloc1(tlen,&fieldname[i]);CHKERRQ(ierr);
        ierr = PetscStrcpy(fieldname[i],"Vector-");CHKERRQ(ierr);
        for (b=0;b<bss[i]-1;b++) {
          ierr = PetscStrcat(fieldname[i],dafieldname[s+b]);CHKERRQ(ierr);
          ierr = PetscStrcat(fieldname[i],"-");CHKERRQ(ierr);
        }
        ierr = PetscStrcat(fieldname[i],dafieldname[s+b]);CHKERRQ(ierr);
      }
      dims[i] = dim;
      nlocal[i] = M*N*P*bss[i];
      s += bss[i];
    }

    /* the viewer context takes ownership of xlocal and destroys it in DMDAFieldDestroyGLVisViewerCtx_Private */
    ierr = PetscNew(&ctx);CHKERRQ(ierr);
    ctx->xlocal = xlocal;

    /* create work vectors */
    for (i=0;i<nf;i++) {
      ierr = VecCreateMPI(PetscObjectComm((PetscObject)da),nlocal[i],PETSC_DECIDE,&Ufield[i]);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)Ufield[i],fieldname[i]);CHKERRQ(ierr);
      ierr = VecSetBlockSize(Ufield[i],bss[i]);CHKERRQ(ierr);
      ierr = VecSetDM(Ufield[i],da);CHKERRQ(ierr);
    }

    ierr = PetscViewerGLVisSetFields(viewer,nf,(const char**)fec_type,dims,DMDASampleGLVisFields_Private,(PetscObject*)Ufield,ctx,DMDAFieldDestroyGLVisViewerCtx_Private);CHKERRQ(ierr);
    for (i=0;i<nf;i++) {
      ierr = PetscFree(fec_type[i]);CHKERRQ(ierr);
      ierr = PetscFree(fieldname[i]);CHKERRQ(ierr);
      ierr = VecDestroy(&Ufield[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree6(fec_type,nlocal,bss,dims,fieldname,Ufield);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDAView_GLVis_ASCII(DM dm, PetscViewer viewer)
{
  DM                da,cda;
  Vec               xcoorl;
  PetscMPIInt       size;
  const PetscScalar *array;
  PetscContainer    glvis_container;
  PetscInt          dim,sdim,i,vid[8],mid,cid,cdof;
  PetscInt          sx,sy,sz,ie,je,ke,ien,jen,ken,nel;
  PetscInt          gsx,gsy,gsz,gm,gn,gp,kst,jst,ist;
  PetscBool         enabled = PETSC_TRUE, isascii;
  PetscErrorCode    ierr;
  const char        *fmt;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Viewer must be of type VIEWERASCII");
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)viewer),&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Use single sequential viewers for parallel visualization");
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);

  /* get container: determines if a process visualizes is portion of the data or not */
  ierr = PetscObjectQuery((PetscObject)viewer,"_glvis_info_container",(PetscObject*)&glvis_container);CHKERRQ(ierr);
  if (!glvis_container) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Missing GLVis container");
  {
    PetscViewerGLVisInfo glvis_info;
    ierr = PetscContainerGetPointer(glvis_container,(void**)&glvis_info);CHKERRQ(ierr);
    enabled = glvis_info->enabled;
    fmt = glvis_info->fmt;
  }
  /* this can happen if we are calling DMView outside of VecView_GLVis */
  ierr = PetscObjectQuery((PetscObject)dm,"GLVisGraphicsDMDAGhosted",(PetscObject*)&da);CHKERRQ(ierr);
  if (!da) {ierr = DMSetUpGLVisViewer_DMDA((PetscObject)dm,NULL);CHKERRQ(ierr);}
  ierr = PetscObjectQuery((PetscObject)dm,"GLVisGraphicsDMDAGhosted",(PetscObject*)&da);CHKERRQ(ierr);
  if (!da) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Missing GLVis ghosted DMDA");
  ierr = DMGetCoordinateDim(da,&sdim);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"MFEM mesh v1.1\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\ndimension\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%D\n",dim);CHKERRQ(ierr);

  if (!enabled) {
    ierr = PetscViewerASCIIPrintf(viewer,"\nelements\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%D\n",0);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"\nboundary\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%D\n",0);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"\nvertices\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%D\n",0);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%D\n",sdim);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = DMDAGetNumElementsGhosted(da,&ien,&jen,&ken);CHKERRQ(ierr);
  nel  = ien;
  if (dim > 1) nel *= jen;
  if (dim > 2) nel *= ken;
  ierr = PetscViewerASCIIPrintf(viewer,"\nelements\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%D\n",nel);CHKERRQ(ierr);
  switch (dim) {
  case 1:
    for (ie = 0; ie < ien; ie++) {
      vid[0] = ie;
      vid[1] = ie+1;
      mid    = 1; /* material id */
      cid    = 1; /* segment */
      ierr   = PetscViewerASCIIPrintf(viewer,"%D %D %D %D\n",mid,cid,vid[0],vid[1]);CHKERRQ(ierr);
    }
    break;
  case 2:
    for (je = 0; je < jen; je++) {
      for (ie = 0; ie < ien; ie++) {
        vid[0] =     je*(ien+1) + ie;
        vid[1] =     je*(ien+1) + ie+1;
        vid[2] = (je+1)*(ien+1) + ie+1;
        vid[3] = (je+1)*(ien+1) + ie;
        mid    = 1; /* material id */
        cid    = 3; /* quad */
        ierr   = PetscViewerASCIIPrintf(viewer,"%D %D %D %D %D %D\n",mid,cid,vid[0],vid[1],vid[2],vid[3]);CHKERRQ(ierr);
      }
    }
    break;
  case 3:
    for (ke = 0; ke < ken; ke++) {
      for (je = 0; je < jen; je++) {
        for (ie = 0; ie < ien; ie++) {
          vid[0] =     ke*(jen+1)*(ien+1) +     je*(ien+1) + ie;
          vid[1] =     ke*(jen+1)*(ien+1) +     je*(ien+1) + ie+1;
          vid[2] =     ke*(jen+1)*(ien+1) + (je+1)*(ien+1) + ie+1;
          vid[3] =     ke*(jen+1)*(ien+1) + (je+1)*(ien+1) + ie;
          vid[4] = (ke+1)*(jen+1)*(ien+1) +     je*(ien+1) + ie;
          vid[5] = (ke+1)*(jen+1)*(ien+1) +     je*(ien+1) + ie+1;
          vid[6] = (ke+1)*(jen+1)*(ien+1) + (je+1)*(ien+1) + ie+1;
          vid[7] = (ke+1)*(jen+1)*(ien+1) + (je+1)*(ien+1) + ie;
          mid    = 1; /* material id */
          cid    = 5; /* hex */
          ierr   = PetscViewerASCIIPrintf(viewer,"%D %D %D %D %D %D %D %D %D %D\n",mid,cid,vid[0],vid[1],vid[2],vid[3],vid[4],vid[5],vid[6],vid[7]);CHKERRQ(ierr);
        }
      }
    }
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Unsupported dimension %D",dim);
    break;
  }
  ierr = PetscViewerASCIIPrintf(viewer,"\nboundary\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%D\n",0);CHKERRQ(ierr);

  /* vertex coordinates */
  ierr = PetscObjectQuery((PetscObject)da,"GLVisGraphicsCoordsGhosted",(PetscObject*)&xcoorl);CHKERRQ(ierr);
  if (!xcoorl) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Missing GLVis ghosted coords");
  ierr = DMDAGetNumVerticesGhosted(da,&ien,&jen,&ken);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\nvertices\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%D\n",ien*jen*ken);CHKERRQ(ierr);
  if (nel) {
    ierr = VecGetDM(xcoorl,&cda);CHKERRQ(ierr);
    ierr = VecGetArrayRead(xcoorl,&array);CHKERRQ(ierr);
    if (!cda) { /* HO viz */
      const char *fecname;
      PetscInt   nc,nl;

      ierr = PetscObjectGetName((PetscObject)xcoorl,&fecname);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"nodes\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"FiniteElementSpace\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%s\n",fecname);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"VDim: %D\n",sdim);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Ordering: 1\n\n");CHKERRQ(ierr); /*Ordering::byVDIM*/
      /* L2 coordinates */
      ierr = DMDAGetNumElementsGhosted(da,&ien,&jen,&ken);CHKERRQ(ierr);
      ierr = VecGetLocalSize(xcoorl,&nl);CHKERRQ(ierr);
      nc   = ien*(jen>0 ? jen : 1)*(ken>0 ? ken : 1);
      cdof = nc ? nl/nc : 0;
      if (!ien) ien++;
      if (!jen) jen++;
      if (!ken) ken++;
      ist = jst = kst = 0;
      gm = ien;
      gn = jen;
      gp = ken;
    } else {
      DMDAGhostedGLVisViewerCtx *dactx;

      ierr = DMGetApplicationContext(da,(void**)&dactx);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%D\n",sdim);CHKERRQ(ierr);
      cdof = sdim;
      ierr = DMDAGetCorners(da,&sx,&sy,&sz,NULL,NULL,NULL);CHKERRQ(ierr);
      ierr = DMDAGetGhostCorners(da,&gsx,&gsy,&gsz,&gm,&gn,&gp);CHKERRQ(ierr);
      if (dactx->ll) {
        kst = jst = ist = 0;
      } else {
        kst  = gsz != sz ? 1 : 0;
        jst  = gsy != sy ? 1 : 0;
        ist  = gsx != sx ? 1 : 0;
      }
    }
    for (ke = kst; ke < kst + ken; ke++) {
      for (je = jst; je < jst + jen; je++) {
        for (ie = ist; ie < ist + ien; ie++) {
          PetscInt c;

          i = ke * gm * gn + je * gm + ie;
          for (c=0;c<cdof/sdim;c++) {
            PetscInt d;
            for (d=0;d<sdim;d++) {
              ierr = PetscViewerASCIIPrintf(viewer,fmt,PetscRealPart(array[cdof*i+c*sdim+d]));CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
          }
        }
      }
    }
    ierr = VecRestoreArrayRead(xcoorl,&array);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMView_DA_GLVis(DM dm, PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMView_GLVis(dm,viewer,DMDAView_GLVis_ASCII);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
