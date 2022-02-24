/* Routines to visualize DMDAs and fields through GLVis */

#include <petsc/private/dmdaimpl.h>
#include <petsc/private/glvisviewerimpl.h>

typedef struct {
  PetscBool ll;
} DMDAGhostedGLVisViewerCtx;

static PetscErrorCode DMDAGhostedDestroyGLVisViewerCtx_Private(void **vctx)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(*vctx));
  PetscFunctionReturn(0);
}

typedef struct {
  Vec xlocal;
} DMDAFieldGLVisViewerCtx;

static PetscErrorCode DMDAFieldDestroyGLVisViewerCtx_Private(void *vctx)
{
  DMDAFieldGLVisViewerCtx *ctx = (DMDAFieldGLVisViewerCtx*)vctx;

  PetscFunctionBegin;
  CHKERRQ(VecDestroy(&ctx->xlocal));
  CHKERRQ(PetscFree(vctx));
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

  PetscFunctionBegin;
  /* Appease -Wmaybe-uninitialized */
  if (nex) *nex = -1;
  if (ney) *ney = -1;
  if (nez) *nez = -1;
  CHKERRQ(DMDAGetCorners(da,&sx,&sy,&sz,&ien,&jen,&ken));
  CHKERRQ(DMGetApplicationContext(da,&dactx));
  if (dactx->ll) {
    PetscInt dim;

    CHKERRQ(DMGetDimension(da,&dim));
    if (!sx) ien--;
    if (!sy && dim > 1) jen--;
    if (!sz && dim > 2) ken--;
  } else {
    PetscInt M,N,P;

    CHKERRQ(DMDAGetInfo(da,NULL,&M,&N,&P,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
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

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(da,&dim));
  CHKERRQ(DMDAGetNumElementsGhosted(da,&ien,&jen,&ken));
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

  PetscFunctionBegin;
  CHKERRQ(VecGetDM(ctx->xlocal,&da));
  PetscCheck(da,PetscObjectComm(oX),PETSC_ERR_ARG_WRONG,"Vector not generated from a DMDA");
  CHKERRQ(DMGetApplicationContext(da,&dactx));
  CHKERRQ(VecGetBlockSize(ctx->xlocal,&bs));
  CHKERRQ(DMGlobalToLocalBegin(da,(Vec)oX,INSERT_VALUES,ctx->xlocal));
  CHKERRQ(DMGlobalToLocalEnd(da,(Vec)oX,INSERT_VALUES,ctx->xlocal));
  CHKERRQ(DMDAGetNumVerticesGhosted(da,&ien,&jen,&ken));
  CHKERRQ(DMDAGetGhostCorners(da,&gsx,&gsy,&gsz,&gm,&gn,&gp));
  CHKERRQ(DMDAGetCorners(da,&sx,&sy,&sz,NULL,NULL,NULL));
  if (dactx->ll) {
    kst = jst = ist = 0;
  } else {
    kst  = gsz != sz ? 1 : 0;
    jst  = gsy != sy ? 1 : 0;
    ist  = gsx != sx ? 1 : 0;
  }
  CHKERRQ(PetscMalloc2(nf,&arrayf,nf,&bss));
  CHKERRQ(VecGetArrayRead(ctx->xlocal,&array));
  for (f=0;f<nf;f++) {
    CHKERRQ(VecGetBlockSize((Vec)oXf[f],&bss[f]));
    CHKERRQ(VecGetArray((Vec)oXf[f],&arrayf[f]));
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
  for (f=0;f<nf;f++) CHKERRQ(VecRestoreArray((Vec)oXf[f],&arrayf[f]));
  CHKERRQ(VecRestoreArrayRead(ctx->xlocal,&array));
  CHKERRQ(PetscFree2(arrayf,bss));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMSetUpGLVisViewer_DMDA(PetscObject oda, PetscViewer viewer)
{
  DM             da = (DM)oda,daview;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectQuery(oda,"GLVisGraphicsDMDAGhosted",(PetscObject*)&daview));
  if (!daview) {
    DMDAGhostedGLVisViewerCtx *dactx;
    DM                        dacoord = NULL;
    Vec                       xcoor,xcoorl;
    PetscBool                 hashocoord = PETSC_FALSE;
    const PetscInt            *lx,*ly,*lz;
    PetscInt                  dim,M,N,P,m,n,p,dof,s,i;

    CHKERRQ(PetscNew(&dactx));
    dactx->ll = PETSC_TRUE; /* default to match elements layout obtained by DMDAGetElements */
    ierr = PetscOptionsBegin(PetscObjectComm(oda),oda->prefix,"GLVis PetscViewer DMDA Options","PetscViewer");CHKERRQ(ierr);
    CHKERRQ(PetscOptionsBool("-viewer_glvis_dm_da_ll","Left-looking subdomain view",NULL,dactx->ll,&dactx->ll,NULL));
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    /* Create a properly ghosted DMDA to visualize the mesh and the fields associated with */
    CHKERRQ(DMDAGetInfo(da,&dim,&M,&N,&P,&m,&n,&p,&dof,&s,NULL,NULL,NULL,NULL));
    CHKERRQ(DMDAGetOwnershipRanges(da,&lx,&ly,&lz));
    CHKERRQ(PetscObjectQuery((PetscObject)da,"_glvis_mesh_coords",(PetscObject*)&xcoor));
    if (!xcoor) {
      CHKERRQ(DMGetCoordinates(da,&xcoor));
    } else {
      hashocoord = PETSC_TRUE;
    }
    CHKERRQ(PetscInfo(da,"Creating auxilary DMDA for managing GLVis graphics\n"));
    switch (dim) {
    case 1:
      CHKERRQ(DMDACreate1d(PetscObjectComm((PetscObject)da),DM_BOUNDARY_NONE,M,dof,1,lx,&daview));
      if (!hashocoord) {
        CHKERRQ(DMDACreate1d(PetscObjectComm((PetscObject)da),DM_BOUNDARY_NONE,M,1,1,lx,&dacoord));
      }
      break;
    case 2:
      CHKERRQ(DMDACreate2d(PetscObjectComm((PetscObject)da),DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,N,m,n,dof,1,lx,ly,&daview));
      if (!hashocoord) {
        CHKERRQ(DMDACreate2d(PetscObjectComm((PetscObject)da),DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,N,m,n,2,1,lx,ly,&dacoord));
      }
      break;
    case 3:
      CHKERRQ(DMDACreate3d(PetscObjectComm((PetscObject)da),DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,N,P,m,n,p,dof,1,lx,ly,lz,&daview));
      if (!hashocoord) {
        CHKERRQ(DMDACreate3d(PetscObjectComm((PetscObject)da),DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,N,P,m,n,p,3,1,lx,ly,lz,&dacoord));
      }
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Unsupported dimension %D",dim);
    }
    CHKERRQ(DMSetApplicationContext(daview,dactx));
    CHKERRQ(DMSetApplicationContextDestroy(daview,DMDAGhostedDestroyGLVisViewerCtx_Private));
    CHKERRQ(DMSetUp(daview));
    if (!xcoor) {
      CHKERRQ(DMDASetUniformCoordinates(daview,0.0,1.0,0.0,1.0,0.0,1.0));
      CHKERRQ(DMGetCoordinates(daview,&xcoor));
    }
    if (dacoord) {
      CHKERRQ(DMSetUp(dacoord));
      CHKERRQ(DMCreateLocalVector(dacoord,&xcoorl));
      CHKERRQ(DMGlobalToLocalBegin(dacoord,xcoor,INSERT_VALUES,xcoorl));
      CHKERRQ(DMGlobalToLocalEnd(dacoord,xcoor,INSERT_VALUES,xcoorl));
      CHKERRQ(DMDestroy(&dacoord));
    } else {
      PetscInt   ien,jen,ken,nc,nl,cdof,deg;
      char       fecmesh[64];
      const char *name;
      PetscBool  flg;

      CHKERRQ(DMDAGetNumElementsGhosted(daview,&ien,&jen,&ken));
      nc   = ien*(jen>0 ? jen : 1)*(ken>0 ? ken : 1);

      CHKERRQ(VecGetLocalSize(xcoor,&nl));
      PetscCheckFalse(nc && nl % nc,PETSC_COMM_SELF,PETSC_ERR_SUP,"Incompatible local coordinate size %D and number of cells %D",nl,nc);
      CHKERRQ(VecDuplicate(xcoor,&xcoorl));
      CHKERRQ(VecCopy(xcoor,xcoorl));
      CHKERRQ(VecSetDM(xcoorl,NULL));
      CHKERRQ(PetscObjectGetName((PetscObject)xcoor,&name));
      CHKERRQ(PetscStrbeginswith(name,"FiniteElementCollection:",&flg));
      if (!flg) {
        deg = 0;
        if (nc && nl) {
          cdof = nl/(nc*dim);
          deg  = 1;
          while (1) {
            PetscInt degd = 1;
            for (i=0;i<dim;i++) degd *= (deg+1);
            PetscCheckFalse(degd > cdof,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cell dofs %D",cdof);
            if (degd == cdof) break;
            deg++;
          }
        }
        CHKERRQ(PetscSNPrintf(fecmesh,sizeof(fecmesh),"FiniteElementCollection: L2_T1_%DD_P%D",dim,deg));
        CHKERRQ(PetscObjectSetName((PetscObject)xcoorl,fecmesh));
      } else {
        CHKERRQ(PetscObjectSetName((PetscObject)xcoorl,name));
      }
    }

    /* xcoorl is composed with the ghosted DMDA, the ghosted coordinate DMDA (if present) is only available through this vector */
    CHKERRQ(PetscObjectCompose((PetscObject)daview,"GLVisGraphicsCoordsGhosted",(PetscObject)xcoorl));
    CHKERRQ(PetscObjectDereference((PetscObject)xcoorl));

    /* daview is composed with the original DMDA */
    CHKERRQ(PetscObjectCompose(oda,"GLVisGraphicsDMDAGhosted",(PetscObject)daview));
    CHKERRQ(PetscObjectDereference((PetscObject)daview));
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

    CHKERRQ(DMDAGetInfo(daview,&dim,NULL,NULL,NULL,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL));
    CHKERRQ(DMGetApplicationContext(daview,&dactx));
    CHKERRQ(DMCreateLocalVector(daview,&xlocal));
    CHKERRQ(DMDAGetFieldNames(da,(const char * const **)&dafieldname));
    CHKERRQ(DMDAGetNumVerticesGhosted(daview,&M,&N,&P));
    CHKERRQ(PetscSNPrintf(fec,sizeof(fec),"FiniteElementCollection: H1_%DD_P1",dim));
    CHKERRQ(PetscMalloc6(dof,&fec_type,dof,&nlocal,dof,&bss,dof,&dims,dof,&fieldname,dof,&Ufield));
    for (i=0;i<dof;i++) bss[i] = 1;
    nf = dof;

    ierr = PetscOptionsBegin(PetscObjectComm(oda),oda->prefix,"GLVis PetscViewer DMDA Field options","PetscViewer");CHKERRQ(ierr);
    CHKERRQ(PetscOptionsIntArray("-viewer_glvis_dm_da_bs","Block sizes for subfields; enable vector representation",NULL,bss,&nf,&bsset));
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (bsset) {
      PetscInt t;
      for (i=0,t=0;i<nf;i++) t += bss[i];
      PetscCheckFalse(t != dof,PetscObjectComm(oda),PETSC_ERR_USER,"Sum of block sizes %D should equal %D",t,dof);
    } else nf = dof;

    for (i=0,s=0;i<nf;i++) {
      CHKERRQ(PetscStrallocpy(fec,&fec_type[i]));
      if (bss[i] == 1) {
        CHKERRQ(PetscStrallocpy(dafieldname[s],&fieldname[i]));
      } else {
        PetscInt b;
        size_t tlen = 9; /* "Vector-" + end */
        for (b=0;b<bss[i];b++) {
          size_t len;
          CHKERRQ(PetscStrlen(dafieldname[s+b],&len));
          tlen += len + 1; /* field + "-" */
        }
        CHKERRQ(PetscMalloc1(tlen,&fieldname[i]));
        CHKERRQ(PetscStrcpy(fieldname[i],"Vector-"));
        for (b=0;b<bss[i]-1;b++) {
          CHKERRQ(PetscStrcat(fieldname[i],dafieldname[s+b]));
          CHKERRQ(PetscStrcat(fieldname[i],"-"));
        }
        CHKERRQ(PetscStrcat(fieldname[i],dafieldname[s+b]));
      }
      dims[i] = dim;
      nlocal[i] = M*N*P*bss[i];
      s += bss[i];
    }

    /* the viewer context takes ownership of xlocal and destroys it in DMDAFieldDestroyGLVisViewerCtx_Private */
    CHKERRQ(PetscNew(&ctx));
    ctx->xlocal = xlocal;

    /* create work vectors */
    for (i=0;i<nf;i++) {
      CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject)da),nlocal[i],PETSC_DECIDE,&Ufield[i]));
      CHKERRQ(PetscObjectSetName((PetscObject)Ufield[i],fieldname[i]));
      CHKERRQ(VecSetBlockSize(Ufield[i],bss[i]));
      CHKERRQ(VecSetDM(Ufield[i],da));
    }

    CHKERRQ(PetscViewerGLVisSetFields(viewer,nf,(const char**)fec_type,dims,DMDASampleGLVisFields_Private,(PetscObject*)Ufield,ctx,DMDAFieldDestroyGLVisViewerCtx_Private));
    for (i=0;i<nf;i++) {
      CHKERRQ(PetscFree(fec_type[i]));
      CHKERRQ(PetscFree(fieldname[i]));
      CHKERRQ(VecDestroy(&Ufield[i]));
    }
    CHKERRQ(PetscFree6(fec_type,nlocal,bss,dims,fieldname,Ufield));
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
  const char        *fmt;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  PetscCheck(isascii,PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Viewer must be of type VIEWERASCII");
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)viewer),&size));
  PetscCheckFalse(size > 1,PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Use single sequential viewers for parallel visualization");
  CHKERRQ(DMGetDimension(dm,&dim));

  /* get container: determines if a process visualizes is portion of the data or not */
  CHKERRQ(PetscObjectQuery((PetscObject)viewer,"_glvis_info_container",(PetscObject*)&glvis_container));
  PetscCheck(glvis_container,PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Missing GLVis container");
  {
    PetscViewerGLVisInfo glvis_info;
    CHKERRQ(PetscContainerGetPointer(glvis_container,(void**)&glvis_info));
    enabled = glvis_info->enabled;
    fmt = glvis_info->fmt;
  }
  /* this can happen if we are calling DMView outside of VecView_GLVis */
  CHKERRQ(PetscObjectQuery((PetscObject)dm,"GLVisGraphicsDMDAGhosted",(PetscObject*)&da));
  if (!da) CHKERRQ(DMSetUpGLVisViewer_DMDA((PetscObject)dm,NULL));
  CHKERRQ(PetscObjectQuery((PetscObject)dm,"GLVisGraphicsDMDAGhosted",(PetscObject*)&da));
  PetscCheck(da,PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Missing GLVis ghosted DMDA");
  CHKERRQ(DMGetCoordinateDim(da,&sdim));

  CHKERRQ(PetscViewerASCIIPrintf(viewer,"MFEM mesh v1.0\n"));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"\ndimension\n"));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"%D\n",dim));

  if (!enabled) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"\nelements\n"));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"%D\n",0));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"\nboundary\n"));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"%D\n",0));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"\nvertices\n"));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"%D\n",0));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"%D\n",sdim));
    PetscFunctionReturn(0);
  }

  CHKERRQ(DMDAGetNumElementsGhosted(da,&ien,&jen,&ken));
  nel  = ien;
  if (dim > 1) nel *= jen;
  if (dim > 2) nel *= ken;
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"\nelements\n"));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"%D\n",nel));
  switch (dim) {
  case 1:
    for (ie = 0; ie < ien; ie++) {
      vid[0] = ie;
      vid[1] = ie+1;
      mid    = 1; /* material id */
      cid    = 1; /* segment */
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"%D %D %D %D\n",mid,cid,vid[0],vid[1]));
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
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"%D %D %D %D %D %D\n",mid,cid,vid[0],vid[1],vid[2],vid[3]));
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
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"%D %D %D %D %D %D %D %D %D %D\n",mid,cid,vid[0],vid[1],vid[2],vid[3],vid[4],vid[5],vid[6],vid[7]));
        }
      }
    }
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Unsupported dimension %D",dim);
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"\nboundary\n"));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"%D\n",0));

  /* vertex coordinates */
  CHKERRQ(PetscObjectQuery((PetscObject)da,"GLVisGraphicsCoordsGhosted",(PetscObject*)&xcoorl));
  PetscCheck(xcoorl,PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Missing GLVis ghosted coords");
  CHKERRQ(DMDAGetNumVerticesGhosted(da,&ien,&jen,&ken));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"\nvertices\n"));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"%D\n",ien*jen*ken));
  if (nel) {
    CHKERRQ(VecGetDM(xcoorl,&cda));
    CHKERRQ(VecGetArrayRead(xcoorl,&array));
    if (!cda) { /* HO viz */
      const char *fecname;
      PetscInt   nc,nl;

      CHKERRQ(PetscObjectGetName((PetscObject)xcoorl,&fecname));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"nodes\n"));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"FiniteElementSpace\n"));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s\n",fecname));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"VDim: %D\n",sdim));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"Ordering: 1\n\n")); /*Ordering::byVDIM*/
      /* L2 coordinates */
      CHKERRQ(DMDAGetNumElementsGhosted(da,&ien,&jen,&ken));
      CHKERRQ(VecGetLocalSize(xcoorl,&nl));
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

      CHKERRQ(DMGetApplicationContext(da,&dactx));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"%D\n",sdim));
      cdof = sdim;
      CHKERRQ(DMDAGetCorners(da,&sx,&sy,&sz,NULL,NULL,NULL));
      CHKERRQ(DMDAGetGhostCorners(da,&gsx,&gsy,&gsz,&gm,&gn,&gp));
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
              CHKERRQ(PetscViewerASCIIPrintf(viewer,fmt,PetscRealPart(array[cdof*i+c*sdim+d])));
            }
            CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
          }
        }
      }
    }
    CHKERRQ(VecRestoreArrayRead(xcoorl,&array));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMView_DA_GLVis(DM dm, PetscViewer viewer)
{
  PetscFunctionBegin;
  CHKERRQ(DMView_GLVis(dm,viewer,DMDAView_GLVis_ASCII));
  PetscFunctionReturn(0);
}
