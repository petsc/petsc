/* Routines to visualize DMDAs and fields through GLVis */

#include <petsc/private/dmdaimpl.h>
#include <petsc/private/glvisviewerimpl.h>

typedef struct {
  Vec xlocal;
} DMDAGLVisViewerCtx;

static PetscErrorCode DMDADestroyGLVisViewerCtx_Private(void *vctx)
{
  DMDAGLVisViewerCtx *ctx = (DMDAGLVisViewerCtx*)vctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&ctx->xlocal);CHKERRQ(ierr);
  ierr = PetscFree(vctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* all but the last proc per dimension claim the ghosted node */
static PetscErrorCode DMDAGetNumElementsGhosted(DM da, PetscInt *nex, PetscInt *ney, PetscInt *nez)
{
  PetscInt       M,N,P,sx,sy,sz,ien,jen,ken;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Appease -Wmaybe-uninitialized */
  if (nex) *nex = -1;
  if (ney) *ney = -1;
  if (nez) *nez = -1;
  ierr = DMDAGetInfo(da,NULL,&M,&N,&P,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&sx,&sy,&sz,&ien,&jen,&ken);CHKERRQ(ierr);
  if (sx+ien == M) ien--;
  if (sy+jen == N) jen--;
  if (sz+ken == P) ken--;
  if (nex) *nex = ien;
  if (ney) *ney = jen;
  if (nez) *nez = ken;
  PetscFunctionReturn(0);
}

/* inherits number of vertices from DMDAGetNumElementsGhosted */
static PetscErrorCode DMDAGetNumVerticesGhosted(DM da, PetscInt *nvx, PetscInt *nvy, PetscInt *nvz)
{
  PetscInt       ien,jen,ken,dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(da,&dim);CHKERRQ(ierr);
  ierr = DMDAGetNumElementsGhosted(da,&ien,&jen,&ken);CHKERRQ(ierr);
  ien  = ien+1;
  jen  = dim > 1 ? jen+1 : 1;
  ken  = dim > 2 ? ken+1 : 1;
  if (nvx) *nvx = ien;
  if (nvy) *nvy = jen;
  if (nvz) *nvz = ken;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDASampleGLVisFields_Private(PetscObject oX, PetscInt nf, PetscObject oXf[], void *vctx)
{
  DM                 da;
  DMDAGLVisViewerCtx *ctx = (DMDAGLVisViewerCtx*)vctx;
  const PetscScalar  *array;
  PetscScalar        **arrayf;
  PetscInt           i,f,ii,ien,jen,ken,ie,je,ke,bs,*bss;
  PetscInt           sx,sy,sz,gsx,gsy,gsz,ist,jst,kst,gm,gn,gp;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(ctx->xlocal,&da);CHKERRQ(ierr);
  if (!da) SETERRQ(PetscObjectComm(oX),PETSC_ERR_ARG_WRONG,"Vector not generated from a DMDA");
  ierr = VecGetBlockSize(ctx->xlocal,&bs);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,(Vec)oX,INSERT_VALUES,ctx->xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,(Vec)oX,INSERT_VALUES,ctx->xlocal);CHKERRQ(ierr);
  ierr = DMDAGetNumVerticesGhosted(da,&ien,&jen,&ken);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gsx,&gsy,&gsz,&gm,&gn,&gp);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&sx,&sy,&sz,NULL,NULL,NULL);CHKERRQ(ierr);
  kst  = gsz != sz ? 1 : 0;
  jst  = gsy != sy ? 1 : 0;
  ist  = gsx != sx ? 1 : 0;
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
  DM                 da = (DM)oda,daview,dacoord;
  Vec                xcoor,xcoorl,xlocal;
  DMDAGLVisViewerCtx *ctx;
  const char         **dafieldname;
  char               **fec_type,**fieldname,fec[64];
  const PetscInt     *lx,*ly,*lz;
  PetscInt           *nlocal,*bss,*dims;
  PetscInt           dim,M,N,P,m,n,p,dof,s,i,nf;
  PetscBool          bsset;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* Create a properly ghosted DMDA to visualize the mesh and the fields associated with */
  ierr = DMDAGetInfo(da,&dim,&M,&N,&P,&m,&n,&p,&dof,&s,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetOwnershipRanges(da,&lx,&ly,&lz);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&xcoor);CHKERRQ(ierr);
  ierr = PetscInfo(da,"Creating auxilary DMDA for managing GLVis graphics\n");CHKERRQ(ierr);
  switch (dim) {
  case 1:
    ierr = DMDACreate1d(PetscObjectComm((PetscObject)da),DM_BOUNDARY_NONE,M,dof,1,lx,&daview);CHKERRQ(ierr);
    ierr = DMDACreate1d(PetscObjectComm((PetscObject)da),DM_BOUNDARY_NONE,M,1,1,lx,&dacoord);CHKERRQ(ierr);
    ierr = PetscStrcpy(fec,"FiniteElementCollection: H1_1D_P1");CHKERRQ(ierr);
    break;
  case 2:
    ierr = DMDACreate2d(PetscObjectComm((PetscObject)da),DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,N,m,n,dof,1,lx,ly,&daview);CHKERRQ(ierr);
    ierr = DMDACreate2d(PetscObjectComm((PetscObject)da),DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,N,m,n,2,1,lx,ly,&dacoord);CHKERRQ(ierr);
    ierr = PetscStrcpy(fec,"FiniteElementCollection: H1_2D_P1");CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMDACreate3d(PetscObjectComm((PetscObject)da),DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,N,P,m,n,p,dof,1,lx,ly,lz,&daview);CHKERRQ(ierr);
    ierr = DMDACreate3d(PetscObjectComm((PetscObject)da),DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,N,P,m,n,p,3,1,lx,ly,lz,&dacoord);CHKERRQ(ierr);
    ierr = PetscStrcpy(fec,"FiniteElementCollection: H1_3D_P1");CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Unsupported dimension %D",dim);
    break;
  }
  ierr = DMSetUp(daview);CHKERRQ(ierr);
  ierr = DMSetUp(dacoord);CHKERRQ(ierr);
  if (!xcoor) {
    ierr = DMDASetUniformCoordinates(daview,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
    ierr = DMGetCoordinates(daview,&xcoor);CHKERRQ(ierr);
  }
  ierr = DMCreateLocalVector(daview,&xlocal);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dacoord,&xcoorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dacoord,xcoor,INSERT_VALUES,xcoorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dacoord,xcoor,INSERT_VALUES,xcoorl);CHKERRQ(ierr);

  /* xcoorl is composed with the original DMDA, ghosted coordinate DMDA is only available through this vector */
  ierr = PetscObjectCompose(oda,"GLVisGraphicsCoordsGhosted",(PetscObject)xcoorl);CHKERRQ(ierr);
  ierr = PetscObjectDereference((PetscObject)xcoorl);CHKERRQ(ierr);

  /* customize the viewer */
  ierr = DMDAGetFieldNames(da,(const char * const **)&dafieldname);CHKERRQ(ierr);
  ierr = DMDAGetNumVerticesGhosted(daview,&M,&N,&P);CHKERRQ(ierr);
  ierr = PetscMalloc5(dof,&fec_type,dof,&nlocal,dof,&bss,dof,&dims,dof,&fieldname);CHKERRQ(ierr);
  for (i=0;i<dof;i++) bss[i] = 1;
  nf = dof;

  ierr = PetscOptionsBegin(PetscObjectComm(oda),oda->prefix,"GLVis PetscViewer DMDA Options","PetscViewer");CHKERRQ(ierr);
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

  /* the viewer context takes ownership of xlocal (and the the properly ghosted DMDA associated with it) */
  ierr = PetscNew(&ctx);CHKERRQ(ierr);
  ctx->xlocal = xlocal;

  ierr = PetscViewerGLVisSetFields(viewer,nf,(const char**)fieldname,(const char**)fec_type,nlocal,bss,dims,DMDASampleGLVisFields_Private,ctx,DMDADestroyGLVisViewerCtx_Private);CHKERRQ(ierr);
  for (i=0;i<nf;i++) {
    ierr = PetscFree(fec_type[i]);CHKERRQ(ierr);
    ierr = PetscFree(fieldname[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree5(fec_type,nlocal,bss,dims,fieldname);CHKERRQ(ierr);
  ierr = DMDestroy(&dacoord);CHKERRQ(ierr);
  ierr = DMDestroy(&daview);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDAView_GLVis_ASCII(DM dm, PetscViewer viewer)
{
  DM                da;
  Vec               xcoorl;
  PetscMPIInt       commsize;
  const PetscScalar *array;
  PetscContainer    glvis_container;
  PetscInt          dim,sdim,i,vid[8],mid,cid;
  PetscInt          sx,sy,sz,ie,je,ke,ien,jen,ken;
  PetscInt          gsx,gsy,gsz,gm,gn,gp,kst,jst,ist;
  PetscBool         enabled = PETSC_TRUE, isascii;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Viewer must be of type VIEWERASCII");
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)viewer),&commsize);CHKERRQ(ierr);
  if (commsize > 1) SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Use single sequential viewers for parallel visualization");
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);

  /* get container: determines if a process visualizes is portion of the data or not */
  ierr = PetscObjectQuery((PetscObject)viewer,"_glvis_info_container",(PetscObject*)&glvis_container);CHKERRQ(ierr);
  if (!glvis_container) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Missing GLVis container");
  {
    PetscViewerGLVisInfo glvis_info;
    ierr = PetscContainerGetPointer(glvis_container,(void**)&glvis_info);CHKERRQ(ierr);
    enabled = glvis_info->enabled;
  }
  ierr = PetscObjectQuery((PetscObject)dm,"GLVisGraphicsCoordsGhosted",(PetscObject*)&xcoorl);CHKERRQ(ierr);
  if (!xcoorl) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Missing GLVis ghosted coords");
  ierr = VecGetDM(xcoorl,&da);CHKERRQ(ierr);
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
  i    = ien;
  if (dim > 1) i *= jen;
  if (dim > 2) i *= ken;
  ierr = PetscViewerASCIIPrintf(viewer,"\nelements\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%D\n",i);CHKERRQ(ierr);
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

  ierr = DMDAGetNumVerticesGhosted(da,&ien,&jen,&ken);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&sx,&sy,&sz,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gsx,&gsy,&gsz,&gm,&gn,&gp);CHKERRQ(ierr);
  kst  = gsz != sz ? 1 : 0;
  jst  = gsy != sy ? 1 : 0;
  ist  = gsx != sx ? 1 : 0;
  ierr = PetscViewerASCIIPrintf(viewer,"\nvertices\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%D\n",ien*jen*ken);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%D\n",sdim);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xcoorl,&array);CHKERRQ(ierr);
  for (ke = kst; ke < kst + ken; ke++) {
    for (je = jst; je < jst + jen; je++) {
      for (ie = ist; ie < ist + ien; ie++) {
        PetscInt d;

        i = ke * gm * gn + je * gm + ie;
        for (d=0;d<sdim-1;d++) {
          ierr = PetscViewerASCIIPrintf(viewer,"%g ",PetscRealPart(array[sdim*i+d]));CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer,"%g\n",PetscRealPart(array[sdim*i+d]));CHKERRQ(ierr);
      }
    }
  }
  ierr = VecRestoreArrayRead(xcoorl,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* dispatching, prints through the socket by prepending the mesh keyword to the usual ASCII dump: duplicated code as in plexglvis.c, should be merged together */
PETSC_INTERN PetscErrorCode DMView_DA_GLVis(DM dm, PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isglvis,isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERGLVIS,&isglvis);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (!isglvis && !isascii) SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Viewer must be of type VIEWERGLVIS or VIEWERASCII");
  if (isglvis) {
    PetscViewer          view;
    PetscViewerGLVisType type;

    ierr = PetscViewerGLVisGetType_Private(viewer,&type);CHKERRQ(ierr);
    ierr = PetscViewerGLVisGetDMWindow_Private(viewer,&view);CHKERRQ(ierr);
    if (view) { /* in the socket case, it may happen that the connection failed */
      if (type == PETSC_VIEWER_GLVIS_SOCKET) {
        PetscMPIInt size,rank;
        ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm),&size);CHKERRQ(ierr);
        ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(view,"parallel %D %D\nmesh\n",size,rank);CHKERRQ(ierr);
      }
      ierr = DMDAView_GLVis_ASCII(dm,view);CHKERRQ(ierr);
      ierr = PetscViewerFlush(view);CHKERRQ(ierr);
      if (type == PETSC_VIEWER_GLVIS_SOCKET) {
        PetscInt    dim;
        const char* name;

        ierr = PetscObjectGetName((PetscObject)dm,&name);CHKERRQ(ierr);
        ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
        ierr = PetscViewerGLVisInitWindow_Private(view,PETSC_TRUE,dim,name);CHKERRQ(ierr);
        ierr = PetscBarrier((PetscObject)dm);CHKERRQ(ierr);
      }
    }
  } else {
    ierr = DMDAView_GLVis_ASCII(dm,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
