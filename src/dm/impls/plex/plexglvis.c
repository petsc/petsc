#include <petsc/private/glvisviewerimpl.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/dmpleximpl.h>
#include <petscbt.h>
#include <petscdmplex.h>
#include <petscsf.h>
#include <petscds.h>

typedef struct {
  PetscInt   nf;
  VecScatter *scctx;
} GLVisViewerCtx;

static PetscErrorCode DestroyGLVisViewerCtx_Private(void *vctx)
{
  GLVisViewerCtx *ctx = (GLVisViewerCtx*)vctx;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0;i<ctx->nf;i++) {
    ierr = VecScatterDestroy(&ctx->scctx[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(ctx->scctx);CHKERRQ(ierr);
  ierr = PetscFree(vctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexSampleGLVisFields_Private(PetscObject oX, PetscInt nf, PetscObject oXfield[], void *vctx)
{
  GLVisViewerCtx *ctx = (GLVisViewerCtx*)vctx;
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (f=0;f<nf;f++) {
    ierr = VecScatterBegin(ctx->scctx[f],(Vec)oX,(Vec)oXfield[f],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ctx->scctx[f],(Vec)oX,(Vec)oXfield[f],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* for FEM, it works for H1 fields only and extracts dofs at cell vertices, discarding any other dof */
PetscErrorCode DMSetUpGLVisViewer_Plex(PetscObject odm, PetscViewer viewer)
{
  DM             dm = (DM)odm;
  Vec            xlocal,xfield;
  PetscDS        ds;
  IS             globalNum,isfield;
  PetscBT        vown;
  char           **fieldname = NULL,**fec_type = NULL;
  const PetscInt *gNum;
  PetscInt       *nlocal,*bs,*idxs,*dims;
  PetscInt       f,maxfields,nfields,c,totc,totdofs,Nv,cum,i;
  PetscInt       dim,cStart,cEnd,cEndInterior,vStart,vEnd;
  GLVisViewerCtx *ctx;
  PetscSection   s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm,0,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  cEnd = cEndInterior < 0 ? cEnd : cEndInterior;
  ierr = DMPlexGetCellNumbering(dm,&globalNum);CHKERRQ(ierr);
  ierr = ISGetIndices(globalNum,&gNum);CHKERRQ(ierr);
  ierr = PetscBTCreate(vEnd-vStart,&vown);CHKERRQ(ierr);
  for (c = cStart, totc = 0; c < cEnd; c++) {
    if (gNum[c-cStart] >= 0) {
      PetscInt i,numPoints,*points = NULL;

      totc++;
      ierr = DMPlexGetTransitiveClosure(dm,c,PETSC_TRUE,&numPoints,&points);CHKERRQ(ierr);
      for (i=0;i<numPoints*2;i+= 2) {
        if ((points[i] >= vStart) && (points[i] < vEnd)) {
          ierr = PetscBTSet(vown,points[i]-vStart);CHKERRQ(ierr);
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(dm,c,PETSC_TRUE,&numPoints,&points);CHKERRQ(ierr);
    }
  }
  for (f=0,Nv=0;f<vEnd-vStart;f++) if (PetscBTLookup(vown,f)) Nv++;

  ierr = DMCreateLocalVector(dm,&xlocal);CHKERRQ(ierr);
  ierr = VecGetLocalSize(xlocal,&totdofs);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm,&s);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(s,&nfields);CHKERRQ(ierr);
  for (f=0,maxfields=0;f<nfields;f++) {
    PetscInt bs;

    ierr = PetscSectionGetFieldComponents(s,f,&bs);CHKERRQ(ierr);
    maxfields += bs;
  }
  ierr = PetscCalloc6(maxfields,&fieldname,maxfields,&nlocal,maxfields,&bs,maxfields,&dims,maxfields,&fec_type,totdofs,&idxs);CHKERRQ(ierr);
  ierr = PetscNew(&ctx);CHKERRQ(ierr);
  ierr = PetscCalloc1(maxfields,&ctx->scctx);CHKERRQ(ierr);
  ierr = DMGetDS(dm,&ds);CHKERRQ(ierr);
  if (ds) {
    for (f=0;f<nfields;f++) {
      const char* fname;
      char        name[256];
      PetscObject disc;
      size_t      len;

      ierr = PetscSectionGetFieldName(s,f,&fname);CHKERRQ(ierr);
      ierr = PetscStrlen(fname,&len);CHKERRQ(ierr);
      if (len) {
        ierr = PetscStrcpy(name,fname);CHKERRQ(ierr);
      } else {
        ierr = PetscSNPrintf(name,256,"Field%d",f);CHKERRQ(ierr);
      }
      ierr = PetscDSGetDiscretization(ds,f,&disc);CHKERRQ(ierr);
      if (disc) {
        PetscClassId id;
        PetscInt     Nc;
        char         fec[64];

        ierr = PetscObjectGetClassId(disc, &id);CHKERRQ(ierr);
        if (id == PETSCFE_CLASSID) {
          PetscFE            fem = (PetscFE)disc;
          PetscDualSpace     sp;
          PetscDualSpaceType spname;
          PetscInt           order;
          PetscBool          islag,continuous,H1 = PETSC_TRUE;

          ierr = PetscFEGetNumComponents(fem,&Nc);CHKERRQ(ierr);
          ierr = PetscFEGetDualSpace(fem,&sp);CHKERRQ(ierr);
          ierr = PetscDualSpaceGetType(sp,&spname);CHKERRQ(ierr);
          ierr = PetscStrcmp(spname,PETSCDUALSPACELAGRANGE,&islag);CHKERRQ(ierr);
          if (!islag) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported dual space");
          ierr = PetscDualSpaceLagrangeGetContinuity(sp,&continuous);CHKERRQ(ierr);
          if (!continuous) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Discontinuous space visualization currently unsupported");
          ierr = PetscDualSpaceGetOrder(sp,&order);CHKERRQ(ierr);
          if (continuous && order > 0) {
            ierr = PetscSNPrintf(fec,64,"FiniteElementCollection: H1_%dD_P1",dim);CHKERRQ(ierr);
          } else {
            H1   = PETSC_FALSE;
            ierr = PetscSNPrintf(fec,64,"FiniteElementCollection: L2_%dD_P%d",dim,order);CHKERRQ(ierr);
          }
          ierr = PetscStrallocpy(name,&fieldname[ctx->nf]);CHKERRQ(ierr);
          bs[ctx->nf]   = Nc;
          dims[ctx->nf] = dim;
          if (H1) {
            nlocal[ctx->nf] = Nc * Nv;
            ierr = PetscStrallocpy(fec,&fec_type[ctx->nf]);CHKERRQ(ierr);
            ierr = VecCreateSeq(PETSC_COMM_SELF,Nv*Nc,&xfield);CHKERRQ(ierr);
            for (i=0,cum=0;i<vEnd-vStart;i++) {
              PetscInt j,off;

              if (PetscUnlikely(!PetscBTLookup(vown,i))) continue;
              ierr = PetscSectionGetFieldOffset(s,i+vStart,f,&off);CHKERRQ(ierr);
              for (j=0;j<Nc;j++) idxs[cum++] = off + j;
            }
            ierr = ISCreateGeneral(PetscObjectComm((PetscObject)xlocal),Nv*Nc,idxs,PETSC_USE_POINTER,&isfield);CHKERRQ(ierr);
          } else {
            nlocal[ctx->nf] = Nc * totc;
            ierr = PetscStrallocpy(fec,&fec_type[ctx->nf]);CHKERRQ(ierr);
            ierr = VecCreateSeq(PETSC_COMM_SELF,Nc*totc,&xfield);CHKERRQ(ierr);
            for (i=0,cum=0;i<cEnd-cStart;i++) {
              PetscInt j,off;

              if (PetscUnlikely(gNum[i] < 0)) continue;
              ierr = PetscSectionGetFieldOffset(s,i+cStart,f,&off);CHKERRQ(ierr);
              for (j=0;j<Nc;j++) idxs[cum++] = off + j;
            }
            ierr = ISCreateGeneral(PetscObjectComm((PetscObject)xlocal),totc*Nc,idxs,PETSC_USE_POINTER,&isfield);CHKERRQ(ierr);
          }
          ierr = VecScatterCreate(xlocal,isfield,xfield,NULL,&ctx->scctx[ctx->nf]);CHKERRQ(ierr);
          ierr = VecDestroy(&xfield);CHKERRQ(ierr);
          ierr = ISDestroy(&isfield);CHKERRQ(ierr);
          ctx->nf++;
        } else if (id == PETSCFV_CLASSID) {
          PetscInt c;

          ierr = PetscFVGetNumComponents((PetscFV)disc,&Nc);CHKERRQ(ierr);
          ierr = PetscSNPrintf(fec,64,"FiniteElementCollection: L2_%dD_P0",dim);CHKERRQ(ierr);
          for (c = 0; c < Nc; c++) {
            char comp[256];
            ierr = PetscSNPrintf(comp,256,"%s-Comp%d",name,c);CHKERRQ(ierr);
            ierr = PetscStrallocpy(comp,&fieldname[ctx->nf]);CHKERRQ(ierr);
            bs[ctx->nf] = 1; /* Does PetscFV support components with different block size? */
            nlocal[ctx->nf] = totc;
            dims[ctx->nf] = dim;
            ierr = PetscStrallocpy(fec,&fec_type[ctx->nf]);CHKERRQ(ierr);
            ierr = VecCreateSeq(PETSC_COMM_SELF,totc,&xfield);CHKERRQ(ierr);
            for (i=0,cum=0;i<cEnd-cStart;i++) {
              PetscInt off;

              if (PetscUnlikely(gNum[i])<0) continue;
              ierr = PetscSectionGetFieldOffset(s,i+cStart,f,&off);CHKERRQ(ierr);
              idxs[cum++] = off + c;
            }
            ierr = ISCreateGeneral(PetscObjectComm((PetscObject)xlocal),totc,idxs,PETSC_USE_POINTER,&isfield);CHKERRQ(ierr);
            ierr = VecScatterCreate(xlocal,isfield,xfield,NULL,&ctx->scctx[ctx->nf]);CHKERRQ(ierr);
            ierr = VecDestroy(&xfield);CHKERRQ(ierr);
            ierr = ISDestroy(&isfield);CHKERRQ(ierr);
            ctx->nf++;
          }
        } else SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONG,"Unknown discretization type for field %D",f);
      } else SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Missing discretization for field %D",f);
    }
  } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Needs a DS attached to the DM");
  ierr = PetscBTDestroy(&vown);CHKERRQ(ierr);
  ierr = VecDestroy(&xlocal);CHKERRQ(ierr);
  ierr = ISRestoreIndices(globalNum,&gNum);CHKERRQ(ierr);

  /* customize the viewer */
  ierr = PetscViewerGLVisSetFields(viewer,ctx->nf,(const char**)fieldname,(const char**)fec_type,nlocal,bs,dims,DMPlexSampleGLVisFields_Private,ctx,DestroyGLVisViewerCtx_Private);CHKERRQ(ierr);
  for (f=0;f<ctx->nf;f++) {
    ierr = PetscFree(fieldname[f]);CHKERRQ(ierr);
    ierr = PetscFree(fec_type[f]);CHKERRQ(ierr);
  }
  ierr = PetscFree6(fieldname,nlocal,bs,dims,fec_type,idxs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef enum {MFEM_POINT=0,MFEM_SEGMENT,MFEM_TRIANGLE,MFEM_SQUARE,MFEM_TETRAHEDRON,MFEM_CUBE,MFEM_UNDEF} MFEM_cid;

MFEM_cid mfem_table_cid[4][7] = { {MFEM_POINT,MFEM_UNDEF,MFEM_UNDEF  ,MFEM_UNDEF   ,MFEM_UNDEF      ,MFEM_UNDEF, MFEM_UNDEF},
                                  {MFEM_POINT,MFEM_UNDEF,MFEM_SEGMENT,MFEM_UNDEF   ,MFEM_UNDEF      ,MFEM_UNDEF, MFEM_UNDEF},
                                  {MFEM_POINT,MFEM_UNDEF,MFEM_SEGMENT,MFEM_TRIANGLE,MFEM_SQUARE     ,MFEM_UNDEF, MFEM_UNDEF},
                                  {MFEM_POINT,MFEM_UNDEF,MFEM_SEGMENT,MFEM_UNDEF   ,MFEM_TETRAHEDRON,MFEM_UNDEF, MFEM_CUBE } };

static PetscErrorCode DMPlexGetPointMFEMCellID_Internal(DM dm, DMLabel label, PetscInt p, PetscInt *mid, PetscInt *cid)
{
  DMLabel        dlabel;
  PetscInt       depth,csize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepthLabel(dm,&dlabel);CHKERRQ(ierr);
  ierr = DMLabelGetValue(dlabel,p,&depth);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm,p,&csize);CHKERRQ(ierr);
  if (label) {
    ierr = DMLabelGetValue(label,p,mid);CHKERRQ(ierr);
  } else {
    *mid = 1;
  }
  *cid = mfem_table_cid[depth][csize];
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexGetPointMFEMVertexIDs_Internal(DM dm, PetscInt p, PetscSection csec, PetscInt *nv, int vids[])
{
  PetscInt       dim,sdim,dof = 0,off = 0,i,q,vStart,vEnd,numPoints,*points = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepthStratum(dm,0,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  sdim = dim;
  if (csec) {
    ierr = DMGetCoordinateDim(dm,&sdim);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(csec,vStart,&off);CHKERRQ(ierr);
    off  = off/sdim;
    ierr = PetscSectionGetDof(csec,p,&dof);CHKERRQ(ierr);
  }
  if (!dof) {
    ierr = DMPlexGetTransitiveClosure(dm,p,PETSC_TRUE,&numPoints,&points);CHKERRQ(ierr);
    for (i=0,q=0;i<numPoints*2;i+= 2)
      if ((points[i] >= vStart) && (points[i] < vEnd))
        vids[q++] = points[i]-vStart+off;
    ierr = DMPlexRestoreTransitiveClosure(dm,p,PETSC_TRUE,&numPoints,&points);CHKERRQ(ierr);
  } else {
    ierr = PetscSectionGetOffset(csec,p,&off);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(csec,p,&dof);CHKERRQ(ierr);
    for (q=0;q<dof/sdim;q++) vids[q] = off/sdim + q;
  }
  ierr = DMPlexInvertCell(dim,q,vids);CHKERRQ(ierr);
  *nv  = q;
  PetscFunctionReturn(0);
}

/* ASCII visualization/dump: full support for simplices and tensor product cells. It supports AMR */
static PetscErrorCode DMPlexView_GLVis_ASCII(DM dm, PetscViewer viewer)
{
  DMLabel              label;
  PetscSection         coordSection,parentSection;
  Vec                  coordinates;
  IS                   globalNum = NULL;
  const PetscScalar    *array;
  const PetscInt       *gNum;
  PetscInt             bf,p,sdim,dim,depth,novl;
  PetscInt             cStart,cEnd,cEndInterior,vStart,vEnd,nvert;
  PetscMPIInt          commsize;
  PetscBool            localized,ovl,isascii,enable_boundary,enable_ncmesh;
  PetscBT              pown;
  PetscErrorCode       ierr;
  PetscContainer       glvis_container;
  PetscBool            enabled = PETSC_TRUE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Viewer must be of type VIEWERASCII");
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)viewer),&commsize);CHKERRQ(ierr);
  if (commsize > 1) SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Use single sequential viewers for parallel visualization");
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm,&depth);CHKERRQ(ierr);
  if (depth >= 0 && dim != depth) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Mesh must be interpolated");

  /* get container: determines if a process visualizes is portion of the data or not */
  ierr = PetscObjectQuery((PetscObject)viewer,"_glvis_info_container",(PetscObject*)&glvis_container);CHKERRQ(ierr);
  if (!glvis_container) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Missing GLVis container");
  {
    PetscViewerGLVisInfo glvis_info;
    ierr = PetscContainerGetPointer(glvis_container,(void**)&glvis_info);CHKERRQ(ierr);
    enabled = glvis_info->enabled;
  }
  ierr = DMPlexGetHybridBounds(dm, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  cEnd = cEndInterior < 0 ? cEnd : cEndInterior;
  ierr = DMPlexGetDepthStratum(dm,0,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocalized(dm,&localized);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm,&coordSection);CHKERRQ(ierr);

  /*
     a couple of sections of the mesh specification are disabled
       - boundary: unless we want to visualize boundary attributes, the boundary is not needed for proper mesh visualization
       - vertex_parents: used for non-conforming meshes only when we want to use MFEM as a discretization package and be able to derefine the mesh
  */
  enable_boundary = PETSC_FALSE;
  enable_ncmesh   = PETSC_FALSE;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)dm),((PetscObject)dm)->prefix,"GLVis PetscViewer DMPlex Options","PetscViewer");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-viewer_glvis_dm_plex_enable_boundary","Enable boundary section in mesh representation; useful for debugging purposes",NULL,enable_boundary,&enable_boundary,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-viewer_glvis_dm_plex_enable_ncmesh","Enable vertex_parents section in mesh representation; useful for debugging purposes",NULL,enable_ncmesh,&enable_ncmesh,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* Identify possible cells in the overlap */
  gNum = NULL;
  novl = 0;
  ovl  = PETSC_FALSE;
  pown = NULL;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm),&commsize);CHKERRQ(ierr);
  if (commsize > 1) {
    ierr = DMPlexGetCellNumbering(dm,&globalNum);CHKERRQ(ierr);
    ierr = ISGetIndices(globalNum,&gNum);CHKERRQ(ierr);
    for (p=cStart; p<cEnd; p++) {
      if (gNum[p-cStart] < 0) {
        ovl = PETSC_TRUE;
        novl++;
      }
    }
    if (ovl) {
      /* it may happen that pown get not destroyed, if the user closes the window while this function is running.
         TODO: garbage collector? attach pown to dm?  */
      ierr = PetscBTCreate(PetscMax(cEnd-cStart,vEnd-vStart),&pown);CHKERRQ(ierr);
    }
  }

  if (!enabled) {
    ierr = PetscViewerASCIIPrintf(viewer,"MFEM mesh v1.1\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"\ndimension\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%D\n",dim);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"\nelements\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%D\n",0);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"\nboundary\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%D\n",0);CHKERRQ(ierr);
    ierr = DMGetCoordinateDim(dm,&sdim);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"\nvertices\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%D\n",0);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%D\n",sdim);CHKERRQ(ierr);
    ierr = PetscBTDestroy(&pown);CHKERRQ(ierr);
    if (globalNum) {
      ierr = ISRestoreIndices(globalNum,&gNum);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }

  /* header */
  ierr = PetscViewerASCIIPrintf(viewer,"MFEM mesh v1.1\n");CHKERRQ(ierr);

  /* topological dimension */
  ierr = PetscViewerASCIIPrintf(viewer,"\ndimension\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%D\n",dim);CHKERRQ(ierr);

  /* elements */
  /* TODO: is this the label we want to use for marking material IDs?
     We should decide to have a single marker for these stuff
     Proposal: DMSetMaterialLabel?
  */
  ierr = DMGetLabel(dm,"Cell Sets",&label);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\nelements\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%D\n",cEnd-cStart-novl);CHKERRQ(ierr);
  for (p=cStart;p<cEnd;p++) {
    int      vids[8];
    PetscInt i,nv = 0,cid = -1,mid = 1;

    if (ovl) {
      if (gNum[p-cStart] < 0) continue;
      else {
        ierr = PetscBTSet(pown,p-cStart);CHKERRQ(ierr);
      }
    }
    ierr = DMPlexGetPointMFEMCellID_Internal(dm,label,p,&mid,&cid);CHKERRQ(ierr);
    ierr = DMPlexGetPointMFEMVertexIDs_Internal(dm,p,localized ? coordSection : NULL,&nv,vids);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%D %D",mid,cid);CHKERRQ(ierr);
    for (i=0;i<nv;i++) {
      ierr = PetscViewerASCIIPrintf(viewer," %d",vids[i]);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  }

  /* boundary */
  ierr = PetscViewerASCIIPrintf(viewer,"\nboundary\n");CHKERRQ(ierr);
  if (!enable_boundary) {
    ierr = PetscViewerASCIIPrintf(viewer,"%D\n",0);CHKERRQ(ierr);
  } else {
    PetscInt fStart,fEnd,fEndInterior;
    PetscInt *faces = NULL,fpc = 0,vpf = 0;
    PetscInt fv1[]     = {0,1},
             fv2tri[]  = {0,1,
                          1,2,
                          2,0},
             fv2quad[] = {0,1,
                          1,2,
                          2,3,
                          3,0},
             fv3tet[]  = {0,1,2,
                          0,3,1,
                          0,2,3,
                          2,1,3},
             fv3hex[]  = {0,1,2,3,
                       4,5,6,7,
                       0,3,5,4,
                       2,1,7,6,
                       3,2,6,5,
                       0,4,7,1};

    if (localized) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Boundary specification with periodic mesh");
    /* determine orientation of boundary mesh */
    if (cEnd-cStart) {
      ierr = DMPlexGetConeSize(dm,cStart,&fpc);CHKERRQ(ierr);
      switch(dim) {
        case 1:
          if (fpc != 2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unhandled case faces per cell %D",fpc);
          faces = fv1;
          vpf = 1;
          break;
        case 2:
          switch (fpc) {
            case 3:
              faces = fv2tri;
              vpf   = 2;
              break;
            case 4:
              faces = fv2quad;
              vpf   = 2;
              break;
            default:
              SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unhandled case: faces per cell %D",fpc);
              break;
          }
          break;
        case 3:
          switch (fpc) {
            case 4:
              faces = fv3tet;
              vpf   = 3;
              break;
            case 6:
              faces = fv3hex;
              vpf   = 4;
              break;
            default:
              SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unhandled case: faces per cell %D",fpc);
              break;
          }
          break;
        default:
          SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unhandled dim");
          break;
      }
    }

    ierr = DMPlexGetHybridBounds(dm, NULL, &fEndInterior, NULL, NULL);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd);CHKERRQ(ierr);
    fEnd = fEndInterior < 0 ? fEnd : fEndInterior;
    if (!ovl) {
      for (p=fStart,bf=0;p<fEnd;p++) {
        PetscInt supportSize;

        ierr = DMPlexGetSupportSize(dm,p,&supportSize);CHKERRQ(ierr);
        if (supportSize == 1) bf++;
      }
    } else {
      for (p=fStart,bf=0;p<fEnd;p++) {
        const PetscInt *support;
        PetscInt       i,supportSize;
        PetscBool      has_owned = PETSC_FALSE, has_ghost = PETSC_FALSE;

        ierr = DMPlexGetSupportSize(dm,p,&supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm,p,&support);CHKERRQ(ierr);
        for (i=0;i<supportSize;i++) {
          if (PetscBTLookup(pown,support[i]-cStart)) has_owned = PETSC_TRUE;
          else has_ghost = PETSC_TRUE;
        }
        if ((supportSize == 1 && has_owned) || (supportSize > 1 && has_owned && has_ghost)) bf++;
      }
    }
    /* TODO: is this the label we want to use for marking boundary facets?
       We should decide to have a single marker for these stuff
       Proposal: DMSetBoundaryLabel?
    */
    ierr = DMGetLabel(dm,"marker",&label);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%D\n",bf);CHKERRQ(ierr);
    if (!ovl) {
      for (p=fStart;p<fEnd;p++) {
        PetscInt supportSize;

        ierr = DMPlexGetSupportSize(dm,p,&supportSize);CHKERRQ(ierr);
        if (supportSize == 1) {
          const    PetscInt *support, *cone;
          PetscInt i,c,v,cid = -1,mid = -1;
          PetscInt cellClosureSize,*cellClosure = NULL;

          ierr = DMPlexGetSupport(dm,p,&support);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm,support[0],&cone);CHKERRQ(ierr);
          for (c=0;c<fpc;c++)
            if (cone[c] == p)
              break;

          ierr = DMPlexGetTransitiveClosure(dm,support[0],PETSC_TRUE,&cellClosureSize,&cellClosure);CHKERRQ(ierr);
          for (v=0;v<cellClosureSize;v++)
            if (cellClosure[2*v] >= vStart && cellClosure[2*v] < vEnd)
              break;
          ierr = DMPlexGetPointMFEMCellID_Internal(dm,label,p,&mid,&cid);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPrintf(viewer,"%D %D",mid,cid);CHKERRQ(ierr);
          for (i=0;i<vpf;i++) {
            ierr = PetscViewerASCIIPrintf(viewer," %D",cellClosure[2*(v+faces[c*vpf+i])]-vStart);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
          ierr = DMPlexRestoreTransitiveClosure(dm,support[0],PETSC_TRUE,&cellClosureSize,&cellClosure);CHKERRQ(ierr);
        }
      }
    } else {
      for (p=fStart;p<fEnd;p++) {
        const PetscInt *support;
        PetscInt       i,supportSize,validcell = -1;
        PetscBool      has_owned = PETSC_FALSE, has_ghost = PETSC_FALSE;

        ierr = DMPlexGetSupportSize(dm,p,&supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm,p,&support);CHKERRQ(ierr);
        for (i=0;i<supportSize;i++) {
          if (PetscBTLookup(pown,support[i]-cStart)) {
            has_owned = PETSC_TRUE;
            validcell = support[i];
          } else has_ghost = PETSC_TRUE;
        }
        if ((supportSize == 1 && has_owned) || (supportSize > 1 && has_owned && has_ghost)) {
          const    PetscInt *support, *cone;
          PetscInt i,c,v,cid = -1,mid = -1;
          PetscInt cellClosureSize,*cellClosure = NULL;

          ierr = DMPlexGetSupport(dm,p,&support);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm,validcell,&cone);CHKERRQ(ierr);
          for (c=0;c<fpc;c++)
            if (cone[c] == p)
              break;

          ierr = DMPlexGetTransitiveClosure(dm,validcell,PETSC_TRUE,&cellClosureSize,&cellClosure);CHKERRQ(ierr);
          for (v=0;v<cellClosureSize;v++)
            if (cellClosure[2*v] >= vStart && cellClosure[2*v] < vEnd)
              break;
          ierr = DMPlexGetPointMFEMCellID_Internal(dm,label,p,&mid,&cid);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPrintf(viewer,"%D %D",mid,cid);CHKERRQ(ierr);
          for (i=0;i<vpf;i++) {
            ierr = PetscViewerASCIIPrintf(viewer," %D",cellClosure[2*(v+faces[c*vpf+i])]-vStart);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
          ierr = DMPlexRestoreTransitiveClosure(dm,validcell,PETSC_TRUE,&cellClosureSize,&cellClosure);CHKERRQ(ierr);
        }
      }
    }
  }

  /* mark owned vertices */
  if (ovl) {
    ierr = PetscBTMemzero(vEnd-vStart,pown);CHKERRQ(ierr);
    for (p=cStart;p<cEnd;p++) {
      PetscInt i,closureSize,*closure = NULL;

      if (gNum[p-cStart] < 0) continue;
      ierr = DMPlexGetTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      for (i=0;i<closureSize;i++) {
        const PetscInt pp = closure[2*i];

        if (pp >= vStart && pp < vEnd) {
          ierr = PetscBTSet(pown,pp-vStart);CHKERRQ(ierr);
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
    }
  }
  if (globalNum) {
    ierr = ISRestoreIndices(globalNum,&gNum);CHKERRQ(ierr);
  }

  /* vertex_parents (Non-conforming meshes) */
  parentSection  = NULL;
  if (enable_ncmesh) {
    ierr = DMPlexGetTree(dm,&parentSection,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  }
  if (parentSection) {
    PetscInt vp,gvp;

    for (vp=0,p=vStart;p<vEnd;p++) {
      DMLabel  dlabel;
      PetscInt parent,depth;

      if (PetscUnlikely(ovl && !PetscBTLookup(pown,p-vStart))) continue;
      ierr = DMPlexGetDepthLabel(dm,&dlabel);CHKERRQ(ierr);
      ierr = DMLabelGetValue(dlabel,p,&depth);CHKERRQ(ierr);
      ierr = DMPlexGetTreeParent(dm,p,&parent,NULL);CHKERRQ(ierr);
      if (parent != p) vp++;
    }
    ierr = MPIU_Allreduce(&vp,&gvp,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
    if (gvp) {
      PetscInt  maxsupp;
      PetscBool *skip = NULL;

      ierr = PetscViewerASCIIPrintf(viewer,"\nvertex_parents\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%D\n",vp);CHKERRQ(ierr);
      ierr = DMPlexGetMaxSizes(dm,NULL,&maxsupp);CHKERRQ(ierr);
      ierr = PetscMalloc1(maxsupp,&skip);CHKERRQ(ierr);
      for (p=vStart;p<vEnd;p++) {
        DMLabel  dlabel;
        PetscInt parent;

        if (PetscUnlikely(ovl && !PetscBTLookup(pown,p-vStart))) continue;
        ierr = DMPlexGetDepthLabel(dm,&dlabel);CHKERRQ(ierr);
        ierr = DMPlexGetTreeParent(dm,p,&parent,NULL);CHKERRQ(ierr);
        if (parent != p) {
          int            vids[8] = { -1, -1, -1, -1, -1, -1, -1, -1 }; /* silent overzealous clang static analyzer */
          PetscInt       i,nv,size,n,numChildren,depth = -1;
          const PetscInt *children;
          ierr = DMPlexGetConeSize(dm,parent,&size);CHKERRQ(ierr);
          switch (size) {
            case 2: /* edge */
              nv   = 0;
              ierr = DMPlexGetPointMFEMVertexIDs_Internal(dm,parent,localized ? coordSection : NULL,&nv,vids);CHKERRQ(ierr);
              ierr = PetscViewerASCIIPrintf(viewer,"%D",p-vStart);CHKERRQ(ierr);
              for (i=0;i<nv;i++) {
                ierr = PetscViewerASCIIPrintf(viewer," %d",vids[i]);CHKERRQ(ierr);
              }
              ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
              vp--;
              break;
            case 4: /* face */
              ierr = DMPlexGetTreeChildren(dm,parent,&numChildren,&children);CHKERRQ(ierr);
              for (n=0;n<numChildren;n++) {
                ierr = DMLabelGetValue(dlabel,children[n],&depth);CHKERRQ(ierr);
                if (!depth) {
                  const PetscInt *hvsupp,*hesupp,*cone;
                  PetscInt       hvsuppSize,hesuppSize,coneSize;
                  PetscInt       hv = children[n],he = -1,f;

                  ierr = PetscMemzero(skip,maxsupp*sizeof(PetscBool));CHKERRQ(ierr);
                  ierr = DMPlexGetSupportSize(dm,hv,&hvsuppSize);CHKERRQ(ierr);
                  ierr = DMPlexGetSupport(dm,hv,&hvsupp);CHKERRQ(ierr);
                  for (i=0;i<hvsuppSize;i++) {
                    PetscInt ep;
                    ierr = DMPlexGetTreeParent(dm,hvsupp[i],&ep,NULL);CHKERRQ(ierr);
                    if (ep != hvsupp[i]) {
                      he = hvsupp[i];
                    } else {
                      skip[i] = PETSC_TRUE;
                    }
                  }
                  if (he == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Vertex %D support size %D: hanging edge not found",hv,hvsuppSize);
                  ierr    = DMPlexGetCone(dm,he,&cone);CHKERRQ(ierr);
                  vids[0] = (int)((cone[0] == hv) ? cone[1] : cone[0]);
                  ierr    = DMPlexGetSupportSize(dm,he,&hesuppSize);CHKERRQ(ierr);
                  ierr    = DMPlexGetSupport(dm,he,&hesupp);CHKERRQ(ierr);
                  for (f=0;f<hesuppSize;f++) {
                    PetscInt j;

                    ierr = DMPlexGetCone(dm,hesupp[f],&cone);CHKERRQ(ierr);
                    ierr = DMPlexGetConeSize(dm,hesupp[f],&coneSize);CHKERRQ(ierr);
                    for (j=0;j<coneSize;j++) {
                      PetscInt k;
                      for (k=0;k<hvsuppSize;k++) {
                        if (hvsupp[k] == cone[j]) {
                          skip[k] = PETSC_TRUE;
                          break;
                        }
                      }
                    }
                  }
                  for (i=0;i<hvsuppSize;i++) {
                    if (!skip[i]) {
                      ierr = DMPlexGetCone(dm,hvsupp[i],&cone);CHKERRQ(ierr);
                      vids[1] = (int)((cone[0] == hv) ? cone[1] : cone[0]);
                    }
                  }
                  ierr = PetscViewerASCIIPrintf(viewer,"%D",hv-vStart);CHKERRQ(ierr);
                  for (i=0;i<2;i++) {
                    ierr = PetscViewerASCIIPrintf(viewer," %d",vids[i]-vStart);CHKERRQ(ierr);
                  }
                  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
                  vp--;
                }
              }
              break;
            default:
              SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Don't know how to deal with support size %d",size);
          }
        }
      }
      ierr = PetscFree(skip);CHKERRQ(ierr);
    }
    if (vp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected %D hanging vertices",vp);
  }
  ierr = PetscBTDestroy(&pown);CHKERRQ(ierr);

  /* vertices */
  ierr = DMGetCoordinateDim(dm,&sdim);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm,&coordinates);CHKERRQ(ierr);
  ierr = VecGetLocalSize(coordinates,&nvert);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\nvertices\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%D\n",nvert/sdim);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%D\n",sdim);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates,&array);CHKERRQ(ierr);
  for (p=0;p<nvert/sdim;p++) {
    PetscInt s;
    for (s=0;s<sdim;s++) {
      ierr = PetscViewerASCIIPrintf(viewer,"%g ",PetscRealPart(array[p*sdim+s]));CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(coordinates,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* dispatching, prints through the socket by prepending the mesh keyword to the usual ASCII dump */
PETSC_INTERN PetscErrorCode DMPlexView_GLVis(DM dm, PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isglvis,isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
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
      ierr = DMPlexView_GLVis_ASCII(dm,view);CHKERRQ(ierr);
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
    ierr = DMPlexView_GLVis_ASCII(dm,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
