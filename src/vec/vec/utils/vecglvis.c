#include <petsc/private/glvisviewerimpl.h>
#include <petsc/private/glvisvecimpl.h>

static PetscErrorCode PetscViewerGLVisVecInfoDestroy_Private(void *ptr)
{
  PetscViewerGLVisVecInfo info = (PetscViewerGLVisVecInfo)ptr;

  PetscFunctionBeginUser;
  CHKERRQ(PetscFree(info->fec_type));
  CHKERRQ(PetscFree(info));
  PetscFunctionReturn(0);
}

/* the main function to visualize vectors using GLVis */
PetscErrorCode VecView_GLVis(Vec U,PetscViewer viewer)
{
  PetscErrorCode         (*g2lfields)(PetscObject,PetscInt,PetscObject[],void*);
  Vec                    *Ufield;
  const char             **fec_type;
  PetscViewerGLVisStatus sockstatus;
  PetscViewerGLVisType   socktype;
  void                   *userctx;
  PetscInt               i,nfields,*spacedim;
  PetscBool              pause = PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerGLVisGetStatus_Private(viewer,&sockstatus));
  if (sockstatus == PETSCVIEWERGLVIS_DISABLED) PetscFunctionReturn(0);
  /* if the user did not customize the viewer through the API, we need extra data that can be attached to the Vec */
  CHKERRQ(PetscViewerGLVisGetFields_Private(viewer,&nfields,NULL,NULL,NULL,NULL,NULL));
  if (!nfields) {
    PetscObject dm;

    CHKERRQ(PetscObjectQuery((PetscObject)U, "__PETSc_dm",&dm));
    if (dm) {
      CHKERRQ(PetscViewerGLVisSetDM_Private(viewer,dm));
    } else SETERRQ(PetscObjectComm((PetscObject)U),PETSC_ERR_SUP,"You need to provide a DM or use PetscViewerGLVisSetFields()");
  }
  CHKERRQ(PetscViewerGLVisGetFields_Private(viewer,&nfields,&fec_type,&spacedim,&g2lfields,(PetscObject**)&Ufield,&userctx));
  if (!nfields) PetscFunctionReturn(0);

  CHKERRQ(PetscViewerGLVisGetType_Private(viewer,&socktype));

  for (i=0;i<nfields;i++) {
    PetscObject    fdm;
    PetscContainer container;

    /* attach visualization info to the vector */
    CHKERRQ(PetscObjectQuery((PetscObject)Ufield[i],"_glvis_info_container",(PetscObject*)&container));
    if (!container) {
      PetscViewerGLVisVecInfo info;

      CHKERRQ(PetscNew(&info));
      CHKERRQ(PetscStrallocpy(fec_type[i],&info->fec_type));
      CHKERRQ(PetscContainerCreate(PetscObjectComm((PetscObject)U),&container));
      CHKERRQ(PetscContainerSetPointer(container,(void*)info));
      CHKERRQ(PetscContainerSetUserDestroy(container,PetscViewerGLVisVecInfoDestroy_Private));
      CHKERRQ(PetscObjectCompose((PetscObject)Ufield[i],"_glvis_info_container",(PetscObject)container));
      CHKERRQ(PetscContainerDestroy(&container));
    }
    /* attach the mesh to the viz vectors */
    CHKERRQ(PetscObjectQuery((PetscObject)Ufield[i], "__PETSc_dm",&fdm));
    if (!fdm) {
      PetscObject dm;

      CHKERRQ(PetscViewerGLVisGetDM_Private(viewer,&dm));
      if (!dm) {
        CHKERRQ(PetscObjectQuery((PetscObject)U, "__PETSc_dm",&dm));
      }
      PetscCheck(dm,PetscObjectComm((PetscObject)U),PETSC_ERR_SUP,"Mesh not present");
      CHKERRQ(PetscObjectCompose((PetscObject)Ufield[i], "__PETSc_dm",dm));
    }
  }

  /* user-provided sampling */
  if (g2lfields) {
    CHKERRQ((*g2lfields)((PetscObject)U,nfields,(PetscObject*)Ufield,userctx));
  } else {
    PetscCheckFalse(nfields > 1,PetscObjectComm((PetscObject)U),PETSC_ERR_SUP,"Don't know how to sample %" PetscInt_FMT " fields",nfields);
    CHKERRQ(VecCopy(U,Ufield[0]));
  }

  /* TODO callback to user routine to disable/enable subdomains */
  for (i=0; i<nfields; i++) {
    PetscObject dm;
    PetscViewer view;

    CHKERRQ(PetscObjectQuery((PetscObject)Ufield[i], "__PETSc_dm",&dm));
    CHKERRQ(PetscViewerGLVisGetWindow_Private(viewer,i,&view));
    if (!view) continue; /* socket window has been closed */
    if (socktype == PETSC_VIEWER_GLVIS_SOCKET) {
      PetscMPIInt size,rank;
      const char *name;

      CHKERRMPI(MPI_Comm_size(PetscObjectComm(dm),&size));
      CHKERRMPI(MPI_Comm_rank(PetscObjectComm(dm),&rank));
      CHKERRQ(PetscObjectGetName((PetscObject)Ufield[i],&name));

      CHKERRQ(PetscGLVisCollectiveBegin(PetscObjectComm(dm),&view));
      CHKERRQ(PetscViewerASCIIPrintf(view,"parallel %d %d\nsolution\n",size,rank));
      CHKERRQ(PetscObjectView(dm,view));
      CHKERRQ(VecView(Ufield[i],view));
      CHKERRQ(PetscViewerGLVisInitWindow_Private(view,PETSC_FALSE,spacedim[i],name));
      CHKERRQ(PetscGLVisCollectiveEnd(PetscObjectComm(dm),&view));
      if (view) pause = PETSC_TRUE; /* at least one window is connected */
    } else {
      CHKERRQ(PetscObjectView(dm,view));
      CHKERRQ(VecView(Ufield[i],view));
    }
    CHKERRQ(PetscViewerGLVisRestoreWindow_Private(viewer,i,&view));
  }
  if (pause) CHKERRQ(PetscViewerGLVisPause_Private(viewer));
  PetscFunctionReturn(0);
}
