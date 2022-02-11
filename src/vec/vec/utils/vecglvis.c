#include <petsc/private/glvisviewerimpl.h>
#include <petsc/private/glvisvecimpl.h>

static PetscErrorCode PetscViewerGLVisVecInfoDestroy_Private(void *ptr)
{
  PetscViewerGLVisVecInfo info = (PetscViewerGLVisVecInfo)ptr;
  PetscErrorCode          ierr;

  PetscFunctionBeginUser;
  ierr = PetscFree(info->fec_type);CHKERRQ(ierr);
  ierr = PetscFree(info);CHKERRQ(ierr);
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
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGLVisGetStatus_Private(viewer,&sockstatus);CHKERRQ(ierr);
  if (sockstatus == PETSCVIEWERGLVIS_DISABLED) PetscFunctionReturn(0);
  /* if the user did not customize the viewer through the API, we need extra data that can be attached to the Vec */
  ierr = PetscViewerGLVisGetFields_Private(viewer,&nfields,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  if (!nfields) {
    PetscObject dm;

    ierr = PetscObjectQuery((PetscObject)U, "__PETSc_dm",&dm);CHKERRQ(ierr);
    if (dm) {
      ierr = PetscViewerGLVisSetDM_Private(viewer,dm);CHKERRQ(ierr);
    } else SETERRQ(PetscObjectComm((PetscObject)U),PETSC_ERR_SUP,"You need to provide a DM or use PetscViewerGLVisSetFields()");
  }
  ierr = PetscViewerGLVisGetFields_Private(viewer,&nfields,&fec_type,&spacedim,&g2lfields,(PetscObject**)&Ufield,&userctx);CHKERRQ(ierr);
  if (!nfields) PetscFunctionReturn(0);

  ierr = PetscViewerGLVisGetType_Private(viewer,&socktype);CHKERRQ(ierr);

  for (i=0;i<nfields;i++) {
    PetscObject    fdm;
    PetscContainer container;

    /* attach visualization info to the vector */
    ierr = PetscObjectQuery((PetscObject)Ufield[i],"_glvis_info_container",(PetscObject*)&container);CHKERRQ(ierr);
    if (!container) {
      PetscViewerGLVisVecInfo info;

      ierr = PetscNew(&info);CHKERRQ(ierr);
      ierr = PetscStrallocpy(fec_type[i],&info->fec_type);CHKERRQ(ierr);
      ierr = PetscContainerCreate(PetscObjectComm((PetscObject)U),&container);CHKERRQ(ierr);
      ierr = PetscContainerSetPointer(container,(void*)info);CHKERRQ(ierr);
      ierr = PetscContainerSetUserDestroy(container,PetscViewerGLVisVecInfoDestroy_Private);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)Ufield[i],"_glvis_info_container",(PetscObject)container);CHKERRQ(ierr);
      ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
    }
    /* attach the mesh to the viz vectors */
    ierr = PetscObjectQuery((PetscObject)Ufield[i], "__PETSc_dm",&fdm);CHKERRQ(ierr);
    if (!fdm) {
      PetscObject dm;

      ierr = PetscViewerGLVisGetDM_Private(viewer,&dm);CHKERRQ(ierr);
      if (!dm) {
        ierr = PetscObjectQuery((PetscObject)U, "__PETSc_dm",&dm);CHKERRQ(ierr);
      }
      PetscCheckFalse(!dm,PetscObjectComm((PetscObject)U),PETSC_ERR_SUP,"Mesh not present");
      ierr = PetscObjectCompose((PetscObject)Ufield[i], "__PETSc_dm",dm);CHKERRQ(ierr);
    }
  }

  /* user-provided sampling */
  if (g2lfields) {
    ierr = (*g2lfields)((PetscObject)U,nfields,(PetscObject*)Ufield,userctx);CHKERRQ(ierr);
  } else {
    PetscCheckFalse(nfields > 1,PetscObjectComm((PetscObject)U),PETSC_ERR_SUP,"Don't know how to sample %" PetscInt_FMT " fields",nfields);
    ierr = VecCopy(U,Ufield[0]);CHKERRQ(ierr);
  }

  /* TODO callback to user routine to disable/enable subdomains */
  for (i=0; i<nfields; i++) {
    PetscObject dm;
    PetscViewer view;

    ierr = PetscObjectQuery((PetscObject)Ufield[i], "__PETSc_dm",&dm);CHKERRQ(ierr);
    ierr = PetscViewerGLVisGetWindow_Private(viewer,i,&view);CHKERRQ(ierr);
    if (!view) continue; /* socket window has been closed */
    if (socktype == PETSC_VIEWER_GLVIS_SOCKET) {
      PetscMPIInt size,rank;
      const char *name;

      ierr = MPI_Comm_size(PetscObjectComm(dm),&size);CHKERRMPI(ierr);
      ierr = MPI_Comm_rank(PetscObjectComm(dm),&rank);CHKERRMPI(ierr);
      ierr = PetscObjectGetName((PetscObject)Ufield[i],&name);CHKERRQ(ierr);

      ierr = PetscGLVisCollectiveBegin(PetscObjectComm(dm),&view);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(view,"parallel %d %d\nsolution\n",size,rank);CHKERRQ(ierr);
      ierr = PetscObjectView(dm,view);CHKERRQ(ierr);
      ierr = VecView(Ufield[i],view);CHKERRQ(ierr);
      ierr = PetscViewerGLVisInitWindow_Private(view,PETSC_FALSE,spacedim[i],name);CHKERRQ(ierr);
      ierr = PetscGLVisCollectiveEnd(PetscObjectComm(dm),&view);CHKERRQ(ierr);
      if (view) pause = PETSC_TRUE; /* at least one window is connected */
    } else {
      ierr = PetscObjectView(dm,view);CHKERRQ(ierr);
      ierr = VecView(Ufield[i],view);CHKERRQ(ierr);
    }
    ierr = PetscViewerGLVisRestoreWindow_Private(viewer,i,&view);CHKERRQ(ierr);
  }
  if (pause) {ierr = PetscViewerGLVisPause_Private(viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
