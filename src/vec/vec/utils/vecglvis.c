#include <petsc/private/glvisviewerimpl.h>
#include <petsc/private/glvisvecimpl.h>

static PetscErrorCode PetscViewerGLVisVecInfoDestroy_Private(void *ptr)
{
  PetscViewerGLVisVecInfo info = (PetscViewerGLVisVecInfo)ptr;

  PetscFunctionBeginUser;
  PetscCall(PetscFree(info->fec_type));
  PetscCall(PetscFree(info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* the main function to visualize vectors using GLVis */
PetscErrorCode VecView_GLVis(Vec U, PetscViewer viewer)
{
  PetscErrorCode (*g2lfields)(PetscObject, PetscInt, PetscObject[], void *);
  Vec                   *Ufield;
  const char           **fec_type;
  PetscViewerGLVisStatus sockstatus;
  PetscViewerGLVisType   socktype;
  void                  *userctx;
  PetscInt               i, nfields, *spacedim;
  PetscBool              pause = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscViewerGLVisGetStatus_Private(viewer, &sockstatus));
  if (sockstatus == PETSCVIEWERGLVIS_DISABLED) PetscFunctionReturn(PETSC_SUCCESS);
  /* if the user did not customize the viewer through the API, we need extra data that can be attached to the Vec */
  PetscCall(PetscViewerGLVisGetFields_Private(viewer, &nfields, NULL, NULL, NULL, NULL, NULL));
  if (!nfields) {
    PetscObject dm;

    PetscCall(PetscObjectQuery((PetscObject)U, "__PETSc_dm", &dm));
    if (dm) {
      PetscCall(PetscViewerGLVisSetDM_Private(viewer, dm));
    } else SETERRQ(PetscObjectComm((PetscObject)U), PETSC_ERR_SUP, "You need to provide a DM or use PetscViewerGLVisSetFields()");
  }
  PetscCall(PetscViewerGLVisGetFields_Private(viewer, &nfields, &fec_type, &spacedim, &g2lfields, (PetscObject **)&Ufield, &userctx));
  if (!nfields) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscViewerGLVisGetType_Private(viewer, &socktype));

  for (i = 0; i < nfields; i++) {
    PetscObject    fdm;
    PetscContainer container;

    /* attach visualization info to the vector */
    PetscCall(PetscObjectQuery((PetscObject)Ufield[i], "_glvis_info_container", (PetscObject *)&container));
    if (!container) {
      PetscViewerGLVisVecInfo info;

      PetscCall(PetscNew(&info));
      PetscCall(PetscStrallocpy(fec_type[i], &info->fec_type));
      PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)U), &container));
      PetscCall(PetscContainerSetPointer(container, (void *)info));
      PetscCall(PetscContainerSetUserDestroy(container, PetscViewerGLVisVecInfoDestroy_Private));
      PetscCall(PetscObjectCompose((PetscObject)Ufield[i], "_glvis_info_container", (PetscObject)container));
      PetscCall(PetscContainerDestroy(&container));
    }
    /* attach the mesh to the viz vectors */
    PetscCall(PetscObjectQuery((PetscObject)Ufield[i], "__PETSc_dm", &fdm));
    if (!fdm) {
      PetscObject dm;

      PetscCall(PetscViewerGLVisGetDM_Private(viewer, &dm));
      if (!dm) PetscCall(PetscObjectQuery((PetscObject)U, "__PETSc_dm", &dm));
      PetscCheck(dm, PetscObjectComm((PetscObject)U), PETSC_ERR_SUP, "Mesh not present");
      PetscCall(PetscObjectCompose((PetscObject)Ufield[i], "__PETSc_dm", dm));
    }
  }

  /* user-provided sampling */
  if (g2lfields) {
    PetscCall((*g2lfields)((PetscObject)U, nfields, (PetscObject *)Ufield, userctx));
  } else {
    PetscCheck(nfields <= 1, PetscObjectComm((PetscObject)U), PETSC_ERR_SUP, "Don't know how to sample %" PetscInt_FMT " fields", nfields);
    PetscCall(VecCopy(U, Ufield[0]));
  }

  /* TODO callback to user routine to disable/enable subdomains */
  for (i = 0; i < nfields; i++) {
    PetscObject dm;
    PetscViewer view;

    PetscCall(PetscObjectQuery((PetscObject)Ufield[i], "__PETSc_dm", &dm));
    PetscCall(PetscViewerGLVisGetWindow_Private(viewer, i, &view));
    if (!view) continue; /* socket window has been closed */
    if (socktype == PETSC_VIEWER_GLVIS_SOCKET) {
      PetscMPIInt size, rank;
      const char *name;

      PetscCallMPI(MPI_Comm_size(PetscObjectComm(dm), &size));
      PetscCallMPI(MPI_Comm_rank(PetscObjectComm(dm), &rank));
      PetscCall(PetscObjectGetName((PetscObject)Ufield[i], &name));

      PetscCall(PetscGLVisCollectiveBegin(PetscObjectComm(dm), &view));
      PetscCall(PetscViewerASCIIPrintf(view, "parallel %d %d\nsolution\n", size, rank));
      PetscCall(PetscObjectView(dm, view));
      PetscCall(VecView(Ufield[i], view));
      PetscCall(PetscViewerGLVisInitWindow_Private(view, PETSC_FALSE, spacedim[i], name));
      PetscCall(PetscGLVisCollectiveEnd(PetscObjectComm(dm), &view));
      if (view) pause = PETSC_TRUE; /* at least one window is connected */
    } else {
      PetscCall(PetscObjectView(dm, view));
      PetscCall(VecView(Ufield[i], view));
    }
    PetscCall(PetscViewerGLVisRestoreWindow_Private(viewer, i, &view));
  }
  if (pause) PetscCall(PetscViewerGLVisPause_Private(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}
