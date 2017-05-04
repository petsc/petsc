#include <petsc/private/glvisviewerimpl.h>
#include <petsc/private/glvisvecimpl.h>
#include <petsc/private/petscfeimpl.h>
#include <petscdm.h>
#include <petscsf.h>

static PetscErrorCode PetscViewerGLVisVecInfoDestroy_Private(void *ptr)
{
  PetscViewerGLVisVecInfo info = (PetscViewerGLVisVecInfo)ptr;
  PetscErrorCode          ierr;

  PetscFunctionBeginUser;
  ierr = PetscFree(info->fec_type);CHKERRQ(ierr);
  ierr = PetscFree(info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_SETJMP_H) && !defined(PETSC_MISSING_SIGPIPE)
#include <setjmp.h>
#include <signal.h>

#if defined(PETSC_HAVE_WINDOWS_H)
#define DEV_NULL "NUL"
#else
#define DEV_NULL "/dev/null"
#endif

static jmp_buf PetscGLVisSigPipeJmpBuf;

static void PetscGLVisSigPipeHandler(PETSC_UNUSED int sig)
{
  longjmp(PetscGLVisSigPipeJmpBuf,1);
}
#endif

/* the main function to visualize vectors using GLVis */
PetscErrorCode VecView_GLVis(Vec U,PetscViewer viewer)
{
  DM                     dm = NULL;
  PetscErrorCode         (*g2lfields)(PetscObject,PetscInt,PetscObject[],void*);
  Vec                    *Ufield;
  const char             **fec_type,**name;
  PetscViewerGLVisStatus sockstatus;
  PetscViewerGLVisType   socktype;
  void                   *userctx;
  PetscInt               i,nfields,*locandbs;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGLVisGetStatus_Private(viewer,&sockstatus);CHKERRQ(ierr);
  if (sockstatus == PETSCVIEWERGLVIS_DISABLED) PetscFunctionReturn(0);
  /* if the user did not customize the viewer through the API, we need extra data that can be attached to the Vec */
  ierr = PetscViewerGLVisGetFields_Private(viewer,&nfields,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  if (!nfields) {
    ierr = VecGetDM(U,&dm);CHKERRQ(ierr);
    if (dm) {
      ierr = PetscViewerGLVisSetDM_Private(viewer,(PetscObject)dm);CHKERRQ(ierr);
    } else SETERRQ(PetscObjectComm((PetscObject)U),PETSC_ERR_SUP,"You need to provide a DM or use PetscViewerGLVisSetFields()");
  }
  ierr = PetscViewerGLVisGetFields_Private(viewer,&nfields,&name,&fec_type,&locandbs,&g2lfields,(PetscObject**)&Ufield,&userctx);CHKERRQ(ierr);
  ierr = PetscViewerGLVisGetType_Private(viewer,&socktype);CHKERRQ(ierr);
  ierr = PetscViewerGLVisGetDM_Private(viewer,(PetscObject*)&dm);CHKERRQ(ierr);
  if (!dm) {
    ierr = VecGetDM(U,&dm);CHKERRQ(ierr);
  }
  if (!dm) SETERRQ(PetscObjectComm((PetscObject)U),PETSC_ERR_SUP,"Mesh not present");

  if (g2lfields) {
    if (!Ufield[0]) {
      for (i=0;i<nfields;i++) {
        PetscViewerGLVisVecInfo info;
        PetscContainer          container;

        ierr = VecCreateMPI(PetscObjectComm((PetscObject)U),locandbs[2*i],PETSC_DECIDE,&Ufield[i]);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)Ufield[i],name[i]);CHKERRQ(ierr);
        ierr = VecSetBlockSize(Ufield[i],locandbs[2*i+1]);CHKERRQ(ierr);

        /* attach visualization info to the vector */
        ierr = PetscNew(&info);CHKERRQ(ierr);
        ierr = PetscStrallocpy(fec_type[i],&info->fec_type);CHKERRQ(ierr);
        ierr = PetscContainerCreate(PetscObjectComm((PetscObject)U),&container);CHKERRQ(ierr);
        ierr = PetscContainerSetPointer(container,(void*)info);CHKERRQ(ierr);
        ierr = PetscContainerSetUserDestroy(container,PetscViewerGLVisVecInfoDestroy_Private);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)Ufield[i],"_glvis_info_container",(PetscObject)container);CHKERRQ(ierr);
        ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
      }
    }
    ierr = (*g2lfields)((PetscObject)U,nfields,(PetscObject*)Ufield,userctx);CHKERRQ(ierr);
    for (i=0;i<nfields;i++) {
      DM fdm;

      /* attach the mesh to the vector if not done before */
      ierr = VecGetDM(Ufield[i],&fdm);CHKERRQ(ierr);
      if (!fdm) {
        DM udm;

        ierr = VecGetDM(U,&udm);CHKERRQ(ierr);
        if (udm) {
          ierr = VecSetDM(Ufield[i],udm);CHKERRQ(ierr);
        } else {
          ierr = VecSetDM(Ufield[i],dm);CHKERRQ(ierr);
        }
      }
    }
  } else Ufield = &U;

  /* TODO callback to user routine to disable/enable subdomains */
  for (i=0;i<nfields;i++) {
    DM          dmf;
    PetscViewer view;

    ierr = VecGetDM(Ufield[i],&dmf);CHKERRQ(ierr);
    if (!dmf) dmf = dm;
    ierr = PetscViewerGLVisGetWindow_Private(viewer,i,&view);CHKERRQ(ierr);
    if (!view) continue; /* socket window has been closed */
    if (socktype == PETSC_VIEWER_GLVIS_DUMP) {
      ierr = DMView(dmf,view);CHKERRQ(ierr);
      ierr = VecView(Ufield[i],view);CHKERRQ(ierr);
    } else {
      /* It may happen that the user has closed the GLVis window */
#if defined(PETSC_HAVE_SETJMP_H) && !defined(PETSC_MISSING_SIGPIPE)
      void (*sighdl)(int) = signal(SIGPIPE,PetscGLVisSigPipeHandler);
      if (!setjmp(PetscGLVisSigPipeJmpBuf)) {
#endif
        PetscMPIInt size,rank;
        PetscInt    dim;

        ierr = DMGetDimension(dmf,&dim);CHKERRQ(ierr);
        ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dmf),&size);CHKERRQ(ierr);
        ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dmf),&rank);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(view,"parallel %D %D\nsolution\n",size,rank);CHKERRQ(ierr);
        ierr = DMView(dmf,view);CHKERRQ(ierr);
        ierr = VecView(Ufield[i],view);CHKERRQ(ierr);
        ierr = PetscViewerGLVisInitWindow_Private(view,PETSC_FALSE,dim,name[i]);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SETJMP_H) && !defined(PETSC_MISSING_SIGPIPE)
      } else {
        FILE     *sock,*null = fopen(DEV_NULL,"w");
        PetscInt readonly;

        ierr = VecLockGet(Ufield[i],&readonly);CHKERRQ(ierr);
        if (readonly) {
          ierr = VecLockPop(Ufield[i]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIGetPointer(view,&sock);CHKERRQ(ierr);
        ierr = PetscViewerASCIISetFILE(view,null);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
        (void)fclose(sock);
      }
      (void)signal(SIGPIPE,sighdl);
#endif
    }
    ierr = PetscViewerGLVisRestoreWindow_Private(viewer,i,&view);CHKERRQ(ierr);
  }
  ierr = PetscViewerGLVisPause_Private(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
