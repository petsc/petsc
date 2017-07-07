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
  PetscObject            dm;
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
    ierr = PetscObjectQuery((PetscObject)U, "__PETSc_dm",&dm);CHKERRQ(ierr);
    if (dm) {
      ierr = PetscViewerGLVisSetDM_Private(viewer,dm);CHKERRQ(ierr);
    } else SETERRQ(PetscObjectComm((PetscObject)U),PETSC_ERR_SUP,"You need to provide a DM or use PetscViewerGLVisSetFields()");
  }
  ierr = PetscViewerGLVisGetFields_Private(viewer,&nfields,&name,&fec_type,&locandbs,&g2lfields,(PetscObject**)&Ufield,&userctx);CHKERRQ(ierr);
  ierr = PetscViewerGLVisGetType_Private(viewer,&socktype);CHKERRQ(ierr);
  ierr = PetscViewerGLVisGetDM_Private(viewer,&dm);CHKERRQ(ierr);
  if (!dm) {
    ierr = PetscObjectQuery((PetscObject)U, "__PETSc_dm",&dm);CHKERRQ(ierr);
  }
  if (!dm) SETERRQ(PetscObjectComm((PetscObject)U),PETSC_ERR_SUP,"Mesh not present");

  if (!Ufield[0]) {
    for (i=0;i<nfields;i++) {
      PetscViewerGLVisVecInfo info;
      PetscContainer          container;

      ierr = VecCreateMPI(PetscObjectComm((PetscObject)U),locandbs[3*i],PETSC_DECIDE,&Ufield[i]);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)Ufield[i],name[i]);CHKERRQ(ierr);
      ierr = VecSetBlockSize(Ufield[i],locandbs[3*i+1]);CHKERRQ(ierr);

      /* attach visualization info to the vector */
      ierr = PetscNew(&info);CHKERRQ(ierr);
      ierr = PetscStrallocpy(fec_type[i],&info->fec_type);CHKERRQ(ierr);
      ierr = PetscContainerCreate(PetscObjectComm((PetscObject)U),&container);CHKERRQ(ierr);
      ierr = PetscContainerSetPointer(container,(void*)info);CHKERRQ(ierr);
      ierr = PetscContainerSetUserDestroy(container,PetscViewerGLVisVecInfoDestroy_Private);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)Ufield[i],"_glvis_info_container",(PetscObject)container);CHKERRQ(ierr);
      ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);

      /* attach the mesh to the viz vectors */
      ierr = PetscObjectCompose((PetscObject)Ufield[i], "__PETSc_dm",dm);CHKERRQ(ierr);
    }
  }

  /* user-provided sampling */
  if (g2lfields) {
    ierr = (*g2lfields)((PetscObject)U,nfields,(PetscObject*)Ufield,userctx);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(U,Ufield[0]);CHKERRQ(ierr);
  }

  /* TODO callback to user routine to disable/enable subdomains */
  for (i=0;i<nfields;i++) {
    PetscViewer view;

    ierr = PetscObjectQuery((PetscObject)Ufield[i], "__PETSc_dm",&dm);CHKERRQ(ierr);
    ierr = PetscViewerGLVisGetWindow_Private(viewer,i,&view);CHKERRQ(ierr);
    if (!view) continue; /* socket window has been closed */
    if (socktype == PETSC_VIEWER_GLVIS_DUMP) {
      ierr = PetscObjectView(dm,view);CHKERRQ(ierr);
      ierr = VecView(Ufield[i],view);CHKERRQ(ierr);
    } else {
      /* It may happen that the user has closed the GLVis window */
#if defined(PETSC_HAVE_SETJMP_H) && !defined(PETSC_MISSING_SIGPIPE)
      void (*sighdl)(int) = signal(SIGPIPE,PetscGLVisSigPipeHandler);
      if (!setjmp(PetscGLVisSigPipeJmpBuf)) {
#endif
        PetscMPIInt size,rank;

        ierr = MPI_Comm_size(PetscObjectComm(dm),&size);CHKERRQ(ierr);
        ierr = MPI_Comm_rank(PetscObjectComm(dm),&rank);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(view,"parallel %D %D\nsolution\n",size,rank);CHKERRQ(ierr);
        ierr = PetscObjectView(dm,view);CHKERRQ(ierr);
        ierr = VecView(Ufield[i],view);CHKERRQ(ierr);
        ierr = PetscViewerGLVisInitWindow_Private(view,PETSC_FALSE,locandbs[3*i+2],name[i]);CHKERRQ(ierr);
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
