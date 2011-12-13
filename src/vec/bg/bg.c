#include <private/bgimpl.h>

#undef __FUNCT__
#define __FUNCT__ "PetscBGCreate"
/*@C
   PetscBGCreate - create a bipartite graph communication context

   Not Collective

   Input Arguments:
.  comm - communicator on which the bipartite graph will operate

   Output Arguments:
.  bg - new bipartite graph context

   Level: intermediate

.seealso: PetscBGSetGraph(), PetscBGDestroy()
@*/
PetscErrorCode PetscBGCreate(MPI_Comm comm,PetscBG *bg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(bg,2);
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = PetscBGInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(*bg,_p_PetscBG,struct _PetscBGOps,PETSCBG_CLASSID,-1,"PetscBG","Bipartite Graph","PetscBG",comm,PetscBGDestroy,PetscBGView);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGReset"
/*@C
   PetscBGReset - Reset a bipartite graph so that different sizes or neighbors can be used

   Collective

   Input Arguments:
.  bg - bipartite graph

   Level: advanced

.seealso: PetscBGCreate(), PetscBGSetGraph(), PetscBGDestroy()
@*/
PetscErrorCode PetscBGReset(PetscBG bg)
{
  PetscErrorCode ierr;
  PetscBGDataLink link,next;
  PetscInt i;

  PetscFunctionBegin;
  ierr = PetscFree(bg->mine_alloc);CHKERRQ(ierr);
  ierr = PetscFree(bg->remote_alloc);CHKERRQ(ierr);
  for (link=bg->link; link; link=next) {
    next = link->next;
    ierr = MPI_Type_free(&link->atom);CHKERRQ(ierr);
    for (i=0; i<bg->nranks; i++) {
      ierr = MPI_Type_free(&link->mine[i]);CHKERRQ(ierr);
      ierr = MPI_Type_free(&link->remote[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree2(link->mine,link->remote);CHKERRQ(ierr);
    ierr = PetscFree(link);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGDestroy"
/*@C
   PetscBGDestroy - destroy bipartite graph

   Collective

   Input Arguments:
.  bg - bipartite graph context

   Level: intermediate

.seealso: PetscBGCreate(), PetscBGReset()
@*/
PetscErrorCode PetscBGDestroy(PetscBG *bg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*bg) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*bg),PETSCBG_CLASSID,1);
  if (--((PetscObject)(*bg))->refct > 0) {*bg = 0; PetscFunctionReturn(0);}
  ierr = PetscBGReset(*bg);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(bg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGView"
/*@
   PetscBGView - view a bipartite graph

   Collective

   Input Arguments:
+  bg - bipartite graph
-  viewer - viewer to display graph, for example PETSC_VIEWER_STDOUT_WORLD

   Level: beginner

.seealso: PetscBGCreate(), PetscBGSetGraph()
@*/
PetscErrorCode PetscBGView(PetscBG bg,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bg,PETSCBG_CLASSID,1);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(((PetscObject)bg)->comm,&viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(bg,1,viewer,2);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscMPIInt rank;
    PetscInt i;
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)bg,viewer,"Bipartite Graph Object");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(((PetscObject)bg)->comm,&rank);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number of outgoing edges=%D, remote ranks=%D\n",rank,bg->nlocal,bg->nranks);CHKERRQ(ierr);
    for (i=0; i<bg->nlocal; i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D -> (%D,%D)\n",rank,bg->mine[i],bg->remote[i].rank,bg->remote[i].index);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
