#include <private/bgimpl.h>
#include <petscctable.h>

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
  bg->mine = PETSC_NULL;
  ierr = PetscFree(bg->mine_alloc);CHKERRQ(ierr);
  bg->remote = PETSC_NULL;
  ierr = PetscFree(bg->remote_alloc);CHKERRQ(ierr);
  ierr = PetscFree2(bg->ranks,bg->counts);CHKERRQ(ierr);
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
#define __FUNCT__ "PetscBGSetGraph"
/*@
   PetscBGSetGraph - Set a parallel bipartite graph

   Collective

   Input Arguments:

   Output Arguments:

   Level: intermediate

.seealso:
@*/
PetscErrorCode PetscBGSetGraph(PetscBG bg,PetscInt nlocal,const PetscInt *ilocal,PetscCopyMode localmode,const PetscBGNode *iremote,PetscCopyMode remotemode)
{
  PetscErrorCode ierr;
  PetscTable table;
  PetscTablePosition pos;
  PetscMPIInt size;
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bg,PETSCBG_CLASSID,1);
  if (ilocal) PetscValidIntPointer(ilocal,3);
  PetscValidPointer(iremote,5);
  ierr = PetscBGReset(bg);CHKERRQ(ierr);
  bg->nlocal = nlocal;
  if (ilocal) {
    switch (localmode) {
    case PETSC_COPY_VALUES:
      ierr = PetscMalloc(nlocal*sizeof(*bg->mine),&bg->mine_alloc);CHKERRQ(ierr);
      bg->mine = bg->mine_alloc;
      ierr = PetscMemcpy(bg->mine,ilocal,nlocal*sizeof(*bg->mine));CHKERRQ(ierr);
      break;
    case PETSC_OWN_POINTER:
      bg->mine_alloc = (PetscInt*)ilocal;
      bg->mine = bg->mine_alloc;
      break;
    case PETSC_USE_POINTER:
      bg->mine = (PetscInt*)ilocal;
      break;
    default: SETERRQ(((PetscObject)bg)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Unknown localmode");
    }
  }
  switch (remotemode) {
  case PETSC_COPY_VALUES:
    ierr = PetscMalloc(nlocal*sizeof(*bg->remote),&bg->remote_alloc);CHKERRQ(ierr);
    bg->remote = bg->remote_alloc;
    ierr = PetscMemcpy(bg->remote,iremote,nlocal*sizeof(*bg->remote));CHKERRQ(ierr);
    break;
  case PETSC_OWN_POINTER:
    bg->remote_alloc = (PetscBGNode*)iremote;
    bg->remote = bg->remote_alloc;
    break;
  case PETSC_USE_POINTER:
    bg->remote = (PetscBGNode*)iremote;
    break;
  default: SETERRQ(((PetscObject)bg)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Unknown remotemode");
  }

  ierr = MPI_Comm_size(((PetscObject)bg)->comm,&size);CHKERRQ(ierr);
  ierr = PetscTableCreate(10,size,&table);CHKERRQ(ierr);
  for (i=0; i<nlocal; i++) {
    /* Log 1-based rank */
    ierr = PetscTableAdd(table,iremote[i].rank+1,1,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscTableGetCount(table,&bg->nranks);CHKERRQ(ierr);
  ierr = PetscMalloc2(bg->nranks,PetscInt,&bg->ranks,bg->nranks,PetscInt,&bg->counts);CHKERRQ(ierr);
  ierr = PetscTableGetHeadPosition(table,&pos);CHKERRQ(ierr);
  for (i=0; i<bg->nranks; i++) {
    ierr = PetscTableGetNext(table,&pos,&bg->ranks[i],&bg->counts[i]);CHKERRQ(ierr);
    bg->ranks[i]--;             /* Convert back to 0-based */
  }
  ierr = PetscTableDestroy(&table);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGView"
/*@C
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
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D -> (%D,%D)\n",rank,bg->mine?bg->mine[i]:i,bg->remote[i].rank,bg->remote[i].index);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number of outgoing edges by rank\n",rank);CHKERRQ(ierr);
    for (i=0; i<bg->nranks; i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D: %D edges\n",rank,bg->ranks[i],bg->counts[i]);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
