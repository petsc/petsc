#include <petscsnes.h> 
#include <petsc-private/dmimpl.h>     /*I      "petscdm.h"     I*/

PetscClassId  DM_CLASSID;
PetscLogEvent DM_Convert, DM_GlobalToLocal, DM_LocalToGlobal;

#undef __FUNCT__  
#define __FUNCT__ "DMCreate"
/*@
  DMCreate - Creates an empty DM object. The type can then be set with DMSetType().

   If you never  call DMSetType()  it will generate an
   error when you try to use the vector.

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the DM object

  Output Parameter:
. dm - The DM object

  Level: beginner

.seealso: DMSetType(), DMDA, DMSLICED, DMCOMPOSITE
@*/
PetscErrorCode  DMCreate(MPI_Comm comm,DM *dm)
{
  DM             v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm,2);
  *dm = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = VecInitializePackage(PETSC_NULL);CHKERRQ(ierr);
  ierr = MatInitializePackage(PETSC_NULL);CHKERRQ(ierr);
  ierr = DMInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(v, _p_DM, struct _DMOps, DM_CLASSID, -1, "DM", "Distribution Manager", "DM", comm, DMDestroy, DMView);CHKERRQ(ierr);
  ierr = PetscMemzero(v->ops, sizeof(struct _DMOps));CHKERRQ(ierr);


  v->workSize     = 0;
  v->workArray    = PETSC_NULL;
  v->ltogmap      = PETSC_NULL;
  v->ltogmapb     = PETSC_NULL;
  v->bs           = 1;
  v->coloringtype = IS_COLORING_GLOBAL;
  v->lf           = PETSC_NULL;
  v->lj           = PETSC_NULL;
  ierr = PetscSFCreate(comm, &v->sf);CHKERRQ(ierr);
  ierr = PetscSFCreate(comm, &v->defaultSF);CHKERRQ(ierr);
  v->defaultSection       = PETSC_NULL;
  v->defaultGlobalSection = PETSC_NULL;

  *dm = v;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DMSetVecType"
/*@C
       DMSetVecType - Sets the type of vector created with DMCreateLocalVector() and DMCreateGlobalVector()

   Logically Collective on DMDA

   Input Parameter:
+  da - initial distributed array
.  ctype - the vector type, currently either VECSTANDARD or VECCUSP

   Options Database:
.   -dm_vec_type ctype

   Level: intermediate

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDestroy(), DMDA, DMDAInterpolationType, VecType
@*/
PetscErrorCode  DMSetVecType(DM da,const VecType ctype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  ierr = PetscFree(da->vectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(ctype,&da->vectype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMSetMatType"
/*@C
       DMSetMatType - Sets the type of matrix created with DMCreateMatrix()

   Logically Collective on DM

   Input Parameter:
+  dm - the DM context
.  ctype - the matrix type

   Options Database:
.   -dm_mat_type ctype

   Level: intermediate

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMCreateMatrix(), DMSetMatrixPreallocateOnly(), MatType
@*/
PetscErrorCode  DMSetMatType(DM dm,const MatType ctype)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscFree(dm->mattype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(ctype,&dm->mattype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMSetOptionsPrefix"
/*@C
   DMSetOptionsPrefix - Sets the prefix used for searching for all 
   DMDA options in the database.

   Logically Collective on DMDA

   Input Parameter:
+  da - the DMDA context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.keywords: DMDA, set, options, prefix, database

.seealso: DMSetFromOptions()
@*/
PetscErrorCode  DMSetOptionsPrefix(DM dm,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)dm,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMDestroy"
/*@
    DMDestroy - Destroys a vector packer or DMDA.

    Collective on DM

    Input Parameter:
.   dm - the DM object to destroy

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix()

@*/
PetscErrorCode  DMDestroy(DM *dm)
{
  PetscInt       i, cnt = 0;
  DMNamedVecLink nlink,nnext;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*dm) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*dm),DM_CLASSID,1);

  /* count all the circular references of DM and its contained Vecs */
  for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
    if ((*dm)->localin[i])  {cnt++;}
    if ((*dm)->globalin[i]) {cnt++;}
  }
  for (nlink=(*dm)->namedglobal; nlink; nlink=nlink->next) cnt++;
  if ((*dm)->x) {
    PetscObject obj;
    ierr = PetscObjectQuery((PetscObject)(*dm)->x,"DM",&obj);CHKERRQ(ierr);
    if (obj == (PetscObject)*dm) cnt++;
  }

  if (--((PetscObject)(*dm))->refct - cnt > 0) {*dm = 0; PetscFunctionReturn(0);}
  /*
     Need this test because the dm references the vectors that
     reference the dm, so destroying the dm calls destroy on the
     vectors that cause another destroy on the dm
  */
  if (((PetscObject)(*dm))->refct < 0) PetscFunctionReturn(0);
  ((PetscObject) (*dm))->refct = 0;
  for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
    if ((*dm)->localout[i]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Destroying a DM that has a local vector obtained with DMGetLocalVector()");
    ierr = VecDestroy(&(*dm)->localin[i]);CHKERRQ(ierr);
  }
  for (nlink=(*dm)->namedglobal; nlink; nlink=nnext) { /* Destroy the named vectors */
    nnext = nlink->next;
    if (nlink->status != DMVEC_STATUS_IN) SETERRQ1(((PetscObject)*dm)->comm,PETSC_ERR_ARG_WRONGSTATE,"DM still has Vec named '%s' checked out",nlink->name);
    ierr = PetscFree(nlink->name);CHKERRQ(ierr);
    ierr = VecDestroy(&nlink->X);CHKERRQ(ierr);
    ierr = PetscFree(nlink);CHKERRQ(ierr);
  }
  (*dm)->namedglobal = PETSC_NULL;

  /* Destroy the list of hooks */
  {
    DMCoarsenHookLink link,next;
    for (link=(*dm)->coarsenhook; link; link=next) {
      next = link->next;
      ierr = PetscFree(link);CHKERRQ(ierr);
    }
    (*dm)->coarsenhook = PETSC_NULL;
  }
  {
    DMRefineHookLink link,next;
    for (link=(*dm)->refinehook; link; link=next) {
      next = link->next;
      ierr = PetscFree(link);CHKERRQ(ierr);
    }
    (*dm)->refinehook = PETSC_NULL;
  }

  if ((*dm)->ctx && (*dm)->ctxdestroy) {
    ierr = (*(*dm)->ctxdestroy)(&(*dm)->ctx);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&(*dm)->x);CHKERRQ(ierr);
  ierr = MatFDColoringDestroy(&(*dm)->fd);CHKERRQ(ierr);
  ierr = DMClearGlobalVectors(*dm);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&(*dm)->ltogmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&(*dm)->ltogmapb);CHKERRQ(ierr);
  ierr = PetscFree((*dm)->vectype);CHKERRQ(ierr);
  ierr = PetscFree((*dm)->mattype);CHKERRQ(ierr);
  ierr = PetscFree((*dm)->workArray);CHKERRQ(ierr);

  ierr = PetscSectionDestroy(&(*dm)->defaultSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&(*dm)->defaultGlobalSection);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&(*dm)->sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&(*dm)->defaultSF);CHKERRQ(ierr);
  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(*dm);CHKERRQ(ierr);

  ierr = (*(*dm)->ops->destroy)(*dm);CHKERRQ(ierr);
  ierr = PetscFree((*dm)->data);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMSetUp"
/*@
    DMSetUp - sets up the data structures inside a DM object

    Collective on DM

    Input Parameter:
.   dm - the DM object to setup

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix()

@*/
PetscErrorCode  DMSetUp(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (dm->setupcalled) PetscFunctionReturn(0);
  if (dm->ops->setup) {
    ierr = (*dm->ops->setup)(dm);CHKERRQ(ierr);
  }
  dm->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMSetFromOptions"
/*@
    DMSetFromOptions - sets parameters in a DM from the options database

    Collective on DM

    Input Parameter:
.   dm - the DM object to set options for

    Options Database:
+   -dm_preallocate_only: Only preallocate the matrix for DMCreateMatrix(), but do not fill it with zeros
.   -dm_vec_type <type>  type of vector to create inside DM
.   -dm_mat_type <type>  type of matrix to create inside DM
-   -dm_coloring_type <global or ghosted> 

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix()

@*/
PetscErrorCode  DMSetFromOptions(DM dm)
{
  PetscBool      flg1 = PETSC_FALSE,flg2 = PETSC_FALSE,flg3 = PETSC_FALSE,flg4 = PETSC_FALSE,flg;
  PetscErrorCode ierr;
  char           typeName[256] = MATAIJ;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectOptionsBegin((PetscObject)dm);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-dm_view", "Information on DM", "DMView", flg1, &flg1, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-dm_view_detail", "Exhaustive mesh description", "DMView", flg2, &flg2, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-dm_view_vtk", "Output mesh in VTK format", "DMView", flg3, &flg3, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-dm_view_latex", "Output mesh in LaTeX TikZ format", "DMView", flg4, &flg4, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-dm_preallocate_only","only preallocate matrix, but do not set column indices","DMSetMatrixPreallocateOnly",dm->prealloc_only,&dm->prealloc_only,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsList("-dm_vec_type","Vector type used for created vectors","DMSetVecType",VecList,dm->vectype,typeName,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = DMSetVecType(dm,typeName);CHKERRQ(ierr);
    }
    ierr = PetscOptionsList("-dm_mat_type","Matrix type used for created matrices","DMSetMatType",MatList,dm->mattype?dm->mattype:typeName,typeName,sizeof typeName,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = DMSetMatType(dm,typeName);CHKERRQ(ierr);
    }
    ierr = PetscOptionsEnum("-dm_is_coloring_type","Global or local coloring of Jacobian","ISColoringType",ISColoringTypes,(PetscEnum)dm->coloringtype,(PetscEnum*)&dm->coloringtype,PETSC_NULL);CHKERRQ(ierr);
    if (dm->ops->setfromoptions) {
      ierr = (*dm->ops->setfromoptions)(dm);CHKERRQ(ierr);
    }
    /* process any options handlers added with PetscObjectAddOptionsHandler() */
    ierr = PetscObjectProcessOptionsHandlers((PetscObject) dm);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (flg1) {
    ierr = DMView(dm, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  if (flg2) {
    PetscViewer viewer;

    ierr = PetscViewerCreate(((PetscObject) dm)->comm, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
    ierr = DMView(dm, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  if (flg3) {
    PetscViewer viewer;

    ierr = PetscViewerCreate(((PetscObject) dm)->comm, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, "mesh.vtk");CHKERRQ(ierr);
    ierr = DMView(dm, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  if (flg4) {
    PetscViewer viewer;

    ierr = PetscViewerCreate(((PetscObject) dm)->comm, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_LATEX);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, "mesh.tex");CHKERRQ(ierr);
    ierr = DMView(dm, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMView"
/*@C
    DMView - Views a vector packer or DMDA.

    Collective on DM

    Input Parameter:
+   dm - the DM object to view
-   v - the viewer

    Level: developer

.seealso DMDestroy(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix()

@*/
PetscErrorCode  DMView(DM dm,PetscViewer v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
 if (!v) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)dm)->comm,&v);CHKERRQ(ierr);
  }
  if (dm->ops->view) {
    ierr = (*dm->ops->view)(dm,v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateGlobalVector"
/*@
    DMCreateGlobalVector - Creates a global vector from a DMDA or DMComposite object

    Collective on DM

    Input Parameter:
.   dm - the DM object

    Output Parameter:
.   vec - the global vector

    Level: beginner

.seealso DMDestroy(), DMView(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix()

@*/
PetscErrorCode  DMCreateGlobalVector(DM dm,Vec *vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (dm->defaultSection) {
    PetscSection gSection;
    PetscInt     localSize;

    ierr = DMGetDefaultGlobalSection(dm, &gSection);CHKERRQ(ierr);
    ierr = PetscSectionGetConstrainedStorageSize(dm->defaultGlobalSection, &localSize);CHKERRQ(ierr);
    ierr = VecCreate(((PetscObject) dm)->comm, vec);CHKERRQ(ierr);
    ierr = VecSetSizes(*vec, localSize, PETSC_DETERMINE);CHKERRQ(ierr);
    /* ierr = VecSetType(*vec, dm->vectype);CHKERRQ(ierr); */
    ierr = VecSetFromOptions(*vec);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) *vec, "DM", (PetscObject) dm);CHKERRQ(ierr);
    /* ierr = VecSetLocalToGlobalMapping(*vec, dm->ltogmap);CHKERRQ(ierr); */
    /* ierr = VecSetLocalToGlobalMappingBlock(*vec, dm->ltogmapb);CHKERRQ(ierr); */
    /* ierr = VecSetOperation(*vec, VECOP_DUPLICATE, (void(*)(void)) VecDuplicate_MPI_DM);CHKERRQ(ierr); */
  } else {
    ierr = (*dm->ops->createglobalvector)(dm,vec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateLocalVector"
/*@
    DMCreateLocalVector - Creates a local vector from a DMDA or DMComposite object

    Not Collective

    Input Parameter:
.   dm - the DM object

    Output Parameter:
.   vec - the local vector

    Level: beginner

.seealso DMDestroy(), DMView(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix()

@*/
PetscErrorCode  DMCreateLocalVector(DM dm,Vec *vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (dm->defaultSection) {
    PetscInt localSize;

    ierr = PetscSectionGetStorageSize(dm->defaultSection, &localSize);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_SELF, vec);CHKERRQ(ierr);
    ierr = VecSetSizes(*vec, localSize, localSize);CHKERRQ(ierr);
    ierr = VecSetFromOptions(*vec);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) *vec, "DM", (PetscObject) dm);CHKERRQ(ierr);
  } else {
    ierr = (*dm->ops->createlocalvector)(dm,vec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGetLocalToGlobalMapping"
/*@
   DMGetLocalToGlobalMapping - Accesses the local-to-global mapping in a DM.

   Collective on DM

   Input Parameter:
.  dm - the DM that provides the mapping

   Output Parameter:
.  ltog - the mapping

   Level: intermediate

   Notes:
   This mapping can then be used by VecSetLocalToGlobalMapping() or
   MatSetLocalToGlobalMapping().

.seealso: DMCreateLocalVector(), DMGetLocalToGlobalMappingBlock()
@*/
PetscErrorCode  DMGetLocalToGlobalMapping(DM dm,ISLocalToGlobalMapping *ltog)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(ltog,2);
  if (!dm->ltogmap) {
    PetscSection section, sectionGlobal;

    ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
    if (section) {
      PetscInt      *ltog;
      PetscInt       pStart, pEnd, size, p, l;

      ierr = DMGetDefaultGlobalSection(dm, &sectionGlobal);CHKERRQ(ierr);
      ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
      ierr = PetscSectionGetStorageSize(section, &size);CHKERRQ(ierr);
      ierr = PetscMalloc(size * sizeof(PetscInt), &ltog);CHKERRQ(ierr); /* We want the local+overlap size */
      for(p = pStart, l = 0; p < pEnd; ++p) {
        PetscInt dof, off, c;

        /* Should probably use constrained dofs */
        ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(sectionGlobal, p, &off);CHKERRQ(ierr);
        for(c = 0; c < dof; ++c, ++l) {
          ltog[l] = off+c;
        }
      }
      ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, size, ltog, PETSC_OWN_POINTER, &dm->ltogmap);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(dm, dm->ltogmap);CHKERRQ(ierr);
    } else {
      if (!dm->ops->createlocaltoglobalmapping) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"DM can not create LocalToGlobalMapping");
      ierr = (*dm->ops->createlocaltoglobalmapping)(dm);CHKERRQ(ierr);
    }
  }
  *ltog = dm->ltogmap;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGetLocalToGlobalMappingBlock"
/*@
   DMGetLocalToGlobalMappingBlock - Accesses the blocked local-to-global mapping in a DM.

   Collective on DM

   Input Parameter:
.  da - the distributed array that provides the mapping

   Output Parameter:
.  ltog - the block mapping

   Level: intermediate

   Notes:
   This mapping can then be used by VecSetLocalToGlobalMappingBlock() or
   MatSetLocalToGlobalMappingBlock().

.seealso: DMCreateLocalVector(), DMGetLocalToGlobalMapping(), DMGetBlockSize(), VecSetBlockSize(), MatSetBlockSize()
@*/
PetscErrorCode  DMGetLocalToGlobalMappingBlock(DM dm,ISLocalToGlobalMapping *ltog)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(ltog,2);
  if (!dm->ltogmapb) {
    PetscInt bs;
    ierr = DMGetBlockSize(dm,&bs);CHKERRQ(ierr);
    if (bs > 1) {
      if (!dm->ops->createlocaltoglobalmappingblock) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"DM can not create LocalToGlobalMappingBlock");
      ierr = (*dm->ops->createlocaltoglobalmappingblock)(dm);CHKERRQ(ierr);
    } else {
      ierr = DMGetLocalToGlobalMapping(dm,&dm->ltogmapb);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)dm->ltogmapb);CHKERRQ(ierr);
    }
  }
  *ltog = dm->ltogmapb;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGetBlockSize"
/*@
   DMGetBlockSize - Gets the inherent block size associated with a DM

   Not Collective

   Input Parameter:
.  dm - the DM with block structure

   Output Parameter:
.  bs - the block size, 1 implies no exploitable block structure

   Level: intermediate

.seealso: ISCreateBlock(), VecSetBlockSize(), MatSetBlockSize(), DMGetLocalToGlobalMappingBlock()
@*/
PetscErrorCode  DMGetBlockSize(DM dm,PetscInt *bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(bs,2);
  if (dm->bs < 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"DM does not have enough information to provide a block size yet");
  *bs = dm->bs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateInterpolation"
/*@
    DMCreateInterpolation - Gets interpolation matrix between two DMDA or DMComposite objects

    Collective on DM

    Input Parameter:
+   dm1 - the DM object
-   dm2 - the second, finer DM object

    Output Parameter:
+  mat - the interpolation
-  vec - the scaling (optional)

    Level: developer

    Notes:  For DMDA objects this only works for "uniform refinement", that is the refined mesh was obtained DMRefine() or the coarse mesh was obtained by 
        DMCoarsen(). The coordinates set into the DMDA are completely ignored in computing the interpolation.

        For DMDA objects you can use this interpolation (more precisely the interpolation from the DMDAGetCoordinateDA()) to interpolate the mesh coordinate vectors
        EXCEPT in the periodic case where it does not make sense since the coordinate vectors are not periodic.
   

.seealso DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateColoring(), DMCreateMatrix(), DMRefine(), DMCoarsen()

@*/
PetscErrorCode  DMCreateInterpolation(DM dm1,DM dm2,Mat *mat,Vec *vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm1,DM_CLASSID,1);
  PetscValidHeaderSpecific(dm2,DM_CLASSID,2);
  ierr = (*dm1->ops->createinterpolation)(dm1,dm2,mat,vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateInjection"
/*@
    DMCreateInjection - Gets injection matrix between two DMDA or DMComposite objects

    Collective on DM

    Input Parameter:
+   dm1 - the DM object
-   dm2 - the second, finer DM object

    Output Parameter:
.   ctx - the injection

    Level: developer

   Notes:  For DMDA objects this only works for "uniform refinement", that is the refined mesh was obtained DMRefine() or the coarse mesh was obtained by 
        DMCoarsen(). The coordinates set into the DMDA are completely ignored in computing the injection.

.seealso DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateColoring(), DMCreateMatrix(), DMCreateInterpolation()

@*/
PetscErrorCode  DMCreateInjection(DM dm1,DM dm2,VecScatter *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm1,DM_CLASSID,1);
  PetscValidHeaderSpecific(dm2,DM_CLASSID,2);
  ierr = (*dm1->ops->getinjection)(dm1,dm2,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateColoring"
/*@C
    DMCreateColoring - Gets coloring for a DMDA or DMComposite

    Collective on DM

    Input Parameter:
+   dm - the DM object
.   ctype - IS_COLORING_GHOSTED or IS_COLORING_GLOBAL
-   matype - either MATAIJ or MATBAIJ

    Output Parameter:
.   coloring - the coloring

    Level: developer

.seealso DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateMatrix()

@*/
PetscErrorCode  DMCreateColoring(DM dm,ISColoringType ctype,const MatType mtype,ISColoring *coloring)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (!dm->ops->getcoloring) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"No coloring for this type of DM yet");
  ierr = (*dm->ops->getcoloring)(dm,ctype,mtype,coloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateMatrix"
/*@C
    DMCreateMatrix - Gets empty Jacobian for a DMDA or DMComposite

    Collective on DM

    Input Parameter:
+   dm - the DM object
-   mtype - Supported types are MATSEQAIJ, MATMPIAIJ, MATSEQBAIJ, MATMPIBAIJ, or
            any type which inherits from one of these (such as MATAIJ)

    Output Parameter:
.   mat - the empty Jacobian 

    Level: beginner

    Notes: This properly preallocates the number of nonzeros in the sparse matrix so you 
       do not need to do it yourself. 

       By default it also sets the nonzero structure and puts in the zero entries. To prevent setting 
       the nonzero pattern call DMDASetMatPreallocateOnly()

       For structured grid problems, when you call MatView() on this matrix it is displayed using the global natural ordering, NOT in the ordering used
       internally by PETSc.

       For structured grid problems, in general it is easiest to use MatSetValuesStencil() or MatSetValuesLocal() to put values into the matrix because MatSetValues() requires 
       the indices for the global numbering for DMDAs which is complicated.

.seealso DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation()

@*/
PetscErrorCode  DMCreateMatrix(DM dm,const MatType mtype,Mat *mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = MatInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(mat,3);
  if (dm->mattype) {
    ierr = (*dm->ops->creatematrix)(dm,dm->mattype,mat);CHKERRQ(ierr);
  } else {
    ierr = (*dm->ops->creatematrix)(dm,mtype,mat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetMatrixPreallocateOnly"
/*@
  DMSetMatrixPreallocateOnly - When DMCreateMatrix() is called the matrix will be properly
    preallocated but the nonzero structure and zero values will not be set.

  Logically Collective on DMDA

  Input Parameter:
+ dm - the DM
- only - PETSC_TRUE if only want preallocation

  Level: developer
.seealso DMCreateMatrix()
@*/
PetscErrorCode DMSetMatrixPreallocateOnly(DM dm, PetscBool only)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->prealloc_only = only;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGetWorkArray"
/*@C
  DMGetWorkArray - Gets a work array guaranteed to be at least the input size

  Not Collective

  Input Parameters:
+ dm - the DM object
- size - The minium size

  Output Parameter:
. array - the work array

  Level: developer

.seealso DMDestroy(), DMCreate()
@*/
PetscErrorCode DMGetWorkArray(DM dm,PetscInt size,PetscScalar **array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(array,3);
  if (size > dm->workSize) {
    dm->workSize = size;
    ierr = PetscFree(dm->workArray);CHKERRQ(ierr);
    ierr = PetscMalloc(dm->workSize * sizeof(PetscScalar), &dm->workArray);CHKERRQ(ierr);
  }
  *array = dm->workArray;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DMCreateFieldIS"
/*@C
  DMCreateFieldIS - Creates a set of IS objects with the global indices of dofs for each field

  Not collective

  Input Parameter:
. dm - the DM object

  Output Parameters:
+ numFields  - The number of fields (or PETSC_NULL if not requested)
. fieldNames - The name for each field (or PETSC_NULL if not requested)
- fields     - The global indices for each field (or PETSC_NULL if not requested)

  Level: intermediate

  Notes:
  The user is responsible for freeing all requested arrays. In particular, every entry of names should be freed with
  PetscFree(), every entry of fields should be destroyed with ISDestroy(), and both arrays should be freed with
  PetscFree().

.seealso DMDestroy(), DMView(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix()
@*/
PetscErrorCode DMCreateFieldIS(DM dm, PetscInt *numFields, char ***fieldNames, IS **fields)
{
  PetscSection   section, sectionGlobal;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (numFields) {
    PetscValidPointer(numFields,2);
    *numFields = 0;
  }
  if (fieldNames) {
    PetscValidPointer(fieldNames,3);
    *fieldNames = PETSC_NULL;
  }
  if (fields) {
    PetscValidPointer(fields,4);
    *fields = PETSC_NULL;
  }
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  if (section) {
    PetscInt *fieldSizes, **fieldIndices;
    PetscInt  nF, f, pStart, pEnd, p;

    ierr = DMGetDefaultGlobalSection(dm, &sectionGlobal);CHKERRQ(ierr);
    ierr = PetscSectionGetNumFields(section, &nF);CHKERRQ(ierr);
    ierr = PetscMalloc2(nF,PetscInt,&fieldSizes,nF,PetscInt *,&fieldIndices);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(sectionGlobal, &pStart, &pEnd);CHKERRQ(ierr);
    for(f = 0; f < nF; ++f) {
      fieldSizes[f] = 0;
    }
    for(p = pStart; p < pEnd; ++p) {
      PetscInt gdof;

      ierr = PetscSectionGetDof(sectionGlobal, p, &gdof);CHKERRQ(ierr);
      if (gdof > 0) {
        for(f = 0; f < nF; ++f) {
          PetscInt fdof, fcdof;

          ierr = PetscSectionGetFieldDof(section, p, f, &fdof);CHKERRQ(ierr);
          ierr = PetscSectionGetFieldConstraintDof(section, p, f, &fcdof);CHKERRQ(ierr);
          fieldSizes[f] += fdof-fcdof;
        }
      }
    }
    for(f = 0; f < nF; ++f) {
      ierr = PetscMalloc(fieldSizes[f] * sizeof(PetscInt), &fieldIndices[f]);CHKERRQ(ierr);
      fieldSizes[f] = 0;
    }
    for(p = pStart; p < pEnd; ++p) {
      PetscInt gdof, goff;

      ierr = PetscSectionGetDof(sectionGlobal, p, &gdof);CHKERRQ(ierr);
      if (gdof > 0) {
        ierr = PetscSectionGetOffset(sectionGlobal, p, &goff);CHKERRQ(ierr);
        for(f = 0; f < nF; ++f) {
          PetscInt fdof, fcdof, fc;

          ierr = PetscSectionGetFieldDof(section, p, f, &fdof);CHKERRQ(ierr);
          ierr = PetscSectionGetFieldConstraintDof(section, p, f, &fcdof);CHKERRQ(ierr);
          for(fc = 0; fc < fdof-fcdof; ++fc, ++fieldSizes[f]) {
            fieldIndices[f][fieldSizes[f]] = goff++;
          }
        }
      }
    }
    if (numFields) {*numFields = nF;}
    if (fieldNames) {
      ierr = PetscMalloc(nF * sizeof(char *), fieldNames);CHKERRQ(ierr);
      for(f = 0; f < nF; ++f) {
        const char *fieldName;

        ierr = PetscSectionGetFieldName(section, f, &fieldName);CHKERRQ(ierr);
        ierr = PetscStrallocpy(fieldName, (char **) &(*fieldNames)[f]);CHKERRQ(ierr);
      }
    }
    if (fields) {
      ierr = PetscMalloc(nF * sizeof(IS), fields);CHKERRQ(ierr);
      for(f = 0; f < nF; ++f) {
        ierr = ISCreateGeneral(((PetscObject) dm)->comm, fieldSizes[f], fieldIndices[f], PETSC_OWN_POINTER, &(*fields)[f]);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree2(fieldSizes,fieldIndices);CHKERRQ(ierr);
  } else {
    if(dm->ops->createfieldis) {ierr = (*dm->ops->createfieldis)(dm, numFields, fieldNames, fields);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMCreateFieldDecompositionDM"
/*@C
  DMCreateFieldDecompositionDM - creates a DM that encapsulates a decomposition of the original DM into fields.

  Not Collective

  Input Parameters:
+ dm   - the DM object
- name - the name of the field decomposition

  Output Parameter:
. ddm  - the field decomposition DM (PETSC_NULL, if no such decomposition is known)

  Level: advanced

.seealso DMDestroy(), DMCreate(), DMCreateFieldDecomposition(), DMCreateDomainDecompositionDM()
@*/
PetscErrorCode DMCreateFieldDecompositionDM(DM dm, const char* name, DM *ddm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidCharPointer(name,2);
  PetscValidPointer(ddm,3);
  *ddm = PETSC_NULL;
  if(dm->ops->createfielddecompositiondm) {
    ierr = (*dm->ops->createfielddecompositiondm)(dm,name,ddm); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DMCreateFieldDecomposition"
/*@C
  DMCreateFieldDecomposition - Returns a list of IS objects defining a decomposition of a problem into subproblems 
                          corresponding to different fields: each IS contains the global indices of the dofs of the 
                          corresponding field. The optional list of DMs define the DM for each subproblem.
                          Generalizes DMCreateFieldIS().

  Not collective

  Input Parameter:
. dm - the DM object

  Output Parameters:
+ len       - The number of subproblems in the field decomposition (or PETSC_NULL if not requested)
. namelist  - The name for each field (or PETSC_NULL if not requested)
. islist    - The global indices for each field (or PETSC_NULL if not requested)
- dmlist    - The DMs for each field subproblem (or PETSC_NULL, if not requested; if PETSC_NULL is returned, no DMs are defined)

  Level: intermediate

  Notes:
  The user is responsible for freeing all requested arrays. In particular, every entry of names should be freed with
  PetscFree(), every entry of is should be destroyed with ISDestroy(), every entry of dm should be destroyed with DMDestroy(),
  and all of the arrays should be freed with PetscFree().

.seealso DMDestroy(), DMView(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMCreateFieldIS()
@*/
PetscErrorCode DMCreateFieldDecomposition(DM dm, PetscInt *len, char ***namelist, IS **islist, DM **dmlist)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (len)      {PetscValidPointer(len,2);      *len      = 0;}
  if (namelist) {PetscValidPointer(namelist,3); *namelist = 0;}
  if (islist)   {PetscValidPointer(islist,4);   *islist   = 0;}
  if (dmlist)   {PetscValidPointer(dmlist,5);   *dmlist   = 0;}
  if(!dm->ops->createfielddecomposition) {
    ierr = DMCreateFieldIS(dm, len, namelist, islist);CHKERRQ(ierr);
    /* By default there are no DMs associated with subproblems. */
    if(dmlist) *dmlist = PETSC_NULL;
  }
  else {
    ierr = (*dm->ops->createfielddecomposition)(dm,len,namelist,islist,dmlist); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateDomainDecompositionDM"
/*@C
  DMCreateDomainDecompositionDM - creates a DM that encapsulates a decomposition of the original DM into subdomains.

  Not Collective

  Input Parameters:
+ dm   - the DM object
- name - the name of the subdomain decomposition

  Output Parameter:
. ddm  - the subdomain decomposition DM (PETSC_NULL, if no such decomposition is known)

  Level: advanced

.seealso DMDestroy(), DMCreate(), DMCreateFieldDecomposition(), DMCreateDomainDecompositionDM()
@*/
PetscErrorCode DMCreateDomainDecompositionDM(DM dm, const char* name, DM *ddm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidCharPointer(name,2);
  PetscValidPointer(ddm,3);
  *ddm = PETSC_NULL;
  if(dm->ops->createdomaindecompositiondm) {
    ierr = (*dm->ops->createdomaindecompositiondm)(dm,name,ddm); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DMCreateDomainDecomposition"
/*@C
  DMCreateDomainDecomposition - Returns lists of IS objects defining a decomposition of a problem into subproblems 
                          corresponding to restrictions to pairs nested subdomains: each IS contains the global 
                          indices of the dofs of the corresponding subdomains.  The inner subdomains conceptually
                          define a nonoverlapping covering, while outer subdomains can overlap.
                          The optional list of DMs define the DM for each subproblem.

  Not collective

  Input Parameter:
. dm - the DM object

  Output Parameters:
+ len         - The number of subproblems in the domain decomposition (or PETSC_NULL if not requested)
. namelist    - The name for each subdomain (or PETSC_NULL if not requested)
. innerislist - The global indices for each inner subdomain (or PETSC_NULL, if not requested)
. outerislist - The global indices for each outer subdomain (or PETSC_NULL, if not requested)
- dmlist      - The DMs for each subdomain subproblem (or PETSC_NULL, if not requested; if PETSC_NULL is returned, no DMs are defined)

  Level: intermediate

  Notes:
  The user is responsible for freeing all requested arrays. In particular, every entry of names should be freed with
  PetscFree(), every entry of is should be destroyed with ISDestroy(), every entry of dm should be destroyed with DMDestroy(),
  and all of the arrays should be freed with PetscFree().

.seealso DMDestroy(), DMView(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMCreateDomainDecompositionDM(), DMCreateFieldDecomposition()
@*/
PetscErrorCode DMCreateDomainDecomposition(DM dm, PetscInt *len, char ***namelist, IS **innerislist, IS **outerislist, DM **dmlist)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (len)           {PetscValidPointer(len,2);            *len         = PETSC_NULL;}
  if (namelist)      {PetscValidPointer(namelist,3);       *namelist    = PETSC_NULL;}
  if (innerislist)   {PetscValidPointer(innerislist,4);    *innerislist = PETSC_NULL;}
  if (outerislist)   {PetscValidPointer(outerislist,5);    *outerislist = PETSC_NULL;}
  if (dmlist)        {PetscValidPointer(dmlist,6);         *dmlist      = PETSC_NULL;}
  if(dm->ops->createdomaindecomposition) {
    ierr = (*dm->ops->createdomaindecomposition)(dm,len,namelist,innerislist,outerislist,dmlist); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMRefine"
/*@
  DMRefine - Refines a DM object

  Collective on DM

  Input Parameter:
+ dm   - the DM object
- comm - the communicator to contain the new DM object (or MPI_COMM_NULL)

  Output Parameter:
. dmf - the refined DM, or PETSC_NULL

  Note: If no refinement was done, the return value is PETSC_NULL

  Level: developer

.seealso DMCoarsen(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation()
@*/
PetscErrorCode  DMRefine(DM dm,MPI_Comm comm,DM *dmf)
{
  PetscErrorCode ierr;
  DMRefineHookLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = (*dm->ops->refine)(dm,comm,dmf);CHKERRQ(ierr);
  if (*dmf) {
    (*dmf)->ops->creatematrix = dm->ops->creatematrix;
    (*dmf)->ops->initialguess = dm->ops->initialguess;
    (*dmf)->ops->function     = dm->ops->function;
    (*dmf)->ops->functionj    = dm->ops->functionj;
    if (dm->ops->jacobian != DMComputeJacobianDefault) {
      (*dmf)->ops->jacobian     = dm->ops->jacobian;
    }
    ierr = PetscObjectCopyFortranFunctionPointers((PetscObject)dm,(PetscObject)*dmf);CHKERRQ(ierr);
    (*dmf)->ctx       = dm->ctx;
    (*dmf)->leveldown = dm->leveldown;
    (*dmf)->levelup   = dm->levelup + 1;
    ierr = DMSetMatType(*dmf,dm->mattype);CHKERRQ(ierr);
    for (link=dm->refinehook; link; link=link->next) {
      if (link->refinehook) {ierr = (*link->refinehook)(dm,*dmf,link->ctx);CHKERRQ(ierr);}
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRefineHookAdd"
/*@
   DMRefineHookAdd - adds a callback to be run when interpolating a nonlinear problem to a finer grid

   Logically Collective

   Input Arguments:
+  coarse - nonlinear solver context on which to run a hook when restricting to a coarser level
.  refinehook - function to run when setting up a coarser level
.  interphook - function to run to update data on finer levels (once per SNESSolve())
-  ctx - [optional] user-defined context for provide data for the hooks (may be PETSC_NULL)

   Calling sequence of refinehook:
$    refinehook(DM coarse,DM fine,void *ctx);

+  coarse - coarse level DM
.  fine - fine level DM to interpolate problem to
-  ctx - optional user-defined function context

   Calling sequence for interphook:
$    interphook(DM coarse,Mat interp,DM fine,void *ctx)

+  coarse - coarse level DM
.  interp - matrix interpolating a coarse-level solution to the finer grid
.  fine - fine level DM to update
-  ctx - optional user-defined function context

   Level: advanced

   Notes:
   This function is only needed if auxiliary data needs to be passed to fine grids while grid sequencing

   If this function is called multiple times, the hooks will be run in the order they are added.

.seealso: DMCoarsenHookAdd(), SNESFASGetInterpolation(), SNESFASGetInjection(), PetscObjectCompose(), PetscContainerCreate()
@*/
PetscErrorCode DMRefineHookAdd(DM coarse,PetscErrorCode (*refinehook)(DM,DM,void*),PetscErrorCode (*interphook)(DM,Mat,DM,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMRefineHookLink link,*p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarse,DM_CLASSID,1);
  for (p=&coarse->refinehook; *p; p=&(*p)->next) {} /* Scan to the end of the current list of hooks */
  ierr = PetscMalloc(sizeof(struct _DMRefineHookLink),&link);CHKERRQ(ierr);
  link->refinehook = refinehook;
  link->interphook = interphook;
  link->ctx = ctx;
  link->next = PETSC_NULL;
  *p = link;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMInterpolate"
/*@
   DMInterpolate - interpolates user-defined problem data to a finer DM by running hooks registered by DMRefineHookAdd()

   Collective if any hooks are

   Input Arguments:
+  coarse - coarser DM to use as a base
.  restrct - interpolation matrix, apply using MatInterpolate()
-  fine - finer DM to update

   Level: developer

.seealso: DMRefineHookAdd(), MatInterpolate()
@*/
PetscErrorCode DMInterpolate(DM coarse,Mat interp,DM fine)
{
  PetscErrorCode ierr;
  DMRefineHookLink link;

  PetscFunctionBegin;
  for (link=fine->refinehook; link; link=link->next) {
    if (link->interphook) {ierr = (*link->interphook)(coarse,interp,fine,link->ctx);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGetRefineLevel"
/*@
    DMGetRefineLevel - Get's the number of refinements that have generated this DM.

    Not Collective

    Input Parameter:
.   dm - the DM object

    Output Parameter:
.   level - number of refinements

    Level: developer

.seealso DMCoarsen(), DMGetCoarsenLevel(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation()

@*/
PetscErrorCode  DMGetRefineLevel(DM dm,PetscInt *level)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *level = dm->levelup;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGlobalToLocalBegin"
/*@
    DMGlobalToLocalBegin - Begins updating local vectors from global vector

    Neighbor-wise Collective on DM

    Input Parameters:
+   dm - the DM object
.   g - the global vector
.   mode - INSERT_VALUES or ADD_VALUES
-   l - the local vector


    Level: beginner

.seealso DMCoarsen(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMGlobalToLocalEnd(), DMLocalToGlobalBegin()

@*/
PetscErrorCode  DMGlobalToLocalBegin(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscSF        sf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDefaultSF(dm, &sf);CHKERRQ(ierr);
  if (sf) {
    PetscScalar *lArray, *gArray;

    if (mode == ADD_VALUES) SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid insertion mode %D", mode);
    ierr = VecGetArray(l, &lArray);CHKERRQ(ierr);
    ierr = VecGetArray(g, &gArray);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(sf, MPIU_SCALAR, gArray, lArray);CHKERRQ(ierr);
    ierr = VecRestoreArray(l, &lArray);CHKERRQ(ierr);
    ierr = VecRestoreArray(g, &gArray);CHKERRQ(ierr);
  } else {
    ierr = (*dm->ops->globaltolocalbegin)(dm,g,mode == INSERT_ALL_VALUES ? INSERT_VALUES : (mode == ADD_ALL_VALUES ? ADD_VALUES : mode),l);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGlobalToLocalEnd"
/*@
    DMGlobalToLocalEnd - Ends updating local vectors from global vector

    Neighbor-wise Collective on DM

    Input Parameters:
+   dm - the DM object
.   g - the global vector
.   mode - INSERT_VALUES or ADD_VALUES
-   l - the local vector


    Level: beginner

.seealso DMCoarsen(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMGlobalToLocalEnd(), DMLocalToGlobalBegin()

@*/
PetscErrorCode  DMGlobalToLocalEnd(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscSF        sf;
  PetscErrorCode ierr;
  PetscScalar    *lArray, *gArray;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDefaultSF(dm, &sf);CHKERRQ(ierr);
  if (sf) {
  if (mode == ADD_VALUES) SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid insertion mode %D", mode);

    ierr = VecGetArray(l, &lArray);CHKERRQ(ierr);
    ierr = VecGetArray(g, &gArray);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf, MPIU_SCALAR, gArray, lArray);CHKERRQ(ierr);
    ierr = VecRestoreArray(l, &lArray);CHKERRQ(ierr);
    ierr = VecRestoreArray(g, &gArray);CHKERRQ(ierr);
  } else {
    ierr = (*dm->ops->globaltolocalend)(dm,g,mode == INSERT_ALL_VALUES ? INSERT_VALUES : (mode == ADD_ALL_VALUES ? ADD_VALUES : mode),l);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMLocalToGlobalBegin"
/*@
    DMLocalToGlobalBegin - updates global vectors from local vectors

    Neighbor-wise Collective on DM

    Input Parameters:
+   dm - the DM object
.   l - the local vector
.   mode - if INSERT_VALUES then no parallel communication is used, if ADD_VALUES then all ghost points from the same base point accumulate into that
           base point.
- - the global vector

    Notes: In the ADD_VALUES case you normally would zero the receiving vector before beginning this operation. If you would like to simply add the non-ghosted values in the local
           array into the global array you need to either (1) zero the ghosted locations and use ADD_VALUES or (2) use INSERT_VALUES into a work global array and then add the work
           global array to the final global array with VecAXPY().

    Level: beginner

.seealso DMCoarsen(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMGlobalToLocalEnd(), DMGlobalToLocalBegin()

@*/
PetscErrorCode  DMLocalToGlobalBegin(DM dm,Vec l,InsertMode mode,Vec g)
{
  PetscSF        sf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDefaultSF(dm, &sf);CHKERRQ(ierr);
  if (sf) {
    MPI_Op       op;
    PetscScalar *lArray, *gArray;

    switch(mode) {
    case INSERT_VALUES:
    case INSERT_ALL_VALUES:
#if defined(PETSC_HAVE_MPI_REPLACE)
      op = MPI_REPLACE; break;
#else
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"No support for INSERT_VALUES without an MPI-2 implementation");
#endif
    case ADD_VALUES:
    case ADD_ALL_VALUES:
      op = MPI_SUM; break;
  default:
    SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid insertion mode %D", mode);
    }
    ierr = VecGetArray(l, &lArray);CHKERRQ(ierr);
    ierr = VecGetArray(g, &gArray);CHKERRQ(ierr);
    ierr = PetscSFReduceBegin(sf, MPIU_SCALAR, lArray, gArray, op);CHKERRQ(ierr);
    ierr = VecRestoreArray(l, &lArray);CHKERRQ(ierr);
    ierr = VecRestoreArray(g, &gArray);CHKERRQ(ierr);
  } else {
    ierr = (*dm->ops->localtoglobalbegin)(dm,l,mode == INSERT_ALL_VALUES ? INSERT_VALUES : (mode == ADD_ALL_VALUES ? ADD_VALUES : mode),g);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMLocalToGlobalEnd"
/*@
    DMLocalToGlobalEnd - updates global vectors from local vectors

    Neighbor-wise Collective on DM

    Input Parameters:
+   dm - the DM object
.   l - the local vector
.   mode - INSERT_VALUES or ADD_VALUES
-   g - the global vector


    Level: beginner

.seealso DMCoarsen(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMGlobalToLocalEnd(), DMGlobalToLocalEnd()

@*/
PetscErrorCode  DMLocalToGlobalEnd(DM dm,Vec l,InsertMode mode,Vec g)
{
  PetscSF        sf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDefaultSF(dm, &sf);CHKERRQ(ierr);
  if (sf) {
    MPI_Op       op;
    PetscScalar *lArray, *gArray;

    switch(mode) {
    case INSERT_VALUES:
    case INSERT_ALL_VALUES:
#if defined(PETSC_HAVE_MPI_REPLACE)
      op = MPI_REPLACE; break;
#else
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"No support for INSERT_VALUES without an MPI-2 implementation");
#endif
    case ADD_VALUES:
    case ADD_ALL_VALUES:
      op = MPI_SUM; break;
    default:
      SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid insertion mode %D", mode);
    }
    ierr = VecGetArray(l, &lArray);CHKERRQ(ierr);
    ierr = VecGetArray(g, &gArray);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(sf, MPIU_SCALAR, lArray, gArray, op);CHKERRQ(ierr);
    ierr = VecRestoreArray(l, &lArray);CHKERRQ(ierr);
    ierr = VecRestoreArray(g, &gArray);CHKERRQ(ierr);
  } else {
    ierr = (*dm->ops->localtoglobalend)(dm,l,mode == INSERT_ALL_VALUES ? INSERT_VALUES : (mode == ADD_ALL_VALUES ? ADD_VALUES : mode),g);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMComputeJacobianDefault"
/*@
    DMComputeJacobianDefault - computes the Jacobian using the DMComputeFunction() if Jacobian computer is not provided

    Collective on DM

    Input Parameter:
+   dm - the DM object 
.   x - location to compute Jacobian at; may be ignored for linear problems
.   A - matrix that defines the operator for the linear solve
-   B - the matrix used to construct the preconditioner

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMGetApplicationContext(), DMSetInitialGuess(), 
         DMSetFunction()

@*/
PetscErrorCode  DMComputeJacobianDefault(DM dm,Vec x,Mat A,Mat B,MatStructure *stflag)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *stflag = SAME_NONZERO_PATTERN;
  ierr  = MatFDColoringApply(B,dm->fd,x,stflag,dm);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCoarsen"
/*@
    DMCoarsen - Coarsens a DM object

    Collective on DM

    Input Parameter:
+   dm - the DM object
-   comm - the communicator to contain the new DM object (or MPI_COMM_NULL)

    Output Parameter:
.   dmc - the coarsened DM

    Level: developer

.seealso DMRefine(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation()

@*/
PetscErrorCode  DMCoarsen(DM dm, MPI_Comm comm, DM *dmc)
{
  PetscErrorCode ierr;
  DMCoarsenHookLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = (*dm->ops->coarsen)(dm, comm, dmc);CHKERRQ(ierr);
  (*dmc)->ops->creatematrix = dm->ops->creatematrix;
  (*dmc)->ops->initialguess = dm->ops->initialguess;
  (*dmc)->ops->function     = dm->ops->function;
  (*dmc)->ops->functionj    = dm->ops->functionj;
  if (dm->ops->jacobian != DMComputeJacobianDefault) {
    (*dmc)->ops->jacobian     = dm->ops->jacobian;
  }
  ierr = PetscObjectCopyFortranFunctionPointers((PetscObject)dm,(PetscObject)*dmc);CHKERRQ(ierr);
  (*dmc)->ctx       = dm->ctx;
  (*dmc)->levelup   = dm->levelup;
  (*dmc)->leveldown = dm->leveldown + 1;
  ierr = DMSetMatType(*dmc,dm->mattype);CHKERRQ(ierr);
  for (link=dm->coarsenhook; link; link=link->next) {
    if (link->coarsenhook) {ierr = (*link->coarsenhook)(dm,*dmc,link->ctx);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCoarsenHookAdd"
/*@
   DMCoarsenHookAdd - adds a callback to be run when restricting a nonlinear problem to the coarse grid

   Logically Collective

   Input Arguments:
+  fine - nonlinear solver context on which to run a hook when restricting to a coarser level
.  coarsenhook - function to run when setting up a coarser level
.  restricthook - function to run to update data on coarser levels (once per SNESSolve())
-  ctx - [optional] user-defined context for provide data for the hooks (may be PETSC_NULL)

   Calling sequence of coarsenhook:
$    coarsenhook(DM fine,DM coarse,void *ctx);

+  fine - fine level DM
.  coarse - coarse level DM to restrict problem to
-  ctx - optional user-defined function context

   Calling sequence for restricthook:
$    restricthook(DM fine,Mat mrestrict,Vec rscale,Mat inject,DM coarse,void *ctx)

+  fine - fine level DM
.  mrestrict - matrix restricting a fine-level solution to the coarse grid
.  rscale - scaling vector for restriction
.  inject - matrix restricting by injection
.  coarse - coarse level DM to update
-  ctx - optional user-defined function context

   Level: advanced

   Notes:
   This function is only needed if auxiliary data needs to be set up on coarse grids.

   If this function is called multiple times, the hooks will be run in the order they are added.

   In order to compose with nonlinear preconditioning without duplicating storage, the hook should be implemented to
   extract the finest level information from its context (instead of from the SNES).

.seealso: DMRefineHookAdd(), SNESFASGetInterpolation(), SNESFASGetInjection(), PetscObjectCompose(), PetscContainerCreate()
@*/
PetscErrorCode DMCoarsenHookAdd(DM fine,PetscErrorCode (*coarsenhook)(DM,DM,void*),PetscErrorCode (*restricthook)(DM,Mat,Vec,Mat,DM,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMCoarsenHookLink link,*p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fine,DM_CLASSID,1);
  for (p=&fine->coarsenhook; *p; p=&(*p)->next) {} /* Scan to the end of the current list of hooks */
  ierr = PetscMalloc(sizeof(struct _DMCoarsenHookLink),&link);CHKERRQ(ierr);
  link->coarsenhook = coarsenhook;
  link->restricthook = restricthook;
  link->ctx = ctx;
  link->next = PETSC_NULL;
  *p = link;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRestrict"
/*@
   DMRestrict - restricts user-defined problem data to a coarser DM by running hooks registered by DMCoarsenHookAdd()

   Collective if any hooks are

   Input Arguments:
+  fine - finer DM to use as a base
.  restrct - restriction matrix, apply using MatRestrict()
.  inject - injection matrix, also use MatRestrict()
-  coarse - coarer DM to update

   Level: developer

.seealso: DMCoarsenHookAdd(), MatRestrict()
@*/
PetscErrorCode DMRestrict(DM fine,Mat restrct,Vec rscale,Mat inject,DM coarse)
{
  PetscErrorCode ierr;
  DMCoarsenHookLink link;

  PetscFunctionBegin;
  for (link=fine->coarsenhook; link; link=link->next) {
    if (link->restricthook) {ierr = (*link->restricthook)(fine,restrct,rscale,inject,coarse,link->ctx);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGetCoarsenLevel"
/*@
    DMGetCoarsenLevel - Get's the number of coarsenings that have generated this DM.

    Not Collective

    Input Parameter:
.   dm - the DM object

    Output Parameter:
.   level - number of coarsenings

    Level: developer

.seealso DMCoarsen(), DMGetRefineLevel(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation()

@*/
PetscErrorCode  DMGetCoarsenLevel(DM dm,PetscInt *level)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *level = dm->leveldown;
  PetscFunctionReturn(0);
}



#undef __FUNCT__  
#define __FUNCT__ "DMRefineHierarchy"
/*@C
    DMRefineHierarchy - Refines a DM object, all levels at once

    Collective on DM

    Input Parameter:
+   dm - the DM object
-   nlevels - the number of levels of refinement

    Output Parameter:
.   dmf - the refined DM hierarchy

    Level: developer

.seealso DMCoarsenHierarchy(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation()

@*/
PetscErrorCode  DMRefineHierarchy(DM dm,PetscInt nlevels,DM dmf[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (nlevels < 0) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_ARG_OUTOFRANGE,"nlevels cannot be negative");
  if (nlevels == 0) PetscFunctionReturn(0);
  if (dm->ops->refinehierarchy) {
    ierr = (*dm->ops->refinehierarchy)(dm,nlevels,dmf);CHKERRQ(ierr);
  } else if (dm->ops->refine) {
    PetscInt i;

    ierr = DMRefine(dm,((PetscObject)dm)->comm,&dmf[0]);CHKERRQ(ierr);
    for (i=1; i<nlevels; i++) {
      ierr = DMRefine(dmf[i-1],((PetscObject)dm)->comm,&dmf[i]);CHKERRQ(ierr);
    }
  } else {
    SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"No RefineHierarchy for this DM yet");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCoarsenHierarchy"
/*@C
    DMCoarsenHierarchy - Coarsens a DM object, all levels at once

    Collective on DM

    Input Parameter:
+   dm - the DM object
-   nlevels - the number of levels of coarsening

    Output Parameter:
.   dmc - the coarsened DM hierarchy

    Level: developer

.seealso DMRefineHierarchy(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation()

@*/
PetscErrorCode  DMCoarsenHierarchy(DM dm, PetscInt nlevels, DM dmc[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (nlevels < 0) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_ARG_OUTOFRANGE,"nlevels cannot be negative");
  if (nlevels == 0) PetscFunctionReturn(0);
  PetscValidPointer(dmc,3);
  if (dm->ops->coarsenhierarchy) {
    ierr = (*dm->ops->coarsenhierarchy)(dm, nlevels, dmc);CHKERRQ(ierr);
  } else if (dm->ops->coarsen) {
    PetscInt i;

    ierr = DMCoarsen(dm,((PetscObject)dm)->comm,&dmc[0]);CHKERRQ(ierr);
    for (i=1; i<nlevels; i++) {
      ierr = DMCoarsen(dmc[i-1],((PetscObject)dm)->comm,&dmc[i]);CHKERRQ(ierr);
    }
  } else {
    SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"No CoarsenHierarchy for this DM yet");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateAggregates"
/*@
   DMCreateAggregates - Gets the aggregates that map between 
   grids associated with two DMs.

   Collective on DM

   Input Parameters:
+  dmc - the coarse grid DM
-  dmf - the fine grid DM

   Output Parameters:
.  rest - the restriction matrix (transpose of the projection matrix)

   Level: intermediate

.keywords: interpolation, restriction, multigrid 

.seealso: DMRefine(), DMCreateInjection(), DMCreateInterpolation()
@*/
PetscErrorCode  DMCreateAggregates(DM dmc, DM dmf, Mat *rest) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmc,DM_CLASSID,1);
  PetscValidHeaderSpecific(dmf,DM_CLASSID,2);
  ierr = (*dmc->ops->getaggregates)(dmc, dmf, rest);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMSetApplicationContextDestroy"
/*@C
    DMSetApplicationContextDestroy - Sets a user function that will be called to destroy the application context when the DM is destroyed

    Not Collective

    Input Parameters:
+   dm - the DM object 
-   destroy - the destroy function

    Level: intermediate

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMGetApplicationContext()

@*/
PetscErrorCode  DMSetApplicationContextDestroy(DM dm,PetscErrorCode (*destroy)(void**))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->ctxdestroy = destroy;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMSetApplicationContext"
/*@
    DMSetApplicationContext - Set a user context into a DM object

    Not Collective

    Input Parameters:
+   dm - the DM object 
-   ctx - the user context

    Level: intermediate

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMGetApplicationContext()

@*/
PetscErrorCode  DMSetApplicationContext(DM dm,void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->ctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGetApplicationContext"
/*@
    DMGetApplicationContext - Gets a user context from a DM object

    Not Collective

    Input Parameter:
.   dm - the DM object 

    Output Parameter:
.   ctx - the user context

    Level: intermediate

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMGetApplicationContext()

@*/
PetscErrorCode  DMGetApplicationContext(DM dm,void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *(void**)ctx = dm->ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMSetInitialGuess"
/*@C
    DMSetInitialGuess - sets a function to compute an initial guess vector entries for the solvers

    Logically Collective on DM

    Input Parameter:
+   dm - the DM object to destroy
-   f - the function to compute the initial guess

    Level: intermediate

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMGetApplicationContext(), DMSetFunction(), DMSetJacobian()

@*/
PetscErrorCode  DMSetInitialGuess(DM dm,PetscErrorCode (*f)(DM,Vec))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->ops->initialguess = f;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMSetFunction"
/*@C
    DMSetFunction - sets a function to compute the right hand side vector entries for the KSP solver or nonlinear function for SNES

    Logically Collective on DM

    Input Parameter:
+   dm - the DM object 
-   f - the function to compute (use PETSC_NULL to cancel a previous function that was set)

    Level: intermediate

    Notes: This sets both the function for function evaluations and the function used to compute Jacobians via finite differences if no Jacobian 
           computer is provided with DMSetJacobian(). Canceling cancels the function, but not the function used to compute the Jacobian.

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMGetApplicationContext(), DMSetInitialGuess(),
         DMSetJacobian()

@*/
PetscErrorCode  DMSetFunction(DM dm,PetscErrorCode (*f)(DM,Vec,Vec))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->ops->function = f;
  if (f) {
    dm->ops->functionj = f;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMSetJacobian"
/*@C
    DMSetJacobian - sets a function to compute the matrix entries for the KSP solver or Jacobian for SNES

    Logically Collective on DM

    Input Parameter:
+   dm - the DM object to destroy
-   f - the function to compute the matrix entries

    Level: intermediate

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMGetApplicationContext(), DMSetInitialGuess(), 
         DMSetFunction()

@*/
PetscErrorCode  DMSetJacobian(DM dm,PetscErrorCode (*f)(DM,Vec,Mat,Mat,MatStructure*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->ops->jacobian = f;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMSetVariableBounds"
/*@C
    DMSetVariableBounds - sets a function to compute the the lower and upper bound vectors for SNESVI.

    Logically Collective on DM

    Input Parameter:
+   dm - the DM object 
-   f - the function that computes variable bounds used by SNESVI (use PETSC_NULL to cancel a previous function that was set)

    Level: intermediate

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMGetApplicationContext(), DMSetInitialGuess(),
         DMSetJacobian()

@*/
PetscErrorCode  DMSetVariableBounds(DM dm,PetscErrorCode (*f)(DM,Vec,Vec))
{
  PetscFunctionBegin;
  dm->ops->computevariablebounds = f;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMHasVariableBounds"
/*@
    DMHasVariableBounds - does the DM object have a variable bounds function?

    Not Collective

    Input Parameter:
.   dm - the DM object to destroy

    Output Parameter:
.   flg - PETSC_TRUE if the variable bounds function exists

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMGetApplicationContext(), DMSetFunction(), DMSetJacobian()

@*/
PetscErrorCode  DMHasVariableBounds(DM dm,PetscBool  *flg)
{
  PetscFunctionBegin;
  *flg =  (dm->ops->computevariablebounds) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMComputeVariableBounds"
/*@C
    DMComputeVariableBounds - compute variable bounds used by SNESVI.

    Logically Collective on DM

    Input Parameters:
+   dm - the DM object to destroy
-   x  - current solution at which the bounds are computed

    Output parameters:
+   xl - lower bound
-   xu - upper bound

    Level: intermediate

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMGetApplicationContext(), DMSetInitialGuess(), 
         DMSetFunction(), DMSetVariableBounds()

@*/
PetscErrorCode  DMComputeVariableBounds(DM dm, Vec xl, Vec xu)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(xl,VEC_CLASSID,2); 
  PetscValidHeaderSpecific(xu,VEC_CLASSID,2); 
  if(dm->ops->computevariablebounds) {
    ierr = (*dm->ops->computevariablebounds)(dm, xl,xu); CHKERRQ(ierr);
  }
  else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "This DM is incapable of computing variable bounds.");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComputeInitialGuess"
/*@
    DMComputeInitialGuess - computes an initial guess vector entries for the KSP solvers

    Collective on DM

    Input Parameter:
+   dm - the DM object to destroy
-   x - the vector to hold the initial guess values

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMGetApplicationContext(), DMSetRhs(), DMSetMat()

@*/
PetscErrorCode  DMComputeInitialGuess(DM dm,Vec x)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (!dm->ops->initialguess) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_ARG_WRONGSTATE,"Need to provide function with DMSetInitialGuess()");
  ierr = (*dm->ops->initialguess)(dm,x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMHasInitialGuess"
/*@
    DMHasInitialGuess - does the DM object have an initial guess function

    Not Collective

    Input Parameter:
.   dm - the DM object to destroy

    Output Parameter:
.   flg - PETSC_TRUE if function exists

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMGetApplicationContext(), DMSetFunction(), DMSetJacobian()

@*/
PetscErrorCode  DMHasInitialGuess(DM dm,PetscBool  *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *flg =  (dm->ops->initialguess) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMHasFunction"
/*@
    DMHasFunction - does the DM object have a function

    Not Collective

    Input Parameter:
.   dm - the DM object to destroy

    Output Parameter:
.   flg - PETSC_TRUE if function exists

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMGetApplicationContext(), DMSetFunction(), DMSetJacobian()

@*/
PetscErrorCode  DMHasFunction(DM dm,PetscBool  *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *flg =  (dm->ops->function) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMHasJacobian"
/*@
    DMHasJacobian - does the DM object have a matrix function

    Not Collective

    Input Parameter:
.   dm - the DM object to destroy

    Output Parameter:
.   flg - PETSC_TRUE if function exists

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMGetApplicationContext(), DMSetFunction(), DMSetJacobian()

@*/
PetscErrorCode  DMHasJacobian(DM dm,PetscBool  *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *flg =  (dm->ops->jacobian) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMSetVec"
/*@C
    DMSetVec - set the vector at which to compute residual, Jacobian and VI bounds, if the problem is nonlinear.

    Collective on DM

    Input Parameter:
+   dm - the DM object 
-   x - location to compute residual and Jacobian, if PETSC_NULL is passed to those routines; will be PETSC_NULL for linear problems.

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMGetApplicationContext(), DMSetInitialGuess(), 
         DMSetFunction(), DMSetJacobian(), DMSetVariableBounds()

@*/
PetscErrorCode  DMSetVec(DM dm,Vec x) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (x) {
    if (!dm->x) {
      ierr = DMCreateGlobalVector(dm,&dm->x);CHKERRQ(ierr);
    }
    ierr = VecCopy(x,dm->x);CHKERRQ(ierr);
  }
  else if(dm->x) {
    ierr = VecDestroy(&dm->x);  CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DMComputeFunction"
/*@
    DMComputeFunction - computes the right hand side vector entries for the KSP solver or nonlinear function for SNES

    Collective on DM

    Input Parameter:
+   dm - the DM object to destroy
.   x - the location where the function is evaluationed, may be ignored for linear problems
-   b - the vector to hold the right hand side entries

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMGetApplicationContext(), DMSetInitialGuess(),
         DMSetJacobian()

@*/
PetscErrorCode  DMComputeFunction(DM dm,Vec x,Vec b)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (!dm->ops->function) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_ARG_WRONGSTATE,"Need to provide function with DMSetFunction()");
  PetscStackPush("DM user function");
  ierr = (*dm->ops->function)(dm,x,b);CHKERRQ(ierr);
  PetscStackPop;
  PetscFunctionReturn(0);
}



#undef __FUNCT__ 
#define __FUNCT__ "DMComputeJacobian"
/*@
    DMComputeJacobian - compute the matrix entries for the solver

    Collective on DM

    Input Parameter:
+   dm - the DM object 
.   x - location to compute Jacobian at; will be PETSC_NULL for linear problems, for nonlinear problems if not provided then pulled from DM
.   A - matrix that defines the operator for the linear solve
-   B - the matrix used to construct the preconditioner

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMGetApplicationContext(), DMSetInitialGuess(), 
         DMSetFunction()

@*/
PetscErrorCode  DMComputeJacobian(DM dm,Vec x,Mat A,Mat B,MatStructure *stflag)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (!dm->ops->jacobian) {
    ISColoring     coloring;
    MatFDColoring  fd;
    const MatType  mtype;
    
    ierr = PetscObjectGetType((PetscObject)B,&mtype);CHKERRQ(ierr);
    ierr = DMCreateColoring(dm,dm->coloringtype,mtype,&coloring);CHKERRQ(ierr);
    ierr = MatFDColoringCreate(B,coloring,&fd);CHKERRQ(ierr);
    ierr = ISColoringDestroy(&coloring);CHKERRQ(ierr);
    ierr = MatFDColoringSetFunction(fd,(PetscErrorCode (*)(void))dm->ops->functionj,dm);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject)fd,((PetscObject)dm)->prefix);CHKERRQ(ierr);
    ierr = MatFDColoringSetFromOptions(fd);CHKERRQ(ierr);

    dm->fd = fd;
    dm->ops->jacobian = DMComputeJacobianDefault;

    /* don't know why this is needed */
    ierr = PetscObjectDereference((PetscObject)dm);CHKERRQ(ierr);
  }
  if (!x) x = dm->x;
  ierr = (*dm->ops->jacobian)(dm,x,A,B,stflag);CHKERRQ(ierr);

  /* if matrix depends on x; i.e. nonlinear problem, keep copy of input vector since needed by multigrid methods to generate coarse grid matrices */
  if (x) {
    if (!dm->x) {
      ierr = DMCreateGlobalVector(dm,&dm->x);CHKERRQ(ierr);
    }
    ierr = VecCopy(x,dm->x);CHKERRQ(ierr);
  }
  if (A != B) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


PetscFList DMList                       = PETSC_NULL;
PetscBool  DMRegisterAllCalled          = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "DMSetType"
/*@C
  DMSetType - Builds a DM, for a particular DM implementation.

  Collective on DM

  Input Parameters:
+ dm     - The DM object
- method - The name of the DM type

  Options Database Key:
. -dm_type <type> - Sets the DM type; use -help for a list of available types

  Notes:
  See "petsc/include/petscdm.h" for available DM types (for instance, DM1D, DM2D, or DM3D).

  Level: intermediate

.keywords: DM, set, type
.seealso: DMGetType(), DMCreate()
@*/
PetscErrorCode  DMSetType(DM dm, const DMType method)
{
  PetscErrorCode (*r)(DM);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject) dm, method, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (!DMRegisterAllCalled) {ierr = DMRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr = PetscFListFind(DMList, ((PetscObject)dm)->comm, method,PETSC_TRUE,(void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown DM type: %s", method);

  if (dm->ops->destroy) {
    ierr = (*dm->ops->destroy)(dm);CHKERRQ(ierr);
    dm->ops->destroy = PETSC_NULL;
  } 
  ierr = (*r)(dm);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)dm,method);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGetType"
/*@C
  DMGetType - Gets the DM type name (as a string) from the DM.

  Not Collective

  Input Parameter:
. dm  - The DM

  Output Parameter:
. type - The DM type name

  Level: intermediate

.keywords: DM, get, type, name
.seealso: DMSetType(), DMCreate()
@*/
PetscErrorCode  DMGetType(DM dm, const DMType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID,1);
  PetscValidCharPointer(type,2);
  if (!DMRegisterAllCalled) {
    ierr = DMRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  }
  *type = ((PetscObject)dm)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMConvert"
/*@C
  DMConvert - Converts a DM to another DM, either of the same or different type.

  Collective on DM

  Input Parameters:
+ dm - the DM
- newtype - new DM type (use "same" for the same type)

  Output Parameter:
. M - pointer to new DM

  Notes:
  Cannot be used to convert a sequential DM to parallel or parallel to sequential,
  the MPI communicator of the generated DM is always the same as the communicator
  of the input DM.

  Level: intermediate

.seealso: DMCreate()
@*/
PetscErrorCode DMConvert(DM dm, const DMType newtype, DM *M)
{
  DM             B;
  char           convname[256];
  PetscBool      sametype, issame;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidType(dm,1);
  PetscValidPointer(M,3);
  ierr = PetscObjectTypeCompare((PetscObject) dm, newtype, &sametype);CHKERRQ(ierr);
  ierr = PetscStrcmp(newtype, "same", &issame);CHKERRQ(ierr);
  {
    PetscErrorCode (*conv)(DM, const DMType, DM *) = PETSC_NULL;

    /*
       Order of precedence:
       1) See if a specialized converter is known to the current DM.
       2) See if a specialized converter is known to the desired DM class.
       3) See if a good general converter is registered for the desired class
       4) See if a good general converter is known for the current matrix.
       5) Use a really basic converter.
    */

    /* 1) See if a specialized converter is known to the current DM and the desired class */
    ierr = PetscStrcpy(convname,"DMConvert_");CHKERRQ(ierr);
    ierr = PetscStrcat(convname,((PetscObject) dm)->type_name);CHKERRQ(ierr);
    ierr = PetscStrcat(convname,"_");CHKERRQ(ierr);
    ierr = PetscStrcat(convname,newtype);CHKERRQ(ierr);
    ierr = PetscStrcat(convname,"_C");CHKERRQ(ierr);
    ierr = PetscObjectQueryFunction((PetscObject)dm,convname,(void (**)(void))&conv);CHKERRQ(ierr);
    if (conv) goto foundconv;

    /* 2)  See if a specialized converter is known to the desired DM class. */
    ierr = DMCreate(((PetscObject) dm)->comm, &B);CHKERRQ(ierr);
    ierr = DMSetType(B, newtype);CHKERRQ(ierr);
    ierr = PetscStrcpy(convname,"DMConvert_");CHKERRQ(ierr);
    ierr = PetscStrcat(convname,((PetscObject) dm)->type_name);CHKERRQ(ierr);
    ierr = PetscStrcat(convname,"_");CHKERRQ(ierr);
    ierr = PetscStrcat(convname,newtype);CHKERRQ(ierr);
    ierr = PetscStrcat(convname,"_C");CHKERRQ(ierr);
    ierr = PetscObjectQueryFunction((PetscObject)B,convname,(void (**)(void))&conv);CHKERRQ(ierr);
    if (conv) {
      ierr = DMDestroy(&B);CHKERRQ(ierr);
      goto foundconv;
    }

#if 0
    /* 3) See if a good general converter is registered for the desired class */
    conv = B->ops->convertfrom;
    ierr = DMDestroy(&B);CHKERRQ(ierr);
    if (conv) goto foundconv;

    /* 4) See if a good general converter is known for the current matrix */
    if (dm->ops->convert) {
      conv = dm->ops->convert;
    }
    if (conv) goto foundconv;
#endif

    /* 5) Use a really basic converter. */
    SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_SUP, "No conversion possible between DM types %s and %s", ((PetscObject) dm)->type_name, newtype);

    foundconv:
    ierr = PetscLogEventBegin(DM_Convert,dm,0,0,0);CHKERRQ(ierr);
    ierr = (*conv)(dm,newtype,M);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(DM_Convert,dm,0,0,0);CHKERRQ(ierr);
  }
  ierr = PetscObjectStateIncrease((PetscObject) *M);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*--------------------------------------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "DMRegister"
/*@C
  DMRegister - See DMRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode  DMRegister(const char sname[], const char path[], const char name[], PetscErrorCode (*function)(DM))
{
  char fullname[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrcpy(fullname, path);CHKERRQ(ierr);
  ierr = PetscStrcat(fullname, ":");CHKERRQ(ierr);
  ierr = PetscStrcat(fullname, name);CHKERRQ(ierr);
  ierr = PetscFListAdd(&DMList, sname, fullname, (void (*)(void)) function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*--------------------------------------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "DMRegisterDestroy"
/*@C
   DMRegisterDestroy - Frees the list of DM methods that were registered by DMRegister()/DMRegisterDynamic().

   Not Collective

   Level: advanced

.keywords: DM, register, destroy
.seealso: DMRegister(), DMRegisterAll(), DMRegisterDynamic()
@*/
PetscErrorCode  DMRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&DMList);CHKERRQ(ierr);
  DMRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MATLAB_ENGINE)
#include <mex.h>

typedef struct {char *funcname; char *jacname; mxArray *ctx;} DMMatlabContext;

#undef __FUNCT__  
#define __FUNCT__ "DMComputeFunction_Matlab"
/*
   DMComputeFunction_Matlab - Calls the function that has been set with
                         DMSetFunctionMatlab().  

   For linear problems x is null
   
.seealso: DMSetFunction(), DMGetFunction()
*/
PetscErrorCode  DMComputeFunction_Matlab(DM dm,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  DMMatlabContext   *sctx;
  int               nlhs = 1,nrhs = 4;
  mxArray	    *plhs[1],*prhs[4];
  long long int     lx = 0,ly = 0,ls = 0;
      
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscCheckSameComm(dm,1,y,3);

  /* call Matlab function in ctx with arguments x and y */
  ierr = DMGetApplicationContext(dm,&sctx);CHKERRQ(ierr);
  ierr = PetscMemcpy(&ls,&dm,sizeof(dm));CHKERRQ(ierr); 
  ierr = PetscMemcpy(&lx,&x,sizeof(x));CHKERRQ(ierr); 
  ierr = PetscMemcpy(&ly,&y,sizeof(y));CHKERRQ(ierr); 
  prhs[0] =  mxCreateDoubleScalar((double)ls);
  prhs[1] =  mxCreateDoubleScalar((double)lx);
  prhs[2] =  mxCreateDoubleScalar((double)ly);
  prhs[3] =  mxCreateString(sctx->funcname);
  ierr    =  mexCallMATLAB(nlhs,plhs,nrhs,prhs,"PetscDMComputeFunctionInternal");CHKERRQ(ierr);
  ierr    =  mxGetScalar(plhs[0]);CHKERRQ(ierr);
  mxDestroyArray(prhs[0]);
  mxDestroyArray(prhs[1]);
  mxDestroyArray(prhs[2]);
  mxDestroyArray(prhs[3]);
  mxDestroyArray(plhs[0]);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DMSetFunctionMatlab"
/*
   DMSetFunctionMatlab - Sets the function evaluation routine 

*/
PetscErrorCode  DMSetFunctionMatlab(DM dm,const char *func)
{
  PetscErrorCode    ierr;
  DMMatlabContext   *sctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  /* currently sctx is memory bleed */
  ierr = DMGetApplicationContext(dm,&sctx);CHKERRQ(ierr);
  if (!sctx) {
    ierr = PetscMalloc(sizeof(DMMatlabContext),&sctx);CHKERRQ(ierr);
  }
  ierr = PetscStrallocpy(func,&sctx->funcname);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm,sctx);CHKERRQ(ierr);
  ierr = DMSetFunction(dm,DMComputeFunction_Matlab);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMComputeJacobian_Matlab"
/*
   DMComputeJacobian_Matlab - Calls the function that has been set with
                         DMSetJacobianMatlab().  

   For linear problems x is null
   
.seealso: DMSetFunction(), DMGetFunction()
*/
PetscErrorCode  DMComputeJacobian_Matlab(DM dm,Vec x,Mat A,Mat B,MatStructure *str)
{
  PetscErrorCode    ierr;
  DMMatlabContext   *sctx;
  int               nlhs = 2,nrhs = 5;
  mxArray	    *plhs[2],*prhs[5];
  long long int     lx = 0,lA = 0,lB = 0,ls = 0;
      
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,3);

  /* call MATLAB function in ctx with arguments x, A, and B */
  ierr = DMGetApplicationContext(dm,&sctx);CHKERRQ(ierr);
  ierr = PetscMemcpy(&ls,&dm,sizeof(dm));CHKERRQ(ierr); 
  ierr = PetscMemcpy(&lx,&x,sizeof(x));CHKERRQ(ierr); 
  ierr = PetscMemcpy(&lA,&A,sizeof(A));CHKERRQ(ierr); 
  ierr = PetscMemcpy(&lB,&B,sizeof(B));CHKERRQ(ierr); 
  prhs[0] =  mxCreateDoubleScalar((double)ls);
  prhs[1] =  mxCreateDoubleScalar((double)lx);
  prhs[2] =  mxCreateDoubleScalar((double)lA);
  prhs[3] =  mxCreateDoubleScalar((double)lB);
  prhs[4] =  mxCreateString(sctx->jacname);
  ierr    =  mexCallMATLAB(nlhs,plhs,nrhs,prhs,"PetscDMComputeJacobianInternal");CHKERRQ(ierr);
  *str    =  (MatStructure) mxGetScalar(plhs[0]);
  ierr    =  (PetscInt) mxGetScalar(plhs[1]);CHKERRQ(ierr);
  mxDestroyArray(prhs[0]);
  mxDestroyArray(prhs[1]);
  mxDestroyArray(prhs[2]);
  mxDestroyArray(prhs[3]);
  mxDestroyArray(prhs[4]);
  mxDestroyArray(plhs[0]);
  mxDestroyArray(plhs[1]);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DMSetJacobianMatlab"
/*
   DMSetJacobianMatlab - Sets the Jacobian function evaluation routine 

*/
PetscErrorCode  DMSetJacobianMatlab(DM dm,const char *func)
{
  PetscErrorCode    ierr;
  DMMatlabContext   *sctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  /* currently sctx is memory bleed */
  ierr = DMGetApplicationContext(dm,&sctx);CHKERRQ(ierr);
  if (!sctx) {
    ierr = PetscMalloc(sizeof(DMMatlabContext),&sctx);CHKERRQ(ierr);
  }
  ierr = PetscStrallocpy(func,&sctx->jacname);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm,sctx);CHKERRQ(ierr);
  ierr = DMSetJacobian(dm,DMComputeJacobian_Matlab);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "DMLoad"
/*@C
  DMLoad - Loads a DM that has been stored in binary or HDF5 format
  with DMView().

  Collective on PetscViewer 

  Input Parameters:
+ newdm - the newly loaded DM, this needs to have been created with DMCreate() or
           some related function before a call to DMLoad(). 
- viewer - binary file viewer, obtained from PetscViewerBinaryOpen() or
           HDF5 file viewer, obtained from PetscViewerHDF5Open()

   Level: intermediate

  Notes:
  Defaults to the DM DA.

  Notes for advanced users:
  Most users should not need to know the details of the binary storage
  format, since DMLoad() and DMView() completely hide these details.
  But for anyone who's interested, the standard binary matrix storage
  format is  
.vb
     has not yet been determined
.ve

   In addition, PETSc automatically does the byte swapping for
machines that store the bytes reversed, e.g.  DEC alpha, freebsd,
linux, Windows and the paragon; thus if you write your own binary
read/write routines you have to swap the bytes; see PetscBinaryRead()
and PetscBinaryWrite() to see how this may be done.

  Concepts: vector^loading from file

.seealso: PetscViewerBinaryOpen(), DMView(), MatLoad(), VecLoad() 
@*/  
PetscErrorCode  DMLoad(DM newdm, PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(newdm,DM_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);

  if (!((PetscObject)newdm)->type_name) {
    ierr = DMSetType(newdm, DMDA);CHKERRQ(ierr);
  }
  ierr = (*newdm->ops->load)(newdm,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/******************************** FEM Support **********************************/

#undef __FUNCT__
#define __FUNCT__ "DMPrintCellVector"
PetscErrorCode DMPrintCellVector(PetscInt c, const char name[], PetscInt len, const PetscScalar x[]) {
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPrintf(PETSC_COMM_SELF, "Cell %D Element %s\n", c, name);CHKERRQ(ierr);
  for(f = 0; f < len; ++f) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "  | %G |\n", PetscRealPart(x[f]));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPrintCellMatrix"
PetscErrorCode DMPrintCellMatrix(PetscInt c, const char name[], PetscInt rows, PetscInt cols, const PetscScalar A[]) {
  PetscInt       f, g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPrintf(PETSC_COMM_SELF, "Cell %D Element %s\n", c, name);CHKERRQ(ierr);
  for(f = 0; f < rows; ++f) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "  |");CHKERRQ(ierr);
    for(g = 0; g < cols; ++g) {
      ierr = PetscPrintf(PETSC_COMM_SELF, " % 9.5G", PetscRealPart(A[f*cols+g]));CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_SELF, " |\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGetLocalFunction"
/*@C
  DMGetLocalFunction - Get the local residual function from this DM

  Not collective

  Input Parameter:
. dm - The DM

  Output Parameter:
. lf - The local residual function

   Calling sequence of lf:
$    lf (SNES snes, Vec x, Vec f, void *ctx);

+  snes - the SNES context
.  x - local vector with the state at which to evaluate residual
.  f - local vector to put residual in
-  ctx - optional user-defined function context

  Level: intermediate

.seealso DMSetLocalFunction(), DMGetLocalJacobian(), DMSetLocalJacobian()
@*/
PetscErrorCode DMGetLocalFunction(DM dm, PetscErrorCode (**lf)(DM, Vec, Vec, void *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (lf) *lf = dm->lf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetLocalFunction"
/*@C
  DMSetLocalFunction - Set the local residual function from this DM

  Not collective

  Input Parameters:
+ dm - The DM
- lf - The local residual function

   Calling sequence of lf:
$    lf (SNES snes, Vec x, Vec f, void *ctx);

+  snes - the SNES context
.  x - local vector with the state at which to evaluate residual
.  f - local vector to put residual in
-  ctx - optional user-defined function context

  Level: intermediate

.seealso DMGetLocalFunction(), DMGetLocalJacobian(), DMSetLocalJacobian()
@*/
PetscErrorCode DMSetLocalFunction(DM dm, PetscErrorCode (*lf)(DM, Vec, Vec, void *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  dm->lf = lf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGetLocalJacobian"
/*@C
  DMGetLocalJacobian - Get the local Jacobian function from this DM

  Not collective

  Input Parameter:
. dm - The DM

  Output Parameter:
. lj - The local Jacobian function

   Calling sequence of lj:
$    lj (SNES snes, Vec x, Mat J, Mat M, void *ctx);

+  snes - the SNES context
.  x - local vector with the state at which to evaluate residual
.  J - matrix to put Jacobian in
.  M - matrix to use for defining Jacobian preconditioner
-  ctx - optional user-defined function context

  Level: intermediate

.seealso DMSetLocalJacobian(), DMGetLocalFunction(), DMSetLocalFunction()
@*/
PetscErrorCode DMGetLocalJacobian(DM dm, PetscErrorCode (**lj)(DM, Vec, Mat, Mat, void *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (lj) *lj = dm->lj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetLocalJacobian"
/*@C
  DMSetLocalJacobian - Set the local Jacobian function from this DM

  Not collective

  Input Parameters:
+ dm - The DM
- lj - The local Jacobian function

   Calling sequence of lj:
$    lj (SNES snes, Vec x, Mat J, Mat M, void *ctx);

+  snes - the SNES context
.  x - local vector with the state at which to evaluate residual
.  J - matrix to put Jacobian in
.  M - matrix to use for defining Jacobian preconditioner
-  ctx - optional user-defined function context

  Level: intermediate

.seealso DMGetLocalJacobian(), DMGetLocalFunction(), DMSetLocalFunction()
@*/
PetscErrorCode DMSetLocalJacobian(DM dm, PetscErrorCode (*lj)(DM, Vec, Mat,  Mat, void *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  dm->lj = lj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGetDefaultSection"
/*@
  DMGetDefaultSection - Get the PetscSection encoding the local data layout for the DM.

  Input Parameter:
. dm - The DM

  Output Parameter:
. section - The PetscSection

  Level: intermediate

  Note: This gets a borrowed reference, so the user should not destroy this PetscSection.

.seealso: DMSetDefaultSection(), DMGetDefaultGlobalSection()
@*/
PetscErrorCode DMGetDefaultSection(DM dm, PetscSection *section) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(section, 2);
  *section = dm->defaultSection;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetDefaultSection"
/*@
  DMSetDefaultSection - Set the PetscSection encoding the local data layout for the DM.

  Input Parameters:
+ dm - The DM
- section - The PetscSection

  Level: intermediate

  Note: Any existing Section will be destroyed

.seealso: DMSetDefaultSection(), DMGetDefaultGlobalSection()
@*/
PetscErrorCode DMSetDefaultSection(DM dm, PetscSection section) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionDestroy(&dm->defaultSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&dm->defaultGlobalSection);CHKERRQ(ierr);
  dm->defaultSection = section;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGetDefaultGlobalSection"
/*@
  DMGetDefaultGlobalSection - Get the PetscSection encoding the global data layout for the DM.

  Input Parameter:
. dm - The DM

  Output Parameter:
. section - The PetscSection

  Level: intermediate

  Note: This gets a borrowed reference, so the user should not destroy this PetscSection.

.seealso: DMSetDefaultSection(), DMGetDefaultSection()
@*/
PetscErrorCode DMGetDefaultGlobalSection(DM dm, PetscSection *section) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(section, 2);
  if (!dm->defaultGlobalSection) {
    ierr = PetscSectionCreateGlobalSection(dm->defaultSection, dm->sf, &dm->defaultGlobalSection);CHKERRQ(ierr);
  }
  *section = dm->defaultGlobalSection;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGetDefaultSF"
/*@
  DMGetDefaultSF - Get the PetscSF encoding the parallel dof overlap for the DM. If it has not been set,
  it is created from the default PetscSection layouts in the DM.

  Input Parameter:
. dm - The DM

  Output Parameter:
. sf - The PetscSF

  Level: intermediate

  Note: This gets a borrowed reference, so the user should not destroy this PetscSF.

.seealso: DMSetDefaultSF(), DMCreateDefaultSF()
@*/
PetscErrorCode DMGetDefaultSF(DM dm, PetscSF *sf) {
  PetscInt       nroots;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(sf, 2);
  ierr = PetscSFGetGraph(dm->defaultSF, &nroots, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  if (nroots < 0) {
    PetscSection section, gSection;

    ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
    if (section) {
      ierr = DMGetDefaultGlobalSection(dm, &gSection);CHKERRQ(ierr);
      ierr = DMCreateDefaultSF(dm, section, gSection);CHKERRQ(ierr);
    } else {
      *sf = PETSC_NULL;
      PetscFunctionReturn(0);
    }
  }
  *sf = dm->defaultSF;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetDefaultSF"
/*@
  DMSetDefaultSF - Set the PetscSF encoding the parallel dof overlap for the DM

  Input Parameters:
+ dm - The DM
- sf - The PetscSF

  Level: intermediate

  Note: Any previous SF is destroyed

.seealso: DMGetDefaultSF(), DMCreateDefaultSF()
@*/
PetscErrorCode DMSetDefaultSF(DM dm, PetscSF sf) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 2);
  ierr = PetscSFDestroy(&dm->defaultSF);CHKERRQ(ierr);
  dm->defaultSF = sf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateDefaultSF"
/*@C
  DMCreateDefaultSF - Create the PetscSF encoding the parallel dof overlap for the DM based upon the PetscSections
  describing the data layout.

  Input Parameters:
+ dm - The DM
. localSection - PetscSection describing the local data layout
- globalSection - PetscSection describing the global data layout

  Level: intermediate

.seealso: DMGetDefaultSF(), DMSetDefaultSF()
@*/
PetscErrorCode DMCreateDefaultSF(DM dm, PetscSection localSection, PetscSection globalSection)
{
  MPI_Comm        comm = ((PetscObject) dm)->comm;
  PetscLayout     layout;
  const PetscInt *ranges;
  PetscInt       *local;
  PetscSFNode    *remote;
  PetscInt        pStart, pEnd, p, nroots, nleaves, l;
  PetscMPIInt     size, rank;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(globalSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(globalSection, &nroots);CHKERRQ(ierr);
  ierr = PetscLayoutCreate(comm, &layout);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(layout, 1);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(layout, nroots);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(layout);CHKERRQ(ierr);
  ierr = PetscLayoutGetRanges(layout, &ranges);CHKERRQ(ierr);
  for(p = pStart, nleaves = 0; p < pEnd; ++p) {
    PetscInt dof, cdof;

    ierr = PetscSectionGetDof(globalSection, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(globalSection, p, &cdof);CHKERRQ(ierr);
    nleaves += dof < 0 ? -(dof+1)-cdof : dof-cdof;
  }
  ierr = PetscMalloc(nleaves * sizeof(PetscInt), &local);CHKERRQ(ierr);
  ierr = PetscMalloc(nleaves * sizeof(PetscSFNode), &remote);CHKERRQ(ierr);
  for(p = pStart, l = 0; p < pEnd; ++p) {
    PetscInt *cind;
    PetscInt  dof, gdof, cdof, dim, off, goff, d, c;

    ierr = PetscSectionGetDof(localSection, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(localSection, p, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(localSection, p, &cdof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintIndices(localSection, p, &cind);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(globalSection, p, &gdof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(globalSection, p, &goff);CHKERRQ(ierr);
    dim  = dof-cdof;
    for(d = 0, c = 0; d < dof; ++d) {
      if ((c < cdof) && (cind[c] == d)) {++c; continue;}
      local[l+d-c] = off+d;
    }
    if (gdof < 0) {
      for(d = 0; d < dim; ++d, ++l) {
        PetscInt offset = -(goff+1) + d, r;

        for(r = 0; r < size; ++r) {
          if ((offset >= ranges[r]) && (offset < ranges[r+1])) break;
        }
        remote[l].rank  = r;
        remote[l].index = offset - ranges[r];
      }
    } else {
      for(d = 0; d < dim; ++d, ++l) {
        remote[l].rank  = rank;
        remote[l].index = goff+d - ranges[rank];
      }
    }
  }
  ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(dm->defaultSF, nroots, nleaves, local, PETSC_OWN_POINTER, remote, PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
