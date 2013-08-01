#include <petsc-private/dmmbimpl.h> /*I  "petscdm.h"   I*/
#include <petsc-private/vecimpl.h> /*I  "petscdm.h"   I*/

#include <petscdmmoab.h>
#include <MBTagConventions.hpp>
#include <moab/Skinner.hpp>

#undef __FUNCT__
#define __FUNCT__ "DMDestroy_Moab"
PetscErrorCode DMDestroy_Moab(DM dm)
{
  PetscErrorCode ierr;
  DM_Moab         *dmmoab = (DM_Moab*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (dmmoab->icreatedinstance) {
    delete dmmoab->mbiface;
  }
  dmmoab->mbiface = NULL;
  dmmoab->pcomm = NULL;
  delete dmmoab->vlocal;
  delete dmmoab->vowned;
  delete dmmoab->vghost;
  delete dmmoab->elocal;
  delete dmmoab->eghost;
  delete dmmoab->bndyvtx;
  delete dmmoab->bndyfaces;
  ierr = VecScatterDestroy(&dmmoab->ltog_sendrecv);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&dmmoab->ltog_map);CHKERRQ(ierr);
  ierr = PetscFree(dm->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetUp_Moab"
PetscErrorCode DMSetUp_Moab(DM dm)
{
  PetscErrorCode          ierr;
  moab::ErrorCode         merr;
  Vec                     local, global;
  IS                      from;
  moab::Range::iterator   iter;
  PetscInt                i,j,bs,gsiz,lsiz,*gsindices;
  DM_Moab                *dmmoab = (DM_Moab*)dm->data;
  PetscInt                totsize,local_min,global_min;
  PetscSection            section;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  /* Get the local and shared vertices and cache it */
  if (dmmoab->mbiface == PETSC_NULL || dmmoab->pcomm == PETSC_NULL) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ORDER, "Set the MOAB Interface and ParallelComm objects before calling SetUp.");

  /* store the current mesh dimension */
  merr = dmmoab->mbiface->get_dimension(dmmoab->dim);MBERRNM(merr);
 
  /* Get the entities recursively in the current part of the mesh, if user did not set the local vertices explicitly */
  if (dmmoab->vlocal->empty()) {
    merr = dmmoab->mbiface->get_entities_by_type(dmmoab->fileset,moab::MBVERTEX,*dmmoab->vlocal,true);MBERRNM(merr);
    *dmmoab->vowned = *dmmoab->vlocal;

    /* filter based on parallel status */
    merr = dmmoab->pcomm->filter_pstatus(*dmmoab->vowned,PSTATUS_NOT_OWNED,PSTATUS_NOT);MBERRNM(merr);
    *dmmoab->vghost = moab::subtract(*dmmoab->vlocal, *dmmoab->vowned);

    dmmoab->nloc = dmmoab->vowned->size();
    dmmoab->nghost = dmmoab->vghost->size();
    ierr = MPI_Allreduce(&dmmoab->nloc, &dmmoab->n, 1, MPI_INTEGER, MPI_SUM, ((PetscObject)dm)->comm);CHKERRQ(ierr);
  }

  /* get the information about the local elements in the mesh */
  {
    dmmoab->elocal->clear();
    dmmoab->eghost->clear();
    merr = dmmoab->mbiface->get_entities_by_dimension(dmmoab->fileset, dmmoab->dim, *dmmoab->elocal, true);CHKERRQ(merr);
    *dmmoab->eghost = *dmmoab->elocal;
    merr = dmmoab->pcomm->filter_pstatus(*dmmoab->elocal,PSTATUS_NOT_OWNED,PSTATUS_NOT);MBERRNM(merr);
    *dmmoab->eghost = moab::subtract(*dmmoab->eghost, *dmmoab->elocal);

    dmmoab->neleloc = dmmoab->elocal->size();
    ierr = MPI_Allreduce(&dmmoab->neleloc, &dmmoab->nele, 1, MPI_INTEGER, MPI_SUM, ((PetscObject)dm)->comm);CHKERRQ(ierr);
  }

  bs = dmmoab->bs;
  if (!dmmoab->ltog_tag) {
    /* Get the global ID tag. The global ID tag is applied to each
       vertex. It acts as an global identifier which MOAB uses to
       assemble the individual pieces of the mesh */
    merr = dmmoab->mbiface->tag_get_handle(GLOBAL_ID_TAG_NAME, dmmoab->ltog_tag);MBERRNM(merr);
  }

  totsize=dmmoab->vlocal->size();
  ierr = PetscMalloc(totsize*sizeof(PetscInt), &gsindices);CHKERRQ(ierr);
  {
    /* first get the local indices */
    merr = dmmoab->mbiface->tag_get_data(dmmoab->ltog_tag,*dmmoab->vowned,&gsindices[0]);MBERRNM(merr);
    /* next get the ghosted indices */
    if (dmmoab->vghost->size()) {
      merr = dmmoab->mbiface->tag_get_data(dmmoab->ltog_tag,*dmmoab->vghost,&gsindices[dmmoab->nloc]);MBERRNM(merr);
    }

    /* find out the local and global minima of GLOBAL_ID */
    local_min=gsindices[0];
    for (i=1; i<totsize; ++i)
      if(local_min>gsindices[i]) local_min=gsindices[i];

    ierr = MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, ((PetscObject)dm)->comm);CHKERRQ(ierr);
    PetscInfo2(dm, "GLOBAL_ID: Local minima - %D, Global minima - %D.\n", local_min, global_min);
  }
    
  {
    ierr = PetscSectionCreate(((PetscObject)dm)->comm, &section);CHKERRQ(ierr);
    ierr = PetscSectionSetNumFields(section, dmmoab->nfields);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(section, gsindices[0], gsindices[dmmoab->nloc-1]+1);CHKERRQ(ierr);
    for (j=gsindices[0]; j<=gsindices[dmmoab->nloc-1]; ++j) {
      PetscInt locgid = j-global_min;
      for (i=0; i < dmmoab->nfields; ++i) {
        ierr = PetscSectionSetFieldName(section, i, dmmoab->fields[i]);CHKERRQ(ierr);
        if (bs>1) {
          ierr = PetscSectionSetFieldDof(section, j, i, locgid*dmmoab->nfields+i);CHKERRQ(ierr);
          ierr = PetscSectionSetFieldOffset(section, j, i, dmmoab->nfields);
        }
        else {
          ierr = PetscSectionSetFieldDof(section, j, i, totsize*i+locgid);CHKERRQ(ierr);
          ierr = PetscSectionSetFieldOffset(section, j, i, totsize);
        }
      }
      ierr = PetscSectionSetDof(section, j, dmmoab->nfields);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(section);CHKERRQ(ierr);
    ierr = DMSetDefaultSection(dm, section);CHKERRQ(ierr);
  }

  {
    /* Create Global to Local Vector Scatter Context */
    ierr = DMCreateGlobalVector_Moab(dm, &global);CHKERRQ(ierr);
    ierr = DMCreateLocalVector_Moab(dm, &local);CHKERRQ(ierr);

    for (i=0; i<totsize; ++i)
      gsindices[i]-=global_min;   /* zero based index needed for IS */

    /* global to local must retrieve ghost points */
    ierr = ISCreateBlock(((PetscObject)dm)->comm,bs,totsize,&gsindices[0],PETSC_COPY_VALUES,&from);CHKERRQ(ierr);

    ierr = VecGetLocalSize(global,&gsiz);CHKERRQ(ierr);
    ierr = VecGetLocalSize(local,&lsiz);CHKERRQ(ierr);

    ierr = VecScatterCreate(local,from,global,from,&dmmoab->ltog_sendrecv);CHKERRQ(ierr);
    ierr = ISDestroy(&from);CHKERRQ(ierr);
    ierr = VecDestroy(&local);CHKERRQ(ierr);
    ierr = VecDestroy(&global);CHKERRQ(ierr);
  }

  /* skin the boundary and store nodes */
  {
    // get the skin vertices of those faces and mark them as fixed; we don't want to fix the vertices on a
    // part boundary, but since we exchanged a layer of ghost faces, those vertices aren't on the skin locally
    // ok to mark non-owned skin vertices too, I won't move those anyway
    // use MOAB's skinner class to find the skin
    moab::Skinner skinner(dmmoab->mbiface);
    dmmoab->bndyvtx = new moab::Range();
    dmmoab->bndyfaces = new moab::Range();
    merr = skinner.find_skin(dmmoab->fileset, *dmmoab->elocal, true, *dmmoab->bndyvtx);MBERRNM(merr); // 'true' param indicates we want vertices back, not faces
    merr = skinner.find_skin(dmmoab->fileset, *dmmoab->elocal, false, *dmmoab->bndyfaces);MBERRNM(merr); // 'false' param indicates we want faces back, not vertices
    PetscInfo2(dm, "Found %D boundary vertices and %D faces.\n", dmmoab->bndyvtx->size(), dmmoab->bndyvtx->size());
  }

  ierr = PetscFree(gsindices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMCreate_Moab"
PETSC_EXTERN PetscErrorCode DMCreate_Moab(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscNewLog(dm,&dm->data);CHKERRQ(ierr);

  ((DM_Moab*)dm->data)->bs = 1;
  ((DM_Moab*)dm->data)->n = 0;
  ((DM_Moab*)dm->data)->nloc = 0;
  ((DM_Moab*)dm->data)->nele = 0;
  ((DM_Moab*)dm->data)->neleloc = 0;
  ((DM_Moab*)dm->data)->nghost = 0;
  ((DM_Moab*)dm->data)->ltog_map = PETSC_NULL;
  ((DM_Moab*)dm->data)->ltog_sendrecv = PETSC_NULL;

  ((DM_Moab*)dm->data)->vlocal = new moab::Range();
  ((DM_Moab*)dm->data)->vowned = new moab::Range();
  ((DM_Moab*)dm->data)->vghost = new moab::Range();
  ((DM_Moab*)dm->data)->elocal = new moab::Range();
  ((DM_Moab*)dm->data)->eghost = new moab::Range();
  
  dm->ops->createglobalvector              = DMCreateGlobalVector_Moab;
  dm->ops->createlocalvector               = DMCreateLocalVector_Moab;
  dm->ops->creatematrix                    = DMCreateMatrix_Moab;
  dm->ops->setup                           = DMSetUp_Moab;
  dm->ops->destroy                         = DMDestroy_Moab;
  dm->ops->globaltolocalbegin              = DMGlobalToLocalBegin_Moab;
  dm->ops->globaltolocalend                = DMGlobalToLocalEnd_Moab;
  dm->ops->localtoglobalbegin              = DMLocalToGlobalBegin_Moab;
  dm->ops->localtoglobalend                = DMLocalToGlobalEnd_Moab;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMoabCreate"
/*@
  DMMoabCreate - Creates a DMMoab object, which encapsulates a moab instance

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the DMMoab object

  Output Parameter:
. dmb  - The DMMoab object

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabCreate(MPI_Comm comm, DM *dmb)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dmb,2);
  ierr = DMCreate(comm, dmb);CHKERRQ(ierr);
  ierr = DMSetType(*dmb, DMMOAB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMoabCreateMoab"
/*@
  DMMoabCreate - Creates a DMMoab object, optionally from an instance and other data

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the DMMoab object
. mbiface - (ptr to) the MOAB Instance; if passed in NULL, MOAB instance is created inside PETSc, and destroyed
         along with the DMMoab
. pcomm - (ptr to) a ParallelComm; if NULL, creates one internally for the whole communicator
. ltog_tag - A tag to use to retrieve global id for an entity; if 0, will use GLOBAL_ID_TAG_NAME/tag
. range - If non-NULL, contains range of entities to which DOFs will be assigned

  Output Parameter:
. dmb  - The DMMoab object

  Level: intermediate

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabCreateMoab(MPI_Comm comm, moab::Interface *mbiface, moab::ParallelComm *pcomm, moab::Tag *ltog_tag, moab::Range *range, DM *dmb)
{
  PetscErrorCode ierr;
  moab::ErrorCode merr;
  moab::EntityHandle partnset;
  PetscInt rank, nprocs;
  DM_Moab        *dmmoab;

  PetscFunctionBegin;
  PetscValidPointer(dmb,6);
  ierr = DMMoabCreate(comm, dmb);CHKERRQ(ierr);
  dmmoab = (DM_Moab*)(*dmb)->data;

  if (!mbiface) {
    dmmoab->mbiface = new moab::Core();
    dmmoab->icreatedinstance = PETSC_TRUE;
  }
  else {
    dmmoab->mbiface = mbiface;
    dmmoab->icreatedinstance = PETSC_FALSE;
  }

  /* create a fileset to store the hierarchy of entities belonging to current DM */
  merr = dmmoab->mbiface->create_meshset(moab::MESHSET_ORDERED, dmmoab->fileset);MBERR("Creating file set failed", merr);

  if (!pcomm) {
    ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &nprocs);CHKERRQ(ierr);

    /* Create root sets for each mesh.  Then pass these
       to the load_file functions to be populated. */
    merr = dmmoab->mbiface->create_meshset(moab::MESHSET_SET, partnset);
    MBERR("Creating partition set failed", merr);

    /* Create the parallel communicator object with the partition handle associated with MOAB */
    dmmoab->pcomm = moab::ParallelComm::get_pcomm(dmmoab->mbiface, partnset, &comm);
  }
  else {
    ierr = DMMoabSetParallelComm(*dmb, pcomm);CHKERRQ(ierr);
  }

  /* do the remaining initializations for DMMoab */
  dmmoab->bs       = 1;

  /* set global ID tag handle */
  if (!ltog_tag) {
    merr = dmmoab->mbiface->tag_get_handle(GLOBAL_ID_TAG_NAME, dmmoab->ltog_tag);MBERRNM(merr);
  }
  else {
    ierr = DMMoabSetLocalToGlobalTag(*dmb, *ltog_tag);CHKERRQ(ierr);
  }

  /* set the local range of entities (vertices) of interest */
  if (range) {
    ierr = DMMoabSetLocalVertices(*dmb, range);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMoabSetParallelComm"
/*@
  DMMoabSetParallelComm - Set the ParallelComm used with this DMMoab

  Collective on MPI_Comm

  Input Parameter:
. dm    - The DMMoab object being set
. pcomm - The ParallelComm being set on the DMMoab

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabSetParallelComm(DM dm,moab::ParallelComm *pcomm)
{
  DM_Moab        *dmmoab = (DM_Moab*)(dm)->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(pcomm,2);
  dmmoab->pcomm = pcomm;
  dmmoab->mbiface = pcomm->get_moab();
  dmmoab->icreatedinstance = PETSC_FALSE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetParallelComm"
/*@
  DMMoabGetParallelComm - Get the ParallelComm used with this DMMoab

  Collective on MPI_Comm

  Input Parameter:
. dm    - The DMMoab object being set

  Output Parameter:
. pcomm - The ParallelComm for the DMMoab

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabGetParallelComm(DM dm,moab::ParallelComm **pcomm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *pcomm = ((DM_Moab*)(dm)->data)->pcomm;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabSetInterface"
/*@
  DMMoabSetInterface - Set the MOAB instance used with this DMMoab

  Collective on MPI_Comm

  Input Parameter:
. dm      - The DMMoab object being set
. mbiface - The MOAB instance being set on this DMMoab

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabSetInterface(DM dm,moab::Interface *mbiface)
{
  DM_Moab        *dmmoab = (DM_Moab*)(dm)->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(mbiface,2);
  dmmoab->pcomm = NULL;
  dmmoab->mbiface = mbiface;
  dmmoab->icreatedinstance = PETSC_FALSE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetInterface"
/*@
  DMMoabGetInterface - Get the MOAB instance used with this DMMoab

  Collective on MPI_Comm

  Input Parameter:
. dm      - The DMMoab object being set

  Output Parameter:
. mbiface - The MOAB instance set on this DMMoab

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabGetInterface(DM dm,moab::Interface **mbiface)
{
  PetscErrorCode   ierr;
  static PetscBool cite = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscCitationsRegister("@techreport{tautges_moab:_2004,\n  type = {{SAND2004-1592}},\n  title = {{MOAB:} A Mesh-Oriented Database},  institution = {Sandia National Laboratories},\n  author = {Tautges, T. J. and Meyers, R. and Merkley, K. and Stimpson, C. and Ernst, C.},\n  year = {2004},  note = {Report}\n}\n",&cite);CHKERRQ(ierr);
  *mbiface = ((DM_Moab*)dm->data)->mbiface;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabSetLocalVertices"
/*@
  DMMoabSetLocalVertices - Set the entities having DOFs on this DMMoab

  Collective on MPI_Comm

  Input Parameter:
. dm    - The DMMoab object being set
. range - The entities treated by this DMMoab

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabSetLocalVertices(DM dm,moab::Range *range)
{
  moab::ErrorCode merr;
  PetscErrorCode  ierr;
  DM_Moab        *dmmoab = (DM_Moab*)(dm)->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dmmoab->vlocal->clear();
  dmmoab->vowned->clear();
  dmmoab->vlocal->insert(range->begin(), range->end());
  *dmmoab->vowned = *dmmoab->vlocal;
  merr = dmmoab->pcomm->filter_pstatus(*dmmoab->vowned,PSTATUS_NOT_OWNED,PSTATUS_NOT);MBERRNM(merr);
  *dmmoab->vghost = moab::subtract(*range, *dmmoab->vowned);
  dmmoab->nloc=dmmoab->vowned->size();
  dmmoab->nghost=dmmoab->vghost->size();
  ierr = MPI_Allreduce(&dmmoab->nloc, &dmmoab->n, 1, MPI_INTEGER, MPI_SUM, ((PetscObject)dm)->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetLocalVertices"
/*@
  DMMoabGetLocalVertices - Get the entities having DOFs on this DMMoab

  Collective on MPI_Comm

  Input Parameter:
. dm    - The DMMoab object being set

  Output Parameter:
. owned - The owned vertex entities in this DMMoab
. ghost - The ghosted entities (non-owned) stored locally in this partition

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabGetLocalVertices(DM dm,moab::Range *owned,moab::Range *ghost)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (owned) *owned = *((DM_Moab*)dm->data)->vowned;
  if (ghost) *ghost = *((DM_Moab*)dm->data)->vghost;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMoabGetLocalElements"
/*@
  DMMoabGetLocalElements - Get the higher-dimensional entities that are locally owned

  Collective on MPI_Comm

  Input Parameter:
. dm    - The DMMoab object being set

  Output Parameter:
. range - The entities owned locally

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabGetLocalElements(DM dm,moab::Range *range)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (range) *range = *((DM_Moab*)dm->data)->elocal;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabSetLocalElements"
/*@
  DMMoabSetLocalElements - Set the entities having DOFs on this DMMoab

  Collective on MPI_Comm

  Input Parameter:
. dm    - The DMMoab object being set
. range - The entities treated by this DMMoab

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabSetLocalElements(DM dm,moab::Range *range)
{
  moab::ErrorCode merr;
  PetscErrorCode  ierr;
  DM_Moab        *dmmoab = (DM_Moab*)(dm)->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dmmoab->elocal->clear();
  dmmoab->eghost->clear();
  dmmoab->elocal->insert(range->begin(), range->end());
  PetscInfo2(dm, "Range size = %D; elocal size = %D.\n", range->size(), dmmoab->elocal->size());
  merr = dmmoab->pcomm->filter_pstatus(*dmmoab->elocal,PSTATUS_NOT_OWNED,PSTATUS_NOT);MBERRNM(merr);
  *dmmoab->eghost = moab::subtract(*range, *dmmoab->elocal);
  dmmoab->neleloc=dmmoab->elocal->size();
  ierr = MPI_Allreduce(&dmmoab->nele, &dmmoab->neleloc, 1, MPI_INTEGER, MPI_SUM, ((PetscObject)dm)->comm);CHKERRQ(ierr);
  PetscInfo2(dm, "Created %D local and %D glocal elements.\n", dmmoab->neleloc, dmmoab->nele);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabSetLocalToGlobalTag"
/*@
  DMMoabSetLocalToGlobalTag - Set the tag used for local to global numbering

  Collective on MPI_Comm

  Input Parameter:
. dm      - The DMMoab object being set
. ltogtag - The MOAB tag used for local to global ids

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabSetLocalToGlobalTag(DM dm,moab::Tag ltogtag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ((DM_Moab*)dm->data)->ltog_tag = ltogtag;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetLocalToGlobalTag"
/*@
  DMMoabGetLocalToGlobalTag - Get the tag used for local to global numbering

  Collective on MPI_Comm

  Input Parameter:
. dm      - The DMMoab object being set

  Output Parameter:
. ltogtag - The MOAB tag used for local to global ids

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabGetLocalToGlobalTag(DM dm,moab::Tag *ltog_tag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *ltog_tag = ((DM_Moab*)dm->data)->ltog_tag;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabSetBlockSize"
/*@
  DMMoabSetBlockSize - Set the block size used with this DMMoab

  Collective on MPI_Comm

  Input Parameter:
. dm - The DMMoab object being set
. bs - The block size used with this DMMoab

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabSetBlockSize(DM dm,PetscInt bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ((DM_Moab*)dm->data)->bs = bs;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetBlockSize"
/*@
  DMMoabGetBlockSize - Get the block size used with this DMMoab

  Collective on MPI_Comm

  Input Parameter:
. dm - The DMMoab object being set

  Output Parameter:
. bs - The block size used with this DMMoab

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabGetBlockSize(DM dm,PetscInt *bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *bs = ((DM_Moab*)dm->data)->bs;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabSetFieldVector"
PetscErrorCode DMMoabSetFieldVector(DM dm, PetscInt ifield, Vec fvec)
{
  DM_Moab        *dmmoab;
  moab::Tag     vtag,ntag;
  PetscScalar   *varray;
  moab::ErrorCode merr;
  PetscErrorCode  ierr;
  PetscSection section;
  PetscInt doff;
  std::string tag_name;
  moab::Range::iterator iter;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dmmoab = (DM_Moab*)(dm)->data;

  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);

  /* Create a tag in MOAB mesh to index and keep track of number of Petsc vec tags */
  merr = dmmoab->mbiface->tag_get_handle(dmmoab->fields[ifield],1,moab::MB_TYPE_DOUBLE,ntag,
                                          moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT);MBERRNM(merr);

  ierr = DMMoabGetVecTag(fvec,&vtag);CHKERRQ(ierr);

  merr = dmmoab->mbiface->tag_get_name(vtag, tag_name);
  if (!tag_name.length() && merr !=moab::MB_SUCCESS) {
    ierr = DMMoabVecGetArray(dm,fvec,&varray);CHKERRQ(ierr);
    for(iter = dmmoab->vowned->begin(); iter != dmmoab->vowned->end(); iter++) {
      moab::EntityHandle vtx = (*iter);

      /* get field dof index */
      ierr = PetscSectionGetFieldOffset(section, vtx, ifield, &doff);

      /* use the entity handle and the Dof index to set the right value */
      merr = dmmoab->mbiface->tag_set_data(ntag, &vtx, 1, (const void*)&varray[doff]);MBERRNM(merr);
    }
    ierr = DMMoabVecRestoreArray(dm,fvec,&varray);CHKERRQ(ierr);
  }
  else {
    ierr = PetscMalloc(dmmoab->nloc*sizeof(PetscScalar),&varray);CHKERRQ(ierr);
    /* we are using a MOAB Vec - directly copy the tag data to new one */
    merr = dmmoab->mbiface->tag_get_data(vtag, *dmmoab->vowned, (void*)varray);MBERRNM(merr);
    merr = dmmoab->mbiface->tag_set_data(ntag, *dmmoab->vowned, (const void*)varray);MBERRNM(merr);
    /* make sure the parallel exchange for ghosts are done appropriately */
    merr = dmmoab->pcomm->exchange_tags(ntag, *dmmoab->vlocal);MBERRNM(merr);
    ierr = PetscFree(varray);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetBoundaryEntities"
PetscErrorCode DMMoabGetBoundaryEntities(DM dm,moab::Range *bdvtx,moab::Range* bdfaces)
{
  DM_Moab        *dmmoab;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dmmoab = (DM_Moab*)(dm)->data;

  if (bdvtx)  *bdvtx = *dmmoab->bndyvtx;
  if (bdfaces)  *bdfaces = *dmmoab->bndyfaces;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabOutput"
PetscErrorCode DMMoabOutput(DM dm,const char* fname,const char* wopts)
{
  DM_Moab        *dmmoab;  
  moab::ErrorCode merr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dmmoab = (DM_Moab*)(dm)->data;

  // output file, using parallel write
  merr = dmmoab->mbiface->write_file(fname, NULL, wopts);MBERRVM(dmmoab->mbiface,"Writing output of DMMoab failed.",merr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabSetFields"
PetscErrorCode DMMoabSetFields(DM dm,PetscInt nfields,const char** fields)
{
  DM_Moab        *dmmoab;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dmmoab = (DM_Moab*)(dm)->data;

  dmmoab->fields = fields;
  dmmoab->nfields = nfields;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetFieldDof"
PetscErrorCode DMMoabGetFieldDof(DM dm,moab::EntityHandle point,PetscInt field,PetscInt* dof)
{
  PetscSection section;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetFieldDof(section, (PetscInt)point, field, dof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetFieldDofs"
PetscErrorCode DMMoabGetFieldDofs(DM dm,PetscInt npoints,const moab::EntityHandle* points,PetscInt field,PetscInt* dof)
{
  PetscInt i;
  PetscSection section;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  if (!dof) {
    ierr = PetscMalloc(sizeof(PetscScalar)*npoints, &dof);CHKERRQ(ierr);
  }
  for (i=0; i<npoints; ++i) {
    ierr = PetscSectionGetFieldDof(section, (PetscInt)points[i], field, &dof[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


