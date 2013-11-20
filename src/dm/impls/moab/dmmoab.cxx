#include <petsc-private/dmmbimpl.h> /*I  "petscdm.h"   I*/
#include <petsc-private/vecimpl.h> /*I  "petscdm.h"   I*/

#include <petscdmmoab.h>
#include <MBTagConventions.hpp>

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
  PetscInt                bs, *gsindices,gsiz,lsiz;
  DM_Moab                *dmmoab = (DM_Moab*)dm->data;
  PetscInt                count,dof,totsize;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  /* Get the local and shared vertices and cache it */
  if (dmmoab->mbiface == PETSC_NULL || dmmoab->pcomm == PETSC_NULL) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ORDER, "Set the MOAB Interface and ParallelComm objects before calling SetUp.");

  /* store the current mesh dimension */
  merr = dmmoab->mbiface->get_dimension(dmmoab->dim);MBERRNM(merr);
 
  /* Get the entities recursively in the current part of the mesh */
  if (dmmoab->vlocal->empty()) {
    merr = dmmoab->mbiface->get_entities_by_type(0,moab::MBVERTEX,*dmmoab->vlocal,true);MBERRNM(merr);
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
    merr = dmmoab->mbiface->get_entities_by_dimension(0, dmmoab->dim, *dmmoab->elocal, true);CHKERRQ(merr);
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

  {
    count=0;
    totsize=dmmoab->vlocal->size();
    ierr = PetscMalloc(totsize*sizeof(PetscInt), &gsindices);CHKERRQ(ierr);
    /* first get the local indices */
    for(iter = dmmoab->vowned->begin(); iter != dmmoab->vowned->end(); iter++) {
      merr = dmmoab->mbiface->tag_get_data(dmmoab->ltog_tag,&(*iter),1,&dof);MBERRNM(merr);
      gsindices[count++] = (dof);
    }
    /* now get the ghosted indices */
    for(iter = dmmoab->vghost->begin(); iter != dmmoab->vghost->end(); iter++) {
      merr = dmmoab->mbiface->tag_get_data(dmmoab->ltog_tag,&(*iter),1,&dof);MBERRNM(merr);
      gsindices[count++] = (dof);
    }
    
    /* Create Global to Local Vector Scatter Context */
    ierr = DMCreateGlobalVector_Moab(dm, &global);CHKERRQ(ierr);
    ierr = DMCreateLocalVector_Moab(dm, &local);CHKERRQ(ierr);

    /* global to local must retrieve ghost points */
    ierr = ISCreateBlock(((PetscObject)dm)->comm,bs,totsize,gsindices,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);

    ierr = VecGetLocalSize(global,&gsiz);CHKERRQ(ierr);
    ierr = VecGetLocalSize(local,&lsiz);CHKERRQ(ierr);

    ierr = VecScatterCreate(local,from,global,from,&dmmoab->ltog_sendrecv);CHKERRQ(ierr);
    ierr = ISDestroy(&from);CHKERRQ(ierr);
    ierr = VecDestroy(&local);CHKERRQ(ierr);
    ierr = VecDestroy(&global);CHKERRQ(ierr);
    ierr = PetscFree(gsindices);CHKERRQ(ierr);
  }

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
  DM_Moab        *dmmoab;

  PetscFunctionBegin;
  PetscValidPointer(dmb,6);
  ierr = DMMoabCreate(comm, dmb);CHKERRQ(ierr);
  dmmoab = (DM_Moab*)(*dmb)->data;

  if (!mbiface) {
    mbiface = new moab::Core();
    dmmoab->icreatedinstance = PETSC_TRUE;
  }
  else
    dmmoab->icreatedinstance = PETSC_FALSE;

  if (!pcomm) {
    moab::EntityHandle rootset,partnset;
    PetscInt rank, nprocs;
    ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &nprocs);CHKERRQ(ierr);

    /* Create root sets for each mesh.  Then pass these
       to the load_file functions to be populated. */
    merr = mbiface->create_meshset(moab::MESHSET_SET, rootset);
    MBERR("Creating root set failed", merr);
    merr = mbiface->create_meshset(moab::MESHSET_SET, partnset);
    MBERR("Creating partition set failed", merr);

    /* Create the parallel communicator object with the partition handle associated with MOAB */
    pcomm = moab::ParallelComm::get_pcomm(mbiface, partnset, &comm);
  }

  if (!ltog_tag) {
    merr = mbiface->tag_get_handle(GLOBAL_ID_TAG_NAME, *ltog_tag);MBERRNM(merr);
  }
  else {
    ierr = DMMoabSetLocalToGlobalTag(*dmb, *ltog_tag);CHKERRQ(ierr);
  }

  /* do the initialization of the DM */
  dmmoab->bs       = 1;
  ierr = DMMoabSetParallelComm(*dmb, pcomm);CHKERRQ(ierr);
  dmmoab->ltog_tag = *ltog_tag;  
  if (range) {
    ierr = DMMoabSetRange(*dmb, range);CHKERRQ(ierr);
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
#define __FUNCT__ "DMMoabSetRange"
/*@
  DMMoabSetRange - Set the entities having DOFs on this DMMoab

  Collective on MPI_Comm

  Input Parameter:
. dm    - The DMMoab object being set
. range - The entities treated by this DMMoab

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabSetRange(DM dm,moab::Range *range)
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
#define __FUNCT__ "DMMoabGetRange"
/*@
  DMMoabGetRange - Get the entities having DOFs on this DMMoab

  Collective on MPI_Comm

  Input Parameter:
. dm    - The DMMoab object being set

  Output Parameter:
. range - The entities treated by this DMMoab

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabGetRange(DM dm,moab::Range **range)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *range = ((DM_Moab*)dm->data)->vowned;
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

