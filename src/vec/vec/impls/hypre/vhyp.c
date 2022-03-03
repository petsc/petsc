
/*
    Creates hypre ijvector from PETSc vector
*/

#include <petsc/private/vecimpl.h>          /*I "petscvec.h" I*/
#include <../src/vec/vec/impls/hypre/vhyp.h>
#include <HYPRE.h>

PetscErrorCode VecHYPRE_IJVectorCreate(PetscLayout map,VecHYPRE_IJVector *ij)
{
  VecHYPRE_IJVector nij;

  PetscFunctionBegin;
  CHKERRQ(PetscNew(&nij));
  CHKERRQ(PetscLayoutSetUp(map));
  CHKERRQ(HYPRE_IJVectorCreate(map->comm,map->rstart,map->rend-1,&nij->ij));
  CHKERRQ(HYPRE_IJVectorSetObjectType(nij->ij,HYPRE_PARCSR));
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  CHKERRQ(HYPRE_IJVectorInitialize_v2(nij->ij,HYPRE_MEMORY_DEVICE));
#else
  CHKERRQ(HYPRE_IJVectorInitialize(nij->ij));
#endif
  CHKERRQ(HYPRE_IJVectorAssemble(nij->ij));
  *ij  = nij;
  PetscFunctionReturn(0);
}

PetscErrorCode VecHYPRE_IJVectorDestroy(VecHYPRE_IJVector *ij)
{
  PetscFunctionBegin;
  if (!*ij) PetscFunctionReturn(0);
  PetscCheckFalse((*ij)->pvec,PetscObjectComm((PetscObject)((*ij)->pvec)),PETSC_ERR_ORDER,"Forgot to call VecHYPRE_IJVectorPopVec()");
  PetscStackCallStandard(HYPRE_IJVectorDestroy,(*ij)->ij);
  CHKERRQ(PetscFree(*ij));
  PetscFunctionReturn(0);
}

PetscErrorCode VecHYPRE_IJVectorCopy(Vec v,VecHYPRE_IJVector ij)
{
  const PetscScalar *array;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  CHKERRQ(HYPRE_IJVectorInitialize_v2(ij->ij,HYPRE_MEMORY_DEVICE));
#else
  CHKERRQ(HYPRE_IJVectorInitialize(ij->ij));
#endif
  CHKERRQ(VecGetArrayRead(v,&array));
  CHKERRQ(HYPRE_IJVectorSetValues(ij->ij,v->map->n,NULL,(HYPRE_Complex*)array));
  CHKERRQ(VecRestoreArrayRead(v,&array));
  CHKERRQ(HYPRE_IJVectorAssemble(ij->ij));
  PetscFunctionReturn(0);
}

/*
    Replaces the address where the HYPRE vector points to its data with the address of
  PETSc's data. Saves the old address so it can be reset when we are finished with it.
  Allows use to get the data into a HYPRE vector without the cost of memcopies
*/
#define VecHYPRE_ParVectorReplacePointer(b,newvalue,savedvalue) {                               \
  hypre_ParVector *par_vector   = (hypre_ParVector*)hypre_IJVectorObject(((hypre_IJVector*)b)); \
  hypre_Vector    *local_vector = hypre_ParVectorLocalVector(par_vector);                       \
  savedvalue         = local_vector->data;                                               \
  local_vector->data = newvalue;                                                         \
}

/*
  This routine access the pointer to the raw data of the "v" to be passed to HYPRE
   - rw values indicate the type of access : 0 -> read, 1 -> write, 2 -> read-write
   - hmem is the location HYPRE is expecting
   - the function returns a pointer to the data (ptr) and the corresponding restore
  Could be extended to VECKOKKOS if we had a way to access the raw pointer to device data.
*/
static inline PetscErrorCode VecGetArrayForHYPRE(Vec v, int rw, HYPRE_MemoryLocation hmem, PetscScalar **ptr, PetscErrorCode(**res)(Vec,PetscScalar**))
{
  PetscMemType   mtype;
  MPI_Comm       comm;

  PetscFunctionBegin;
#if !defined(PETSC_HAVE_HYPRE_DEVICE)
  hmem = HYPRE_MEMORY_HOST; /* this is just a convenience because HYPRE_MEMORY_HOST and HYPRE_MEMORY_DEVICE are the same in this case */
#endif
  *ptr = NULL;
  *res = NULL;
  CHKERRQ(PetscObjectGetComm((PetscObject)v,&comm));
  switch (rw) {
  case 0: /* read */
    if (hmem == HYPRE_MEMORY_HOST) {
      CHKERRQ(VecGetArrayRead(v,(const PetscScalar**)ptr));
      *res = (PetscErrorCode(*)(Vec,PetscScalar**))VecRestoreArrayRead;
    } else {
      CHKERRQ(VecGetArrayReadAndMemType(v,(const PetscScalar**)ptr,&mtype));
      PetscCheck(PetscMemTypeDevice(mtype),comm,PETSC_ERR_ARG_WRONG,"HYPRE_MEMORY_DEVICE expects a device vector. You need to enable PETSc device support, for example, in some cases, -vec_type cuda");
      *res = (PetscErrorCode(*)(Vec,PetscScalar**))VecRestoreArrayReadAndMemType;
    }
    break;
  case 1: /* write */
    if (hmem == HYPRE_MEMORY_HOST) {
      CHKERRQ(VecGetArrayWrite(v,ptr));
      *res = VecRestoreArrayWrite;
    } else {
      CHKERRQ(VecGetArrayWriteAndMemType(v,(PetscScalar**)ptr,&mtype));
      PetscCheck(PetscMemTypeDevice(mtype),comm,PETSC_ERR_ARG_WRONG,"HYPRE_MEMORY_DEVICE expects a device vector. You need to enable PETSc device support, for example, in some cases, -vec_type cuda");
      *res = VecRestoreArrayWriteAndMemType;
    }
    break;
  case 2: /* read/write */
    if (hmem == HYPRE_MEMORY_HOST) {
      CHKERRQ(VecGetArray(v,ptr));
      *res = VecRestoreArray;
    } else {
      CHKERRQ(VecGetArrayAndMemType(v,(PetscScalar**)ptr,&mtype));
      PetscCheck(PetscMemTypeDevice(mtype),comm,PETSC_ERR_ARG_WRONG,"HYPRE_MEMORY_DEVICE expects a device vector. You need to enable PETSc device support, for example, in some cases, -vec_type cuda");
      *res = VecRestoreArrayAndMemType;
    }
    break;
  default:
    SETERRQ(comm,PETSC_ERR_SUP,"Unhandled case %d",rw);
  }
  PetscFunctionReturn(0);
}

#define VecHYPRE_IJVectorMemoryLocation(v) hypre_IJVectorMemoryLocation((hypre_IJVector*)(v))

/* Temporarily pushes the array of the data in v to ij (read access)
   depending on the value of the ij memory location
   Must be completed with a call to VecHYPRE_IJVectorPopVec */
PetscErrorCode VecHYPRE_IJVectorPushVecRead(VecHYPRE_IJVector ij, Vec v)
{
  HYPRE_Complex  *pv;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,2);
  PetscCheck(!ij->pvec,PetscObjectComm((PetscObject)v),PETSC_ERR_ORDER,"Forgot to call VecHYPRE_IJVectorPopVec()");
  PetscCheck(!ij->hv,PetscObjectComm((PetscObject)v),PETSC_ERR_ORDER,"Forgot to call VecHYPRE_IJVectorPopVec()");
  CHKERRQ(VecGetArrayForHYPRE(v,0,VecHYPRE_IJVectorMemoryLocation(ij->ij),(PetscScalar**)&pv,&ij->restore));
  VecHYPRE_ParVectorReplacePointer(ij->ij,pv,ij->hv);
  ij->pvec = v;
  PetscFunctionReturn(0);
}

/* Temporarily pushes the array of the data in v to ij (write access)
   depending on the value of the ij memory location
   Must be completed with a call to VecHYPRE_IJVectorPopVec */
PetscErrorCode VecHYPRE_IJVectorPushVecWrite(VecHYPRE_IJVector ij, Vec v)
{
  HYPRE_Complex  *pv;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,2);
  PetscCheck(!ij->pvec,PetscObjectComm((PetscObject)v),PETSC_ERR_ORDER,"Forgot to call VecHYPRE_IJVectorPopVec()");
  PetscCheck(!ij->hv,PetscObjectComm((PetscObject)v),PETSC_ERR_ORDER,"Forgot to call VecHYPRE_IJVectorPopVec()");
  CHKERRQ(VecGetArrayForHYPRE(v,1,VecHYPRE_IJVectorMemoryLocation(ij->ij),(PetscScalar**)&pv,&ij->restore));
  VecHYPRE_ParVectorReplacePointer(ij->ij,pv,ij->hv);
  ij->pvec = v;
  PetscFunctionReturn(0);
}

/* Temporarily pushes the array of the data in v to ij (read/write access)
   depending on the value of the ij memory location
   Must be completed with a call to VecHYPRE_IJVectorPopVec */
PetscErrorCode VecHYPRE_IJVectorPushVec(VecHYPRE_IJVector ij, Vec v)
{
  HYPRE_Complex  *pv;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,2);
  PetscCheck(!ij->pvec,PetscObjectComm((PetscObject)v),PETSC_ERR_ORDER,"Forgot to call VecHYPRE_IJVectorPopVec()");
  PetscCheck(!ij->hv,PetscObjectComm((PetscObject)v),PETSC_ERR_ORDER,"Forgot to call VecHYPRE_IJVectorPopVec()");
  CHKERRQ(VecGetArrayForHYPRE(v,2,VecHYPRE_IJVectorMemoryLocation(ij->ij),(PetscScalar**)&pv,&ij->restore));
  VecHYPRE_ParVectorReplacePointer(ij->ij,pv,ij->hv);
  ij->pvec = v;
  PetscFunctionReturn(0);
}

/* Restores the pointer data to v */
PetscErrorCode VecHYPRE_IJVectorPopVec(VecHYPRE_IJVector ij)
{
  HYPRE_Complex  *pv;

  PetscFunctionBegin;
  PetscCheck(ij->pvec,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Forgot to call VecHYPRE_IJVectorPushVec()");
  PetscCheck(ij->restore,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Forgot to call VecHYPRE_IJVectorPushVec()");
  VecHYPRE_ParVectorReplacePointer(ij->ij,ij->hv,pv);
  CHKERRQ((*ij->restore)(ij->pvec,(PetscScalar**)&pv));
  ij->hv = NULL;
  ij->pvec = NULL;
  ij->restore = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode VecHYPRE_IJBindToCPU(VecHYPRE_IJVector ij,PetscBool bind)
{
  HYPRE_MemoryLocation hmem = bind ? HYPRE_MEMORY_HOST : HYPRE_MEMORY_DEVICE;
  hypre_ParVector      *hij;

  PetscFunctionBegin;
  PetscCheck(!ij->pvec,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Forgot to call VecHYPRE_IJVectorPopVec()");
  PetscCheck(!ij->hv,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Forgot to call VecHYPRE_IJVectorPopVec()");
#if !defined(PETSC_HAVE_HYPRE_DEVICE)
  hmem = HYPRE_MEMORY_HOST;
#endif
#if PETSC_PKG_HYPRE_VERSION_GT(2,19,0)
  if (hmem != VecHYPRE_IJVectorMemoryLocation(ij->ij)) {
    PetscStackCallStandard(HYPRE_IJVectorGetObject,ij->ij,(void**)&hij);
    PetscStackCallStandard(hypre_ParVectorMigrate,hij,hmem);
  }
#endif
  PetscFunctionReturn(0);
}
