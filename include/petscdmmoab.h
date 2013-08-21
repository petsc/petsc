#if !defined(__PETSCMOAB_H)
#define __PETSCMOAB_H

#include <petscvec.h>
#include <petscmat.h>
#include <petscdm.h>

#include <string>
#include <moab/Core.hpp>
#include <moab/ParallelComm.hpp>

/* The MBERR macro is used to save typing. It checks a MOAB error code
 * (rval) and calls SETERRQ if not MB_SUCCESS. A message (msg) can
 * also be passed in. */
#define MBERR(msg,rval) do{if(rval != moab::MB_SUCCESS) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"MOAB ERROR (%i): %s",(PetscErrorCode)rval,msg);} while(0)
#define MBERRNM(rval) do{if(rval != moab::MB_SUCCESS) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"MOAB ERROR (%i)",rval);} while(0)
#define MBERRV(mbif,rval) do{if(rval != moab::MB_SUCCESS) { std::string emsg; mbif->get_last_error(emsg); SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"MOAB ERROR (%i): %s",(PetscErrorCode)rval,emsg.c_str());} } while(0)
#define MBERRVM(mbif,msg,rval) do{if(rval != moab::MB_SUCCESS) { std::string emsg; mbif->get_last_error(emsg); SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_LIB,"MOAB ERROR (%i): %s :: %s",(PetscErrorCode)rval,msg,emsg.c_str());} } while(0)

#define MOAB_PARROPTS_READ_PART    "READ_PART"
#define MOAB_PARROPTS_READ_DELETE  "READ_DELETE"
#define MOAB_PARROPTS_BCAST_DELETE "BCAST_DELETE"

#define MOAB_PARWOPTS_WRITE_PART "WRITE_PART"
#define MOAB_PARWOPTS_WRITE_FORMAT "FORMAT"

typedef const char* MoabReadMode;
typedef const char* MoabWriteMode;

PETSC_EXTERN PetscErrorCode DMMoabCreate(MPI_Comm comm, DM *moab);
PETSC_EXTERN PetscErrorCode DMMoabCreateMoab(MPI_Comm comm, moab::Interface *mbiface, moab::ParallelComm *pcomm, moab::Tag *ltog_tag, moab::Range *range, DM *moab);
PETSC_EXTERN PetscErrorCode DMMoabOutput(DM,const char*,const char*);

PETSC_EXTERN PetscErrorCode DMMoabSetParallelComm(DM dm,moab::ParallelComm *pcomm);
PETSC_EXTERN PetscErrorCode DMMoabGetParallelComm(DM dm,moab::ParallelComm **pcomm);
PETSC_EXTERN PetscErrorCode DMMoabSetInterface(DM dm,moab::Interface *iface);
PETSC_EXTERN PetscErrorCode DMMoabGetInterface(DM dm,moab::Interface **iface);
PETSC_EXTERN PetscErrorCode DMMoabSetLocalVertices(DM dm,moab::Range *range);
PETSC_EXTERN PetscErrorCode DMMoabGetLocalVertices(DM dm,moab::Range *owned,moab::Range *ghost);
PETSC_EXTERN PetscErrorCode DMMoabSetLocalElements(DM dm,moab::Range *range);
PETSC_EXTERN PetscErrorCode DMMoabGetLocalElements(DM dm,moab::Range *range);
PETSC_EXTERN PetscErrorCode DMMoabSetLocalToGlobalTag(DM dm,moab::Tag ltogtag);
PETSC_EXTERN PetscErrorCode DMMoabGetLocalToGlobalTag(DM dm,moab::Tag *ltog_tag);
PETSC_EXTERN PetscErrorCode DMMoabSetBlockSize(DM dm,PetscInt bs);
PETSC_EXTERN PetscErrorCode DMMoabGetBlockSize(DM dm,PetscInt *bs);
PETSC_EXTERN PetscErrorCode DMMoabGetDimension(DM dm,PetscInt *dim);
PETSC_EXTERN PetscErrorCode DMMoabGetBoundaryEntities(DM dm,moab::Range *bdvtx,moab::Range* bdfaces,moab::Range* bdelems);

PETSC_EXTERN PetscErrorCode DMMoabGetSize(DM dm,PetscInt *ng);
PETSC_EXTERN PetscErrorCode DMMoabGetLocalSize(DM dm,PetscInt *nl,PetscInt *ng);

PETSC_EXTERN PetscErrorCode DMMoabVecGetArrayRead(DM,Vec,void*);
PETSC_EXTERN PetscErrorCode DMMoabVecRestoreArrayRead(DM,Vec,void*);
PETSC_EXTERN PetscErrorCode DMMoabVecGetArray(DM,Vec,void*);
PETSC_EXTERN PetscErrorCode DMMoabVecRestoreArray(DM,Vec,void*);

PETSC_EXTERN PetscErrorCode DMMoabCreateVector(DM dm, moab::Tag tag,PetscInt tag_size,moab::Range *range,PetscBool serial, PetscBool destroy_tag,Vec *X);
PETSC_EXTERN PetscErrorCode DMMoabCreateMatrix(DM dm, MatType mtype,Mat *J);
PETSC_EXTERN PetscErrorCode DMMoabGetVecTag(Vec vec,moab::Tag *tag);
PETSC_EXTERN PetscErrorCode DMMoabGetVecRange(Vec vec,moab::Range *range);

PETSC_EXTERN PetscErrorCode DMMoabSetFieldVector(DM, PetscInt, Vec);
PETSC_EXTERN PetscErrorCode DMMoabSetGlobalFieldVector(DM, Vec);

PETSC_EXTERN PetscErrorCode DMMoabSetFields(DM dm,PetscInt nfields,const char** fields);
PETSC_EXTERN PetscErrorCode DMMoabGetFieldDof(DM dm,moab::EntityHandle point, PetscInt field,PetscInt* dof);
PETSC_EXTERN PetscErrorCode DMMoabGetFieldDofs(DM dm,PetscInt npoints,const moab::EntityHandle* points, PetscInt field,PetscInt* dof);
PETSC_EXTERN PetscErrorCode DMMoabGetFieldDofsLocal(DM dm,PetscInt npoints,const moab::EntityHandle* points, PetscInt field,PetscInt* dof);
PETSC_EXTERN PetscErrorCode DMMoabGetDofs(DM dm,PetscInt npoints,const moab::EntityHandle* points,PetscInt* dof);
PETSC_EXTERN PetscErrorCode DMMoabGetDofsLocal(DM dm,PetscInt npoints,const moab::EntityHandle* points,PetscInt* dof);
PETSC_EXTERN PetscErrorCode DMMoabGetDofsBlocked(DM dm,PetscInt npoints,const moab::EntityHandle* points,PetscInt* dof);
PETSC_EXTERN PetscErrorCode DMMoabGetDofsBlockedLocal(DM dm,PetscInt npoints,const moab::EntityHandle* points,PetscInt* dof);

/* discretization and assembly specific DMMoab interface functions */
PETSC_EXTERN PetscErrorCode DMMoabGetElementConnectivity(DM dm,moab::EntityHandle ehandle,PetscInt* nconn,const moab::EntityHandle **conn);
PETSC_EXTERN PetscErrorCode DMMoabGetVertexCoordinates(DM dm,PetscInt nconn,const moab::EntityHandle *conn,PetscScalar *vpos);
PETSC_EXTERN PetscErrorCode DMMoabIsEntityOnBoundary(DM dm,const moab::EntityHandle ent,PetscBool* ent_on_boundary);
PETSC_EXTERN PetscErrorCode DMMoabCheckBoundaryVertices(DM dm,PetscInt nconn,const moab::EntityHandle *cnt,PetscBool* isbdvtx);

/* DM utility creation interface */
PETSC_EXTERN PetscErrorCode DMMoabCreateBoxMesh(MPI_Comm,PetscInt,PetscInt,PetscInt,DM*);
PETSC_EXTERN PetscErrorCode DMMoabLoadFromFile(MPI_Comm,PetscInt,const char*,const char*,DM*);

#endif
