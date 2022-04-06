#define PETSCDM_DLL
#include <petsc/private/dmswarmimpl.h>    /*I   "petscdmswarm.h"   I*/
#include <petsc/private/hashsetij.h>
#include <petsc/private/petscfeimpl.h>
#include <petscviewer.h>
#include <petscdraw.h>
#include <petscdmplex.h>
#include <petscblaslapack.h>
#include "../src/dm/impls/swarm/data_bucket.h"
#include <petscdmlabel.h>
#include <petscsection.h>

PetscLogEvent DMSWARM_Migrate, DMSWARM_SetSizes, DMSWARM_AddPoints, DMSWARM_RemovePoints, DMSWARM_Sort;
PetscLogEvent DMSWARM_DataExchangerTopologySetup, DMSWARM_DataExchangerBegin, DMSWARM_DataExchangerEnd;
PetscLogEvent DMSWARM_DataExchangerSendCount, DMSWARM_DataExchangerPack;

const char* DMSwarmTypeNames[] = { "basic", "pic", NULL };
const char* DMSwarmMigrateTypeNames[] = { "basic", "dmcellnscatter", "dmcellexact", "user", NULL };
const char* DMSwarmCollectTypeNames[] = { "basic", "boundingbox", "general", "user", NULL  };
const char* DMSwarmPICLayoutTypeNames[] = { "regular", "gauss", "subdivision", NULL  };

const char DMSwarmField_pid[] = "DMSwarm_pid";
const char DMSwarmField_rank[] = "DMSwarm_rank";
const char DMSwarmPICField_coor[] = "DMSwarmPIC_coor";
const char DMSwarmPICField_cellid[] = "DMSwarm_cellid";

#if defined(PETSC_HAVE_HDF5)
#include <petscviewerhdf5.h>

PetscErrorCode VecView_Swarm_HDF5_Internal(Vec v, PetscViewer viewer)
{
  DM             dm;
  PetscReal      seqval;
  PetscInt       seqnum, bs;
  PetscBool      isseq;

  PetscFunctionBegin;
  PetscCall(VecGetDM(v, &dm));
  PetscCall(VecGetBlockSize(v, &bs));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "/particle_fields"));
  PetscCall(PetscObjectTypeCompare((PetscObject) v, VECSEQ, &isseq));
  PetscCall(DMGetOutputSequenceNumber(dm, &seqnum, &seqval));
  PetscCall(PetscViewerHDF5SetTimestep(viewer, seqnum));
  /* PetscCall(DMSequenceView_HDF5(dm, "time", seqnum, (PetscScalar) seqval, viewer)); */
  PetscCall(VecViewNative(v, viewer));
  PetscCall(PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject) v, "Nc", PETSC_INT, (void *) &bs));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmView_HDF5(DM dm, PetscViewer viewer)
{
  Vec            coordinates;
  PetscInt       Np;
  PetscBool      isseq;

  PetscFunctionBegin;
  PetscCall(DMSwarmGetSize(dm, &Np));
  PetscCall(DMSwarmCreateGlobalVectorFromField(dm, DMSwarmPICField_coor, &coordinates));
  PetscCall(PetscObjectSetName((PetscObject) coordinates, "coordinates"));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "/particles"));
  PetscCall(PetscObjectTypeCompare((PetscObject) coordinates, VECSEQ, &isseq));
  PetscCall(VecViewNative(coordinates, viewer));
  PetscCall(PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject) coordinates, "Np", PETSC_INT, (void *) &Np));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(dm, DMSwarmPICField_coor, &coordinates));
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode VecView_Swarm(Vec v, PetscViewer viewer)
{
  DM             dm;
#if defined(PETSC_HAVE_HDF5)
  PetscBool      ishdf5;
#endif

  PetscFunctionBegin;
  PetscCall(VecGetDM(v, &dm));
  PetscCheck(dm,PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector not generated from a DM");
#if defined(PETSC_HAVE_HDF5)
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,  &ishdf5));
  if (ishdf5) {
      PetscCall(VecView_Swarm_HDF5_Internal(v, viewer));
      PetscFunctionReturn(0);
  }
#endif
  PetscCall(VecViewNative(v, viewer));
  PetscFunctionReturn(0);
}

/*@C
   DMSwarmVectorDefineField - Sets the field from which to define a Vec object
                             when DMCreateLocalVector(), or DMCreateGlobalVector() is called

   Collective on dm

   Input parameters:
+  dm - a DMSwarm
-  fieldname - the textual name given to a registered field

   Level: beginner

   Notes:

   The field with name fieldname must be defined as having a data type of PetscScalar.

   This function must be called prior to calling DMCreateLocalVector(), DMCreateGlobalVector().
   Mutiple calls to DMSwarmVectorDefineField() are permitted.

.seealso: DMSwarmRegisterPetscDatatypeField(), DMCreateGlobalVector(), DMCreateLocalVector()
@*/
PetscErrorCode DMSwarmVectorDefineField(DM dm,const char fieldname[])
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;
  PetscInt       bs,n;
  PetscScalar    *array;
  PetscDataType  type;

  PetscFunctionBegin;
  if (!swarm->issetup) PetscCall(DMSetUp(dm));
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db,&n,NULL,NULL));
  PetscCall(DMSwarmGetField(dm,fieldname,&bs,&type,(void**)&array));

  /* Check all fields are of type PETSC_REAL or PETSC_SCALAR */
  PetscCheck(type == PETSC_REAL,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only valid for PETSC_REAL");
  PetscCall(PetscSNPrintf(swarm->vec_field_name,PETSC_MAX_PATH_LEN-1,"%s",fieldname));
  swarm->vec_field_set = PETSC_TRUE;
  swarm->vec_field_bs = bs;
  swarm->vec_field_nlocal = n;
  PetscCall(DMSwarmRestoreField(dm,fieldname,&bs,&type,(void**)&array));
  PetscFunctionReturn(0);
}

/* requires DMSwarmDefineFieldVector has been called */
PetscErrorCode DMCreateGlobalVector_Swarm(DM dm,Vec *vec)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;
  Vec            x;
  char           name[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  if (!swarm->issetup) PetscCall(DMSetUp(dm));
  PetscCheck(swarm->vec_field_set,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Must call DMSwarmVectorDefineField first");
  PetscCheck(swarm->vec_field_nlocal == swarm->db->L,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"DMSwarm sizes have changed since last call to VectorDefineField first"); /* Stale data */

  PetscCall(PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"DMSwarmField_%s",swarm->vec_field_name));
  PetscCall(VecCreate(PetscObjectComm((PetscObject)dm),&x));
  PetscCall(PetscObjectSetName((PetscObject)x,name));
  PetscCall(VecSetSizes(x,swarm->db->L*swarm->vec_field_bs,PETSC_DETERMINE));
  PetscCall(VecSetBlockSize(x,swarm->vec_field_bs));
  PetscCall(VecSetDM(x,dm));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSetDM(x, dm));
  PetscCall(VecSetOperation(x, VECOP_VIEW, (void (*)(void)) VecView_Swarm));
  *vec = x;
  PetscFunctionReturn(0);
}

/* requires DMSwarmDefineFieldVector has been called */
PetscErrorCode DMCreateLocalVector_Swarm(DM dm,Vec *vec)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;
  Vec            x;
  char           name[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  if (!swarm->issetup) PetscCall(DMSetUp(dm));
  PetscCheck(swarm->vec_field_set,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Must call DMSwarmVectorDefineField first");
  PetscCheck(swarm->vec_field_nlocal == swarm->db->L,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"DMSwarm sizes have changed since last call to VectorDefineField first");

  PetscCall(PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"DMSwarmField_%s",swarm->vec_field_name));
  PetscCall(VecCreate(PETSC_COMM_SELF,&x));
  PetscCall(PetscObjectSetName((PetscObject)x,name));
  PetscCall(VecSetSizes(x,swarm->db->L*swarm->vec_field_bs,PETSC_DETERMINE));
  PetscCall(VecSetBlockSize(x,swarm->vec_field_bs));
  PetscCall(VecSetDM(x,dm));
  PetscCall(VecSetFromOptions(x));
  *vec = x;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSwarmDestroyVectorFromField_Private(DM dm, const char fieldname[], Vec *vec)
{
  DM_Swarm         *swarm = (DM_Swarm *) dm->data;
  DMSwarmDataField gfield;
  void             (*fptr)(void);
  PetscInt         bs, nlocal;
  char             name[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(*vec, &nlocal));
  PetscCall(VecGetBlockSize(*vec, &bs));
  PetscCheck(nlocal/bs == swarm->db->L,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"DMSwarm sizes have changed since vector was created - cannot ensure pointers are valid");
  PetscCall(DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db, fieldname, &gfield));
  /* check vector is an inplace array */
  PetscCall(PetscSNPrintf(name, PETSC_MAX_PATH_LEN-1, "DMSwarm_VecFieldInPlace_%s", fieldname));
  PetscCall(PetscObjectQueryFunction((PetscObject) *vec, name, &fptr));
  PetscCheck(fptr,PetscObjectComm((PetscObject) dm), PETSC_ERR_USER, "Vector being destroyed was not created from DMSwarm field(%s)", fieldname);
  PetscCall(DMSwarmDataFieldRestoreAccess(gfield));
  PetscCall(VecResetArray(*vec));
  PetscCall(VecDestroy(vec));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSwarmCreateVectorFromField_Private(DM dm, const char fieldname[], MPI_Comm comm, Vec *vec)
{
  DM_Swarm      *swarm = (DM_Swarm *) dm->data;
  PetscDataType  type;
  PetscScalar   *array;
  PetscInt       bs, n;
  char           name[PETSC_MAX_PATH_LEN];
  PetscMPIInt    size;
  PetscBool      iscuda,iskokkos;

  PetscFunctionBegin;
  if (!swarm->issetup) PetscCall(DMSetUp(dm));
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &n, NULL, NULL));
  PetscCall(DMSwarmGetField(dm, fieldname, &bs, &type, (void **) &array));
  PetscCheck(type == PETSC_REAL,PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Only valid for PETSC_REAL");

  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscStrcmp(dm->vectype,VECKOKKOS,&iskokkos));
  PetscCall(PetscStrcmp(dm->vectype,VECCUDA,&iscuda));
  PetscCall(VecCreate(comm,vec));
  PetscCall(VecSetSizes(*vec,n*bs,PETSC_DETERMINE));
  PetscCall(VecSetBlockSize(*vec,bs));
  if (iskokkos) PetscCall(VecSetType(*vec,VECKOKKOS));
  else if (iscuda) PetscCall(VecSetType(*vec,VECCUDA));
  else PetscCall(VecSetType(*vec,VECSTANDARD));
  PetscCall(VecPlaceArray(*vec,array));

  PetscCall(PetscSNPrintf(name, PETSC_MAX_PATH_LEN-1, "DMSwarmSharedField_%s", fieldname));
  PetscCall(PetscObjectSetName((PetscObject) *vec, name));

  /* Set guard */
  PetscCall(PetscSNPrintf(name, PETSC_MAX_PATH_LEN-1, "DMSwarm_VecFieldInPlace_%s", fieldname));
  PetscCall(PetscObjectComposeFunction((PetscObject) *vec, name, DMSwarmDestroyVectorFromField_Private));

  PetscCall(VecSetDM(*vec, dm));
  PetscCall(VecSetOperation(*vec, VECOP_VIEW, (void (*)(void)) VecView_Swarm));
  PetscFunctionReturn(0);
}

/* This creates a "mass matrix" between a finite element and particle space. If a finite element interpolant is given by

     \hat f = \sum_i f_i \phi_i

   and a particle function is given by

     f = \sum_i w_i \delta(x - x_i)

   then we want to require that

     M \hat f = M_p f

   where the particle mass matrix is given by

     (M_p)_{ij} = \int \phi_i \delta(x - x_j)

   The way Dave May does particles, they amount to quadratue weights rather than delta functions, so he has |J| is in
   his integral. We allow this with the boolean flag.
*/
static PetscErrorCode DMSwarmComputeMassMatrix_Private(DM dmc, DM dmf, Mat mass, PetscBool useDeltaFunction, void *ctx)
{
  const char    *name = "Mass Matrix";
  MPI_Comm       comm;
  PetscDS        prob;
  PetscSection   fsection, globalFSection;
  PetscHSetIJ    ht;
  PetscLayout    rLayout, colLayout;
  PetscInt      *dnz, *onz;
  PetscInt       locRows, locCols, rStart, colStart, colEnd, *rowIDXs;
  PetscReal     *xi, *v0, *J, *invJ, detJ = 1.0, v0ref[3] = {-1.0, -1.0, -1.0};
  PetscScalar   *elemMat;
  PetscInt       dim, Nf, field, cStart, cEnd, cell, totDim, maxC = 0;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject) mass, &comm));
  PetscCall(DMGetCoordinateDim(dmf, &dim));
  PetscCall(DMGetDS(dmf, &prob));
  PetscCall(PetscDSGetNumFields(prob, &Nf));
  PetscCall(PetscDSGetTotalDimension(prob, &totDim));
  PetscCall(PetscMalloc3(dim, &v0, dim*dim, &J, dim*dim,&invJ));
  PetscCall(DMGetLocalSection(dmf, &fsection));
  PetscCall(DMGetGlobalSection(dmf, &globalFSection));
  PetscCall(DMPlexGetHeightStratum(dmf, 0, &cStart, &cEnd));
  PetscCall(MatGetLocalSize(mass, &locRows, &locCols));

  PetscCall(PetscLayoutCreate(comm, &colLayout));
  PetscCall(PetscLayoutSetLocalSize(colLayout, locCols));
  PetscCall(PetscLayoutSetBlockSize(colLayout, 1));
  PetscCall(PetscLayoutSetUp(colLayout));
  PetscCall(PetscLayoutGetRange(colLayout, &colStart, &colEnd));
  PetscCall(PetscLayoutDestroy(&colLayout));

  PetscCall(PetscLayoutCreate(comm, &rLayout));
  PetscCall(PetscLayoutSetLocalSize(rLayout, locRows));
  PetscCall(PetscLayoutSetBlockSize(rLayout, 1));
  PetscCall(PetscLayoutSetUp(rLayout));
  PetscCall(PetscLayoutGetRange(rLayout, &rStart, NULL));
  PetscCall(PetscLayoutDestroy(&rLayout));

  PetscCall(PetscCalloc2(locRows, &dnz, locRows, &onz));
  PetscCall(PetscHSetIJCreate(&ht));

  PetscCall(PetscSynchronizedFlush(comm, NULL));
  /* count non-zeros */
  PetscCall(DMSwarmSortGetAccess(dmc));
  for (field = 0; field < Nf; ++field) {
    for (cell = cStart; cell < cEnd; ++cell) {
      PetscInt  c, i;
      PetscInt *findices,   *cindices; /* fine is vertices, coarse is particles */
      PetscInt  numFIndices, numCIndices;

      PetscCall(DMPlexGetClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL));
      PetscCall(DMSwarmSortGetPointsPerCell(dmc, cell, &numCIndices, &cindices));
      maxC = PetscMax(maxC, numCIndices);
      {
        PetscHashIJKey key;
        PetscBool      missing;
        for (i = 0; i < numFIndices; ++i) {
          key.j = findices[i]; /* global column (from Plex) */
          if (key.j >= 0) {
            /* Get indices for coarse elements */
            for (c = 0; c < numCIndices; ++c) {
              key.i = cindices[c] + rStart; /* global cols (from Swarm) */
              if (key.i < 0) continue;
              PetscCall(PetscHSetIJQueryAdd(ht, key, &missing));
              if (missing) {
                if ((key.j >= colStart) && (key.j < colEnd)) ++dnz[key.i - rStart];
                else                                         ++onz[key.i - rStart];
              } else SETERRQ(PetscObjectComm((PetscObject) dmf), PETSC_ERR_SUP, "Set new value at %D,%D", key.i, key.j);
            }
          }
        }
        PetscCall(PetscFree(cindices));
      }
      PetscCall(DMPlexRestoreClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL));
    }
  }
  PetscCall(PetscHSetIJDestroy(&ht));
  PetscCall(MatXAIJSetPreallocation(mass, 1, dnz, onz, NULL, NULL));
  PetscCall(MatSetOption(mass, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscCall(PetscFree2(dnz, onz));
  PetscCall(PetscMalloc3(maxC*totDim, &elemMat, maxC, &rowIDXs, maxC*dim, &xi));
  for (field = 0; field < Nf; ++field) {
    PetscTabulation Tcoarse;
    PetscObject     obj;
    PetscReal       *coords;
    PetscInt        Nc, i;

    PetscCall(PetscDSGetDiscretization(prob, field, &obj));
    PetscCall(PetscFEGetNumComponents((PetscFE) obj, &Nc));
    PetscCheck(Nc == 1,PetscObjectComm((PetscObject) dmf), PETSC_ERR_SUP, "Can only interpolate a scalar field from particles, Nc = %D", Nc);
    PetscCall(DMSwarmGetField(dmc, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
    for (cell = cStart; cell < cEnd; ++cell) {
      PetscInt *findices  , *cindices;
      PetscInt  numFIndices, numCIndices;
      PetscInt  p, c;

      /* TODO: Use DMField instead of assuming affine */
      PetscCall(DMPlexComputeCellGeometryFEM(dmf, cell, NULL, v0, J, invJ, &detJ));
      PetscCall(DMPlexGetClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL));
      PetscCall(DMSwarmSortGetPointsPerCell(dmc, cell, &numCIndices, &cindices));
      for (p = 0; p < numCIndices; ++p) {
        CoordinatesRealToRef(dim, dim, v0ref, v0, invJ, &coords[cindices[p]*dim], &xi[p*dim]);
      }
      PetscCall(PetscFECreateTabulation((PetscFE) obj, 1, numCIndices, xi, 0, &Tcoarse));
      /* Get elemMat entries by multiplying by weight */
      PetscCall(PetscArrayzero(elemMat, numCIndices*totDim));
      for (i = 0; i < numFIndices; ++i) {
        for (p = 0; p < numCIndices; ++p) {
          for (c = 0; c < Nc; ++c) {
            /* B[(p*pdim + i)*Nc + c] is the value at point p for basis function i and component c */
            elemMat[p*numFIndices+i] += Tcoarse->T[0][(p*numFIndices + i)*Nc + c]*(useDeltaFunction ? 1.0 : detJ);
          }
        }
      }
      for (p = 0; p < numCIndices; ++p) rowIDXs[p] = cindices[p] + rStart;
      if (0) PetscCall(DMPrintCellMatrix(cell, name, 1, numCIndices, elemMat));
      PetscCall(MatSetValues(mass, numCIndices, rowIDXs, numFIndices, findices, elemMat, ADD_VALUES));
      PetscCall(PetscFree(cindices));
      PetscCall(DMPlexRestoreClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL));
      PetscCall(PetscTabulationDestroy(&Tcoarse));
    }
    PetscCall(DMSwarmRestoreField(dmc, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  }
  PetscCall(PetscFree3(elemMat, rowIDXs, xi));
  PetscCall(DMSwarmSortRestoreAccess(dmc));
  PetscCall(PetscFree3(v0, J, invJ));
  PetscCall(MatAssemblyBegin(mass, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mass, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/* Returns empty matrix for use with SNES FD */
static PetscErrorCode DMCreateMatrix_Swarm(DM sw, Mat* m)
{
  Vec            field;
  PetscInt       size;

  PetscFunctionBegin;
  PetscCall(DMGetGlobalVector(sw, &field));
  PetscCall(VecGetLocalSize(field, &size));
  PetscCall(DMRestoreGlobalVector(sw, &field));
  PetscCall(MatCreate(PETSC_COMM_WORLD, m));
  PetscCall(MatSetFromOptions(*m));
  PetscCall(MatSetSizes(*m, PETSC_DECIDE, PETSC_DECIDE, size, size));
  PetscCall(MatSeqAIJSetPreallocation(*m, 1, NULL));
  PetscCall(MatZeroEntries(*m));
  PetscCall(MatAssemblyBegin(*m, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*m, MAT_FINAL_ASSEMBLY));
  PetscCall(MatShift(*m, 1.0));
  PetscCall(MatSetDM(*m, sw));
  PetscFunctionReturn(0);
}

/* FEM cols, Particle rows */
static PetscErrorCode DMCreateMassMatrix_Swarm(DM dmCoarse, DM dmFine, Mat *mass)
{
  PetscSection   gsf;
  PetscInt       m, n;
  void          *ctx;

  PetscFunctionBegin;
  PetscCall(DMGetGlobalSection(dmFine, &gsf));
  PetscCall(PetscSectionGetConstrainedStorageSize(gsf, &m));
  PetscCall(DMSwarmGetLocalSize(dmCoarse, &n));
  PetscCall(MatCreate(PetscObjectComm((PetscObject) dmCoarse), mass));
  PetscCall(MatSetSizes(*mass, n, m, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(*mass, dmCoarse->mattype));
  PetscCall(DMGetApplicationContext(dmFine, &ctx));

  PetscCall(DMSwarmComputeMassMatrix_Private(dmCoarse, dmFine, *mass, PETSC_TRUE, ctx));
  PetscCall(MatViewFromOptions(*mass, NULL, "-mass_mat_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSwarmComputeMassMatrixSquare_Private(DM dmc, DM dmf, Mat mass, PetscBool useDeltaFunction, void *ctx)
{
  const char    *name = "Mass Matrix Square";
  MPI_Comm       comm;
  PetscDS        prob;
  PetscSection   fsection, globalFSection;
  PetscHSetIJ    ht;
  PetscLayout    rLayout, colLayout;
  PetscInt      *dnz, *onz, *adj, depth, maxConeSize, maxSupportSize, maxAdjSize;
  PetscInt       locRows, locCols, rStart, colStart, colEnd, *rowIDXs;
  PetscReal     *xi, *v0, *J, *invJ, detJ = 1.0, v0ref[3] = {-1.0, -1.0, -1.0};
  PetscScalar   *elemMat, *elemMatSq;
  PetscInt       cdim, Nf, field, cStart, cEnd, cell, totDim, maxC = 0;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject) mass, &comm));
  PetscCall(DMGetCoordinateDim(dmf, &cdim));
  PetscCall(DMGetDS(dmf, &prob));
  PetscCall(PetscDSGetNumFields(prob, &Nf));
  PetscCall(PetscDSGetTotalDimension(prob, &totDim));
  PetscCall(PetscMalloc3(cdim, &v0, cdim*cdim, &J, cdim*cdim,&invJ));
  PetscCall(DMGetLocalSection(dmf, &fsection));
  PetscCall(DMGetGlobalSection(dmf, &globalFSection));
  PetscCall(DMPlexGetHeightStratum(dmf, 0, &cStart, &cEnd));
  PetscCall(MatGetLocalSize(mass, &locRows, &locCols));

  PetscCall(PetscLayoutCreate(comm, &colLayout));
  PetscCall(PetscLayoutSetLocalSize(colLayout, locCols));
  PetscCall(PetscLayoutSetBlockSize(colLayout, 1));
  PetscCall(PetscLayoutSetUp(colLayout));
  PetscCall(PetscLayoutGetRange(colLayout, &colStart, &colEnd));
  PetscCall(PetscLayoutDestroy(&colLayout));

  PetscCall(PetscLayoutCreate(comm, &rLayout));
  PetscCall(PetscLayoutSetLocalSize(rLayout, locRows));
  PetscCall(PetscLayoutSetBlockSize(rLayout, 1));
  PetscCall(PetscLayoutSetUp(rLayout));
  PetscCall(PetscLayoutGetRange(rLayout, &rStart, NULL));
  PetscCall(PetscLayoutDestroy(&rLayout));

  PetscCall(DMPlexGetDepth(dmf, &depth));
  PetscCall(DMPlexGetMaxSizes(dmf, &maxConeSize, &maxSupportSize));
  maxAdjSize = PetscPowInt(maxConeSize*maxSupportSize, depth);
  PetscCall(PetscMalloc1(maxAdjSize, &adj));

  PetscCall(PetscCalloc2(locRows, &dnz, locRows, &onz));
  PetscCall(PetscHSetIJCreate(&ht));
  /* Count nonzeros
       This is just FVM++, but we cannot use the Plex P0 allocation since unknowns in a cell will not be contiguous
  */
  PetscCall(DMSwarmSortGetAccess(dmc));
  for (cell = cStart; cell < cEnd; ++cell) {
    PetscInt  i;
    PetscInt *cindices;
    PetscInt  numCIndices;
  #if 0
    PetscInt  adjSize = maxAdjSize, a, j;
  #endif

    PetscCall(DMSwarmSortGetPointsPerCell(dmc, cell, &numCIndices, &cindices));
    maxC = PetscMax(maxC, numCIndices);
    /* Diagonal block */
    for (i = 0; i < numCIndices; ++i) {dnz[cindices[i]] += numCIndices;}
#if 0
    /* Off-diagonal blocks */
    PetscCall(DMPlexGetAdjacency(dmf, cell, &adjSize, &adj));
    for (a = 0; a < adjSize; ++a) {
      if (adj[a] >= cStart && adj[a] < cEnd && adj[a] != cell) {
        const PetscInt ncell = adj[a];
        PetscInt      *ncindices;
        PetscInt       numNCIndices;

        PetscCall(DMSwarmSortGetPointsPerCell(dmc, ncell, &numNCIndices, &ncindices));
        {
          PetscHashIJKey key;
          PetscBool      missing;

          for (i = 0; i < numCIndices; ++i) {
            key.i = cindices[i] + rStart; /* global rows (from Swarm) */
            if (key.i < 0) continue;
            for (j = 0; j < numNCIndices; ++j) {
              key.j = ncindices[j] + rStart; /* global column (from Swarm) */
              if (key.j < 0) continue;
              PetscCall(PetscHSetIJQueryAdd(ht, key, &missing));
              if (missing) {
                if ((key.j >= colStart) && (key.j < colEnd)) ++dnz[key.i - rStart];
                else                                         ++onz[key.i - rStart];
              }
            }
          }
        }
        PetscCall(PetscFree(ncindices));
      }
    }
#endif
    PetscCall(PetscFree(cindices));
  }
  PetscCall(PetscHSetIJDestroy(&ht));
  PetscCall(MatXAIJSetPreallocation(mass, 1, dnz, onz, NULL, NULL));
  PetscCall(MatSetOption(mass, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscCall(PetscFree2(dnz, onz));
  PetscCall(PetscMalloc4(maxC*totDim, &elemMat, maxC*maxC, &elemMatSq, maxC, &rowIDXs, maxC*cdim, &xi));
  /* Fill in values
       Each entry is a sum of terms \phi_i(x_p) \phi_i(x_q)
       Start just by producing block diagonal
       Could loop over adjacent cells
         Produce neighboring element matrix
         TODO Determine which columns and rows correspond to shared dual vector
         Do MatMatMult with rectangular matrices
         Insert block
  */
  for (field = 0; field < Nf; ++field) {
    PetscTabulation Tcoarse;
    PetscObject     obj;
    PetscReal       *coords;
    PetscInt        Nc, i;

    PetscCall(PetscDSGetDiscretization(prob, field, &obj));
    PetscCall(PetscFEGetNumComponents((PetscFE) obj, &Nc));
    PetscCheck(Nc == 1,PetscObjectComm((PetscObject) dmf), PETSC_ERR_SUP, "Can only interpolate a scalar field from particles, Nc = %D", Nc);
    PetscCall(DMSwarmGetField(dmc, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
    for (cell = cStart; cell < cEnd; ++cell) {
      PetscInt *findices  , *cindices;
      PetscInt  numFIndices, numCIndices;
      PetscInt  p, c;

      /* TODO: Use DMField instead of assuming affine */
      PetscCall(DMPlexComputeCellGeometryFEM(dmf, cell, NULL, v0, J, invJ, &detJ));
      PetscCall(DMPlexGetClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL));
      PetscCall(DMSwarmSortGetPointsPerCell(dmc, cell, &numCIndices, &cindices));
      for (p = 0; p < numCIndices; ++p) {
        CoordinatesRealToRef(cdim, cdim, v0ref, v0, invJ, &coords[cindices[p]*cdim], &xi[p*cdim]);
      }
      PetscCall(PetscFECreateTabulation((PetscFE) obj, 1, numCIndices, xi, 0, &Tcoarse));
      /* Get elemMat entries by multiplying by weight */
      PetscCall(PetscArrayzero(elemMat, numCIndices*totDim));
      for (i = 0; i < numFIndices; ++i) {
        for (p = 0; p < numCIndices; ++p) {
          for (c = 0; c < Nc; ++c) {
            /* B[(p*pdim + i)*Nc + c] is the value at point p for basis function i and component c */
            elemMat[p*numFIndices+i] += Tcoarse->T[0][(p*numFIndices + i)*Nc + c]*(useDeltaFunction ? 1.0 : detJ);
          }
        }
      }
      PetscCall(PetscTabulationDestroy(&Tcoarse));
      for (p = 0; p < numCIndices; ++p) rowIDXs[p] = cindices[p] + rStart;
      if (0) PetscCall(DMPrintCellMatrix(cell, name, 1, numCIndices, elemMat));
      /* Block diagonal */
      if (numCIndices) {
        PetscBLASInt blasn, blask;
        PetscScalar  one = 1.0, zero = 0.0;

        PetscCall(PetscBLASIntCast(numCIndices, &blasn));
        PetscCall(PetscBLASIntCast(numFIndices, &blask));
        PetscStackCallBLAS("BLASgemm",BLASgemm_("T","N",&blasn,&blasn,&blask,&one,elemMat,&blask,elemMat,&blask,&zero,elemMatSq,&blasn));
      }
      PetscCall(MatSetValues(mass, numCIndices, rowIDXs, numCIndices, rowIDXs, elemMatSq, ADD_VALUES));
      /* TODO Off-diagonal */
      PetscCall(PetscFree(cindices));
      PetscCall(DMPlexRestoreClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL));
    }
    PetscCall(DMSwarmRestoreField(dmc, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  }
  PetscCall(PetscFree4(elemMat, elemMatSq, rowIDXs, xi));
  PetscCall(PetscFree(adj));
  PetscCall(DMSwarmSortRestoreAccess(dmc));
  PetscCall(PetscFree3(v0, J, invJ));
  PetscCall(MatAssemblyBegin(mass, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mass, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*@C
  DMSwarmCreateMassMatrixSquare - Creates the block-diagonal of the square, M^T_p M_p, of the particle mass matrix M_p

  Collective on dmCoarse

  Input parameters:
+ dmCoarse - a DMSwarm
- dmFine   - a DMPlex

  Output parameter:
. mass     - the square of the particle mass matrix

  Level: advanced

  Notes:
  We only compute the block diagonal since this provides a good preconditioner and is completely local. It would be possible in the
  future to compute the full normal equations.

.seealso: DMCreateMassMatrix()
@*/
PetscErrorCode DMSwarmCreateMassMatrixSquare(DM dmCoarse, DM dmFine, Mat *mass)
{
  PetscInt       n;
  void          *ctx;

  PetscFunctionBegin;
  PetscCall(DMSwarmGetLocalSize(dmCoarse, &n));
  PetscCall(MatCreate(PetscObjectComm((PetscObject) dmCoarse), mass));
  PetscCall(MatSetSizes(*mass, n, n, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(*mass, dmCoarse->mattype));
  PetscCall(DMGetApplicationContext(dmFine, &ctx));

  PetscCall(DMSwarmComputeMassMatrixSquare_Private(dmCoarse, dmFine, *mass, PETSC_TRUE, ctx));
  PetscCall(MatViewFromOptions(*mass, NULL, "-mass_sq_mat_view"));
  PetscFunctionReturn(0);
}

/*@C
   DMSwarmCreateGlobalVectorFromField - Creates a Vec object sharing the array associated with a given field

   Collective on dm

   Input parameters:
+  dm - a DMSwarm
-  fieldname - the textual name given to a registered field

   Output parameter:
.  vec - the vector

   Level: beginner

   Notes:
   The vector must be returned using a matching call to DMSwarmDestroyGlobalVectorFromField().

.seealso: DMSwarmRegisterPetscDatatypeField(), DMSwarmDestroyGlobalVectorFromField()
@*/
PetscErrorCode DMSwarmCreateGlobalVectorFromField(DM dm,const char fieldname[],Vec *vec)
{
  MPI_Comm       comm = PetscObjectComm((PetscObject) dm);

  PetscFunctionBegin;
  PetscCall(DMSwarmCreateVectorFromField_Private(dm, fieldname, comm, vec));
  PetscFunctionReturn(0);
}

/*@C
   DMSwarmDestroyGlobalVectorFromField - Destroys the Vec object which share the array associated with a given field

   Collective on dm

   Input parameters:
+  dm - a DMSwarm
-  fieldname - the textual name given to a registered field

   Output parameter:
.  vec - the vector

   Level: beginner

.seealso: DMSwarmRegisterPetscDatatypeField(), DMSwarmCreateGlobalVectorFromField()
@*/
PetscErrorCode DMSwarmDestroyGlobalVectorFromField(DM dm,const char fieldname[],Vec *vec)
{
  PetscFunctionBegin;
  PetscCall(DMSwarmDestroyVectorFromField_Private(dm, fieldname, vec));
  PetscFunctionReturn(0);
}

/*@C
   DMSwarmCreateLocalVectorFromField - Creates a Vec object sharing the array associated with a given field

   Collective on dm

   Input parameters:
+  dm - a DMSwarm
-  fieldname - the textual name given to a registered field

   Output parameter:
.  vec - the vector

   Level: beginner

   Notes:
   The vector must be returned using a matching call to DMSwarmDestroyLocalVectorFromField().

.seealso: DMSwarmRegisterPetscDatatypeField(), DMSwarmDestroyLocalVectorFromField()
@*/
PetscErrorCode DMSwarmCreateLocalVectorFromField(DM dm,const char fieldname[],Vec *vec)
{
  MPI_Comm       comm = PETSC_COMM_SELF;

  PetscFunctionBegin;
  PetscCall(DMSwarmCreateVectorFromField_Private(dm, fieldname, comm, vec));
  PetscFunctionReturn(0);
}

/*@C
   DMSwarmDestroyLocalVectorFromField - Destroys the Vec object which share the array associated with a given field

   Collective on dm

   Input parameters:
+  dm - a DMSwarm
-  fieldname - the textual name given to a registered field

   Output parameter:
.  vec - the vector

   Level: beginner

.seealso: DMSwarmRegisterPetscDatatypeField(), DMSwarmCreateLocalVectorFromField()
@*/
PetscErrorCode DMSwarmDestroyLocalVectorFromField(DM dm,const char fieldname[],Vec *vec)
{
  PetscFunctionBegin;
  PetscCall(DMSwarmDestroyVectorFromField_Private(dm, fieldname, vec));
  PetscFunctionReturn(0);
}

/*@C
   DMSwarmInitializeFieldRegister - Initiates the registration of fields to a DMSwarm

   Collective on dm

   Input parameter:
.  dm - a DMSwarm

   Level: beginner

   Notes:
   After all fields have been registered, you must call DMSwarmFinalizeFieldRegister().

.seealso: DMSwarmFinalizeFieldRegister(), DMSwarmRegisterPetscDatatypeField(),
          DMSwarmRegisterUserStructField(), DMSwarmRegisterUserDatatypeField()
@*/
PetscErrorCode DMSwarmInitializeFieldRegister(DM dm)
{
  DM_Swarm      *swarm = (DM_Swarm *) dm->data;

  PetscFunctionBegin;
  if (!swarm->field_registration_initialized) {
    swarm->field_registration_initialized = PETSC_TRUE;
    PetscCall(DMSwarmRegisterPetscDatatypeField(dm,DMSwarmField_pid,1,PETSC_INT64)); /* unique identifer */
    PetscCall(DMSwarmRegisterPetscDatatypeField(dm,DMSwarmField_rank,1,PETSC_INT)); /* used for communication */
  }
  PetscFunctionReturn(0);
}

/*@
   DMSwarmFinalizeFieldRegister - Finalizes the registration of fields to a DMSwarm

   Collective on dm

   Input parameter:
.  dm - a DMSwarm

   Level: beginner

   Notes:
   After DMSwarmFinalizeFieldRegister() has been called, no new fields can be defined on the DMSwarm.

.seealso: DMSwarmInitializeFieldRegister(), DMSwarmRegisterPetscDatatypeField(),
          DMSwarmRegisterUserStructField(), DMSwarmRegisterUserDatatypeField()
@*/
PetscErrorCode DMSwarmFinalizeFieldRegister(DM dm)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;

  PetscFunctionBegin;
  if (!swarm->field_registration_finalized) {
    PetscCall(DMSwarmDataBucketFinalize(swarm->db));
  }
  swarm->field_registration_finalized = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   DMSwarmSetLocalSizes - Sets the length of all registered fields on the DMSwarm

   Not collective

   Input parameters:
+  dm - a DMSwarm
.  nlocal - the length of each registered field
-  buffer - the length of the buffer used to efficient dynamic re-sizing

   Level: beginner

.seealso: DMSwarmGetLocalSize()
@*/
PetscErrorCode DMSwarmSetLocalSizes(DM dm,PetscInt nlocal,PetscInt buffer)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMSWARM_SetSizes,0,0,0,0));
  PetscCall(DMSwarmDataBucketSetSizes(swarm->db,nlocal,buffer));
  PetscCall(PetscLogEventEnd(DMSWARM_SetSizes,0,0,0,0));
  PetscFunctionReturn(0);
}

/*@
   DMSwarmSetCellDM - Attachs a DM to a DMSwarm

   Collective on dm

   Input parameters:
+  dm - a DMSwarm
-  dmcell - the DM to attach to the DMSwarm

   Level: beginner

   Notes:
   The attached DM (dmcell) will be queried for point location and
   neighbor MPI-rank information if DMSwarmMigrate() is called.

.seealso: DMSwarmSetType(), DMSwarmGetCellDM(), DMSwarmMigrate()
@*/
PetscErrorCode DMSwarmSetCellDM(DM dm,DM dmcell)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;

  PetscFunctionBegin;
  swarm->dmcell = dmcell;
  PetscFunctionReturn(0);
}

/*@
   DMSwarmGetCellDM - Fetches the attached cell DM

   Collective on dm

   Input parameter:
.  dm - a DMSwarm

   Output parameter:
.  dmcell - the DM which was attached to the DMSwarm

   Level: beginner

.seealso: DMSwarmSetCellDM()
@*/
PetscErrorCode DMSwarmGetCellDM(DM dm,DM *dmcell)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;

  PetscFunctionBegin;
  *dmcell = swarm->dmcell;
  PetscFunctionReturn(0);
}

/*@
   DMSwarmGetLocalSize - Retrives the local length of fields registered

   Not collective

   Input parameter:
.  dm - a DMSwarm

   Output parameter:
.  nlocal - the length of each registered field

   Level: beginner

.seealso: DMSwarmGetSize(), DMSwarmSetLocalSizes()
@*/
PetscErrorCode DMSwarmGetLocalSize(DM dm,PetscInt *nlocal)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;

  PetscFunctionBegin;
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db,nlocal,NULL,NULL));
  PetscFunctionReturn(0);
}

/*@
   DMSwarmGetSize - Retrives the total length of fields registered

   Collective on dm

   Input parameter:
.  dm - a DMSwarm

   Output parameter:
.  n - the total length of each registered field

   Level: beginner

   Note:
   This calls MPI_Allreduce upon each call (inefficient but safe)

.seealso: DMSwarmGetLocalSize(), DMSwarmSetLocalSizes()
@*/
PetscErrorCode DMSwarmGetSize(DM dm,PetscInt *n)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;
  PetscInt       nlocal;

  PetscFunctionBegin;
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db,&nlocal,NULL,NULL));
  PetscCallMPI(MPI_Allreduce(&nlocal,n,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)dm)));
  PetscFunctionReturn(0);
}

/*@C
   DMSwarmRegisterPetscDatatypeField - Register a field to a DMSwarm with a native PETSc data type

   Collective on dm

   Input parameters:
+  dm - a DMSwarm
.  fieldname - the textual name to identify this field
.  blocksize - the number of each data type
-  type - a valid PETSc data type (PETSC_CHAR, PETSC_SHORT, PETSC_INT, PETSC_FLOAT, PETSC_REAL, PETSC_LONG)

   Level: beginner

   Notes:
   The textual name for each registered field must be unique.

.seealso: DMSwarmRegisterUserStructField(), DMSwarmRegisterUserDatatypeField()
@*/
PetscErrorCode DMSwarmRegisterPetscDatatypeField(DM dm,const char fieldname[],PetscInt blocksize,PetscDataType type)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;
  size_t         size;

  PetscFunctionBegin;
  PetscCheck(swarm->field_registration_initialized,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Must call DMSwarmInitializeFieldRegister() first");
  PetscCheck(!swarm->field_registration_finalized,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Cannot register additional fields after calling DMSwarmFinalizeFieldRegister() first");

  PetscCheck(type != PETSC_OBJECT,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Valid for {char,short,int,long,float,double}");
  PetscCheck(type != PETSC_FUNCTION,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Valid for {char,short,int,long,float,double}");
  PetscCheck(type != PETSC_STRING,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Valid for {char,short,int,long,float,double}");
  PetscCheck(type != PETSC_STRUCT,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Valid for {char,short,int,long,float,double}");
  PetscCheck(type != PETSC_DATATYPE_UNKNOWN,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Valid for {char,short,int,long,float,double}");

  PetscCall(PetscDataTypeGetSize(type, &size));
  /* Load a specific data type into data bucket, specifying textual name and its size in bytes */
  PetscCall(DMSwarmDataBucketRegisterField(swarm->db,"DMSwarmRegisterPetscDatatypeField",fieldname,blocksize*size,NULL));
  {
    DMSwarmDataField gfield;

    PetscCall(DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db,fieldname,&gfield));
    PetscCall(DMSwarmDataFieldSetBlockSize(gfield,blocksize));
  }
  swarm->db->field[swarm->db->nfields-1]->petsc_type = type;
  PetscFunctionReturn(0);
}

/*@C
   DMSwarmRegisterUserStructField - Register a user defined struct to a DMSwarm

   Collective on dm

   Input parameters:
+  dm - a DMSwarm
.  fieldname - the textual name to identify this field
-  size - the size in bytes of the user struct of each data type

   Level: beginner

   Notes:
   The textual name for each registered field must be unique.

.seealso: DMSwarmRegisterPetscDatatypeField(), DMSwarmRegisterUserDatatypeField()
@*/
PetscErrorCode DMSwarmRegisterUserStructField(DM dm,const char fieldname[],size_t size)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;

  PetscFunctionBegin;
  PetscCall(DMSwarmDataBucketRegisterField(swarm->db,"DMSwarmRegisterUserStructField",fieldname,size,NULL));
  swarm->db->field[swarm->db->nfields-1]->petsc_type = PETSC_STRUCT ;
  PetscFunctionReturn(0);
}

/*@C
   DMSwarmRegisterUserDatatypeField - Register a user defined data type to a DMSwarm

   Collective on dm

   Input parameters:
+  dm - a DMSwarm
.  fieldname - the textual name to identify this field
.  size - the size in bytes of the user data type
-  blocksize - the number of each data type

   Level: beginner

   Notes:
   The textual name for each registered field must be unique.

.seealso: DMSwarmRegisterPetscDatatypeField(), DMSwarmRegisterUserStructField(), DMSwarmRegisterUserDatatypeField()
@*/
PetscErrorCode DMSwarmRegisterUserDatatypeField(DM dm,const char fieldname[],size_t size,PetscInt blocksize)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;

  PetscFunctionBegin;
  PetscCall(DMSwarmDataBucketRegisterField(swarm->db,"DMSwarmRegisterUserDatatypeField",fieldname,blocksize*size,NULL));
  {
    DMSwarmDataField gfield;

    PetscCall(DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db,fieldname,&gfield));
    PetscCall(DMSwarmDataFieldSetBlockSize(gfield,blocksize));
  }
  swarm->db->field[swarm->db->nfields-1]->petsc_type = PETSC_DATATYPE_UNKNOWN;
  PetscFunctionReturn(0);
}

/*@C
   DMSwarmGetField - Get access to the underlying array storing all entries associated with a registered field

   Not collective

   Input parameters:
+  dm - a DMSwarm
-  fieldname - the textual name to identify this field

   Output parameters:
+  blocksize - the number of each data type
.  type - the data type
-  data - pointer to raw array

   Level: beginner

   Notes:
   The array must be returned using a matching call to DMSwarmRestoreField().

.seealso: DMSwarmRestoreField()
@*/
PetscErrorCode DMSwarmGetField(DM dm,const char fieldname[],PetscInt *blocksize,PetscDataType *type,void **data)
{
  DM_Swarm         *swarm = (DM_Swarm*)dm->data;
  DMSwarmDataField gfield;

  PetscFunctionBegin;
  if (!swarm->issetup) PetscCall(DMSetUp(dm));
  PetscCall(DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db,fieldname,&gfield));
  PetscCall(DMSwarmDataFieldGetAccess(gfield));
  PetscCall(DMSwarmDataFieldGetEntries(gfield,data));
  if (blocksize) {*blocksize = gfield->bs; }
  if (type) { *type = gfield->petsc_type; }
  PetscFunctionReturn(0);
}

/*@C
   DMSwarmRestoreField - Restore access to the underlying array storing all entries associated with a registered field

   Not collective

   Input parameters:
+  dm - a DMSwarm
-  fieldname - the textual name to identify this field

   Output parameters:
+  blocksize - the number of each data type
.  type - the data type
-  data - pointer to raw array

   Level: beginner

   Notes:
   The user must call DMSwarmGetField() prior to calling DMSwarmRestoreField().

.seealso: DMSwarmGetField()
@*/
PetscErrorCode DMSwarmRestoreField(DM dm,const char fieldname[],PetscInt *blocksize,PetscDataType *type,void **data)
{
  DM_Swarm         *swarm = (DM_Swarm*)dm->data;
  DMSwarmDataField gfield;

  PetscFunctionBegin;
  PetscCall(DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db,fieldname,&gfield));
  PetscCall(DMSwarmDataFieldRestoreAccess(gfield));
  if (data) *data = NULL;
  PetscFunctionReturn(0);
}

/*@
   DMSwarmAddPoint - Add space for one new point in the DMSwarm

   Not collective

   Input parameter:
.  dm - a DMSwarm

   Level: beginner

   Notes:
   The new point will have all fields initialized to zero.

.seealso: DMSwarmAddNPoints()
@*/
PetscErrorCode DMSwarmAddPoint(DM dm)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;

  PetscFunctionBegin;
  if (!swarm->issetup) PetscCall(DMSetUp(dm));
  PetscCall(PetscLogEventBegin(DMSWARM_AddPoints,0,0,0,0));
  PetscCall(DMSwarmDataBucketAddPoint(swarm->db));
  PetscCall(PetscLogEventEnd(DMSWARM_AddPoints,0,0,0,0));
  PetscFunctionReturn(0);
}

/*@
   DMSwarmAddNPoints - Add space for a number of new points in the DMSwarm

   Not collective

   Input parameters:
+  dm - a DMSwarm
-  npoints - the number of new points to add

   Level: beginner

   Notes:
   The new point will have all fields initialized to zero.

.seealso: DMSwarmAddPoint()
@*/
PetscErrorCode DMSwarmAddNPoints(DM dm,PetscInt npoints)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;
  PetscInt       nlocal;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMSWARM_AddPoints,0,0,0,0));
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db,&nlocal,NULL,NULL));
  nlocal = nlocal + npoints;
  PetscCall(DMSwarmDataBucketSetSizes(swarm->db,nlocal,DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
  PetscCall(PetscLogEventEnd(DMSWARM_AddPoints,0,0,0,0));
  PetscFunctionReturn(0);
}

/*@
   DMSwarmRemovePoint - Remove the last point from the DMSwarm

   Not collective

   Input parameter:
.  dm - a DMSwarm

   Level: beginner

.seealso: DMSwarmRemovePointAtIndex()
@*/
PetscErrorCode DMSwarmRemovePoint(DM dm)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMSWARM_RemovePoints,0,0,0,0));
  PetscCall(DMSwarmDataBucketRemovePoint(swarm->db));
  PetscCall(PetscLogEventEnd(DMSWARM_RemovePoints,0,0,0,0));
  PetscFunctionReturn(0);
}

/*@
   DMSwarmRemovePointAtIndex - Removes a specific point from the DMSwarm

   Not collective

   Input parameters:
+  dm - a DMSwarm
-  idx - index of point to remove

   Level: beginner

.seealso: DMSwarmRemovePoint()
@*/
PetscErrorCode DMSwarmRemovePointAtIndex(DM dm,PetscInt idx)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMSWARM_RemovePoints,0,0,0,0));
  PetscCall(DMSwarmDataBucketRemovePointAtIndex(swarm->db,idx));
  PetscCall(PetscLogEventEnd(DMSWARM_RemovePoints,0,0,0,0));
  PetscFunctionReturn(0);
}

/*@
   DMSwarmCopyPoint - Copy point pj to point pi in the DMSwarm

   Not collective

   Input parameters:
+  dm - a DMSwarm
.  pi - the index of the point to copy
-  pj - the point index where the copy should be located

 Level: beginner

.seealso: DMSwarmRemovePoint()
@*/
PetscErrorCode DMSwarmCopyPoint(DM dm,PetscInt pi,PetscInt pj)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;

  PetscFunctionBegin;
  if (!swarm->issetup) PetscCall(DMSetUp(dm));
  PetscCall(DMSwarmDataBucketCopyPoint(swarm->db,pi,swarm->db,pj));
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmMigrate_Basic(DM dm,PetscBool remove_sent_points)
{
  PetscFunctionBegin;
  PetscCall(DMSwarmMigrate_Push_Basic(dm,remove_sent_points));
  PetscFunctionReturn(0);
}

/*@
   DMSwarmMigrate - Relocates points defined in the DMSwarm to other MPI-ranks

   Collective on dm

   Input parameters:
+  dm - the DMSwarm
-  remove_sent_points - flag indicating if sent points should be removed from the current MPI-rank

   Notes:
   The DM will be modified to accommodate received points.
   If remove_sent_points = PETSC_TRUE, any points that were sent will be removed from the DM.
   Different styles of migration are supported. See DMSwarmSetMigrateType().

   Level: advanced

.seealso: DMSwarmSetMigrateType()
@*/
PetscErrorCode DMSwarmMigrate(DM dm,PetscBool remove_sent_points)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMSWARM_Migrate,0,0,0,0));
  switch (swarm->migrate_type) {
    case DMSWARM_MIGRATE_BASIC:
      PetscCall(DMSwarmMigrate_Basic(dm,remove_sent_points));
      break;
    case DMSWARM_MIGRATE_DMCELLNSCATTER:
      PetscCall(DMSwarmMigrate_CellDMScatter(dm,remove_sent_points));
      break;
    case DMSWARM_MIGRATE_DMCELLEXACT:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"DMSWARM_MIGRATE_DMCELLEXACT not implemented");
    case DMSWARM_MIGRATE_USER:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"DMSWARM_MIGRATE_USER not implemented");
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"DMSWARM_MIGRATE type unknown");
  }
  PetscCall(PetscLogEventEnd(DMSWARM_Migrate,0,0,0,0));
  PetscCall(DMClearGlobalVectors(dm));
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmMigrate_GlobalToLocal_Basic(DM dm,PetscInt *globalsize);

/*
 DMSwarmCollectViewCreate

 * Applies a collection method and gathers point neighbour points into dm

 Notes:
 Users should call DMSwarmCollectViewDestroy() after
 they have finished computations associated with the collected points
*/

/*@
   DMSwarmCollectViewCreate - Applies a collection method and gathers points
                              in neighbour ranks into the DMSwarm

   Collective on dm

   Input parameter:
.  dm - the DMSwarm

   Notes:
   Users should call DMSwarmCollectViewDestroy() after
   they have finished computations associated with the collected points
   Different collect methods are supported. See DMSwarmSetCollectType().

   Level: advanced

.seealso: DMSwarmCollectViewDestroy(), DMSwarmSetCollectType()
@*/
PetscErrorCode DMSwarmCollectViewCreate(DM dm)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;
  PetscInt       ng;

  PetscFunctionBegin;
  PetscCheck(!swarm->collect_view_active,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"CollectView currently active");
  PetscCall(DMSwarmGetLocalSize(dm,&ng));
  switch (swarm->collect_type) {

    case DMSWARM_COLLECT_BASIC:
      PetscCall(DMSwarmMigrate_GlobalToLocal_Basic(dm,&ng));
      break;
    case DMSWARM_COLLECT_DMDABOUNDINGBOX:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"DMSWARM_COLLECT_DMDABOUNDINGBOX not implemented");
    case DMSWARM_COLLECT_GENERAL:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"DMSWARM_COLLECT_GENERAL not implemented");
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"DMSWARM_COLLECT type unknown");
  }
  swarm->collect_view_active = PETSC_TRUE;
  swarm->collect_view_reset_nlocal = ng;
  PetscFunctionReturn(0);
}

/*@
   DMSwarmCollectViewDestroy - Resets the DMSwarm to the size prior to calling DMSwarmCollectViewCreate()

   Collective on dm

   Input parameters:
.  dm - the DMSwarm

   Notes:
   Users should call DMSwarmCollectViewCreate() before this function is called.

   Level: advanced

.seealso: DMSwarmCollectViewCreate(), DMSwarmSetCollectType()
@*/
PetscErrorCode DMSwarmCollectViewDestroy(DM dm)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;

  PetscFunctionBegin;
  PetscCheck(swarm->collect_view_active,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"CollectView is currently not active");
  PetscCall(DMSwarmSetLocalSizes(dm,swarm->collect_view_reset_nlocal,-1));
  swarm->collect_view_active = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmSetUpPIC(DM dm)
{
  PetscInt       dim;

  PetscFunctionBegin;
  PetscCall(DMSwarmSetNumSpecies(dm, 1));
  PetscCall(DMGetDimension(dm,&dim));
  PetscCheck(dim >= 1,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Dimension must be 1,2,3 - found %D",dim);
  PetscCheck(dim <= 3,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Dimension must be 1,2,3 - found %D",dim);
  PetscCall(DMSwarmRegisterPetscDatatypeField(dm,DMSwarmPICField_coor,dim,PETSC_DOUBLE));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dm,DMSwarmPICField_cellid,1,PETSC_INT));
  PetscFunctionReturn(0);
}

/*@
  DMSwarmSetPointCoordinatesRandom - Sets initial coordinates for particles in each cell

  Collective on dm

  Input parameters:
+ dm  - the DMSwarm
- Npc - The number of particles per cell in the cell DM

  Notes:
  The user must use DMSwarmSetCellDM() to set the cell DM first. The particles are placed randomly inside each cell. If only
  one particle is in each cell, it is placed at the centroid.

  Level: intermediate

.seealso: DMSwarmSetCellDM()
@*/
PetscErrorCode DMSwarmSetPointCoordinatesRandom(DM dm, PetscInt Npc)
{
  DM             cdm;
  PetscRandom    rnd;
  DMPolytopeType ct;
  PetscBool      simplex;
  PetscReal     *centroid, *coords, *xi0, *v0, *J, *invJ, detJ;
  PetscInt       dim, d, cStart, cEnd, c, p;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject) dm), &rnd));
  PetscCall(PetscRandomSetInterval(rnd, -1.0, 1.0));
  PetscCall(PetscRandomSetType(rnd, PETSCRAND48));

  PetscCall(DMSwarmGetCellDM(dm, &cdm));
  PetscCall(DMGetDimension(cdm, &dim));
  PetscCall(DMPlexGetHeightStratum(cdm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetCellType(cdm, cStart, &ct));
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;

  PetscCall(PetscMalloc5(dim, &centroid, dim, &xi0, dim, &v0, dim*dim, &J, dim*dim, &invJ));
  for (d = 0; d < dim; ++d) xi0[d] = -1.0;
  PetscCall(DMSwarmGetField(dm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  for (c = cStart; c < cEnd; ++c) {
    if (Npc == 1) {
      PetscCall(DMPlexComputeCellGeometryFVM(cdm, c, NULL, centroid, NULL));
      for (d = 0; d < dim; ++d) coords[c*dim+d] = centroid[d];
    } else {
      PetscCall(DMPlexComputeCellGeometryFEM(cdm, c, NULL, v0, J, invJ, &detJ)); /* affine */
      for (p = 0; p < Npc; ++p) {
        const PetscInt n   = c*Npc + p;
        PetscReal      sum = 0.0, refcoords[3];

        for (d = 0; d < dim; ++d) {
          PetscCall(PetscRandomGetValueReal(rnd, &refcoords[d]));
          sum += refcoords[d];
        }
        if (simplex && sum > 0.0) for (d = 0; d < dim; ++d) refcoords[d] -= PetscSqrtReal(dim)*sum;
        CoordinatesRefToReal(dim, dim, xi0, v0, J, refcoords, &coords[n*dim]);
      }
    }
  }
  PetscCall(DMSwarmRestoreField(dm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  PetscCall(PetscFree5(centroid, xi0, v0, J, invJ));
  PetscCall(PetscRandomDestroy(&rnd));
  PetscFunctionReturn(0);
}

/*@
   DMSwarmSetType - Set particular flavor of DMSwarm

   Collective on dm

   Input parameters:
+  dm - the DMSwarm
-  stype - the DMSwarm type (e.g. DMSWARM_PIC)

   Level: advanced

.seealso: DMSwarmSetMigrateType(), DMSwarmSetCollectType(), DMSwarmType, DMSWARM_PIC, DMSWARM_BASIC
@*/
PetscErrorCode DMSwarmSetType(DM dm,DMSwarmType stype)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;

  PetscFunctionBegin;
  swarm->swarm_type = stype;
  if (swarm->swarm_type == DMSWARM_PIC) {
    PetscCall(DMSwarmSetUpPIC(dm));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMSetup_Swarm(DM dm)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;
  PetscMPIInt    rank;
  PetscInt       p,npoints,*rankval;

  PetscFunctionBegin;
  if (swarm->issetup) PetscFunctionReturn(0);
  swarm->issetup = PETSC_TRUE;

  if (swarm->swarm_type == DMSWARM_PIC) {
    /* check dmcell exists */
    PetscCheck(swarm->dmcell,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"DMSWARM_PIC requires you call DMSwarmSetCellDM");

    if (swarm->dmcell->ops->locatepointssubdomain) {
      /* check methods exists for exact ownership identificiation */
      PetscCall(PetscInfo(dm, "DMSWARM_PIC: Using method CellDM->ops->LocatePointsSubdomain\n"));
      swarm->migrate_type = DMSWARM_MIGRATE_DMCELLEXACT;
    } else {
      /* check methods exist for point location AND rank neighbor identification */
      if (swarm->dmcell->ops->locatepoints) {
        PetscCall(PetscInfo(dm, "DMSWARM_PIC: Using method CellDM->LocatePoints\n"));
      } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"DMSWARM_PIC requires the method CellDM->ops->locatepoints be defined");

      if (swarm->dmcell->ops->getneighbors) {
        PetscCall(PetscInfo(dm, "DMSWARM_PIC: Using method CellDM->GetNeigbors\n"));
      } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"DMSWARM_PIC requires the method CellDM->ops->getneighbors be defined");

      swarm->migrate_type = DMSWARM_MIGRATE_DMCELLNSCATTER;
    }
  }

  PetscCall(DMSwarmFinalizeFieldRegister(dm));

  /* check some fields were registered */
  PetscCheck(swarm->db->nfields > 2,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"At least one field user must be registered via DMSwarmRegisterXXX()");

  /* check local sizes were set */
  PetscCheck(swarm->db->L != -1,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Local sizes must be set via DMSwarmSetLocalSizes()");

  /* initialize values in pid and rank placeholders */
  /* TODO: [pid - use MPI_Scan] */
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL));
  PetscCall(DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval));
  for (p=0; p<npoints; p++) {
    rankval[p] = (PetscInt)rank;
  }
  PetscCall(DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval));
  PetscFunctionReturn(0);
}

extern PetscErrorCode DMSwarmSortDestroy(DMSwarmSort *_ctx);

PetscErrorCode DMDestroy_Swarm(DM dm)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;

  PetscFunctionBegin;
  if (--swarm->refct > 0) PetscFunctionReturn(0);
  PetscCall(DMSwarmDataBucketDestroy(&swarm->db));
  if (swarm->sort_context) {
    PetscCall(DMSwarmSortDestroy(&swarm->sort_context));
  }
  PetscCall(PetscFree(swarm));
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmView_Draw(DM dm, PetscViewer viewer)
{
  DM             cdm;
  PetscDraw      draw;
  PetscReal     *coords, oldPause, radius = 0.01;
  PetscInt       Np, p, bs;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetReal(NULL, ((PetscObject) dm)->prefix, "-dm_view_swarm_radius", &radius, NULL));
  PetscCall(PetscViewerDrawGetDraw(viewer, 0, &draw));
  PetscCall(DMSwarmGetCellDM(dm, &cdm));
  PetscCall(PetscDrawGetPause(draw, &oldPause));
  PetscCall(PetscDrawSetPause(draw, 0.0));
  PetscCall(DMView(cdm, viewer));
  PetscCall(PetscDrawSetPause(draw, oldPause));

  PetscCall(DMSwarmGetLocalSize(dm, &Np));
  PetscCall(DMSwarmGetField(dm, DMSwarmPICField_coor, &bs, NULL, (void **) &coords));
  for (p = 0; p < Np; ++p) {
    const PetscInt i = p*bs;

    PetscCall(PetscDrawEllipse(draw, coords[i], coords[i+1], radius, radius, PETSC_DRAW_BLUE));
  }
  PetscCall(DMSwarmRestoreField(dm, DMSwarmPICField_coor, &bs, NULL, (void **) &coords));
  PetscCall(PetscDrawFlush(draw));
  PetscCall(PetscDrawPause(draw));
  PetscCall(PetscDrawSave(draw));
  PetscFunctionReturn(0);
}

PetscErrorCode DMView_Swarm(DM dm, PetscViewer viewer)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;
  PetscBool      iascii,ibinary,isvtk,isdraw;
#if defined(PETSC_HAVE_HDF5)
  PetscBool      ishdf5;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY,&ibinary));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERVTK,   &isvtk));
#if defined(PETSC_HAVE_HDF5)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERHDF5,  &ishdf5));
#endif
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERDRAW,  &isdraw));
  if (iascii) {
    PetscCall(DMSwarmDataBucketView(PetscObjectComm((PetscObject)dm),swarm->db,NULL,DATABUCKET_VIEW_STDOUT));
  } else PetscCheck(!ibinary,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"NO Binary support");
#if defined(PETSC_HAVE_HDF5)
  else if (ishdf5) {
    PetscCall(DMSwarmView_HDF5(dm, viewer));
  }
#endif
  else if (isdraw) {
    PetscCall(DMSwarmView_Draw(dm, viewer));
  }
  PetscFunctionReturn(0);
}

/*@C
   DMSwarmGetCellSwarm - Extracts a single cell from the DMSwarm object, returns it as a single cell DMSwarm.
   The cell DM is filtered for fields of that cell, and the filtered DM is used as the cell DM of the new swarm object.

   Important: Changes to this cell of the swarm will be lost if they are made prior to restoring this cell.

   Noncollective

   Input parameters:
+  sw - the DMSwarm
.  cellID - the integer id of the cell to be extracted and filtered
-  cellswarm - The DMSwarm to receive the cell

   Level: beginner

   Notes:
      This presently only supports DMSWARM_PIC type

      Should be restored with DMSwarmRestoreCellSwarm()

.seealso: DMSwarmRestoreCellSwarm()
@*/
PETSC_EXTERN PetscErrorCode DMSwarmGetCellSwarm(DM sw, PetscInt cellID, DM cellswarm)
{
  DM_Swarm      *original = (DM_Swarm*) sw->data;
  DMLabel        label;
  DM             dmc, subdmc;
  PetscInt      *pids, particles, dim;

  PetscFunctionBegin;
  /* Configure new swarm */
  PetscCall(DMSetType(cellswarm, DMSWARM));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSetDimension(cellswarm, dim));
  PetscCall(DMSwarmSetType(cellswarm, DMSWARM_PIC));
  /* Destroy the unused, unconfigured data bucket to prevent stragglers in memory */
  PetscCall(DMSwarmDataBucketDestroy(&((DM_Swarm*)cellswarm->data)->db));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMSwarmSortGetNumberOfPointsPerCell(sw, cellID, &particles));
  PetscCall(DMSwarmSortGetPointsPerCell(sw, cellID, &particles, &pids));
  PetscCall(DMSwarmDataBucketCreateFromSubset(original->db, particles, pids, &((DM_Swarm*)cellswarm->data)->db));
  PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCall(PetscFree(pids));
  PetscCall(DMSwarmGetCellDM(sw, &dmc));
  PetscCall(DMLabelCreate(PetscObjectComm((PetscObject)sw), "singlecell", &label));
  PetscCall(DMAddLabel(dmc, label));
  PetscCall(DMLabelSetValue(label, cellID, 1));
  PetscCall(DMPlexFilter(dmc, label, 1, &subdmc));
  PetscCall(DMSwarmSetCellDM(cellswarm, subdmc));
  PetscCall(DMLabelDestroy(&label));
  PetscFunctionReturn(0);
}

/*@C
   DMSwarmRestoreCellSwarm - Restores a DMSwarm object obtained with DMSwarmGetCellSwarm(). All fields are copied back into the parent swarm.

   Noncollective

   Input parameters:
+  sw - the parent DMSwarm
.  cellID - the integer id of the cell to be copied back into the parent swarm
-  cellswarm - the cell swarm object

   Level: beginner

   Note:
    This only supports DMSWARM_PIC types of DMSwarms

.seealso: DMSwarmGetCellSwarm()
@*/
PETSC_EXTERN PetscErrorCode DMSwarmRestoreCellSwarm(DM sw, PetscInt cellID, DM cellswarm)
{
  DM             dmc;
  PetscInt       *pids, particles, p;

  PetscFunctionBegin;
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMSwarmSortGetPointsPerCell(sw, cellID, &particles, &pids));
  PetscCall(DMSwarmSortRestoreAccess(sw));
  /* Pointwise copy of each particle based on pid. The parent swarm may not be altered during this process. */
  for (p=0; p<particles; ++p) {
    PetscCall(DMSwarmDataBucketCopyPoint(((DM_Swarm*)cellswarm->data)->db,pids[p],((DM_Swarm*)sw->data)->db,pids[p]));
  }
  /* Free memory, destroy cell dm */
  PetscCall(DMSwarmGetCellDM(cellswarm, &dmc));
  PetscCall(DMDestroy(&dmc));
  PetscCall(PetscFree(pids));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMClone_Swarm(DM, DM *);

static PetscErrorCode DMInitialize_Swarm(DM sw)
{
  PetscFunctionBegin;
  sw->dim  = 0;
  sw->ops->view                            = DMView_Swarm;
  sw->ops->load                            = NULL;
  sw->ops->setfromoptions                  = NULL;
  sw->ops->clone                           = DMClone_Swarm;
  sw->ops->setup                           = DMSetup_Swarm;
  sw->ops->createlocalsection              = NULL;
  sw->ops->createdefaultconstraints        = NULL;
  sw->ops->createglobalvector              = DMCreateGlobalVector_Swarm;
  sw->ops->createlocalvector               = DMCreateLocalVector_Swarm;
  sw->ops->getlocaltoglobalmapping         = NULL;
  sw->ops->createfieldis                   = NULL;
  sw->ops->createcoordinatedm              = NULL;
  sw->ops->getcoloring                     = NULL;
  sw->ops->creatematrix                    = DMCreateMatrix_Swarm;
  sw->ops->createinterpolation             = NULL;
  sw->ops->createinjection                 = NULL;
  sw->ops->createmassmatrix                = DMCreateMassMatrix_Swarm;
  sw->ops->refine                          = NULL;
  sw->ops->coarsen                         = NULL;
  sw->ops->refinehierarchy                 = NULL;
  sw->ops->coarsenhierarchy                = NULL;
  sw->ops->globaltolocalbegin              = NULL;
  sw->ops->globaltolocalend                = NULL;
  sw->ops->localtoglobalbegin              = NULL;
  sw->ops->localtoglobalend                = NULL;
  sw->ops->destroy                         = DMDestroy_Swarm;
  sw->ops->createsubdm                     = NULL;
  sw->ops->getdimpoints                    = NULL;
  sw->ops->locatepoints                    = NULL;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMClone_Swarm(DM dm, DM *newdm)
{
  DM_Swarm       *swarm = (DM_Swarm *) dm->data;

  PetscFunctionBegin;
  swarm->refct++;
  (*newdm)->data = swarm;
  PetscCall(PetscObjectChangeTypeName((PetscObject) *newdm, DMSWARM));
  PetscCall(DMInitialize_Swarm(*newdm));
  (*newdm)->dim = dm->dim;
  PetscFunctionReturn(0);
}

/*MC

 DMSWARM = "swarm" - A DM object used to represent arrays of data (fields) of arbitrary data type.
 This implementation was designed for particle methods in which the underlying
 data required to be represented is both (i) dynamic in length, (ii) and of arbitrary data type.

 User data can be represented by DMSwarm through a registering "fields".
 To register a field, the user must provide;
 (a) a unique name;
 (b) the data type (or size in bytes);
 (c) the block size of the data.

 For example, suppose the application requires a unique id, energy, momentum and density to be stored
 on a set of particles. Then the following code could be used

$    DMSwarmInitializeFieldRegister(dm)
$    DMSwarmRegisterPetscDatatypeField(dm,"uid",1,PETSC_LONG);
$    DMSwarmRegisterPetscDatatypeField(dm,"energy",1,PETSC_REAL);
$    DMSwarmRegisterPetscDatatypeField(dm,"momentum",3,PETSC_REAL);
$    DMSwarmRegisterPetscDatatypeField(dm,"density",1,PETSC_FLOAT);
$    DMSwarmFinalizeFieldRegister(dm)

 The fields represented by DMSwarm are dynamic and can be re-sized at any time.
 The only restriction imposed by DMSwarm is that all fields contain the same number of points.

 To support particle methods, "migration" techniques are provided. These methods migrate data
 between ranks.

 DMSwarm supports the methods DMCreateGlobalVector() and DMCreateLocalVector().
 As a DMSwarm may internally define and store values of different data types,
 before calling DMCreateGlobalVector() or DMCreateLocalVector(), the user must inform DMSwarm which
 fields should be used to define a Vec object via
   DMSwarmVectorDefineField()
 The specified field can be changed at any time - thereby permitting vectors
 compatible with different fields to be created.

 A dual representation of fields in the DMSwarm and a Vec object is permitted via
   DMSwarmCreateGlobalVectorFromField()
 Here the data defining the field in the DMSwarm is shared with a Vec.
 This is inherently unsafe if you alter the size of the field at any time between
 calls to DMSwarmCreateGlobalVectorFromField() and DMSwarmDestroyGlobalVectorFromField().
 If the local size of the DMSwarm does not match the local size of the global vector
 when DMSwarmDestroyGlobalVectorFromField() is called, an error is thrown.

 Additional high-level support is provided for Particle-In-Cell methods.
 Please refer to the man page for DMSwarmSetType().

 Level: beginner

.seealso: DMType, DMCreate(), DMSetType()
M*/
PETSC_EXTERN PetscErrorCode DMCreate_Swarm(DM dm)
{
  DM_Swarm      *swarm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscNewLog(dm,&swarm));
  dm->data = swarm;
  PetscCall(DMSwarmDataBucketCreate(&swarm->db));
  PetscCall(DMSwarmInitializeFieldRegister(dm));
  swarm->refct = 1;
  swarm->vec_field_set                  = PETSC_FALSE;
  swarm->issetup                        = PETSC_FALSE;
  swarm->swarm_type                     = DMSWARM_BASIC;
  swarm->migrate_type                   = DMSWARM_MIGRATE_BASIC;
  swarm->collect_type                   = DMSWARM_COLLECT_BASIC;
  swarm->migrate_error_on_missing_point = PETSC_FALSE;
  swarm->dmcell                         = NULL;
  swarm->collect_view_active            = PETSC_FALSE;
  swarm->collect_view_reset_nlocal      = -1;
  PetscCall(DMInitialize_Swarm(dm));
  PetscFunctionReturn(0);
}
