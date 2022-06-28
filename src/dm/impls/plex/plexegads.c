#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/hashmapi.h>

#ifdef PETSC_HAVE_EGADS
#include <egads.h>
#endif

/* We need to understand how to natively parse STEP files. There seems to be only one open source implementation of
   the STEP parser contained in the OpenCASCADE package. It is enough to make a strong man weep:

     https://github.com/tpaviot/oce/tree/master/src/STEPControl

   The STEP, and inner EXPRESS, formats are ISO standards, so they are documented

     https://stackoverflow.com/questions/26774037/documentation-or-specification-for-step-and-stp-files
     http://stepmod.sourceforge.net/express_model_spec/

   but again it seems that there has been a deliberate effort at obfuscation, probably to raise the bar for entrants.
*/

#ifdef PETSC_HAVE_EGADS
PETSC_INTERN PetscErrorCode DMPlexSnapToGeomModel_EGADS_Internal(DM, PetscInt, ego, PetscInt, PetscInt, PetscInt, const PetscScalar[], PetscScalar[]);
PETSC_INTERN PetscErrorCode DMPlexSnapToGeomModel_EGADSLite_Internal(DM, PetscInt, ego, PetscInt, PetscInt, PetscInt, const PetscScalar[], PetscScalar[]);

PetscErrorCode DMPlexSnapToGeomModel_EGADS_Internal(DM dm, PetscInt p, ego model, PetscInt bodyID, PetscInt faceID, PetscInt edgeID, const PetscScalar mcoords[], PetscScalar gcoords[])
{
  DM             cdm;
  ego           *bodies;
  ego            geom, body, obj;
  /* result has to hold derivatives, along with the value */
  double         params[3], result[18], paramsV[16*3], resultV[16*3], range[4];
  int            Nb, oclass, mtype, *senses, peri;
  Vec            coordinatesLocal;
  PetscScalar   *coords = NULL;
  PetscInt       Nv, v, Np = 0, pm;
  PetscInt       dE, d;

  PetscFunctionBeginHot;
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinateDim(dm, &dE));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinatesLocal));
  PetscCall(EG_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));
  PetscCheck(bodyID < Nb,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Body %" PetscInt_FMT " is not in [0, %d)", bodyID, Nb);
  body = bodies[bodyID];

  if (edgeID >= 0)      {PetscCall(EG_objectBodyTopo(body, EDGE, edgeID, &obj)); Np = 1;}
  else if (faceID >= 0) {PetscCall(EG_objectBodyTopo(body, FACE, faceID, &obj)); Np = 2;}
  else {
    for (d = 0; d < dE; ++d) gcoords[d] = mcoords[d];
    PetscFunctionReturn(0);
  }

  /* Calculate parameters (t or u,v) for vertices */
  PetscCall(DMPlexVecGetClosure(cdm, NULL, coordinatesLocal, p, &Nv, &coords));
  Nv  /= dE;
  if (Nv == 1) {
    PetscCall(DMPlexVecRestoreClosure(cdm, NULL, coordinatesLocal, p, &Nv, &coords));
    for (d = 0; d < dE; ++d) gcoords[d] = mcoords[d];
    PetscFunctionReturn(0);
  }
  PetscCheck(Nv <= 16,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle %" PetscInt_FMT " coordinates associated to point %" PetscInt_FMT, Nv, p);

  /* Correct EGADSlite 2pi bug when calculating nearest point on Periodic Surfaces */
  PetscCall(EG_getRange(obj, range, &peri));
  for (v = 0; v < Nv; ++v) {
    PetscCall(EG_invEvaluate(obj, &coords[v*dE], &paramsV[v*3], &resultV[v*3]));
#if 1
    if (peri > 0) {
      if      (paramsV[v*3+0] + 1.e-4 < range[0]) {paramsV[v*3+0] += 2. * PETSC_PI;}
      else if (paramsV[v*3+0] - 1.e-4 > range[1]) {paramsV[v*3+0] -= 2. * PETSC_PI;}
    }
    if (peri > 1) {
      if      (paramsV[v*3+1] + 1.e-4 < range[2]) {paramsV[v*3+1] += 2. * PETSC_PI;}
      else if (paramsV[v*3+1] - 1.e-4 > range[3]) {paramsV[v*3+1] -= 2. * PETSC_PI;}
    }
#endif
  }
  PetscCall(DMPlexVecRestoreClosure(cdm, NULL, coordinatesLocal, p, &Nv, &coords));
  /* Calculate parameters (t or u,v) for new vertex at edge midpoint */
  for (pm = 0; pm < Np; ++pm) {
    params[pm] = 0.;
    for (v = 0; v < Nv; ++v) params[pm] += paramsV[v*3+pm];
    params[pm] /= Nv;
  }
  PetscCheck(!(params[0] < range[0]) && !(params[0] > range[1]),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point %" PetscInt_FMT " had bad interpolation", p);
  PetscCheck(Np <= 1 || (params[1] >= range[2] && params[1] <= range[3]),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point %" PetscInt_FMT " had bad interpolation", p);
  /* Put coordinates for new vertex in result[] */
  PetscCall(EG_evaluate(obj, params, result));
  for (d = 0; d < dE; ++d) gcoords[d] = result[d];
  PetscFunctionReturn(0);
}
#endif

/*@
  DMPlexSnapToGeomModel - Given a coordinate point 'mcoords' on the mesh point 'p', return the closest coordinate point 'gcoords' on the geometry model associated with that point.

  Not collective

  Input Parameters:
+ dm      - The DMPlex object
. p       - The mesh point
. dE      - The coordinate dimension
- mcoords - A coordinate point lying on the mesh point

  Output Parameter:
. gcoords - The closest coordinate point on the geometry model associated with 'p' to the given point

  Note: Returns the original coordinates if no geometry model is found. Right now the only supported geometry model is EGADS. The coordinate dimension may be different from the coordinate dimension of the dm, for example if the transformation is extrusion.

  Level: intermediate

.seealso: `DMRefine()`, `DMPlexCreate()`, `DMPlexSetRefinementUniform()`
@*/
PetscErrorCode DMPlexSnapToGeomModel(DM dm, PetscInt p, PetscInt dE, const PetscScalar mcoords[], PetscScalar gcoords[])
{
  PetscInt d;

  PetscFunctionBeginHot;
#ifdef PETSC_HAVE_EGADS
  {
    DM_Plex       *plex = (DM_Plex *) dm->data;
    DMLabel        bodyLabel, faceLabel, edgeLabel;
    PetscInt       bodyID, faceID, edgeID;
    PetscContainer modelObj;
    ego            model;
    PetscBool      islite = PETSC_FALSE;

    PetscCall(DMGetLabel(dm, "EGADS Body ID", &bodyLabel));
    PetscCall(DMGetLabel(dm, "EGADS Face ID", &faceLabel));
    PetscCall(DMGetLabel(dm, "EGADS Edge ID", &edgeLabel));
    if (!bodyLabel || !faceLabel || !edgeLabel || plex->ignoreModel) {
      for (d = 0; d < dE; ++d) gcoords[d] = mcoords[d];
      PetscFunctionReturn(0);
    }
    PetscCall(PetscObjectQuery((PetscObject) dm, "EGADS Model", (PetscObject *) &modelObj));
    if (!modelObj) {
      PetscCall(PetscObjectQuery((PetscObject) dm, "EGADSLite Model", (PetscObject *) &modelObj));
      islite = PETSC_TRUE;
    }
    if (!modelObj) {
      for (d = 0; d < dE; ++d) gcoords[d] = mcoords[d];
      PetscFunctionReturn(0);
    }
    PetscCall(PetscContainerGetPointer(modelObj, (void **) &model));
    PetscCall(DMLabelGetValue(bodyLabel, p, &bodyID));
    PetscCall(DMLabelGetValue(faceLabel, p, &faceID));
    PetscCall(DMLabelGetValue(edgeLabel, p, &edgeID));
    /* Allows for "Connective" Plex Edges present in models with multiple non-touching Entities */
    if (bodyID < 0) {
      for (d = 0; d < dE; ++d) gcoords[d] = mcoords[d];
      PetscFunctionReturn(0);
    }
    if (islite) PetscCall(DMPlexSnapToGeomModel_EGADSLite_Internal(dm, p, model, bodyID, faceID, edgeID, mcoords, gcoords));
    else        PetscCall(DMPlexSnapToGeomModel_EGADS_Internal(dm, p, model, bodyID, faceID, edgeID, mcoords, gcoords));
  }
#else
  for (d = 0; d < dE; ++d) gcoords[d] = mcoords[d];
#endif
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_EGADS)
static PetscErrorCode DMPlexEGADSPrintModel_Internal(ego model)
{
  ego geom, *bodies, *objs, *nobjs, *mobjs, *lobjs;
  int oclass, mtype, *senses;
  int Nb, b;

  PetscFunctionBeginUser;
  /* test bodyTopo functions */
  PetscCall(EG_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, " Number of BODIES (nbodies): %d \n", Nb));

  for (b = 0; b < Nb; ++b) {
    ego body = bodies[b];
    int id, Nsh, Nf, Nl, l, Ne, e, Nv, v;

    /* Output Basic Model Topology */
    PetscCall(EG_getBodyTopos(body, NULL, SHELL, &Nsh, &objs));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "   Number of SHELLS: %d \n", Nsh));
    EG_free(objs);

    PetscCall(EG_getBodyTopos(body, NULL, FACE,  &Nf, &objs));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "   Number of FACES: %d \n", Nf));
    EG_free(objs);

    PetscCall(EG_getBodyTopos(body, NULL, LOOP,  &Nl, &lobjs));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "   Number of LOOPS: %d \n", Nl));

    PetscCall(EG_getBodyTopos(body, NULL, EDGE,  &Ne, &objs));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "   Number of EDGES: %d \n", Ne));
    EG_free(objs);

    PetscCall(EG_getBodyTopos(body, NULL, NODE,  &Nv, &objs));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "   Number of NODES: %d \n", Nv));
    EG_free(objs);

    for (l = 0; l < Nl; ++l) {
      ego loop = lobjs[l];

      id   = EG_indexBodyTopo(body, loop);
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "          LOOP ID: %d\n", id));

      /* Get EDGE info which associated with the current LOOP */
      PetscCall(EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses));

      for (e = 0; e < Ne; ++e) {
        ego    edge      = objs[e];
        double range[4]  = {0., 0., 0., 0.};
        double point[3]  = {0., 0., 0.};
        double params[3] = {0., 0., 0.};
        double result[18];
        int    peri;

        id   = EG_indexBodyTopo(body, edge);
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "            EDGE ID: %d (%d)\n", id, e));

        PetscCall(EG_getRange(edge, range, &peri));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Range = %lf, %lf, %lf, %lf \n", range[0], range[1], range[2], range[3]));

        /* Get NODE info which associated with the current EDGE */
        PetscCall(EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
        if (mtype == DEGENERATE) {
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "  EDGE %d is DEGENERATE \n", id));
        } else {
          params[0] = range[0];
          PetscCall(EG_evaluate(edge, params, result));
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "   between (%lf, %lf, %lf)", result[0], result[1], result[2]));
          params[0] = range[1];
          PetscCall(EG_evaluate(edge, params, result));
          PetscCall(PetscPrintf(PETSC_COMM_SELF, " and (%lf, %lf, %lf)\n", result[0], result[1], result[2]));
        }

        for (v = 0; v < Nv; ++v) {
          ego    vertex = nobjs[v];
          double limits[4];
          int    dummy;

          PetscCall(EG_getTopology(vertex, &geom, &oclass, &mtype, limits, &dummy, &mobjs, &senses));
          id   = EG_indexBodyTopo(body, vertex);
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "              NODE ID: %d \n", id));
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "                 (x, y, z) = (%lf, %lf, %lf) \n", limits[0], limits[1], limits[2]));

          point[0] = point[0] + limits[0];
          point[1] = point[1] + limits[1];
          point[2] = point[2] + limits[2];
        }
      }
    }
    EG_free(lobjs);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexEGADSDestroy_Private(void *context)
{
  if (context) EG_close((ego) context);
  return 0;
}

static PetscErrorCode DMPlexCreateEGADS_Internal(MPI_Comm comm, ego context, ego model, DM *newdm)
{
  DMLabel        bodyLabel, faceLabel, edgeLabel, vertexLabel;
  PetscInt       cStart, cEnd, c;
  /* EGADSLite variables */
  ego            geom, *bodies, *objs, *nobjs, *mobjs, *lobjs;
  int            oclass, mtype, nbodies, *senses;
  int            b;
  /* PETSc variables */
  DM             dm;
  PetscHMapI     edgeMap = NULL;
  PetscInt       dim = -1, cdim = -1, numCorners = 0, maxCorners = 0, numVertices = 0, newVertices = 0, numEdges = 0, numCells = 0, newCells = 0, numQuads = 0, cOff = 0, fOff = 0;
  PetscInt      *cells  = NULL, *cone = NULL;
  PetscReal     *coords = NULL;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    const PetscInt debug = 0;

    /* ---------------------------------------------------------------------------------------------------
    Generate Petsc Plex
      Get all Nodes in model, record coordinates in a correctly formatted array
      Cycle through bodies, cycle through loops, recorde NODE IDs in a correctly formatted array
      We need to uniformly refine the initial geometry to guarantee a valid mesh
    */

    /* Calculate cell and vertex sizes */
    PetscCall(EG_getTopology(model, &geom, &oclass, &mtype, NULL, &nbodies, &bodies, &senses));
    PetscCall(PetscHMapICreate(&edgeMap));
    numEdges = 0;
    for (b = 0; b < nbodies; ++b) {
      ego body = bodies[b];
      int id, Nl, l, Nv, v;

      PetscCall(EG_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs));
      for (l = 0; l < Nl; ++l) {
        ego loop = lobjs[l];
        int Ner  = 0, Ne, e, Nc;

        PetscCall(EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses));
        for (e = 0; e < Ne; ++e) {
          ego edge = objs[e];
          int Nv, id;
          PetscHashIter iter;
          PetscBool     found;

          PetscCall(EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
          if (mtype == DEGENERATE) continue;
          id = EG_indexBodyTopo(body, edge);
          PetscCall(PetscHMapIFind(edgeMap, id-1, &iter, &found));
          if (!found) PetscCall(PetscHMapISet(edgeMap, id-1, numEdges++));
          ++Ner;
        }
        if (Ner == 2)      {Nc = 2;}
        else if (Ner == 3) {Nc = 4;}
        else if (Ner == 4) {Nc = 8; ++numQuads;}
        else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot support loop with %d edges", Ner);
        numCells += Nc;
        newCells += Nc-1;
        maxCorners = PetscMax(Ner*2+1, maxCorners);
      }
      PetscCall(EG_getBodyTopos(body, NULL, NODE, &Nv, &nobjs));
      for (v = 0; v < Nv; ++v) {
        ego vertex = nobjs[v];

        id = EG_indexBodyTopo(body, vertex);
        /* TODO: Instead of assuming contiguous ids, we could use a hash table */
        numVertices = PetscMax(id, numVertices);
      }
      EG_free(lobjs);
      EG_free(nobjs);
    }
    PetscCall(PetscHMapIGetSize(edgeMap, &numEdges));
    newVertices  = numEdges + numQuads;
    numVertices += newVertices;

    dim        = 2; /* Assume 3D Models :: Need to update to handle 2D Models in the future */
    cdim       = 3; /* Assume 3D Models :: Need to update to handle 2D Models in the future */
    numCorners = 3; /* Split cells into triangles */
    PetscCall(PetscMalloc3(numVertices*cdim, &coords, numCells*numCorners, &cells, maxCorners, &cone));

    /* Get vertex coordinates */
    for (b = 0; b < nbodies; ++b) {
      ego body = bodies[b];
      int id, Nv, v;

      PetscCall(EG_getBodyTopos(body, NULL, NODE, &Nv, &nobjs));
      for (v = 0; v < Nv; ++v) {
        ego    vertex = nobjs[v];
        double limits[4];
        int    dummy;

        PetscCall(EG_getTopology(vertex, &geom, &oclass, &mtype, limits, &dummy, &mobjs, &senses));
        id   = EG_indexBodyTopo(body, vertex);
        coords[(id-1)*cdim+0] = limits[0];
        coords[(id-1)*cdim+1] = limits[1];
        coords[(id-1)*cdim+2] = limits[2];
      }
      EG_free(nobjs);
    }
    PetscCall(PetscHMapIClear(edgeMap));
    fOff     = numVertices - newVertices + numEdges;
    numEdges = 0;
    numQuads = 0;
    for (b = 0; b < nbodies; ++b) {
      ego body = bodies[b];
      int Nl, l;

      PetscCall(EG_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs));
      for (l = 0; l < Nl; ++l) {
        ego loop = lobjs[l];
        int lid, Ner = 0, Ne, e;

        lid  = EG_indexBodyTopo(body, loop);
        PetscCall(EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses));
        for (e = 0; e < Ne; ++e) {
          ego       edge = objs[e];
          int       eid, Nv;
          PetscHashIter iter;
          PetscBool     found;

          PetscCall(EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
          if (mtype == DEGENERATE) continue;
          ++Ner;
          eid  = EG_indexBodyTopo(body, edge);
          PetscCall(PetscHMapIFind(edgeMap, eid-1, &iter, &found));
          if (!found) {
            PetscInt v = numVertices - newVertices + numEdges;
            double range[4], params[3] = {0., 0., 0.}, result[18];
            int    periodic[2];

            PetscCall(PetscHMapISet(edgeMap, eid-1, numEdges++));
            PetscCall(EG_getRange(edge, range, periodic));
            params[0] = 0.5*(range[0] + range[1]);
            PetscCall(EG_evaluate(edge, params, result));
            coords[v*cdim+0] = result[0];
            coords[v*cdim+1] = result[1];
            coords[v*cdim+2] = result[2];
          }
        }
        if (Ner == 4) {
          PetscInt v = fOff + numQuads++;
          ego     *fobjs, face;
          double   range[4], params[3] = {0., 0., 0.}, result[18];
          int      Nf, fid, periodic[2];

          PetscCall(EG_getBodyTopos(body, loop, FACE, &Nf, &fobjs));
          face = fobjs[0];
          fid  = EG_indexBodyTopo(body, face);
          PetscCheck(Nf == 1,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Loop %d has %d faces, instead of 1 (%d)", lid-1, Nf, fid);
          PetscCall(EG_getRange(face, range, periodic));
          params[0] = 0.5*(range[0] + range[1]);
          params[1] = 0.5*(range[2] + range[3]);
          PetscCall(EG_evaluate(face, params, result));
          coords[v*cdim+0] = result[0];
          coords[v*cdim+1] = result[1];
          coords[v*cdim+2] = result[2];
        }
      }
    }
    PetscCheck(numEdges + numQuads == newVertices,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of new vertices %" PetscInt_FMT " != %" PetscInt_FMT " previous count", numEdges + numQuads, newVertices);

    /* Get cell vertices by traversing loops */
    numQuads = 0;
    cOff     = 0;
    for (b = 0; b < nbodies; ++b) {
      ego body = bodies[b];
      int id, Nl, l;

      PetscCall(EG_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs));
      for (l = 0; l < Nl; ++l) {
        ego loop = lobjs[l];
        int lid, Ner = 0, Ne, e, nc = 0, c, Nt, t;

        lid  = EG_indexBodyTopo(body, loop);
        PetscCall(EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses));

        for (e = 0; e < Ne; ++e) {
          ego edge = objs[e];
          int points[3];
          int eid, Nv, v, tmp;

          eid = EG_indexBodyTopo(body, edge);
          PetscCall(EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
          if (mtype == DEGENERATE) continue;
          else                     ++Ner;
          PetscCheck(Nv == 2,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Edge %d has %d vertices != 2", eid, Nv);

          for (v = 0; v < Nv; ++v) {
            ego vertex = nobjs[v];

            id = EG_indexBodyTopo(body, vertex);
            points[v*2] = id-1;
          }
          {
            PetscInt edgeNum;

            PetscCall(PetscHMapIGet(edgeMap, eid-1, &edgeNum));
            points[1] = numVertices - newVertices + edgeNum;
          }
          /* EGADS loops are not oriented, but seem to be in order, so we must piece them together */
          if (!nc) {
            for (v = 0; v < Nv+1; ++v) cone[nc++] = points[v];
          } else {
            if (cone[nc-1] == points[0])      {cone[nc++] = points[1]; if (cone[0] != points[2]) cone[nc++] = points[2];}
            else if (cone[nc-1] == points[2]) {cone[nc++] = points[1]; if (cone[0] != points[0]) cone[nc++] = points[0];}
            else if (cone[nc-3] == points[0]) {tmp = cone[nc-3]; cone[nc-3] = cone[nc-1]; cone[nc-1] = tmp; cone[nc++] = points[1]; if (cone[0] != points[2]) cone[nc++] = points[2];}
            else if (cone[nc-3] == points[2]) {tmp = cone[nc-3]; cone[nc-3] = cone[nc-1]; cone[nc-1] = tmp; cone[nc++] = points[1]; if (cone[0] != points[0]) cone[nc++] = points[0];}
            else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Edge %d does not match its predecessor", eid);
          }
        }
        PetscCheck(nc == 2*Ner,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of corners %" PetscInt_FMT " != %" PetscInt_FMT, nc, 2*Ner);
        if (Ner == 4) {cone[nc++] = numVertices - newVertices + numEdges + numQuads++;}
        PetscCheck(nc <= maxCorners,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of corners %" PetscInt_FMT " > %" PetscInt_FMT " max", nc, maxCorners);
        /* Triangulate the loop */
        switch (Ner) {
          case 2: /* Bi-Segment -> 2 triangles */
            Nt = 2;
            cells[cOff*numCorners+0] = cone[0];
            cells[cOff*numCorners+1] = cone[1];
            cells[cOff*numCorners+2] = cone[2];
            ++cOff;
            cells[cOff*numCorners+0] = cone[0];
            cells[cOff*numCorners+1] = cone[2];
            cells[cOff*numCorners+2] = cone[3];
            ++cOff;
            break;
          case 3: /* Triangle   -> 4 triangles */
            Nt = 4;
            cells[cOff*numCorners+0] = cone[0];
            cells[cOff*numCorners+1] = cone[1];
            cells[cOff*numCorners+2] = cone[5];
            ++cOff;
            cells[cOff*numCorners+0] = cone[1];
            cells[cOff*numCorners+1] = cone[2];
            cells[cOff*numCorners+2] = cone[3];
            ++cOff;
            cells[cOff*numCorners+0] = cone[5];
            cells[cOff*numCorners+1] = cone[3];
            cells[cOff*numCorners+2] = cone[4];
            ++cOff;
            cells[cOff*numCorners+0] = cone[1];
            cells[cOff*numCorners+1] = cone[3];
            cells[cOff*numCorners+2] = cone[5];
            ++cOff;
            break;
          case 4: /* Quad       -> 8 triangles */
            Nt = 8;
            cells[cOff*numCorners+0] = cone[0];
            cells[cOff*numCorners+1] = cone[1];
            cells[cOff*numCorners+2] = cone[7];
            ++cOff;
            cells[cOff*numCorners+0] = cone[1];
            cells[cOff*numCorners+1] = cone[2];
            cells[cOff*numCorners+2] = cone[3];
            ++cOff;
            cells[cOff*numCorners+0] = cone[3];
            cells[cOff*numCorners+1] = cone[4];
            cells[cOff*numCorners+2] = cone[5];
            ++cOff;
            cells[cOff*numCorners+0] = cone[5];
            cells[cOff*numCorners+1] = cone[6];
            cells[cOff*numCorners+2] = cone[7];
            ++cOff;
            cells[cOff*numCorners+0] = cone[8];
            cells[cOff*numCorners+1] = cone[1];
            cells[cOff*numCorners+2] = cone[3];
            ++cOff;
            cells[cOff*numCorners+0] = cone[8];
            cells[cOff*numCorners+1] = cone[3];
            cells[cOff*numCorners+2] = cone[5];
            ++cOff;
            cells[cOff*numCorners+0] = cone[8];
            cells[cOff*numCorners+1] = cone[5];
            cells[cOff*numCorners+2] = cone[7];
            ++cOff;
            cells[cOff*numCorners+0] = cone[8];
            cells[cOff*numCorners+1] = cone[7];
            cells[cOff*numCorners+2] = cone[1];
            ++cOff;
            break;
          default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Loop %d has %d edges, which we do not support", lid, Ner);
        }
        if (debug) {
          for (t = 0; t < Nt; ++t) {
            PetscCall(PetscPrintf(PETSC_COMM_SELF, "  LOOP Corner NODEs Triangle %" PetscInt_FMT " (", t));
            for (c = 0; c < numCorners; ++c) {
              if (c > 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, ", "));
              PetscCall(PetscPrintf(PETSC_COMM_SELF, "%" PetscInt_FMT, cells[(cOff-Nt+t)*numCorners+c]));
            }
            PetscCall(PetscPrintf(PETSC_COMM_SELF, ")\n"));
          }
        }
      }
      EG_free(lobjs);
    }
  }
  PetscCheck(cOff == numCells,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Count of total cells %" PetscInt_FMT " != %" PetscInt_FMT " previous count", cOff, numCells);
  PetscCall(DMPlexCreateFromCellListPetsc(PETSC_COMM_WORLD, dim, numCells, numVertices, numCorners, PETSC_TRUE, cells, cdim, coords, &dm));
  PetscCall(PetscFree3(coords, cells, cone));
  PetscCall(PetscInfo(dm, " Total Number of Unique Cells    = %" PetscInt_FMT " (%" PetscInt_FMT ")\n", numCells, newCells));
  PetscCall(PetscInfo(dm, " Total Number of Unique Vertices = %" PetscInt_FMT " (%" PetscInt_FMT ")\n", numVertices, newVertices));
  /* Embed EGADS model in DM */
  {
    PetscContainer modelObj, contextObj;

    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &modelObj));
    PetscCall(PetscContainerSetPointer(modelObj, model));
    PetscCall(PetscObjectCompose((PetscObject) dm, "EGADS Model", (PetscObject) modelObj));
    PetscCall(PetscContainerDestroy(&modelObj));

    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &contextObj));
    PetscCall(PetscContainerSetPointer(contextObj, context));
    PetscCall(PetscContainerSetUserDestroy(contextObj, DMPlexEGADSDestroy_Private));
    PetscCall(PetscObjectCompose((PetscObject) dm, "EGADS Context", (PetscObject) contextObj));
    PetscCall(PetscContainerDestroy(&contextObj));
  }
  /* Label points */
  PetscCall(DMCreateLabel(dm, "EGADS Body ID"));
  PetscCall(DMGetLabel(dm, "EGADS Body ID", &bodyLabel));
  PetscCall(DMCreateLabel(dm, "EGADS Face ID"));
  PetscCall(DMGetLabel(dm, "EGADS Face ID", &faceLabel));
  PetscCall(DMCreateLabel(dm, "EGADS Edge ID"));
  PetscCall(DMGetLabel(dm, "EGADS Edge ID", &edgeLabel));
  PetscCall(DMCreateLabel(dm, "EGADS Vertex ID"));
  PetscCall(DMGetLabel(dm, "EGADS Vertex ID", &vertexLabel));
  cOff = 0;
  for (b = 0; b < nbodies; ++b) {
    ego body = bodies[b];
    int id, Nl, l;

    PetscCall(EG_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs));
    for (l = 0; l < Nl; ++l) {
      ego  loop = lobjs[l];
      ego *fobjs;
      int  lid, Nf, fid, Ner = 0, Ne, e, Nt = 0, t;

      lid  = EG_indexBodyTopo(body, loop);
      PetscCall(EG_getBodyTopos(body, loop, FACE, &Nf, &fobjs));
      PetscCheck(Nf <= 1,PETSC_COMM_SELF, PETSC_ERR_SUP, "Loop %d has %d > 1 faces, which is not supported", lid, Nf);
      fid  = EG_indexBodyTopo(body, fobjs[0]);
      EG_free(fobjs);
      PetscCall(EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses));
      for (e = 0; e < Ne; ++e) {
        ego             edge = objs[e];
        int             eid, Nv, v;
        PetscInt        points[3], support[2], numEdges, edgeNum;
        const PetscInt *edges;

        eid = EG_indexBodyTopo(body, edge);
        PetscCall(EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
        if (mtype == DEGENERATE) continue;
        else                     ++Ner;
        for (v = 0; v < Nv; ++v) {
          ego vertex = nobjs[v];

          id   = EG_indexBodyTopo(body, vertex);
          PetscCall(DMLabelSetValue(edgeLabel, numCells + id-1, eid));
          points[v*2] = numCells + id-1;
        }
        PetscCall(PetscHMapIGet(edgeMap, eid-1, &edgeNum));
        points[1] = numCells + numVertices - newVertices + edgeNum;

        PetscCall(DMLabelSetValue(edgeLabel, points[1], eid));
        support[0] = points[0];
        support[1] = points[1];
        PetscCall(DMPlexGetJoin(dm, 2, support, &numEdges, &edges));
        PetscCheck(numEdges == 1,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Vertices (%" PetscInt_FMT ", %" PetscInt_FMT ") should only bound 1 edge, not %" PetscInt_FMT, support[0], support[1], numEdges);
        PetscCall(DMLabelSetValue(edgeLabel, edges[0], eid));
        PetscCall(DMPlexRestoreJoin(dm, 2, support, &numEdges, &edges));
        support[0] = points[1];
        support[1] = points[2];
        PetscCall(DMPlexGetJoin(dm, 2, support, &numEdges, &edges));
        PetscCheck(numEdges == 1,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Vertices (%" PetscInt_FMT ", %" PetscInt_FMT ") should only bound 1 edge, not %" PetscInt_FMT, support[0], support[1], numEdges);
        PetscCall(DMLabelSetValue(edgeLabel, edges[0], eid));
        PetscCall(DMPlexRestoreJoin(dm, 2, support, &numEdges, &edges));
      }
      switch (Ner) {
        case 2: Nt = 2;break;
        case 3: Nt = 4;break;
        case 4: Nt = 8;break;
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Loop with %d edges is unsupported", Ner);
      }
      for (t = 0; t < Nt; ++t) {
        PetscCall(DMLabelSetValue(bodyLabel, cOff+t, b));
        PetscCall(DMLabelSetValue(faceLabel, cOff+t, fid));
      }
      cOff += Nt;
    }
    EG_free(lobjs);
  }
  PetscCall(PetscHMapIDestroy(&edgeMap));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL;
    PetscInt  clSize, cl, bval, fval;

    PetscCall(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &clSize, &closure));
    PetscCall(DMLabelGetValue(bodyLabel, c, &bval));
    PetscCall(DMLabelGetValue(faceLabel, c, &fval));
    for (cl = 0; cl < clSize*2; cl += 2) {
      PetscCall(DMLabelSetValue(bodyLabel, closure[cl], bval));
      PetscCall(DMLabelSetValue(faceLabel, closure[cl], fval));
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &clSize, &closure));
  }
  *newdm = dm;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateEGADS(MPI_Comm comm, ego context, ego model, DM *newdm)
{
  DMLabel         bodyLabel, faceLabel, edgeLabel, vertexLabel;
  // EGADS/EGADSLite variables
  ego             geom, *bodies, *mobjs, *fobjs, *lobjs, *eobjs, *nobjs;
  ego             topRef, prev, next;
  int             oclass, mtype, nbodies, *senses, *lSenses, *eSenses;
  int             b;
  // PETSc variables
  DM              dm;
  PetscHMapI      edgeMap = NULL, bodyIndexMap = NULL, bodyVertexMap = NULL, bodyEdgeMap = NULL, bodyFaceMap = NULL, bodyEdgeGlobalMap = NULL;
  PetscInt        dim = -1, cdim = -1, numCorners = 0, numVertices = 0, numEdges = 0, numFaces = 0, numCells = 0, edgeCntr = 0;
  PetscInt        cellCntr = 0, numPoints = 0;
  PetscInt        *cells  = NULL;
  const PetscInt  *cone = NULL;
  PetscReal       *coords = NULL;
  PetscMPIInt      rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    // ---------------------------------------------------------------------------------------------------
    // Generate Petsc Plex
    //  Get all Nodes in model, record coordinates in a correctly formatted array
    //  Cycle through bodies, cycle through loops, recorde NODE IDs in a correctly formatted array
    //  We need to uniformly refine the initial geometry to guarantee a valid mesh

  // Caluculate cell and vertex sizes
  PetscCall(EG_getTopology(model, &geom, &oclass, &mtype, NULL, &nbodies, &bodies, &senses));

    PetscCall(PetscHMapICreate(&edgeMap));
  PetscCall(PetscHMapICreate(&bodyIndexMap));
  PetscCall(PetscHMapICreate(&bodyVertexMap));
  PetscCall(PetscHMapICreate(&bodyEdgeMap));
  PetscCall(PetscHMapICreate(&bodyEdgeGlobalMap));
  PetscCall(PetscHMapICreate(&bodyFaceMap));

  for (b = 0; b < nbodies; ++b) {
      ego             body = bodies[b];
    int             Nf, Ne, Nv;
    PetscHashIter   BIiter, BViter, BEiter, BEGiter, BFiter, EMiter;
    PetscBool       BIfound, BVfound, BEfound, BEGfound, BFfound, EMfound;

    PetscCall(PetscHMapIFind(bodyIndexMap, b, &BIiter, &BIfound));
    PetscCall(PetscHMapIFind(bodyVertexMap, b, &BViter, &BVfound));
    PetscCall(PetscHMapIFind(bodyEdgeMap, b, &BEiter, &BEfound));
    PetscCall(PetscHMapIFind(bodyEdgeGlobalMap, b, &BEGiter, &BEGfound));
    PetscCall(PetscHMapIFind(bodyFaceMap, b, &BFiter, &BFfound));

    if (!BIfound)  PetscCall(PetscHMapISet(bodyIndexMap, b, numFaces + numEdges + numVertices));
    if (!BVfound)  PetscCall(PetscHMapISet(bodyVertexMap, b, numVertices));
    if (!BEfound)  PetscCall(PetscHMapISet(bodyEdgeMap, b, numEdges));
    if (!BEGfound) PetscCall(PetscHMapISet(bodyEdgeGlobalMap, b, edgeCntr));
    if (!BFfound)  PetscCall(PetscHMapISet(bodyFaceMap, b, numFaces));

    PetscCall(EG_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
    PetscCall(EG_getBodyTopos(body, NULL, EDGE, &Ne, &eobjs));
    PetscCall(EG_getBodyTopos(body, NULL, NODE, &Nv, &nobjs));
    EG_free(fobjs);
    EG_free(eobjs);
    EG_free(nobjs);

    // Remove DEGENERATE EDGES from Edge count
    PetscCall(EG_getBodyTopos(body, NULL, EDGE, &Ne, &eobjs));
    int Netemp = 0;
    for (int e = 0; e < Ne; ++e) {
      ego     edge = eobjs[e];
      int     eid;

      PetscCall(EG_getInfo(edge, &oclass, &mtype, &topRef, &prev, &next));
      eid = EG_indexBodyTopo(body, edge);

      PetscCall(PetscHMapIFind(edgeMap, edgeCntr + eid - 1, &EMiter, &EMfound));
      if (mtype == DEGENERATE) {
        if (!EMfound) PetscCall(PetscHMapISet(edgeMap, edgeCntr + eid - 1, -1));
      }
      else {
      ++Netemp;
        if (!EMfound) PetscCall(PetscHMapISet(edgeMap, edgeCntr + eid - 1, Netemp));
      }
    }
    EG_free(eobjs);

    // Determine Number of Cells
    PetscCall(EG_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
    for (int f = 0; f < Nf; ++f) {
        ego     face = fobjs[f];
    int     edgeTemp = 0;

      PetscCall(EG_getBodyTopos(body, face, EDGE, &Ne, &eobjs));
      for (int e = 0; e < Ne; ++e) {
        ego     edge = eobjs[e];

        PetscCall(EG_getInfo(edge, &oclass, &mtype, &topRef, &prev, &next));
        if (mtype != DEGENERATE) {++edgeTemp;}
      }
      numCells += (2 * edgeTemp);
      EG_free(eobjs);
    }
    EG_free(fobjs);

    numFaces    += Nf;
    numEdges    += Netemp;
    numVertices += Nv;
    edgeCntr    += Ne;
  }

  // Set up basic DMPlex parameters
  dim        = 2;    // Assumes 3D Models :: Need to handle 2D modles in the future
  cdim       = 3;     // Assumes 3D Models :: Need to update to handle 2D modles in future
  numCorners = 3;     // Split Faces into triangles
    numPoints  = numVertices + numEdges + numFaces;   // total number of coordinate points

  PetscCall(PetscMalloc2(numPoints*cdim, &coords, numCells*numCorners, &cells));

  // Get Vertex Coordinates and Set up Cells
  for (b = 0; b < nbodies; ++b) {
    ego             body = bodies[b];
    int             Nf, Ne, Nv;
    PetscInt        bodyVertexIndexStart, bodyEdgeIndexStart, bodyEdgeGlobalIndexStart, bodyFaceIndexStart;
    PetscHashIter   BViter, BEiter, BEGiter, BFiter, EMiter;
    PetscBool       BVfound, BEfound, BEGfound, BFfound, EMfound;

    // Vertices on Current Body
    PetscCall(EG_getBodyTopos(body, NULL, NODE, &Nv, &nobjs));

    PetscCall(PetscHMapIFind(bodyVertexMap, b, &BViter, &BVfound));
    PetscCheck(BVfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %d not found in bodyVertexMap", b);
    PetscCall(PetscHMapIGet(bodyVertexMap, b, &bodyVertexIndexStart));

    for (int v = 0; v < Nv; ++v) {
      ego    vertex = nobjs[v];
    double limits[4];
    int    id, dummy;

    PetscCall(EG_getTopology(vertex, &geom, &oclass, &mtype, limits, &dummy, &mobjs, &senses));
    id = EG_indexBodyTopo(body, vertex);

    coords[(bodyVertexIndexStart + id - 1)*cdim + 0] = limits[0];
    coords[(bodyVertexIndexStart + id - 1)*cdim + 1] = limits[1];
    coords[(bodyVertexIndexStart + id - 1)*cdim + 2] = limits[2];
    }
    EG_free(nobjs);

    // Edge Midpoint Vertices on Current Body
    PetscCall(EG_getBodyTopos(body, NULL, EDGE, &Ne, &eobjs));

    PetscCall(PetscHMapIFind(bodyEdgeMap, b, &BEiter, &BEfound));
    PetscCheck(BEfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %d not found in bodyEdgeMap", b);
    PetscCall(PetscHMapIGet(bodyEdgeMap, b, &bodyEdgeIndexStart));

    PetscCall(PetscHMapIFind(bodyEdgeGlobalMap, b, &BEGiter, &BEGfound));
    PetscCheck(BEGfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %d not found in bodyEdgeGlobalMap", b);
    PetscCall(PetscHMapIGet(bodyEdgeGlobalMap, b, &bodyEdgeGlobalIndexStart));

    for (int e = 0; e < Ne; ++e) {
      ego          edge = eobjs[e];
    double       range[2], avgt[1], cntrPnt[9];
    int          eid, eOffset;
    int          periodic;

    PetscCall(EG_getInfo(edge, &oclass, &mtype, &topRef, &prev, &next));
    if (mtype == DEGENERATE) {continue;}

    eid = EG_indexBodyTopo(body, edge);

    // get relative offset from globalEdgeID Vector
    PetscCall(PetscHMapIFind(edgeMap, bodyEdgeGlobalIndexStart + eid - 1, &EMiter, &EMfound));
      PetscCheck(EMfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "Edge %d not found in edgeMap", bodyEdgeGlobalIndexStart + eid - 1);
      PetscCall(PetscHMapIGet(edgeMap, bodyEdgeGlobalIndexStart + eid - 1, &eOffset));

    PetscCall(EG_getRange(edge, range, &periodic));
    avgt[0] = (range[0] + range[1]) /  2.;

    PetscCall(EG_evaluate(edge, avgt, cntrPnt));
    coords[(numVertices + bodyEdgeIndexStart + eOffset - 1)*cdim + 0] = cntrPnt[0];
        coords[(numVertices + bodyEdgeIndexStart + eOffset - 1)*cdim + 1] = cntrPnt[1];
    coords[(numVertices + bodyEdgeIndexStart + eOffset - 1)*cdim + 2] = cntrPnt[2];
    }
    EG_free(eobjs);

    // Face Midpoint Vertices on Current Body
    PetscCall(EG_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));

    PetscCall(PetscHMapIFind(bodyFaceMap, b, &BFiter, &BFfound));
    PetscCheck(BFfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %d not found in bodyFaceMap", b);
    PetscCall(PetscHMapIGet(bodyFaceMap, b, &bodyFaceIndexStart));

    for (int f = 0; f < Nf; ++f) {
    ego       face = fobjs[f];
    double    range[4], avgUV[2], cntrPnt[18];
    int       peri, id;

    id = EG_indexBodyTopo(body, face);
    PetscCall(EG_getRange(face, range, &peri));

    avgUV[0] = (range[0] + range[1]) / 2.;
    avgUV[1] = (range[2] + range[3]) / 2.;
    PetscCall(EG_evaluate(face, avgUV, cntrPnt));

    coords[(numVertices + numEdges + bodyFaceIndexStart + id - 1)*cdim + 0] = cntrPnt[0];
    coords[(numVertices + numEdges + bodyFaceIndexStart + id - 1)*cdim + 1] = cntrPnt[1];
    coords[(numVertices + numEdges + bodyFaceIndexStart + id - 1)*cdim + 2] = cntrPnt[2];
    }
    EG_free(fobjs);

    // Define Cells :: Note - This could be incorporated in the Face Midpoint Vertices Loop but was kept separate for clarity
    PetscCall(EG_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
    for (int f = 0; f < Nf; ++f) {
    ego      face = fobjs[f];
    int      fID, midFaceID, midPntID, startID, endID, Nl;

    fID = EG_indexBodyTopo(body, face);
    midFaceID = numVertices + numEdges + bodyFaceIndexStart + fID - 1;
    // Must Traverse Loop to ensure we have all necessary information like the sense (+/- 1) of the edges.
    // TODO :: Only handles single loop faces (No holes). The choices for handling multiloop faces are:
    //            1) Use the DMPlexCreateEGADSFromFile() with the -dm_plex_egads_with_tess = 1 option.
    //               This will use a default EGADS tessellation as an initial surface mesh.
    //            2) Create the initial surface mesh via a 2D mesher :: Currently not availble (?future?)
    //               May I suggest the XXXX as a starting point?

    PetscCall(EG_getTopology(face, &geom, &oclass, &mtype, NULL, &Nl, &lobjs, &lSenses));

      PetscCheck(Nl <= 1,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face has %d Loops. Can only handle Faces with 1 Loop. Please use --dm_plex_egads_with_tess = 1 Option", Nl);
    for (int l = 0; l < Nl; ++l) {
          ego      loop = lobjs[l];

          PetscCall(EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &eobjs, &eSenses));
      for (int e = 0; e < Ne; ++e) {
        ego     edge = eobjs[e];
        int     eid, eOffset;

        PetscCall(EG_getInfo(edge, &oclass, &mtype, &topRef, &prev, &next));
      eid = EG_indexBodyTopo(body, edge);
        if (mtype == DEGENERATE) { continue; }

        // get relative offset from globalEdgeID Vector
        PetscCall(PetscHMapIFind(edgeMap, bodyEdgeGlobalIndexStart + eid - 1, &EMiter, &EMfound));
          PetscCheck(EMfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "Edge %d of Body %d not found in edgeMap. Global Edge ID :: %d", eid, b, bodyEdgeGlobalIndexStart + eid - 1);
          PetscCall(PetscHMapIGet(edgeMap, bodyEdgeGlobalIndexStart + eid - 1, &eOffset));

      midPntID = numVertices + bodyEdgeIndexStart + eOffset - 1;

        PetscCall(EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));

        if (eSenses[e] > 0) { startID = EG_indexBodyTopo(body, nobjs[0]); endID = EG_indexBodyTopo(body, nobjs[1]); }
        else { startID = EG_indexBodyTopo(body, nobjs[1]); endID = EG_indexBodyTopo(body, nobjs[0]); }

      // Define 2 Cells per Edge with correct orientation
      cells[cellCntr*numCorners + 0] = midFaceID;
      cells[cellCntr*numCorners + 1] = bodyVertexIndexStart + startID - 1;
      cells[cellCntr*numCorners + 2] = midPntID;

      cells[cellCntr*numCorners + 3] = midFaceID;
      cells[cellCntr*numCorners + 4] = midPntID;
      cells[cellCntr*numCorners + 5] = bodyVertexIndexStart + endID - 1;

      cellCntr = cellCntr + 2;
      }
    }
    }
    EG_free(fobjs);
  }
  }

  // Generate DMPlex
  PetscCall(DMPlexCreateFromCellListPetsc(PETSC_COMM_WORLD, dim, numCells, numPoints, numCorners, PETSC_TRUE, cells, cdim, coords, &dm));
  PetscCall(PetscFree2(coords, cells));
  PetscCall(PetscInfo(dm, " Total Number of Unique Cells    = %" PetscInt_FMT " \n", numCells));
  PetscCall(PetscInfo(dm, " Total Number of Unique Vertices = %" PetscInt_FMT " \n", numVertices));

  // Embed EGADS model in DM
  {
    PetscContainer modelObj, contextObj;

    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &modelObj));
    PetscCall(PetscContainerSetPointer(modelObj, model));
    PetscCall(PetscObjectCompose((PetscObject) dm, "EGADS Model", (PetscObject) modelObj));
    PetscCall(PetscContainerDestroy(&modelObj));

    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &contextObj));
    PetscCall(PetscContainerSetPointer(contextObj, context));
    PetscCall(PetscContainerSetUserDestroy(contextObj, DMPlexEGADSDestroy_Private));
    PetscCall(PetscObjectCompose((PetscObject) dm, "EGADS Context", (PetscObject) contextObj));
    PetscCall(PetscContainerDestroy(&contextObj));
  }
  // Label points
  PetscInt   nStart, nEnd;

  PetscCall(DMCreateLabel(dm, "EGADS Body ID"));
  PetscCall(DMGetLabel(dm, "EGADS Body ID", &bodyLabel));
  PetscCall(DMCreateLabel(dm, "EGADS Face ID"));
  PetscCall(DMGetLabel(dm, "EGADS Face ID", &faceLabel));
  PetscCall(DMCreateLabel(dm, "EGADS Edge ID"));
  PetscCall(DMGetLabel(dm, "EGADS Edge ID", &edgeLabel));
  PetscCall(DMCreateLabel(dm, "EGADS Vertex ID"));
  PetscCall(DMGetLabel(dm, "EGADS Vertex ID", &vertexLabel));

  PetscCall(DMPlexGetHeightStratum(dm, 2, &nStart, &nEnd));

  cellCntr = 0;
  for (b = 0; b < nbodies; ++b) {
    ego             body = bodies[b];
  int             Nv, Ne, Nf;
  PetscInt        bodyVertexIndexStart, bodyEdgeIndexStart, bodyEdgeGlobalIndexStart, bodyFaceIndexStart;
  PetscHashIter   BViter, BEiter, BEGiter, BFiter, EMiter;
  PetscBool       BVfound, BEfound, BEGfound, BFfound, EMfound;

  PetscCall(PetscHMapIFind(bodyVertexMap, b, &BViter, &BVfound));
  PetscCheck(BVfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %d not found in bodyVertexMap", b);
  PetscCall(PetscHMapIGet(bodyVertexMap, b, &bodyVertexIndexStart));

  PetscCall(PetscHMapIFind(bodyEdgeMap, b, &BEiter, &BEfound));
  PetscCheck(BEfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %d not found in bodyEdgeMap", b);
  PetscCall(PetscHMapIGet(bodyEdgeMap, b, &bodyEdgeIndexStart));

    PetscCall(PetscHMapIFind(bodyFaceMap, b, &BFiter, &BFfound));
  PetscCheck(BFfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %d not found in bodyFaceMap", b);
  PetscCall(PetscHMapIGet(bodyFaceMap, b, &bodyFaceIndexStart));

    PetscCall(PetscHMapIFind(bodyEdgeGlobalMap, b, &BEGiter, &BEGfound));
    PetscCheck(BEGfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %d not found in bodyEdgeGlobalMap", b);
    PetscCall(PetscHMapIGet(bodyEdgeGlobalMap, b, &bodyEdgeGlobalIndexStart));

  PetscCall(EG_getBodyTopos(body, NULL, FACE,  &Nf, &fobjs));
  for (int f = 0; f < Nf; ++f) {
    ego   face = fobjs[f];
      int   fID, Nl;

    fID  = EG_indexBodyTopo(body, face);

    PetscCall(EG_getBodyTopos(body, face, LOOP, &Nl, &lobjs));
    for (int l = 0; l < Nl; ++l) {
        ego  loop = lobjs[l];
    int  lid;

    lid  = EG_indexBodyTopo(body, loop);
      PetscCheck(Nl <= 1,PETSC_COMM_SELF, PETSC_ERR_SUP, "Loop %d has %d > 1 faces, which is not supported", lid, Nf);

    PetscCall(EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &eobjs, &eSenses));
    for (int e = 0; e < Ne; ++e) {
      ego     edge = eobjs[e];
      int     eid, eOffset;

      // Skip DEGENERATE Edges
      PetscCall(EG_getInfo(edge, &oclass, &mtype, &topRef, &prev, &next));
      if (mtype == DEGENERATE) {continue;}
      eid = EG_indexBodyTopo(body, edge);

      // get relative offset from globalEdgeID Vector
      PetscCall(PetscHMapIFind(edgeMap, bodyEdgeGlobalIndexStart + eid - 1, &EMiter, &EMfound));
      PetscCheck(EMfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "Edge %d of Body %d not found in edgeMap. Global Edge ID :: %d", eid, b, bodyEdgeGlobalIndexStart + eid - 1);
      PetscCall(PetscHMapIGet(edgeMap, bodyEdgeGlobalIndexStart + eid - 1, &eOffset));

      PetscCall(EG_getBodyTopos(body, edge, NODE, &Nv, &nobjs));
      for (int v = 0; v < Nv; ++v){
        ego vertex = nobjs[v];
        int vID;

        vID = EG_indexBodyTopo(body, vertex);
        PetscCall(DMLabelSetValue(bodyLabel, nStart + bodyVertexIndexStart + vID - 1, b));
        PetscCall(DMLabelSetValue(vertexLabel, nStart + bodyVertexIndexStart + vID - 1, vID));
      }
      EG_free(nobjs);

      PetscCall(DMLabelSetValue(bodyLabel, nStart + numVertices + bodyEdgeIndexStart + eOffset - 1, b));
      PetscCall(DMLabelSetValue(edgeLabel, nStart + numVertices + bodyEdgeIndexStart + eOffset - 1, eid));

      // Define Cell faces
      for (int jj = 0; jj < 2; ++jj){
        PetscCall(DMLabelSetValue(bodyLabel, cellCntr, b));
        PetscCall(DMLabelSetValue(faceLabel, cellCntr, fID));
        PetscCall(DMPlexGetCone(dm, cellCntr, &cone));

        PetscCall(DMLabelSetValue(bodyLabel, cone[0], b));
        PetscCall(DMLabelSetValue(faceLabel, cone[0], fID));

        PetscCall(DMLabelSetValue(bodyLabel, cone[1], b));
        PetscCall(DMLabelSetValue(edgeLabel, cone[1], eid));

       PetscCall(DMLabelSetValue(bodyLabel, cone[2], b));
       PetscCall(DMLabelSetValue(faceLabel, cone[2], fID));

       cellCntr = cellCntr + 1;
      }
    }
    }
    EG_free(lobjs);

    PetscCall(DMLabelSetValue(bodyLabel, nStart + numVertices + numEdges + bodyFaceIndexStart + fID - 1, b));
    PetscCall(DMLabelSetValue(faceLabel, nStart + numVertices + numEdges + bodyFaceIndexStart + fID - 1, fID));
  }
  EG_free(fobjs);
  }

  PetscCall(PetscHMapIDestroy(&edgeMap));
  PetscCall(PetscHMapIDestroy(&bodyIndexMap));
  PetscCall(PetscHMapIDestroy(&bodyVertexMap));
  PetscCall(PetscHMapIDestroy(&bodyEdgeMap));
  PetscCall(PetscHMapIDestroy(&bodyEdgeGlobalMap));
  PetscCall(PetscHMapIDestroy(&bodyFaceMap));

  *newdm = dm;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateEGADS_Tess_Internal(MPI_Comm comm, ego context, ego model, DM *newdm)
{
  DMLabel              bodyLabel, faceLabel, edgeLabel, vertexLabel;
  /* EGADSLite variables */
  ego                  geom, *bodies, *fobjs;
  int                  b, oclass, mtype, nbodies, *senses;
  int                  totalNumTris = 0, totalNumPoints = 0;
  double               boundBox[6] = {0., 0., 0., 0., 0., 0.}, tessSize;
  /* PETSc variables */
  DM                   dm;
  PetscHMapI           pointIndexStartMap = NULL, triIndexStartMap = NULL, pTypeLabelMap = NULL, pIndexLabelMap = NULL;
  PetscHMapI           pBodyIndexLabelMap = NULL, triFaceIDLabelMap = NULL, triBodyIDLabelMap = NULL;
  PetscInt             dim = -1, cdim = -1, numCorners = 0, counter = 0;
  PetscInt            *cells  = NULL;
  const PetscInt      *cone = NULL;
  PetscReal           *coords = NULL;
  PetscMPIInt          rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    // ---------------------------------------------------------------------------------------------------
    // Generate Petsc Plex from EGADSlite created Tessellation of geometry
    // ---------------------------------------------------------------------------------------------------

  // Caluculate cell and vertex sizes
  PetscCall(EG_getTopology(model, &geom, &oclass, &mtype, NULL, &nbodies, &bodies, &senses));

  PetscCall(PetscHMapICreate(&pointIndexStartMap));
  PetscCall(PetscHMapICreate(&triIndexStartMap));
  PetscCall(PetscHMapICreate(&pTypeLabelMap));
  PetscCall(PetscHMapICreate(&pIndexLabelMap));
  PetscCall(PetscHMapICreate(&pBodyIndexLabelMap));
  PetscCall(PetscHMapICreate(&triFaceIDLabelMap));
  PetscCall(PetscHMapICreate(&triBodyIDLabelMap));

  /* Create Tessellation of Bodies */
  ego tessArray[nbodies];

  for (b = 0; b < nbodies; ++b) {
    ego             body = bodies[b];
    double          params[3] = {0.0, 0.0, 0.0};    // Parameters for Tessellation
    int             Nf, bodyNumPoints = 0, bodyNumTris = 0;
    PetscHashIter   PISiter, TISiter;
    PetscBool       PISfound, TISfound;

    /* Store Start Index for each Body's Point and Tris */
    PetscCall(PetscHMapIFind(pointIndexStartMap, b, &PISiter, &PISfound));
    PetscCall(PetscHMapIFind(triIndexStartMap, b, &TISiter, &TISfound));

    if (!PISfound)  PetscCall(PetscHMapISet(pointIndexStartMap, b, totalNumPoints));
    if (!TISfound)  PetscCall(PetscHMapISet(triIndexStartMap, b, totalNumTris));

    /* Calculate Tessellation parameters based on Bounding Box */
    /* Get Bounding Box Dimensions of the BODY */
    PetscCall(EG_getBoundingBox(body, boundBox));
    tessSize = boundBox[3] - boundBox[0];
    if (tessSize < boundBox[4] - boundBox[1]) tessSize = boundBox[4] - boundBox[1];
    if (tessSize < boundBox[5] - boundBox[2]) tessSize = boundBox[5] - boundBox[2];

    // TODO :: May want to give users tessellation parameter options //
    params[0] = 0.0250 * tessSize;
    params[1] = 0.0075 * tessSize;
    params[2] = 15.0;

    PetscCall(EG_makeTessBody(body, params, &tessArray[b]));

    PetscCall(EG_getBodyTopos(body, NULL, FACE,  &Nf, &fobjs));

    for (int f = 0; f < Nf; ++f) {
      ego             face = fobjs[f];
    int             len, fID, ntris;
    const int      *ptype, *pindex, *ptris, *ptric;
    const double   *pxyz, *puv;

    // Get Face ID //
    fID = EG_indexBodyTopo(body, face);

    // Checkout the Surface Tessellation //
    PetscCall(EG_getTessFace(tessArray[b], fID, &len, &pxyz, &puv, &ptype, &pindex, &ntris, &ptris, &ptric));

    // Determine total number of triangle cells in the tessellation //
    bodyNumTris += (int) ntris;

    // Check out the point index and coordinate //
    for (int p = 0; p < len; ++p) {
        int global;

        PetscCall(EG_localToGlobal(tessArray[b], fID, p+1, &global));

      // Determine the total number of points in the tessellation //
        bodyNumPoints = PetscMax(bodyNumPoints, global);
    }
    }
    EG_free(fobjs);

    totalNumPoints += bodyNumPoints;
    totalNumTris += bodyNumTris;
    }
  //}  - Original End of (rank == 0)

  dim = 2;
  cdim = 3;
  numCorners = 3;
  //PetscInt counter = 0;

  /* NEED TO DEFINE MATRICES/VECTORS TO STORE GEOM REFERENCE DATA   */
  /* Fill in below and use to define DMLabels after DMPlex creation */
  PetscCall(PetscMalloc2(totalNumPoints*cdim, &coords, totalNumTris*numCorners, &cells));

  for (b = 0; b < nbodies; ++b) {
  ego             body = bodies[b];
  int             Nf;
  PetscInt        pointIndexStart;
  PetscHashIter   PISiter;
  PetscBool       PISfound;

  PetscCall(PetscHMapIFind(pointIndexStartMap, b, &PISiter, &PISfound));
  PetscCheck(PISfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %d not found in pointIndexStartMap", b);
  PetscCall(PetscHMapIGet(pointIndexStartMap, b, &pointIndexStart));

  PetscCall(EG_getBodyTopos(body, NULL, FACE,  &Nf, &fobjs));

  for (int f = 0; f < Nf; ++f) {
    /* Get Face Object */
    ego              face = fobjs[f];
    int              len, fID, ntris;
    const int       *ptype, *pindex, *ptris, *ptric;
    const double    *pxyz, *puv;

    /* Get Face ID */
    fID = EG_indexBodyTopo(body, face);

    /* Checkout the Surface Tessellation */
    PetscCall(EG_getTessFace(tessArray[b], fID, &len, &pxyz, &puv, &ptype, &pindex, &ntris, &ptris, &ptric));

    /* Check out the point index and coordinate */
    for (int p = 0; p < len; ++p) {
    int              global;
    PetscHashIter    PTLiter, PILiter, PBLiter;
    PetscBool        PTLfound, PILfound, PBLfound;

    PetscCall(EG_localToGlobal(tessArray[b], fID, p+1, &global));

    /* Set the coordinates array for DAG */
    coords[((global-1+pointIndexStart)*3) + 0] = pxyz[(p*3)+0];
    coords[((global-1+pointIndexStart)*3) + 1] = pxyz[(p*3)+1];
    coords[((global-1+pointIndexStart)*3) + 2] = pxyz[(p*3)+2];

    /* Store Geometry Label Information for DMLabel assignment later */
    PetscCall(PetscHMapIFind(pTypeLabelMap, global-1+pointIndexStart, &PTLiter, &PTLfound));
    PetscCall(PetscHMapIFind(pIndexLabelMap, global-1+pointIndexStart, &PILiter, &PILfound));
    PetscCall(PetscHMapIFind(pBodyIndexLabelMap, global-1+pointIndexStart, &PBLiter, &PBLfound));

    if (!PTLfound) PetscCall(PetscHMapISet(pTypeLabelMap, global-1+pointIndexStart, ptype[p]));
    if (!PILfound) PetscCall(PetscHMapISet(pIndexLabelMap, global-1+pointIndexStart, pindex[p]));
    if (!PBLfound) PetscCall(PetscHMapISet(pBodyIndexLabelMap, global-1+pointIndexStart, b));

    if (ptype[p] < 0) PetscCall(PetscHMapISet(pIndexLabelMap, global-1+pointIndexStart, fID));
    }

    for (int t = 0; t < (int) ntris; ++t){
    int           global, globalA, globalB;
    PetscHashIter TFLiter, TBLiter;
    PetscBool     TFLfound, TBLfound;

    PetscCall(EG_localToGlobal(tessArray[b], fID, ptris[(t*3) + 0], &global));
    cells[(counter*3) +0] = global-1+pointIndexStart;

    PetscCall(EG_localToGlobal(tessArray[b], fID, ptris[(t*3) + 1], &globalA));
    cells[(counter*3) +1] = globalA-1+pointIndexStart;

    PetscCall(EG_localToGlobal(tessArray[b], fID, ptris[(t*3) + 2], &globalB));
    cells[(counter*3) +2] = globalB-1+pointIndexStart;

    PetscCall(PetscHMapIFind(triFaceIDLabelMap, counter, &TFLiter, &TFLfound));
        PetscCall(PetscHMapIFind(triBodyIDLabelMap, counter, &TBLiter, &TBLfound));

    if (!TFLfound)  PetscCall(PetscHMapISet(triFaceIDLabelMap, counter, fID));
        if (!TBLfound)  PetscCall(PetscHMapISet(triBodyIDLabelMap, counter, b));

    counter += 1;
    }
  }
  EG_free(fobjs);
  }
}

  //Build DMPlex
  PetscCall(DMPlexCreateFromCellListPetsc(PETSC_COMM_WORLD, dim, totalNumTris, totalNumPoints, numCorners, PETSC_TRUE, cells, cdim, coords, &dm));
  PetscCall(PetscFree2(coords, cells));

  // Embed EGADS model in DM
  {
    PetscContainer modelObj, contextObj;

    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &modelObj));
    PetscCall(PetscContainerSetPointer(modelObj, model));
    PetscCall(PetscObjectCompose((PetscObject) dm, "EGADS Model", (PetscObject) modelObj));
    PetscCall(PetscContainerDestroy(&modelObj));

    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &contextObj));
    PetscCall(PetscContainerSetPointer(contextObj, context));
    PetscCall(PetscContainerSetUserDestroy(contextObj, DMPlexEGADSDestroy_Private));
    PetscCall(PetscObjectCompose((PetscObject) dm, "EGADS Context", (PetscObject) contextObj));
    PetscCall(PetscContainerDestroy(&contextObj));
  }

  // Label Points
  PetscCall(DMCreateLabel(dm, "EGADS Body ID"));
  PetscCall(DMGetLabel(dm, "EGADS Body ID", &bodyLabel));
  PetscCall(DMCreateLabel(dm, "EGADS Face ID"));
  PetscCall(DMGetLabel(dm, "EGADS Face ID", &faceLabel));
  PetscCall(DMCreateLabel(dm, "EGADS Edge ID"));
  PetscCall(DMGetLabel(dm, "EGADS Edge ID", &edgeLabel));
  PetscCall(DMCreateLabel(dm, "EGADS Vertex ID"));
  PetscCall(DMGetLabel(dm, "EGADS Vertex ID", &vertexLabel));

   /* Get Number of DAG Nodes at each level */
  int   fStart, fEnd, eStart, eEnd, nStart, nEnd;

  PetscCall(DMPlexGetHeightStratum(dm, 0, &fStart, &fEnd));
  PetscCall(DMPlexGetHeightStratum(dm, 1, &eStart, &eEnd));
  PetscCall(DMPlexGetHeightStratum(dm, 2, &nStart, &nEnd));

  /* Set DMLabels for NODES */
  for (int n = nStart; n < nEnd; ++n) {
    int             pTypeVal, pIndexVal, pBodyVal;
    PetscHashIter   PTLiter, PILiter, PBLiter;
    PetscBool       PTLfound, PILfound, PBLfound;

    //Converted to Hash Tables
    PetscCall(PetscHMapIFind(pTypeLabelMap, n - nStart, &PTLiter, &PTLfound));
    PetscCheck(PTLfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "DAG Point %d not found in pTypeLabelMap", n);
    PetscCall(PetscHMapIGet(pTypeLabelMap, n - nStart, &pTypeVal));

    PetscCall(PetscHMapIFind(pIndexLabelMap, n - nStart, &PILiter, &PILfound));
    PetscCheck(PILfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "DAG Point %d not found in pIndexLabelMap", n);
    PetscCall(PetscHMapIGet(pIndexLabelMap, n - nStart, &pIndexVal));

    PetscCall(PetscHMapIFind(pBodyIndexLabelMap, n - nStart, &PBLiter, &PBLfound));
    PetscCheck(PBLfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "DAG Point %d not found in pBodyLabelMap", n);
    PetscCall(PetscHMapIGet(pBodyIndexLabelMap, n - nStart, &pBodyVal));

    PetscCall(DMLabelSetValue(bodyLabel, n, pBodyVal));
    if (pTypeVal == 0) PetscCall(DMLabelSetValue(vertexLabel, n, pIndexVal));
    if (pTypeVal >  0) PetscCall(DMLabelSetValue(edgeLabel, n, pIndexVal));
    if (pTypeVal <  0) PetscCall(DMLabelSetValue(faceLabel, n, pIndexVal));
  }

  /* Set DMLabels for Edges - Based on the DMLabels of the EDGE's NODES */
  for (int e = eStart; e < eEnd; ++e) {
  int    bodyID_0, vertexID_0, vertexID_1, edgeID_0, edgeID_1, faceID_0, faceID_1;

  PetscCall(DMPlexGetCone(dm, e, &cone));
  PetscCall(DMLabelGetValue(bodyLabel, cone[0], &bodyID_0));    // Do I need to check the other end?
  PetscCall(DMLabelGetValue(vertexLabel, cone[0], &vertexID_0));
  PetscCall(DMLabelGetValue(vertexLabel, cone[1], &vertexID_1));
  PetscCall(DMLabelGetValue(edgeLabel, cone[0], &edgeID_0));
  PetscCall(DMLabelGetValue(edgeLabel, cone[1], &edgeID_1));
  PetscCall(DMLabelGetValue(faceLabel, cone[0], &faceID_0));
  PetscCall(DMLabelGetValue(faceLabel, cone[1], &faceID_1));

  PetscCall(DMLabelSetValue(bodyLabel, e, bodyID_0));

  if (edgeID_0 == edgeID_1) PetscCall(DMLabelSetValue(edgeLabel, e, edgeID_0));
  else if (vertexID_0 > 0 && edgeID_1 > 0) PetscCall(DMLabelSetValue(edgeLabel, e, edgeID_1));
  else if (vertexID_1 > 0 && edgeID_0 > 0) PetscCall(DMLabelSetValue(edgeLabel, e, edgeID_0));
  else { /* Do Nothing */ }
  }

  /* Set DMLabels for Cells */
  for (int f = fStart; f < fEnd; ++f){
  int             edgeID_0;
  PetscInt        triBodyVal, triFaceVal;
  PetscHashIter   TFLiter, TBLiter;
  PetscBool       TFLfound, TBLfound;

    // Convert to Hash Table
  PetscCall(PetscHMapIFind(triFaceIDLabelMap, f - fStart, &TFLiter, &TFLfound));
  PetscCheck(TFLfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "DAG Point %d not found in triFaceIDLabelMap", f);
  PetscCall(PetscHMapIGet(triFaceIDLabelMap, f - fStart, &triFaceVal));

  PetscCall(PetscHMapIFind(triBodyIDLabelMap, f - fStart, &TBLiter, &TBLfound));
  PetscCheck(TBLfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "DAG Point %d not found in triBodyIDLabelMap", f);
    PetscCall(PetscHMapIGet(triBodyIDLabelMap, f - fStart, &triBodyVal));

  PetscCall(DMLabelSetValue(bodyLabel, f, triBodyVal));
  PetscCall(DMLabelSetValue(faceLabel, f, triFaceVal));

  /* Finish Labeling previously unlabeled DMPlex Edges - Assumes Triangular Cell (3 Edges Max) */
  PetscCall(DMPlexGetCone(dm, f, &cone));

  for (int jj = 0; jj < 3; ++jj) {
    PetscCall(DMLabelGetValue(edgeLabel, cone[jj], &edgeID_0));

    if (edgeID_0 < 0) {
    PetscCall(DMLabelSetValue(bodyLabel, cone[jj], triBodyVal));
      PetscCall(DMLabelSetValue(faceLabel, cone[jj], triFaceVal));
    }
  }
  }

  *newdm = dm;
  PetscFunctionReturn(0);
}
#endif

/*@
  DMPlexInflateToGeomModel - Snaps the vertex coordinates of a DMPlex object representing the mesh to its geometry if some vertices depart from the model. This usually happens with non-conforming refinement.

  Collective on dm

  Input Parameter:
. dm - The uninflated DM object representing the mesh

  Output Parameter:
. dm - The inflated DM object representing the mesh

  Level: intermediate

.seealso: `DMPLEX`, `DMCreate()`, `DMPlexCreateEGADS()`
@*/
PetscErrorCode DMPlexInflateToGeomModel(DM dm)
{
#if defined(PETSC_HAVE_EGADS)
  /* EGADS Variables */
  ego            model, geom, body, face, edge;
  ego           *bodies;
  int            Nb, oclass, mtype, *senses;
  double         result[3];
  /* PETSc Variables */
  DM             cdm;
  PetscContainer modelObj;
  DMLabel        bodyLabel, faceLabel, edgeLabel, vertexLabel;
  Vec            coordinates;
  PetscScalar   *coords;
  PetscInt       bodyID, faceID, edgeID, vertexID;
  PetscInt       cdim, d, vStart, vEnd, v;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_EGADS)
  PetscCall(PetscObjectQuery((PetscObject) dm, "EGADS Model", (PetscObject *) &modelObj));
  if (!modelObj) PetscFunctionReturn(0);
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMGetLabel(dm, "EGADS Body ID", &bodyLabel));
  PetscCall(DMGetLabel(dm, "EGADS Face ID", &faceLabel));
  PetscCall(DMGetLabel(dm, "EGADS Edge ID", &edgeLabel));
  PetscCall(DMGetLabel(dm, "EGADS Vertex ID", &vertexLabel));

  PetscCall(PetscContainerGetPointer(modelObj, (void **) &model));
  PetscCall(EG_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));

  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(VecGetArrayWrite(coordinates, &coords));
  for (v = vStart; v < vEnd; ++v) {
    PetscScalar *vcoords;

    PetscCall(DMLabelGetValue(bodyLabel, v, &bodyID));
    PetscCall(DMLabelGetValue(faceLabel, v, &faceID));
    PetscCall(DMLabelGetValue(edgeLabel, v, &edgeID));
    PetscCall(DMLabelGetValue(vertexLabel, v, &vertexID));

    PetscCheck(bodyID < Nb,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Body %" PetscInt_FMT " is not in [0, %d)", bodyID, Nb);
    body = bodies[bodyID];

    PetscCall(DMPlexPointLocalRef(cdm, v, coords, (void *) &vcoords));
    if (edgeID > 0) {
      /* Snap to EDGE at nearest location */
      double params[1];
      PetscCall(EG_objectBodyTopo(body, EDGE, edgeID, &edge));
      PetscCall(EG_invEvaluate(edge, vcoords, params, result)); // Get (x,y,z) of nearest point on EDGE
      for (d = 0; d < cdim; ++d) vcoords[d] = result[d];
    } else if (faceID > 0) {
      /* Snap to FACE at nearest location */
      double params[2];
      PetscCall(EG_objectBodyTopo(body, FACE, faceID, &face));
      PetscCall(EG_invEvaluate(face, vcoords, params, result)); // Get (x,y,z) of nearest point on FACE
      for (d = 0; d < cdim; ++d) vcoords[d] = result[d];
    }
  }
  PetscCall(VecRestoreArrayWrite(coordinates, &coords));
  /* Clear out global coordinates */
  PetscCall(VecDestroy(&dm->coordinates[0].x));
#endif
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreateEGADSFromFile - Create a DMPlex mesh from an EGADS, IGES, or STEP file.

  Collective

  Input Parameters:
+ comm     - The MPI communicator
- filename - The name of the EGADS, IGES, or STEP file

  Output Parameter:
. dm       - The DM object representing the mesh

  Level: beginner

.seealso: `DMPLEX`, `DMCreate()`, `DMPlexCreateEGADS()`, `DMPlexCreateEGADSLiteFromFile()`
@*/
PetscErrorCode DMPlexCreateEGADSFromFile(MPI_Comm comm, const char filename[], DM *dm)
{
  PetscMPIInt    rank;
#if defined(PETSC_HAVE_EGADS)
  ego            context= NULL, model = NULL;
#endif
  PetscBool      printModel = PETSC_FALSE, tessModel = PETSC_FALSE, newModel = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidCharPointer(filename, 2);
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-dm_plex_egads_print_model", &printModel, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-dm_plex_egads_tess_model", &tessModel, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-dm_plex_egads_new_model", &newModel, NULL));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
#if defined(PETSC_HAVE_EGADS)
  if (rank == 0) {

    PetscCall(EG_open(&context));
    PetscCall(EG_loadModel(context, 0, filename, &model));
    if (printModel) PetscCall(DMPlexEGADSPrintModel_Internal(model));

  }
  if (tessModel)     PetscCall(DMPlexCreateEGADS_Tess_Internal(comm, context, model, dm));
  else if (newModel) PetscCall(DMPlexCreateEGADS(comm, context, model, dm));
  else               PetscCall(DMPlexCreateEGADS_Internal(comm, context, model, dm));
  PetscFunctionReturn(0);
#else
  SETERRQ(comm, PETSC_ERR_SUP, "This method requires EGADS support. Reconfigure using --download-egads");
#endif
}
