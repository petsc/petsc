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
  CHKERRQ(DMGetCoordinateDM(dm, &cdm));
  CHKERRQ(DMGetCoordinateDim(dm, &dE));
  CHKERRQ(DMGetCoordinatesLocal(dm, &coordinatesLocal));
  CHKERRQ(EG_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));
  PetscCheckFalse(bodyID >= Nb,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Body %D is not in [0, %d)", bodyID, Nb);
  body = bodies[bodyID];

  if (edgeID >= 0)      {CHKERRQ(EG_objectBodyTopo(body, EDGE, edgeID, &obj)); Np = 1;}
  else if (faceID >= 0) {CHKERRQ(EG_objectBodyTopo(body, FACE, faceID, &obj)); Np = 2;}
  else {
    for (d = 0; d < dE; ++d) gcoords[d] = mcoords[d];
    PetscFunctionReturn(0);
  }

  /* Calculate parameters (t or u,v) for vertices */
  CHKERRQ(DMPlexVecGetClosure(cdm, NULL, coordinatesLocal, p, &Nv, &coords));
  Nv  /= dE;
  if (Nv == 1) {
    CHKERRQ(DMPlexVecRestoreClosure(cdm, NULL, coordinatesLocal, p, &Nv, &coords));
    for (d = 0; d < dE; ++d) gcoords[d] = mcoords[d];
    PetscFunctionReturn(0);
  }
  PetscCheckFalse(Nv > 16,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle %D coordinates associated to point %D", Nv, p);

  /* Correct EGADSlite 2pi bug when calculating nearest point on Periodic Surfaces */
  CHKERRQ(EG_getRange(obj, range, &peri));
  for (v = 0; v < Nv; ++v) {
    CHKERRQ(EG_invEvaluate(obj, &coords[v*dE], &paramsV[v*3], &resultV[v*3]));
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
  CHKERRQ(DMPlexVecRestoreClosure(cdm, NULL, coordinatesLocal, p, &Nv, &coords));
  /* Calculate parameters (t or u,v) for new vertex at edge midpoint */
  for (pm = 0; pm < Np; ++pm) {
    params[pm] = 0.;
    for (v = 0; v < Nv; ++v) params[pm] += paramsV[v*3+pm];
    params[pm] /= Nv;
  }
  PetscCheckFalse((params[0] < range[0]) || (params[0] > range[1]),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point %D had bad interpolation", p);
  PetscCheckFalse(Np > 1 && ((params[1] < range[2]) || (params[1] > range[3])),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point %D had bad interpolation", p);
  /* Put coordinates for new vertex in result[] */
  CHKERRQ(EG_evaluate(obj, params, result));
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

.seealso: DMRefine(), DMPlexCreate(), DMPlexSetRefinementUniform()
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

    CHKERRQ(DMGetLabel(dm, "EGADS Body ID", &bodyLabel));
    CHKERRQ(DMGetLabel(dm, "EGADS Face ID", &faceLabel));
    CHKERRQ(DMGetLabel(dm, "EGADS Edge ID", &edgeLabel));
    if (!bodyLabel || !faceLabel || !edgeLabel || plex->ignoreModel) {
      for (d = 0; d < dE; ++d) gcoords[d] = mcoords[d];
      PetscFunctionReturn(0);
    }
    CHKERRQ(PetscObjectQuery((PetscObject) dm, "EGADS Model", (PetscObject *) &modelObj));
    if (!modelObj) {
      CHKERRQ(PetscObjectQuery((PetscObject) dm, "EGADSLite Model", (PetscObject *) &modelObj));
      islite = PETSC_TRUE;
    }
    if (!modelObj) {
      for (d = 0; d < dE; ++d) gcoords[d] = mcoords[d];
      PetscFunctionReturn(0);
    }
    CHKERRQ(PetscContainerGetPointer(modelObj, (void **) &model));
    CHKERRQ(DMLabelGetValue(bodyLabel, p, &bodyID));
    CHKERRQ(DMLabelGetValue(faceLabel, p, &faceID));
    CHKERRQ(DMLabelGetValue(edgeLabel, p, &edgeID));
    /* Allows for "Connective" Plex Edges present in models with multiple non-touching Entities */
    if (bodyID < 0) {
      for (d = 0; d < dE; ++d) gcoords[d] = mcoords[d];
      PetscFunctionReturn(0);
    }
    if (islite) CHKERRQ(DMPlexSnapToGeomModel_EGADSLite_Internal(dm, p, model, bodyID, faceID, edgeID, mcoords, gcoords));
    else        CHKERRQ(DMPlexSnapToGeomModel_EGADS_Internal(dm, p, model, bodyID, faceID, edgeID, mcoords, gcoords));
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
  CHKERRQ(EG_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF, " Number of BODIES (nbodies): %d \n", Nb));

  for (b = 0; b < Nb; ++b) {
    ego body = bodies[b];
    int id, Nsh, Nf, Nl, l, Ne, e, Nv, v;

    /* Output Basic Model Topology */
    CHKERRQ(EG_getBodyTopos(body, NULL, SHELL, &Nsh, &objs));
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "   Number of SHELLS: %d \n", Nsh));
    EG_free(objs);

    CHKERRQ(EG_getBodyTopos(body, NULL, FACE,  &Nf, &objs));
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "   Number of FACES: %d \n", Nf));
    EG_free(objs);

    CHKERRQ(EG_getBodyTopos(body, NULL, LOOP,  &Nl, &lobjs));
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "   Number of LOOPS: %d \n", Nl));

    CHKERRQ(EG_getBodyTopos(body, NULL, EDGE,  &Ne, &objs));
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "   Number of EDGES: %d \n", Ne));
    EG_free(objs);

    CHKERRQ(EG_getBodyTopos(body, NULL, NODE,  &Nv, &objs));
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "   Number of NODES: %d \n", Nv));
    EG_free(objs);

    for (l = 0; l < Nl; ++l) {
      ego loop = lobjs[l];

      id   = EG_indexBodyTopo(body, loop);
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "          LOOP ID: %d\n", id));

      /* Get EDGE info which associated with the current LOOP */
      CHKERRQ(EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses));

      for (e = 0; e < Ne; ++e) {
        ego    edge      = objs[e];
        double range[4]  = {0., 0., 0., 0.};
        double point[3]  = {0., 0., 0.};
        double params[3] = {0., 0., 0.};
        double result[18];
        int    peri;

        id   = EG_indexBodyTopo(body, edge);
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "            EDGE ID: %d (%d)\n", id, e));

        CHKERRQ(EG_getRange(edge, range, &peri));
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "  Range = %lf, %lf, %lf, %lf \n", range[0], range[1], range[2], range[3]));

        /* Get NODE info which associated with the current EDGE */
        CHKERRQ(EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
        if (mtype == DEGENERATE) {
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "  EDGE %d is DEGENERATE \n", id));
        } else {
          params[0] = range[0];
          CHKERRQ(EG_evaluate(edge, params, result));
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "   between (%lf, %lf, %lf)", result[0], result[1], result[2]));
          params[0] = range[1];
          CHKERRQ(EG_evaluate(edge, params, result));
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF, " and (%lf, %lf, %lf)\n", result[0], result[1], result[2]));
        }

        for (v = 0; v < Nv; ++v) {
          ego    vertex = nobjs[v];
          double limits[4];
          int    dummy;

          CHKERRQ(EG_getTopology(vertex, &geom, &oclass, &mtype, limits, &dummy, &mobjs, &senses));
          id   = EG_indexBodyTopo(body, vertex);
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "              NODE ID: %d \n", id));
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "                 (x, y, z) = (%lf, %lf, %lf) \n", limits[0], limits[1], limits[2]));

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
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    const PetscInt debug = 0;

    /* ---------------------------------------------------------------------------------------------------
    Generate Petsc Plex
      Get all Nodes in model, record coordinates in a correctly formatted array
      Cycle through bodies, cycle through loops, recorde NODE IDs in a correctly formatted array
      We need to uniformly refine the initial geometry to guarantee a valid mesh
    */

    /* Calculate cell and vertex sizes */
    CHKERRQ(EG_getTopology(model, &geom, &oclass, &mtype, NULL, &nbodies, &bodies, &senses));
    CHKERRQ(PetscHMapICreate(&edgeMap));
    numEdges = 0;
    for (b = 0; b < nbodies; ++b) {
      ego body = bodies[b];
      int id, Nl, l, Nv, v;

      CHKERRQ(EG_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs));
      for (l = 0; l < Nl; ++l) {
        ego loop = lobjs[l];
        int Ner  = 0, Ne, e, Nc;

        CHKERRQ(EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses));
        for (e = 0; e < Ne; ++e) {
          ego edge = objs[e];
          int Nv, id;
          PetscHashIter iter;
          PetscBool     found;

          CHKERRQ(EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
          if (mtype == DEGENERATE) continue;
          id = EG_indexBodyTopo(body, edge);
          CHKERRQ(PetscHMapIFind(edgeMap, id-1, &iter, &found));
          if (!found) CHKERRQ(PetscHMapISet(edgeMap, id-1, numEdges++));
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
      CHKERRQ(EG_getBodyTopos(body, NULL, NODE, &Nv, &nobjs));
      for (v = 0; v < Nv; ++v) {
        ego vertex = nobjs[v];

        id = EG_indexBodyTopo(body, vertex);
        /* TODO: Instead of assuming contiguous ids, we could use a hash table */
        numVertices = PetscMax(id, numVertices);
      }
      EG_free(lobjs);
      EG_free(nobjs);
    }
    CHKERRQ(PetscHMapIGetSize(edgeMap, &numEdges));
    newVertices  = numEdges + numQuads;
    numVertices += newVertices;

    dim        = 2; /* Assume 3D Models :: Need to update to handle 2D Models in the future */
    cdim       = 3; /* Assume 3D Models :: Need to update to handle 2D Models in the future */
    numCorners = 3; /* Split cells into triangles */
    CHKERRQ(PetscMalloc3(numVertices*cdim, &coords, numCells*numCorners, &cells, maxCorners, &cone));

    /* Get vertex coordinates */
    for (b = 0; b < nbodies; ++b) {
      ego body = bodies[b];
      int id, Nv, v;

      CHKERRQ(EG_getBodyTopos(body, NULL, NODE, &Nv, &nobjs));
      for (v = 0; v < Nv; ++v) {
        ego    vertex = nobjs[v];
        double limits[4];
        int    dummy;

        CHKERRQ(EG_getTopology(vertex, &geom, &oclass, &mtype, limits, &dummy, &mobjs, &senses));
        id   = EG_indexBodyTopo(body, vertex);
        coords[(id-1)*cdim+0] = limits[0];
        coords[(id-1)*cdim+1] = limits[1];
        coords[(id-1)*cdim+2] = limits[2];
      }
      EG_free(nobjs);
    }
    CHKERRQ(PetscHMapIClear(edgeMap));
    fOff     = numVertices - newVertices + numEdges;
    numEdges = 0;
    numQuads = 0;
    for (b = 0; b < nbodies; ++b) {
      ego body = bodies[b];
      int Nl, l;

      CHKERRQ(EG_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs));
      for (l = 0; l < Nl; ++l) {
        ego loop = lobjs[l];
        int lid, Ner = 0, Ne, e;

        lid  = EG_indexBodyTopo(body, loop);
        CHKERRQ(EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses));
        for (e = 0; e < Ne; ++e) {
          ego       edge = objs[e];
          int       eid, Nv;
          PetscHashIter iter;
          PetscBool     found;

          CHKERRQ(EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
          if (mtype == DEGENERATE) continue;
          ++Ner;
          eid  = EG_indexBodyTopo(body, edge);
          CHKERRQ(PetscHMapIFind(edgeMap, eid-1, &iter, &found));
          if (!found) {
            PetscInt v = numVertices - newVertices + numEdges;
            double range[4], params[3] = {0., 0., 0.}, result[18];
            int    periodic[2];

            CHKERRQ(PetscHMapISet(edgeMap, eid-1, numEdges++));
            CHKERRQ(EG_getRange(edge, range, periodic));
            params[0] = 0.5*(range[0] + range[1]);
            CHKERRQ(EG_evaluate(edge, params, result));
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

          CHKERRQ(EG_getBodyTopos(body, loop, FACE, &Nf, &fobjs));
          face = fobjs[0];
          fid  = EG_indexBodyTopo(body, face);
          PetscCheckFalse(Nf != 1,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Loop %d has %d faces, instead of 1 (%d)", lid-1, Nf, fid);
          CHKERRQ(EG_getRange(face, range, periodic));
          params[0] = 0.5*(range[0] + range[1]);
          params[1] = 0.5*(range[2] + range[3]);
          CHKERRQ(EG_evaluate(face, params, result));
          coords[v*cdim+0] = result[0];
          coords[v*cdim+1] = result[1];
          coords[v*cdim+2] = result[2];
        }
      }
    }
    PetscCheckFalse(numEdges + numQuads != newVertices,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of new vertices %D != %D previous count", numEdges + numQuads, newVertices);

    /* Get cell vertices by traversing loops */
    numQuads = 0;
    cOff     = 0;
    for (b = 0; b < nbodies; ++b) {
      ego body = bodies[b];
      int id, Nl, l;

      CHKERRQ(EG_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs));
      for (l = 0; l < Nl; ++l) {
        ego loop = lobjs[l];
        int lid, Ner = 0, Ne, e, nc = 0, c, Nt, t;

        lid  = EG_indexBodyTopo(body, loop);
        CHKERRQ(EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses));

        for (e = 0; e < Ne; ++e) {
          ego edge = objs[e];
          int points[3];
          int eid, Nv, v, tmp;

          eid = EG_indexBodyTopo(body, edge);
          CHKERRQ(EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
          if (mtype == DEGENERATE) continue;
          else                     ++Ner;
          PetscCheckFalse(Nv != 2,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Edge %d has %d vertices != 2", eid, Nv);

          for (v = 0; v < Nv; ++v) {
            ego vertex = nobjs[v];

            id = EG_indexBodyTopo(body, vertex);
            points[v*2] = id-1;
          }
          {
            PetscInt edgeNum;

            CHKERRQ(PetscHMapIGet(edgeMap, eid-1, &edgeNum));
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
        PetscCheckFalse(nc != 2*Ner,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of corners %D != %D", nc, 2*Ner);
        if (Ner == 4) {cone[nc++] = numVertices - newVertices + numEdges + numQuads++;}
        PetscCheckFalse(nc > maxCorners,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of corners %D > %D max", nc, maxCorners);
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
            CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "  LOOP Corner NODEs Triangle %D (", t));
            for (c = 0; c < numCorners; ++c) {
              if (c > 0) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, ", "));
              CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "%D", cells[(cOff-Nt+t)*numCorners+c]));
            }
            CHKERRQ(PetscPrintf(PETSC_COMM_SELF, ")\n"));
          }
        }
      }
      EG_free(lobjs);
    }
  }
  PetscCheckFalse(cOff != numCells,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Count of total cells %D != %D previous count", cOff, numCells);
  CHKERRQ(DMPlexCreateFromCellListPetsc(PETSC_COMM_WORLD, dim, numCells, numVertices, numCorners, PETSC_TRUE, cells, cdim, coords, &dm));
  CHKERRQ(PetscFree3(coords, cells, cone));
  CHKERRQ(PetscInfo(dm, " Total Number of Unique Cells    = %D (%D)\n", numCells, newCells));
  CHKERRQ(PetscInfo(dm, " Total Number of Unique Vertices = %D (%D)\n", numVertices, newVertices));
  /* Embed EGADS model in DM */
  {
    PetscContainer modelObj, contextObj;

    CHKERRQ(PetscContainerCreate(PETSC_COMM_SELF, &modelObj));
    CHKERRQ(PetscContainerSetPointer(modelObj, model));
    CHKERRQ(PetscObjectCompose((PetscObject) dm, "EGADS Model", (PetscObject) modelObj));
    CHKERRQ(PetscContainerDestroy(&modelObj));

    CHKERRQ(PetscContainerCreate(PETSC_COMM_SELF, &contextObj));
    CHKERRQ(PetscContainerSetPointer(contextObj, context));
    CHKERRQ(PetscContainerSetUserDestroy(contextObj, DMPlexEGADSDestroy_Private));
    CHKERRQ(PetscObjectCompose((PetscObject) dm, "EGADS Context", (PetscObject) contextObj));
    CHKERRQ(PetscContainerDestroy(&contextObj));
  }
  /* Label points */
  CHKERRQ(DMCreateLabel(dm, "EGADS Body ID"));
  CHKERRQ(DMGetLabel(dm, "EGADS Body ID", &bodyLabel));
  CHKERRQ(DMCreateLabel(dm, "EGADS Face ID"));
  CHKERRQ(DMGetLabel(dm, "EGADS Face ID", &faceLabel));
  CHKERRQ(DMCreateLabel(dm, "EGADS Edge ID"));
  CHKERRQ(DMGetLabel(dm, "EGADS Edge ID", &edgeLabel));
  CHKERRQ(DMCreateLabel(dm, "EGADS Vertex ID"));
  CHKERRQ(DMGetLabel(dm, "EGADS Vertex ID", &vertexLabel));
  cOff = 0;
  for (b = 0; b < nbodies; ++b) {
    ego body = bodies[b];
    int id, Nl, l;

    CHKERRQ(EG_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs));
    for (l = 0; l < Nl; ++l) {
      ego  loop = lobjs[l];
      ego *fobjs;
      int  lid, Nf, fid, Ner = 0, Ne, e, Nt = 0, t;

      lid  = EG_indexBodyTopo(body, loop);
      CHKERRQ(EG_getBodyTopos(body, loop, FACE, &Nf, &fobjs));
      PetscCheckFalse(Nf > 1,PETSC_COMM_SELF, PETSC_ERR_SUP, "Loop %d has %d > 1 faces, which is not supported", lid, Nf);
      fid  = EG_indexBodyTopo(body, fobjs[0]);
      EG_free(fobjs);
      CHKERRQ(EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses));
      for (e = 0; e < Ne; ++e) {
        ego             edge = objs[e];
        int             eid, Nv, v;
        PetscInt        points[3], support[2], numEdges, edgeNum;
        const PetscInt *edges;

        eid = EG_indexBodyTopo(body, edge);
        CHKERRQ(EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
        if (mtype == DEGENERATE) continue;
        else                     ++Ner;
        for (v = 0; v < Nv; ++v) {
          ego vertex = nobjs[v];

          id   = EG_indexBodyTopo(body, vertex);
          CHKERRQ(DMLabelSetValue(edgeLabel, numCells + id-1, eid));
          points[v*2] = numCells + id-1;
        }
        CHKERRQ(PetscHMapIGet(edgeMap, eid-1, &edgeNum));
        points[1] = numCells + numVertices - newVertices + edgeNum;

        CHKERRQ(DMLabelSetValue(edgeLabel, points[1], eid));
        support[0] = points[0];
        support[1] = points[1];
        CHKERRQ(DMPlexGetJoin(dm, 2, support, &numEdges, &edges));
        PetscCheckFalse(numEdges != 1,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Vertices (%D, %D) should only bound 1 edge, not %D", support[0], support[1], numEdges);
        CHKERRQ(DMLabelSetValue(edgeLabel, edges[0], eid));
        CHKERRQ(DMPlexRestoreJoin(dm, 2, support, &numEdges, &edges));
        support[0] = points[1];
        support[1] = points[2];
        CHKERRQ(DMPlexGetJoin(dm, 2, support, &numEdges, &edges));
        PetscCheckFalse(numEdges != 1,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Vertices (%D, %D) should only bound 1 edge, not %D", support[0], support[1], numEdges);
        CHKERRQ(DMLabelSetValue(edgeLabel, edges[0], eid));
        CHKERRQ(DMPlexRestoreJoin(dm, 2, support, &numEdges, &edges));
      }
      switch (Ner) {
        case 2: Nt = 2;break;
        case 3: Nt = 4;break;
        case 4: Nt = 8;break;
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Loop with %d edges is unsupported", Ner);
      }
      for (t = 0; t < Nt; ++t) {
        CHKERRQ(DMLabelSetValue(bodyLabel, cOff+t, b));
        CHKERRQ(DMLabelSetValue(faceLabel, cOff+t, fid));
      }
      cOff += Nt;
    }
    EG_free(lobjs);
  }
  CHKERRQ(PetscHMapIDestroy(&edgeMap));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL;
    PetscInt  clSize, cl, bval, fval;

    CHKERRQ(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &clSize, &closure));
    CHKERRQ(DMLabelGetValue(bodyLabel, c, &bval));
    CHKERRQ(DMLabelGetValue(faceLabel, c, &fval));
    for (cl = 0; cl < clSize*2; cl += 2) {
      CHKERRQ(DMLabelSetValue(bodyLabel, closure[cl], bval));
      CHKERRQ(DMLabelSetValue(faceLabel, closure[cl], fval));
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &clSize, &closure));
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
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  if (!rank) {
    // ---------------------------------------------------------------------------------------------------
    // Generate Petsc Plex
    //  Get all Nodes in model, record coordinates in a correctly formatted array
    //  Cycle through bodies, cycle through loops, recorde NODE IDs in a correctly formatted array
    //  We need to uniformly refine the initial geometry to guarantee a valid mesh

  // Caluculate cell and vertex sizes
  CHKERRQ(EG_getTopology(model, &geom, &oclass, &mtype, NULL, &nbodies, &bodies, &senses));

    CHKERRQ(PetscHMapICreate(&edgeMap));
  CHKERRQ(PetscHMapICreate(&bodyIndexMap));
  CHKERRQ(PetscHMapICreate(&bodyVertexMap));
  CHKERRQ(PetscHMapICreate(&bodyEdgeMap));
  CHKERRQ(PetscHMapICreate(&bodyEdgeGlobalMap));
  CHKERRQ(PetscHMapICreate(&bodyFaceMap));

  for (b = 0; b < nbodies; ++b) {
      ego             body = bodies[b];
    int             Nf, Ne, Nv;
    PetscHashIter   BIiter, BViter, BEiter, BEGiter, BFiter, EMiter;
    PetscBool       BIfound, BVfound, BEfound, BEGfound, BFfound, EMfound;

    CHKERRQ(PetscHMapIFind(bodyIndexMap, b, &BIiter, &BIfound));
    CHKERRQ(PetscHMapIFind(bodyVertexMap, b, &BViter, &BVfound));
    CHKERRQ(PetscHMapIFind(bodyEdgeMap, b, &BEiter, &BEfound));
    CHKERRQ(PetscHMapIFind(bodyEdgeGlobalMap, b, &BEGiter, &BEGfound));
    CHKERRQ(PetscHMapIFind(bodyFaceMap, b, &BFiter, &BFfound));

    if (!BIfound)  CHKERRQ(PetscHMapISet(bodyIndexMap, b, numFaces + numEdges + numVertices));
    if (!BVfound)  CHKERRQ(PetscHMapISet(bodyVertexMap, b, numVertices));
    if (!BEfound)  CHKERRQ(PetscHMapISet(bodyEdgeMap, b, numEdges));
    if (!BEGfound) CHKERRQ(PetscHMapISet(bodyEdgeGlobalMap, b, edgeCntr));
    if (!BFfound)  CHKERRQ(PetscHMapISet(bodyFaceMap, b, numFaces));

    CHKERRQ(EG_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
    CHKERRQ(EG_getBodyTopos(body, NULL, EDGE, &Ne, &eobjs));
    CHKERRQ(EG_getBodyTopos(body, NULL, NODE, &Nv, &nobjs));
    EG_free(fobjs);
    EG_free(eobjs);
    EG_free(nobjs);

    // Remove DEGENERATE EDGES from Edge count
    CHKERRQ(EG_getBodyTopos(body, NULL, EDGE, &Ne, &eobjs));
    int Netemp = 0;
    for (int e = 0; e < Ne; ++e) {
      ego     edge = eobjs[e];
      int     eid;

      CHKERRQ(EG_getInfo(edge, &oclass, &mtype, &topRef, &prev, &next));
      eid = EG_indexBodyTopo(body, edge);

      CHKERRQ(PetscHMapIFind(edgeMap, edgeCntr + eid - 1, &EMiter, &EMfound));
      if (mtype == DEGENERATE) {
        if (!EMfound) CHKERRQ(PetscHMapISet(edgeMap, edgeCntr + eid - 1, -1));
      }
      else {
      ++Netemp;
        if (!EMfound) CHKERRQ(PetscHMapISet(edgeMap, edgeCntr + eid - 1, Netemp));
      }
    }
    EG_free(eobjs);

    // Determine Number of Cells
    CHKERRQ(EG_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
    for (int f = 0; f < Nf; ++f) {
        ego     face = fobjs[f];
    int     edgeTemp = 0;

      CHKERRQ(EG_getBodyTopos(body, face, EDGE, &Ne, &eobjs));
      for (int e = 0; e < Ne; ++e) {
        ego     edge = eobjs[e];

        CHKERRQ(EG_getInfo(edge, &oclass, &mtype, &topRef, &prev, &next));
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

  CHKERRQ(PetscMalloc2(numPoints*cdim, &coords, numCells*numCorners, &cells));

  // Get Vertex Coordinates and Set up Cells
  for (b = 0; b < nbodies; ++b) {
    ego             body = bodies[b];
    int             Nf, Ne, Nv;
    PetscInt        bodyVertexIndexStart, bodyEdgeIndexStart, bodyEdgeGlobalIndexStart, bodyFaceIndexStart;
    PetscHashIter   BViter, BEiter, BEGiter, BFiter, EMiter;
    PetscBool       BVfound, BEfound, BEGfound, BFfound, EMfound;

    // Vertices on Current Body
    CHKERRQ(EG_getBodyTopos(body, NULL, NODE, &Nv, &nobjs));

    CHKERRQ(PetscHMapIFind(bodyVertexMap, b, &BViter, &BVfound));
    PetscCheck(BVfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %d not found in bodyVertexMap", b);
    CHKERRQ(PetscHMapIGet(bodyVertexMap, b, &bodyVertexIndexStart));

    for (int v = 0; v < Nv; ++v) {
      ego    vertex = nobjs[v];
    double limits[4];
    int    id, dummy;

    CHKERRQ(EG_getTopology(vertex, &geom, &oclass, &mtype, limits, &dummy, &mobjs, &senses));
    id = EG_indexBodyTopo(body, vertex);

    coords[(bodyVertexIndexStart + id - 1)*cdim + 0] = limits[0];
    coords[(bodyVertexIndexStart + id - 1)*cdim + 1] = limits[1];
    coords[(bodyVertexIndexStart + id - 1)*cdim + 2] = limits[2];
    }
    EG_free(nobjs);

    // Edge Midpoint Vertices on Current Body
    CHKERRQ(EG_getBodyTopos(body, NULL, EDGE, &Ne, &eobjs));

    CHKERRQ(PetscHMapIFind(bodyEdgeMap, b, &BEiter, &BEfound));
    PetscCheck(BEfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %d not found in bodyEdgeMap", b);
    CHKERRQ(PetscHMapIGet(bodyEdgeMap, b, &bodyEdgeIndexStart));

    CHKERRQ(PetscHMapIFind(bodyEdgeGlobalMap, b, &BEGiter, &BEGfound));
    PetscCheck(BEGfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %d not found in bodyEdgeGlobalMap", b);
    CHKERRQ(PetscHMapIGet(bodyEdgeGlobalMap, b, &bodyEdgeGlobalIndexStart));

    for (int e = 0; e < Ne; ++e) {
      ego          edge = eobjs[e];
    double       range[2], avgt[1], cntrPnt[9];
    int          eid, eOffset;
    int          periodic;

    CHKERRQ(EG_getInfo(edge, &oclass, &mtype, &topRef, &prev, &next));
    if (mtype == DEGENERATE) {continue;}

    eid = EG_indexBodyTopo(body, edge);

    // get relative offset from globalEdgeID Vector
    CHKERRQ(PetscHMapIFind(edgeMap, bodyEdgeGlobalIndexStart + eid - 1, &EMiter, &EMfound));
      PetscCheck(EMfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "Edge %d not found in edgeMap", bodyEdgeGlobalIndexStart + eid - 1);
      CHKERRQ(PetscHMapIGet(edgeMap, bodyEdgeGlobalIndexStart + eid - 1, &eOffset));

    CHKERRQ(EG_getRange(edge, range, &periodic));
    avgt[0] = (range[0] + range[1]) /  2.;

    CHKERRQ(EG_evaluate(edge, avgt, cntrPnt));
    coords[(numVertices + bodyEdgeIndexStart + eOffset - 1)*cdim + 0] = cntrPnt[0];
        coords[(numVertices + bodyEdgeIndexStart + eOffset - 1)*cdim + 1] = cntrPnt[1];
    coords[(numVertices + bodyEdgeIndexStart + eOffset - 1)*cdim + 2] = cntrPnt[2];
    }
    EG_free(eobjs);

    // Face Midpoint Vertices on Current Body
    CHKERRQ(EG_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));

    CHKERRQ(PetscHMapIFind(bodyFaceMap, b, &BFiter, &BFfound));
    PetscCheck(BFfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %d not found in bodyFaceMap", b);
    CHKERRQ(PetscHMapIGet(bodyFaceMap, b, &bodyFaceIndexStart));

    for (int f = 0; f < Nf; ++f) {
    ego       face = fobjs[f];
    double    range[4], avgUV[2], cntrPnt[18];
    int       peri, id;

    id = EG_indexBodyTopo(body, face);
    CHKERRQ(EG_getRange(face, range, &peri));

    avgUV[0] = (range[0] + range[1]) / 2.;
    avgUV[1] = (range[2] + range[3]) / 2.;
    CHKERRQ(EG_evaluate(face, avgUV, cntrPnt));

    coords[(numVertices + numEdges + bodyFaceIndexStart + id - 1)*cdim + 0] = cntrPnt[0];
    coords[(numVertices + numEdges + bodyFaceIndexStart + id - 1)*cdim + 1] = cntrPnt[1];
    coords[(numVertices + numEdges + bodyFaceIndexStart + id - 1)*cdim + 2] = cntrPnt[2];
    }
    EG_free(fobjs);

    // Define Cells :: Note - This could be incorporated in the Face Midpoint Vertices Loop but was kept separate for clarity
    CHKERRQ(EG_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
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

    CHKERRQ(EG_getTopology(face, &geom, &oclass, &mtype, NULL, &Nl, &lobjs, &lSenses));

      PetscCheckFalse(Nl > 1,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face has %d Loops. Can only handle Faces with 1 Loop. Please use --dm_plex_egads_with_tess = 1 Option", Nl);
    for (int l = 0; l < Nl; ++l) {
          ego      loop = lobjs[l];

          CHKERRQ(EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &eobjs, &eSenses));
      for (int e = 0; e < Ne; ++e) {
        ego     edge = eobjs[e];
        int     eid, eOffset;

        CHKERRQ(EG_getInfo(edge, &oclass, &mtype, &topRef, &prev, &next));
      eid = EG_indexBodyTopo(body, edge);
        if (mtype == DEGENERATE) { continue; }

        // get relative offset from globalEdgeID Vector
        CHKERRQ(PetscHMapIFind(edgeMap, bodyEdgeGlobalIndexStart + eid - 1, &EMiter, &EMfound));
          PetscCheck(EMfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "Edge %d of Body %d not found in edgeMap. Global Edge ID :: %d", eid, b, bodyEdgeGlobalIndexStart + eid - 1);
          CHKERRQ(PetscHMapIGet(edgeMap, bodyEdgeGlobalIndexStart + eid - 1, &eOffset));

      midPntID = numVertices + bodyEdgeIndexStart + eOffset - 1;

        CHKERRQ(EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));

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
  CHKERRQ(DMPlexCreateFromCellListPetsc(PETSC_COMM_WORLD, dim, numCells, numPoints, numCorners, PETSC_TRUE, cells, cdim, coords, &dm));
  CHKERRQ(PetscFree2(coords, cells));
  CHKERRQ(PetscInfo(dm, " Total Number of Unique Cells    = %D \n", numCells));
  CHKERRQ(PetscInfo(dm, " Total Number of Unique Vertices = %D \n", numVertices));

  // Embed EGADS model in DM
  {
    PetscContainer modelObj, contextObj;

    CHKERRQ(PetscContainerCreate(PETSC_COMM_SELF, &modelObj));
    CHKERRQ(PetscContainerSetPointer(modelObj, model));
    CHKERRQ(PetscObjectCompose((PetscObject) dm, "EGADS Model", (PetscObject) modelObj));
    CHKERRQ(PetscContainerDestroy(&modelObj));

    CHKERRQ(PetscContainerCreate(PETSC_COMM_SELF, &contextObj));
    CHKERRQ(PetscContainerSetPointer(contextObj, context));
    CHKERRQ(PetscContainerSetUserDestroy(contextObj, DMPlexEGADSDestroy_Private));
    CHKERRQ(PetscObjectCompose((PetscObject) dm, "EGADS Context", (PetscObject) contextObj));
    CHKERRQ(PetscContainerDestroy(&contextObj));
  }
  // Label points
  PetscInt   nStart, nEnd;

  CHKERRQ(DMCreateLabel(dm, "EGADS Body ID"));
  CHKERRQ(DMGetLabel(dm, "EGADS Body ID", &bodyLabel));
  CHKERRQ(DMCreateLabel(dm, "EGADS Face ID"));
  CHKERRQ(DMGetLabel(dm, "EGADS Face ID", &faceLabel));
  CHKERRQ(DMCreateLabel(dm, "EGADS Edge ID"));
  CHKERRQ(DMGetLabel(dm, "EGADS Edge ID", &edgeLabel));
  CHKERRQ(DMCreateLabel(dm, "EGADS Vertex ID"));
  CHKERRQ(DMGetLabel(dm, "EGADS Vertex ID", &vertexLabel));

  CHKERRQ(DMPlexGetHeightStratum(dm, 2, &nStart, &nEnd));

  cellCntr = 0;
  for (b = 0; b < nbodies; ++b) {
    ego             body = bodies[b];
  int             Nv, Ne, Nf;
  PetscInt        bodyVertexIndexStart, bodyEdgeIndexStart, bodyEdgeGlobalIndexStart, bodyFaceIndexStart;
  PetscHashIter   BViter, BEiter, BEGiter, BFiter, EMiter;
  PetscBool       BVfound, BEfound, BEGfound, BFfound, EMfound;

  CHKERRQ(PetscHMapIFind(bodyVertexMap, b, &BViter, &BVfound));
  PetscCheck(BVfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %d not found in bodyVertexMap", b);
  CHKERRQ(PetscHMapIGet(bodyVertexMap, b, &bodyVertexIndexStart));

  CHKERRQ(PetscHMapIFind(bodyEdgeMap, b, &BEiter, &BEfound));
  PetscCheck(BEfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %d not found in bodyEdgeMap", b);
  CHKERRQ(PetscHMapIGet(bodyEdgeMap, b, &bodyEdgeIndexStart));

    CHKERRQ(PetscHMapIFind(bodyFaceMap, b, &BFiter, &BFfound));
  PetscCheck(BFfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %d not found in bodyFaceMap", b);
  CHKERRQ(PetscHMapIGet(bodyFaceMap, b, &bodyFaceIndexStart));

    CHKERRQ(PetscHMapIFind(bodyEdgeGlobalMap, b, &BEGiter, &BEGfound));
    PetscCheck(BEGfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %d not found in bodyEdgeGlobalMap", b);
    CHKERRQ(PetscHMapIGet(bodyEdgeGlobalMap, b, &bodyEdgeGlobalIndexStart));

  CHKERRQ(EG_getBodyTopos(body, NULL, FACE,  &Nf, &fobjs));
  for (int f = 0; f < Nf; ++f) {
    ego   face = fobjs[f];
      int   fID, Nl;

    fID  = EG_indexBodyTopo(body, face);

    CHKERRQ(EG_getBodyTopos(body, face, LOOP, &Nl, &lobjs));
    for (int l = 0; l < Nl; ++l) {
        ego  loop = lobjs[l];
    int  lid;

    lid  = EG_indexBodyTopo(body, loop);
      PetscCheckFalse(Nl > 1,PETSC_COMM_SELF, PETSC_ERR_SUP, "Loop %d has %d > 1 faces, which is not supported", lid, Nf);

    CHKERRQ(EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &eobjs, &eSenses));
    for (int e = 0; e < Ne; ++e) {
      ego     edge = eobjs[e];
      int     eid, eOffset;

      // Skip DEGENERATE Edges
      CHKERRQ(EG_getInfo(edge, &oclass, &mtype, &topRef, &prev, &next));
      if (mtype == DEGENERATE) {continue;}
      eid = EG_indexBodyTopo(body, edge);

      // get relative offset from globalEdgeID Vector
      CHKERRQ(PetscHMapIFind(edgeMap, bodyEdgeGlobalIndexStart + eid - 1, &EMiter, &EMfound));
      PetscCheck(EMfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "Edge %d of Body %d not found in edgeMap. Global Edge ID :: %d", eid, b, bodyEdgeGlobalIndexStart + eid - 1);
      CHKERRQ(PetscHMapIGet(edgeMap, bodyEdgeGlobalIndexStart + eid - 1, &eOffset));

      CHKERRQ(EG_getBodyTopos(body, edge, NODE, &Nv, &nobjs));
      for (int v = 0; v < Nv; ++v){
        ego vertex = nobjs[v];
        int vID;

        vID = EG_indexBodyTopo(body, vertex);
        CHKERRQ(DMLabelSetValue(bodyLabel, nStart + bodyVertexIndexStart + vID - 1, b));
        CHKERRQ(DMLabelSetValue(vertexLabel, nStart + bodyVertexIndexStart + vID - 1, vID));
      }
      EG_free(nobjs);

      CHKERRQ(DMLabelSetValue(bodyLabel, nStart + numVertices + bodyEdgeIndexStart + eOffset - 1, b));
      CHKERRQ(DMLabelSetValue(edgeLabel, nStart + numVertices + bodyEdgeIndexStart + eOffset - 1, eid));

      // Define Cell faces
      for (int jj = 0; jj < 2; ++jj){
        CHKERRQ(DMLabelSetValue(bodyLabel, cellCntr, b));
        CHKERRQ(DMLabelSetValue(faceLabel, cellCntr, fID));
        CHKERRQ(DMPlexGetCone(dm, cellCntr, &cone));

        CHKERRQ(DMLabelSetValue(bodyLabel, cone[0], b));
        CHKERRQ(DMLabelSetValue(faceLabel, cone[0], fID));

        CHKERRQ(DMLabelSetValue(bodyLabel, cone[1], b));
        CHKERRQ(DMLabelSetValue(edgeLabel, cone[1], eid));

       CHKERRQ(DMLabelSetValue(bodyLabel, cone[2], b));
       CHKERRQ(DMLabelSetValue(faceLabel, cone[2], fID));

       cellCntr = cellCntr + 1;
      }
    }
    }
    EG_free(lobjs);

    CHKERRQ(DMLabelSetValue(bodyLabel, nStart + numVertices + numEdges + bodyFaceIndexStart + fID - 1, b));
    CHKERRQ(DMLabelSetValue(faceLabel, nStart + numVertices + numEdges + bodyFaceIndexStart + fID - 1, fID));
  }
  EG_free(fobjs);
  }

  CHKERRQ(PetscHMapIDestroy(&edgeMap));
  CHKERRQ(PetscHMapIDestroy(&bodyIndexMap));
  CHKERRQ(PetscHMapIDestroy(&bodyVertexMap));
  CHKERRQ(PetscHMapIDestroy(&bodyEdgeMap));
  CHKERRQ(PetscHMapIDestroy(&bodyEdgeGlobalMap));
  CHKERRQ(PetscHMapIDestroy(&bodyFaceMap));

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
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  if (!rank) {
    // ---------------------------------------------------------------------------------------------------
    // Generate Petsc Plex from EGADSlite created Tessellation of geometry
    // ---------------------------------------------------------------------------------------------------

  // Caluculate cell and vertex sizes
  CHKERRQ(EG_getTopology(model, &geom, &oclass, &mtype, NULL, &nbodies, &bodies, &senses));

  CHKERRQ(PetscHMapICreate(&pointIndexStartMap));
  CHKERRQ(PetscHMapICreate(&triIndexStartMap));
  CHKERRQ(PetscHMapICreate(&pTypeLabelMap));
  CHKERRQ(PetscHMapICreate(&pIndexLabelMap));
  CHKERRQ(PetscHMapICreate(&pBodyIndexLabelMap));
  CHKERRQ(PetscHMapICreate(&triFaceIDLabelMap));
  CHKERRQ(PetscHMapICreate(&triBodyIDLabelMap));

  /* Create Tessellation of Bodies */
  ego tessArray[nbodies];

  for (b = 0; b < nbodies; ++b) {
    ego             body = bodies[b];
    double          params[3] = {0.0, 0.0, 0.0};    // Parameters for Tessellation
    int             Nf, bodyNumPoints = 0, bodyNumTris = 0;
    PetscHashIter   PISiter, TISiter;
    PetscBool       PISfound, TISfound;

    /* Store Start Index for each Body's Point and Tris */
    CHKERRQ(PetscHMapIFind(pointIndexStartMap, b, &PISiter, &PISfound));
    CHKERRQ(PetscHMapIFind(triIndexStartMap, b, &TISiter, &TISfound));

    if (!PISfound)  CHKERRQ(PetscHMapISet(pointIndexStartMap, b, totalNumPoints));
    if (!TISfound)  CHKERRQ(PetscHMapISet(triIndexStartMap, b, totalNumTris));

    /* Calculate Tessellation parameters based on Bounding Box */
    /* Get Bounding Box Dimensions of the BODY */
    CHKERRQ(EG_getBoundingBox(body, boundBox));
    tessSize = boundBox[3] - boundBox[0];
    if (tessSize < boundBox[4] - boundBox[1]) tessSize = boundBox[4] - boundBox[1];
    if (tessSize < boundBox[5] - boundBox[2]) tessSize = boundBox[5] - boundBox[2];

    // TODO :: May want to give users tessellation parameter options //
    params[0] = 0.0250 * tessSize;
    params[1] = 0.0075 * tessSize;
    params[2] = 15.0;

    CHKERRQ(EG_makeTessBody(body, params, &tessArray[b]));

    CHKERRQ(EG_getBodyTopos(body, NULL, FACE,  &Nf, &fobjs));

    for (int f = 0; f < Nf; ++f) {
      ego             face = fobjs[f];
    int             len, fID, ntris;
    const int      *ptype, *pindex, *ptris, *ptric;
    const double   *pxyz, *puv;

    // Get Face ID //
    fID = EG_indexBodyTopo(body, face);

    // Checkout the Surface Tessellation //
    CHKERRQ(EG_getTessFace(tessArray[b], fID, &len, &pxyz, &puv, &ptype, &pindex, &ntris, &ptris, &ptric));

    // Determine total number of triangle cells in the tessellation //
    bodyNumTris += (int) ntris;

    // Check out the point index and coordinate //
    for (int p = 0; p < len; ++p) {
        int global;

        CHKERRQ(EG_localToGlobal(tessArray[b], fID, p+1, &global));

      // Determine the total number of points in the tessellation //
        bodyNumPoints = PetscMax(bodyNumPoints, global);
    }
    }
    EG_free(fobjs);

    totalNumPoints += bodyNumPoints;
    totalNumTris += bodyNumTris;
    }
  //}  - Original End of (!rank)

  dim = 2;
  cdim = 3;
  numCorners = 3;
  //PetscInt counter = 0;

  /* NEED TO DEFINE MATRICES/VECTORS TO STORE GEOM REFERENCE DATA   */
  /* Fill in below and use to define DMLabels after DMPlex creation */
  CHKERRQ(PetscMalloc2(totalNumPoints*cdim, &coords, totalNumTris*numCorners, &cells));

  for (b = 0; b < nbodies; ++b) {
  ego             body = bodies[b];
  int             Nf;
  PetscInt        pointIndexStart;
  PetscHashIter   PISiter;
  PetscBool       PISfound;

  CHKERRQ(PetscHMapIFind(pointIndexStartMap, b, &PISiter, &PISfound));
  PetscCheck(PISfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %d not found in pointIndexStartMap", b);
  CHKERRQ(PetscHMapIGet(pointIndexStartMap, b, &pointIndexStart));

  CHKERRQ(EG_getBodyTopos(body, NULL, FACE,  &Nf, &fobjs));

  for (int f = 0; f < Nf; ++f) {
    /* Get Face Object */
    ego              face = fobjs[f];
    int              len, fID, ntris;
    const int       *ptype, *pindex, *ptris, *ptric;
    const double    *pxyz, *puv;

    /* Get Face ID */
    fID = EG_indexBodyTopo(body, face);

    /* Checkout the Surface Tessellation */
    CHKERRQ(EG_getTessFace(tessArray[b], fID, &len, &pxyz, &puv, &ptype, &pindex, &ntris, &ptris, &ptric));

    /* Check out the point index and coordinate */
    for (int p = 0; p < len; ++p) {
    int              global;
    PetscHashIter    PTLiter, PILiter, PBLiter;
    PetscBool        PTLfound, PILfound, PBLfound;

    CHKERRQ(EG_localToGlobal(tessArray[b], fID, p+1, &global));

    /* Set the coordinates array for DAG */
    coords[((global-1+pointIndexStart)*3) + 0] = pxyz[(p*3)+0];
    coords[((global-1+pointIndexStart)*3) + 1] = pxyz[(p*3)+1];
    coords[((global-1+pointIndexStart)*3) + 2] = pxyz[(p*3)+2];

    /* Store Geometry Label Information for DMLabel assignment later */
    CHKERRQ(PetscHMapIFind(pTypeLabelMap, global-1+pointIndexStart, &PTLiter, &PTLfound));
    CHKERRQ(PetscHMapIFind(pIndexLabelMap, global-1+pointIndexStart, &PILiter, &PILfound));
    CHKERRQ(PetscHMapIFind(pBodyIndexLabelMap, global-1+pointIndexStart, &PBLiter, &PBLfound));

    if (!PTLfound) CHKERRQ(PetscHMapISet(pTypeLabelMap, global-1+pointIndexStart, ptype[p]));
    if (!PILfound) CHKERRQ(PetscHMapISet(pIndexLabelMap, global-1+pointIndexStart, pindex[p]));
    if (!PBLfound) CHKERRQ(PetscHMapISet(pBodyIndexLabelMap, global-1+pointIndexStart, b));

    if (ptype[p] < 0) CHKERRQ(PetscHMapISet(pIndexLabelMap, global-1+pointIndexStart, fID));
    }

    for (int t = 0; t < (int) ntris; ++t){
    int           global, globalA, globalB;
    PetscHashIter TFLiter, TBLiter;
    PetscBool     TFLfound, TBLfound;

    CHKERRQ(EG_localToGlobal(tessArray[b], fID, ptris[(t*3) + 0], &global));
    cells[(counter*3) +0] = global-1+pointIndexStart;

    CHKERRQ(EG_localToGlobal(tessArray[b], fID, ptris[(t*3) + 1], &globalA));
    cells[(counter*3) +1] = globalA-1+pointIndexStart;

    CHKERRQ(EG_localToGlobal(tessArray[b], fID, ptris[(t*3) + 2], &globalB));
    cells[(counter*3) +2] = globalB-1+pointIndexStart;

    CHKERRQ(PetscHMapIFind(triFaceIDLabelMap, counter, &TFLiter, &TFLfound));
        CHKERRQ(PetscHMapIFind(triBodyIDLabelMap, counter, &TBLiter, &TBLfound));

    if (!TFLfound)  CHKERRQ(PetscHMapISet(triFaceIDLabelMap, counter, fID));
        if (!TBLfound)  CHKERRQ(PetscHMapISet(triBodyIDLabelMap, counter, b));

    counter += 1;
    }
  }
  EG_free(fobjs);
  }
}

  //Build DMPlex
  CHKERRQ(DMPlexCreateFromCellListPetsc(PETSC_COMM_WORLD, dim, totalNumTris, totalNumPoints, numCorners, PETSC_TRUE, cells, cdim, coords, &dm));
  CHKERRQ(PetscFree2(coords, cells));

  // Embed EGADS model in DM
  {
    PetscContainer modelObj, contextObj;

    CHKERRQ(PetscContainerCreate(PETSC_COMM_SELF, &modelObj));
    CHKERRQ(PetscContainerSetPointer(modelObj, model));
    CHKERRQ(PetscObjectCompose((PetscObject) dm, "EGADS Model", (PetscObject) modelObj));
    CHKERRQ(PetscContainerDestroy(&modelObj));

    CHKERRQ(PetscContainerCreate(PETSC_COMM_SELF, &contextObj));
    CHKERRQ(PetscContainerSetPointer(contextObj, context));
    CHKERRQ(PetscContainerSetUserDestroy(contextObj, DMPlexEGADSDestroy_Private));
    CHKERRQ(PetscObjectCompose((PetscObject) dm, "EGADS Context", (PetscObject) contextObj));
    CHKERRQ(PetscContainerDestroy(&contextObj));
  }

  // Label Points
  CHKERRQ(DMCreateLabel(dm, "EGADS Body ID"));
  CHKERRQ(DMGetLabel(dm, "EGADS Body ID", &bodyLabel));
  CHKERRQ(DMCreateLabel(dm, "EGADS Face ID"));
  CHKERRQ(DMGetLabel(dm, "EGADS Face ID", &faceLabel));
  CHKERRQ(DMCreateLabel(dm, "EGADS Edge ID"));
  CHKERRQ(DMGetLabel(dm, "EGADS Edge ID", &edgeLabel));
  CHKERRQ(DMCreateLabel(dm, "EGADS Vertex ID"));
  CHKERRQ(DMGetLabel(dm, "EGADS Vertex ID", &vertexLabel));

   /* Get Number of DAG Nodes at each level */
  int   fStart, fEnd, eStart, eEnd, nStart, nEnd;

  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &fStart, &fEnd));
  CHKERRQ(DMPlexGetHeightStratum(dm, 1, &eStart, &eEnd));
  CHKERRQ(DMPlexGetHeightStratum(dm, 2, &nStart, &nEnd));

  /* Set DMLabels for NODES */
  for (int n = nStart; n < nEnd; ++n) {
    int             pTypeVal, pIndexVal, pBodyVal;
    PetscHashIter   PTLiter, PILiter, PBLiter;
    PetscBool       PTLfound, PILfound, PBLfound;

    //Converted to Hash Tables
    CHKERRQ(PetscHMapIFind(pTypeLabelMap, n - nStart, &PTLiter, &PTLfound));
    PetscCheck(PTLfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "DAG Point %d not found in pTypeLabelMap", n);
    CHKERRQ(PetscHMapIGet(pTypeLabelMap, n - nStart, &pTypeVal));

    CHKERRQ(PetscHMapIFind(pIndexLabelMap, n - nStart, &PILiter, &PILfound));
    PetscCheck(PILfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "DAG Point %d not found in pIndexLabelMap", n);
    CHKERRQ(PetscHMapIGet(pIndexLabelMap, n - nStart, &pIndexVal));

    CHKERRQ(PetscHMapIFind(pBodyIndexLabelMap, n - nStart, &PBLiter, &PBLfound));
    PetscCheck(PBLfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "DAG Point %d not found in pBodyLabelMap", n);
    CHKERRQ(PetscHMapIGet(pBodyIndexLabelMap, n - nStart, &pBodyVal));

    CHKERRQ(DMLabelSetValue(bodyLabel, n, pBodyVal));
    if (pTypeVal == 0) CHKERRQ(DMLabelSetValue(vertexLabel, n, pIndexVal));
    if (pTypeVal >  0) CHKERRQ(DMLabelSetValue(edgeLabel, n, pIndexVal));
    if (pTypeVal <  0) CHKERRQ(DMLabelSetValue(faceLabel, n, pIndexVal));
  }

  /* Set DMLabels for Edges - Based on the DMLabels of the EDGE's NODES */
  for (int e = eStart; e < eEnd; ++e) {
  int    bodyID_0, vertexID_0, vertexID_1, edgeID_0, edgeID_1, faceID_0, faceID_1;

  CHKERRQ(DMPlexGetCone(dm, e, &cone));
  CHKERRQ(DMLabelGetValue(bodyLabel, cone[0], &bodyID_0));    // Do I need to check the other end?
  CHKERRQ(DMLabelGetValue(vertexLabel, cone[0], &vertexID_0));
  CHKERRQ(DMLabelGetValue(vertexLabel, cone[1], &vertexID_1));
  CHKERRQ(DMLabelGetValue(edgeLabel, cone[0], &edgeID_0));
  CHKERRQ(DMLabelGetValue(edgeLabel, cone[1], &edgeID_1));
  CHKERRQ(DMLabelGetValue(faceLabel, cone[0], &faceID_0));
  CHKERRQ(DMLabelGetValue(faceLabel, cone[1], &faceID_1));

  CHKERRQ(DMLabelSetValue(bodyLabel, e, bodyID_0));

  if (edgeID_0 == edgeID_1) CHKERRQ(DMLabelSetValue(edgeLabel, e, edgeID_0));
  else if (vertexID_0 > 0 && edgeID_1 > 0) CHKERRQ(DMLabelSetValue(edgeLabel, e, edgeID_1));
  else if (vertexID_1 > 0 && edgeID_0 > 0) CHKERRQ(DMLabelSetValue(edgeLabel, e, edgeID_0));
  else { /* Do Nothing */ }
  }

  /* Set DMLabels for Cells */
  for (int f = fStart; f < fEnd; ++f){
  int             edgeID_0;
  PetscInt        triBodyVal, triFaceVal;
  PetscHashIter   TFLiter, TBLiter;
  PetscBool       TFLfound, TBLfound;

    // Convert to Hash Table
  CHKERRQ(PetscHMapIFind(triFaceIDLabelMap, f - fStart, &TFLiter, &TFLfound));
  PetscCheck(TFLfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "DAG Point %d not found in triFaceIDLabelMap", f);
  CHKERRQ(PetscHMapIGet(triFaceIDLabelMap, f - fStart, &triFaceVal));

  CHKERRQ(PetscHMapIFind(triBodyIDLabelMap, f - fStart, &TBLiter, &TBLfound));
  PetscCheck(TBLfound,PETSC_COMM_SELF, PETSC_ERR_SUP, "DAG Point %d not found in triBodyIDLabelMap", f);
    CHKERRQ(PetscHMapIGet(triBodyIDLabelMap, f - fStart, &triBodyVal));

  CHKERRQ(DMLabelSetValue(bodyLabel, f, triBodyVal));
  CHKERRQ(DMLabelSetValue(faceLabel, f, triFaceVal));

  /* Finish Labeling previously unlabeled DMPlex Edges - Assumes Triangular Cell (3 Edges Max) */
  CHKERRQ(DMPlexGetCone(dm, f, &cone));

  for (int jj = 0; jj < 3; ++jj) {
    CHKERRQ(DMLabelGetValue(edgeLabel, cone[jj], &edgeID_0));

    if (edgeID_0 < 0) {
    CHKERRQ(DMLabelSetValue(bodyLabel, cone[jj], triBodyVal));
      CHKERRQ(DMLabelSetValue(faceLabel, cone[jj], triFaceVal));
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

.seealso: DMPLEX, DMCreate(), DMPlexCreateEGADS()
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
  CHKERRQ(PetscObjectQuery((PetscObject) dm, "EGADS Model", (PetscObject *) &modelObj));
  if (!modelObj) PetscFunctionReturn(0);
  CHKERRQ(DMGetCoordinateDim(dm, &cdim));
  CHKERRQ(DMGetCoordinateDM(dm, &cdm));
  CHKERRQ(DMGetCoordinatesLocal(dm, &coordinates));
  CHKERRQ(DMGetLabel(dm, "EGADS Body ID", &bodyLabel));
  CHKERRQ(DMGetLabel(dm, "EGADS Face ID", &faceLabel));
  CHKERRQ(DMGetLabel(dm, "EGADS Edge ID", &edgeLabel));
  CHKERRQ(DMGetLabel(dm, "EGADS Vertex ID", &vertexLabel));

  CHKERRQ(PetscContainerGetPointer(modelObj, (void **) &model));
  CHKERRQ(EG_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));

  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(VecGetArrayWrite(coordinates, &coords));
  for (v = vStart; v < vEnd; ++v) {
    PetscScalar *vcoords;

    CHKERRQ(DMLabelGetValue(bodyLabel, v, &bodyID));
    CHKERRQ(DMLabelGetValue(faceLabel, v, &faceID));
    CHKERRQ(DMLabelGetValue(edgeLabel, v, &edgeID));
    CHKERRQ(DMLabelGetValue(vertexLabel, v, &vertexID));

    PetscCheckFalse(bodyID >= Nb,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Body %D is not in [0, %d)", bodyID, Nb);
    body = bodies[bodyID];

    CHKERRQ(DMPlexPointLocalRef(cdm, v, coords, (void *) &vcoords));
    if (edgeID > 0) {
      /* Snap to EDGE at nearest location */
      double params[1];
      CHKERRQ(EG_objectBodyTopo(body, EDGE, edgeID, &edge));
      CHKERRQ(EG_invEvaluate(edge, vcoords, params, result)); // Get (x,y,z) of nearest point on EDGE
      for (d = 0; d < cdim; ++d) vcoords[d] = result[d];
    } else if (faceID > 0) {
      /* Snap to FACE at nearest location */
      double params[2];
      CHKERRQ(EG_objectBodyTopo(body, FACE, faceID, &face));
      CHKERRQ(EG_invEvaluate(face, vcoords, params, result)); // Get (x,y,z) of nearest point on FACE
      for (d = 0; d < cdim; ++d) vcoords[d] = result[d];
    }
  }
  CHKERRQ(VecRestoreArrayWrite(coordinates, &coords));
  /* Clear out global coordinates */
  CHKERRQ(VecDestroy(&dm->coordinates));
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

.seealso: DMPLEX, DMCreate(), DMPlexCreateEGADS(), DMPlexCreateEGADSLiteFromFile()
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
  CHKERRQ(PetscOptionsGetBool(NULL, NULL, "-dm_plex_egads_print_model", &printModel, NULL));
  CHKERRQ(PetscOptionsGetBool(NULL, NULL, "-dm_plex_egads_tess_model", &tessModel, NULL));
  CHKERRQ(PetscOptionsGetBool(NULL, NULL, "-dm_plex_egads_new_model", &newModel, NULL));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
#if defined(PETSC_HAVE_EGADS)
  if (rank == 0) {

    CHKERRQ(EG_open(&context));
    CHKERRQ(EG_loadModel(context, 0, filename, &model));
    if (printModel) CHKERRQ(DMPlexEGADSPrintModel_Internal(model));

  }
  if (tessModel)     CHKERRQ(DMPlexCreateEGADS_Tess_Internal(comm, context, model, dm));
  else if (newModel) CHKERRQ(DMPlexCreateEGADS(comm, context, model, dm));
  else               CHKERRQ(DMPlexCreateEGADS_Internal(comm, context, model, dm));
  PetscFunctionReturn(0);
#else
  SETERRQ(comm, PETSC_ERR_SUP, "This method requires EGADS support. Reconfigure using --download-egads");
#endif
}
