#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/hashmapi.h>

#ifdef PETSC_HAVE_EGADS
#include <egads_lite.h>

PetscErrorCode DMPlexSnapToGeomModel_EGADSLite_Internal(DM dm, PetscInt p, PetscInt dE, ego model, PetscInt bodyID, PetscInt faceID, PetscInt edgeID, const PetscScalar mcoords[], PetscScalar gcoords[])
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
  PetscInt       d;

  PetscFunctionBeginHot;
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinatesLocal));
  PetscCall(EGlite_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));
  PetscCheck(bodyID < Nb,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Body %D is not in [0, %d)", bodyID, Nb);
  body = bodies[bodyID];

  if (edgeID >= 0)      {PetscCall(EGlite_objectBodyTopo(body, EDGE, edgeID, &obj)); Np = 1;}
  else if (faceID >= 0) {PetscCall(EGlite_objectBodyTopo(body, FACE, faceID, &obj)); Np = 2;}
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
  PetscCheck(Nv <= 16,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle %D coordinates associated to point %D", Nv, p);

  /* Correct EGADSlite 2pi bug when calculating nearest point on Periodic Surfaces */
  PetscCall(EGlite_getRange(obj, range, &peri));
  for (v = 0; v < Nv; ++v) {
    PetscCall(EGlite_invEvaluate(obj, &coords[v*dE], &paramsV[v*3], &resultV[v*3]));
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
  PetscCheck(!(params[0] < range[0]) && !(params[0] > range[1]),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point %D had bad interpolation", p);
  PetscCheckFalse(Np > 1 && ((params[1] < range[2]) || (params[1] > range[3])),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point %D had bad interpolation", p);
  /* Put coordinates for new vertex in result[] */
  PetscCall(EGlite_evaluate(obj, params, result));
  for (d = 0; d < dE; ++d) gcoords[d] = result[d];
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexEGADSLiteDestroy_Private(void *context)
{
  if (context) EGlite_close((ego) context);
  return 0;
}

static PetscErrorCode DMPlexCreateEGADSLite_Internal(MPI_Comm comm, ego context, ego model, DM *newdm)
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
  if (!rank) {
    const PetscInt debug = 0;

    /* ---------------------------------------------------------------------------------------------------
    Generate Petsc Plex
      Get all Nodes in model, record coordinates in a correctly formatted array
      Cycle through bodies, cycle through loops, recorde NODE IDs in a correctly formatted array
      We need to uniformly refine the initial geometry to guarantee a valid mesh
    */

    /* Calculate cell and vertex sizes */
    PetscCall(EGlite_getTopology(model, &geom, &oclass, &mtype, NULL, &nbodies, &bodies, &senses));
    PetscCall(PetscHMapICreate(&edgeMap));
    numEdges = 0;
    for (b = 0; b < nbodies; ++b) {
      ego body = bodies[b];
      int id, Nl, l, Nv, v;

      PetscCall(EGlite_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs));
      for (l = 0; l < Nl; ++l) {
        ego loop = lobjs[l];
        int Ner  = 0, Ne, e, Nc;

        PetscCall(EGlite_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses));
        for (e = 0; e < Ne; ++e) {
          ego edge = objs[e];
          int Nv, id;
          PetscHashIter iter;
          PetscBool     found;

          PetscCall(EGlite_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
          if (mtype == DEGENERATE) continue;
          id   = EGlite_indexBodyTopo(body, edge);
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
      PetscCall(EGlite_getBodyTopos(body, NULL, NODE, &Nv, &nobjs));
      for (v = 0; v < Nv; ++v) {
        ego vertex = nobjs[v];

        id = EGlite_indexBodyTopo(body, vertex);
        /* TODO: Instead of assuming contiguous ids, we could use a hash table */
        numVertices = PetscMax(id, numVertices);
      }
      EGlite_free(lobjs);
      EGlite_free(nobjs);
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

      PetscCall(EGlite_getBodyTopos(body, NULL, NODE, &Nv, &nobjs));
      for (v = 0; v < Nv; ++v) {
        ego    vertex = nobjs[v];
        double limits[4];
        int    dummy;

        PetscCall(EGlite_getTopology(vertex, &geom, &oclass, &mtype, limits, &dummy, &mobjs, &senses));
        id   = EGlite_indexBodyTopo(body, vertex);
        coords[(id-1)*cdim+0] = limits[0];
        coords[(id-1)*cdim+1] = limits[1];
        coords[(id-1)*cdim+2] = limits[2];
      }
      EGlite_free(nobjs);
    }
    PetscCall(PetscHMapIClear(edgeMap));
    fOff     = numVertices - newVertices + numEdges;
    numEdges = 0;
    numQuads = 0;
    for (b = 0; b < nbodies; ++b) {
      ego body = bodies[b];
      int Nl, l;

      PetscCall(EGlite_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs));
      for (l = 0; l < Nl; ++l) {
        ego loop = lobjs[l];
        int lid, Ner = 0, Ne, e;

        lid  = EGlite_indexBodyTopo(body, loop);
        PetscCall(EGlite_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses));
        for (e = 0; e < Ne; ++e) {
          ego       edge = objs[e];
          int       eid, Nv;
          PetscHashIter iter;
          PetscBool     found;

          PetscCall(EGlite_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
          if (mtype == DEGENERATE) continue;
          ++Ner;
          eid  = EGlite_indexBodyTopo(body, edge);
          PetscCall(PetscHMapIFind(edgeMap, eid-1, &iter, &found));
          if (!found) {
            PetscInt v = numVertices - newVertices + numEdges;
            double range[4], params[3] = {0., 0., 0.}, result[18];
            int    periodic[2];

            PetscCall(PetscHMapISet(edgeMap, eid-1, numEdges++));
            PetscCall(EGlite_getRange(edge, range, periodic));
            params[0] = 0.5*(range[0] + range[1]);
            PetscCall(EGlite_evaluate(edge, params, result));
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

          PetscCall(EGlite_getBodyTopos(body, loop, FACE, &Nf, &fobjs));
          face = fobjs[0];
          fid  = EGlite_indexBodyTopo(body, face);
          PetscCheck(Nf == 1,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Loop %d has %d faces, instead of 1 (%d)", lid-1, Nf, fid);
          PetscCall(EGlite_getRange(face, range, periodic));
          params[0] = 0.5*(range[0] + range[1]);
          params[1] = 0.5*(range[2] + range[3]);
          PetscCall(EGlite_evaluate(face, params, result));
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

      PetscCall(EGlite_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs));
      for (l = 0; l < Nl; ++l) {
        ego loop = lobjs[l];
        int lid, Ner = 0, Ne, e, nc = 0, c, Nt, t;

        lid  = EGlite_indexBodyTopo(body, loop);
        PetscCall(EGlite_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses));

        for (e = 0; e < Ne; ++e) {
          ego edge = objs[e];
          int points[3];
          int eid, Nv, v, tmp;

          eid  = EGlite_indexBodyTopo(body, edge);
          PetscCall(EGlite_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
          if (mtype == DEGENERATE) continue;
          else                     ++Ner;
          PetscCheck(Nv == 2,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Edge %d has %d vertices != 2", eid, Nv);

          for (v = 0; v < Nv; ++v) {
            ego vertex = nobjs[v];

            id = EGlite_indexBodyTopo(body, vertex);
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
        PetscCheck(nc == 2*Ner,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of corners %D != %D", nc, 2*Ner);
        if (Ner == 4) {cone[nc++] = numVertices - newVertices + numEdges + numQuads++;}
        PetscCheck(nc <= maxCorners,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of corners %D > %D max", nc, maxCorners);
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
            PetscCall(PetscPrintf(PETSC_COMM_SELF, "  LOOP Corner NODEs Triangle %D (", t));
            for (c = 0; c < numCorners; ++c) {
              if (c > 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, ", "));
              PetscCall(PetscPrintf(PETSC_COMM_SELF, "%D", cells[(cOff-Nt+t)*numCorners+c]));
            }
            PetscCall(PetscPrintf(PETSC_COMM_SELF, ")\n"));
          }
        }
      }
      EGlite_free(lobjs);
    }
  }
  PetscCheck(cOff == numCells,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Count of total cells %D != %D previous count", cOff, numCells);
  PetscCall(DMPlexCreateFromCellListPetsc(PETSC_COMM_WORLD, dim, numCells, numVertices, numCorners, PETSC_TRUE, cells, cdim, coords, &dm));
  PetscCall(PetscFree3(coords, cells, cone));
  PetscCall(PetscInfo(dm, " Total Number of Unique Cells    = %D (%D)\n", numCells, newCells));
  PetscCall(PetscInfo(dm, " Total Number of Unique Vertices = %D (%D)\n", numVertices, newVertices));
  /* Embed EGADS model in DM */
  {
    PetscContainer modelObj, contextObj;

    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &modelObj));
    PetscCall(PetscContainerSetPointer(modelObj, model));
    PetscCall(PetscObjectCompose((PetscObject) dm, "EGADSLite Model", (PetscObject) modelObj));
    PetscCall(PetscContainerDestroy(&modelObj));

    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &contextObj));
    PetscCall(PetscContainerSetPointer(contextObj, context));
    PetscCall(PetscContainerSetUserDestroy(contextObj, DMPlexEGADSLiteDestroy_Private));
    PetscCall(PetscObjectCompose((PetscObject) dm, "EGADSLite Context", (PetscObject) contextObj));
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

    PetscCall(EGlite_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs));
    for (l = 0; l < Nl; ++l) {
      ego  loop = lobjs[l];
      ego *fobjs;
      int  lid, Nf, fid, Ner = 0, Ne, e, Nt = 0, t;

      lid  = EGlite_indexBodyTopo(body, loop);
      PetscCall(EGlite_getBodyTopos(body, loop, FACE, &Nf, &fobjs));
      PetscCheck(Nf <= 1,PETSC_COMM_SELF, PETSC_ERR_SUP, "Loop %d has %d > 1 faces, which is not supported", lid, Nf);
      fid  = EGlite_indexBodyTopo(body, fobjs[0]);
      EGlite_free(fobjs);
      PetscCall(EGlite_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses));
      for (e = 0; e < Ne; ++e) {
        ego             edge = objs[e];
        int             eid, Nv, v;
        PetscInt        points[3], support[2], numEdges, edgeNum;
        const PetscInt *edges;

        eid  = EGlite_indexBodyTopo(body, edge);
        PetscCall(EGlite_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
        if (mtype == DEGENERATE) continue;
        else                     ++Ner;
        for (v = 0; v < Nv; ++v) {
          ego vertex = nobjs[v];

          id   = EGlite_indexBodyTopo(body, vertex);
          PetscCall(DMLabelSetValue(edgeLabel, numCells + id-1, eid));
          points[v*2] = numCells + id-1;
        }
        PetscCall(PetscHMapIGet(edgeMap, eid-1, &edgeNum));
        points[1] = numCells + numVertices - newVertices + edgeNum;

        PetscCall(DMLabelSetValue(edgeLabel, points[1], eid));
        support[0] = points[0];
        support[1] = points[1];
        PetscCall(DMPlexGetJoin(dm, 2, support, &numEdges, &edges));
        PetscCheck(numEdges == 1,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Vertices (%D, %D) should only bound 1 edge, not %D", support[0], support[1], numEdges);
        PetscCall(DMLabelSetValue(edgeLabel, edges[0], eid));
        PetscCall(DMPlexRestoreJoin(dm, 2, support, &numEdges, &edges));
        support[0] = points[1];
        support[1] = points[2];
        PetscCall(DMPlexGetJoin(dm, 2, support, &numEdges, &edges));
        PetscCheck(numEdges == 1,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Vertices (%D, %D) should only bound 1 edge, not %D", support[0], support[1], numEdges);
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
    EGlite_free(lobjs);
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

static PetscErrorCode DMPlexEGADSLitePrintModel_Internal(ego model)
{
  ego geom, *bodies, *objs, *nobjs, *mobjs, *lobjs;
  int oclass, mtype, *senses;
  int Nb, b;

  PetscFunctionBeginUser;
  /* test bodyTopo functions */
  PetscCall(EGlite_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, " Number of BODIES (nbodies): %d \n", Nb));

  for (b = 0; b < Nb; ++b) {
    ego body = bodies[b];
    int id, Nsh, Nf, Nl, l, Ne, e, Nv, v;

    /* Output Basic Model Topology */
    PetscCall(EGlite_getBodyTopos(body, NULL, SHELL, &Nsh, &objs));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "   Number of SHELLS: %d \n", Nsh));
    EGlite_free(objs);

    PetscCall(EGlite_getBodyTopos(body, NULL, FACE,  &Nf, &objs));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "   Number of FACES: %d \n", Nf));
    EGlite_free(objs);

    PetscCall(EGlite_getBodyTopos(body, NULL, LOOP,  &Nl, &lobjs));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "   Number of LOOPS: %d \n", Nl));

    PetscCall(EGlite_getBodyTopos(body, NULL, EDGE,  &Ne, &objs));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "   Number of EDGES: %d \n", Ne));
    EGlite_free(objs);

    PetscCall(EGlite_getBodyTopos(body, NULL, NODE,  &Nv, &objs));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "   Number of NODES: %d \n", Nv));
    EGlite_free(objs);

    for (l = 0; l < Nl; ++l) {
      ego loop = lobjs[l];

      id   = EGlite_indexBodyTopo(body, loop);
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "          LOOP ID: %d\n", id));

      /* Get EDGE info which associated with the current LOOP */
      PetscCall(EGlite_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses));

      for (e = 0; e < Ne; ++e) {
        ego    edge      = objs[e];
        double range[4]  = {0., 0., 0., 0.};
        double point[3]  = {0., 0., 0.};
        double params[3] = {0., 0., 0.};
        double result[18];
        int    peri;

        PetscCall(EGlite_indexBodyTopo(body, edge));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "            EDGE ID: %d (%d)\n", id, e));

        PetscCall(EGlite_getRange(edge, range, &peri));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Range = %lf, %lf, %lf, %lf \n", range[0], range[1], range[2], range[3]));

        /* Get NODE info which associated with the current EDGE */
        PetscCall(EGlite_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
        if (mtype == DEGENERATE) {
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "  EDGE %d is DEGENERATE \n", id));
        } else {
          params[0] = range[0];
          PetscCall(EGlite_evaluate(edge, params, result));
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "   between (%lf, %lf, %lf)", result[0], result[1], result[2]));
          params[0] = range[1];
          PetscCall(EGlite_evaluate(edge, params, result));
          PetscCall(PetscPrintf(PETSC_COMM_SELF, " and (%lf, %lf, %lf)\n", result[0], result[1], result[2]));
        }

        for (v = 0; v < Nv; ++v) {
          ego    vertex = nobjs[v];
          double limits[4];
          int    dummy;

          PetscCall(EGlite_getTopology(vertex, &geom, &oclass, &mtype, limits, &dummy, &mobjs, &senses));
          PetscCall(EGlite_indexBodyTopo(body, vertex));
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "              NODE ID: %d \n", id));
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "                 (x, y, z) = (%lf, %lf, %lf) \n", limits[0], limits[1], limits[2]));

          point[0] = point[0] + limits[0];
          point[1] = point[1] + limits[1];
          point[2] = point[2] + limits[2];
        }
      }
    }
    EGlite_free(lobjs);
  }
  PetscFunctionReturn(0);
}
#endif

/*@C
  DMPlexCreateEGADSLiteFromFile - Create a DMPlex mesh from an EGADSLite file.

  Collective

  Input Parameters:
+ comm     - The MPI communicator
- filename - The name of the EGADSLite file

  Output Parameter:
. dm       - The DM object representing the mesh

  Level: beginner

.seealso: DMPLEX, DMCreate(), DMPlexCreateEGADS(), DMPlexCreateEGADSFromFile()
@*/
PetscErrorCode DMPlexCreateEGADSLiteFromFile(MPI_Comm comm, const char filename[], DM *dm)
{
  PetscMPIInt    rank;
#if defined(PETSC_HAVE_EGADS)
  ego            context= NULL, model = NULL;
#endif
  PetscBool      printModel = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidCharPointer(filename, 2);
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-dm_plex_egads_print_model", &printModel, NULL));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
#if defined(PETSC_HAVE_EGADS)
  if (!rank) {

    PetscCall(EGlite_open(&context));
    PetscCall(EGlite_loadModel(context, 0, filename, &model));
    if (printModel) PetscCall(DMPlexEGADSLitePrintModel_Internal(model));

  }
  PetscCall(DMPlexCreateEGADSLite_Internal(comm, context, model, dm));
  PetscFunctionReturn(0);
#else
  SETERRQ(comm, PETSC_ERR_SUP, "This method requires EGADSLite support. Reconfigure using --download-egads");
#endif
}
