static const char help[] = "Test of EGADSLite CAD functionality";

#include <petscdmplex.h>
#include <petsc/private/hashmapi.h>

#include <egads.h>
#include <petsc.h>

typedef struct {
  char filename[PETSC_MAX_PATH_LEN];
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->filename[0] = '\0';

  ierr = PetscOptionsBegin(comm, "", "EGADSPlex Problem Options", "EGADSLite");CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The EGADSLite file", "ex9.c", options->filename, options->filename, sizeof(options->filename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode PrintEGADS(ego model)
{
  ego geom, *bodies, *objs, *nobjs, *mobjs, *lobjs;
  int oclass, mtype, *senses;
  int Nb, b;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* test bodyTopo functions */
  ierr = EG_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, " Number of BODIES (nbodies): %d \n", Nb);CHKERRQ(ierr);

  for (b = 0; b < Nb; ++b) {
    ego body = bodies[b];
    int id, Nsh, Nf, Nl, l, Ne, e, Nv, v;

    /* Output Basic Model Topology */
    ierr = EG_getBodyTopos(body, NULL, SHELL, &Nsh, &objs);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "   Number of SHELLS: %d \n", Nsh);CHKERRQ(ierr);
    EG_free(objs);

    ierr = EG_getBodyTopos(body, NULL, FACE,  &Nf, &objs);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "   Number of FACES: %d \n", Nf);CHKERRQ(ierr);
    EG_free(objs);

    ierr = EG_getBodyTopos(body, NULL, LOOP,  &Nl, &lobjs);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "   Number of LOOPS: %d \n", Nl);CHKERRQ(ierr);

    ierr = EG_getBodyTopos(body, NULL, EDGE,  &Ne, &objs);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "   Number of EDGES: %d \n", Ne);CHKERRQ(ierr);
    EG_free(objs);

    ierr = EG_getBodyTopos(body, NULL, NODE,  &Nv, &objs);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "   Number of NODES: %d \n", Nv);CHKERRQ(ierr);
    EG_free(objs);

    for (l = 0; l < Nl; ++l) {
      ego loop = lobjs[l];

      id   = EG_indexBodyTopo(body, loop);
      ierr = PetscPrintf(PETSC_COMM_SELF, "          LOOP ID: %d\n", id);CHKERRQ(ierr);

      /* Get EDGE info which associated with the current LOOP */
      ierr = EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses);CHKERRQ(ierr);

      for (e = 0; e < Ne; ++e) {
        ego edge = objs[e];

        id = EG_indexBodyTopo(body, edge);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_SELF, "            EDGE ID: %d\n", id);CHKERRQ(ierr);

        double range[4] = {0., 0., 0., 0.};
        double point[3] = {0., 0., 0.};
        int    peri;

        ierr = EG_getRange(objs[e], range, &peri);
        ierr = PetscPrintf(PETSC_COMM_SELF, " Range = %lf, %lf, %lf, %lf \n", range[0], range[1], range[2], range[3]);

        /* Get NODE info which associated with the current EDGE */
        ierr = EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses);CHKERRQ(ierr);

        for (v = 0; v < Nv; ++v) {
          ego    vertex = nobjs[v];
          double limits[4];
          int    dummy;

          ierr = EG_getTopology(vertex, &geom, &oclass, &mtype, limits, &dummy, &mobjs, &senses);CHKERRQ(ierr);
          id   = EG_indexBodyTopo(body, vertex);
          ierr = PetscPrintf(PETSC_COMM_SELF, "              NODE ID: %d \n", id);CHKERRQ(ierr);
          ierr = PetscPrintf(PETSC_COMM_SELF, "                 (x, y, z) = (%lf, %lf, %lf) \n", limits[0], limits[1], limits[2]);

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

int main(int argc, char *argv[])
{
  DMLabel        bodyLabel, faceLabel, edgeLabel;
  PetscInt       cStart, cEnd, c;
  /* EGADSLite variables */
  ego            context, model, geom, *bodies, *objs, *nobjs, *mobjs, *lobjs;
  int            oclass, mtype, nbodies, *senses;
  int            b;
  /* PETSc variables */
  DM             dm;
  PetscHMapI     edgeMap = NULL;
  PetscInt       dim = -1, cdim = -1, numCorners = 0, maxCorners = 0, numVertices = 0, newVertices = 0, numEdges = 0, numCells = 0, newCells = 0, numQuads = 0, cOff = 0, fOff = 0;
  PetscInt      *cells  = NULL, *cone = NULL;
  PetscReal     *coords = NULL;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  AppCtx         ctx;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &ctx);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    /* Open EGADs file and load EGADs model data */
    ierr = EG_open(&context);CHKERRQ(ierr);
    ierr = EG_loadModel(context, 0, ctx.filename, &model);CHKERRQ(ierr);

    ierr = PrintEGADS(model);CHKERRQ(ierr);

    /* ---------------------------------------------------------------------------------------------------
    Generate Petsc Plex
      Get all Nodes in model, record coordinates in a correctly formatted array
      Cycle through bodies, cycle through loops, recorde NODE IDs in a correctly formatted array
      We need to uniformly refine the initial geometry to guarantee a valid mesh
    */

    /* Calculate cell and vertex sizes */
    ierr = EG_getTopology(model, &geom, &oclass, &mtype, NULL, &nbodies, &bodies, &senses);CHKERRQ(ierr);
    ierr = PetscHMapICreate(&edgeMap);CHKERRQ(ierr);
    numEdges = 0;
    for (b = 0; b < nbodies; ++b) {
      ego body = bodies[b];
      int id, Nl, l, Nv, v;

      ierr = EG_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs);CHKERRQ(ierr);
      for (l = 0; l < Nl; ++l) {
        ego loop = lobjs[l];
        int Ner  = 0, Ne, e, Nc;

        ierr = EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses);CHKERRQ(ierr);
        for (e = 0; e < Ne; ++e) {
          ego edge = objs[e];
          int Nv, id;
          PetscHashIter iter;
          PetscBool     found;

          ierr = EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses);
          if (mtype == DEGENERATE) continue;
          id   = EG_indexBodyTopo(body, edge);CHKERRQ(ierr);
          ierr = PetscHMapIFind(edgeMap, id-1, &iter, &found);CHKERRQ(ierr);
          if (!found) {ierr = PetscHMapISet(edgeMap, id-1, numEdges++);CHKERRQ(ierr);}
          ++Ner;
        }
        if (Ner == 2)      {Nc = 2;}
        else if (Ner == 3) {Nc = 4;}
        else if (Ner == 4) {Nc = 8; ++numQuads;}
        else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot support loop with %d edges", Ner);
        numCells += Nc;
        newCells += Nc-1;
        maxCorners = PetscMax(Ner*2+1, maxCorners);
      }
      ierr = EG_getBodyTopos(body, NULL, NODE, &Nv, &nobjs);CHKERRQ(ierr);
      for (v = 0; v < Nv; ++v) {
        ego vertex = nobjs[v];

        id = EG_indexBodyTopo(body, vertex);
        /* TODO: Instead of assuming contiguous ids, we could use a hash table */
        numVertices = PetscMax(id, numVertices);
      }
      EG_free(lobjs);
      EG_free(nobjs);
    }
    ierr = PetscHMapIGetSize(edgeMap, &numEdges);CHKERRQ(ierr);
    newVertices  = numEdges + numQuads;
    numVertices += newVertices;
    ierr = PetscPrintf(PETSC_COMM_SELF, "\nPLEX Input Array Checkouts\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, " Total Number of Unique Cells    = %D (%D)\n", numCells, newCells);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, " Total Number of Unique Vertices = %D (%D) \n", numVertices, newVertices);CHKERRQ(ierr);

    dim        = 2; /* Assume 3D Models :: Need to update to handle 2D Models in the future */
    cdim       = 3; /* Assume 3D Models :: Need to update to handle 2D Models in the future */
    numCorners = 3; /* Split cells into triangles */
    ierr = PetscMalloc3(numVertices*cdim, &coords, numCells*numCorners, &cells, maxCorners, &cone);CHKERRQ(ierr);

    /* Get vertex coordinates */
    for (b = 0; b < nbodies; ++b) {
      ego body = bodies[b];
      int id, Nv, v;

      ierr = EG_getBodyTopos(body, NULL, NODE, &Nv, &nobjs);CHKERRQ(ierr);
      for (v = 0; v < Nv; ++v) {
        ego    vertex = nobjs[v];
        double limits[4];
        int    dummy;

        ierr = EG_getTopology(vertex, &geom, &oclass, &mtype, limits, &dummy, &mobjs, &senses);CHKERRQ(ierr);
        id   = EG_indexBodyTopo(body, vertex);CHKERRQ(ierr);
        coords[(id-1)*cdim+0] = limits[0];
        coords[(id-1)*cdim+1] = limits[1];
        coords[(id-1)*cdim+2] = limits[2];
        ierr = PetscPrintf(PETSC_COMM_SELF, "    Node ID = %d (%d)\n", id, id-1);
        ierr = PetscPrintf(PETSC_COMM_SELF, "      (x,y,z) = (%lf, %lf, %lf) \n \n", coords[(id-1)*cdim+0], coords[(id-1)*cdim+1],coords[(id-1)*cdim+2]);
      }
      EG_free(nobjs);
    }
    ierr = PetscHMapIClear(edgeMap);CHKERRQ(ierr);
    fOff     = numVertices - newVertices + numEdges;
    numEdges = 0;
    numQuads = 0;
    for (b = 0; b < nbodies; ++b) {
      ego body = bodies[b];
      int Nl, l;

      ierr = EG_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs);CHKERRQ(ierr);
      for (l = 0; l < Nl; ++l) {
        ego loop = lobjs[l];
        int lid, Ner = 0, Ne, e;

        lid  = EG_indexBodyTopo(body, loop);CHKERRQ(ierr);
        ierr = EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses);CHKERRQ(ierr);
        for (e = 0; e < Ne; ++e) {
          ego       edge = objs[e];
          int       eid, Nv;
          PetscHashIter iter;
          PetscBool     found;

          ierr = EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses);
          if (mtype == DEGENERATE) continue;
          ++Ner;
          eid  = EG_indexBodyTopo(body, edge);CHKERRQ(ierr);
          ierr = PetscHMapIFind(edgeMap, eid-1, &iter, &found);CHKERRQ(ierr);
          if (!found) {
            PetscInt v = numVertices - newVertices + numEdges;
            double range[4], params[3] = {0., 0., 0.}, result[18];
            int    periodic[2];

            ierr = PetscHMapISet(edgeMap, eid-1, numEdges++);CHKERRQ(ierr);
            ierr = EG_getRange(edge, range, periodic); CHKERRQ(ierr);
            params[0] = 0.5*(range[0] + range[1]);
            ierr = EG_evaluate(edge, params, result);CHKERRQ(ierr);
            coords[v*cdim+0] = result[0];
            coords[v*cdim+1] = result[1];
            coords[v*cdim+2] = result[2];
            ierr = PetscPrintf(PETSC_COMM_SELF, "    Edge ID = %d (%D) \n", eid, v);
            ierr = PetscPrintf(PETSC_COMM_SELF, "      (x,y,z) = (%lf, %lf, %lf)\n", coords[v*cdim+0], coords[v*cdim+1],coords[v*cdim+2]);
            params[0] = range[0];
            ierr = EG_evaluate(edge, params, result);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_SELF, "      between (%lf, %lf, %lf)", result[0], result[1], result[2]);
            params[0] = range[1];
            ierr = EG_evaluate(edge, params, result);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_SELF, " and (%lf, %lf, %lf)\n\n", result[0], result[1], result[2]);
          }
        }
        if (Ner == 4) {
          PetscInt v = fOff + numQuads++;
          ego     *fobjs;
          double   range[4], params[3] = {0., 0., 0.}, result[18];
          int      Nf, face, periodic[2];

          ierr = EG_getBodyTopos(body, loop, FACE, &Nf, &fobjs);CHKERRQ(ierr);
          if (Nf != 1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Loop %d has %d faces, instead of 1", lid-1, Nf);
          face = EG_indexBodyTopo(body, fobjs[0]);CHKERRQ(ierr);
          ierr = EG_getRange(fobjs[0], range, periodic); CHKERRQ(ierr);
          params[0] = 0.5*(range[0] + range[1]);
          params[1] = 0.5*(range[2] + range[3]);
          ierr = EG_evaluate(fobjs[0], params, result);CHKERRQ(ierr);
          coords[v*cdim+0] = result[0];
          coords[v*cdim+1] = result[1];
          coords[v*cdim+2] = result[2];
          ierr = PetscPrintf(PETSC_COMM_SELF, "    Face ID = %d (%D) \n", face-1, v);
          ierr = PetscPrintf(PETSC_COMM_SELF, "      (x,y,z) = (%lf, %lf, %lf) \n \n", coords[v*cdim+0], coords[v*cdim+1],coords[v*cdim+2]);
        }
      }
    }
    if (numEdges + numQuads != newVertices) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of new vertices %D != %D previous count", numEdges + numQuads, newVertices);
    CHKMEMQ;

    /* Get cell vertices by traversing loops */
    numQuads = 0;
    cOff     = 0;
    for (b = 0; b < nbodies; ++b) {
      ego body = bodies[b];
      int id, Nl, l;

      ierr = EG_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs);CHKERRQ(ierr);
      for (l = 0; l < Nl; ++l) {
        ego loop = lobjs[l];
        int lid, Ner = 0, Ne, e, nc = 0, c, Nt, t;

        lid  = EG_indexBodyTopo(body, loop);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_SELF, "    LOOP ID: %d \n", lid);CHKERRQ(ierr);
        ierr = EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses);CHKERRQ(ierr);

        for (e = 0; e < Ne; ++e) {
          ego edge = objs[e];
          int points[3];
          int eid, Nv, v, tmp;

          eid  = EG_indexBodyTopo(body, edge);
          ierr = PetscPrintf(PETSC_COMM_SELF, "      EDGE ID: %d \n", eid);CHKERRQ(ierr);
          ierr = EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses);
          if (mtype == DEGENERATE) {ierr = PetscPrintf(PETSC_COMM_SELF, "        EGDE %d is DEGENERATE \n", eid);CHKERRQ(ierr); continue;}
          else                     {++Ner;}
          if (Nv != 2) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Edge %d has %d vertices != 2", eid, Nv);

          for (v = 0; v < Nv; ++v) {
            ego vertex = nobjs[v];

            id = EG_indexBodyTopo(body, vertex);
            points[v*2] = id-1;
          }
          {
            PetscInt edgeNum;

            ierr = PetscHMapIGet(edgeMap, eid-1, &edgeNum);CHKERRQ(ierr);
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
            else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Edge %d does not match its predecessor", eid);
          }
        }
        if (nc != 2*Ner)     SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of corners %D != %D", nc, 2*Ner);
        if (Ner == 4) {cone[nc++] = numVertices - newVertices + numEdges + numQuads++;}
        if (nc > maxCorners) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of corners %D > %D max", nc, maxCorners);
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
          default: SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Loop %d has %d edges, which we do not support", lid, Ner);
        }
        for (t = 0; t < Nt; ++t) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "      LOOP Corner NODEs Triangle %D (", t);
          for (c = 0; c < numCorners; ++c) {
            if (c > 0) {ierr = PetscPrintf(PETSC_COMM_SELF, ", ");}
            ierr = PetscPrintf(PETSC_COMM_SELF, "%D", cells[(cOff-Nt+t)*numCorners+c]);
          }
          ierr = PetscPrintf(PETSC_COMM_SELF, ")\n");
        }
      }
      EG_free(lobjs);
    }
  }
  if (cOff != numCells) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Count of total cells %D != %D previous count", cOff, numCells);
  ierr = DMPlexCreateFromCellListPetsc(PETSC_COMM_WORLD, dim, numCells, numVertices, numCorners, PETSC_TRUE, cells, cdim, coords, &dm);CHKERRQ(ierr);
  ierr = PetscFree3(coords, cells, cone);CHKERRQ(ierr);
  /* Embed EGADS model in DM */
  {
    PetscContainer modelObj;

    ierr = PetscContainerCreate(PETSC_COMM_SELF, &modelObj);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(modelObj, model);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) dm, "EGADS Model", (PetscObject) modelObj);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&modelObj);CHKERRQ(ierr);
  }
  /* Label points */
  ierr = DMCreateLabel(dm, "EGADS Body ID");CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "EGADS Body ID", &bodyLabel);CHKERRQ(ierr);
  ierr = DMCreateLabel(dm, "EGADS Face ID");CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "EGADS Face ID", &faceLabel);CHKERRQ(ierr);
  ierr = DMCreateLabel(dm, "EGADS Edge ID");CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "EGADS Edge ID", &edgeLabel);CHKERRQ(ierr);
  cOff = 0;
  for (b = 0; b < nbodies; ++b) {
    ego body = bodies[b];
    int id, Nl, l;

    ierr = EG_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs);CHKERRQ(ierr);
    for (l = 0; l < Nl; ++l) {
      ego  loop = lobjs[l];
      ego *fobjs;
      int  lid, Nf, fid, Ner = 0, Ne, e, Nt = 0, t;

      lid  = EG_indexBodyTopo(body, loop);CHKERRQ(ierr);
      ierr = EG_getBodyTopos(body, loop, FACE, &Nf, &fobjs);CHKERRQ(ierr);
      if (Nf > 1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Loop %d has %d > 1 faces, which is not supported", lid, Nf);CHKERRQ(ierr);
      fid  = EG_indexBodyTopo(body, fobjs[0]);CHKERRQ(ierr);
      EG_free(fobjs);
      ierr = EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses);CHKERRQ(ierr);
      for (e = 0; e < Ne; ++e) {
        ego             edge = objs[e];
        int             eid, Nv, v;
        PetscInt        points[3], support[2], numEdges, edgeNum;
        const PetscInt *edges;

        eid  = EG_indexBodyTopo(body, edge);
        ierr = EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses);
        if (mtype == DEGENERATE) {continue;}
        else                     {++Ner;}
        for (v = 0; v < Nv; ++v) {
          ego vertex = nobjs[v];

          id   = EG_indexBodyTopo(body, vertex);
          ierr = DMLabelSetValue(edgeLabel, numCells + id-1, eid);CHKERRQ(ierr);
          points[v*2] = numCells + id-1;
        }
        ierr = PetscHMapIGet(edgeMap, eid-1, &edgeNum);CHKERRQ(ierr);
        points[1] = numCells + numVertices - newVertices + edgeNum;

        ierr = DMLabelSetValue(edgeLabel, points[1], eid);CHKERRQ(ierr);
        support[0] = points[0];
        support[1] = points[1];
        ierr = DMPlexGetJoin(dm, 2, support, &numEdges, &edges);CHKERRQ(ierr);
        if (numEdges != 1) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Vertices (%D, %D) should only bound 1 edge, not %D", support[0], support[1], numEdges);
        ierr = DMLabelSetValue(edgeLabel, edges[0], eid);CHKERRQ(ierr);
        ierr = DMPlexRestoreJoin(dm, 2, support, &numEdges, &edges);CHKERRQ(ierr);
        support[0] = points[1];
        support[1] = points[2];
        ierr = DMPlexGetJoin(dm, 2, support, &numEdges, &edges);CHKERRQ(ierr);
        if (numEdges != 1) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Vertices (%D, %D) should only bound 1 edge, not %D", support[0], support[1], numEdges);
        ierr = DMLabelSetValue(edgeLabel, edges[0], eid);CHKERRQ(ierr);
        ierr = DMPlexRestoreJoin(dm, 2, support, &numEdges, &edges);CHKERRQ(ierr);
      }
      switch (Ner) {
        case 2: Nt = 2;break;
        case 3: Nt = 4;break;
        case 4: Nt = 8;break;
        default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Loop with %d edges is unsupported", Ner);
      }
      for (t = 0; t < Nt; ++t) {
        ierr = DMLabelSetValue(bodyLabel, cOff+t, b);CHKERRQ(ierr);
        ierr = DMLabelSetValue(faceLabel, cOff+t, fid);CHKERRQ(ierr);
      }
      cOff += Nt;
    }
    EG_free(lobjs);
  }
  ierr = PetscHMapIDestroy(&edgeMap);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL;
    PetscInt  clSize, cl, bval, fval;

    ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &clSize, &closure);CHKERRQ(ierr);
    ierr = DMLabelGetValue(bodyLabel, c, &bval);CHKERRQ(ierr);
    ierr = DMLabelGetValue(faceLabel, c, &fval);CHKERRQ(ierr);
    for (cl = 0; cl < clSize*2; cl += 2) {
      ierr = DMLabelSetValue(bodyLabel, closure[cl], bval);CHKERRQ(ierr);
      ierr = DMLabelSetValue(faceLabel, closure[cl], fval);CHKERRQ(ierr);
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &clSize, &closure);CHKERRQ(ierr);
  }
  ierr = DMLabelView(bodyLabel, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMLabelView(faceLabel, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMLabelView(edgeLabel, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);

  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);

  /* Close EGADSlite file */
  ierr = EG_close(context);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
    requires: egads

  test:
    suffix: sphere_0
    filter: sed "s/DM_[0-9a-zA-Z]*_0/DM__0/g"
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/unit_sphere.egadslite -dm_refine 1 -dm_view -dm_plex_check_all

  test:
    suffix: nozzle_0
    filter: sed "s/DM_[0-9a-zA-Z]*_0/DM__0/g"
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/nozzle.egadslite -dm_refine 1 -dm_view -dm_plex_check_all

TEST*/
