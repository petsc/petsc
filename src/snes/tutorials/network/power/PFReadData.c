#include "petscmat.h"
#include "power.h"
#include <string.h>
#include <ctype.h>

PetscErrorCode PFReadMatPowerData(PFDATA *pf, char *filename)
{
  FILE        *fp;
  VERTEX_Power Bus;
  LOAD         Load;
  GEN          Gen;
  EDGE_Power   Branch;
  PetscInt     line_counter   = 0;
  PetscInt     bus_start_line = -1, bus_end_line = -1; /* xx_end_line points to the next line after the record ends */
  PetscInt     gen_start_line = -1, gen_end_line = -1;
  PetscInt     br_start_line = -1, br_end_line = -1;
  char         line[MAXLINE];
  PetscInt     loadi = 0, geni = 0, bri = 0, busi = 0, i, j;
  int          extbusnum, bustype_i;
  double       Pd, Qd;
  PetscInt     maxbusnum = -1, intbusnum, *busext2intmap, genj, loadj;
  GEN          newgen;
  LOAD         newload;

  PetscFunctionBegin;
  fp = fopen(filename, "r");
  /* Check for valid file */
  PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Can't open Matpower data file %s", filename);
  pf->nload = 0;
  while (fgets(line, MAXLINE, fp)) {
    if (strstr(line, "mpc.bus = [")) bus_start_line = line_counter + 1;                     /* Bus data starts from next line */
    if (strstr(line, "mpc.gen") && gen_start_line == -1) gen_start_line = line_counter + 1; /* Generator data starts from next line */
    if (strstr(line, "mpc.branch")) br_start_line = line_counter + 1;                       /* Branch data starts from next line */
    if (strstr(line, "];")) {
      if (bus_start_line != -1 && bus_end_line == -1) bus_end_line = line_counter;
      if (gen_start_line != -1 && gen_end_line == -1) gen_end_line = line_counter;
      if (br_start_line != -1 && br_end_line == -1) br_end_line = line_counter;
    }

    /* Count the number of pq loads */
    if (bus_start_line != -1 && line_counter >= bus_start_line && bus_end_line == -1) {
      sscanf(line, "%d %d %lf %lf", &extbusnum, &bustype_i, &Pd, &Qd);
      if (!((Pd == 0.0) && (Qd == 0.0))) pf->nload++;
      if (extbusnum > maxbusnum) maxbusnum = extbusnum;
    }
    line_counter++;
  }
  fclose(fp);

  pf->nbus    = bus_end_line - bus_start_line;
  pf->ngen    = gen_end_line - gen_start_line;
  pf->nbranch = br_end_line - br_start_line;

  PetscCall(PetscCalloc1(pf->nbus, &pf->bus));
  PetscCall(PetscCalloc1(pf->ngen, &pf->gen));
  PetscCall(PetscCalloc1(pf->nload, &pf->load));
  PetscCall(PetscCalloc1(pf->nbranch, &pf->branch));
  Bus    = pf->bus;
  Gen    = pf->gen;
  Load   = pf->load;
  Branch = pf->branch;

  /* Setting pf->sbase to 100 */
  pf->sbase = 100.0;

  PetscCall(PetscMalloc1(maxbusnum + 1, &busext2intmap));
  for (i = 0; i < maxbusnum + 1; i++) busext2intmap[i] = -1;

  fp = fopen(filename, "r");
  /* Reading data */
  for (i = 0; i < line_counter; i++) {
    PetscCheck(fgets(line, MAXLINE, fp), PETSC_COMM_SELF, PETSC_ERR_SUP, "File is incorrectly formatted");

    if ((i >= bus_start_line) && (i < bus_end_line)) {
      double gl, bl, vm, va, basekV;
      int    bus_i, ide, area;
      /* Bus data */
      sscanf(line, "%d %d %lf %lf %lf %lf %d %lf %lf %lf", &bus_i, &ide, &Pd, &Qd, &gl, &bl, &area, &vm, &va, &basekV);
      Bus[busi].bus_i                = bus_i;
      Bus[busi].ide                  = ide;
      Bus[busi].area                 = area;
      Bus[busi].gl                   = gl;
      Bus[busi].bl                   = bl;
      Bus[busi].vm                   = vm;
      Bus[busi].va                   = va;
      Bus[busi].basekV               = basekV;
      Bus[busi].internal_i           = busi;
      busext2intmap[Bus[busi].bus_i] = busi;

      if (!((Pd == 0.0) && (Qd == 0.0))) {
        Load[loadi].bus_i                 = Bus[busi].bus_i;
        Load[loadi].status                = 1;
        Load[loadi].pl                    = Pd;
        Load[loadi].ql                    = Qd;
        Load[loadi].area                  = Bus[busi].area;
        Load[loadi].internal_i            = busi;
        Bus[busi].lidx[Bus[busi].nload++] = loadi;
        PetscCheck(Bus[busi].nload <= NLOAD_AT_BUS_MAX, PETSC_COMM_SELF, PETSC_ERR_SUP, "Exceeded maximum number of loads allowed at bus");
        loadi++;
      }
      busi++;
    }

    /* Read generator data */
    if (i >= gen_start_line && i < gen_end_line) {
      double pg, qg, qt, qb, vs, mbase, pt, pb;
      int    bus_i, status;
      sscanf(line, "%d %lf %lf %lf %lf %lf %lf %d %lf %lf", &bus_i, &pg, &qg, &qt, &qb, &vs, &mbase, &status, &pt, &pb);
      Gen[geni].bus_i  = bus_i;
      Gen[geni].status = status;
      Gen[geni].pg     = pg;
      Gen[geni].qg     = qg;
      Gen[geni].qt     = qt;
      Gen[geni].qb     = qb;
      Gen[geni].vs     = vs;
      Gen[geni].mbase  = mbase;
      Gen[geni].pt     = pt;
      Gen[geni].pb     = pb;

      intbusnum                                  = busext2intmap[Gen[geni].bus_i];
      Gen[geni].internal_i                       = intbusnum;
      Bus[intbusnum].gidx[Bus[intbusnum].ngen++] = geni;

      Bus[intbusnum].vm = Gen[geni].vs;

      PetscCheck(Bus[intbusnum].ngen <= NGEN_AT_BUS_MAX, PETSC_COMM_SELF, PETSC_ERR_SUP, "Exceeded maximum number of generators allowed at bus");
      geni++;
    }

    if (i >= br_start_line && i < br_end_line) {
      PetscScalar R, X, Bc, B, G, Zm, tap, shift, tap2, tapr, tapi;
      double      r, x, b, rateA, rateB, rateC, tapratio, phaseshift;
      int         fbus, tbus, status;
      sscanf(line, "%d %d %lf %lf %lf %lf %lf %lf %lf %lf %d", &fbus, &tbus, &r, &x, &b, &rateA, &rateB, &rateC, &tapratio, &phaseshift, &status);
      Branch[bri].fbus       = fbus;
      Branch[bri].tbus       = tbus;
      Branch[bri].status     = status;
      Branch[bri].r          = r;
      Branch[bri].x          = x;
      Branch[bri].b          = b;
      Branch[bri].rateA      = rateA;
      Branch[bri].rateB      = rateB;
      Branch[bri].rateC      = rateC;
      Branch[bri].tapratio   = tapratio;
      Branch[bri].phaseshift = phaseshift;

      if (Branch[bri].tapratio == 0.0) Branch[bri].tapratio = 1.0;
      Branch[bri].phaseshift *= PETSC_PI / 180.0;

      intbusnum              = busext2intmap[Branch[bri].fbus];
      Branch[bri].internal_i = intbusnum;

      intbusnum              = busext2intmap[Branch[bri].tbus];
      Branch[bri].internal_j = intbusnum;

      /* Compute self and transfer admittances */
      R  = Branch[bri].r;
      X  = Branch[bri].x;
      Bc = Branch[bri].b;

      Zm = R * R + X * X;
      G  = R / Zm;
      B  = -X / Zm;

      tap   = Branch[bri].tapratio;
      shift = Branch[bri].phaseshift;
      tap2  = tap * tap;
      tapr  = tap * PetscCosScalar(shift);
      tapi  = tap * PetscSinScalar(shift);

      Branch[bri].yff[0] = G / tap2;
      Branch[bri].yff[1] = (B + Bc / 2.0) / tap2;

      Branch[bri].yft[0] = -(G * tapr - B * tapi) / tap2;
      Branch[bri].yft[1] = -(B * tapr + G * tapi) / tap2;

      Branch[bri].ytf[0] = -(G * tapr + B * tapi) / tap2;
      Branch[bri].ytf[1] = -(B * tapr - G * tapi) / tap2;

      Branch[bri].ytt[0] = G;
      Branch[bri].ytt[1] = B + Bc / 2.0;

      bri++;
    }
  }
  fclose(fp);

  /* Reorder the generator data structure according to bus numbers */
  genj  = 0;
  loadj = 0;
  PetscCall(PetscMalloc1(pf->ngen, &newgen));
  PetscCall(PetscMalloc1(pf->nload, &newload));
  for (i = 0; i < pf->nbus; i++) {
    for (j = 0; j < pf->bus[i].ngen; j++) PetscCall(PetscMemcpy(&newgen[genj++], &pf->gen[pf->bus[i].gidx[j]], sizeof(struct _p_GEN)));
    for (j = 0; j < pf->bus[i].nload; j++) PetscCall(PetscMemcpy(&newload[loadj++], &pf->load[pf->bus[i].lidx[j]], sizeof(struct _p_LOAD)));
  }
  PetscCall(PetscFree(pf->gen));
  PetscCall(PetscFree(pf->load));
  pf->gen  = newgen;
  pf->load = newload;

  PetscCall(PetscFree(busext2intmap));
  PetscFunctionReturn(PETSC_SUCCESS);
}
