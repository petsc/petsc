#include "petscmat.h"
#include "pf.h"
#include <string.h>
#include <ctype.h>
  
#undef __FUNCT__
#define __FUNCT__ "PFReadMatPowerData"
PetscErrorCode PFReadMatPowerData(PFDATA *pf,char *filename)
{
  FILE           *fp;
  PetscErrorCode ierr;
  VERTEXDATA     Bus;
  LOAD           Load;
  GEN            Gen;
  EDGEDATA       Branch;
  PetscInt       line_counter=0;
  PetscInt       bus_start_line=-1,bus_end_line=-1; /* xx_end_line points to the next line after the record ends */
  PetscInt       gen_start_line=-1,gen_end_line=-1;
  PetscInt       br_start_line=-1,br_end_line=-1;
  char           line[MAXLINE];
  PetscInt       loadi=0,geni=0,bri=0,busi=0,i;
  PetscInt       extbusnum,bustype_i,j;
  PetscScalar    Pd,Qd;
  PetscInt       maxbusnum=-1,intbusnum;

  PetscFunctionBegin;

  fp = fopen(filename,"r");
  /* Check for valid file */
  if (fp == NULL)  {
     ierr = PetscPrintf(PETSC_COMM_WORLD,"Can't open Matpower data file\n");CHKERRQ(ierr);
     exit(EXIT_FAILURE);
  }
  pf->nload=0;
  while(fgets(line,MAXLINE,fp) != NULL) {
    if(strstr(line,"mpc.bus") != NULL)    bus_start_line = line_counter+1; /* Bus data starts from next line */
    if(strstr(line,"mpc.gen") != NULL && gen_start_line == -1)    gen_start_line = line_counter+1; /* Generator data starts from next line */
    if(strstr(line,"mpc.branch") != NULL) br_start_line = line_counter+1; /* Branch data starts from next line */
    if(strstr(line,"];") != NULL) {
      if (bus_start_line != -1 && bus_end_line == -1) bus_end_line = line_counter;
      if (gen_start_line != -1 && gen_end_line == -1) gen_end_line = line_counter;
      if (br_start_line  != -1 && br_end_line == -1) br_end_line = line_counter;
    }

    /* Count the number of pq loads */
    if(bus_start_line != -1 && line_counter >= bus_start_line && bus_end_line == -1) {
      sscanf(line,"%d %d %lf %lf",&extbusnum,&bustype_i,&Pd,&Qd);
      if(!((Pd == 0.0) && (Qd == 0.0))) pf->nload++;
      if (extbusnum > maxbusnum) maxbusnum = extbusnum;
    }
    line_counter++;
  }
  fclose(fp);

  pf->nbus    = bus_end_line - bus_start_line;
  pf->ngen    = gen_end_line - gen_start_line;
  pf->nbranch = br_end_line  - br_start_line;

  ierr = PetscPrintf(PETSC_COMM_SELF,"nb = %d, ngen = %d, nload = %d, nbranch = %d\n",pf->nbus,pf->ngen,pf->nload,pf->nbranch);CHKERRQ(ierr);

  ierr = PetscMalloc(pf->nbus*sizeof(struct _p_VERTEXDATA),&pf->bus);CHKERRQ(ierr);
  ierr = PetscMalloc(pf->ngen*sizeof(struct _p_GEN),&pf->gen);CHKERRQ(ierr);
  ierr = PetscMalloc(pf->nload*sizeof(struct _p_LOAD),&pf->load);CHKERRQ(ierr);
  ierr = PetscMalloc(pf->nbranch*sizeof(struct _p_EDGEDATA),&pf->branch);CHKERRQ(ierr);
  Bus = pf->bus; Gen = pf->gen; Load = pf->load; Branch = pf->branch;

  for(i=0; i < pf->nbus; i++) {
    pf->bus[i].ngen = pf->bus[i].nload = 0;
  }

  /* Setting pf->sbase to 100 */
  pf->sbase = 100.0;

  PetscInt *busext2intmap;
  ierr = PetscMalloc1(maxbusnum+1,&busext2intmap);CHKERRQ(ierr);
  for(i=0; i < maxbusnum+1; i++) busext2intmap[i] = -1;

  fp = fopen(filename,"r");
  /* Reading data */
  for(i=0;i<line_counter;i++) {
    fgets(line,MAXLINE,fp);

    if((i >= bus_start_line) && (i < bus_end_line)) {
      /* Bus data */
      sscanf(line,"%d %d %lf %lf %lf %lf %d %lf %lf %lf",		\
	     &Bus[busi].bus_i,&Bus[busi].ide,&Pd,&Qd,&Bus[busi].gl,	\
	     &Bus[busi].bl,&Bus[busi].area,&Bus[busi].vm,&Bus[busi].va,&Bus[busi].basekV);
      Bus[busi].internal_i = busi;
      busext2intmap[Bus[busi].bus_i] = busi;

      if(!((Pd == 0.0) && (Qd == 0.0))) {
	Load[loadi].bus_i = Bus[busi].bus_i;
	Load[loadi].status = 1;
	Load[loadi].pl = Pd;
	Load[loadi].ql = Qd;
	Load[loadi].area = Bus[busi].area;
	Load[loadi].internal_i = busi;
	Bus[busi].lidx[Bus[busi].nload++] = loadi;
	if (Bus[busi].nload > NLOAD_AT_BUS_MAX)
	  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Exceeded maximum number of loads allowed at bus");
	loadi++;
      }
      busi++;
    }

    /* Read generator data */
    if(i >= gen_start_line && i < gen_end_line) {
      sscanf(line,"%d %lf %lf %lf %lf %lf %lf %d %lf %lf",&Gen[geni].bus_i, \
	     &Gen[geni].pg,&Gen[geni].qg,&Gen[geni].qt,&Gen[geni].qb, \
	     &Gen[geni].vs,&Gen[geni].mbase,&Gen[geni].status,&Gen[geni].pt, \
	     &Gen[geni].pb);

      intbusnum = busext2intmap[Gen[geni].bus_i];
      Gen[geni].internal_i = intbusnum;
      Bus[intbusnum].gidx[Bus[intbusnum].ngen++] = geni;

      Bus[intbusnum].vm = Gen[geni].vs;

      if (Bus[intbusnum].ngen > NGEN_AT_BUS_MAX) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Exceeded maximum number of generators allowed at bus");
      geni++;
    }
    
    if(i >= br_start_line && i < br_end_line) {
      sscanf(line,"%d %d %lf %lf %lf %lf %lf %lf %lf %lf %d",&Branch[bri].fbus,&Branch[bri].tbus, \
	     &Branch[bri].r,&Branch[bri].x,&Branch[bri].b,		\
	     &Branch[bri].rateA,&Branch[bri].rateB,&Branch[bri].rateC, \
	     &Branch[bri].tapratio,&Branch[bri].phaseshift,&Branch[bri].status);
      if(!Branch[bri].tapratio) Branch[bri].tapratio = 1.0;
      Branch[bri].phaseshift *= PETSC_PI/180.0;

      intbusnum = busext2intmap[Branch[bri].fbus];
      Branch[bri].internal_i = intbusnum;

      intbusnum = busext2intmap[Branch[bri].tbus];
      Branch[bri].internal_j = intbusnum;

      /* Compute self and transfer admittances */
      PetscScalar R,X,Bc,B,G,Zm,tap,shift,tap2,tapr,tapi;
      R = Branch[bri].r;
      X = Branch[bri].x;
      Bc = Branch[bri].b;

      Zm = R*R + X*X;
      G  = R/Zm;
      B  = -X/Zm;

      tap = Branch[bri].tapratio;
      shift = Branch[bri].phaseshift;
      tap2 = tap*tap;
      tapr = tap*cos(shift);
      tapi = tap*sin(shift);

      Branch[bri].yff[0] = G/tap2; 
      Branch[bri].yff[1] = (B+Bc/2.0)/tap2;
      
      Branch[bri].yft[0] = -(G*tapr - B*tapi)/tap2;
      Branch[bri].yft[1] = -(B*tapr + G*tapi)/tap2;

      Branch[bri].ytf[0] = -(G*tapr + B*tapi)/tap2;
      Branch[bri].ytf[1] = -(B*tapr - G*tapi)/tap2;

      Branch[bri].ytt[0] = G;
      Branch[bri].ytt[1] = B+Bc/2.0;

      bri++;
    }
  }
  fclose(fp);
  
  /* Reorder the generator data structure according to bus numbers */
  GEN  newgen;
  LOAD newload;
  PetscInt genj=0,loadj=0;
  ierr = PetscMalloc(pf->ngen*sizeof(struct _p_GEN),&newgen);CHKERRQ(ierr);
  ierr = PetscMalloc(pf->nload*sizeof(struct _p_LOAD),&newload);CHKERRQ(ierr);
  for (i = 0; i < pf->nbus; i++) {
    for (j = 0; j < pf->bus[i].ngen; j++) {
      ierr = PetscMemcpy(&newgen[genj++],&pf->gen[pf->bus[i].gidx[j]],sizeof(struct _p_GEN));
    }
    for (j = 0; j < pf->bus[i].nload; j++) {
      ierr = PetscMemcpy(&newload[loadj++],&pf->load[pf->bus[i].lidx[j]],sizeof(struct _p_LOAD));
    }
  }
  ierr = PetscFree(pf->gen);CHKERRQ(ierr);
  ierr = PetscFree(pf->load);CHKERRQ(ierr);
  pf->gen = newgen;
  pf->load = newload;

  ierr = PetscFree(busext2intmap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
