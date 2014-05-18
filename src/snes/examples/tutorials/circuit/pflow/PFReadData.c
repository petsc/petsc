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
  PetscInt       line_counter=0,lindex[7]; /* lindex saves the line numbers of data start/end for sections */
  char           line[MAXLINE];
  PetscInt       k=0,loadi=0,geni=0,bri=0,busi=0,i;
  PetscInt       dummy,bustype_i,j;
  PetscScalar    Pd,Qd;

  PetscFunctionBegin;

  fp = fopen(filename,"r");
  /* Check for valid file */
  if (fp == NULL)  {
     ierr = PetscPrintf(PETSC_COMM_WORLD,"Can't open Matpower data file\n");CHKERRQ(ierr);
     exit(EXIT_FAILURE);
  }
  lindex[0]=lindex[1]=lindex[2]=lindex[3]=0;
  lindex[4]=lindex[5]=lindex[6]=0;
  pf->nload=0;
  while(fgets(line,MAXLINE,fp) != NULL) {
    if(strncmp(line,"mpc.baseMVA",11) == 0) lindex[k++] = line_counter;
    if(strncmp(line,"mpc.bus",7) == 0) lindex[k++] = line_counter+1; /* Bus data starts from next line */
    if(strncmp(line,"mpc.gen",7) == 0) lindex[k++] = line_counter+1; /* Generator data starts from next line */
    if(strncmp(line,"mpc.branch",10) == 0) lindex[k++] = line_counter+1; /* Branch data starts from next line */
    if(strncmp(line,"];",2) == 0) {
      if(k <= 6) lindex[k++] = line_counter;
    }
    /* Count the number of pq loads */
    if((lindex[1]) && (line_counter >= lindex[1]) &&(!lindex[2])) {
      sscanf(line,"%d %d %lf %lf",&dummy,&bustype_i,&Pd,&Qd);
      if(!((Pd == 0.0) && (Qd == 0.0))) pf->nload++;
    }
    line_counter++;
  }
  fclose(fp);

  pf->nbus    = lindex[2]-lindex[1];
  pf->ngen    = lindex[4]-lindex[3];
  pf->nbranch = lindex[6]-lindex[5];

  ierr = PetscMalloc(pf->nbus*sizeof(struct _p_VERTEXDATA),&pf->bus);CHKERRQ(ierr);
  ierr = PetscMalloc(pf->ngen*sizeof(struct _p_GEN),&pf->gen);CHKERRQ(ierr);
  ierr = PetscMalloc(pf->nload*sizeof(struct _p_LOAD),&pf->load);CHKERRQ(ierr);
  ierr = PetscMalloc(pf->nbranch*sizeof(struct _p_EDGEDATA),&pf->branch);CHKERRQ(ierr);
  Bus = pf->bus; Gen = pf->gen; Load = pf->load; Branch = pf->branch;

  /* Setting pf->sbase to 100 */
  pf->sbase = 100.0;

  fp = fopen(filename,"r");
  /* Reading data */
  for(i=0;i<line_counter;i++) {
    fgets(line,MAXLINE,fp);

    if((i >= lindex[1]) && (i < lindex[2])) {
      /* Bus data */
      sscanf(line,"%d %d %lf %lf %lf %lf %d %lf %lf %lf",\
	     &Bus[busi].bus_i,&Bus[busi].ide,&Pd,&Qd,&Bus[busi].gl,\
	     &Bus[busi].bl,&Bus[busi].area,&Bus[busi].vm,&Bus[busi].va,&Bus[busi].basekV);
      Bus[busi].internal_i = busi;
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
    if((i >= lindex[3]) && (i < lindex[4])) {
      sscanf(line,"%d %lf %lf %lf %lf %lf %lf %d %lf %lf",&Gen[geni].bus_i, \
	     &Gen[geni].pg,&Gen[geni].qg,&Gen[geni].qt,&Gen[geni].qb, \
	     &Gen[geni].vs,&Gen[geni].mbase,&Gen[geni].status,&Gen[geni].pt, \
	     &Gen[geni].pb);
      for(j=0;j<pf->nbus;j++) {
	if(Gen[geni].bus_i == Bus[j].bus_i) { 
	  Gen[geni].internal_i = Bus[j].internal_i;
	  Bus[j].gidx[Bus[j].ngen++] = geni;
	  if (Bus[j].ngen > NGEN_AT_BUS_MAX)
	    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Exceeded maximum number of generators allowed at bus");
	  break;
	}
      }
      geni++;
    }
    
    if((i >= lindex[5]) && (i < lindex[6])) {
      sscanf(line,"%d %d %lf %lf %lf %lf %lf %lf %lf %lf %d",&Branch[bri].fbus,&Branch[bri].tbus, \
	     &Branch[bri].r,&Branch[bri].x,&Branch[bri].b,		\
	     &Branch[bri].rateA,&Branch[bri].rateB,&Branch[bri].rateC, \
	     &Branch[bri].tapratio,&Branch[bri].phaseshift,&Branch[bri].status);
      if(!Branch[bri].tapratio) Branch[bri].tapratio = 1.0;
      Branch[bri].phaseshift *= PETSC_PI/180.0;
      for(j=0;j<pf->nbus;j++) {
	if(Branch[bri].fbus == Bus[j].bus_i) {
	  Branch[bri].internal_i = Bus[j].internal_i;
	  break;
	}
      }
      for(j=0;j<pf->nbus;j++) {
	if(Branch[bri].tbus == Bus[j].bus_i) {
	  Branch[bri].internal_j = Bus[j].internal_i;
	  break;
	}
      }
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


  PetscFunctionReturn(0);
}
