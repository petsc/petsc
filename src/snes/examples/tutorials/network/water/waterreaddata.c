#include "water.h"
#include <string.h>
#include <ctype.h>

PetscErrorCode PumpHeadCurveResidual(SNES snes,Vec X, Vec F,void *ctx)
{
  PetscErrorCode ierr;
  const PetscScalar *x;
  PetscScalar *f;
  Pump        *pump=(Pump*)ctx;
  PetscScalar *head=pump->headcurve.head,*flow=pump->headcurve.flow;
  PetscInt i;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  f[0] = f[1] = f[2] = 0;
  for(i=0; i < pump->headcurve.npt;i++) {
    f[0] +=   x[0] - x[1]*PetscPowScalar(flow[i],x[2]) - head[i]; /* Partial w.r.t x[0] */
    f[1] +=  (x[0] - x[1]*PetscPowScalar(flow[i],x[2]) - head[i])*-1*PetscPowScalar(flow[i],x[2]); /*Partial w.r.t x[1] */
    f[2] +=  (x[0] - x[1]*PetscPowScalar(flow[i],x[2]) - head[i])*-1*x[1]*x[2]*PetscPowScalar(flow[i],x[2]-1); /*Partial w.r.t x[2] */
  }

  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode SetPumpHeadCurveParams(Pump *pump)
{
  PetscErrorCode ierr;
  SNES           snes;
  Vec            X,F;
  PetscScalar   *head,*flow,*x;
  SNESConvergedReason reason;

  PetscFunctionBegin;
  head = pump->headcurve.head;
  flow = pump->headcurve.flow;
  if(pump->headcurve.npt == 1) {
    /* Single point head curve, set the other two data points */
    flow[1] = 0;
    head[1] = 1.33*head[0]; /* 133% of design head -- From EPANET manual */
    flow[2] = 2*flow[0];    /* 200% of design flow -- From EPANET manual */
    head[2] = 0;
    pump->headcurve.npt += 2;
  }

  ierr = SNESCreate(PETSC_COMM_SELF,&snes);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_SELF,&X);CHKERRQ(ierr);
  ierr = VecSetSizes(X,3,3);CHKERRQ(ierr);
  ierr = VecSetFromOptions(X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&F);CHKERRQ(ierr);

  ierr = SNESSetFunction(snes,F,PumpHeadCurveResidual,(void*)pump);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,NULL,NULL,SNESComputeJacobianDefault,NULL);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  x[0] = head[1]; x[1] = 10; x[2] = 3;
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);

  ierr = SNESGetConvergedReason(snes,&reason);CHKERRQ(ierr);
  if(reason < 0) {
    SETERRQ(PETSC_COMM_SELF,0,"Pump head curve did not converge\n");
  }

  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  pump->h0 = x[0];
  pump->r  = x[1];
  pump->n  = x[2];
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int LineStartsWith(const char *a, const char *b)
{
  if(strncmp(a, b, strlen(b)) == 0) return 1;
  return 0;
}

int CheckDataSegmentEnd(const char *line)
{
  if(LineStartsWith(line,"[JUNCTIONS]") || \
     LineStartsWith(line,"[RESERVOIRS]") || \
     LineStartsWith(line,"[TANKS]") || \
     LineStartsWith(line,"[PIPES]") || \
     LineStartsWith(line,"[PUMPS]") || \
     LineStartsWith(line,"[CURVES]") || \
     LineStartsWith(line,"[VALVES]") || \
     LineStartsWith(line,"[PATTERNS]") || \
     LineStartsWith(line,"[VALVES]") || \
     LineStartsWith(line,"[QUALITY]") || \
     LineStartsWith(line,"\n") || LineStartsWith(line,"\r\n")) {
    return 1;
  }
  return 0;
}

/* Gets the file pointer positiion for the start of the data segment and the 
   number of data segments (lines) read
*/
void GetDataSegment(FILE *fp,char *line,fpos_t *data_segment_start_pos,PetscInt *ndatalines)
{
  PetscInt data_segment_end;
  PetscInt nlines=0;

  data_segment_end = 0;
  fgetpos(fp,data_segment_start_pos);
  fgets(line,MAXLINE,fp);
  while(LineStartsWith(line,";")) {
    fgetpos(fp,data_segment_start_pos);
    fgets(line,MAXLINE,fp);
  }
  while(!data_segment_end) {
    fgets(line,MAXLINE,fp);
    nlines++;
    data_segment_end = CheckDataSegmentEnd(line);
  }
  *ndatalines = nlines;
}


PetscErrorCode WaterReadData(WATERDATA *water,char *filename)
{
  FILE           *fp;
  PetscErrorCode ierr;
  VERTEX_Water   vert;
  EDGE_Water     edge;
  fpos_t         junc_start_pos,res_start_pos,tank_start_pos,pipe_start_pos,pump_start_pos;
  fpos_t         curve_start_pos,title_start_pos;
  char           line[MAXLINE];
  PetscInt       i,j,nv=0,ne=0,ncurve=0,ntitle=0,nlines,ndata,curve_id;
  Junction       *junction=NULL;
  Reservoir      *reservoir=NULL;
  Tank           *tank=NULL;
  Pipe           *pipe=NULL;
  Pump           *pump=NULL;
  PetscScalar    curve_x,curve_y;

  PetscFunctionBegin;
  water->nvertex = water->nedge = 0;
  fp = fopen(filename,"r");
  /* Check for valid file */
  if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Can't open EPANET data file %s",filename);

  /* Read file and get line numbers for different data segments */
  while(fgets(line,MAXLINE,fp)) {

    if (strstr(line,"[TITLE]")) {
      GetDataSegment(fp,line,&title_start_pos,&ntitle);
    }

    if (strstr(line,"[JUNCTIONS]")) {
      GetDataSegment(fp,line,&junc_start_pos,&nlines);
      water->nvertex += nlines;
      water->njunction = nlines;
    }

    if (strstr(line,"[RESERVOIRS]")) {
      GetDataSegment(fp,line,&res_start_pos,&nlines);
      water->nvertex += nlines;
      water->nreservoir = nlines;
    }

    if (strstr(line,"[TANKS]")) {
      GetDataSegment(fp,line,&tank_start_pos,&nlines);
      water->nvertex += nlines;
      water->ntank = nlines;
    }

    if (strstr(line,"[PIPES]")) {
      GetDataSegment(fp,line,&pipe_start_pos,&nlines);
      water->nedge += nlines;
      water->npipe = nlines;
    }

    if (strstr(line,"[PUMPS]")) {
      GetDataSegment(fp,line,&pump_start_pos,&nlines);
      water->nedge += nlines;
      water->npump = nlines;
    }

    if (strstr(line,"[CURVES]")) {
      GetDataSegment(fp,line,&curve_start_pos,&ncurve);
    }
  }

  /* Allocate vertex and edge data structs */
  ierr = PetscCalloc1(water->nvertex,&water->vertex);CHKERRQ(ierr);
  ierr = PetscCalloc1(water->nedge,&water->edge);CHKERRQ(ierr);
  vert = water->vertex;
  edge = water->edge;

  /* Junctions */
  fsetpos(fp,&junc_start_pos);
  for (i=0; i < water->njunction; i++) {
    fgets(line,MAXLINE,fp);
    vert[nv].type = VERTEX_TYPE_JUNCTION;
    /*    printf("%s\n",line); */
    junction = &vert[nv].junc;
    ndata = sscanf(line,"%d %lf %lf %d",&vert[nv].id,&junction->elev,&junction->demand,&junction->dempattern);
    junction->demand *= GPM_CFS;
    junction->id = vert[nv].id;
    nv++;
  }

  /* Reservoirs */
  fsetpos(fp,&res_start_pos);
  for (i=0; i < water->nreservoir; i++) {
    fgets(line,MAXLINE,fp);
    vert[nv].type = VERTEX_TYPE_RESERVOIR;
    /*    printf("%s\n",line); */
    reservoir = &vert[nv].res;
    ndata = sscanf(line,"%d %lf %d",&vert[nv].id,&reservoir->head,&reservoir->headpattern);
    reservoir->id = vert[nv].id;
    nv++;
  }

  /* Tanks */
  fsetpos(fp,&tank_start_pos);
  for (i=0; i < water->ntank; i++) {
    fgets(line,MAXLINE,fp);
    vert[nv].type = VERTEX_TYPE_TANK;
    /*    printf("%s\n",line); */
    tank = &vert[nv].tank;
    ndata = sscanf(line,"%d %lf %lf %lf %lf %lf %lf %d",&vert[nv].id,&tank->elev,&tank->initlvl,&tank->minlvl,&tank->maxlvl,&tank->diam,&tank->minvolume,&tank->volumecurve);
    tank->id = vert[nv].id;
    nv++;
  }

  /* Pipes */
  fsetpos(fp,&pipe_start_pos);
  for (i=0; i < water->npipe; i++) {
    fgets(line,MAXLINE,fp);
    edge[ne].type = EDGE_TYPE_PIPE;
    /*    printf("%s\n",line); */
    pipe = &edge[ne].pipe;
    ndata = sscanf(line,"%d %d %d %lf %lf %lf %lf %s",&pipe->id,&pipe->node1,&pipe->node2,&pipe->length,&pipe->diam,&pipe->roughness,&pipe->minorloss,pipe->stat);
    edge[ne].id = pipe->id;
    if (strcmp(pipe->stat,"OPEN") == 0) pipe->status = PIPE_STATUS_OPEN;
    if (ndata < 8) {
      strcpy(pipe->stat,"OPEN"); /* default OPEN */
      pipe->status = PIPE_STATUS_OPEN;
    }
    if (ndata < 7) pipe->minorloss = 0.;
    pipe->n = 1.85;
    pipe->k = 4.72*pipe->length/(PetscPowScalar(pipe->roughness,pipe->n)*PetscPowScalar(0.0833333*pipe->diam,4.87));
    ne++;
  }

  /* Pumps */
  fsetpos(fp,&pump_start_pos);
  for (i=0; i < water->npump; i++) {
    fgets(line,MAXLINE,fp);
    edge[ne].type = EDGE_TYPE_PUMP;
    /*    printf("%s\n",line); */
    pump = &edge[ne].pump;
    ndata = sscanf(line,"%d %d %d %s %d",&pump->id,&pump->node1,&pump->node2,pump->param,&pump->paramid);
    edge[ne].id = pump->id;
    ne++;
  }

  /* Curves */
  fsetpos(fp,&curve_start_pos);
  for (i=0; i < ncurve; i++) {
    fgets(line,MAXLINE,fp);
    /*    printf("%s\n",line); */
    ndata = sscanf(line,"%d %lf %lf",&curve_id,&curve_x,&curve_y);
    /* Check for pump with the curve_id */
    for (j=water->npipe;j < water->npipe+water->npump;j++) {
      if(water->edge[j].pump.paramid == curve_id) {
	if(pump->headcurve.npt == 3) {
	  SETERRQ3(PETSC_COMM_SELF,0,"Pump %d [%d --> %d]: No support for more than 3-pt head-flow curve",pump->id,pump->node1,pump->node2);
	}
	pump = &water->edge[j].pump;
	pump->headcurve.flow[pump->headcurve.npt] = curve_x*GPM_CFS;
	pump->headcurve.head[pump->headcurve.npt] = curve_y;
	pump->headcurve.npt++;
	break;
      }
    }
  }

  fclose(fp);

  /* Get pump curve parameters */
  for(j=water->npipe;j < water->npipe+water->npump;j++) {
    pump = &water->edge[j].pump;
    if (strcmp(pump->param,"HEAD") == 0) {
      /* Head-flow curve */
      ierr = SetPumpHeadCurveParams(pump);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
