#include "water.h"
#include <string.h>
#include <ctype.h>

PetscErrorCode PumpHeadCurveResidual(SNES snes,Vec X, Vec F,void *ctx)
{
  const PetscScalar *x;
  PetscScalar *f;
  Pump        *pump=(Pump*)ctx;
  PetscScalar *head=pump->headcurve.head,*flow=pump->headcurve.flow;
  PetscInt i;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(X,&x));
  PetscCall(VecGetArray(F,&f));

  f[0] = f[1] = f[2] = 0;
  for (i=0; i < pump->headcurve.npt;i++) {
    f[0] +=   x[0] - x[1]*PetscPowScalar(flow[i],x[2]) - head[i]; /* Partial w.r.t x[0] */
    f[1] +=  (x[0] - x[1]*PetscPowScalar(flow[i],x[2]) - head[i])*-1*PetscPowScalar(flow[i],x[2]); /*Partial w.r.t x[1] */
    f[2] +=  (x[0] - x[1]*PetscPowScalar(flow[i],x[2]) - head[i])*-1*x[1]*x[2]*PetscPowScalar(flow[i],x[2]-1); /*Partial w.r.t x[2] */
  }

  PetscCall(VecRestoreArrayRead(X,&x));
  PetscCall(VecRestoreArray(F,&f));

  PetscFunctionReturn(0);
}

PetscErrorCode SetPumpHeadCurveParams(Pump *pump)
{
  SNES           snes;
  Vec            X,F;
  PetscScalar   *head,*flow,*x;
  SNESConvergedReason reason;

  PetscFunctionBegin;
  head = pump->headcurve.head;
  flow = pump->headcurve.flow;
  if (pump->headcurve.npt == 1) {
    /* Single point head curve, set the other two data points */
    flow[1] = 0;
    head[1] = 1.33*head[0]; /* 133% of design head -- From EPANET manual */
    flow[2] = 2*flow[0];    /* 200% of design flow -- From EPANET manual */
    head[2] = 0;
    pump->headcurve.npt += 2;
  }

  PetscCall(SNESCreate(PETSC_COMM_SELF,&snes));

  PetscCall(VecCreate(PETSC_COMM_SELF,&X));
  PetscCall(VecSetSizes(X,3,3));
  PetscCall(VecSetFromOptions(X));
  PetscCall(VecDuplicate(X,&F));

  PetscCall(SNESSetFunction(snes,F,PumpHeadCurveResidual,(void*)pump));
  PetscCall(SNESSetJacobian(snes,NULL,NULL,SNESComputeJacobianDefault,NULL));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(VecGetArray(X,&x));
  x[0] = head[1]; x[1] = 10; x[2] = 3;
  PetscCall(VecRestoreArray(X,&x));

  PetscCall(SNESSolve(snes,NULL,X));

  PetscCall(SNESGetConvergedReason(snes,&reason));
  if (reason < 0) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_CONV_FAILED,"Pump head curve did not converge");
  }

  PetscCall(VecGetArray(X,&x));
  pump->h0 = x[0];
  pump->r  = x[1];
  pump->n  = x[2];
  PetscCall(VecRestoreArray(X,&x));

  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&F));
  PetscCall(SNESDestroy(&snes));
  PetscFunctionReturn(0);
}

int LineStartsWith(const char *a, const char *b)
{
  if (strncmp(a, b, strlen(b)) == 0) return 1;
  return 0;
}

int CheckDataSegmentEnd(const char *line)
{
  if (LineStartsWith(line,"[JUNCTIONS]") || \
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
PetscErrorCode GetDataSegment(FILE *fp,char *line,fpos_t *data_segment_start_pos,PetscInt *ndatalines)
{
  PetscInt data_segment_end;
  PetscInt nlines=0;

  PetscFunctionBegin;
  data_segment_end = 0;
  fgetpos(fp,data_segment_start_pos);
  PetscCheckFalse(!fgets(line,MAXLINE,fp),PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Cannot read data segment from file");
  while (LineStartsWith(line,";")) {
    fgetpos(fp,data_segment_start_pos);
    PetscCheckFalse(!fgets(line,MAXLINE,fp),PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Cannot read data segment from file");
  }
  while (!data_segment_end) {
    PetscCheckFalse(!fgets(line,MAXLINE,fp),PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Cannot read data segment from file");
    nlines++;
    data_segment_end = CheckDataSegmentEnd(line);
  }
  *ndatalines = nlines;
  PetscFunctionReturn(0);
}

PetscErrorCode WaterReadData(WATERDATA *water,char *filename)
{
  FILE           *fp=NULL;
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
  double         v1,v2,v3,v4,v5,v6;

  PetscFunctionBegin;
  water->nvertex = water->nedge = 0;
  fp = fopen(filename,"rb");
  /* Check for valid file */
  PetscCheck(fp,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Can't open EPANET data file %s",filename);

  /* Read file and get line numbers for different data segments */
  while (fgets(line,MAXLINE,fp)) {

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
  PetscCall(PetscCalloc1(water->nvertex,&water->vertex));
  PetscCall(PetscCalloc1(water->nedge,&water->edge));
  vert = water->vertex;
  edge = water->edge;

  /* Junctions */
  fsetpos(fp,&junc_start_pos);
  for (i=0; i < water->njunction; i++) {
    int id=0,pattern=0;
    PetscCheckFalse(!fgets(line,MAXLINE,fp),PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Cannot read junction from file");
    vert[nv].type = VERTEX_TYPE_JUNCTION;
    junction = &vert[nv].junc;
    ndata = sscanf(line,"%d %lf %lf %d",&id,&v1,&v2,&pattern);PetscCheckFalse(ndata < 3,PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Unable to read junction data");
    vert[nv].id          = id;
    junction->dempattern = pattern;
    junction->elev   = (PetscScalar)v1;
    junction->demand = (PetscScalar)v2;
    junction->demand *= GPM_CFS;
    junction->id = vert[nv].id;
    nv++;
  }

  /* Reservoirs */
  fsetpos(fp,&res_start_pos);
  for (i=0; i < water->nreservoir; i++) {
    int id=0,pattern=0;
    PetscCheckFalse(!fgets(line,MAXLINE,fp),PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Cannot read reservoir from file");
    vert[nv].type = VERTEX_TYPE_RESERVOIR;
    reservoir = &vert[nv].res;
    ndata = sscanf(line,"%d %lf %d",&id,&v1,&pattern);PetscCheckFalse(ndata < 2,PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Unable to read reservoir data");
    vert[nv].id            = id;
    reservoir->headpattern = pattern;
    reservoir->head = (PetscScalar)v1;
    reservoir->id   = vert[nv].id;
    nv++;
  }

  /* Tanks */
  fsetpos(fp,&tank_start_pos);
  for (i=0; i < water->ntank; i++) {
    int id=0,curve=0;
    PetscCheckFalse(!fgets(line,MAXLINE,fp),PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Cannot read data tank from file");
    vert[nv].type = VERTEX_TYPE_TANK;
    tank = &vert[nv].tank;
    ndata = sscanf(line,"%d %lf %lf %lf %lf %lf %lf %d",&id,&v1,&v2,&v3,&v4,&v5,&v6,&curve);PetscCheckFalse(ndata < 7,PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Unable to read tank data");
    vert[nv].id       = id;
    tank->volumecurve = curve;
    tank->elev      = (PetscScalar)v1;
    tank->initlvl   = (PetscScalar)v2;
    tank->minlvl    = (PetscScalar)v3;
    tank->maxlvl    = (PetscScalar)v4;
    tank->diam      = (PetscScalar)v5;
    tank->minvolume = (PetscScalar)v6;
    tank->id        = vert[nv].id;
    nv++;
  }

  /* Pipes */
  fsetpos(fp,&pipe_start_pos);
  for (i=0; i < water->npipe; i++) {
    int id=0,node1=0,node2=0;
    PetscCheckFalse(!fgets(line,MAXLINE,fp),PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Cannot read data pipe from file");
    edge[ne].type = EDGE_TYPE_PIPE;
    pipe = &edge[ne].pipe;
    ndata = sscanf(line,"%d %d %d %lf %lf %lf %lf %s",&id,&node1,&node2,&v1,&v2,&v3,&v4,pipe->stat);
    pipe->id        = id;
    pipe->node1     = node1;
    pipe->node2     = node2;
    pipe->length    = (PetscScalar)v1;
    pipe->diam      = (PetscScalar)v2;
    pipe->roughness = (PetscScalar)v3;
    pipe->minorloss = (PetscScalar)v4;
    edge[ne].id     = pipe->id;
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
    int id=0,node1=0,node2=0,paramid=0;
    PetscCheckFalse(!fgets(line,MAXLINE,fp),PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Cannot read data pump from file");
    edge[ne].type = EDGE_TYPE_PUMP;
    pump = &edge[ne].pump;
    ndata = sscanf(line,"%d %d %d %s %d",&id,&node1,&node2,pump->param,&paramid);PetscCheckFalse(ndata != 5,PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Unable to read pump data");
    pump->id      = id;
    pump->node1   = node1;
    pump->node2   = node2;
    pump->paramid = paramid;
    edge[ne].id   = pump->id;
    ne++;
  }

  /* Curves */
  fsetpos(fp,&curve_start_pos);
  for (i=0; i < ncurve; i++) {
    int icurve_id=0;
    PetscCheckFalse(!fgets(line,MAXLINE,fp),PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Cannot read data curve from file");
    ndata = sscanf(line,"%d %lf %lf",&icurve_id,&v1,&v2);PetscCheckFalse(ndata != 3,PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Unable to read curve data");
    curve_id = icurve_id;
    curve_x  = (PetscScalar)v1;
    curve_y  = (PetscScalar)v2;
    /* Check for pump with the curve_id */
    for (j=water->npipe;j < water->npipe+water->npump;j++) {
      if (water->edge[j].pump.paramid == curve_id) {
        if (pump->headcurve.npt == 3) {
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Pump %d [%d --> %d]: No support for more than 3-pt head-flow curve",pump->id,pump->node1,pump->node2);
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
  for (j=water->npipe;j < water->npipe+water->npump;j++) {
    pump = &water->edge[j].pump;
    if (strcmp(pump->param,"HEAD") == 0) {
      /* Head-flow curve */
      PetscCall(SetPumpHeadCurveParams(pump));
    }
  }
  PetscFunctionReturn(0);
}
