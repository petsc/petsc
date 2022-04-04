#include <petsc/private/dmpleximpl.h>
#include <../src/sys/classes/viewer/impls/vtk/vtkvimpl.h>

typedef struct {
  PetscInt nvertices;
  PetscInt ncells;
  PetscInt nconn;               /* number of entries in cell->vertex connectivity array */
} PieceInfo;

#if defined(PETSC_USE_REAL_SINGLE) || defined(PETSC_USE_REAL___FP16)
/* output in float if single or half precision in memory */
static const char precision[] = "Float32";
typedef float PetscVTUReal;
#define MPIU_VTUREAL MPI_FLOAT
#elif defined(PETSC_USE_REAL_DOUBLE) || defined(PETSC_USE_REAL___FLOAT128)
/* output in double if double or quad precision in memory */
static const char precision[] = "Float64";
typedef double PetscVTUReal;
#define MPIU_VTUREAL MPI_DOUBLE
#else
static const char precision[] = "UnknownPrecision";
typedef PetscReal PetscVTUReal;
#define MPIU_VTUREAL MPIU_REAL
#endif

static PetscErrorCode TransferWrite(MPI_Comm comm, PetscViewer viewer,FILE *fp,PetscMPIInt srank,PetscMPIInt root,const void *send,void *recv,PetscMPIInt count,MPI_Datatype mpidatatype,PetscMPIInt tag)
{
  PetscMPIInt    rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  if (rank == srank && rank != root) {
    PetscCallMPI(MPI_Send((void*)send,count,mpidatatype,root,tag,comm));
  } else if (rank == root) {
    const void *buffer;
    if (root == srank) {        /* self */
      buffer = send;
    } else {
      MPI_Status  status;
      PetscMPIInt nrecv;
      PetscCallMPI(MPI_Recv(recv,count,mpidatatype,srank,tag,comm,&status));
      PetscCallMPI(MPI_Get_count(&status,mpidatatype,&nrecv));
      PetscCheck(count == nrecv,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Array size mismatch");
      buffer = recv;
    }
    PetscCall(PetscViewerVTKFWrite(viewer,fp,buffer,count,mpidatatype));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexGetVTKConnectivity(DM dm, PetscBool localized, PieceInfo *piece,PetscVTKInt **oconn,PetscVTKInt **ooffsets,PetscVTKType **otypes)
{
  PetscSection   coordSection;
  PetscVTKInt    *conn,*offsets;
  PetscVTKType   *types;
  PetscInt       dim,vStart,vEnd,cStart,cEnd,pStart,pEnd,cellHeight,numLabelCells,hasLabel,c,v,countcell,countconn;

  PetscFunctionBegin;
  PetscCall(PetscMalloc3(piece->nconn,&conn,piece->ncells,&offsets,piece->ncells,&types));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(DMGetDimension(dm,&dim));
  PetscCall(DMPlexGetChart(dm,&pStart,&pEnd));
  PetscCall(DMPlexGetVTKCellHeight(dm, &cellHeight));
  PetscCall(DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMGetStratumSize(dm, "vtk", 1, &numLabelCells));
  hasLabel = numLabelCells > 0 ? PETSC_TRUE : PETSC_FALSE;

  countcell = 0;
  countconn = 0;
  for (c = cStart; c < cEnd; ++c) {
    PetscInt nverts,dof = 0,celltype,startoffset,nC=0;

    if (hasLabel) {
      PetscInt value;

      PetscCall(DMGetLabelValue(dm, "vtk", c, &value));
      if (value != 1) continue;
    }
    startoffset = countconn;
    if (localized) {
      PetscCall(PetscSectionGetDof(coordSection, c, &dof));
    }
    if (!dof) {
      PetscInt *closure = NULL;
      PetscInt  closureSize;

      PetscCall(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
      for (v = 0; v < closureSize*2; v += 2) {
        if ((closure[v] >= vStart) && (closure[v] < vEnd)) {
          if (!localized) conn[countconn++] = closure[v] - vStart;
          else conn[countconn++] = startoffset + nC;
          ++nC;
        }
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    } else {
      for (nC = 0; nC < dof/dim; nC++) conn[countconn++] = startoffset + nC;
    }

    {
      PetscInt n = PetscMin(nC, 8), s = countconn - nC, i, cone[8];
      for (i = 0; i < n; ++i) cone[i] = conn[s+i];
      PetscCall(DMPlexReorderCell(dm, c, cone));
      for (i = 0; i < n; ++i) conn[s+i] = (int)cone[i];
    }

    offsets[countcell] = countconn;

    nverts = countconn - startoffset;
    PetscCall(DMPlexVTKGetCellType_Internal(dm,dim,nverts,&celltype));

    types[countcell] = celltype;
    countcell++;
  }
  PetscCheck(countcell == piece->ncells,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Inconsistent cell count");
  PetscCheck(countconn == piece->nconn,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Inconsistent connectivity count");
  *oconn    = conn;
  *ooffsets = offsets;
  *otypes   = types;
  PetscFunctionReturn(0);
}

/*
  Write all fields that have been provided to the viewer
  Multi-block XML format with binary appended data.
*/
PetscErrorCode DMPlexVTKWriteAll_VTU(DM dm,PetscViewer viewer)
{
  MPI_Comm                 comm;
  PetscSection             coordSection;
  PetscViewer_VTK          *vtk = (PetscViewer_VTK*)viewer->data;
  PetscViewerVTKObjectLink link;
  FILE                     *fp;
  PetscMPIInt              rank,size,tag;
  PetscInt                 dimEmbed,cellHeight,cStart,cEnd,vStart,vEnd,numLabelCells,hasLabel,c,v,r,i;
  PetscBool                localized;
  PieceInfo                piece,*gpiece = NULL;
  void                     *buffer = NULL;
  const char               *byte_order = PetscBinaryBigEndian() ? "BigEndian" : "LittleEndian";
  PetscInt                 loops_per_scalar;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));
#if defined(PETSC_USE_COMPLEX)
  loops_per_scalar = 2;
#else
  loops_per_scalar = 1;
#endif
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCall(PetscCommGetNewTag(comm,&tag));

  PetscCall(PetscFOpen(comm,vtk->filename,"wb",&fp));
  PetscCall(PetscFPrintf(comm,fp,"<?xml version=\"1.0\"?>\n"));
  PetscCall(PetscFPrintf(comm,fp,"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"%s\">\n", byte_order));
  PetscCall(PetscFPrintf(comm,fp,"  <UnstructuredGrid>\n"));

  PetscCall(DMGetCoordinateDim(dm, &dimEmbed));
  PetscCall(DMPlexGetVTKCellHeight(dm, &cellHeight));
  PetscCall(DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMGetStratumSize(dm, "vtk", 1, &numLabelCells));
  PetscCall(DMGetCoordinatesLocalized(dm, &localized));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));

  hasLabel        = numLabelCells > 0 ? PETSC_TRUE : PETSC_FALSE;
  piece.nvertices = 0;
  piece.ncells    = 0;
  piece.nconn     = 0;
  if (!localized) piece.nvertices = vEnd - vStart;
  for (c = cStart; c < cEnd; ++c) {
    PetscInt dof = 0;
    if (hasLabel) {
      PetscInt value;

      PetscCall(DMGetLabelValue(dm, "vtk", c, &value));
      if (value != 1) continue;
    }
    if (localized) {
      PetscCall(PetscSectionGetDof(coordSection, c, &dof));
    }
    if (!dof) {
      PetscInt *closure = NULL;
      PetscInt closureSize;

      PetscCall(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
      for (v = 0; v < closureSize*2; v += 2) {
        if ((closure[v] >= vStart) && (closure[v] < vEnd)) {
          piece.nconn++;
          if (localized) piece.nvertices++;
        }
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    } else {
      piece.nvertices += dof/dimEmbed;
      piece.nconn     += dof/dimEmbed;
    }
    piece.ncells++;
  }
  if (rank == 0) PetscCall(PetscMalloc1(size,&gpiece));
  PetscCallMPI(MPI_Gather((PetscInt*)&piece,sizeof(piece)/sizeof(PetscInt),MPIU_INT,(PetscInt*)gpiece,sizeof(piece)/sizeof(PetscInt),MPIU_INT,0,comm));

  /*
   * Write file header
   */
  if (rank == 0) {
    PetscInt boffset = 0;

    for (r=0; r<size; r++) {
      PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"    <Piece NumberOfPoints=\"%D\" NumberOfCells=\"%D\">\n",gpiece[r].nvertices,gpiece[r].ncells));
      /* Coordinate positions */
      PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"      <Points>\n"));
      PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"        <DataArray type=\"%s\" Name=\"Position\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%D\" />\n",precision,boffset));
      boffset += gpiece[r].nvertices*3*sizeof(PetscVTUReal) + sizeof(int);
      PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"      </Points>\n"));
      /* Cell connectivity */
      PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"      <Cells>\n"));
      PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"        <DataArray type=\"Int32\" Name=\"connectivity\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%D\" />\n",boffset));
      boffset += gpiece[r].nconn*sizeof(PetscInt) + sizeof(int);
      PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"        <DataArray type=\"Int32\" Name=\"offsets\"      NumberOfComponents=\"1\" format=\"appended\" offset=\"%D\" />\n",boffset));
      boffset += gpiece[r].ncells*sizeof(PetscInt) + sizeof(int);
      PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"        <DataArray type=\"UInt8\" Name=\"types\"        NumberOfComponents=\"1\" format=\"appended\" offset=\"%D\" />\n",boffset));
      boffset += gpiece[r].ncells*sizeof(unsigned char) + sizeof(int);
      PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"      </Cells>\n"));

      /*
       * Cell Data headers
       */
      PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"      <CellData>\n"));
      PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"        <DataArray type=\"Int32\" Name=\"Rank\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%D\" />\n",boffset));
      boffset += gpiece[r].ncells*sizeof(int) + sizeof(int);
      /* all the vectors */
      for (link=vtk->link; link; link=link->next) {
        Vec          X = (Vec)link->vec;
        DM           dmX = NULL;
        PetscInt     bs = 1,nfields,field;
        const char   *vecname = "";
        PetscSection section;
        if ((link->ft != PETSC_VTK_CELL_FIELD) && (link->ft != PETSC_VTK_CELL_VECTOR_FIELD)) continue;
        if (((PetscObject)X)->name || link != vtk->link) { /* If the object is already named, use it. If it is past the first link, name it to disambiguate. */
          PetscCall(PetscObjectGetName((PetscObject)X,&vecname));
        }
        PetscCall(VecGetDM(X, &dmX));
        if (!dmX) dmX = dm;
        PetscCall(PetscObjectQuery(link->vec, "section", (PetscObject*) &section));
        if (!section) PetscCall(DMGetLocalSection(dmX, &section));
        if (cEnd > cStart) PetscCall(PetscSectionGetDof(section,cStart,&bs));
        PetscCall(PetscSectionGetNumFields(section,&nfields));
        field = 0;
        if (link->field >= 0) {
          field = link->field;
          nfields = field + 1;
        }
        for (i=0; field<(nfields?nfields:1); field++) {
          PetscInt     fbs,j;
          PetscFV      fv = NULL;
          PetscObject  f;
          PetscClassId fClass;
          const char *fieldname = NULL;
          char       buf[256];
          PetscBool    vector;
          if (nfields) {        /* We have user-defined fields/components */
            PetscCall(PetscSectionGetFieldDof(section,cStart,field,&fbs));
            PetscCall(PetscSectionGetFieldName(section,field,&fieldname));
          } else fbs = bs;      /* Say we have one field with 'bs' components */
          PetscCall(DMGetField(dmX,field,NULL,&f));
          PetscCall(PetscObjectGetClassId(f,&fClass));
          if (fClass == PETSCFV_CLASSID) {
            fv = (PetscFV) f;
          }
          if (nfields && !fieldname) {
            PetscCall(PetscSNPrintf(buf,sizeof(buf),"CellField%D",field));
            fieldname = buf;
          }
          vector = PETSC_FALSE;
          if (link->ft == PETSC_VTK_CELL_VECTOR_FIELD) {
            vector = PETSC_TRUE;
            PetscCheck(fbs <= 3,PetscObjectComm((PetscObject)viewer),PETSC_ERR_ARG_SIZ,"Cell vector fields can have at most 3 components, %D given", fbs);
            for (j = 0; j < fbs; j++) {
              const char *compName = NULL;
              if (fv) {
                PetscCall(PetscFVGetComponentName(fv,j,&compName));
                if (compName) break;
              }
            }
            if (j < fbs) vector = PETSC_FALSE;
          }
          if (vector) {
#if defined(PETSC_USE_COMPLEX)
            PetscCall(PetscFPrintf(comm,fp,"        <DataArray type=\"%s\" Name=\"%s%s.Re\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%D\" />\n",precision,vecname,fieldname,boffset));
            boffset += gpiece[r].ncells*3*sizeof(PetscVTUReal) + sizeof(int);
            PetscCall(PetscFPrintf(comm,fp,"        <DataArray type=\"%s\" Name=\"%s%s.Im\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%D\" />\n",precision,vecname,fieldname,boffset));
            boffset += gpiece[r].ncells*3*sizeof(PetscVTUReal) + sizeof(int);
#else
            PetscCall(PetscFPrintf(comm,fp,"        <DataArray type=\"%s\" Name=\"%s%s\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%D\" />\n",precision,vecname,fieldname,boffset));
            boffset += gpiece[r].ncells*3*sizeof(PetscVTUReal) + sizeof(int);
#endif
            i+=fbs;
          } else {
            for (j=0; j<fbs; j++) {
              const char *compName = NULL;
              char finalname[256];
              if (fv) {
                PetscCall(PetscFVGetComponentName(fv,j,&compName));
              }
              if (compName) {
                PetscCall(PetscSNPrintf(finalname,255,"%s%s.%s",vecname,fieldname,compName));
              }
              else if (fbs > 1) {
                PetscCall(PetscSNPrintf(finalname,255,"%s%s.%D",vecname,fieldname,j));
              } else {
                PetscCall(PetscSNPrintf(finalname,255,"%s%s",vecname,fieldname));
              }
#if defined(PETSC_USE_COMPLEX)
              PetscCall(PetscFPrintf(comm,fp,"        <DataArray type=\"%s\" Name=\"%s.Re\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%D\" />\n",precision,finalname,boffset));
              boffset += gpiece[r].ncells*sizeof(PetscVTUReal) + sizeof(int);
              PetscCall(PetscFPrintf(comm,fp,"        <DataArray type=\"%s\" Name=\"%s.Im\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%D\" />\n",precision,finalname,boffset));
              boffset += gpiece[r].ncells*sizeof(PetscVTUReal) + sizeof(int);
#else
              PetscCall(PetscFPrintf(comm,fp,"        <DataArray type=\"%s\" Name=\"%s\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%D\" />\n",precision,finalname,boffset));
              boffset += gpiece[r].ncells*sizeof(PetscVTUReal) + sizeof(int);
#endif
              i++;
            }
          }
        }
        //PetscCheck(i == bs,comm,PETSC_ERR_PLIB,"Total number of field components %D != block size %D",i,bs);
      }
      PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"      </CellData>\n"));

      /*
       * Point Data headers
       */
      PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"      <PointData>\n"));
      for (link=vtk->link; link; link=link->next) {
        Vec          X = (Vec)link->vec;
        DM           dmX;
        PetscInt     bs = 1,nfields,field;
        const char   *vecname = "";
        PetscSection section;
        if ((link->ft != PETSC_VTK_POINT_FIELD) && (link->ft != PETSC_VTK_POINT_VECTOR_FIELD)) continue;
        if (((PetscObject)X)->name || link != vtk->link) { /* If the object is already named, use it. If it is past the first link, name it to disambiguate. */
          PetscCall(PetscObjectGetName((PetscObject)X,&vecname));
        }
        PetscCall(VecGetDM(X, &dmX));
        if (!dmX) dmX = dm;
        PetscCall(PetscObjectQuery(link->vec, "section", (PetscObject*) &section));
        if (!section) PetscCall(DMGetLocalSection(dmX, &section));
        if (vEnd > vStart) PetscCall(PetscSectionGetDof(section,vStart,&bs));
        PetscCall(PetscSectionGetNumFields(section,&nfields));
        field = 0;
        if (link->field >= 0) {
          field = link->field;
          nfields = field + 1;
        }
        for (i=0; field<(nfields?nfields:1); field++) {
          PetscInt   fbs,j;
          const char *fieldname = NULL;
          char       buf[256];
          if (nfields) {        /* We have user-defined fields/components */
            PetscCall(PetscSectionGetFieldDof(section,vStart,field,&fbs));
            PetscCall(PetscSectionGetFieldName(section,field,&fieldname));
          } else fbs = bs;      /* Say we have one field with 'bs' components */
          if (nfields && !fieldname) {
            PetscCall(PetscSNPrintf(buf,sizeof(buf),"PointField%D",field));
            fieldname = buf;
          }
          if (link->ft == PETSC_VTK_POINT_VECTOR_FIELD) {
            PetscCheck(fbs <= 3,PetscObjectComm((PetscObject)viewer),PETSC_ERR_ARG_SIZ,"Point vector fields can have at most 3 components, %D given", fbs);
#if defined(PETSC_USE_COMPLEX)
            PetscCall(PetscFPrintf(comm,fp,"        <DataArray type=\"%s\" Name=\"%s%s.Re\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%D\" />\n",precision,vecname,fieldname,boffset));
            boffset += gpiece[r].nvertices*3*sizeof(PetscVTUReal) + sizeof(int);
            PetscCall(PetscFPrintf(comm,fp,"        <DataArray type=\"%s\" Name=\"%s%s.Im\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%D\" />\n",precision,vecname,fieldname,boffset));
            boffset += gpiece[r].nvertices*3*sizeof(PetscVTUReal) + sizeof(int);
#else
            PetscCall(PetscFPrintf(comm,fp,"        <DataArray type=\"%s\" Name=\"%s%s\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%D\" />\n",precision,vecname,fieldname,boffset));
            boffset += gpiece[r].nvertices*3*sizeof(PetscVTUReal) + sizeof(int);
#endif
          } else {
            for (j=0; j<fbs; j++) {
              const char *compName = NULL;
              char finalname[256];
              PetscCall(PetscSectionGetComponentName(section,field,j,&compName));
              PetscCall(PetscSNPrintf(finalname,255,"%s%s.%s",vecname,fieldname,compName));
#if defined(PETSC_USE_COMPLEX)
              PetscCall(PetscFPrintf(comm,fp,"        <DataArray type=\"%s\" Name=\"%s.Re\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%D\" />\n",precision,finalname,boffset));
              boffset += gpiece[r].nvertices*sizeof(PetscVTUReal) + sizeof(int);
              PetscCall(PetscFPrintf(comm,fp,"        <DataArray type=\"%s\" Name=\"%s.Im\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%D\" />\n",precision,finalname,boffset));
              boffset += gpiece[r].nvertices*sizeof(PetscVTUReal) + sizeof(int);
#else
              PetscCall(PetscFPrintf(comm,fp,"        <DataArray type=\"%s\" Name=\"%s\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%D\" />\n",precision,finalname,boffset));
              boffset += gpiece[r].nvertices*sizeof(PetscVTUReal) + sizeof(int);
#endif
            }
          }
        }
      }
      PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"      </PointData>\n"));
      PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"    </Piece>\n"));
    }
  }

  PetscCall(PetscFPrintf(comm,fp,"  </UnstructuredGrid>\n"));
  PetscCall(PetscFPrintf(comm,fp,"  <AppendedData encoding=\"raw\">\n"));
  PetscCall(PetscFPrintf(comm,fp,"_"));

  if (rank == 0) {
    PetscInt maxsize = 0;
    for (r=0; r<size; r++) {
      maxsize = PetscMax(maxsize, (PetscInt) (gpiece[r].nvertices*3*sizeof(PetscVTUReal)));
      maxsize = PetscMax(maxsize, (PetscInt) (gpiece[r].ncells*3*sizeof(PetscVTUReal)));
      maxsize = PetscMax(maxsize, (PetscInt) (gpiece[r].nconn*sizeof(PetscVTKInt)));
    }
    PetscCall(PetscMalloc(maxsize,&buffer));
  }
  for (r=0; r<size; r++) {
    if (r == rank) {
      PetscInt nsend;
      {                         /* Position */
        const PetscScalar *x;
        PetscVTUReal      *y = NULL;
        Vec               coords;
        PetscBool         copy;

        PetscCall(DMGetCoordinatesLocal(dm,&coords));
        PetscCall(VecGetArrayRead(coords,&x));
#if defined(PETSC_USE_COMPLEX)
        copy = PETSC_TRUE;
#else
        copy = (PetscBool) (dimEmbed != 3 || localized || (sizeof(PetscReal) != sizeof(PetscVTUReal)));
#endif
        if (copy) {
          PetscCall(PetscMalloc1(piece.nvertices*3,&y));
          if (localized) {
            PetscInt cnt;
            for (c=cStart,cnt=0; c<cEnd; c++) {
              PetscInt off, dof;

              PetscCall(PetscSectionGetDof(coordSection, c, &dof));
              if (!dof) {
                PetscInt *closure = NULL;
                PetscInt closureSize;

                PetscCall(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
                for (v = 0; v < closureSize*2; v += 2) {
                  if ((closure[v] >= vStart) && (closure[v] < vEnd)) {
                    PetscCall(PetscSectionGetOffset(coordSection, closure[v], &off));
                    if (dimEmbed != 3) {
                      y[cnt*3+0] = (PetscVTUReal) PetscRealPart(x[off+0]);
                      y[cnt*3+1] = (PetscVTUReal) ((dimEmbed > 1) ? PetscRealPart(x[off+1]) : 0.0);
                      y[cnt*3+2] = (PetscVTUReal) 0.0;
                    } else {
                      y[cnt*3+0] = (PetscVTUReal) PetscRealPart(x[off+0]);
                      y[cnt*3+1] = (PetscVTUReal) PetscRealPart(x[off+1]);
                      y[cnt*3+2] = (PetscVTUReal) PetscRealPart(x[off+2]);
                    }
                    cnt++;
                  }
                }
                PetscCall(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
              } else {
                PetscCall(PetscSectionGetOffset(coordSection, c, &off));
                if (dimEmbed != 3) {
                  for (i=0; i<dof/dimEmbed; i++) {
                    y[cnt*3+0] = (PetscVTUReal) PetscRealPart(x[off + i*dimEmbed + 0]);
                    y[cnt*3+1] = (PetscVTUReal) ((dimEmbed > 1) ? PetscRealPart(x[off + i*dimEmbed + 1]) : 0.0);
                    y[cnt*3+2] = (PetscVTUReal) 0.0;
                    cnt++;
                  }
                } else {
                  for (i=0; i<dof; i ++) {
                    y[cnt*3+i] = (PetscVTUReal) PetscRealPart(x[off + i]);
                  }
                  cnt += dof/dimEmbed;
                }
              }
            }
            PetscCheck(cnt == piece.nvertices,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Count does not match");
          } else {
            for (i=0; i<piece.nvertices; i++) {
              y[i*3+0] = (PetscVTUReal) PetscRealPart(x[i*dimEmbed+0]);
              y[i*3+1] = (PetscVTUReal) ((dimEmbed > 1) ? PetscRealPart(x[i*dimEmbed+1]) : 0.);
              y[i*3+2] = (PetscVTUReal) ((dimEmbed > 2) ? PetscRealPart(x[i*dimEmbed+2]) : 0.);
            }
          }
        }
        nsend = piece.nvertices*3;
        PetscCall(TransferWrite(comm,viewer,fp,r,0,copy ? (const void *) y : (const void *) x,buffer,nsend,MPIU_VTUREAL,tag));
        PetscCall(PetscFree(y));
        PetscCall(VecRestoreArrayRead(coords,&x));
      }
      {                           /* Connectivity, offsets, types */
        PetscVTKInt  *connectivity = NULL, *offsets = NULL;
        PetscVTKType *types = NULL;
        PetscCall(DMPlexGetVTKConnectivity(dm,localized,&piece,&connectivity,&offsets,&types));
        PetscCall(TransferWrite(comm,viewer,fp,r,0,connectivity,buffer,piece.nconn,MPI_INT,tag));
        PetscCall(TransferWrite(comm,viewer,fp,r,0,offsets,buffer,piece.ncells,MPI_INT,tag));
        PetscCall(TransferWrite(comm,viewer,fp,r,0,types,buffer,piece.ncells,MPI_CHAR,tag));
        PetscCall(PetscFree3(connectivity,offsets,types));
      }
      {                         /* Owners (cell data) */
        PetscVTKInt *owners;
        PetscCall(PetscMalloc1(piece.ncells,&owners));
        for (i=0; i<piece.ncells; i++) owners[i] = rank;
        PetscCall(TransferWrite(comm,viewer,fp,r,0,owners,buffer,piece.ncells,MPI_INT,tag));
        PetscCall(PetscFree(owners));
      }
      /* Cell data */
      for (link=vtk->link; link; link=link->next) {
        Vec               X = (Vec)link->vec;
        DM                dmX;
        const PetscScalar *x;
        PetscVTUReal      *y;
        PetscInt          bs = 1, nfields, field;
        PetscSection      section = NULL;

        if ((link->ft != PETSC_VTK_CELL_FIELD) && (link->ft != PETSC_VTK_CELL_VECTOR_FIELD)) continue;
        PetscCall(VecGetDM(X, &dmX));
        if (!dmX) dmX = dm;
        PetscCall(PetscObjectQuery(link->vec, "section", (PetscObject*) &section));
        if (!section) PetscCall(DMGetLocalSection(dmX, &section));
        if (cEnd > cStart) PetscCall(PetscSectionGetDof(section,cStart,&bs));
        PetscCall(PetscSectionGetNumFields(section,&nfields));
        field = 0;
        if (link->field >= 0) {
          field = link->field;
          nfields = field + 1;
        }
        PetscCall(VecGetArrayRead(X,&x));
        PetscCall(PetscMalloc1(piece.ncells*3,&y));
        for (i=0; field<(nfields?nfields:1); field++) {
          PetscInt     fbs,j;
          PetscFV      fv = NULL;
          PetscObject  f;
          PetscClassId fClass;
          PetscBool    vector;
          if (nfields && cEnd > cStart) {        /* We have user-defined fields/components */
            PetscCall(PetscSectionGetFieldDof(section,cStart,field,&fbs));
          } else fbs = bs;      /* Say we have one field with 'bs' components */
          PetscCall(DMGetField(dmX,field,NULL,&f));
          PetscCall(PetscObjectGetClassId(f,&fClass));
          if (fClass == PETSCFV_CLASSID) {
            fv = (PetscFV) f;
          }
          vector = PETSC_FALSE;
          if (link->ft == PETSC_VTK_CELL_VECTOR_FIELD) {
            vector = PETSC_TRUE;
            for (j = 0; j < fbs; j++) {
              const char *compName = NULL;
              if (fv) {
                PetscCall(PetscFVGetComponentName(fv,j,&compName));
                if (compName) break;
              }
            }
            if (j < fbs) vector = PETSC_FALSE;
          }
          if (vector) {
            PetscInt cnt, l;
            for (l = 0; l < loops_per_scalar; l++) {
              for (c=cStart,cnt=0; c<cEnd; c++) {
                const PetscScalar *xpoint;
                PetscInt off, j;

                if (hasLabel) {     /* Ignore some cells */
                  PetscInt value;
                  PetscCall(DMGetLabelValue(dmX, "vtk", c, &value));
                  if (value != 1) continue;
                }
                if (nfields) {
                  PetscCall(PetscSectionGetFieldOffset(section, c, field, &off));
                } else {
                  PetscCall(PetscSectionGetOffset(section, c, &off));
                }
                xpoint = &x[off];
                for (j = 0; j < fbs; j++) {
                  y[cnt++] = (PetscVTUReal) (l ? PetscImaginaryPart(xpoint[j]) : PetscRealPart(xpoint[j]));
                }
                for (; j < 3; j++) y[cnt++] = 0.;
              }
              PetscCheck(cnt == piece.ncells*3,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Count does not match");
              PetscCall(TransferWrite(comm,viewer,fp,r,0,y,buffer,piece.ncells*3,MPIU_VTUREAL,tag));
            }
          } else {
            for (i=0; i<fbs; i++) {
              PetscInt cnt, l;
              for (l = 0; l < loops_per_scalar; l++) {
                for (c=cStart,cnt=0; c<cEnd; c++) {
                  const PetscScalar *xpoint;
                  PetscInt off;

                  if (hasLabel) {     /* Ignore some cells */
                    PetscInt value;
                    PetscCall(DMGetLabelValue(dmX, "vtk", c, &value));
                    if (value != 1) continue;
                  }
                  if (nfields) {
                    PetscCall(PetscSectionGetFieldOffset(section, c, field, &off));
                  } else {
                    PetscCall(PetscSectionGetOffset(section, c, &off));
                  }
                  xpoint   = &x[off];
                  y[cnt++] = (PetscVTUReal) (l ? PetscImaginaryPart(xpoint[i]) : PetscRealPart(xpoint[i]));
                }
                PetscCheck(cnt == piece.ncells,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Count does not match");
                PetscCall(TransferWrite(comm,viewer,fp,r,0,y,buffer,piece.ncells,MPIU_VTUREAL,tag));
              }
            }
          }
        }
        PetscCall(PetscFree(y));
        PetscCall(VecRestoreArrayRead(X,&x));
      }
      /* point data */
      for (link=vtk->link; link; link=link->next) {
        Vec               X = (Vec)link->vec;
        DM                dmX;
        const PetscScalar *x;
        PetscVTUReal      *y;
        PetscInt          bs = 1, nfields, field;
        PetscSection      section = NULL;

        if ((link->ft != PETSC_VTK_POINT_FIELD) && (link->ft != PETSC_VTK_POINT_VECTOR_FIELD)) continue;
        PetscCall(VecGetDM(X, &dmX));
        if (!dmX) dmX = dm;
        PetscCall(PetscObjectQuery(link->vec, "section", (PetscObject*) &section));
        if (!section) PetscCall(DMGetLocalSection(dmX, &section));
        if (vEnd > vStart) PetscCall(PetscSectionGetDof(section,vStart,&bs));
        PetscCall(PetscSectionGetNumFields(section,&nfields));
        field = 0;
        if (link->field >= 0) {
          field = link->field;
          nfields = field + 1;
        }
        PetscCall(VecGetArrayRead(X,&x));
        PetscCall(PetscMalloc1(piece.nvertices*3,&y));
        for (i=0; field<(nfields?nfields:1); field++) {
          PetscInt fbs,j;
          if (nfields && vEnd > vStart) {        /* We have user-defined fields/components */
            PetscCall(PetscSectionGetFieldDof(section,vStart,field,&fbs));
          } else fbs = bs;      /* Say we have one field with 'bs' components */
          if (link->ft == PETSC_VTK_POINT_VECTOR_FIELD) {
            PetscInt cnt, l;
            for (l = 0; l < loops_per_scalar; l++) {
              if (!localized) {
                for (v=vStart,cnt=0; v<vEnd; v++) {
                  PetscInt          off;
                  const PetscScalar *xpoint;

                  if (nfields) {
                    PetscCall(PetscSectionGetFieldOffset(section,v,field,&off));
                  } else {
                    PetscCall(PetscSectionGetOffset(section,v,&off));
                  }
                  xpoint = &x[off];
                  for (j = 0; j < fbs; j++) {
                    y[cnt++] = (PetscVTUReal) (l ? PetscImaginaryPart(xpoint[j]) : PetscRealPart(xpoint[j]));
                  }
                  for (; j < 3; j++) y[cnt++] = 0.;
                }
              } else {
                for (c=cStart,cnt=0; c<cEnd; c++) {
                  PetscInt *closure = NULL;
                  PetscInt  closureSize, off;

                  PetscCall(DMPlexGetTransitiveClosure(dmX, c, PETSC_TRUE, &closureSize, &closure));
                  for (v = 0, off = 0; v < closureSize*2; v += 2) {
                    if ((closure[v] >= vStart) && (closure[v] < vEnd)) {
                      PetscInt    voff;
                      const PetscScalar *xpoint;

                      if (nfields) {
                        PetscCall(PetscSectionGetFieldOffset(section,closure[v],field,&voff));
                      } else {
                        PetscCall(PetscSectionGetOffset(section,closure[v],&voff));
                      }
                      xpoint = &x[voff];
                      for (j = 0; j < fbs; j++) {
                        y[cnt + off++] = (PetscVTUReal) (l ? PetscImaginaryPart(xpoint[j]) : PetscRealPart(xpoint[j]));
                      }
                      for (; j < 3; j++) y[cnt + off++] = 0.;
                    }
                  }
                  cnt += off;
                  PetscCall(DMPlexRestoreTransitiveClosure(dmX, c, PETSC_TRUE, &closureSize, &closure));
                }
              }
              PetscCheck(cnt == piece.nvertices*3,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Count does not match");
              PetscCall(TransferWrite(comm,viewer,fp,r,0,y,buffer,piece.nvertices*3,MPIU_VTUREAL,tag));
            }
          } else {
            for (i=0; i<fbs; i++) {
              PetscInt cnt, l;
              for (l = 0; l < loops_per_scalar; l++) {
                if (!localized) {
                  for (v=vStart,cnt=0; v<vEnd; v++) {
                    PetscInt          off;
                    const PetscScalar *xpoint;

                    if (nfields) {
                      PetscCall(PetscSectionGetFieldOffset(section,v,field,&off));
                    } else {
                      PetscCall(PetscSectionGetOffset(section,v,&off));
                    }
                    xpoint   = &x[off];
                    y[cnt++] = (PetscVTUReal) (l ? PetscImaginaryPart(xpoint[i]) : PetscRealPart(xpoint[i]));
                  }
                } else {
                  for (c=cStart,cnt=0; c<cEnd; c++) {
                    PetscInt *closure = NULL;
                    PetscInt  closureSize, off;

                    PetscCall(DMPlexGetTransitiveClosure(dmX, c, PETSC_TRUE, &closureSize, &closure));
                    for (v = 0, off = 0; v < closureSize*2; v += 2) {
                      if ((closure[v] >= vStart) && (closure[v] < vEnd)) {
                        PetscInt    voff;
                        const PetscScalar *xpoint;

                        if (nfields) {
                          PetscCall(PetscSectionGetFieldOffset(section,closure[v],field,&voff));
                        } else {
                          PetscCall(PetscSectionGetOffset(section,closure[v],&voff));
                        }
                        xpoint         = &x[voff];
                        y[cnt + off++] = (l ? PetscImaginaryPart(xpoint[i]) : PetscRealPart(xpoint[i]));
                      }
                    }
                    cnt += off;
                    PetscCall(DMPlexRestoreTransitiveClosure(dmX, c, PETSC_TRUE, &closureSize, &closure));
                  }
                }
                PetscCheck(cnt == piece.nvertices,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Count does not match");
                PetscCall(TransferWrite(comm,viewer,fp,r,0,y,buffer,piece.nvertices,MPIU_VTUREAL,tag));
              }
            }
          }
        }
        PetscCall(PetscFree(y));
        PetscCall(VecRestoreArrayRead(X,&x));
      }
    } else if (rank == 0) {
      PetscInt l;

      PetscCall(TransferWrite(comm,viewer,fp,r,0,NULL,buffer,gpiece[r].nvertices*3,MPIU_VTUREAL,tag)); /* positions */
      PetscCall(TransferWrite(comm,viewer,fp,r,0,NULL,buffer,gpiece[r].nconn,MPI_INT,tag)); /* connectivity */
      PetscCall(TransferWrite(comm,viewer,fp,r,0,NULL,buffer,gpiece[r].ncells,MPI_INT,tag)); /* offsets */
      PetscCall(TransferWrite(comm,viewer,fp,r,0,NULL,buffer,gpiece[r].ncells,MPI_CHAR,tag)); /* types */
      PetscCall(TransferWrite(comm,viewer,fp,r,0,NULL,buffer,gpiece[r].ncells,MPI_INT,tag)); /* owner rank (cells) */
      /* all cell data */
      for (link=vtk->link; link; link=link->next) {
        Vec          X = (Vec)link->vec;
        PetscInt     bs = 1, nfields, field;
        DM           dmX;
        PetscSection section = NULL;

        if ((link->ft != PETSC_VTK_CELL_FIELD) && (link->ft != PETSC_VTK_CELL_VECTOR_FIELD)) continue;
        PetscCall(VecGetDM(X, &dmX));
        if (!dmX) dmX = dm;
        PetscCall(PetscObjectQuery(link->vec, "section", (PetscObject*) &section));
        if (!section) PetscCall(DMGetLocalSection(dmX, &section));
        if (cEnd > cStart) PetscCall(PetscSectionGetDof(section,cStart,&bs));
        PetscCall(PetscSectionGetNumFields(section,&nfields));
        field = 0;
        if (link->field >= 0) {
          field = link->field;
          nfields = field + 1;
        }
        for (i=0; field<(nfields?nfields:1); field++) {
          PetscInt     fbs,j;
          PetscFV      fv = NULL;
          PetscObject  f;
          PetscClassId fClass;
          PetscBool    vector;
          if (nfields && cEnd > cStart) {        /* We have user-defined fields/components */
            PetscCall(PetscSectionGetFieldDof(section,cStart,field,&fbs));
          } else fbs = bs;      /* Say we have one field with 'bs' components */
          PetscCall(DMGetField(dmX,field,NULL,&f));
          PetscCall(PetscObjectGetClassId(f,&fClass));
          if (fClass == PETSCFV_CLASSID) {
            fv = (PetscFV) f;
          }
          vector = PETSC_FALSE;
          if (link->ft == PETSC_VTK_CELL_VECTOR_FIELD) {
            vector = PETSC_TRUE;
            for (j = 0; j < fbs; j++) {
              const char *compName = NULL;
              if (fv) {
                PetscCall(PetscFVGetComponentName(fv,j,&compName));
                if (compName) break;
              }
            }
            if (j < fbs) vector = PETSC_FALSE;
          }
          if (vector) {
            for (l = 0; l < loops_per_scalar; l++) {
              PetscCall(TransferWrite(comm,viewer,fp,r,0,NULL,buffer,gpiece[r].ncells*3,MPIU_VTUREAL,tag));
            }
          } else {
            for (i=0; i<fbs; i++) {
              for (l = 0; l < loops_per_scalar; l++) {
                PetscCall(TransferWrite(comm,viewer,fp,r,0,NULL,buffer,gpiece[r].ncells,MPIU_VTUREAL,tag));
              }
            }
          }
        }
      }
      /* all point data */
      for (link=vtk->link; link; link=link->next) {
        Vec          X = (Vec)link->vec;
        DM           dmX;
        PetscInt     bs = 1, nfields, field;
        PetscSection section = NULL;

        if ((link->ft != PETSC_VTK_POINT_FIELD) && (link->ft != PETSC_VTK_POINT_VECTOR_FIELD)) continue;
        PetscCall(VecGetDM(X, &dmX));
        if (!dmX) dmX = dm;
        PetscCall(PetscObjectQuery(link->vec, "section", (PetscObject*) &section));
        if (!section) PetscCall(DMGetLocalSection(dmX, &section));
        if (vEnd > vStart) PetscCall(PetscSectionGetDof(section,vStart,&bs));
        PetscCall(PetscSectionGetNumFields(section,&nfields));
        field = 0;
        if (link->field >= 0) {
          field = link->field;
          nfields = field + 1;
        }
        for (i=0; field<(nfields?nfields:1); field++) {
          PetscInt   fbs;
          if (nfields && vEnd > vStart) {        /* We have user-defined fields/components */
            PetscCall(PetscSectionGetFieldDof(section,vStart,field,&fbs));
          } else fbs = bs;      /* Say we have one field with 'bs' components */
          if (link->ft == PETSC_VTK_POINT_VECTOR_FIELD) {
            for (l = 0; l < loops_per_scalar; l++) {
              PetscCall(TransferWrite(comm,viewer,fp,r,0,NULL,buffer,gpiece[r].nvertices*3,MPIU_VTUREAL,tag));
            }
          } else {
            for (i=0; i<fbs; i++) {
              for (l = 0; l < loops_per_scalar; l++) {
                PetscCall(TransferWrite(comm,viewer,fp,r,0,NULL,buffer,gpiece[r].nvertices,MPIU_VTUREAL,tag));
              }
            }
          }
        }
      }
    }
  }
  PetscCall(PetscFree(gpiece));
  PetscCall(PetscFree(buffer));
  PetscCall(PetscFPrintf(comm,fp,"\n  </AppendedData>\n"));
  PetscCall(PetscFPrintf(comm,fp,"</VTKFile>\n"));
  PetscCall(PetscFClose(comm,fp));
  PetscFunctionReturn(0);
}
