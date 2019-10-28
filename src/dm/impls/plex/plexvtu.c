#include <petsc/private/dmpleximpl.h>
#include <../src/sys/classes/viewer/impls/vtk/vtkvimpl.h>

typedef struct {
  PetscInt nvertices;
  PetscInt ncells;
  PetscInt nconn;               /* number of entries in cell->vertex connectivity array */
} PieceInfo;

#if defined(PETSC_USE_REAL_SINGLE)
static const char precision[] = "Float32";
#elif defined(PETSC_USE_REAL_DOUBLE)
static const char precision[] = "Float64";
#else
static const char precision[] = "UnknownPrecision";
#endif

static PetscErrorCode TransferWrite(PetscViewer viewer,FILE *fp,PetscMPIInt srank,PetscMPIInt root,const void *send,void *recv,PetscMPIInt count,MPI_Datatype mpidatatype,PetscMPIInt tag)
{
  PetscMPIInt    rank;
  PetscErrorCode ierr;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  if (rank == srank && rank != root) {
    ierr = MPI_Send((void*)send,count,mpidatatype,root,tag,comm);CHKERRQ(ierr);
  } else if (rank == root) {
    const void *buffer;
    if (root == srank) {        /* self */
      buffer = send;
    } else {
      MPI_Status  status;
      PetscMPIInt nrecv;
      ierr = MPI_Recv(recv,count,mpidatatype,srank,tag,comm,&status);CHKERRQ(ierr);
      ierr = MPI_Get_count(&status,mpidatatype,&nrecv);CHKERRQ(ierr);
      if (count != nrecv) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Array size mismatch");
      buffer = recv;
    }
    ierr = PetscViewerVTKFWrite(viewer,fp,buffer,count,mpidatatype);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexGetVTKConnectivity(DM dm, PetscBool localized, PieceInfo *piece,PetscVTKInt **oconn,PetscVTKInt **ooffsets,PetscVTKType **otypes)
{
  PetscErrorCode ierr;
  PetscSection   coordSection;
  PetscVTKInt    *conn,*offsets;
  PetscVTKType   *types;
  PetscInt       dim,vStart,vEnd,cStart,cEnd,pStart,pEnd,cellHeight,numLabelCells,hasLabel,c,v,countcell,countconn;

  PetscFunctionBegin;
  ierr = PetscMalloc3(piece->nconn,&conn,piece->ncells,&offsets,piece->ncells,&types);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr     = DMGetStratumSize(dm, "vtk", 1, &numLabelCells);CHKERRQ(ierr);
  hasLabel = numLabelCells > 0 ? PETSC_TRUE : PETSC_FALSE;

  countcell = 0;
  countconn = 0;
  for (c = cStart; c < cEnd; ++c) {
    PetscInt nverts,dof = 0,celltype,startoffset,nC=0;

    if (hasLabel) {
      PetscInt value;

      ierr = DMGetLabelValue(dm, "vtk", c, &value);CHKERRQ(ierr);
      if (value != 1) continue;
    }
    startoffset = countconn;
    if (localized) {
      ierr = PetscSectionGetDof(coordSection, c, &dof);CHKERRQ(ierr);
    }
    if (!dof) {
      PetscInt *closure = NULL;
      PetscInt  closureSize;

      ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for (v = 0; v < closureSize*2; v += 2) {
        if ((closure[v] >= vStart) && (closure[v] < vEnd)) {
          if (!localized) conn[countconn++] = closure[v] - vStart;
          else conn[countconn++] = startoffset + nC;
          ++nC;
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    } else {
      for (nC = 0; nC < dof/dim; nC++) conn[countconn++] = startoffset + nC;
    }
    ierr = DMPlexInvertCell(dim, nC, &conn[countconn-nC]);CHKERRQ(ierr);

    offsets[countcell] = countconn;

    nverts = countconn - startoffset;
    ierr   = DMPlexVTKGetCellType_Internal(dm,dim,nverts,&celltype);CHKERRQ(ierr);

    types[countcell] = celltype;
    countcell++;
  }
  if (countcell != piece->ncells) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Inconsistent cell count");
  if (countconn != piece->nconn) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Inconsistent connectivity count");
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
  PetscErrorCode           ierr;
  PetscInt                 dimEmbed,cellHeight,cStart,cEnd,vStart,vEnd,numLabelCells,hasLabel,c,v,r,i;
  PetscBool                localized;
  PieceInfo                piece,*gpiece = NULL;
  void                     *buffer = NULL;
  const char               *byte_order = PetscBinaryBigEndian() ? "BigEndian" : "LittleEndian";

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Complex values not supported");
#endif
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  tag  = ((PetscObject)viewer)->tag;

  ierr = PetscFOpen(comm,vtk->filename,"wb",&fp);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fp,"<?xml version=\"1.0\"?>\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fp,"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"%s\">\n", byte_order);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fp,"  <UnstructuredGrid>\n");CHKERRQ(ierr);

  ierr = DMGetCoordinateDim(dm, &dimEmbed);CHKERRQ(ierr);
  ierr = DMPlexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMGetStratumSize(dm, "vtk", 1, &numLabelCells);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocalized(dm, &localized);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);

  hasLabel        = numLabelCells > 0 ? PETSC_TRUE : PETSC_FALSE;
  piece.nvertices = 0;
  piece.ncells    = 0;
  piece.nconn     = 0;
  if (!localized) piece.nvertices = vEnd - vStart;
  for (c = cStart; c < cEnd; ++c) {
    PetscInt dof = 0;
    if (hasLabel) {
      PetscInt value;

      ierr = DMGetLabelValue(dm, "vtk", c, &value);CHKERRQ(ierr);
      if (value != 1) continue;
    }
    if (localized) {
      ierr = PetscSectionGetDof(coordSection, c, &dof);CHKERRQ(ierr);
    }
    if (!dof) {
      PetscInt *closure = NULL;
      PetscInt closureSize;

      ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for (v = 0; v < closureSize*2; v += 2) {
        if ((closure[v] >= vStart) && (closure[v] < vEnd)) {
          piece.nconn++;
          if (localized) piece.nvertices++;
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    } else {
      piece.nvertices += dof/dimEmbed;
      piece.nconn     += dof/dimEmbed;
    }
    piece.ncells++;
  }
  if (!rank) {ierr = PetscMalloc1(size,&gpiece);CHKERRQ(ierr);}
  ierr = MPI_Gather((PetscInt*)&piece,sizeof(piece)/sizeof(PetscInt),MPIU_INT,(PetscInt*)gpiece,sizeof(piece)/sizeof(PetscInt),MPIU_INT,0,comm);CHKERRQ(ierr);

  /*
   * Write file header
   */
  if (!rank) {
    PetscInt boffset = 0;

    for (r=0; r<size; r++) {
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"    <Piece NumberOfPoints=\"%D\" NumberOfCells=\"%D\">\n",gpiece[r].nvertices,gpiece[r].ncells);CHKERRQ(ierr);
      /* Coordinate positions */
      ierr     = PetscFPrintf(PETSC_COMM_SELF,fp,"      <Points>\n");CHKERRQ(ierr);
      ierr     = PetscFPrintf(PETSC_COMM_SELF,fp,"        <DataArray type=\"%s\" Name=\"Position\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%D\" />\n",precision,boffset);CHKERRQ(ierr);
      boffset += gpiece[r].nvertices*3*sizeof(PetscScalar) + sizeof(int);
      ierr     = PetscFPrintf(PETSC_COMM_SELF,fp,"      </Points>\n");CHKERRQ(ierr);
      /* Cell connectivity */
      ierr     = PetscFPrintf(PETSC_COMM_SELF,fp,"      <Cells>\n");CHKERRQ(ierr);
      ierr     = PetscFPrintf(PETSC_COMM_SELF,fp,"        <DataArray type=\"Int32\" Name=\"connectivity\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%D\" />\n",boffset);CHKERRQ(ierr);
      boffset += gpiece[r].nconn*sizeof(PetscInt) + sizeof(int);
      ierr     = PetscFPrintf(PETSC_COMM_SELF,fp,"        <DataArray type=\"Int32\" Name=\"offsets\"      NumberOfComponents=\"1\" format=\"appended\" offset=\"%D\" />\n",boffset);CHKERRQ(ierr);
      boffset += gpiece[r].ncells*sizeof(PetscInt) + sizeof(int);
      ierr     = PetscFPrintf(PETSC_COMM_SELF,fp,"        <DataArray type=\"UInt8\" Name=\"types\"        NumberOfComponents=\"1\" format=\"appended\" offset=\"%D\" />\n",boffset);CHKERRQ(ierr);
      boffset += gpiece[r].ncells*sizeof(unsigned char) + sizeof(int);
      ierr     = PetscFPrintf(PETSC_COMM_SELF,fp,"      </Cells>\n");CHKERRQ(ierr);

      /*
       * Cell Data headers
       */
      ierr     = PetscFPrintf(PETSC_COMM_SELF,fp,"      <CellData>\n");CHKERRQ(ierr);
      ierr     = PetscFPrintf(PETSC_COMM_SELF,fp,"        <DataArray type=\"Int32\" Name=\"Rank\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%D\" />\n",boffset);CHKERRQ(ierr);
      boffset += gpiece[r].ncells*sizeof(int) + sizeof(int);
      /* all the vectors */
      for (link=vtk->link; link; link=link->next) {
        Vec        X = (Vec)link->vec;
        DM         dmX = NULL;
        PetscInt   bs,nfields,field;
        const char *vecname = "";
        PetscSection section;
        if ((link->ft != PETSC_VTK_CELL_FIELD) && (link->ft != PETSC_VTK_CELL_VECTOR_FIELD)) continue;
        if (((PetscObject)X)->name || link != vtk->link) { /* If the object is already named, use it. If it is past the first link, name it to disambiguate. */
          ierr = PetscObjectGetName((PetscObject)X,&vecname);CHKERRQ(ierr);
        }
        ierr = VecGetDM(X, &dmX);CHKERRQ(ierr);
        if (!dmX) dmX = dm;
        ierr = PetscObjectQuery(link->vec, "section", (PetscObject*) &section);CHKERRQ(ierr);
        if (!section) {ierr = DMGetLocalSection(dmX, &section);CHKERRQ(ierr);}
        ierr = PetscSectionGetDof(section,cStart,&bs);CHKERRQ(ierr);
        ierr = PetscSectionGetNumFields(section,&nfields);CHKERRQ(ierr);
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
            ierr = PetscSectionGetFieldDof(section,cStart,field,&fbs);CHKERRQ(ierr);
            ierr = PetscSectionGetFieldName(section,field,&fieldname);CHKERRQ(ierr);
          } else fbs = bs;      /* Say we have one field with 'bs' components */
          ierr = DMGetField(dmX,field,NULL,&f);CHKERRQ(ierr);
          ierr = PetscObjectGetClassId(f,&fClass);CHKERRQ(ierr);
          if (fClass == PETSCFV_CLASSID) {
            fv = (PetscFV) f;
          }
          if (nfields && !fieldname) {
            ierr = PetscSNPrintf(buf,sizeof(buf),"CellField%D",field);CHKERRQ(ierr);
            fieldname = buf;
          }
          vector = PETSC_FALSE;
          if (link->ft == PETSC_VTK_CELL_VECTOR_FIELD) {
            vector = PETSC_TRUE;
            if (fbs > 3) SETERRQ1(PetscObjectComm((PetscObject)viewer),PETSC_ERR_ARG_SIZ,"Cell vector fields can have at most 3 components, %D given\n", fbs);
            for (j = 0; j < fbs; j++) {
              const char *compName = NULL;
              if (fv) {
                ierr = PetscFVGetComponentName(fv,j,&compName);CHKERRQ(ierr);
                if (compName) break;
              }
            }
            if (j < fbs) vector = PETSC_FALSE;
          }
          if (vector) {
            ierr = PetscFPrintf(comm,fp,"        <DataArray type=\"%s\" Name=\"%s%s\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%D\" />\n",precision,vecname,fieldname,boffset);CHKERRQ(ierr);
            boffset += gpiece[r].ncells*3*sizeof(PetscScalar) + sizeof(int);
            i+=fbs;
          } else {
            for (j=0; j<fbs; j++) {
              const char *compName = NULL;
              if (fv) {
                ierr = PetscFVGetComponentName(fv,j,&compName);CHKERRQ(ierr);
              }
              if (compName) {
                ierr = PetscFPrintf(comm,fp,"        <DataArray type=\"%s\" Name=\"%s%s.%s\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%D\" />\n",precision,vecname,fieldname,compName,boffset);CHKERRQ(ierr);
              }
              else if (fbs > 1) {
                ierr = PetscFPrintf(comm,fp,"        <DataArray type=\"%s\" Name=\"%s%s.%D\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%D\" />\n",precision,vecname,fieldname,j,boffset);CHKERRQ(ierr);
              } else {
                ierr = PetscFPrintf(comm,fp,"        <DataArray type=\"%s\" Name=\"%s%s\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%D\" />\n",precision,vecname,fieldname,boffset);CHKERRQ(ierr);
              }
              boffset += gpiece[r].ncells*sizeof(PetscScalar) + sizeof(int);
              i++;
            }
          }
        }
        if (i != bs) SETERRQ2(comm,PETSC_ERR_PLIB,"Total number of field components %D != block size %D",i,bs);
      }
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"      </CellData>\n");CHKERRQ(ierr);

      /*
       * Point Data headers
       */
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"      <PointData>\n");CHKERRQ(ierr);
      for (link=vtk->link; link; link=link->next) {
        Vec        X = (Vec)link->vec;
        DM         dmX;
        PetscInt   bs,nfields,field;
        const char *vecname = "";
        PetscSection section;
        if ((link->ft != PETSC_VTK_POINT_FIELD) && (link->ft != PETSC_VTK_POINT_VECTOR_FIELD)) continue;
        if (((PetscObject)X)->name || link != vtk->link) { /* If the object is already named, use it. If it is past the first link, name it to disambiguate. */
          ierr = PetscObjectGetName((PetscObject)X,&vecname);CHKERRQ(ierr);
        }
        ierr = VecGetDM(X, &dmX);CHKERRQ(ierr);
        if (!dmX) dmX = dm;
        ierr = PetscObjectQuery(link->vec, "section", (PetscObject*) &section);CHKERRQ(ierr);
        if (!section) {ierr = DMGetLocalSection(dmX, &section);CHKERRQ(ierr);}
        ierr = PetscSectionGetDof(section,vStart,&bs);CHKERRQ(ierr);
        ierr = PetscSectionGetNumFields(section,&nfields);CHKERRQ(ierr);
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
            ierr = PetscSectionGetFieldDof(section,vStart,field,&fbs);CHKERRQ(ierr);
            ierr = PetscSectionGetFieldName(section,field,&fieldname);CHKERRQ(ierr);
          } else fbs = bs;      /* Say we have one field with 'bs' components */
          if (nfields && !fieldname) {
            ierr = PetscSNPrintf(buf,sizeof(buf),"PointField%D",field);CHKERRQ(ierr);
            fieldname = buf;
          }
          if (link->ft == PETSC_VTK_POINT_VECTOR_FIELD) {
            if (fbs > 3) SETERRQ1(PetscObjectComm((PetscObject)viewer),PETSC_ERR_ARG_SIZ,"Point vector fields can have at most 3 components, %D given\n", fbs);
            ierr = PetscFPrintf(comm,fp,"        <DataArray type=\"%s\" Name=\"%s%s\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%D\" />\n",precision,vecname,fieldname,boffset);CHKERRQ(ierr);
            boffset += gpiece[r].nvertices*3*sizeof(PetscScalar) + sizeof(int);
          } else {
            for (j=0; j<fbs; j++) {
              if (fbs > 1) {
                ierr = PetscFPrintf(comm,fp,"        <DataArray type=\"%s\" Name=\"%s%s.%D\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%D\" />\n",precision,vecname,fieldname,j,boffset);CHKERRQ(ierr);
              } else {
                ierr = PetscFPrintf(comm,fp,"        <DataArray type=\"%s\" Name=\"%s%s\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%D\" />\n",precision,vecname,fieldname,boffset);CHKERRQ(ierr);
              }
              boffset += gpiece[r].nvertices*sizeof(PetscScalar) + sizeof(int);
            }
          }
        }
      }
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"      </PointData>\n");CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"    </Piece>\n");CHKERRQ(ierr);
    }
  }

  ierr = PetscFPrintf(comm,fp,"  </UnstructuredGrid>\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fp,"  <AppendedData encoding=\"raw\">\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fp,"_");CHKERRQ(ierr);

  if (!rank) {
    PetscInt maxsize = 0;
    for (r=0; r<size; r++) {
      maxsize = PetscMax(maxsize, (PetscInt) (gpiece[r].nvertices*3*sizeof(PetscScalar)));
      maxsize = PetscMax(maxsize, (PetscInt) (gpiece[r].ncells*3*sizeof(PetscScalar)));
      maxsize = PetscMax(maxsize, (PetscInt) (gpiece[r].nconn*sizeof(PetscVTKInt)));
    }
    ierr = PetscMalloc(maxsize,&buffer);CHKERRQ(ierr);
  }
  for (r=0; r<size; r++) {
    if (r == rank) {
      PetscInt nsend;
      {                         /* Position */
        const PetscScalar *x;
        PetscScalar       *y = NULL;
        Vec               coords;

        ierr  = DMGetCoordinatesLocal(dm,&coords);CHKERRQ(ierr);
        ierr  = VecGetArrayRead(coords,&x);CHKERRQ(ierr);
        if (dimEmbed != 3 || localized) {
          ierr = PetscMalloc1(piece.nvertices*3,&y);CHKERRQ(ierr);
          if (localized) {
            PetscInt cnt;
            for (c=cStart,cnt=0; c<cEnd; c++) {
              PetscInt off, dof;

              ierr = PetscSectionGetDof(coordSection, c, &dof);CHKERRQ(ierr);
              if (!dof) {
                PetscInt *closure = NULL;
                PetscInt closureSize;

                ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
                for (v = 0; v < closureSize*2; v += 2) {
                  if ((closure[v] >= vStart) && (closure[v] < vEnd)) {
                    ierr = PetscSectionGetOffset(coordSection, closure[v], &off);CHKERRQ(ierr);
                    if (dimEmbed != 3) {
                      y[cnt*3+0] = x[off+0];
                      y[cnt*3+1] = (dimEmbed > 1) ? x[off+1] : 0.0;
                      y[cnt*3+2] = 0.0;
                    } else {
                      y[cnt*3+0] = x[off+0];
                      y[cnt*3+1] = x[off+1];
                      y[cnt*3+2] = x[off+2];
                    }
                    cnt++;
                  }
                }
                ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
              } else {
                ierr = PetscSectionGetOffset(coordSection, c, &off);CHKERRQ(ierr);
                if (dimEmbed != 3) {
                  for (i=0; i<dof/dimEmbed; i++) {
                    y[cnt*3+0] = x[off + i*dimEmbed + 0];
                    y[cnt*3+1] = (dimEmbed > 1) ? x[off + i*dimEmbed + 1] : 0.0;
                    y[cnt*3+2] = 0.0;
                    cnt++;
                  }
                } else {
                  for (i=0; i<dof; i ++) {
                    y[cnt*3+i] = x[off + i];
                  }
                  cnt += dof/dimEmbed;
                }
              }
            }
            if (cnt != piece.nvertices) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Count does not match");
          } else {
            for (i=0; i<piece.nvertices; i++) {
              y[i*3+0] = x[i*dimEmbed+0];
              y[i*3+1] = (dimEmbed > 1) ? x[i*dimEmbed+1] : 0;
              y[i*3+2] = 0.0;
            }
          }
        }
        nsend = piece.nvertices*3;
        ierr  = TransferWrite(viewer,fp,r,0,y ? y : x,buffer,nsend,MPIU_SCALAR,tag);CHKERRQ(ierr);
        ierr  = PetscFree(y);CHKERRQ(ierr);
        ierr  = VecRestoreArrayRead(coords,&x);CHKERRQ(ierr);
      }
      {                           /* Connectivity, offsets, types */
        PetscVTKInt  *connectivity = NULL, *offsets = NULL;
        PetscVTKType *types = NULL;
        ierr = DMPlexGetVTKConnectivity(dm,localized,&piece,&connectivity,&offsets,&types);CHKERRQ(ierr);
        ierr = TransferWrite(viewer,fp,r,0,connectivity,buffer,piece.nconn,MPI_INT,tag);CHKERRQ(ierr);
        ierr = TransferWrite(viewer,fp,r,0,offsets,buffer,piece.ncells,MPI_INT,tag);CHKERRQ(ierr);
        ierr = TransferWrite(viewer,fp,r,0,types,buffer,piece.ncells,MPI_CHAR,tag);CHKERRQ(ierr);
        ierr = PetscFree3(connectivity,offsets,types);CHKERRQ(ierr);
      }
      {                         /* Owners (cell data) */
        PetscVTKInt *owners;
        ierr = PetscMalloc1(piece.ncells,&owners);CHKERRQ(ierr);
        for (i=0; i<piece.ncells; i++) owners[i] = rank;
        ierr = TransferWrite(viewer,fp,r,0,owners,buffer,piece.ncells,MPI_INT,tag);CHKERRQ(ierr);
        ierr = PetscFree(owners);CHKERRQ(ierr);
      }
      /* Cell data */
      for (link=vtk->link; link; link=link->next) {
        Vec               X = (Vec)link->vec;
        DM                dmX;
        const PetscScalar *x;
        PetscScalar       *y;
        PetscInt          bs, nfields, field;
        PetscSection      section = NULL;

        if ((link->ft != PETSC_VTK_CELL_FIELD) && (link->ft != PETSC_VTK_CELL_VECTOR_FIELD)) continue;
        ierr = VecGetDM(X, &dmX);CHKERRQ(ierr);
        if (!dmX) dmX = dm;
        ierr = PetscObjectQuery(link->vec, "section", (PetscObject*) &section);CHKERRQ(ierr);
        if (!section) {ierr = DMGetLocalSection(dmX, &section);CHKERRQ(ierr);}
        ierr = PetscSectionGetDof(section,cStart,&bs);CHKERRQ(ierr);
        ierr = PetscSectionGetNumFields(section,&nfields);CHKERRQ(ierr);
        field = 0;
        if (link->field >= 0) {
          field = link->field;
          nfields = field + 1;
        }
        ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
        ierr = PetscMalloc1(piece.ncells*3,&y);CHKERRQ(ierr);
        for (i=0; field<(nfields?nfields:1); field++) {
          PetscInt     fbs,j;
          PetscFV      fv = NULL;
          PetscObject  f;
          PetscClassId fClass;
          PetscBool    vector;
          if (nfields) {        /* We have user-defined fields/components */
            ierr = PetscSectionGetFieldDof(section,cStart,field,&fbs);CHKERRQ(ierr);
          } else fbs = bs;      /* Say we have one field with 'bs' components */
          ierr = DMGetField(dmX,field,NULL,&f);CHKERRQ(ierr);
          ierr = PetscObjectGetClassId(f,&fClass);CHKERRQ(ierr);
          if (fClass == PETSCFV_CLASSID) {
            fv = (PetscFV) f;
          }
          vector = PETSC_FALSE;
          if (link->ft == PETSC_VTK_CELL_VECTOR_FIELD) {
            vector = PETSC_TRUE;
            for (j = 0; j < fbs; j++) {
              const char *compName = NULL;
              if (fv) {
                ierr = PetscFVGetComponentName(fv,j,&compName);CHKERRQ(ierr);
                if (compName) break;
              }
            }
            if (j < fbs) vector = PETSC_FALSE;
          }
          if (vector) {
            PetscInt cnt;
            for (c=cStart,cnt=0; c<cEnd; c++) {
              const PetscScalar *xpoint;
              PetscInt off, j;

              if (hasLabel) {     /* Ignore some cells */
                PetscInt value;
                ierr = DMGetLabelValue(dmX, "vtk", c, &value);CHKERRQ(ierr);
                if (value != 1) continue;
              }
              if (nfields) {
                ierr = PetscSectionGetFieldOffset(section, c, field, &off);CHKERRQ(ierr);
              } else {
                ierr = PetscSectionGetOffset(section, c, &off);CHKERRQ(ierr);
              }
              xpoint = &x[off];
              for (j = 0; j < fbs; j++) {
                y[cnt++] = xpoint[j];
              }
              for (; j < 3; j++) y[cnt++] = 0.;
            }
            if (cnt != piece.ncells*3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Count does not match");
            ierr = TransferWrite(viewer,fp,r,0,y,buffer,piece.ncells*3,MPIU_SCALAR,tag);CHKERRQ(ierr);
          } else {
            for (i=0; i<fbs; i++) {
              PetscInt cnt;
              for (c=cStart,cnt=0; c<cEnd; c++) {
                const PetscScalar *xpoint;
                PetscInt off;

                if (hasLabel) {     /* Ignore some cells */
                  PetscInt value;
                  ierr = DMGetLabelValue(dmX, "vtk", c, &value);CHKERRQ(ierr);
                  if (value != 1) continue;
                }
                if (nfields) {
                  ierr = PetscSectionGetFieldOffset(section, c, field, &off);CHKERRQ(ierr);
                } else {
                  ierr = PetscSectionGetOffset(section, c, &off);CHKERRQ(ierr);
                }
                xpoint   = &x[off];
                y[cnt++] = xpoint[i];
              }
              if (cnt != piece.ncells) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Count does not match");
              ierr = TransferWrite(viewer,fp,r,0,y,buffer,piece.ncells,MPIU_SCALAR,tag);CHKERRQ(ierr);
            }
          }
        }
        ierr = PetscFree(y);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
      }
      /* point data */
      for (link=vtk->link; link; link=link->next) {
        Vec               X = (Vec)link->vec;
        DM                dmX;
        const PetscScalar *x;
        PetscScalar       *y;
        PetscInt          bs, nfields, field;
        PetscSection      section = NULL;

        if ((link->ft != PETSC_VTK_POINT_FIELD) && (link->ft != PETSC_VTK_POINT_VECTOR_FIELD)) continue;
        ierr = VecGetDM(X, &dmX);CHKERRQ(ierr);
        if (!dmX) dmX = dm;
        ierr = PetscObjectQuery(link->vec, "section", (PetscObject*) &section);CHKERRQ(ierr);
        if (!section) {ierr = DMGetLocalSection(dmX, &section);CHKERRQ(ierr);}
        ierr = PetscSectionGetDof(section,vStart,&bs);CHKERRQ(ierr);
        ierr = PetscSectionGetNumFields(section,&nfields);CHKERRQ(ierr);
        field = 0;
        if (link->field >= 0) {
          field = link->field;
          nfields = field + 1;
        }
        ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
        ierr = PetscMalloc1(piece.nvertices*3,&y);CHKERRQ(ierr);
        for (i=0; field<(nfields?nfields:1); field++) {
          PetscInt   fbs,j;
          if (nfields) {        /* We have user-defined fields/components */
            ierr = PetscSectionGetFieldDof(section,vStart,field,&fbs);CHKERRQ(ierr);
          } else fbs = bs;      /* Say we have one field with 'bs' components */
          if (link->ft == PETSC_VTK_POINT_VECTOR_FIELD) {
            PetscInt cnt;
            if (!localized) {
              for (v=vStart,cnt=0; v<vEnd; v++) {
                PetscInt    off;
                const PetscScalar *xpoint;

                if (nfields) {
                  ierr     = PetscSectionGetFieldOffset(section,v,field,&off);CHKERRQ(ierr);
                } else {
                  ierr     = PetscSectionGetOffset(section,v,&off);CHKERRQ(ierr);
                }
                xpoint   = &x[off];
                for (j = 0; j < fbs; j++) {
                  y[cnt++] = xpoint[j];
                }
                for (; j < 3; j++) y[cnt++] = 0.;
              }
            } else {
              for (c=cStart,cnt=0; c<cEnd; c++) {
                PetscInt *closure = NULL;
                PetscInt  closureSize, off;

                ierr = DMPlexGetTransitiveClosure(dmX, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
                for (v = 0, off = 0; v < closureSize*2; v += 2) {
                  if ((closure[v] >= vStart) && (closure[v] < vEnd)) {
                    PetscInt    voff;
                    const PetscScalar *xpoint;

                    if (nfields) {
                      ierr = PetscSectionGetFieldOffset(section,closure[v],field,&voff);CHKERRQ(ierr);
                    } else {
                      ierr = PetscSectionGetOffset(section,closure[v],&voff);CHKERRQ(ierr);
                    }
                    xpoint         = &x[voff];
                    for (j = 0; j < fbs; j++) {
                      y[cnt + off++] = xpoint[i];
                    }
                    for (; j < 3; j++) y[cnt + off++] = 0.;
                  }
                }
                cnt += off;
                ierr = DMPlexRestoreTransitiveClosure(dmX, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
              }
            }
            if (cnt != piece.nvertices*3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Count does not match");
            ierr = TransferWrite(viewer,fp,r,0,y,buffer,piece.nvertices*3,MPIU_SCALAR,tag);CHKERRQ(ierr);
          } else {
            for (i=0; i<fbs; i++) {
              PetscInt cnt;
              if (!localized) {
                for (v=vStart,cnt=0; v<vEnd; v++) {
                  PetscInt    off;
                  const PetscScalar *xpoint;

                  if (nfields) {
                    ierr     = PetscSectionGetFieldOffset(section,v,field,&off);CHKERRQ(ierr);
                  } else {
                    ierr     = PetscSectionGetOffset(section,v,&off);CHKERRQ(ierr);
                  }
                  xpoint   = &x[off];
                  y[cnt++] = xpoint[i];
                }
              } else {
                for (c=cStart,cnt=0; c<cEnd; c++) {
                  PetscInt *closure = NULL;
                  PetscInt  closureSize, off;

                  ierr = DMPlexGetTransitiveClosure(dmX, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
                  for (v = 0, off = 0; v < closureSize*2; v += 2) {
                    if ((closure[v] >= vStart) && (closure[v] < vEnd)) {
                      PetscInt    voff;
                      const PetscScalar *xpoint;

                      if (nfields) {
                        ierr           = PetscSectionGetFieldOffset(section,closure[v],field,&voff);CHKERRQ(ierr);
                      } else {
                        ierr           = PetscSectionGetOffset(section,closure[v],&voff);CHKERRQ(ierr);
                      }
                      xpoint         = &x[voff];
                      y[cnt + off++] = xpoint[i];
                    }
                  }
                  cnt += off;
                  ierr = DMPlexRestoreTransitiveClosure(dmX, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
                }
              }
              if (cnt != piece.nvertices) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Count does not match");
              ierr = TransferWrite(viewer,fp,r,0,y,buffer,piece.nvertices,MPIU_SCALAR,tag);CHKERRQ(ierr);
            }
          }
        }
        ierr = PetscFree(y);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
      }
    } else if (!rank) {
      ierr = TransferWrite(viewer,fp,r,0,NULL,buffer,gpiece[r].nvertices*3,MPIU_SCALAR,tag);CHKERRQ(ierr); /* positions */
      ierr = TransferWrite(viewer,fp,r,0,NULL,buffer,gpiece[r].nconn,MPI_INT,tag);CHKERRQ(ierr); /* connectivity */
      ierr = TransferWrite(viewer,fp,r,0,NULL,buffer,gpiece[r].ncells,MPI_INT,tag);CHKERRQ(ierr); /* offsets */
      ierr = TransferWrite(viewer,fp,r,0,NULL,buffer,gpiece[r].ncells,MPI_CHAR,tag);CHKERRQ(ierr); /* types */
      ierr = TransferWrite(viewer,fp,r,0,NULL,buffer,gpiece[r].ncells,MPI_INT,tag);CHKERRQ(ierr); /* owner rank (cells) */
      /* all cell data */
      for (link=vtk->link; link; link=link->next) {
        Vec               X = (Vec)link->vec;
        PetscInt bs, nfields, field;
        DM           dmX;
        PetscSection section = NULL;

        if ((link->ft != PETSC_VTK_CELL_FIELD) && (link->ft != PETSC_VTK_CELL_VECTOR_FIELD)) continue;
        ierr = VecGetDM(X, &dmX);CHKERRQ(ierr);
        if (!dmX) dmX = dm;
        ierr = PetscObjectQuery(link->vec, "section", (PetscObject*) &section);CHKERRQ(ierr);
        if (!section) {ierr = DMGetLocalSection(dmX, &section);CHKERRQ(ierr);}
        ierr = PetscSectionGetDof(section,cStart,&bs);CHKERRQ(ierr);
        ierr = PetscSectionGetNumFields(section,&nfields);CHKERRQ(ierr);
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
          if (nfields) {        /* We have user-defined fields/components */
            ierr = PetscSectionGetFieldDof(section,cStart,field,&fbs);CHKERRQ(ierr);
          } else fbs = bs;      /* Say we have one field with 'bs' components */
          ierr = DMGetField(dmX,field,NULL,&f);CHKERRQ(ierr);
          ierr = PetscObjectGetClassId(f,&fClass);CHKERRQ(ierr);
          if (fClass == PETSCFV_CLASSID) {
            fv = (PetscFV) f;
          }
          vector = PETSC_FALSE;
          if (link->ft == PETSC_VTK_CELL_VECTOR_FIELD) {
            vector = PETSC_TRUE;
            for (j = 0; j < fbs; j++) {
              const char *compName = NULL;
              if (fv) {
                ierr = PetscFVGetComponentName(fv,j,&compName);CHKERRQ(ierr);
                if (compName) break;
              }
            }
            if (j < fbs) vector = PETSC_FALSE;
          }
          if (vector) {
            ierr = TransferWrite(viewer,fp,r,0,NULL,buffer,gpiece[r].ncells*3,MPIU_SCALAR,tag);CHKERRQ(ierr);
          } else {
            for (i=0; i<fbs; i++) {
              ierr = TransferWrite(viewer,fp,r,0,NULL,buffer,gpiece[r].ncells,MPIU_SCALAR,tag);CHKERRQ(ierr);
            }
          }
        }
      }
      /* all point data */
      for (link=vtk->link; link; link=link->next) {
        Vec               X = (Vec)link->vec;
        DM                dmX;
        PetscInt bs, nfields, field;
        PetscSection section = NULL;

        if ((link->ft != PETSC_VTK_POINT_FIELD) && (link->ft != PETSC_VTK_POINT_VECTOR_FIELD)) continue;
        ierr = VecGetDM(X, &dmX);CHKERRQ(ierr);
        if (!dmX) dmX = dm;
        ierr = PetscObjectQuery(link->vec, "section", (PetscObject*) &section);CHKERRQ(ierr);
        if (!section) {ierr = DMGetLocalSection(dmX, &section);CHKERRQ(ierr);}
        ierr = PetscSectionGetDof(section,vStart,&bs);CHKERRQ(ierr);
        ierr = PetscSectionGetNumFields(section,&nfields);CHKERRQ(ierr);
        field = 0;
        if (link->field >= 0) {
          field = link->field;
          nfields = field + 1;
        }
        for (i=0; field<(nfields?nfields:1); field++) {
          PetscInt   fbs;
          if (nfields) {        /* We have user-defined fields/components */
            ierr = PetscSectionGetFieldDof(section,vStart,field,&fbs);CHKERRQ(ierr);
          } else fbs = bs;      /* Say we have one field with 'bs' components */
          if (link->ft == PETSC_VTK_POINT_VECTOR_FIELD) {
            ierr = TransferWrite(viewer,fp,r,0,NULL,buffer,gpiece[r].nvertices*3,MPIU_SCALAR,tag);CHKERRQ(ierr);
          } else {
            for (i=0; i<fbs; i++) {
              ierr = TransferWrite(viewer,fp,r,0,NULL,buffer,gpiece[r].nvertices,MPIU_SCALAR,tag);CHKERRQ(ierr);
            }
          }
        }
      }
    }
  }
  ierr = PetscFree(gpiece);CHKERRQ(ierr);
  ierr = PetscFree(buffer);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fp,"\n  </AppendedData>\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fp,"</VTKFile>\n");CHKERRQ(ierr);
  ierr = PetscFClose(comm,fp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
