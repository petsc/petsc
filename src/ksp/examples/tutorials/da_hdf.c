#include <hdf/mfhdf.h>

static int
DAVecHDFOutput2d(DA da, Vec X, char *fname)
{
	int32 sd_id, sds_id, pos[2], dims[2], zero[2], l = 0, b = 0;
	int ierr, xs, ys, xm, ym, i, j, k, w, rank;
	float *vf;
	char str[10], *name;
	PetscScalar **x;
	
	PetscFunctionBegin;

	ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);
	ierr = PetscMalloc(strlen(fname) + 10, &name); CHKERRQ(ierr);
	sprintf(name, "%d-%s", rank, fname);

	sd_id = SDstart(name, DFACC_CREATE);
	if (sd_id < 0) {
		SETERRQ1(1, "SDstart failed for %s", name);
	}
	ierr = PetscFree(name); CHKERRQ(ierr);

	ierr = DAGetInfo(da,0,0,0,0,0,0,0,&w,0,0,0);CHKERRQ(ierr);  
	ierr = DAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
	ierr = DAVecGetArray(da, X, (void *)&x);
	ierr = PetscMalloc(xm*ym * sizeof(*vf), &vf); CHKERRQ(ierr);

	pos [0] = xs; pos [1] = ys;
	dims[0] = xm; dims[1] = ym;
	zero[0] = 0;  zero[1] = 0;
	
	for (k = 0; k < w; k++) {
	  for (j=ys; j<ys+ym; j++){
	    for(i=xs; i<xs+xm; i++){
		vf[(i-xs) + (j-ys) * xm] = x[j][i*w + k];
	    }
	  }
	  sprintf(str, "Vec%d", k);
	  sds_id = SDcreate(sd_id, str, DFNT_FLOAT32, 2, dims);
	  if (sds_id < 0) {
	    SETERRQ1(1, "SDcreate failed for %s", str);
	  }
	  SDsetattr(sds_id, "pos", DFNT_INT32, 2, pos);
	  SDsetattr(sds_id, "bnd", DFNT_INT32, 1, &b);
	  SDsetattr(sds_id, "level", DFNT_INT32, 1, &l);
	  SDwritedata(sds_id, zero, 0, dims, vf);
	  SDendaccess(sds_id);
	}
	SDend(sd_id);

	ierr = PetscFree(vf);
	ierr = DAVecRestoreArray(da, X, (void *)&x); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

static int
DAVecHDFOutput3d(DA da, Vec X, char *fname)
{
	int32 sd_id, sds_id, pos[3], dims[3], zero[3];
	int ierr, xs, ys, zs, xm, ym, zm, i, j, k, l, w, rank;
	float *vf;
	char str[10], *name;
	PetscScalar ***x;
	
	PetscFunctionBegin;

	ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);
	ierr = PetscMalloc(strlen(fname) + 10, &name); CHKERRQ(ierr);
	sprintf(name, "%d-%s", rank, fname);

	sd_id = SDstart(name, DFACC_CREATE);
	if (sd_id < 0) {
		SETERRQ1(1, "SDstart failed for %s", name);
	}
	ierr = PetscFree(name); CHKERRQ(ierr);

	ierr = DAGetInfo(da,0,0,0,0,0,0,0,&w,0,0,0);CHKERRQ(ierr);  
	ierr = DAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
	ierr = DAVecGetArray(da, X, (void *)&x);
	ierr = PetscMalloc(xm*ym*zm * sizeof(*vf), &vf); CHKERRQ(ierr);

	pos [0] = xs; pos [1] = ys; pos [2] = zs;
	dims[0] = xm; dims[1] = ym; dims[2] = zm;
	zero[0] = 0;  zero[1] = 0;  zero[2] = 0;
	
	for (l = 0; l < w; l++) {
	  for (k=zs; k<zs+zm; k++){
	    for (j=ys; j<ys+ym; j++){
	      for(i=xs; i<xs+xm; i++){
		vf[(i-xs) + ((j-ys) + (k-zs) * ym)* xm] = x[k][j][i*w + l];
	      }
	    }
	  }
	  sprintf(str, "Vec%d", k);
	  sds_id = SDcreate(sd_id, str, DFNT_FLOAT32, 3, dims);
	  if (sds_id < 0) {
	    SETERRQ1(1, "SDcreate failed for %s", str);
	  }
	  SDsetattr(sds_id, "pos", DFNT_INT32, 3, pos);
	  SDwritedata(sds_id, zero, 0, dims, vf);
	  SDendaccess(sds_id);
	}
	SDend(sd_id);

	ierr = PetscFree(vf);
	ierr = DAVecRestoreArray(da, X, (void *)&x); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

int
DAVecHDFOutput(DA da, Vec X, char *fname)
{
	int ierr, dim;

	PetscFunctionBegin;
	ierr = DAGetInfo(da,&dim,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);  
	switch (dim) {
	case 2:
		ierr = DAVecHDFOutput2d(da, X, fname); CE;
		break;
	case 3:
		ierr = DAVecHDFOutput3d(da, X, fname); CE;
		break;
	default:
		SETERRQ1(1, "No support for %d dims", dim);
	}
	PetscFunctionReturn(0);
}

