#include "hacks.h"

static char help[] = "MRC 2D.\n\n";

#include "petscda.h"
#include "petscmg.h"
#include "petscsles.h"


#define EQ
/*#define FISHPAK */

#ifdef FISHPAK
#include "cyclic.h"
#endif

typedef struct _p_MRC *MRC;

typedef struct _p_Poi *Poi;

#ifdef FISHPAK

struct _p_Poi {
	MRC mrc;
	DA da;
};

#else

struct _p_Poi {
	MRC  mrc;
	DMMG *dmmg;
};

#endif

struct _p_MRC {
	Vec b, bp;
	Poi poi;
	DA da;
	PetscReal pert;
	PetscReal de;
	PetscReal rhos;
	PetscReal eps;
	PetscReal t, dt;
	PetscReal dt_out;
	PetscReal dt_spec_out;
	PetscReal Sx, Sy;
	PetscReal Lx, Ly;
};

typedef struct {
	PetscScalar U;
	PetscScalar F;
} Fld;

typedef struct {
	PetscScalar phi;
	PetscScalar psi;
} Pot;

static MRC _mrc; // FIXME

int w = sizeof(Fld) / sizeof(PetscScalar);

PetscScalar zero = 0., one = 1., mone = -1., two = 2.;
PetscReal min_dt = .001;
PetscReal cfl_thres = 3; // 1.
PetscReal t_end = 10000;

#undef __FUNCT__
#define __FUNCT__ "CalcRhs"
int CalcRhs(MRC mrc, Vec B, Vec X, Vec Rhs, PetscReal *cfl)
{
	int ierr;
	int i, j, mx, my;
	Vec Xlocal, Blocal;
	DA da = mrc->da;
	PetscReal Hx, Hy, d2Hx, d2Hy, dHx2, dHy2;
	PetscReal de2 = sqr(mrc->de), rhos2 = sqr(mrc->rhos), dt = mrc->dt;
	PetscReal eta = 0, nu = 0;
	PetscScalar by_eq, F_eq_x, ux, uy, bx, by;
	Fld **f, **rhs;
	Pot **p;

	PetscFunctionBegin;
	ierr = DAGetInfo(da, 0, &mx, &my, 0, 0, 0, 0, 0, 0, 0, 0); CE;

	Hx = mrc->Lx / mx;  Hy = mrc->Ly / my;
	d2Hx = .5 / Hx;     d2Hy = .5 / Hy;
	dHx2 = 1./ (Hx*Hx); dHy2 = 1./ (Hy*Hy);

	ierr = DAGetLocalVector(da, &Xlocal); CE;
	ierr = DAGlobalToLocalBegin(da, X, INSERT_VALUES, Xlocal); CE;
	ierr = DAGlobalToLocalEnd  (da, X, INSERT_VALUES, Xlocal); CE;

	ierr = DAGetLocalVector(da, &Blocal); CE;
	ierr = DAGlobalToLocalBegin(da, B, INSERT_VALUES, Blocal); CE;
	ierr = DAGlobalToLocalEnd  (da, B, INSERT_VALUES, Blocal); CE;

	ierr = DAVecGetArray(da, Xlocal, (void **) &p); CE;
	ierr = DAVecGetArray(da, Blocal, (void **) &f); CE;
	ierr = DAVecGetArray(da, Rhs   , (void **) &rhs); CE;

	DA_for_each_point(da, i, j) {
#ifdef EQ
		  PetscReal x = mrc->Lx*i/mx;
		  by_eq  = sin(x);
		  F_eq_x = - (1. + de2) * sin(x);
#else
		  by_eq  = 0.;
		  F_eq_x = 0.;
#endif

  		  ux = - D_y(p, phi);
		  uy =   D_x(p, phi);
		  bx =   D_y(p, psi);
		  by = - D_x(p, psi) + by_eq;
		  rhs[j][i].U = - (ux *  D_x(f, U)           + uy *  D_y(f, U))
			        + (bx * (D_x(f, F) + F_eq_x) + by *  D_y(f, F)) / de2
			        +  nu *  D_2(f, U);
		  rhs[j][i].F = - (ux * (D_x(f, F) + F_eq_x) + uy *  D_y(f, F))
		                + (bx *  D_x(f, U) + by *  D_y(f, U)) * rhos2
			        + eta *  D_2(p, psi);

		  *cfl = PetscMax(*cfl, PetscAbs(ux*dt/Hx));
		  *cfl = PetscMax(*cfl, PetscAbs(uy*dt/Hy));
		  *cfl = PetscMax(*cfl, PetscAbs(bx*dt/Hx));
		  *cfl = PetscMax(*cfl, PetscAbs(by*dt/Hy));
	}

	ierr = DAVecRestoreArray(da, Xlocal, (void **)&p); CE;
	ierr = DAVecRestoreArray(da, Blocal, (void **)&f); CE;
	ierr = DAVecRestoreArray(da, Rhs   , (void **)&rhs); CE;

	ierr = DARestoreLocalVector(da, &Xlocal); CE;
	ierr = DARestoreLocalVector(da, &Blocal); CE;

	PetscFunctionReturn(0);
}

#ifdef FISHPAK

#undef __FUNCT__
#define __FUNCT__ "PoiCreate"
int PoiCreate(MRC mrc, Poi *in_poi)
{
	int ierr;
	Poi poi;

	PetscFunctionBegin;

	ierr = PetscMalloc(sizeof(*poi), &poi); CE;
	
	poi->mrc = mrc;
	ierr = DACreate2d(PETSC_COMM_WORLD,DA_XYPERIODIC, DA_STENCIL_STAR,
			  8 << (6-1), 8 << (6-1), PETSC_DECIDE, PETSC_DECIDE,
			  w, 1, 0, 0, &poi->da); CE;

	*in_poi = poi;
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PoiDestroy"
int PoiDestroy(Poi poi)
{
	int ierr;
	
	PetscFunctionBegin;

	ierr = DADestroy(poi->da); CE;
	ierr = PetscFree(poi); CE;

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PoiGetDA"
int PoiGetDA(Poi poi, DA *in_da)
{
	PetscFunctionBegin;

	*in_da = poi->da;

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PoiCalcRhs"
int PoiCalcRhs(Poi poi, Vec X, Vec B)
{
	int ierr;
	int i, j, mx, my;
	Vec Xlocal;
	MRC mrc = poi->mrc;
	DA da = poi->da;
	PetscReal Hx, Hy, d2Hx, d2Hy, dHx2, dHy2;
	PetscReal de2 = sqr(mrc->de);
	Fld **b;
	Pot **x;

	PetscFunctionBegin;
	ierr = DAGetInfo(da, 0, &mx, &my, 0, 0, 0, 0, 0, 0, 0, 0); CE;

	Hx = mrc->Lx / mx;  Hy = mrc->Ly / my;
	d2Hx = .5 / Hx;     d2Hy = .5 / Hy;
	dHx2 = 1./ (Hx*Hx); dHy2 = 1./ (Hy*Hy);

	ierr = DAGetLocalVector(da, &Xlocal); CE;
	ierr = DAGlobalToLocalBegin(da, X, INSERT_VALUES, Xlocal); CE;
	ierr = DAGlobalToLocalEnd  (da, X, INSERT_VALUES, Xlocal); CE;

	ierr = DAVecGetArray(da, Xlocal, (void *) &x); CE;
	ierr = DAVecGetArray(da, B,      (void *) &b); CE;

	DA_for_each_point(da, i, j) {
		b[j][i].U = D_2(x, phi);
		b[j][i].F = x[j][i].psi - de2 * D_2(x, psi);
	}
	ierr = DAVecRestoreArray(da, Xlocal, (void *) &x); CE;
	ierr = DAVecRestoreArray(da, B,      (void *) &b); CE;

	ierr = DARestoreLocalVector(da, &Xlocal); CE;

	PetscFunctionReturn(0);
}

double log2(double);

#undef __FUNCT__
#define __FUNCT__ "PoiSolve"
int PoiSolve(Poi poi, Vec B, Vec X)
{
	int ierr, i, j, rank, mx, my, mbdcnd, nbdcnd, MX, MY, MXS, sz;
	MRC mrc = poi->mrc;
	DA da = poi->da;
	Fld **b;
	Pot **x;
	PetscReal de2 = sqr(mrc->de);
	double *f, *w, __pertrb[2], xb[2], xe[2], lambda;

	PetscFunctionBegin;

	ierr = MPI_Comm_size(da->comm, &rank); CE;
	if (rank != 1) {
		SETERRQ(1, "serial only");
	}

	ierr = DAGetInfo(da, 0, &mx ,&my, 0, 0, 0, 0, 0, 0, 0, 0); CE;

	MX = mx;
	MY = my;
	sz = (MX+1) * (MY+1);
	MXS = MX + 1;

	xb[0]  = mrc->Sx;           xb[1]  = mrc->Sy;
	xe[0]  = mrc->Sx + mrc->Lx; xe[1]  = mrc->Sy + mrc->Ly;

	mbdcnd = 0; nbdcnd = 0;

	ierr = PetscMalloc(sizeof(*f) * sz, &f); CE;
	ierr = PetscMalloc(sizeof(*w) * 
			   (4*(MY+1) + (13+(int)(log2(MY+1)))*(MX+1)), &w); CE;

	ierr = DAVecGetArray(da, B, (void *) &b); CE;
	ierr = DAVecGetArray(da, X, (void *) &x); CE;

	// U / phi

	lambda = 0;
	DA_for_each_point(da, i, j) {
		f[i + j * MXS] = b[j][i].U;
	}
	for (j = 0; j < MY; j++) {
		f[MX + j * MXS] = f[0 + j * MXS];
	}
	for (i = 0; i <= MX; i++) {
		f[i + MY * MXS] = f[i + 0 * MXS];
	}
	hwscrt_(&xb[0], &xe[0], &MX, &mbdcnd, NULL, NULL,
		&xb[1], &xe[1], &MY, &nbdcnd, NULL, NULL,
		&lambda, f, &MXS, &__pertrb[0], &ierr, w); CE;
	
	DA_for_each_point(da, i, j) {
		x[j][i].phi = f[i + j * MXS];
	}

	// F / psi

	lambda = -1./de2;
	DA_for_each_point(da, i, j) {
		f[i + j * MXS] = lambda * b[j][i].F;
	}
	for (j = 0; j < MY; j++) {
		f[MX + j * MXS] = f[0 + j * MXS];
	}
	for (i = 0; i <= MX; i++) {
		f[i + MY * MXS] = f[i + 0 * MXS];
	}
	hwscrt_(&xb[0], &xe[0], &MX, &mbdcnd, NULL, NULL,
		&xb[1], &xe[1], &MY, &nbdcnd, NULL, NULL,
		&lambda, f, &MXS, &__pertrb[1], &ierr, w); CE;
	
	DA_for_each_point(da, i, j) {
		x[j][i].psi = f[i + j * MXS];
	}

	ierr = DAVecRestoreArray(da, B, (void *) &b); CE;
	ierr = DAVecRestoreArray(da, X, (void *) &x); CE;

	ierr = PetscFree(f); CE;
	ierr = PetscFree(w); CE;

	PetscFunctionReturn(0);
}

#else

static Vec _B;

#undef __FUNCT__
#define __FUNCT__ "ComputeRHS"
int ComputeRHS(DMMG dmmg, Vec b)
{
	int ierr;
	Vec B = _B; // FIXME

	PetscFunctionBegin;
	ierr = VecCopy(B, b); CE;
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeJacobian"
int ComputeJacobian(DMMG dmmg, Mat J)
{
	int          ierr, i, j, mx, my;
	DA           da = (DA)dmmg->dm;
	MRC          mrc = _mrc; // FIXME
	PetscScalar  v[5];
	PetscReal    Hx, Hy, HxdHy, HydHx, HxHy, Hdiag;
	PetscReal    de2 = sqr(mrc->de);
	MatStencil   row, col[7];
	
	ierr = DAGetInfo(da, 0, &mx ,&my, 0, 0, 0, 0, 0, 0, 0, 0); CE;

	Hx = mrc->Lx / mx;  Hy = mrc->Ly / my;
	HxdHy = Hx / Hy;    HydHx = Hy / Hx;
	HxHy = Hx * Hy;     Hdiag = -2.*(HxdHy + HydHx);

	DA_for_each_point(da, i, j) {
		row.i = i; row.j = j; row.c = 0;
		v[0] = HydHx;          col[0].i = i+1; col[0].j = j;   col[0].c = 0;
		v[1] = HydHx;          col[1].i = i-1; col[1].j = j;   col[1].c = 0;
		v[2] = HxdHy;          col[2].i = i;   col[2].j = j+1; col[2].c = 0;
		v[3] = HxdHy;          col[3].i = i;   col[3].j = j-1; col[3].c = 0;
		v[4] = Hdiag;          col[4].i = i;   col[4].j = j;   col[4].c = 0;
		ierr = MatSetValuesStencil(J, 1, &row, 5, col, v, INSERT_VALUES); CE;

		row.i = i; row.j = j; row.c = 1;
		v[0] = -de2*HydHx;     col[0].i = i+1; col[0].j = j;   col[0].c = 1;
		v[1] = -de2*HydHx;     col[1].i = i-1; col[1].j = j;   col[1].c = 1;
		v[2] = -de2*HxdHy;     col[2].i = i;   col[2].j = j+1; col[2].c = 1;
		v[3] = -de2*HxdHy;     col[3].i = i;   col[3].j = j-1; col[3].c = 1;
		v[4] = HxHy-de2*Hdiag; col[4].i = i;   col[4].j = j;   col[4].c = 1;
		ierr = MatSetValuesStencil(J, 1, &row, 5, col, v, INSERT_VALUES); CE;
	}
	ierr = MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY); CE;
	ierr = MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY); CE;
	return 0;
}

#undef __FUNCT__
#define __FUNCT__ "AttachNullSpace"
int AttachNullSpace(PC pc, Vec model)
{
	int ierr, i, N, start, end;
	PetscScalar scale;
	Vec V;
	Fld *v;
	MatNullSpace sp;

	PetscFunctionBegin;

	ierr  = VecDuplicate(model, &V); CE;
	ierr  = VecGetSize(model, &N); CE;
	scale = 1.0 / sqrt(N/2);
	ierr  = VecGetOwnershipRange(V, &start, &end); CE;

	ierr  = VecGetArray(V, (PetscScalar **) &v); CE;
	for (i = start/w; i < end/w; i++) {
		v[i].U = scale;
		v[i].F = 0.;
	}
	ierr = VecRestoreArray(V, (PetscScalar **) &v); CE;
	ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, 0, 1, &V, &sp); CE;
	ierr = VecDestroy(V); CE;
	ierr = PCNullSpaceAttach(pc, sp); CE;
	ierr = MatNullSpaceDestroy(sp); CE;

	PetscFunctionReturn(0);
}
 
#undef __FUNCT__
#define __FUNCT__ "PoiCreate"
int PoiCreate(MRC mrc, Poi *in_poi)
{
	int ierr, i;
	Poi poi;
	DA da;
	SLES sles, subsles;
	PC pc, subpc;
	PetscTruth flg;

	PetscFunctionBegin;

	ierr = PetscMalloc(sizeof(*poi), &poi); CE;
	
	poi->mrc = mrc;
	ierr = DMMGCreate(PETSC_COMM_WORLD, 6, PETSC_NULL, &poi->dmmg); CE;
	ierr = DACreate2d(PETSC_COMM_WORLD,DA_XYPERIODIC, DA_STENCIL_STAR,
			  8, 8, PETSC_DECIDE, PETSC_DECIDE,
			  w, 1, 0, 0, &da); CE;
	ierr = DMMGSetDM(poi->dmmg,(DM)da);
	ierr = DADestroy(da); CE;

	ierr = DMMGSetSLES(poi->dmmg, ComputeRHS, ComputeJacobian); CE;

	sles = DMMGGetSLES(poi->dmmg); CE;

	ierr = SLESGetPC(sles, &pc); CE;
	ierr = AttachNullSpace(pc, DMMGGetx(poi->dmmg)); CE;
	ierr = PetscTypeCompare((PetscObject)pc, PCMG, &flg); CE;
	if (flg) {
		for (i = 0; i < DMMGGetLevels(poi->dmmg); i++) {
			ierr = MGGetSmoother(pc, i, &subsles); CE;
			ierr = SLESGetPC(subsles, &subpc); CE;
			ierr = AttachNullSpace(subpc, poi->dmmg[i]->x); CE;
			ierr = PetscTypeCompare((PetscObject)subpc, PCLU, &flg); CE;
			if (flg) {
				PCLUSetDamping(subpc, 1e-12);
			}
		}
	}
	

	*in_poi = poi;
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PoiDestroy"
int PoiDestroy(Poi poi)
{
	int ierr;

	PetscFunctionBegin;

	ierr = DMMGDestroy(poi->dmmg); CE;
	ierr = PetscFree(poi); CE;

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PoiGetDA"
int PoiGetDA(Poi poi, DA *in_da)
{
	PetscFunctionBegin;

	*in_da = DMMGGetDA(poi->dmmg);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PoiCalcRhs"
int PoiCalcRhs(Poi poi, Vec X, Vec B)
{
	int ierr, mx, my;
	MRC mrc = poi->mrc;
	PetscScalar Hx, Hy, dHxHy;

	PetscFunctionBegin;

	ierr = DAGetInfo(DMMGGetDA(poi->dmmg), 0, &mx ,&my, 0, 0, 0, 0, 0, 0, 0, 0); CE;

	Hx = mrc->Lx / mx;  Hy = mrc->Ly / my;
	dHxHy = 1. / (Hx * Hy);

	ierr = MatMult(DMMGGetJ(poi->dmmg), X, B); CE;
	ierr = VecScale(&dHxHy, B); CE;
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PoiSolve"
int PoiSolve(Poi poi, Vec B, Vec X)
{
	int ierr, mx, my;
	MRC mrc = poi->mrc;
	PetscScalar Hx, Hy, HxHy, dHxHy;

	PetscFunctionBegin;

	ierr = DAGetInfo(DMMGGetDA(poi->dmmg), 0, &mx ,&my, 0, 0, 0, 0, 0, 0, 0, 0); CE;

	Hx = mrc->Lx / mx;  Hy = mrc->Ly / my;
	HxHy = Hx * Hy;     dHxHy = 1./HxHy;

	_B = B;
	ierr = DMMGSolve(poi->dmmg); CE;
	ierr = VecCopy(DMMGGetx(poi->dmmg), X); CE;
	ierr = VecScale(&HxHy, X); CE;
	
	PetscFunctionReturn(0);
}

#endif

#undef __FUNCT__
#define __FUNCT__ "IniPorcelli"
int IniPorcelli(MRC mrc)
{
	int ierr, mx, my, i, j;
	DA da = mrc->da;
	PetscReal x, y;
	Vec X;
	Pot **v;
	PetscReal k = 1. * mrc->eps;
	PetscReal gamma = k * mrc->de;
	PetscReal pert = mrc->pert;
	PetscReal de = mrc->de;
	
	PetscFunctionBegin;

	ierr = DAGetInfo(da, 0, &mx ,&my, 0, 0, 0, 0, 0, 0, 0, 0); CE;

	ierr = DAGetGlobalVector(da, &X); CE;
	ierr = DAVecGetArray(da, X, (void **)&v); CE;
	DA_for_each_point(da, i, j) {
		x = mrc->Sx + mrc->Lx*i/mx;
		y = mrc->Sy + mrc->Ly*j/my;
		
		if (x < -M_PI/2) {
			v[j][i].phi = pert * gamma / k * erf((x + M_PI) / (sqrt(2) * de)) * (-sin(k*y));
		} else if (x < M_PI/2) {
			v[j][i].phi = - pert * gamma / k * erf(x / (sqrt(2) * de)) * (-sin(k*y));
		} else if (x < 3*M_PI/2) {
			v[j][i].phi = pert * gamma / k * erf((x - M_PI) / (sqrt(2) * de)) * (-sin(k*y));
		} else {
			v[j][i].phi = - pert * gamma / k * erf((x - 2*M_PI) / (sqrt(2) * de)) * (-sin(k*y));
		}
#ifdef EQ
		v[j][i].psi = 0.;
#else
		v[j][i].psi = cos(x);
#endif
	}
	ierr = DAVecRestoreArray(da, X, (void **)&v); CE;

	ierr = PoiCalcRhs(mrc->poi, X, mrc->b); CE;
	ierr = VecSet(&zero, X);
	ierr = DARestoreGlobalVector(da, &X); CE;

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MRCCreate"
int MRCCreate(MRC *in_mrc)
{
	int ierr;
	MRC mrc;

	PetscFunctionBegin;

	ierr = PetscMalloc(sizeof(*mrc), &mrc); CE;

	_mrc = mrc; // FIXME

	mrc->t           = 0;
	mrc->dt          = 0.1;
	mrc->dt_out      = 10;
	mrc->dt_spec_out = 1;
	
	mrc->de     = 0.2;
	mrc->rhos   = 0;
	mrc->eps    = 0.5;
	mrc->pert   = 1e-4;

	mrc->Sx     = 0;
	mrc->Sy     = 0;
	mrc->Lx     = 2*M_PI;
	mrc->Ly     = 2*M_PI / mrc->eps;

	ierr = PetscOptionsBegin(PETSC_COMM_WORLD, PETSC_NULL, "MRC Options",
				 "MRC"); CE;
	ierr = PetscOptionsReal("-de", "electron skin depth", "MRC",
				mrc->de, &mrc->de, 0); CE;
	ierr = PetscOptionsReal("-rhos", "ion sound lamor radius", "MRC", 
				mrc->rhos, &mrc->rhos, 0); CE;
	ierr = PetscOptionsReal("-eps", "aspect ratio", "MRC", 
				mrc->eps, &mrc->eps, 0); CE;
	ierr = PetscOptionsReal("-pert", "initial perturbation", "MRC", 
				mrc->pert, &mrc->pert, 0); CE;
	ierr = PetscOptionsReal("-dt", "initial dt", "MRC", 
				mrc->dt, &mrc->dt, 0); CE;
	ierr = PetscOptionsReal("-dt_out", "dt between checkpoints", "MRC", 
				mrc->dt_out, &mrc->dt_out, 0); CE;
	ierr = PetscOptionsReal("-cfl_thres", "maximum cfl for time refinement", "MRC", 
				cfl_thres, &cfl_thres, 0); CE;
	ierr = PetscOptionsEnd(); CE;

	ierr = PetscPrintf(PETSC_COMM_WORLD, "de        = %g\n", mrc->de); CE;
	ierr = PetscPrintf(PETSC_COMM_WORLD, "rhos      = %g\n", mrc->rhos); CE;
	ierr = PetscPrintf(PETSC_COMM_WORLD, "eps       = %g\n", mrc->eps); CE;
	ierr = PetscPrintf(PETSC_COMM_WORLD, "pert      = %g\n", mrc->pert); CE;
	ierr = PetscPrintf(PETSC_COMM_WORLD, "dt        = %g\n", mrc->dt); CE;
	ierr = PetscPrintf(PETSC_COMM_WORLD, "dt_out    = %g\n", mrc->dt_out); CE;
	ierr = PetscPrintf(PETSC_COMM_WORLD, "cfl_thres = %g\n", cfl_thres); CE;

	ierr = PoiCreate(mrc, &mrc->poi); CE;
	ierr = PoiGetDA(mrc->poi, &mrc->da); CE;

	ierr = DACreateGlobalVector(mrc->da, &mrc->b); CE;
	ierr = DACreateGlobalVector(mrc->da, &mrc->bp); CE;

	*in_mrc = mrc;
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MRCSetUp"
int MRCSetUp(MRC mrc, int (*ini)(MRC))
{
	int ierr;

	PetscFunctionBegin;

	ierr = ini(mrc); CE;

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MRCDestroy"
int MRCDestroy(MRC mrc)
{
	int ierr;

	PetscFunctionBegin;

	ierr = VecDestroy(mrc->b); CE;
	ierr = VecDestroy(mrc->bp); CE;
	ierr = PoiDestroy(mrc->poi); CE;
	ierr = PetscFree(mrc); CE;

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MRCSpo1Output"
int MRCSpo1Output(MRC mrc, Vec X, FILE *spo)
{
	int ierr, mx, my, rank, i, j, tag, xs, ys, xm, ym;
	DA da = mrc->da;
	Pot **p;
	PetscReal values[4];
	PetscReal Hx, Hy, d2Hx, d2Hy, dHx2, dHy2;
	MPI_Status status;

	PetscFunctionBegin;

	ierr = DAGetInfo(da, 0, &mx, &my, 0, 0, 0, 0, 0, 0, 0, 0); CE;

	i = mx / 2; j = my / 2;

	Hx = mrc->Lx / mx;  Hy = mrc->Ly / my;
	d2Hx = .5 / Hx;     d2Hy = .5 / Hy;
	dHx2 = 1./ (Hx*Hx); dHy2 = 1./ (Hy*Hy);

	ierr = MPI_Comm_rank(da->comm, &rank); CE;
	ierr = PetscCommGetNewTag(da->comm, &tag); CE;
	ierr = DAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0); CE;
	if (i >= xs && i < xs + xm &&
	    j >= ys && j < ys + ym) {
		ierr = DAVecGetArray(da, X, (void **)&p); CE;
		values[0] = D_x2(p, psi);
		values[1] = D_y2(p, psi);
		values[2] = D_xy(p, phi);
		values[3] = p[j][i].psi;
		ierr = DAVecRestoreArray(da, X, (void **)&p); CE;
		if (rank != 0) {
			ierr = MPI_Send(values, 4, MPIU_REAL, 0, tag, da->comm); CE;
		}
	} else {
		if (rank == 0) {
			ierr = MPI_Recv(values, 4, MPIU_REAL, 0, tag, da->comm, &status); CE;
		}
	}
	ierr = PetscFPrintf(da->comm, spo, "\t%g\t%g\t%g\t%g", 
			    values[0], values[1], values[2], values[3]); CE;
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MRCSpecOutput"
int MRCSpecOutput(MRC mrc, Vec X)
{
	int ierr;
	DA da = mrc->da;
	PetscReal b_max[2] = {}, x_max[2] = {};
	static FILE *spo;
	static int opened = 0;

	PetscFunctionBegin;

	HERE;
	if (!opened) {
		ierr = PetscFOpen(da->comm, "spo.asc", "w", &spo); CE;
		ierr = PetscFPrintf(da->comm, spo, 
				    "time max(b[0]) max(b[1]) max(x[0]) max(x[1])\n"); CE;
		opened = 1;
	}
	ierr = VecStrideNorm(mrc->b, 0, NORM_MAX, &b_max[0]); CE;
	ierr = VecStrideNorm(mrc->b, 1, NORM_MAX, &b_max[1]); CE;
	ierr = VecStrideNorm(X, 0, NORM_MAX, &x_max[0]); CE;
	ierr = VecStrideNorm(X, 1, NORM_MAX, &x_max[1]); CE;

	PetscFPrintf(da->comm, spo, "%g %g %g %g %g", mrc->t,
		     b_max[0], b_max[1], x_max[0], x_max[1]);
	MRCSpo1Output(mrc, X, spo);
	PetscFPrintf(da->comm, spo, "\n");

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MRCOutput"
int MRCOutput(MRC mrc, const char *s, Vec V)
{
	int ierr;
	char fname[256];

	PetscFunctionBegin;

	HERE;
	sprintf(fname, "%s-%g.hdf", s, mrc->t);
	/*	ierr = DAVecHDFOutput(mrc->da, V, fname); CE; */

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MRCRun"
int MRCRun(MRC mrc)
{
	int ierr, antialias = 0;
	DA da = mrc->da;
	PetscReal next_out = mrc->t + mrc->dt_out;
	PetscReal next_spec_out = mrc->t + mrc->dt_spec_out;
	PetscReal cfl, two_dt, m_dt, dt;
	Vec x, rhs, b = mrc->b, bp = mrc->bp;

	PetscFunctionBegin;

	ierr = DAGetGlobalVector(da, &x); CE;
	ierr = DAGetGlobalVector(da, &rhs); CE;

	ierr = VecSet(&zero, x); CE;

	ierr = PoiSolve(mrc->poi, b, x); CE;

	ierr = MRCOutput(mrc, "b", b); CE;
	ierr = MRCOutput(mrc, "x", x); CE;

	// euler
	cfl = 0;
	ierr = CalcRhs(mrc, b, x, rhs, &cfl); CE;
	ierr = VecCopy(rhs, bp); CE;
	ierr = VecAXPY(&mrc->dt, rhs, b); CE;
	mrc->t += mrc->dt;

	do {
		ierr = PoiSolve(mrc->poi, b, x); CE;

		if (mrc->t - next_out >= -1e-7) {
			ierr = MRCOutput(mrc, "b", b); CE;
			ierr = MRCOutput(mrc, "x", x); CE;
			while (next_out <= mrc->t + 1e-7 )
				next_out += mrc->dt_out;
		}
		if (mrc->t - next_spec_out >= -1e-7) {
			ierr = MRCSpecOutput(mrc, x); CE;
			while (next_spec_out <= mrc->t + 1e-7 )
				next_spec_out += mrc->dt_spec_out;
		}

		ierr = PetscPrintf(PETSC_COMM_WORLD, "\t\t\t\t\tt = %g   \tcfl = %g\n",
				   mrc->t, cfl); CE;
		if (cfl > cfl_thres) {
			mrc->dt /= 2;
			if (mrc->dt < min_dt) {
				SETERRQ1(1, "dt = %g is too small\n", mrc->dt);
			}
		}

		// lf
		dt = mrc->dt;
		two_dt = 2. * mrc->dt;
		m_dt = -mrc->dt;
		cfl = 0;
		ierr = CalcRhs(mrc, b, x, rhs, &cfl); CE;
		if ((++antialias % 10) == 0) {
			ierr = VecAXPY(&m_dt, bp, b); CE;
			ierr = VecAXPY(&two_dt, rhs, b); CE;
			ierr = VecCopy(rhs, bp); CE;
		} else {
			ierr = VecScale(&two, rhs); CE;
			ierr = VecAYPX(&mone, rhs, bp); CE;
			ierr = VecAXPY(&dt, bp, b); CE;
		}
		mrc->t += dt;
	} while (mrc->t < t_end);

	ierr = DARestoreGlobalVector(da, &rhs); CE;
	ierr = DARestoreGlobalVector(da, &x); CE;

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
	int ierr;
	MRC mrc;
	
	PetscInitialize(&argc, &argv, (char *)0, help);

	ierr = MRCCreate(&mrc); CE;
	ierr = MRCSetUp(mrc, IniPorcelli); CE;
	ierr = MRCRun(mrc); CE;
	ierr = MRCDestroy(mrc); CE;

	ierr = PetscFinalize(); CE;
	
	return 0;
}


