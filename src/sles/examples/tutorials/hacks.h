#undef __FUNCT__
#define __FUNCT__ __FUNCTION__

#define HERE do { int ierr, rank; ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] LINE %d (%s)\n", rank, __LINE__, __FUNCTION__);CHKERRQ(ierr); } while (0)

#define CE do { CHKERRQ(ierr); } while (0)


#define sqr(a) ((a)*(a))

#include "src/dm/da/daimpl.h"    /*I   "petscda.h"   I*/

static inline void
__for_each_point_first_2d(DA da, int *i, int *j)
{
	*i = da->xs / da->w - 1;
	*j = da->ys;
}

static inline int
__for_each_point_next_2d(DA da, int *i, int *j)
{
	if (++(*i) >= da->xe / da->w) {
		if (++(*j) >= da->ye)
			return 0;
		*i = da->xs / da->w;
	}
	return 1;
}

#define DA_for_each_point_2d(da, i, j) \
        for (__for_each_point_first_2d(da, &(i), &(j)); \
             __for_each_point_next_2d(da, &(i), &(j)); )



#define D_x(p, f)  (d2Hx * (p[j][i+1].f - p[j][i-1].f))
#define D_y(p, f)  (d2Hy * (p[j+1][i].f - p[j-1][i].f))
#define D_xy(p, f) (d2Hx*d2Hy * (p[j+1][i+1].f - p[j-1][i+1].f - p[j+1][i-1].f + p[j-1][i-1].f))
#define D_x2(p, f) (dHx2 * (p[j][i+1].f - 2.* p[j][i].f + p[j][i-1].f))
#define D_y2(p, f) (dHy2 * (p[j+1][i].f - 2.* p[j][i].f + p[j-1][i].f))
#define D_2(p, f)  (D_x2(p, f) + D_y2(p, f))
