/*                       
    This code implements a slight variant of the Enhanced
    BiCGStab(L) algorithm in (3) and (2).  The variation
    concerns cases when either kappa0**2 or kappa1**2 is
    negative due to round-off. Kappa0 has also been pulled
    out of the denominator in the formula for ghat, see
    KSPSolve_BCGSL for details.

    References:
      1. G.L.G. Sleijpen, H.A. van der Vorst, "An overview of
  approaches for the stable computation of hybrid BiCG
  methods", Applied Numerical Mathematics: Transactions 
  f IMACS, 19(3), pp 235-54, 1996. 
      2. G.L.G. Sleijpen, H.A. van der Vorst, D.R. Fokkema,
  "BiCGStab(L) and other hybrid Bi-CG methods",
  Numerical Algorithms, 7, pp 75-109, 1994.
      3. D.R. Fokkema, "Enhanced implementation of BiCGStab(L)
  for solving linear systems of equations", preprint
  from www.citeseer.com.

    Written by: Joel M. Malard, email jm.malard@pnl.gov

    Warning: anything free comes with no garantee ...,
    but do send me bug reports and your comments.
 
*/
#include "petsc.h"
#include "src/ksp/ksp/kspimpl.h"
#include "bcgsl.h"

/****************************************************
 *
 * Internal linear algebra functions.
 *
 ****************************************************/

#undef __FUNCT__  
#define __FUNCT__ "bcgsl_factr_i" 
int bcgsl_factr_i(int m, PetscScalar *A, int lda)
{
	int i, j, k;
	PetscScalar u, v, w;

	for ( i=0; i<m; i++ )
	{
		w = A[i+lda*i];

		/* we don't pivot but let's check anyways */
		if ( w!=w ) return 1;
		u = w;

		for ( j=i+1; j<m; j++ )
			if ( A[i+lda*i]+w==A[i+lda*i] )
			{
				return 1;
			}

		for( j=i+1; j<m; j++ )
		{
			v = -A[j+lda*i] / w;
			A[j+lda*i] = v;
			for ( k=i+1; k<m; k++ )
			{
				A[j+lda*k] = A[j+lda*k] + v*A[i+lda*k];
			}
		}
	}

	return 0;
}
#undef __FUNCT__  
/*--------------------------------------------------------------------------*/
#define __FUNCT__ "bcgsl_solve_i" 
int bcgsl_solve_i(int m, PetscScalar *A, int lda, PetscScalar *b,
	PetscScalar *x)
{
	int i, j;

	for ( i=0; i<m; i++ ) x[i] = b[i];

	for ( i=0; i<m; i++ )
	{
		for( j=i+1; j<m; j++ )
		{
			x[j] = x[j] + A[j+lda*i]*x[i];
		}
	}

	for ( i=m-1; i>=0; i-- )
	{
		x[i] = x[i]/A[i+lda*i];
		for ( j=i-1; j>=0; j-- )
		{
			x[j] = x[j] - A[j+lda*i]*x[i];
		}
	}
	return 0;
}
#undef __FUNCT__  
/*--------------------------------------------------------------------------*/
#define __FUNCT__ "bcgsl_mvmul_i" 
int bcgsl_mvmul_i(int m, PetscScalar *A, int lda, PetscScalar *b,
	PetscScalar *x)
{
	int i, j;

	for ( i=0; i<m; i++ ) x[i] = 0;

	for ( i=0; i<m; i++ )
	{
		for( j=0; j<m; j++ )
		{
			x[j] = x[j] + A[j+lda*i]*b[i];
		}
	}

	return 0;
}
#undef __FUNCT__  
/*--------------------------------------------------------------------------*/
#define __FUNCT__ "bcgsl_dot_i" 
int bcgsl_dot_i(int m, PetscScalar *x, PetscScalar *y,
	PetscScalar *w)
{
	int i, j;
	PetscScalar u;

	*w = 0.0;

	for ( i=0; i<m; i++ )
	{
		*w = *w + (x[i]*y[i]);
	}

	return 0;
}

/****************************************************
 *
 * Some memory allocation functions 
 *
 ****************************************************/

#undef __FUNCT__  
#define __FUNCT__ "bcgsl_cleanup_i" 
int bcgsl_cleanup_i(KSP ksp)
{
	KSP_BiCGStabL *bcgsl = (KSP_BiCGStabL *)ksp->data;

	int       ierr;

	PetscFunctionBegin;

	/* free all workspace */
	ierr = VecDestroy(VB);					CHKERRQ(ierr);
	ierr = VecDestroy(VRT);					CHKERRQ(ierr);
	ierr = VecDestroyVecs(VVR,ELL+1);			CHKERRQ(ierr);
	ierr = VecDestroyVecs(VVU,ELL+1);			CHKERRQ(ierr);
	ierr = VecDestroy(VTM);					CHKERRQ(ierr);

	if ( DELTA>0.0 )
	{
		ierr = VecDestroy(VXR);				CHKERRQ(ierr);
	}

	ierr = PetscFree(AY0c);					CHKERRQ(ierr);
	ierr = PetscFree(AYlc);					CHKERRQ(ierr);
	ierr = PetscFree(AYtc);					CHKERRQ(ierr);
	ierr = PetscFree(MZc);					CHKERRQ(ierr);

	if ( LDZ>0 )
	{
		ierr = PetscFree(MZ);				CHKERRQ(ierr);
		ierr = PetscFree(AY0t);				CHKERRQ(ierr);
		ierr = PetscFree(AYlt);				CHKERRQ(ierr);
	}

	PetscFunctionReturn(0);
}
#undef __FUNCT__  
/*--------------------------------------------------------------------------*/
#define __FUNCT__ "bcgsl_setup_i" 
int bcgsl_setup_i(KSP ksp)
{
	KSP_BiCGStabL *bcgsl = (KSP_BiCGStabL *)ksp->data;
	int ell = ELL;
	int ierr;
	
	PetscFunctionBegin;

	ierr = VecDuplicate(ksp->vec_rhs,&VB);			CHKERRQ(ierr);
	ierr = VecDuplicate(ksp->vec_sol,&VRT);			CHKERRQ(ierr);
	ierr = VecDuplicate(ksp->vec_sol,&VTM);			CHKERRQ(ierr);
	ierr = VecDuplicateVecs(ksp->vec_sol, ell+1, &VVR);	CHKERRQ(ierr);
	ierr = VecDuplicateVecs(ksp->vec_sol, ell+1, &VVU);	CHKERRQ(ierr);

	/* restart vectors */
	if ( DELTA>0.0 )
	{
		ierr = VecDuplicate(VX,&VXR);			CHKERRQ(ierr);
	}

	/* Allocate serial work space. */
	LDZ  = ell-1;
	LDZc = ell+1;

	ierr = PetscMalloc( LDZc*sizeof(PetscScalar),
		&AY0c);						CHKERRQ(ierr);
	ierr = PetscMalloc( LDZc*sizeof(PetscScalar),
		&AYlc);						CHKERRQ(ierr);
	ierr = PetscMalloc( LDZc*sizeof(PetscScalar),
		&AYtc);						CHKERRQ(ierr);
	ierr = PetscMalloc(LDZc*LDZc*sizeof(
			PetscScalar), &MZc);			CHKERRQ(ierr);

	if ( LDZ>0 )
	{
		ierr = PetscMalloc( LDZ*sizeof(PetscScalar),
			&AY0t);					CHKERRQ(ierr);
		ierr = PetscMalloc( LDZ*sizeof(PetscScalar),
			&AYlt);					CHKERRQ(ierr);
		ierr = PetscMalloc( LDZ*LDZ*sizeof(PetscScalar),
			&MZ);					CHKERRQ(ierr);
	}


	PetscFunctionReturn(0);
}
#undef __FUNCT__  

/****************************************************
 *
 * BiCGSTab(L) solver 
 *
 ****************************************************/

#define __FUNCT__ "KSPsolve_BCGS"
/*@C
   KSPSolve_BCGSL - Solve a linear system of equations using
BiCGStab(L).

   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate
-  its - Number of outer iteration steps done

   Options Database Keys:

+  -ksp_type bcgsl

   Level: intermediate

.keywords: KSP, BiCGStab(L), solve

.seealso: @()
@*/
static int  KSPSolve_BCGSL(KSP ksp)
{
	KSP_BiCGStabL *bcgsl = (KSP_BiCGStabL *) ksp->data;

	PetscScalar	alpha, beta, nu, omega, sigma;
	PetscScalar	zero = 0;
	PetscScalar	rho0, rho1;
	PetscScalar	kappa0, kappaA, kappa1;
	PetscReal	ghat, bidule, epsilon, atol;
	PetscReal	zeta, zeta0, mxres, myres, nrm0;
	PetscReal	xx0, xxt;

	PetscTruth	bUpdateX;
	PetscTruth	bComputeR;
	PetscTruth	bBombed = PETSC_FALSE;

	int	maxit, zits, ierr, jerr;
	int	h, i, j, k;
	int	rank;
	
	/* Prime the iterative solver */
	ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

	ierr = KSPInitialResidual(ksp, VX, VTM, VB,
		VVR[0], ksp->vec_rhs);				CHKERRQ(ierr);
	ierr = VecNorm(VVR[0], NORM_2, &zeta0);			CHKERRQ(ierr);
	mxres = zeta0;

	ierr = (*ksp->converged)( ksp, 0, zeta0,
		&ksp->reason, ksp->cnvP);		CHKERRQ(ierr);
	if (ksp->reason)
	{
		return;
	}

	ierr = VecSet( &zero, VVU[0] );				CHKERRQ(ierr);
	alpha = 0;
	rho0 = omega = 1;

	if ( DELTA>0.0 )
	{
		ierr = VecCopy(VX,VXR);				CHKERRQ(ierr);
		ierr = VecSet(&zero,VX);			CHKERRQ(ierr);
		ierr = VecCopy(VVR[0],VB);			CHKERRQ(ierr);
	}
	else
	{
		ierr = VecCopy(ksp->vec_rhs, VB);		CHKERRQ(ierr);
	}

	/* Life goes on */
	ierr = VecCopy(VVR[0], VRT);				CHKERRQ(ierr);
	zeta = zeta0;
	
	ierr = KSPGetTolerances( ksp, &epsilon,
		&atol, PETSC_NULL, &maxit);			CHKERRQ(ierr);

	for ( k=0; k<maxit; k += ELL )
	{
		ierr = PetscObjectTakeAccess(ksp);		CHKERRQ(ierr);
		ksp->its   = k;
		ksp->rnorm = zeta;
		ierr = PetscObjectGrantAccess(ksp);		CHKERRQ(ierr);

		KSPLogResidualHistory(ksp,zeta);
        	KSPMonitor(ksp,ksp->its,zeta); 

		ierr = (*ksp->converged)( ksp, k, zeta,
			&ksp->reason, ksp->cnvP);		CHKERRQ(ierr);
		if (ksp->reason)
		{
			break;
		}

		/* BiCG part */
		rho0 = -omega*rho0;
		nrm0 = zeta;
		for ( j=0; j<ELL; j++ )
		{
			/* rho1 <- r_j' * r_tilde */
			ierr = VecDot(VVR[j],VRT,&rho1);	CHKERRQ(ierr);
			if (rho1 == 0.0)
			{
         			ksp->reason = KSP_DIVERGED_BREAKDOWN_BICG;
				bBombed = PETSC_TRUE;
				break;
			}
			beta = alpha*(rho1/rho0);
			rho0 = rho1;
			nu = -beta;
			for ( i=0; i<=j; i++ )
			{
				/* u_i <- r_i - beta*u_i */
				ierr = VecAYPX( &nu, VVR[i],
					VVU[i]);		CHKERRQ(ierr);
			}
			/* u_{j+1} <- inv(K)*A*u_j */
			ierr = KSP_PCApplyBAorAB(ksp,
				VVU[j], VVU[j+1],
				VTM);				CHKERRQ(ierr); 

			ierr = VecDot(VVU[j+1],VRT,&sigma);	CHKERRQ(ierr);
			if (sigma == 0.0)
			{
         			ksp->reason = KSP_DIVERGED_BREAKDOWN_BICG;
				bBombed = PETSC_TRUE;
				break;
			}
			alpha = rho1/sigma;

			/* x <- x + alpha*u_0 */
			ierr = VecAXPY( &alpha, VVU[0], VX);	CHKERRQ(ierr);

			nu = -alpha;
			for ( i=0; i<=j; i++ )
			{

				/* r_i <- r_i - alpha*u_{i+1} */
				ierr = VecAXPY( &nu, VVU[i+1],
					VVR[i]);		CHKERRQ(ierr);
			}

			/* r_{j+1} <- inv(K)*A*r_j */
			ierr = KSP_PCApplyBAorAB(ksp,
				VVR[j], VVR[j+1],
				VTM);				CHKERRQ(ierr); 


			if ( DELTA>0.0 )
			{
				ierr = VecNorm(VVR[0],NORM_2,
					&nrm0);			CHKERRQ(ierr);
				if ( mxres<nrm0 ) mxres = nrm0;
				if ( myres<nrm0 ) myres = nrm0;
			}
		}

		if (bBombed==PETSC_TRUE)
		{
			break;
		} 

		ierr = (*ksp->converged)( ksp, k, nrm0,
			&ksp->reason, ksp->cnvP);		CHKERRQ(ierr);
		if (ksp->reason)
		{
			break;
		}

		/* Polynomial part */

		for ( i=0; i<=ELL; i++ )
		{
			for ( j=0; j<i; j++ )
			{
				ierr = VecDot(VVR[j],VVR[i],&nu);CHKERRQ(ierr);
				MZc[i+LDZc*j] = nu;
				MZc[j+LDZc*i] = nu;
			}

			ierr = VecDot(VVR[i],VVR[i], &nu);	CHKERRQ(ierr);
			MZc[i+LDZc*i] = nu;
		}

		if ( ELL==1 )
		/* KSP_BCGSL_SetEll has been set top prevent this case
		 * because BiCGstab is typically better, but the routine
		 * stands alone anyways.
		 */
		{
			nu = MZc[1+LDZc];
			if ( nu==0.0 )
			{ 
         			ksp->reason = KSP_DIVERGED_BREAKDOWN_BICG;
				break;
			}
			AY0c[0] = -1;
			AY0c[1] = MZc[0]/nu;
		}
		else
		{
			for ( i=0; i<LDZ; i++ )
			{
				for ( j=0; j<LDZ; j++ )
				{
					MZ[i+LDZ*j] = MZc[(i+1)+LDZc*(j+1)];
				}
				AY0t[i] = MZc[i+1];
				AYlt[i] = MZc[i+1+LDZc*ELL];
			}

			ierr = bcgsl_factr_i(LDZ,MZ,LDZ);
			if ( ierr!=0 )
			{
         			ksp->reason = KSP_DIVERGED_BREAKDOWN;
				bBombed = PETSC_TRUE;
				break;
			}
			ierr = bcgsl_solve_i(LDZ,MZ,LDZ,AY0t,
				&AY0c[1]);			CHKERRQ(ierr);
			AY0c[0] = -1;	AY0c[ELL] = 0;

			ierr = bcgsl_solve_i(LDZ,MZ,LDZ,AYlt,
				&AYlc[1]);			CHKERRQ(ierr);
			AYlc[0] = 0;	AYlc[ELL] = -1;

			ierr = bcgsl_mvmul_i(LDZc,MZc,LDZc,
				AY0c,AYtc);			CHKERRQ(ierr);
			ierr = bcgsl_dot_i(LDZc,AY0c,AYtc,
				&kappa0);			CHKERRQ(ierr);

			/* round-off can cause negative kappa's */
			if ( kappa0<0 ) kappa0 = -kappa0;
			kappa0 = PetscSqrtScalar(kappa0); 

			ierr = bcgsl_dot_i(LDZc,AYlc,AYtc,
				&kappaA);			CHKERRQ(ierr);

			ierr = bcgsl_mvmul_i(LDZc,MZc,LDZc,
				AYlc, AYtc);			CHKERRQ(ierr);
			ierr = bcgsl_dot_i(LDZc,AYlc,AYtc,
				&kappa1);			CHKERRQ(ierr);

			if ( kappa1<0 ) kappa1 = -kappa1;
			kappa1 = PetscSqrtScalar(kappa1); 

			if ( kappa0!=0.0 && kappa1!=0.0)
			{
				if ( kappaA<0.7*kappa0*kappa1 )
				{
					ghat = ( kappaA<0.0 ) ?
						-0.7*kappa0/kappa1 :
						0.7*kappa0/kappa1  ; 
				}
				else
				{
					ghat = kappaA/(kappa1*kappa1);
				}

				for ( i=0; i<=ELL; i++ )
				{
					AY0c[i] = AY0c[i] - ghat* AYlc[i];
				}
			}
		}

		h = ELL; omega = 0.0;
		for ( h=ELL; h>0 && omega==0.0; h-- )
		{
			omega = AY0c[h];
		}

		if ( h==0 )
		{
         		ksp->reason = KSP_DIVERGED_BREAKDOWN;
			break;
		}
			
		for ( i=1; i<=ELL; i++ )
		{
			PetscReal xxx, xxu, xxr;
			nu = -AY0c[i];
			ierr = VecAXPY(&nu,VVU[i],VVU[0]);	CHKERRQ(ierr);
			nu = AY0c[i];
			ierr = VecAXPY(&nu,VVR[i-1],VX);	CHKERRQ(ierr);
			nu = -AY0c[i];
			ierr = VecAXPY(&nu,VVR[i],VVR[0]);	CHKERRQ(ierr);
		}

		ierr = VecNorm(VVR[0],NORM_2,&zeta);		CHKERRQ(ierr);

		/* Accurate Update */
		if ( DELTA>0.0 )
		{
			if ( mxres<zeta ) mxres = zeta;
			if ( myres<zeta ) myres = zeta;

			bUpdateX = (zeta<DELTA*zeta0 && zeta0<=mxres);
			if ( (zeta<DELTA*myres && zeta0<=myres) ||
				bUpdateX )
			{
				/* r0 <- b-inv(K)*A*X */
				ierr = KSP_PCApplyBAorAB( ksp,
					VX, VVR[0], VTM);	CHKERRQ(ierr); 
				nu = -1;
				ierr = VecAYPX(&nu,VB, VVR[0]);
								CHKERRQ(ierr);
				myres = zeta;

				if ( bUpdateX )
				{
					nu = 1;
					ierr = VecAXPY(&nu,VX, VXR);
								CHKERRQ(ierr);
					ierr = VecSet(&zero,VX);
							CHKERRQ(ierr); 
					ierr = VecCopy(VVR[0], VB);
							CHKERRQ(ierr); 
					mxres = zeta;
				}
			}
		}
	}

        KSPMonitor(ksp,ksp->its,zeta); 

	if ( DELTA>0.0 )
	{
		nu = 1;
		ierr = VecAXPY(&nu,VXR,VX);			CHKERRQ(ierr);
	}

	ierr = (*ksp->converged)( ksp, k, zeta,
		&ksp->reason, ksp->cnvP);			CHKERRQ(ierr);
	if ( !ksp->reason ) {
		ksp->reason = KSP_DIVERGED_ITS;
	}

	PetscFunctionReturn(0);

}
#undef __FUNCT__  

/****************************************************
 *
 * Interface functions to the BiCGSTab(L) solver 
 *
 ****************************************************/

EXTERN_C_BEGIN
#define __FUNCT__ "KSP_BCGSL_SetXRes" 
/*@C
   KSP_BCGSL_SetXRes - Sets the parameter governing when
exact residuals will be used instead of computed residuals. 

   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate
-  delta - computed residuals are used alone when delta is not positive

   Options Database Keys:

+  -ksp_bcgsl_xres delta 

   Level: intermediate

.keywords: KSP, BiCGStab(L), set, exact residuals

.seealso: @()
@*/
int KSP_BCGSL_SetXRes(KSP ksp, PetscReal delta)
{
	KSP_BiCGStabL *bcgsl = (KSP_BiCGStabL *)ksp->data;
	int	ierr;

	PetscFunctionBegin;

	if ( ksp->setupcalled )
	{
		if (( delta<=0 && DELTA>0 ) ||
		   ( delta>0 && DELTA<=0 )) 
		{
			ierr = bcgsl_cleanup_i(ksp);		CHKERRQ(ierr);
			ksp->setupcalled = 0;
		}
	}

	DELTA = delta; 

	PetscFunctionReturn(0);
}
#undef __FUNCT__  
EXTERN_C_END
/*--------------------------------------------------------------------------*/
EXTERN_C_BEGIN
#define __FUNCT__ "KSP_BCGSL_SetEll" 
/*@C
   KSP_BCGSL_SetEll - Sets the number of search directions in BiCGStab(L).

   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate
-  ell - number of search directions

   Options Database Keys:

+  -ksp_bcgsl_ell ell 

   Level: intermediate

.keywords: KSP, BiCGStab(L), set, exact residuals, 

.seealso: @()
@*/
int KSP_BCGSL_SetEll(KSP ksp, int ell)
{
	KSP_BiCGStabL *bcgsl = (KSP_BiCGStabL *)ksp->data;
	int       ierr;

	PetscFunctionBegin;

	if (ell <= 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,
		"KSP_BCGSL_SetEll: second argument must exceed 1");

	if (!ksp->setupcalled)
	{
		ELL = ell;
	}
	else if (ELL != ell)
	{
		/* free the data structures,
		   then create them again
		 */
		ierr = bcgsl_cleanup_i(ksp);			CHKERRQ(ierr);
		ELL = ell;
		ksp->setupcalled = 0;
	}

	PetscFunctionReturn(0);
}
#undef __FUNCT__  
EXTERN_C_END
/*--------------------------------------------------------------------------*/
#define __FUNCT__ "KSPView_BCGSL" 
int KSPView_BCGSL(KSP ksp,PetscViewer viewer)
{
	KSP_BiCGStabL  *bcgsl = (KSP_BiCGStabL *)ksp->data; 
	int        ierr;
	PetscTruth isascii,isstring;

	PetscFunctionBegin;

	ierr = PetscTypeCompare( (PetscObject)viewer,
		PETSC_VIEWER_ASCII, &isascii);			CHKERRQ(ierr);

	ierr = PetscTypeCompare( (PetscObject)viewer,
		PETSC_VIEWER_STRING, &isstring);		CHKERRQ(ierr);

	if (isascii)
	{
		ierr = PetscViewerASCIIPrintf( viewer,
			"  BCGSL: Ell = %d\n", ELL);		CHKERRQ(ierr);
		ierr = PetscViewerASCIIPrintf( viewer,
			"  BCGSL: Delta = %lg\n", DELTA);	CHKERRQ(ierr);
	}
	else if (isstring)
	{
		ierr = PetscViewerStringSPrintf( viewer,
			" BCGSL: Ell = %d", ELL);		CHKERRQ(ierr);
		ierr = PetscViewerStringSPrintf( viewer,
			" BCGSL: Delta = %lg", DELTA);		CHKERRQ(ierr);
	}
	else
	{
		SETERRQ1(1,
			"Viewer type %s not supported for KSP BCGSL",
			((PetscObject)viewer)->type_name);
	}

	PetscFunctionReturn(0);
}
#undef __FUNCT__  
/*--------------------------------------------------------------------------*/
#define __FUNCT__ "KSPSetFromOptions_BCGSL"
/*@C
   KSP_BCGSL_SetXRes - Sets the parameter governing when
exact residuals will be used instead of computed residuals. 

   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate
-  delta - only computed residuals are used when delta is not positive

   Options Database Keys:

+  -ksp_bcgsl_xres delta 

   Level: intermediate

.keywords: KSP, BiCGStab(L), set, exact residuals, 

.seealso: @()
@*/
int KSPSetFromOptions_BCGSL(KSP ksp)
{
	KSP_BiCGStabL	*bcgsl = (KSP_BiCGStabL *)ksp->data; 
	int	ierr;
	int	this_ell;
	PetscReal	delta;
	PetscTruth	flga, flg;

	PetscFunctionBegin;

	/* PetscOptionsBegin/End are called in KSPSetFromOptions. They
	don't need to be called here.
	*/
	ierr = PetscOptionsHead("KSP BiCGStab(L) Options");	CHKERRQ(ierr);

	/* Set number of search directions */
	ierr = PetscOptionsInt("-ksp_bcgsl_ell",
		"Number of Krylov search directions",
		"KSP_BCGSL_SetEll", ELL,
		&this_ell, &flg);				CHKERRQ(ierr);
	if (flg)
	{
		ierr = KSP_BCGSL_SetEll(ksp,this_ell);		CHKERRQ(ierr);
	}

	/* Will computed residual be refreshed? */
	ierr = PetscOptionsReal("-ksp_bcgsl_xres", 
		"Threshold used to decide when to refresh computed residuals",
		"KSP_BCGSL_SetXRes", DELTA, &delta, &flg);	CHKERRQ(ierr);
	if (flg)
	{
		ierr = KSP_BCGSL_SetXRes(ksp,delta);	CHKERRQ(ierr);
	}


	ierr = PetscOptionsTail();				CHKERRQ(ierr);

	PetscFunctionReturn(0);
}
#undef __FUNCT__  
/*--------------------------------------------------------------------------*/
#define __FUNCT__ "KSPSetUp_BCGSL"
static int KSPSetUp_BCGSL(KSP ksp)
{
	int ierr;

	PetscFunctionBegin;

	/* Support left preconditioners only */
	if (ksp->pc_side == PC_SYMMETRIC)
	{
		SETERRQ(PETSC_ERR_SUP,
			"no symmetric preconditioning for KSPBCGSL");
	}
	else if (ksp->pc_side == PC_RIGHT)
	{
		SETERRQ(PETSC_ERR_SUP,
			"no right preconditioning for KSPBCGSL");
	}

	ierr = bcgsl_setup_i(ksp);			CHKERRQ(ierr);

	PetscFunctionReturn(0);
}
#undef __FUNCT__  
/*--------------------------------------------------------------------------*/
EXTERN_C_BEGIN
#define __FUNCT__ "KSPCreate_BCGSL"
/*@C
   KSP_BCGSL_SetXRes - Sets the parameter governing when
exact residuals will be used instead of computed residuals. 

   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate
-  delta - only computed residuals are used when delta is not positive

   Options Database Keys:

+  -ksp_bcgsl_xres delta 

   Level: intermediate

.keywords: KSP, BiCGStab(L), set, exact residuals, 

.seealso: @()
@*/
int KSPCreate_BCGSL(KSP ksp)
{

	int ierr;
	KSP_BiCGStabL *bcgsl;

	PetscFunctionBegin;

	/* allocate BiCGStab(L) context */
	ierr = PetscNew(KSP_BiCGStabL,&bcgsl);			CHKERRQ(ierr);
	ierr = PetscMemzero(bcgsl,sizeof(KSP_BiCGStabL));	CHKERRQ(ierr);
	ksp->data = (void*)bcgsl;

	ksp->pc_side              = PC_LEFT;
	ksp->ops->setup           = KSPSetUp_BCGSL;
	ksp->ops->solve           = KSPSolve_BCGSL;
	ksp->ops->destroy         = KSPDefaultDestroy;
	ksp->ops->buildsolution   = KSPDefaultBuildSolution;
	ksp->ops->buildresidual   = KSPDefaultBuildResidual;
	ksp->ops->setfromoptions  = KSPSetFromOptions_BCGSL;
	ksp->ops->view            = KSPView_BCGSL;

	/* Let the user redefine the number of directions vectors */
	ELL = 2;
	ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,
		"KSPBCGSLSetEll_C", "KSP_BCGS_SetEll",
		KSP_BCGSL_SetEll);				CHKERRQ(ierr);

	/* Set the threshold for when exact residuals will be used */
	DELTA = 0.01;
	ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,
		"KSP_BCGSL_SetXRes_C", "KSP_BCGS_SetXRes",
		KSP_BCGSL_SetXRes);				CHKERRQ(ierr);

	PetscFunctionReturn(0);
}
#undef __FUNCT__  
EXTERN_C_END
/*--------------------------------------------------------------------------*/
