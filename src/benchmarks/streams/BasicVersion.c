
#include <sys/time.h>
/* int gettimeofday(struct timeval *tp, struct timezone *tzp); */

double second()
{
/* struct timeval { long	tv_sec;	
	    long	tv_usec;	};

struct timezone { int	tz_minuteswest;
	     int	tz_dsttime;	 };	*/

	struct timeval tp;
	struct timezone tzp;
	int i;

	i = gettimeofday(&tp,&tzp);
	return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}
# include <stdio.h>
# include <math.h>
# include <float.h>
# include <limits.h>
# include <sys/time.h>

/*
* Program: Stream
* Programmer: Joe R. Zagar
* Revision: 4.0-BETA, October 24, 1995
* Original code developed by John D. McCalpin
*
* This program measures memory transfer rates in MB/s for simple 
* computational kernels coded in C.  These numbers reveal the quality
* of code generation for simple uncacheable kernels as well as showing
* the cost of floating-point operations relative to memory accesses.
*
* INSTRUCTIONS:
*
*	1) Stream requires a good bit of memory to run.  Adjust the
*          value of 'N' (below) to give a 'timing calibration' of 
*          at least 20 clock-ticks.  This will provide rate estimates
*          that should be good to about 5% precision.
*/

# define N	2000000
# define NTIMES	50
# define OFFSET	0

/*
*	3) Compile the code with full optimization.  Many compilers
*	   generate unreasonably bad code before the optimizer tightens
*	   things up.  If the results are unreasonably good, on the
*	   other hand, the optimizer might be too smart for me!
*
*         Try compiling with:
*               cc -O stream_d.c second.c -o stream_d -lm
*
*         This is known to work on Cray, SGI, IBM, and Sun machines.
*
*
*	4) Mail the results to mccalpin@cs.virginia.edu
*	   Be sure to include:
*		a) computer hardware model number and software revision
*		b) the compiler flags
*		c) all of the output from the test case.
* Thanks!
*
*/

# define HLINE "-------------------------------------------------------------\n"

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

static double	a[N+OFFSET],
		b[N+OFFSET],
		c[N+OFFSET];
/*double *a,*b,*c;*/

static double	rmstime[4] = {0}, maxtime[4] = {0},
		mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};

static char	*label[4] = {"Copy:      ", "Scale:     ",
   "Add:       ", "Triad:     "};

static double	bytes[4] = {
   2 * sizeof(double) * N,
   2 * sizeof(double) * N,
   3 * sizeof(double) * N,
   3 * sizeof(double) * N
   };

extern double second();

int
main()
   {
   int			quantum, checktick();
   int			BytesPerWord;
   register int	j, k;
   double		scalar, t, times[4][NTIMES];

   /* --- SETUP --- determine precision and check timing --- */

   printf(HLINE);
   BytesPerWord = sizeof(double);
   printf("This system uses %d bytes per DOUBLE PRECISION word.\n",
	BytesPerWord);

   printf(HLINE);
   printf("Array size = %d, Offset = %d\n" , N, OFFSET);
   printf("Total memory required = %.1f MB.\n",
	(3 * N * BytesPerWord) / 1048576.0);
   printf("Each test is run %d times, but only\n", NTIMES);
   printf("the *best* time for each is used.\n");

   /* Get initial value for system clock. */

   /*  a = malloc(N*sizeof(double));
   b = malloc(N*sizeof(double));
   c = malloc(N*sizeof(double));*/
   for (j=0; j<N; j++) {
	a[j] = 1.0;
	b[j] = 2.0;
	c[j] = 0.0;
	}

   printf(HLINE);

   if  ( (quantum = checktick()) >= 1) 
	printf("Your clock granularity/precision appears to be "
	    "%d microseconds.\n", quantum);
   else
	printf("Your clock granularity appears to be "
	    "less than one microsecond.\n");

   t = second();
   for (j = 0; j < N; j++)
	a[j] = 2.0E0 * a[j];
   t = 1.0E6 * (second() - t);

   printf("Each test below will take on the order"
	" of %d microseconds.\n", (int) t  );
   printf("   (= %d clock ticks)\n", (int) (t/quantum) );
   printf("Increase the size of the arrays if this shows that\n");
   printf("you are not getting at least 20 clock ticks per test.\n");

   printf(HLINE);

   printf("WARNING -- The above is only a rough guideline.\n");
   printf("For best results, please be sure you know the\n");
   printf("precision of your system timer.\n");
   printf(HLINE);

   /*	--- MAIN LOOP --- repeat test cases NTIMES times --- */

   scalar = 3.0;
   for (k=0; k<NTIMES; k++)
	{
	times[0][k] = second();
	for (j=0; j<N; j++)
	    c[j] = a[j];
	times[0][k] = second() - times[0][k];
	
	times[1][k] = second();
	for (j=0; j<N; j++)
	    b[j] = scalar*c[j];
	times[1][k] = second() - times[1][k];
	
	times[2][k] = second();
	for (j=0; j<N; j++)
	    c[j] = a[j]+b[j];
	times[2][k] = second() - times[2][k];
	
	times[3][k] = second();
	for (j=0; j<N; j++)
	    a[j] = b[j]+scalar*c[j];
	times[3][k] = second() - times[3][k];
	}

   /*	--- SUMMARY --- */

       for (k=0; k<NTIMES; k++)
	{
	for (j=0; j<4; j++)
	    {
	    rmstime[j] = rmstime[j] + (times[j][k] * times[j][k]);
	    mintime[j] = MIN(mintime[j], times[j][k]);
	    maxtime[j] = MAX(maxtime[j], times[j][k]);
	    }
	}

   printf("Function      Rate (MB/s)   RMS time     Min time     Max time\n");
   for (j=0; j<4; j++) {
	rmstime[j] = sqrt(rmstime[j]/(double)NTIMES);

	printf("%s%11.4f  %11.4f  %11.4f  %11.4f\n", label[j],
	       1.0E-06 * bytes[j]/mintime[j],
	       rmstime[j],
	       mintime[j],
	       maxtime[j]);
   }
   return 0;
}

# define	M	20

int
checktick()
   {
   int		i, minDelta, Delta;
   double	t1, t2, timesfound[M];

/*  Collect a sequence of M unique time values from the system. */

   for (i = 0; i < M; i++) {
	t1 = second();
	while( ((t2=second()) - t1) < 1.0E-6 )
	    ;
	timesfound[i] = t1 = t2;
	}

/*
* Determine the minimum difference between these M values.
* This result will be our estimate (in microseconds) for the
* clock granularity.
*/

   minDelta = 1000000;
   for (i = 1; i < M; i++) {
	Delta = (int)( 1.0E6 * (timesfound[i]-timesfound[i-1]));
	minDelta = MIN(minDelta, MAX(Delta,0));
	}

   return(minDelta);
   }

