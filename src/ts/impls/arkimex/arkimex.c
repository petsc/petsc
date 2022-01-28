/*
  Code for timestepping with additive Runge-Kutta IMEX method

  Notes:
  The general system is written as

  F(t,U,Udot) = G(t,U)

  where F represents the stiff part of the physics and G represents the non-stiff part.

*/
#include <petsc/private/tsimpl.h>                /*I   "petscts.h"   I*/
#include <petscdm.h>

static TSARKIMEXType  TSARKIMEXDefault = TSARKIMEX3;
static PetscBool      TSARKIMEXRegisterAllCalled;
static PetscBool      TSARKIMEXPackageInitialized;
static PetscErrorCode TSExtrapolate_ARKIMEX(TS,PetscReal,Vec);

typedef struct _ARKTableau *ARKTableau;
struct _ARKTableau {
  char      *name;
  PetscInt  order;                /* Classical approximation order of the method */
  PetscInt  s;                    /* Number of stages */
  PetscBool stiffly_accurate;     /* The implicit part is stiffly accurate*/
  PetscBool FSAL_implicit;        /* The implicit part is FSAL*/
  PetscBool explicit_first_stage; /* The implicit part has an explicit first stage*/
  PetscInt  pinterp;              /* Interpolation order */
  PetscReal *At,*bt,*ct;          /* Stiff tableau */
  PetscReal *A,*b,*c;             /* Non-stiff tableau */
  PetscReal *bembedt,*bembed;     /* Embedded formula of order one less (order-1) */
  PetscReal *binterpt,*binterp;   /* Dense output formula */
  PetscReal ccfl;                 /* Placeholder for CFL coefficient relative to forward Euler */
};
typedef struct _ARKTableauLink *ARKTableauLink;
struct _ARKTableauLink {
  struct _ARKTableau tab;
  ARKTableauLink     next;
};
static ARKTableauLink ARKTableauList;

typedef struct {
  ARKTableau   tableau;
  Vec          *Y;               /* States computed during the step */
  Vec          *YdotI;           /* Time derivatives for the stiff part */
  Vec          *YdotRHS;         /* Function evaluations for the non-stiff part */
  Vec          *Y_prev;          /* States computed during the previous time step */
  Vec          *YdotI_prev;      /* Time derivatives for the stiff part for the previous time step*/
  Vec          *YdotRHS_prev;    /* Function evaluations for the non-stiff part for the previous time step*/
  Vec          Ydot0;            /* Holds the slope from the previous step in FSAL case */
  Vec          Ydot;             /* Work vector holding Ydot during residual evaluation */
  Vec          Z;                /* Ydot = shift(Y-Z) */
  PetscScalar  *work;            /* Scalar work */
  PetscReal    scoeff;           /* shift = scoeff/dt */
  PetscReal    stage_time;
  PetscBool    imex;
  PetscBool    extrapolate;      /* Extrapolate initial guess from previous time-step stage values */
  TSStepStatus status;
} TS_ARKIMEX;
/*MC
     TSARKIMEXARS122 - Second order ARK IMEX scheme.

     This method has one explicit stage and one implicit stage.

     Options Database:
.      -ts_arkimex_type ars122

     References:
.   1. -  U. Ascher, S. Ruuth, R. J. Spiteri, Implicit explicit Runge Kutta methods for time dependent Partial Differential Equations. Appl. Numer. Math. 25, (1997).

     Level: advanced

.seealso: TSARKIMEX, TSARKIMEXType, TSARKIMEXSetType()
M*/
/*MC
     TSARKIMEXA2 - Second order ARK IMEX scheme with A-stable implicit part.

     This method has an explicit stage and one implicit stage, and has an A-stable implicit scheme. This method was provided by Emil Constantinescu.

     Options Database:
.      -ts_arkimex_type a2

     Level: advanced

.seealso: TSARKIMEX, TSARKIMEXType, TSARKIMEXSetType()
M*/
/*MC
     TSARKIMEXL2 - Second order ARK IMEX scheme with L-stable implicit part.

     This method has two implicit stages, and L-stable implicit scheme.

     Options Database:
.      -ts_arkimex_type l2

    References:
.   1. -  L. Pareschi, G. Russo, Implicit Explicit Runge Kutta schemes and applications to hyperbolic systems with relaxations. Journal of Scientific Computing Volume: 25, Issue: 1, October, 2005.

     Level: advanced

.seealso: TSARKIMEX, TSARKIMEXType, TSARKIMEXSetType()
M*/
/*MC
     TSARKIMEX1BEE - First order backward Euler represented as an ARK IMEX scheme with extrapolation as error estimator. This is a 3-stage method.

     This method is aimed at starting the integration of implicit DAEs when explicit first-stage ARK methods are used.

     Options Database:
.      -ts_arkimex_type 1bee

     Level: advanced

.seealso: TSARKIMEX, TSARKIMEXType, TSARKIMEXSetType()
M*/
/*MC
     TSARKIMEX2C - Second order ARK IMEX scheme with L-stable implicit part.

     This method has one explicit stage and two implicit stages. The implicit part is the same as in TSARKIMEX2D and TSARKIMEX2E, but the explicit part has a larger stability region on the negative real axis. This method was provided by Emil Constantinescu.

     Options Database:
.      -ts_arkimex_type 2c

     Level: advanced

.seealso: TSARKIMEX, TSARKIMEXType, TSARKIMEXSetType()
M*/
/*MC
     TSARKIMEX2D - Second order ARK IMEX scheme with L-stable implicit part.

     This method has one explicit stage and two implicit stages. The stability function is independent of the explicit part in the infinity limit of the implict component. This method was provided by Emil Constantinescu.

     Options Database:
.      -ts_arkimex_type 2d

     Level: advanced

.seealso: TSARKIMEX, TSARKIMEXType, TSARKIMEXSetType()
M*/
/*MC
     TSARKIMEX2E - Second order ARK IMEX scheme with L-stable implicit part.

     This method has one explicit stage and two implicit stages. It is is an optimal method developed by Emil Constantinescu.

     Options Database:
.      -ts_arkimex_type 2e

    Level: advanced

.seealso: TSARKIMEX, TSARKIMEXType, TSARKIMEXSetType()
M*/
/*MC
     TSARKIMEXPRSSP2 - Second order SSP ARK IMEX scheme.

     This method has three implicit stages.

     References:
.   1. -  L. Pareschi, G. Russo, Implicit Explicit Runge Kutta schemes and applications to hyperbolic systems with relaxations. Journal of Scientific Computing Volume: 25, Issue: 1, October, 2005.

     This method is referred to as SSP2-(3,3,2) in https://arxiv.org/abs/1110.4375

     Options Database:
.      -ts_arkimex_type prssp2

     Level: advanced

.seealso: TSARKIMEX, TSARKIMEXType, TSARKIMEXSetType()
M*/
/*MC
     TSARKIMEX3 - Third order ARK IMEX scheme with L-stable implicit part.

     This method has one explicit stage and three implicit stages.

     Options Database:
.      -ts_arkimex_type 3

     References:
.   1. -  Kennedy and Carpenter 2003.

     Level: advanced

.seealso: TSARKIMEX, TSARKIMEXType, TSARKIMEXSetType()
M*/
/*MC
     TSARKIMEXARS443 - Third order ARK IMEX scheme.

     This method has one explicit stage and four implicit stages.

     Options Database:
.      -ts_arkimex_type ars443

     References:
+   1. -  U. Ascher, S. Ruuth, R. J. Spiteri, Implicit explicit Runge Kutta methods for time dependent Partial Differential Equations. Appl. Numer. Math. 25, (1997).
-   2. -  This method is referred to as ARS(4,4,3) in https://arxiv.org/abs/1110.4375

     Level: advanced

.seealso: TSARKIMEX, TSARKIMEXType, TSARKIMEXSetType()
M*/
/*MC
     TSARKIMEXBPR3 - Third order ARK IMEX scheme.

     This method has one explicit stage and four implicit stages.

     Options Database:
.      -ts_arkimex_type bpr3

     References:
 .    This method is referred to as ARK3 in https://arxiv.org/abs/1110.4375

     Level: advanced

.seealso: TSARKIMEX, TSARKIMEXType, TSARKIMEXSetType()
M*/
/*MC
     TSARKIMEX4 - Fourth order ARK IMEX scheme with L-stable implicit part.

     This method has one explicit stage and four implicit stages.

     Options Database:
.      -ts_arkimex_type 4

     References:
.     Kennedy and Carpenter 2003.

     Level: advanced

.seealso: TSARKIMEX, TSARKIMEXType, TSARKIMEXSetType()
M*/
/*MC
     TSARKIMEX5 - Fifth order ARK IMEX scheme with L-stable implicit part.

     This method has one explicit stage and five implicit stages.

     Options Database:
.      -ts_arkimex_type 5

     References:
.     Kennedy and Carpenter 2003.

     Level: advanced

.seealso: TSARKIMEX, TSARKIMEXType, TSARKIMEXSetType()
M*/

/*@C
  TSARKIMEXRegisterAll - Registers all of the additive Runge-Kutta implicit-explicit methods in TSARKIMEX

  Not Collective, but should be called by all processes which will need the schemes to be registered

  Level: advanced

.seealso:  TSARKIMEXRegisterDestroy()
@*/
PetscErrorCode TSARKIMEXRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSARKIMEXRegisterAllCalled) PetscFunctionReturn(0);
  TSARKIMEXRegisterAllCalled = PETSC_TRUE;

  {
    const PetscReal
      A[3][3] = {{0.0,0.0,0.0},
                 {0.0,0.0,0.0},
                 {0.0,0.5,0.0}},
      At[3][3] = {{1.0,0.0,0.0},
                  {0.0,0.5,0.0},
                  {0.0,0.5,0.5}},
      b[3]       = {0.0,0.5,0.5},
      bembedt[3] = {1.0,0.0,0.0};
    ierr = TSARKIMEXRegister(TSARKIMEX1BEE,2,3,&At[0][0],b,NULL,&A[0][0],b,NULL,bembedt,bembedt,1,b,NULL);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[2][2] = {{0.0,0.0},
                 {0.5,0.0}},
      At[2][2] = {{0.0,0.0},
                  {0.0,0.5}},
      b[2]       = {0.0,1.0},
      bembedt[2] = {0.5,0.5};
    /* binterpt[2][2] = {{1.0,-1.0},{0.0,1.0}};  second order dense output has poor stability properties and hence it is not currently in use*/
    ierr = TSARKIMEXRegister(TSARKIMEXARS122,2,2,&At[0][0],b,NULL,&A[0][0],b,NULL,bembedt,bembedt,1,b,NULL);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[2][2] = {{0.0,0.0},
                 {1.0,0.0}},
      At[2][2] = {{0.0,0.0},
                  {0.5,0.5}},
      b[2]       = {0.5,0.5},
      bembedt[2] = {0.0,1.0};
    /* binterpt[2][2] = {{1.0,-0.5},{0.0,0.5}}  second order dense output has poor stability properties and hence it is not currently in use*/
    ierr = TSARKIMEXRegister(TSARKIMEXA2,2,2,&At[0][0],b,NULL,&A[0][0],b,NULL,bembedt,bembedt,1,b,NULL);CHKERRQ(ierr);
  }
  {
    /* const PetscReal us2 = 1.0-1.0/PetscSqrtReal((PetscReal)2.0);    Direct evaluation: 0.2928932188134524755992. Used below to ensure all values are available at compile time   */
    const PetscReal
      A[2][2] = {{0.0,0.0},
                 {1.0,0.0}},
      At[2][2] = {{0.2928932188134524755992,0.0},
                  {1.0-2.0*0.2928932188134524755992,0.2928932188134524755992}},
      b[2]       = {0.5,0.5},
      bembedt[2] = {0.0,1.0},
      binterpt[2][2] = {{  (0.2928932188134524755992-1.0)/(2.0*0.2928932188134524755992-1.0),-1/(2.0*(1.0-2.0*0.2928932188134524755992))},
                        {1-(0.2928932188134524755992-1.0)/(2.0*0.2928932188134524755992-1.0),-1/(2.0*(1.0-2.0*0.2928932188134524755992))}},
      binterp[2][2] = {{1.0,-0.5},{0.0,0.5}};
    ierr = TSARKIMEXRegister(TSARKIMEXL2,2,2,&At[0][0],b,NULL,&A[0][0],b,NULL,bembedt,bembedt,2,binterpt[0],binterp[0]);CHKERRQ(ierr);
  }
  {
    /* const PetscReal s2 = PetscSqrtReal((PetscReal)2.0),  Direct evaluation: 1.414213562373095048802. Used below to ensure all values are available at compile time   */
    const PetscReal
      A[3][3] = {{0,0,0},
                 {2-1.414213562373095048802,0,0},
                 {0.5,0.5,0}},
      At[3][3] = {{0,0,0},
                  {1-1/1.414213562373095048802,1-1/1.414213562373095048802,0},
                  {1/(2*1.414213562373095048802),1/(2*1.414213562373095048802),1-1/1.414213562373095048802}},
      bembedt[3] = {(4.-1.414213562373095048802)/8.,(4.-1.414213562373095048802)/8.,1/(2.*1.414213562373095048802)},
      binterpt[3][2] = {{1.0/1.414213562373095048802,-1.0/(2.0*1.414213562373095048802)},
                        {1.0/1.414213562373095048802,-1.0/(2.0*1.414213562373095048802)},
                        {1.0-1.414213562373095048802,1.0/1.414213562373095048802}};
    ierr = TSARKIMEXRegister(TSARKIMEX2C,2,3,&At[0][0],NULL,NULL,&A[0][0],NULL,NULL,bembedt,bembedt,2,binterpt[0],NULL);CHKERRQ(ierr);
  }
  {
    /* const PetscReal s2 = PetscSqrtReal((PetscReal)2.0),  Direct evaluation: 1.414213562373095048802. Used below to ensure all values are available at compile time   */
    const PetscReal
      A[3][3] = {{0,0,0},
                 {2-1.414213562373095048802,0,0},
                 {0.75,0.25,0}},
      At[3][3] = {{0,0,0},
                  {1-1/1.414213562373095048802,1-1/1.414213562373095048802,0},
                  {1/(2*1.414213562373095048802),1/(2*1.414213562373095048802),1-1/1.414213562373095048802}},
      bembedt[3] = {(4.-1.414213562373095048802)/8.,(4.-1.414213562373095048802)/8.,1/(2.*1.414213562373095048802)},
      binterpt[3][2] =  {{1.0/1.414213562373095048802,-1.0/(2.0*1.414213562373095048802)},
                         {1.0/1.414213562373095048802,-1.0/(2.0*1.414213562373095048802)},
                         {1.0-1.414213562373095048802,1.0/1.414213562373095048802}};
    ierr = TSARKIMEXRegister(TSARKIMEX2D,2,3,&At[0][0],NULL,NULL,&A[0][0],NULL,NULL,bembedt,bembedt,2,binterpt[0],NULL);CHKERRQ(ierr);
  }
  {                             /* Optimal for linear implicit part */
    /* const PetscReal s2 = PetscSqrtReal((PetscReal)2.0),  Direct evaluation: 1.414213562373095048802. Used below to ensure all values are available at compile time   */
    const PetscReal
      A[3][3] = {{0,0,0},
                 {2-1.414213562373095048802,0,0},
                 {(3-2*1.414213562373095048802)/6,(3+2*1.414213562373095048802)/6,0}},
      At[3][3] = {{0,0,0},
                  {1-1/1.414213562373095048802,1-1/1.414213562373095048802,0},
                  {1/(2*1.414213562373095048802),1/(2*1.414213562373095048802),1-1/1.414213562373095048802}},
      bembedt[3] = {(4.-1.414213562373095048802)/8.,(4.-1.414213562373095048802)/8.,1/(2.*1.414213562373095048802)},
      binterpt[3][2] =  {{1.0/1.414213562373095048802,-1.0/(2.0*1.414213562373095048802)},
                         {1.0/1.414213562373095048802,-1.0/(2.0*1.414213562373095048802)},
                         {1.0-1.414213562373095048802,1.0/1.414213562373095048802}};
    ierr = TSARKIMEXRegister(TSARKIMEX2E,2,3,&At[0][0],NULL,NULL,&A[0][0],NULL,NULL,bembedt,bembedt,2,binterpt[0],NULL);CHKERRQ(ierr);
  }
  {                             /* Optimal for linear implicit part */
    const PetscReal
      A[3][3] = {{0,0,0},
                 {0.5,0,0},
                 {0.5,0.5,0}},
      At[3][3] = {{0.25,0,0},
                  {0,0.25,0},
                  {1./3,1./3,1./3}};
    ierr = TSARKIMEXRegister(TSARKIMEXPRSSP2,2,3,&At[0][0],NULL,NULL,&A[0][0],NULL,NULL,NULL,NULL,0,NULL,NULL);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[4][4] = {{0,0,0,0},
                 {1767732205903./2027836641118.,0,0,0},
                 {5535828885825./10492691773637.,788022342437./10882634858940.,0,0},
                 {6485989280629./16251701735622.,-4246266847089./9704473918619.,10755448449292./10357097424841.,0}},
      At[4][4] = {{0,0,0,0},
                  {1767732205903./4055673282236.,1767732205903./4055673282236.,0,0},
                  {2746238789719./10658868560708.,-640167445237./6845629431997.,1767732205903./4055673282236.,0},
                  {1471266399579./7840856788654.,-4482444167858./7529755066697.,11266239266428./11593286722821.,1767732205903./4055673282236.}},
      bembedt[4]     = {2756255671327./12835298489170.,-10771552573575./22201958757719.,9247589265047./10645013368117.,2193209047091./5459859503100.},
      binterpt[4][2] = {{4655552711362./22874653954995., -215264564351./13552729205753.},
                        {-18682724506714./9892148508045.,17870216137069./13817060693119.},
                        {34259539580243./13192909600954.,-28141676662227./17317692491321.},
                        {584795268549./6622622206610.,   2508943948391./7218656332882.}};
    ierr = TSARKIMEXRegister(TSARKIMEX3,3,4,&At[0][0],NULL,NULL,&A[0][0],NULL,NULL,bembedt,bembedt,2,binterpt[0],NULL);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[5][5] = {{0,0,0,0,0},
                 {1./2,0,0,0,0},
                 {11./18,1./18,0,0,0},
                 {5./6,-5./6,.5,0,0},
                 {1./4,7./4,3./4,-7./4,0}},
      At[5][5] = {{0,0,0,0,0},
                  {0,1./2,0,0,0},
                  {0,1./6,1./2,0,0},
                  {0,-1./2,1./2,1./2,0},
                  {0,3./2,-3./2,1./2,1./2}},
    *bembedt = NULL;
    ierr = TSARKIMEXRegister(TSARKIMEXARS443,3,5,&At[0][0],NULL,NULL,&A[0][0],NULL,NULL,bembedt,bembedt,0,NULL,NULL);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[5][5] = {{0,0,0,0,0},
                 {1,0,0,0,0},
                 {4./9,2./9,0,0,0},
                 {1./4,0,3./4,0,0},
                 {1./4,0,3./5,0,0}},
      At[5][5] = {{0,0,0,0,0},
                  {.5,.5,0,0,0},
                  {5./18,-1./9,.5,0,0},
                  {.5,0,0,.5,0},
                  {.25,0,.75,-.5,.5}},
    *bembedt = NULL;
    ierr = TSARKIMEXRegister(TSARKIMEXBPR3,3,5,&At[0][0],NULL,NULL,&A[0][0],NULL,NULL,bembedt,bembedt,0,NULL,NULL);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[6][6] = {{0,0,0,0,0,0},
                 {1./2,0,0,0,0,0},
                 {13861./62500.,6889./62500.,0,0,0,0},
                 {-116923316275./2393684061468.,-2731218467317./15368042101831.,9408046702089./11113171139209.,0,0,0},
                 {-451086348788./2902428689909.,-2682348792572./7519795681897.,12662868775082./11960479115383.,3355817975965./11060851509271.,0,0},
                 {647845179188./3216320057751.,73281519250./8382639484533.,552539513391./3454668386233.,3354512671639./8306763924573.,4040./17871.,0}},
      At[6][6] = {{0,0,0,0,0,0},
                  {1./4,1./4,0,0,0,0},
                  {8611./62500.,-1743./31250.,1./4,0,0,0},
                  {5012029./34652500.,-654441./2922500.,174375./388108.,1./4,0,0},
                  {15267082809./155376265600.,-71443401./120774400.,730878875./902184768.,2285395./8070912.,1./4,0},
                  {82889./524892.,0,15625./83664.,69875./102672.,-2260./8211,1./4}},
      bembedt[6]     = {4586570599./29645900160.,0,178811875./945068544.,814220225./1159782912.,-3700637./11593932.,61727./225920.},
      binterpt[6][3] = {{6943876665148./7220017795957.,-54480133./30881146.,6818779379841./7100303317025.},
                        {0,0,0},
                        {7640104374378./9702883013639.,-11436875./14766696.,2173542590792./12501825683035.},
                        {-20649996744609./7521556579894.,174696575./18121608.,-31592104683404./5083833661969.},
                        {8854892464581./2390941311638.,-12120380./966161.,61146701046299./7138195549469.},
                        {-11397109935349./6675773540249.,3843./706.,-17219254887155./4939391667607.}};
    ierr = TSARKIMEXRegister(TSARKIMEX4,4,6,&At[0][0],NULL,NULL,&A[0][0],NULL,NULL,bembedt,bembedt,3,binterpt[0],NULL);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[8][8] = {{0,0,0,0,0,0,0,0},
                 {41./100,0,0,0,0,0,0,0},
                 {367902744464./2072280473677.,677623207551./8224143866563.,0,0,0,0,0,0},
                 {1268023523408./10340822734521.,0,1029933939417./13636558850479.,0,0,0,0,0},
                 {14463281900351./6315353703477.,0,66114435211212./5879490589093.,-54053170152839./4284798021562.,0,0,0,0},
                 {14090043504691./34967701212078.,0,15191511035443./11219624916014.,-18461159152457./12425892160975.,-281667163811./9011619295870.,0,0,0},
                 {19230459214898./13134317526959.,0,21275331358303./2942455364971.,-38145345988419./4862620318723.,-1./8,-1./8,0,0},
                 {-19977161125411./11928030595625.,0,-40795976796054./6384907823539.,177454434618887./12078138498510.,782672205425./8267701900261.,-69563011059811./9646580694205.,7356628210526./4942186776405.,0}},
      At[8][8] = {{0,0,0,0,0,0,0,0},
                  {41./200.,41./200.,0,0,0,0,0,0},
                  {41./400.,-567603406766./11931857230679.,41./200.,0,0,0,0,0},
                  {683785636431./9252920307686.,0,-110385047103./1367015193373.,41./200.,0,0,0,0},
                  {3016520224154./10081342136671.,0,30586259806659./12414158314087.,-22760509404356./11113319521817.,41./200.,0,0,0},
                  {218866479029./1489978393911.,0,638256894668./5436446318841.,-1179710474555./5321154724896.,-60928119172./8023461067671.,41./200.,0,0},
                  {1020004230633./5715676835656.,0,25762820946817./25263940353407.,-2161375909145./9755907335909.,-211217309593./5846859502534.,-4269925059573./7827059040749.,41./200,0},
                  {-872700587467./9133579230613.,0,0,22348218063261./9555858737531.,-1143369518992./8141816002931.,-39379526789629./19018526304540.,32727382324388./42900044865799.,41./200.}},
      bembedt[8]     = {-975461918565./9796059967033.,0,0,78070527104295./32432590147079.,-548382580838./3424219808633.,-33438840321285./15594753105479.,3629800801594./4656183773603.,4035322873751./18575991585200.},
      binterpt[8][3] = {{-17674230611817./10670229744614.,  43486358583215./12773830924787., -9257016797708./5021505065439.},
                        {0,  0, 0                            },
                        {0,  0, 0                            },
                        {65168852399939./7868540260826.,  -91478233927265./11067650958493., 26096422576131./11239449250142.},
                        {15494834004392./5936557850923.,  -79368583304911./10890268929626., 92396832856987./20362823103730.},
                        {-99329723586156./26959484932159.,  -12239297817655./9152339842473., 30029262896817./10175596800299.},
                        {-19024464361622./5461577185407.,  115839755401235./10719374521269., -26136350496073./3983972220547.},
                        {-6511271360970./6095937251113.,  5843115559534./2180450260947., -5289405421727./3760307252460. }};
    ierr = TSARKIMEXRegister(TSARKIMEX5,5,8,&At[0][0],NULL,NULL,&A[0][0],NULL,NULL,bembedt,bembedt,3,binterpt[0],NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   TSARKIMEXRegisterDestroy - Frees the list of schemes that were registered by TSARKIMEXRegister().

   Not Collective

   Level: advanced

.seealso: TSARKIMEXRegister(), TSARKIMEXRegisterAll()
@*/
PetscErrorCode TSARKIMEXRegisterDestroy(void)
{
  PetscErrorCode ierr;
  ARKTableauLink link;

  PetscFunctionBegin;
  while ((link = ARKTableauList)) {
    ARKTableau t = &link->tab;
    ARKTableauList = link->next;
    ierr = PetscFree6(t->At,t->bt,t->ct,t->A,t->b,t->c);CHKERRQ(ierr);
    ierr = PetscFree2(t->bembedt,t->bembed);CHKERRQ(ierr);
    ierr = PetscFree2(t->binterpt,t->binterp);CHKERRQ(ierr);
    ierr = PetscFree(t->name);CHKERRQ(ierr);
    ierr = PetscFree(link);CHKERRQ(ierr);
  }
  TSARKIMEXRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  TSARKIMEXInitializePackage - This function initializes everything in the TSARKIMEX package. It is called
  from TSInitializePackage().

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode TSARKIMEXInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSARKIMEXPackageInitialized) PetscFunctionReturn(0);
  TSARKIMEXPackageInitialized = PETSC_TRUE;
  ierr = TSARKIMEXRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(TSARKIMEXFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSARKIMEXFinalizePackage - This function destroys everything in the TSARKIMEX package. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode TSARKIMEXFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TSARKIMEXPackageInitialized = PETSC_FALSE;
  ierr = TSARKIMEXRegisterDestroy();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSARKIMEXRegister - register an ARK IMEX scheme by providing the entries in the Butcher tableau and optionally embedded approximations and interpolation

   Not Collective, but the same schemes should be registered on all processes on which they will be used

   Input Parameters:
+  name - identifier for method
.  order - approximation order of method
.  s - number of stages, this is the dimension of the matrices below
.  At - Butcher table of stage coefficients for stiff part (dimension s*s, row-major)
.  bt - Butcher table for completing the stiff part of the step (dimension s; NULL to use the last row of At)
.  ct - Abscissa of each stiff stage (dimension s, NULL to use row sums of At)
.  A - Non-stiff stage coefficients (dimension s*s, row-major)
.  b - Non-stiff step completion table (dimension s; NULL to use last row of At)
.  c - Non-stiff abscissa (dimension s; NULL to use row sums of A)
.  bembedt - Stiff part of completion table for embedded method (dimension s; NULL if not available)
.  bembed - Non-stiff part of completion table for embedded method (dimension s; NULL to use bembedt if provided)
.  pinterp - Order of the interpolation scheme, equal to the number of columns of binterpt and binterp
.  binterpt - Coefficients of the interpolation formula for the stiff part (dimension s*pinterp)
-  binterp - Coefficients of the interpolation formula for the non-stiff part (dimension s*pinterp; NULL to reuse binterpt)

   Notes:
   Several ARK IMEX methods are provided, this function is only needed to create new methods.

   Level: advanced

.seealso: TSARKIMEX
@*/
PetscErrorCode TSARKIMEXRegister(TSARKIMEXType name,PetscInt order,PetscInt s,
                                 const PetscReal At[],const PetscReal bt[],const PetscReal ct[],
                                 const PetscReal A[],const PetscReal b[],const PetscReal c[],
                                 const PetscReal bembedt[],const PetscReal bembed[],
                                 PetscInt pinterp,const PetscReal binterpt[],const PetscReal binterp[])
{
  PetscErrorCode ierr;
  ARKTableauLink link;
  ARKTableau     t;
  PetscInt       i,j;

  PetscFunctionBegin;
  ierr     = TSARKIMEXInitializePackage();CHKERRQ(ierr);
  ierr     = PetscNew(&link);CHKERRQ(ierr);
  t        = &link->tab;
  ierr     = PetscStrallocpy(name,&t->name);CHKERRQ(ierr);
  t->order = order;
  t->s     = s;
  ierr     = PetscMalloc6(s*s,&t->At,s,&t->bt,s,&t->ct,s*s,&t->A,s,&t->b,s,&t->c);CHKERRQ(ierr);
  ierr     = PetscArraycpy(t->At,At,s*s);CHKERRQ(ierr);
  ierr     = PetscArraycpy(t->A,A,s*s);CHKERRQ(ierr);
  if (bt) { ierr = PetscArraycpy(t->bt,bt,s);CHKERRQ(ierr); }
  else for (i=0; i<s; i++) t->bt[i] = At[(s-1)*s+i];
  if (b)  { ierr = PetscArraycpy(t->b,b,s);CHKERRQ(ierr); }
  else for (i=0; i<s; i++) t->b[i] = t->bt[i];
  if (ct) { ierr = PetscArraycpy(t->ct,ct,s);CHKERRQ(ierr); }
  else for (i=0; i<s; i++) for (j=0,t->ct[i]=0; j<s; j++) t->ct[i] += At[i*s+j];
  if (c)  { ierr = PetscArraycpy(t->c,c,s);CHKERRQ(ierr); }
  else for (i=0; i<s; i++) for (j=0,t->c[i]=0; j<s; j++) t->c[i] += A[i*s+j];
  t->stiffly_accurate = PETSC_TRUE;
  for (i=0; i<s; i++) if (t->At[(s-1)*s+i] != t->bt[i]) t->stiffly_accurate = PETSC_FALSE;
  t->explicit_first_stage = PETSC_TRUE;
  for (i=0; i<s; i++) if (t->At[i] != 0.0) t->explicit_first_stage = PETSC_FALSE;
  /*def of FSAL can be made more precise*/
  t->FSAL_implicit = (PetscBool)(t->explicit_first_stage && t->stiffly_accurate);
  if (bembedt) {
    ierr = PetscMalloc2(s,&t->bembedt,s,&t->bembed);CHKERRQ(ierr);
    ierr = PetscArraycpy(t->bembedt,bembedt,s);CHKERRQ(ierr);
    ierr = PetscArraycpy(t->bembed,bembed ? bembed : bembedt,s);CHKERRQ(ierr);
  }

  t->pinterp     = pinterp;
  ierr           = PetscMalloc2(s*pinterp,&t->binterpt,s*pinterp,&t->binterp);CHKERRQ(ierr);
  ierr           = PetscArraycpy(t->binterpt,binterpt,s*pinterp);CHKERRQ(ierr);
  ierr           = PetscArraycpy(t->binterp,binterp ? binterp : binterpt,s*pinterp);CHKERRQ(ierr);
  link->next     = ARKTableauList;
  ARKTableauList = link;
  PetscFunctionReturn(0);
}

/*
 The step completion formula is

 x1 = x0 - h bt^T YdotI + h b^T YdotRHS

 This function can be called before or after ts->vec_sol has been updated.
 Suppose we have a completion formula (bt,b) and an embedded formula (bet,be) of different order.
 We can write

 x1e = x0 - h bet^T YdotI + h be^T YdotRHS
     = x1 + h bt^T YdotI - h b^T YdotRHS - h bet^T YdotI + h be^T YdotRHS
     = x1 - h (bet - bt)^T YdotI + h (be - b)^T YdotRHS

 so we can evaluate the method with different order even after the step has been optimistically completed.
*/
static PetscErrorCode TSEvaluateStep_ARKIMEX(TS ts,PetscInt order,Vec X,PetscBool *done)
{
  TS_ARKIMEX     *ark = (TS_ARKIMEX*)ts->data;
  ARKTableau     tab  = ark->tableau;
  PetscScalar    *w   = ark->work;
  PetscReal      h;
  PetscInt       s = tab->s,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (ark->status) {
  case TS_STEP_INCOMPLETE:
  case TS_STEP_PENDING:
    h = ts->time_step; break;
  case TS_STEP_COMPLETE:
    h = ts->ptime - ts->ptime_prev; break;
  default: SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Invalid TSStepStatus");
  }
  if (order == tab->order) {
    if (ark->status == TS_STEP_INCOMPLETE) {
      if (!ark->imex && tab->stiffly_accurate) { /* Only the stiffly accurate implicit formula is used */
        ierr = VecCopy(ark->Y[s-1],X);CHKERRQ(ierr);
      } else { /* Use the standard completion formula (bt,b) */
        ierr = VecCopy(ts->vec_sol,X);CHKERRQ(ierr);
        for (j=0; j<s; j++) w[j] = h*tab->bt[j];
        ierr = VecMAXPY(X,s,w,ark->YdotI);CHKERRQ(ierr);
        if (ark->imex) { /* Method is IMEX, complete the explicit formula */
          for (j=0; j<s; j++) w[j] = h*tab->b[j];
          ierr = VecMAXPY(X,s,w,ark->YdotRHS);CHKERRQ(ierr);
        }
      }
    } else {ierr = VecCopy(ts->vec_sol,X);CHKERRQ(ierr);}
    if (done) *done = PETSC_TRUE;
    PetscFunctionReturn(0);
  } else if (order == tab->order-1) {
    if (!tab->bembedt) goto unavailable;
    if (ark->status == TS_STEP_INCOMPLETE) { /* Complete with the embedded method (bet,be) */
      ierr = VecCopy(ts->vec_sol,X);CHKERRQ(ierr);
      for (j=0; j<s; j++) w[j] = h*tab->bembedt[j];
      ierr = VecMAXPY(X,s,w,ark->YdotI);CHKERRQ(ierr);
      for (j=0; j<s; j++) w[j] = h*tab->bembed[j];
      ierr = VecMAXPY(X,s,w,ark->YdotRHS);CHKERRQ(ierr);
    } else { /* Rollback and re-complete using (bet-be,be-b) */
      ierr = VecCopy(ts->vec_sol,X);CHKERRQ(ierr);
      for (j=0; j<s; j++) w[j] = h*(tab->bembedt[j] - tab->bt[j]);
      ierr = VecMAXPY(X,tab->s,w,ark->YdotI);CHKERRQ(ierr);
      for (j=0; j<s; j++) w[j] = h*(tab->bembed[j] - tab->b[j]);
      ierr = VecMAXPY(X,s,w,ark->YdotRHS);CHKERRQ(ierr);
    }
    if (done) *done = PETSC_TRUE;
    PetscFunctionReturn(0);
  }
unavailable:
  if (done) *done = PETSC_FALSE;
  else SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"ARKIMEX '%s' of order %D cannot evaluate step at order %D. Consider using -ts_adapt_type none or a different method that has an embedded estimate.",tab->name,tab->order,order);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSARKIMEXTestMassIdentity(TS ts,PetscBool *id)
{
  PetscErrorCode ierr;
  Vec            Udot,Y1,Y2;
  TS_ARKIMEX     *ark = (TS_ARKIMEX*)ts->data;
  PetscReal      norm;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&Udot);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&Y1);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&Y2);CHKERRQ(ierr);
  ierr = TSComputeIFunction(ts,ts->ptime,ts->vec_sol,Udot,Y1,ark->imex);CHKERRQ(ierr);
  ierr = VecSetRandom(Udot,NULL);CHKERRQ(ierr);
  ierr = TSComputeIFunction(ts,ts->ptime,ts->vec_sol,Udot,Y2,ark->imex);CHKERRQ(ierr);
  ierr = VecAXPY(Y2,-1.0,Y1);CHKERRQ(ierr);
  ierr = VecAXPY(Y2,-1.0,Udot);CHKERRQ(ierr);
  ierr = VecNorm(Y2,NORM_2,&norm);CHKERRQ(ierr);
  if (norm < 100.0*PETSC_MACHINE_EPSILON) {
    *id = PETSC_TRUE;
  } else {
    *id = PETSC_FALSE;
    ierr = PetscInfo((PetscObject)ts,"IFunction(Udot = random) - IFunction(Udot = 0) is not near Udot, %g, suspect mass matrix implied in IFunction() is not the identity as required\n",(double)norm);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&Udot);CHKERRQ(ierr);
  ierr = VecDestroy(&Y1);CHKERRQ(ierr);
  ierr = VecDestroy(&Y2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSRollBack_ARKIMEX(TS ts)
{
  TS_ARKIMEX      *ark = (TS_ARKIMEX*)ts->data;
  ARKTableau      tab  = ark->tableau;
  const PetscInt  s    = tab->s;
  const PetscReal *bt  = tab->bt,*b = tab->b;
  PetscScalar     *w   = ark->work;
  Vec             *YdotI = ark->YdotI,*YdotRHS = ark->YdotRHS;
  PetscInt        j;
  PetscReal       h;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  switch (ark->status) {
  case TS_STEP_INCOMPLETE:
  case TS_STEP_PENDING:
    h = ts->time_step; break;
  case TS_STEP_COMPLETE:
    h = ts->ptime - ts->ptime_prev; break;
  default: SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Invalid TSStepStatus");
  }
  for (j=0; j<s; j++) w[j] = -h*bt[j];
  ierr = VecMAXPY(ts->vec_sol,s,w,YdotI);CHKERRQ(ierr);
  for (j=0; j<s; j++) w[j] = -h*b[j];
  ierr = VecMAXPY(ts->vec_sol,s,w,YdotRHS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSStep_ARKIMEX(TS ts)
{
  TS_ARKIMEX      *ark = (TS_ARKIMEX*)ts->data;
  ARKTableau      tab  = ark->tableau;
  const PetscInt  s    = tab->s;
  const PetscReal *At  = tab->At,*A = tab->A,*ct = tab->ct,*c = tab->c;
  PetscScalar     *w   = ark->work;
  Vec             *Y   = ark->Y,*YdotI = ark->YdotI,*YdotRHS = ark->YdotRHS,Ydot = ark->Ydot,Ydot0 = ark->Ydot0,Z = ark->Z;
  PetscBool       extrapolate = ark->extrapolate;
  TSAdapt         adapt;
  SNES            snes;
  PetscInt        i,j,its,lits;
  PetscInt        rejections = 0;
  PetscBool       stageok,accept = PETSC_TRUE;
  PetscReal       next_time_step = ts->time_step;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (ark->extrapolate && !ark->Y_prev) {
    ierr = VecDuplicateVecs(ts->vec_sol,tab->s,&ark->Y_prev);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(ts->vec_sol,tab->s,&ark->YdotI_prev);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(ts->vec_sol,tab->s,&ark->YdotRHS_prev);CHKERRQ(ierr);
  }

  if (!ts->steprollback) {
    if (ts->equation_type >= TS_EQ_IMPLICIT) { /* Save the initial slope for the next step */
      ierr = VecCopy(YdotI[s-1],Ydot0);CHKERRQ(ierr);
    }
    if (ark->extrapolate && !ts->steprestart) { /* Save the Y, YdotI, YdotRHS for extrapolation initial guess */
      for (i = 0; i<s; i++) {
        ierr = VecCopy(Y[i],ark->Y_prev[i]);CHKERRQ(ierr);
        ierr = VecCopy(YdotRHS[i],ark->YdotRHS_prev[i]);CHKERRQ(ierr);
        ierr = VecCopy(YdotI[i],ark->YdotI_prev[i]);CHKERRQ(ierr);
      }
    }
  }

  if (ts->equation_type >= TS_EQ_IMPLICIT && tab->explicit_first_stage && ts->steprestart) {
    TS ts_start;
    if (PetscDefined(USE_DEBUG)) {
      PetscBool id = PETSC_FALSE;
      ierr = TSARKIMEXTestMassIdentity(ts,&id);CHKERRQ(ierr);
      PetscAssertFalse(!id,PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_INCOMP,"This scheme requires an identity mass matrix, however the TSIFunction you provide does not utilize an identity mass matrix");
    }
    ierr = TSClone(ts,&ts_start);CHKERRQ(ierr);
    ierr = TSSetSolution(ts_start,ts->vec_sol);CHKERRQ(ierr);
    ierr = TSSetTime(ts_start,ts->ptime);CHKERRQ(ierr);
    ierr = TSSetMaxSteps(ts_start,ts->steps+1);CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts_start,ts->ptime+ts->time_step);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts_start,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts_start,ts->time_step);CHKERRQ(ierr);
    ierr = TSSetType(ts_start,TSARKIMEX);CHKERRQ(ierr);
    ierr = TSARKIMEXSetFullyImplicit(ts_start,PETSC_TRUE);CHKERRQ(ierr);
    ierr = TSARKIMEXSetType(ts_start,TSARKIMEX1BEE);CHKERRQ(ierr);

    ierr = TSRestartStep(ts_start);CHKERRQ(ierr);
    ierr = TSSolve(ts_start,ts->vec_sol);CHKERRQ(ierr);
    ierr = TSGetTime(ts_start,&ts->ptime);CHKERRQ(ierr);
    ierr = TSGetTimeStep(ts_start,&ts->time_step);CHKERRQ(ierr);

    { /* Save the initial slope for the next step */
      TS_ARKIMEX *ark_start = (TS_ARKIMEX*)ts_start->data;
      ierr = VecCopy(ark_start->YdotI[ark_start->tableau->s-1],Ydot0);CHKERRQ(ierr);
    }
    ts->steps++;

    /* Set the correct TS in SNES */
    /* We'll try to bypass this by changing the method on the fly */
    {
      ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
      ierr = TSSetSNES(ts,snes);CHKERRQ(ierr);
    }
    ierr = TSDestroy(&ts_start);CHKERRQ(ierr);
  }

  ark->status = TS_STEP_INCOMPLETE;
  while (!ts->reason && ark->status != TS_STEP_COMPLETE) {
    PetscReal t = ts->ptime;
    PetscReal h = ts->time_step;
    for (i=0; i<s; i++) {
      ark->stage_time = t + h*ct[i];
      ierr = TSPreStage(ts,ark->stage_time);CHKERRQ(ierr);
      if (At[i*s+i] == 0) { /* This stage is explicit */
        PetscAssertFalse(i!=0 && ts->equation_type >= TS_EQ_IMPLICIT,PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Explicit stages other than the first one are not supported for implicit problems");
        ierr = VecCopy(ts->vec_sol,Y[i]);CHKERRQ(ierr);
        for (j=0; j<i; j++) w[j] = h*At[i*s+j];
        ierr = VecMAXPY(Y[i],i,w,YdotI);CHKERRQ(ierr);
        for (j=0; j<i; j++) w[j] = h*A[i*s+j];
        ierr = VecMAXPY(Y[i],i,w,YdotRHS);CHKERRQ(ierr);
      } else {
        ark->scoeff = 1./At[i*s+i];
        /* Ydot = shift*(Y-Z) */
        ierr = VecCopy(ts->vec_sol,Z);CHKERRQ(ierr);
        for (j=0; j<i; j++) w[j] = h*At[i*s+j];
        ierr = VecMAXPY(Z,i,w,YdotI);CHKERRQ(ierr);
        for (j=0; j<i; j++) w[j] = h*A[i*s+j];
        ierr = VecMAXPY(Z,i,w,YdotRHS);CHKERRQ(ierr);
        if (extrapolate && !ts->steprestart) {
          /* Initial guess extrapolated from previous time step stage values */
          ierr = TSExtrapolate_ARKIMEX(ts,c[i],Y[i]);CHKERRQ(ierr);
        } else {
          /* Initial guess taken from last stage */
          ierr = VecCopy(i>0 ? Y[i-1] : ts->vec_sol,Y[i]);CHKERRQ(ierr);
        }
        ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
        ierr = SNESSolve(snes,NULL,Y[i]);CHKERRQ(ierr);
        ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
        ierr = SNESGetLinearSolveIterations(snes,&lits);CHKERRQ(ierr);
        ts->snes_its += its; ts->ksp_its += lits;
        ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
        ierr = TSAdaptCheckStage(adapt,ts,ark->stage_time,Y[i],&stageok);CHKERRQ(ierr);
        if (!stageok) {
          /* We are likely rejecting the step because of solver or function domain problems so we should not attempt to
           * use extrapolation to initialize the solves on the next attempt. */
          extrapolate = PETSC_FALSE;
          goto reject_step;
        }
      }
      if (ts->equation_type >= TS_EQ_IMPLICIT) {
        if (i==0 && tab->explicit_first_stage) {
          PetscAssertFalse(!tab->stiffly_accurate,PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"TSARKIMEX %s is not stiffly accurate and therefore explicit-first stage methods cannot be used if the equation is implicit because the slope cannot be evaluated",ark->tableau->name);
          ierr = VecCopy(Ydot0,YdotI[0]);CHKERRQ(ierr);                                      /* YdotI = YdotI(tn-1) */
        } else {
          ierr = VecAXPBYPCZ(YdotI[i],-ark->scoeff/h,ark->scoeff/h,0,Z,Y[i]);CHKERRQ(ierr);  /* YdotI = shift*(X-Z) */
        }
      } else {
        if (i==0 && tab->explicit_first_stage) {
          ierr = VecZeroEntries(Ydot);CHKERRQ(ierr);
          ierr = TSComputeIFunction(ts,t+h*ct[i],Y[i],Ydot,YdotI[i],ark->imex);CHKERRQ(ierr);/* YdotI = -G(t,Y,0)   */
          ierr = VecScale(YdotI[i],-1.0);CHKERRQ(ierr);
        } else {
          ierr = VecAXPBYPCZ(YdotI[i],-ark->scoeff/h,ark->scoeff/h,0,Z,Y[i]);CHKERRQ(ierr);  /* YdotI = shift*(X-Z) */
        }
        if (ark->imex) {
          ierr = TSComputeRHSFunction(ts,t+h*c[i],Y[i],YdotRHS[i]);CHKERRQ(ierr);
        } else {
          ierr = VecZeroEntries(YdotRHS[i]);CHKERRQ(ierr);
        }
      }
      ierr = TSPostStage(ts,ark->stage_time,i,Y);CHKERRQ(ierr);
    }

    ark->status = TS_STEP_INCOMPLETE;
    ierr = TSEvaluateStep_ARKIMEX(ts,tab->order,ts->vec_sol,NULL);CHKERRQ(ierr);
    ark->status = TS_STEP_PENDING;
    ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
    ierr = TSAdaptCandidatesClear(adapt);CHKERRQ(ierr);
    ierr = TSAdaptCandidateAdd(adapt,tab->name,tab->order,1,tab->ccfl,(PetscReal)tab->s,PETSC_TRUE);CHKERRQ(ierr);
    ierr = TSAdaptChoose(adapt,ts,ts->time_step,NULL,&next_time_step,&accept);CHKERRQ(ierr);
    ark->status = accept ? TS_STEP_COMPLETE : TS_STEP_INCOMPLETE;
    if (!accept) { /* Roll back the current step */
      ierr = TSRollBack_ARKIMEX(ts);CHKERRQ(ierr);
      ts->time_step = next_time_step;
      goto reject_step;
    }

    ts->ptime += ts->time_step;
    ts->time_step = next_time_step;
    break;

  reject_step:
    ts->reject++; accept = PETSC_FALSE;
    if (!ts->reason && ++rejections > ts->max_reject && ts->max_reject >= 0) {
      ts->reason = TS_DIVERGED_STEP_REJECTED;
      ierr = PetscInfo(ts,"Step=%D, step rejections %D greater than current TS allowed, stopping solve\n",ts->steps,rejections);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSInterpolate_ARKIMEX(TS ts,PetscReal itime,Vec X)
{
  TS_ARKIMEX      *ark = (TS_ARKIMEX*)ts->data;
  PetscInt        s    = ark->tableau->s,pinterp = ark->tableau->pinterp,i,j;
  PetscReal       h;
  PetscReal       tt,t;
  PetscScalar     *bt,*b;
  const PetscReal *Bt = ark->tableau->binterpt,*B = ark->tableau->binterp;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscAssertFalse(!Bt || !B,PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"TSARKIMEX %s does not have an interpolation formula",ark->tableau->name);
  switch (ark->status) {
  case TS_STEP_INCOMPLETE:
  case TS_STEP_PENDING:
    h = ts->time_step;
    t = (itime - ts->ptime)/h;
    break;
  case TS_STEP_COMPLETE:
    h = ts->ptime - ts->ptime_prev;
    t = (itime - ts->ptime)/h + 1; /* In the interval [0,1] */
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Invalid TSStepStatus");
  }
  ierr = PetscMalloc2(s,&bt,s,&b);CHKERRQ(ierr);
  for (i=0; i<s; i++) bt[i] = b[i] = 0;
  for (j=0,tt=t; j<pinterp; j++,tt*=t) {
    for (i=0; i<s; i++) {
      bt[i] += h * Bt[i*pinterp+j] * tt;
      b[i]  += h * B[i*pinterp+j] * tt;
    }
  }
  ierr = VecCopy(ark->Y[0],X);CHKERRQ(ierr);
  ierr = VecMAXPY(X,s,bt,ark->YdotI);CHKERRQ(ierr);
  ierr = VecMAXPY(X,s,b,ark->YdotRHS);CHKERRQ(ierr);
  ierr = PetscFree2(bt,b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSExtrapolate_ARKIMEX(TS ts,PetscReal c,Vec X)
{
  TS_ARKIMEX      *ark = (TS_ARKIMEX*)ts->data;
  PetscInt        s = ark->tableau->s,pinterp = ark->tableau->pinterp,i,j;
  PetscReal       h,h_prev,t,tt;
  PetscScalar     *bt,*b;
  const PetscReal *Bt = ark->tableau->binterpt,*B = ark->tableau->binterp;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscAssertFalse(!Bt || !B,PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"TSARKIMEX %s does not have an interpolation formula",ark->tableau->name);
  ierr = PetscCalloc2(s,&bt,s,&b);CHKERRQ(ierr);
  h = ts->time_step;
  h_prev = ts->ptime - ts->ptime_prev;
  t = 1 + h/h_prev*c;
  for (j=0,tt=t; j<pinterp; j++,tt*=t) {
    for (i=0; i<s; i++) {
      bt[i] += h * Bt[i*pinterp+j] * tt;
      b[i]  += h * B[i*pinterp+j] * tt;
    }
  }
  PetscAssertFalse(!ark->Y_prev,PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Stages from previous step have not been stored");
  ierr = VecCopy(ark->Y_prev[0],X);CHKERRQ(ierr);
  ierr = VecMAXPY(X,s,bt,ark->YdotI_prev);CHKERRQ(ierr);
  ierr = VecMAXPY(X,s,b,ark->YdotRHS_prev);CHKERRQ(ierr);
  ierr = PetscFree2(bt,b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode TSARKIMEXTableauReset(TS ts)
{
  TS_ARKIMEX     *ark = (TS_ARKIMEX*)ts->data;
  ARKTableau     tab  = ark->tableau;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tab) PetscFunctionReturn(0);
  ierr = PetscFree(ark->work);CHKERRQ(ierr);
  ierr = VecDestroyVecs(tab->s,&ark->Y);CHKERRQ(ierr);
  ierr = VecDestroyVecs(tab->s,&ark->YdotI);CHKERRQ(ierr);
  ierr = VecDestroyVecs(tab->s,&ark->YdotRHS);CHKERRQ(ierr);
  ierr = VecDestroyVecs(tab->s,&ark->Y_prev);CHKERRQ(ierr);
  ierr = VecDestroyVecs(tab->s,&ark->YdotI_prev);CHKERRQ(ierr);
  ierr = VecDestroyVecs(tab->s,&ark->YdotRHS_prev);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSReset_ARKIMEX(TS ts)
{
  TS_ARKIMEX     *ark = (TS_ARKIMEX*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSARKIMEXTableauReset(ts);CHKERRQ(ierr);
  ierr = VecDestroy(&ark->Ydot);CHKERRQ(ierr);
  ierr = VecDestroy(&ark->Ydot0);CHKERRQ(ierr);
  ierr = VecDestroy(&ark->Z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSARKIMEXGetVecs(TS ts,DM dm,Vec *Z,Vec *Ydot)
{
  TS_ARKIMEX     *ax = (TS_ARKIMEX*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (Z) {
    if (dm && dm != ts->dm) {
      ierr = DMGetNamedGlobalVector(dm,"TSARKIMEX_Z",Z);CHKERRQ(ierr);
    } else *Z = ax->Z;
  }
  if (Ydot) {
    if (dm && dm != ts->dm) {
      ierr = DMGetNamedGlobalVector(dm,"TSARKIMEX_Ydot",Ydot);CHKERRQ(ierr);
    } else *Ydot = ax->Ydot;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSARKIMEXRestoreVecs(TS ts,DM dm,Vec *Z,Vec *Ydot)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (Z) {
    if (dm && dm != ts->dm) {
      ierr = DMRestoreNamedGlobalVector(dm,"TSARKIMEX_Z",Z);CHKERRQ(ierr);
    }
  }
  if (Ydot) {
    if (dm && dm != ts->dm) {
      ierr = DMRestoreNamedGlobalVector(dm,"TSARKIMEX_Ydot",Ydot);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*
  This defines the nonlinear equation that is to be solved with SNES
  G(U) = F[t0+Theta*dt, U, (U-U0)*shift] = 0
*/
static PetscErrorCode SNESTSFormFunction_ARKIMEX(SNES snes,Vec X,Vec F,TS ts)
{
  TS_ARKIMEX     *ark = (TS_ARKIMEX*)ts->data;
  DM             dm,dmsave;
  Vec            Z,Ydot;
  PetscReal      shift = ark->scoeff / ts->time_step;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr   = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr   = TSARKIMEXGetVecs(ts,dm,&Z,&Ydot);CHKERRQ(ierr);
  ierr   = VecAXPBYPCZ(Ydot,-shift,shift,0,Z,X);CHKERRQ(ierr); /* Ydot = shift*(X-Z) */
  dmsave = ts->dm;
  ts->dm = dm;

  ierr = TSComputeIFunction(ts,ark->stage_time,X,Ydot,F,ark->imex);CHKERRQ(ierr);

  ts->dm = dmsave;
  ierr   = TSARKIMEXRestoreVecs(ts,dm,&Z,&Ydot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESTSFormJacobian_ARKIMEX(SNES snes,Vec X,Mat A,Mat B,TS ts)
{
  TS_ARKIMEX     *ark = (TS_ARKIMEX*)ts->data;
  DM             dm,dmsave;
  Vec            Ydot;
  PetscReal      shift = ark->scoeff / ts->time_step;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = TSARKIMEXGetVecs(ts,dm,NULL,&Ydot);CHKERRQ(ierr);
  /* ark->Ydot has already been computed in SNESTSFormFunction_ARKIMEX (SNES guarantees this) */
  dmsave = ts->dm;
  ts->dm = dm;

  ierr = TSComputeIJacobian(ts,ark->stage_time,X,Ydot,shift,A,B,ark->imex);CHKERRQ(ierr);

  ts->dm = dmsave;
  ierr   = TSARKIMEXRestoreVecs(ts,dm,NULL,&Ydot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCoarsenHook_TSARKIMEX(DM fine,DM coarse,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRestrictHook_TSARKIMEX(DM fine,Mat restrct,Vec rscale,Mat inject,DM coarse,void *ctx)
{
  TS             ts = (TS)ctx;
  PetscErrorCode ierr;
  Vec            Z,Z_c;

  PetscFunctionBegin;
  ierr = TSARKIMEXGetVecs(ts,fine,&Z,NULL);CHKERRQ(ierr);
  ierr = TSARKIMEXGetVecs(ts,coarse,&Z_c,NULL);CHKERRQ(ierr);
  ierr = MatRestrict(restrct,Z,Z_c);CHKERRQ(ierr);
  ierr = VecPointwiseMult(Z_c,rscale,Z_c);CHKERRQ(ierr);
  ierr = TSARKIMEXRestoreVecs(ts,fine,&Z,NULL);CHKERRQ(ierr);
  ierr = TSARKIMEXRestoreVecs(ts,coarse,&Z_c,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSubDomainHook_TSARKIMEX(DM dm,DM subdm,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSubDomainRestrictHook_TSARKIMEX(DM dm,VecScatter gscat,VecScatter lscat,DM subdm,void *ctx)
{
  TS             ts = (TS)ctx;
  PetscErrorCode ierr;
  Vec            Z,Z_c;

  PetscFunctionBegin;
  ierr = TSARKIMEXGetVecs(ts,dm,&Z,NULL);CHKERRQ(ierr);
  ierr = TSARKIMEXGetVecs(ts,subdm,&Z_c,NULL);CHKERRQ(ierr);

  ierr = VecScatterBegin(gscat,Z,Z_c,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(gscat,Z,Z_c,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = TSARKIMEXRestoreVecs(ts,dm,&Z,NULL);CHKERRQ(ierr);
  ierr = TSARKIMEXRestoreVecs(ts,subdm,&Z_c,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSARKIMEXTableauSetUp(TS ts)
{
  TS_ARKIMEX     *ark = (TS_ARKIMEX*)ts->data;
  ARKTableau     tab  = ark->tableau;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(tab->s,&ark->work);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(ts->vec_sol,tab->s,&ark->Y);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(ts->vec_sol,tab->s,&ark->YdotI);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(ts->vec_sol,tab->s,&ark->YdotRHS);CHKERRQ(ierr);
  if (ark->extrapolate) {
    ierr = VecDuplicateVecs(ts->vec_sol,tab->s,&ark->Y_prev);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(ts->vec_sol,tab->s,&ark->YdotI_prev);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(ts->vec_sol,tab->s,&ark->YdotRHS_prev);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetUp_ARKIMEX(TS ts)
{
  TS_ARKIMEX     *ark = (TS_ARKIMEX*)ts->data;
  PetscErrorCode ierr;
  DM             dm;
  SNES           snes;

  PetscFunctionBegin;
  ierr = TSARKIMEXTableauSetUp(ts);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&ark->Ydot);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&ark->Ydot0);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&ark->Z);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMCoarsenHookAdd(dm,DMCoarsenHook_TSARKIMEX,DMRestrictHook_TSARKIMEX,ts);CHKERRQ(ierr);
  ierr = DMSubDomainHookAdd(dm,DMSubDomainHook_TSARKIMEX,DMSubDomainRestrictHook_TSARKIMEX,ts);CHKERRQ(ierr);
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

static PetscErrorCode TSSetFromOptions_ARKIMEX(PetscOptionItems *PetscOptionsObject,TS ts)
{
  TS_ARKIMEX     *ark = (TS_ARKIMEX*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"ARKIMEX ODE solver options");CHKERRQ(ierr);
  {
    ARKTableauLink link;
    PetscInt       count,choice;
    PetscBool      flg;
    const char     **namelist;
    for (link=ARKTableauList,count=0; link; link=link->next,count++) ;
    ierr = PetscMalloc1(count,(char***)&namelist);CHKERRQ(ierr);
    for (link=ARKTableauList,count=0; link; link=link->next,count++) namelist[count] = link->tab.name;
    ierr = PetscOptionsEList("-ts_arkimex_type","Family of ARK IMEX method","TSARKIMEXSetType",(const char*const*)namelist,count,ark->tableau->name,&choice,&flg);CHKERRQ(ierr);
    if (flg) {ierr = TSARKIMEXSetType(ts,namelist[choice]);CHKERRQ(ierr);}
    ierr = PetscFree(namelist);CHKERRQ(ierr);

    flg  = (PetscBool) !ark->imex;
    ierr = PetscOptionsBool("-ts_arkimex_fully_implicit","Solve the problem fully implicitly","TSARKIMEXSetFullyImplicit",flg,&flg,NULL);CHKERRQ(ierr);
    ark->imex = (PetscBool) !flg;
    ierr = PetscOptionsBool("-ts_arkimex_initial_guess_extrapolate","Extrapolate the initial guess for the stage solution from stage values of the previous time step","",ark->extrapolate,&ark->extrapolate,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSView_ARKIMEX(TS ts,PetscViewer viewer)
{
  TS_ARKIMEX     *ark = (TS_ARKIMEX*)ts->data;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ARKTableau    tab = ark->tableau;
    TSARKIMEXType arktype;
    char          buf[512];
    PetscBool     flg;

    ierr = TSARKIMEXGetType(ts,&arktype);CHKERRQ(ierr);
    ierr = TSARKIMEXGetFullyImplicit(ts,&flg);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  ARK IMEX %s\n",arktype);CHKERRQ(ierr);
    ierr = PetscFormatRealArray(buf,sizeof(buf),"% 8.6f",tab->s,tab->ct);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Stiff abscissa       ct = %s\n",buf);CHKERRQ(ierr);
    ierr = PetscFormatRealArray(buf,sizeof(buf),"% 8.6f",tab->s,tab->c);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Fully implicit: %s\n",flg ? "yes" : "no");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Stiffly accurate: %s\n",tab->stiffly_accurate ? "yes" : "no");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Explicit first stage: %s\n",tab->explicit_first_stage ? "yes" : "no");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"FSAL property: %s\n",tab->FSAL_implicit ? "yes" : "no");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Nonstiff abscissa     c = %s\n",buf);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSLoad_ARKIMEX(TS ts,PetscViewer viewer)
{
  PetscErrorCode ierr;
  SNES           snes;
  TSAdapt        adapt;

  PetscFunctionBegin;
  ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptLoad(adapt,viewer);CHKERRQ(ierr);
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESLoad(snes,viewer);CHKERRQ(ierr);
  /* function and Jacobian context for SNES when used with TS is always ts object */
  ierr = SNESSetFunction(snes,NULL,NULL,ts);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,NULL,NULL,NULL,ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSARKIMEXSetType - Set the type of ARK IMEX scheme

  Logically collective

  Input Parameters:
+  ts - timestepping context
-  arktype - type of ARK-IMEX scheme

  Options Database:
.  -ts_arkimex_type <1bee,a2,l2,ars122,2c,2d,2e,prssp2,3,bpr3,ars443,4,5>

  Level: intermediate

.seealso: TSARKIMEXGetType(), TSARKIMEX, TSARKIMEXType, TSARKIMEX1BEE, TSARKIMEXA2, TSARKIMEXL2, TSARKIMEXARS122, TSARKIMEX2C, TSARKIMEX2D, TSARKIMEX2E, TSARKIMEXPRSSP2,
          TSARKIMEX3, TSARKIMEXBPR3, TSARKIMEXARS443, TSARKIMEX4, TSARKIMEX5
@*/
PetscErrorCode TSARKIMEXSetType(TS ts,TSARKIMEXType arktype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidCharPointer(arktype,2);
  ierr = PetscTryMethod(ts,"TSARKIMEXSetType_C",(TS,TSARKIMEXType),(ts,arktype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSARKIMEXGetType - Get the type of ARK IMEX scheme

  Logically collective

  Input Parameter:
.  ts - timestepping context

  Output Parameter:
.  arktype - type of ARK-IMEX scheme

  Level: intermediate

.seealso: TSARKIMEXGetType()
@*/
PetscErrorCode TSARKIMEXGetType(TS ts,TSARKIMEXType *arktype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscUseMethod(ts,"TSARKIMEXGetType_C",(TS,TSARKIMEXType*),(ts,arktype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TSARKIMEXSetFullyImplicit - Solve both parts of the equation implicitly

  Logically collective

  Input Parameters:
+  ts - timestepping context
-  flg - PETSC_TRUE for fully implicit

  Level: intermediate

.seealso: TSARKIMEXGetType(), TSARKIMEXGetFullyImplicit()
@*/
PetscErrorCode TSARKIMEXSetFullyImplicit(TS ts,PetscBool flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveBool(ts,flg,2);
  ierr = PetscTryMethod(ts,"TSARKIMEXSetFullyImplicit_C",(TS,PetscBool),(ts,flg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TSARKIMEXGetFullyImplicit - Inquires if both parts of the equation are solved implicitly

  Logically collective

  Input Parameter:
.  ts - timestepping context

  Output Parameter:
.  flg - PETSC_TRUE for fully implicit

  Level: intermediate

.seealso: TSARKIMEXGetType(), TSARKIMEXSetFullyImplicit()
@*/
PetscErrorCode TSARKIMEXGetFullyImplicit(TS ts,PetscBool *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(flg,2);
  ierr = PetscUseMethod(ts,"TSARKIMEXGetFullyImplicit_C",(TS,PetscBool*),(ts,flg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  TSARKIMEXGetType_ARKIMEX(TS ts,TSARKIMEXType *arktype)
{
  TS_ARKIMEX     *ark = (TS_ARKIMEX*)ts->data;

  PetscFunctionBegin;
  *arktype = ark->tableau->name;
  PetscFunctionReturn(0);
}
static PetscErrorCode  TSARKIMEXSetType_ARKIMEX(TS ts,TSARKIMEXType arktype)
{
  TS_ARKIMEX     *ark = (TS_ARKIMEX*)ts->data;
  PetscErrorCode ierr;
  PetscBool      match;
  ARKTableauLink link;

  PetscFunctionBegin;
  if (ark->tableau) {
    ierr = PetscStrcmp(ark->tableau->name,arktype,&match);CHKERRQ(ierr);
    if (match) PetscFunctionReturn(0);
  }
  for (link = ARKTableauList; link; link=link->next) {
    ierr = PetscStrcmp(link->tab.name,arktype,&match);CHKERRQ(ierr);
    if (match) {
      if (ts->setupcalled) {ierr = TSARKIMEXTableauReset(ts);CHKERRQ(ierr);}
      ark->tableau = &link->tab;
      if (ts->setupcalled) {ierr = TSARKIMEXTableauSetUp(ts);CHKERRQ(ierr);}
      ts->default_adapt_type = ark->tableau->bembed ? TSADAPTBASIC : TSADAPTNONE;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_UNKNOWN_TYPE,"Could not find '%s'",arktype);
}

static PetscErrorCode  TSARKIMEXSetFullyImplicit_ARKIMEX(TS ts,PetscBool flg)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX*)ts->data;

  PetscFunctionBegin;
  ark->imex = (PetscBool)!flg;
  PetscFunctionReturn(0);
}

static PetscErrorCode  TSARKIMEXGetFullyImplicit_ARKIMEX(TS ts,PetscBool *flg)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX*)ts->data;

  PetscFunctionBegin;
  *flg = (PetscBool)!ark->imex;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDestroy_ARKIMEX(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_ARKIMEX(ts);CHKERRQ(ierr);
  if (ts->dm) {
    ierr = DMCoarsenHookRemove(ts->dm,DMCoarsenHook_TSARKIMEX,DMRestrictHook_TSARKIMEX,ts);CHKERRQ(ierr);
    ierr = DMSubDomainHookRemove(ts->dm,DMSubDomainHook_TSARKIMEX,DMSubDomainRestrictHook_TSARKIMEX,ts);CHKERRQ(ierr);
  }
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSARKIMEXGetType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSARKIMEXSetType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSARKIMEXSetFullyImplicit_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSARKIMEXSetFullyImplicit_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
/*MC
      TSARKIMEX - ODE and DAE solver using additive Runge-Kutta IMEX schemes

  These methods are intended for problems with well-separated time scales, especially when a slow scale is strongly
  nonlinear such that it is expensive to solve with a fully implicit method. The user should provide the stiff part
  of the equation using TSSetIFunction() and the non-stiff part with TSSetRHSFunction().

  Notes:
  The default is TSARKIMEX3, it can be changed with TSARKIMEXSetType() or -ts_arkimex_type

  If the equation is implicit or a DAE, then TSSetEquationType() needs to be set accordingly. Refer to the manual for further information.

  Methods with an explicit stage can only be used with ODE in which the stiff part G(t,X,Xdot) has the form Xdot + Ghat(t,X).

  Consider trying TSROSW if the stiff part is linear or weakly nonlinear.

  Level: beginner

.seealso:  TSCreate(), TS, TSSetType(), TSARKIMEXSetType(), TSARKIMEXGetType(), TSARKIMEXSetFullyImplicit(), TSARKIMEXGetFullyImplicit(),
           TSARKIMEX1BEE, TSARKIMEX2C, TSARKIMEX2D, TSARKIMEX2E, TSARKIMEX3, TSARKIMEXL2, TSARKIMEXA2, TSARKIMEXARS122,
           TSARKIMEX4, TSARKIMEX5, TSARKIMEXPRSSP2, TSARKIMEXARS443, TSARKIMEXBPR3, TSARKIMEXType, TSARKIMEXRegister()

M*/
PETSC_EXTERN PetscErrorCode TSCreate_ARKIMEX(TS ts)
{
  TS_ARKIMEX     *th;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSARKIMEXInitializePackage();CHKERRQ(ierr);

  ts->ops->reset          = TSReset_ARKIMEX;
  ts->ops->destroy        = TSDestroy_ARKIMEX;
  ts->ops->view           = TSView_ARKIMEX;
  ts->ops->load           = TSLoad_ARKIMEX;
  ts->ops->setup          = TSSetUp_ARKIMEX;
  ts->ops->step           = TSStep_ARKIMEX;
  ts->ops->interpolate    = TSInterpolate_ARKIMEX;
  ts->ops->evaluatestep   = TSEvaluateStep_ARKIMEX;
  ts->ops->rollback       = TSRollBack_ARKIMEX;
  ts->ops->setfromoptions = TSSetFromOptions_ARKIMEX;
  ts->ops->snesfunction   = SNESTSFormFunction_ARKIMEX;
  ts->ops->snesjacobian   = SNESTSFormJacobian_ARKIMEX;

  ts->usessnes = PETSC_TRUE;

  ierr = PetscNewLog(ts,&th);CHKERRQ(ierr);
  ts->data = (void*)th;
  th->imex = PETSC_TRUE;

  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSARKIMEXGetType_C",TSARKIMEXGetType_ARKIMEX);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSARKIMEXSetType_C",TSARKIMEXSetType_ARKIMEX);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSARKIMEXSetFullyImplicit_C",TSARKIMEXSetFullyImplicit_ARKIMEX);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSARKIMEXGetFullyImplicit_C",TSARKIMEXGetFullyImplicit_ARKIMEX);CHKERRQ(ierr);

  ierr = TSARKIMEXSetType(ts,TSARKIMEXDefault);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
