/*
  Code for time stepping with the Runge-Kutta method

  Notes:
  The general system is written as

  Udot = F(t,U)

*/

#include <petsc/private/tsimpl.h>                /*I   "petscts.h"   I*/
#include <petscdm.h>
#include <../src/ts/impls/explicit/rk/rk.h>
#include <../src/ts/impls/explicit/rk/mrk.h>

static TSRKType  TSRKDefault = TSRK3BS;
static PetscBool TSRKRegisterAllCalled;
static PetscBool TSRKPackageInitialized;

static RKTableauLink RKTableauList;

/*MC
     TSRK1FE - First order forward Euler scheme.

     This method has one stage.

     Options database:
.     -ts_rk_type 1fe

     Level: advanced

.seealso: TSRK, TSRKType, TSRKSetType()
M*/
/*MC
     TSRK2A - Second order RK scheme.

     This method has two stages.

     Options database:
.     -ts_rk_type 2a

     Level: advanced

.seealso: TSRK, TSRKType, TSRKSetType()
M*/
/*MC
     TSRK3 - Third order RK scheme.

     This method has three stages.

     Options database:
.     -ts_rk_type 3

     Level: advanced

.seealso: TSRK, TSRKType, TSRKSetType()
M*/
/*MC
     TSRK3BS - Third order RK scheme of Bogacki-Shampine with 2nd order embedded method.

     This method has four stages with the First Same As Last (FSAL) property.

     Options database:
.     -ts_rk_type 3bs

     Level: advanced

     References: https://doi.org/10.1016/0893-9659(89)90079-7

.seealso: TSRK, TSRKType, TSRKSetType()
M*/
/*MC
     TSRK4 - Fourth order RK scheme.

     This is the classical Runge-Kutta method with four stages.

     Options database:
.     -ts_rk_type 4

     Level: advanced

.seealso: TSRK, TSRKType, TSRKSetType()
M*/
/*MC
     TSRK5F - Fifth order Fehlberg RK scheme with a 4th order embedded method.

     This method has six stages.

     Options database:
.     -ts_rk_type 5f

     Level: advanced

.seealso: TSRK, TSRKType, TSRKSetType()
M*/
/*MC
     TSRK5DP - Fifth order Dormand-Prince RK scheme with the 4th order embedded method.

     This method has seven stages with the First Same As Last (FSAL) property.

     Options database:
.     -ts_rk_type 5dp

     Level: advanced

     References: https://doi.org/10.1016/0771-050X(80)90013-3

.seealso: TSRK, TSRKType, TSRKSetType()
M*/
/*MC
     TSRK5BS - Fifth order Bogacki-Shampine RK scheme with 4th order embedded method.

     This method has eight stages with the First Same As Last (FSAL) property.

     Options database:
.     -ts_rk_type 5bs

     Level: advanced

     References: https://doi.org/10.1016/0898-1221(96)00141-1

.seealso: TSRK, TSRKType, TSRKSetType()
M*/
/*MC
     TSRK6VR - Sixth order robust Verner RK scheme with fifth order embedded method.

     This method has nine stages with the First Same As Last (FSAL) property.

     Options database:
.     -ts_rk_type 6vr

     Level: advanced

     References: http://people.math.sfu.ca/~jverner/RKV65.IIIXb.Robust.00010102836.081204.CoeffsOnlyRAT

.seealso: TSRK, TSRKType, TSRKSetType()
M*/
/*MC
     TSRK7VR - Seventh order robust Verner RK scheme with sixth order embedded method.

     This method has ten stages.

     Options database:
.     -ts_rk_type 7vr

     Level: advanced

     References: http://people.math.sfu.ca/~jverner/RKV76.IIa.Robust.000027015646.081206.CoeffsOnlyRAT

.seealso: TSRK, TSRKType, TSRKSetType()
M*/
/*MC
     TSRK8VR - Eigth order robust Verner RK scheme with seventh order embedded method.

     This method has thirteen stages.

     Options database:
.     -ts_rk_type 8vr

     Level: advanced

     References: http://people.math.sfu.ca/~jverner/RKV87.IIa.Robust.00000754677.081208.CoeffsOnlyRATandFLOAT

.seealso: TSRK, TSRKType, TSRKSetType()
M*/

/*@C
  TSRKRegisterAll - Registers all of the Runge-Kutta explicit methods in TSRK

  Not Collective, but should be called by all processes which will need the schemes to be registered

  Level: advanced

.seealso:  TSRKRegisterDestroy()
@*/
PetscErrorCode TSRKRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSRKRegisterAllCalled) PetscFunctionReturn(0);
  TSRKRegisterAllCalled = PETSC_TRUE;

#define RC PetscRealConstant
  {
    const PetscReal
      A[1][1] = {{0}},
      b[1]    = {RC(1.0)};
    ierr = TSRKRegister(TSRK1FE,1,1,&A[0][0],b,NULL,NULL,0,NULL);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[2][2]   = {{0,0},
                   {RC(1.0),0}},
      b[2]      =  {RC(0.5),RC(0.5)},
      bembed[2] =  {RC(1.0),0};
    ierr = TSRKRegister(TSRK2A,2,2,&A[0][0],b,NULL,bembed,0,NULL);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[3][3] = {{0,0,0},
                 {RC(2.0)/RC(3.0),0,0},
                 {RC(-1.0)/RC(3.0),RC(1.0),0}},
      b[3]    =  {RC(0.25),RC(0.5),RC(0.25)};
    ierr = TSRKRegister(TSRK3,3,3,&A[0][0],b,NULL,NULL,0,NULL);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[4][4]   = {{0,0,0,0},
                   {RC(1.0)/RC(2.0),0,0,0},
                   {0,RC(3.0)/RC(4.0),0,0},
                   {RC(2.0)/RC(9.0),RC(1.0)/RC(3.0),RC(4.0)/RC(9.0),0}},
      b[4]      =  {RC(2.0)/RC(9.0),RC(1.0)/RC(3.0),RC(4.0)/RC(9.0),0},
      bembed[4] =  {RC(7.0)/RC(24.0),RC(1.0)/RC(4.0),RC(1.0)/RC(3.0),RC(1.0)/RC(8.0)};
    ierr = TSRKRegister(TSRK3BS,3,4,&A[0][0],b,NULL,bembed,0,NULL);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[4][4] = {{0,0,0,0},
                 {RC(0.5),0,0,0},
                 {0,RC(0.5),0,0},
                 {0,0,RC(1.0),0}},
      b[4]    =  {RC(1.0)/RC(6.0),RC(1.0)/RC(3.0),RC(1.0)/RC(3.0),RC(1.0)/RC(6.0)};
    ierr = TSRKRegister(TSRK4,4,4,&A[0][0],b,NULL,NULL,0,NULL);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[6][6]   = {{0,0,0,0,0,0},
                   {RC(0.25),0,0,0,0,0},
                   {RC(3.0)/RC(32.0),RC(9.0)/RC(32.0),0,0,0,0},
                   {RC(1932.0)/RC(2197.0),RC(-7200.0)/RC(2197.0),RC(7296.0)/RC(2197.0),0,0,0},
                   {RC(439.0)/RC(216.0),RC(-8.0),RC(3680.0)/RC(513.0),RC(-845.0)/RC(4104.0),0,0},
                   {RC(-8.0)/RC(27.0),RC(2.0),RC(-3544.0)/RC(2565.0),RC(1859.0)/RC(4104.0),RC(-11.0)/RC(40.0),0}},
      b[6]      =  {RC(16.0)/RC(135.0),0,RC(6656.0)/RC(12825.0),RC(28561.0)/RC(56430.0),RC(-9.0)/RC(50.0),RC(2.0)/RC(55.0)},
      bembed[6] =  {RC(25.0)/RC(216.0),0,RC(1408.0)/RC(2565.0),RC(2197.0)/RC(4104.0),RC(-1.0)/RC(5.0),0};
    ierr = TSRKRegister(TSRK5F,5,6,&A[0][0],b,NULL,bembed,0,NULL);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[7][7]       = {{0,0,0,0,0,0,0},
                       {RC(1.0)/RC(5.0),0,0,0,0,0,0},
                       {RC(3.0)/RC(40.0),RC(9.0)/RC(40.0),0,0,0,0,0},
                       {RC(44.0)/RC(45.0),RC(-56.0)/RC(15.0),RC(32.0)/RC(9.0),0,0,0,0},
                       {RC(19372.0)/RC(6561.0),RC(-25360.0)/RC(2187.0),RC(64448.0)/RC(6561.0),RC(-212.0)/RC(729.0),0,0,0},
                       {RC(9017.0)/RC(3168.0),RC(-355.0)/RC(33.0),RC(46732.0)/RC(5247.0),RC(49.0)/RC(176.0),RC(-5103.0)/RC(18656.0),0,0},
                       {RC(35.0)/RC(384.0),0,RC(500.0)/RC(1113.0),RC(125.0)/RC(192.0),RC(-2187.0)/RC(6784.0),RC(11.0)/RC(84.0),0}},
      b[7]          =  {RC(35.0)/RC(384.0),0,RC(500.0)/RC(1113.0),RC(125.0)/RC(192.0),RC(-2187.0)/RC(6784.0),RC(11.0)/RC(84.0),0},
      bembed[7]     =  {RC(5179.0)/RC(57600.0),0,RC(7571.0)/RC(16695.0),RC(393.0)/RC(640.0),RC(-92097.0)/RC(339200.0),RC(187.0)/RC(2100.0),RC(1.0)/RC(40.0)},
      binterp[7][5] = {{RC(1.0),RC(-4034104133.0)/RC(1410260304.0),RC(105330401.0)/RC(33982176.0),RC(-13107642775.0)/RC(11282082432.0),RC(6542295.0)/RC(470086768.0)},
                       {0,0,0,0,0},
                       {0,RC(132343189600.0)/RC(32700410799.0),RC(-833316000.0)/RC(131326951.0),RC(91412856700.0)/RC(32700410799.0),RC(-523383600.0)/RC(10900136933.0)},
                       {0,RC(-115792950.0)/RC(29380423.0),RC(185270875.0)/RC(16991088.0),RC(-12653452475.0)/RC(1880347072.0),RC(98134425.0)/RC(235043384.0)},
                       {0,RC(70805911779.0)/RC(24914598704.0),RC(-4531260609.0)/RC(600351776.0),RC(988140236175.0)/RC(199316789632.0),RC(-14307999165.0)/RC(24914598704.0)},
                       {0,RC(-331320693.0)/RC(205662961.0),RC(31361737.0)/RC(7433601.0),RC(-2426908385.0)/RC(822651844.0),RC(97305120.0)/RC(205662961.0)},
                       {0,RC(44764047.0)/RC(29380423.0),RC(-1532549.0)/RC(353981.0),RC(90730570.0)/RC(29380423.0),RC(-8293050.0)/RC(29380423.0)}};
      ierr = TSRKRegister(TSRK5DP,5,7,&A[0][0],b,NULL,bembed,5,binterp[0]);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[8][8]   = {{0,0,0,0,0,0,0,0},
                   {RC(1.0)/RC(6.0),0,0,0,0,0,0,0},
                   {RC(2.0)/RC(27.0),RC(4.0)/RC(27.0),0,0,0,0,0,0},
                   {RC(183.0)/RC(1372.0),RC(-162.0)/RC(343.0),RC(1053.0)/RC(1372.0),0,0,0,0,0},
                   {RC(68.0)/RC(297.0),RC(-4.0)/RC(11.0),RC(42.0)/RC(143.0),RC(1960.0)/RC(3861.0),0,0,0,0},
                   {RC(597.0)/RC(22528.0),RC(81.0)/RC(352.0),RC(63099.0)/RC(585728.0),RC(58653.0)/RC(366080.0),RC(4617.0)/RC(20480.0),0,0,0},
                   {RC(174197.0)/RC(959244.0),RC(-30942.0)/RC(79937.0),RC(8152137.0)/RC(19744439.0),RC(666106.0)/RC(1039181.0),RC(-29421.0)/RC(29068.0),RC(482048.0)/RC(414219.0),0,0},
                   {RC(587.0)/RC(8064.0),0,RC(4440339.0)/RC(15491840.0),RC(24353.0)/RC(124800.0),RC(387.0)/RC(44800.0),RC(2152.0)/RC(5985.0),RC(7267.0)/RC(94080.0),0}},
      b[8]      =  {RC(587.0)/RC(8064.0),0,RC(4440339.0)/RC(15491840.0),RC(24353.0)/RC(124800.0),RC(387.0)/RC(44800.0),RC(2152.0)/RC(5985.0),RC(7267.0)/RC(94080.0),0},
      bembed[8] =  {RC(2479.0)/RC(34992.0),0,RC(123.0)/RC(416.0),RC(612941.0)/RC(3411720.0),RC(43.0)/RC(1440.0),RC(2272.0)/RC(6561.0),RC(79937.0)/RC(1113912.0),RC(3293.0)/RC(556956.0)};
    ierr = TSRKRegister(TSRK5BS,5,8,&A[0][0],b,NULL,bembed,0,NULL);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[9][9]   = {{0,0,0,0,0,0,0,0,0},
                   {RC(1.8000000000000000000000000000000000000000e-01),0,0,0,0,0,0,0,0},
                   {RC(8.9506172839506172839506172839506172839506e-02),RC(7.7160493827160493827160493827160493827160e-02),0,0,0,0,0,0,0},
                   {RC(6.2500000000000000000000000000000000000000e-02),0,RC(1.8750000000000000000000000000000000000000e-01),0,0,0,0,0,0},
                   {RC(3.1651600000000000000000000000000000000000e-01),0,RC(-1.0449480000000000000000000000000000000000e+00),RC(1.2584320000000000000000000000000000000000e+00),0,0,0,0,0},
                   {RC(2.7232612736485626257225065566674305502508e-01),0,RC(-8.2513360323886639676113360323886639676113e-01),RC(1.0480917678812415654520917678812415654521e+00),RC(1.0471570799276856873679117969088177628396e-01),0,0,0,0},
                   {RC(-1.6699418599716514314329607278961797333198e-01),0,RC(6.3170850202429149797570850202429149797571e-01),RC(1.7461044552773876082146758838488161796432e-01),RC(-1.0665356459086066122525194734018680677781e+00),RC(1.2272108843537414965986394557823129251701e+00),0,0,0},
                   {RC(3.6423751686909581646423751686909581646424e-01),0,RC(-2.0404858299595141700404858299595141700405e-01),RC(-3.4883737816068643136312309244640071707741e-01),RC(3.2619323032856867443333608747142581729048e+00),RC(-2.7551020408163265306122448979591836734694e+00),RC(6.8181818181818181818181818181818181818182e-01),0,0},
                   {RC(7.6388888888888888888888888888888888888889e-02),0,0,RC(3.6940836940836940836940836940836940836941e-01),0,RC(2.4801587301587301587301587301587301587302e-01),RC(2.3674242424242424242424242424242424242424e-01),RC(6.9444444444444444444444444444444444444444e-02),0}},
      b[9]      =  {RC(7.6388888888888888888888888888888888888889e-02),0,0,RC(3.6940836940836940836940836940836940836941e-01),0,RC(2.4801587301587301587301587301587301587302e-01),RC(2.3674242424242424242424242424242424242424e-01),RC(6.9444444444444444444444444444444444444444e-02),0},
      bembed[9] =  {RC(5.8700209643605870020964360587002096436059e-02),0,0,RC(4.8072562358276643990929705215419501133787e-01),RC(-8.5341242076919085578832094861228313083563e-01),RC(1.2046485260770975056689342403628117913832e+00),0,RC(-5.9242373072160306202859394348756050883710e-02),RC(1.6858043453788134639198468985703028256220e-01)};
    ierr = TSRKRegister(TSRK6VR,6,9,&A[0][0],b,NULL,bembed,0,NULL);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[10][10]  = {{0,0,0,0,0,0,0,0,0,0},
                    {RC(5.0000000000000000000000000000000000000000e-03),0,0,0,0,0,0,0,0,0},
                    {RC(-1.0767901234567901234567901234567901234568e+00),RC(1.1856790123456790123456790123456790123457e+00),0,0,0,0,0,0,0,0},
                    {RC(4.0833333333333333333333333333333333333333e-02),0,RC(1.2250000000000000000000000000000000000000e-01),0,0,0,0,0,0,0},
                    {RC(6.3607142857142857142857142857142857142857e-01),0,RC(-2.4444642857142857142857142857142857142857e+00),RC(2.2633928571428571428571428571428571428571e+00),0,0,0,0,0,0},
                    {RC(-2.5351211079349245229256383554660215487207e+00),0,RC(1.0299374654449267920438514460756024913612e+01),RC(-7.9513032885990579949493217458266876536482e+00),RC(7.9301148923100592201226014271115261823800e-01),0,0,0,0,0},
                    {RC(1.0018765812524632961969196583094999808207e+00),0,RC(-4.1665712824423798331313938005470971453189e+00),RC(3.8343432929128642412552665218251378665197e+00),RC(-5.0233333560710847547464330228611765612403e-01),RC(6.6768474388416077115385092269857695410259e-01),0,0,0,0},
                    {RC(2.7255018354630767130333963819175005717348e+01),0,RC(-4.2004617278410638355318645443909295369611e+01),RC(-1.0535713126619489917921081600546526103722e+01),RC(8.0495536711411937147983652158926826634202e+01),RC(-6.7343882271790513468549075963212975640927e+01),RC(1.3048657610777937463471187029566964762710e+01),0,0,0},
                    {RC(-3.0397378057114965146943658658755763226883e+00),0,RC(1.0138161410329801111857946190709700150441e+01),RC(-6.4293056748647215721462825629555298064437e+00),RC(-1.5864371483408276587115312853798610579467e+00),RC(1.8921781841968424410864308909131353365021e+00),RC(1.9699335407608869061292360163336442838006e-02),RC(5.4416989827933235465102724247952572977903e-03),0,0},
                    {RC(-1.4449518916777735137351003179355712360517e+00),0,RC(8.0318913859955919224117033223019560435041e+00),RC(-7.5831741663401346820798883023671588604984e+00),RC(3.5816169353190074211247685442452878696855e+00),RC(-2.4369722632199529411183809065693752383733e+00),RC(8.5158999992326179339689766032486142173390e-01),0,0,0}},
      b[10]      =  {RC(4.7425837833706756083569172717574534698932e-02),0,0,RC(2.5622361659370562659961727458274623448160e-01),RC(2.6951376833074206619473817258075952886764e-01),RC(1.2686622409092782845989138364739173247882e-01),RC(2.4887225942060071622046449427647492767292e-01),RC(3.0744837408200631335304388479099184768645e-03),RC(4.8023809989496943308189063347143123323209e-02),0},
      bembed[10] =  {RC(4.7485247699299631037531273805727961552268e-02),0,0,RC(2.5599412588690633297154918245905393870497e-01),RC(2.7058478081067688722530891099268135732387e-01),RC(1.2505618684425992913638822323746917920448e-01),RC(2.5204468723743860507184043820197442562182e-01),0,0,RC(4.8834971521418614557381971303093137592592e-02)};
    ierr = TSRKRegister(TSRK7VR,7,10,&A[0][0],b,NULL,bembed,0,NULL);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[13][13]  = {{0,0,0,0,0,0,0,0,0,0,0,0,0},
                    {RC(2.5000000000000000000000000000000000000000e-01),0,0,0,0,0,0,0,0,0,0,0,0},
                    {RC(8.7400846504915232052686327594877411977046e-02),RC(2.5487604938654321753087950620345685135815e-02),0,0,0,0,0,0,0,0,0,0,0},
                    {RC(4.2333169291338582677165354330708661417323e-02),0,RC(1.2699950787401574803149606299212598425197e-01),0,0,0,0,0,0,0,0,0,0},
                    {RC(4.2609505888742261494881445237572274090942e-01),0,RC(-1.5987952846591523265427733230657181117089e+00),RC(1.5967002257717297115939588706899953707994e+00),0,0,0,0,0,0,0,0,0},
                    {RC(5.0719337296713929515090618138513639239329e-02),0,0,RC(2.5433377264600407582754714408877778031369e-01),RC(2.0394689005728199465736223777270858044698e-01),0,0,0,0,0,0,0,0},
                    {RC(-2.9000374717523110970388379285425896124091e-01),0,0,RC(1.3441873910260789889438681109414337003184e+00),RC(-2.8647779433614427309611103827036562829470e+00),RC(2.6775942995105948517211260646164815438695e+00),0,0,0,0,0,0,0},
                    {RC(9.8535011337993546469740402980727014284756e-02),0,0,0,RC(2.2192680630751384842024036498197387903583e-01),RC(-1.8140622911806994312690338288073952457474e-01),RC(1.0944411472562548236922614918038631254153e-02),0,0,0,0,0,0},
                    {RC(3.8711052545731144679444618165166373405645e-01),0,0,RC(-1.4424454974855277571256745553077927767173e+00),RC(2.9053981890699509317691346449233848441744e+00),RC(-1.8537710696301059290843332675811978025183e+00),RC(1.4003648098728154269497325109771241479223e-01),RC(5.7273940811495816575746774624447706488753e-01),0,0,0,0,0},
                    {RC(-1.6124403444439308100630016197913480595436e-01),0,0,RC(-1.7339602957358984083578404473962567894901e-01),RC(-1.3012892814065147406016812745172492529744e+00),RC(1.1379503751738617308558792131431003472124e+00),RC(-3.1747649663966880106923521138043024698980e-02),RC(9.3351293824933666439811064486056884856590e-01),RC(-8.3786318334733852703300855629616433201504e-02),0,0,0,0},
                    {RC(-1.9199444881589533281510804651483576073142e-02),0,0,RC(2.7330857265264284907942326254016124275617e-01),RC(-6.7534973206944372919691611210942380856240e-01),RC(3.4151849813846016071738489974728382711981e-01),RC(-6.7950064803375772478920516198524629391910e-02),RC(9.6591752247623878884265586491216376509746e-02),RC(1.3253082511182101180721038466545389951226e-01),RC(3.6854959360386113446906329951531666812946e-01),0,0,0},
                    {RC(6.0918774036452898676888412111588817784584e-01),0,0,RC(-2.2725690858980016768999800931413088399719e+00),RC(4.7578983426940290068155255881914785497547e+00),RC(-5.5161067066927584824294689667844248244842e+00),RC(2.9005963696801192709095818565946174378180e-01),RC(5.6914239633590368229109858454801849145630e-01),RC(7.9267957603321670271339916205893327579951e-01),RC(1.5473720453288822894126190771849898232047e-01),RC(1.6149708956621816247083215106334544434974e+00),0,0},
                    {RC(8.8735762208534719663211694051981022704884e-01),0,0,RC(-2.9754597821085367558513632804709301581977e+00),RC(5.6007170094881630597990392548350098923829e+00),RC(-5.9156074505366744680014930189941657351840e+00),RC(2.2029689156134927016879142540807638331238e-01),RC(1.0155097824462216666143271340902996997549e-01),RC(1.1514345647386055909780397752125850553556e+00),RC(1.9297101665271239396134361900805843653065e+00),0,0,0}},
      b[13]      =  {RC(4.4729564666695714203015840429049382466467e-02),0,0,0,0,RC(1.5691033527708199813368698010726645409175e-01),RC(1.8460973408151637740702451873526277892035e-01),RC(2.2516380602086991042479419400350721970920e-01),RC(1.4794615651970234687005179885449141753736e-01),RC(7.6055542444955825269798361910336491012732e-02),RC(1.2277290235018619610824346315921437388535e-01),RC(4.1811958638991631583384842800871882376786e-02),0},
      bembed[13] =  {RC(4.5847111400495925878664730122010282095875e-02),0,0,0,0,RC(2.6231891404152387437443356584845803392392e-01),RC(1.9169372337852611904485738635688429008025e-01),RC(2.1709172327902618330978407422906448568196e-01),RC(1.2738189624833706796803169450656737867900e-01),RC(1.1510530385365326258240515750043192148894e-01),0,0,RC(4.0561327798437566841823391436583608050053e-02)};
    ierr = TSRKRegister(TSRK8VR,8,13,&A[0][0],b,NULL,bembed,0,NULL);CHKERRQ(ierr);
  }
#undef RC
  PetscFunctionReturn(0);
}

/*@C
   TSRKRegisterDestroy - Frees the list of schemes that were registered by TSRKRegister().

   Not Collective

   Level: advanced

.seealso: TSRKRegister(), TSRKRegisterAll()
@*/
PetscErrorCode TSRKRegisterDestroy(void)
{
  PetscErrorCode ierr;
  RKTableauLink  link;

  PetscFunctionBegin;
  while ((link = RKTableauList)) {
    RKTableau t = &link->tab;
    RKTableauList = link->next;
    ierr = PetscFree3(t->A,t->b,t->c);CHKERRQ(ierr);
    ierr = PetscFree(t->bembed);CHKERRQ(ierr);
    ierr = PetscFree(t->binterp);CHKERRQ(ierr);
    ierr = PetscFree(t->name);CHKERRQ(ierr);
    ierr = PetscFree(link);CHKERRQ(ierr);
  }
  TSRKRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  TSRKInitializePackage - This function initializes everything in the TSRK package. It is called
  from TSInitializePackage().

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode TSRKInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSRKPackageInitialized) PetscFunctionReturn(0);
  TSRKPackageInitialized = PETSC_TRUE;
  ierr = TSRKRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(TSRKFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSRKFinalizePackage - This function destroys everything in the TSRK package. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode TSRKFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TSRKPackageInitialized = PETSC_FALSE;
  ierr = TSRKRegisterDestroy();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSRKRegister - register an RK scheme by providing the entries in the Butcher tableau and optionally embedded approximations and interpolation

   Not Collective, but the same schemes should be registered on all processes on which they will be used

   Input Parameters:
+  name - identifier for method
.  order - approximation order of method
.  s - number of stages, this is the dimension of the matrices below
.  A - stage coefficients (dimension s*s, row-major)
.  b - step completion table (dimension s; NULL to use last row of A)
.  c - abscissa (dimension s; NULL to use row sums of A)
.  bembed - completion table for embedded method (dimension s; NULL if not available)
.  p - Order of the interpolation scheme, equal to the number of columns of binterp
-  binterp - Coefficients of the interpolation formula (dimension s*p; NULL to reuse b with p=1)

   Notes:
   Several RK methods are provided, this function is only needed to create new methods.

   Level: advanced

.seealso: TSRK
@*/
PetscErrorCode TSRKRegister(TSRKType name,PetscInt order,PetscInt s,
                            const PetscReal A[],const PetscReal b[],const PetscReal c[],
                            const PetscReal bembed[],PetscInt p,const PetscReal binterp[])
{
  PetscErrorCode  ierr;
  RKTableauLink   link;
  RKTableau       t;
  PetscInt        i,j;

  PetscFunctionBegin;
  PetscValidCharPointer(name,1);
  PetscValidRealPointer(A,4);
  if (b) PetscValidRealPointer(b,5);
  if (c) PetscValidRealPointer(c,6);
  if (bembed) PetscValidRealPointer(bembed,7);
  if (binterp || p > 1) PetscValidRealPointer(binterp,9);

  ierr = TSRKInitializePackage();CHKERRQ(ierr);
  ierr = PetscNew(&link);CHKERRQ(ierr);
  t = &link->tab;

  ierr = PetscStrallocpy(name,&t->name);CHKERRQ(ierr);
  t->order = order;
  t->s = s;
  ierr = PetscMalloc3(s*s,&t->A,s,&t->b,s,&t->c);CHKERRQ(ierr);
  ierr = PetscArraycpy(t->A,A,s*s);CHKERRQ(ierr);
  if (b)  { ierr = PetscArraycpy(t->b,b,s);CHKERRQ(ierr); }
  else for (i=0; i<s; i++) t->b[i] = A[(s-1)*s+i];
  if (c)  { ierr = PetscArraycpy(t->c,c,s);CHKERRQ(ierr); }
  else for (i=0; i<s; i++) for (j=0,t->c[i]=0; j<s; j++) t->c[i] += A[i*s+j];
  t->FSAL = PETSC_TRUE;
  for (i=0; i<s; i++) if (t->A[(s-1)*s+i] != t->b[i]) t->FSAL = PETSC_FALSE;

  if (bembed) {
    ierr = PetscMalloc1(s,&t->bembed);CHKERRQ(ierr);
    ierr = PetscArraycpy(t->bembed,bembed,s);CHKERRQ(ierr);
  }

  if (!binterp) { p = 1; binterp = t->b; }
  t->p = p;
  ierr = PetscMalloc1(s*p,&t->binterp);CHKERRQ(ierr);
  ierr = PetscArraycpy(t->binterp,binterp,s*p);CHKERRQ(ierr);

  link->next = RKTableauList;
  RKTableauList = link;
  PetscFunctionReturn(0);
}

PetscErrorCode TSRKGetTableau_RK(TS ts, PetscInt *s, const PetscReal **A, const PetscReal **b, const PetscReal **c, const PetscReal **bembed,
                                        PetscInt *p, const PetscReal **binterp, PetscBool *FSAL)
{
  TS_RK     *rk   = (TS_RK*)ts->data;
  RKTableau tab  = rk->tableau;

  PetscFunctionBegin;
  if (s) *s = tab->s;
  if (A) *A = tab->A;
  if (b) *b = tab->b;
  if (c) *c = tab->c;
  if (bembed) *bembed = tab->bembed;
  if (p) *p = tab->p;
  if (binterp) *binterp = tab->binterp;
  if (FSAL) *FSAL = tab->FSAL;
  PetscFunctionReturn(0);
}

/*@C
   TSRKGetTableau - Get info on the RK tableau

   Not Collective

   Input Parameter:
.  ts - timestepping context

   Output Parameters:
+  s - number of stages, this is the dimension of the matrices below
.  A - stage coefficients (dimension s*s, row-major)
.  b - step completion table (dimension s)
.  c - abscissa (dimension s)
.  bembed - completion table for embedded method (dimension s; NULL if not available)
.  p - Order of the interpolation scheme, equal to the number of columns of binterp
.  binterp - Coefficients of the interpolation formula (dimension s*p)
-  FSAL - wheather or not the scheme has the First Same As Last property

   Level: developer

.seealso: TSRK
@*/
PetscErrorCode TSRKGetTableau(TS ts, PetscInt *s, const PetscReal **A, const PetscReal **b, const PetscReal **c, const PetscReal **bembed,
                                     PetscInt *p, const PetscReal **binterp, PetscBool *FSAL)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscUseMethod(ts,"TSRKGetTableau_C",(TS,PetscInt*,const PetscReal**,const PetscReal**,const PetscReal**,const PetscReal**,
                                                  PetscInt*,const PetscReal**,PetscBool*),(ts,s,A,b,c,bembed,p,binterp,FSAL));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 This is for single-step RK method
 The step completion formula is

 x1 = x0 + h b^T YdotRHS

 This function can be called before or after ts->vec_sol has been updated.
 Suppose we have a completion formula (b) and an embedded formula (be) of different order.
 We can write

 x1e = x0 + h be^T YdotRHS
     = x1 - h b^T YdotRHS + h be^T YdotRHS
     = x1 + h (be - b)^T YdotRHS

 so we can evaluate the method with different order even after the step has been optimistically completed.
*/
static PetscErrorCode TSEvaluateStep_RK(TS ts,PetscInt order,Vec X,PetscBool *done)
{
  TS_RK          *rk   = (TS_RK*)ts->data;
  RKTableau      tab  = rk->tableau;
  PetscScalar    *w    = rk->work;
  PetscReal      h;
  PetscInt       s    = tab->s,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (rk->status) {
  case TS_STEP_INCOMPLETE:
  case TS_STEP_PENDING:
    h = ts->time_step; break;
  case TS_STEP_COMPLETE:
    h = ts->ptime - ts->ptime_prev; break;
  default: SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Invalid TSStepStatus");
  }
  if (order == tab->order) {
    if (rk->status == TS_STEP_INCOMPLETE) {
      ierr = VecCopy(ts->vec_sol,X);CHKERRQ(ierr);
      for (j=0; j<s; j++) w[j] = h*tab->b[j]/rk->dtratio;
      ierr = VecMAXPY(X,s,w,rk->YdotRHS);CHKERRQ(ierr);
    } else {ierr = VecCopy(ts->vec_sol,X);CHKERRQ(ierr);}
    PetscFunctionReturn(0);
  } else if (order == tab->order-1) {
    if (!tab->bembed) goto unavailable;
    if (rk->status == TS_STEP_INCOMPLETE) { /*Complete with the embedded method (be)*/
      ierr = VecCopy(ts->vec_sol,X);CHKERRQ(ierr);
      for (j=0; j<s; j++) w[j] = h*tab->bembed[j];
      ierr = VecMAXPY(X,s,w,rk->YdotRHS);CHKERRQ(ierr);
    } else {  /*Rollback and re-complete using (be-b) */
      ierr = VecCopy(ts->vec_sol,X);CHKERRQ(ierr);
      for (j=0; j<s; j++) w[j] = h*(tab->bembed[j] - tab->b[j]);
      ierr = VecMAXPY(X,s,w,rk->YdotRHS);CHKERRQ(ierr);
    }
    if (done) *done = PETSC_TRUE;
    PetscFunctionReturn(0);
  }
unavailable:
  if (done) *done = PETSC_FALSE;
  else SETERRQ3(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"RK '%s' of order %D cannot evaluate step at order %D. Consider using -ts_adapt_type none or a different method that has an embedded estimate.",tab->name,tab->order,order);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSForwardCostIntegral_RK(TS ts)
{
  TS_RK           *rk = (TS_RK*)ts->data;
  TS              quadts = ts->quadraturets;
  RKTableau       tab = rk->tableau;
  const PetscInt  s = tab->s;
  const PetscReal *b = tab->b,*c = tab->c;
  Vec             *Y = rk->Y;
  PetscInt        i;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* No need to backup quadts->vec_sol since it can be reverted in TSRollBack_RK */
  for (i=s-1; i>=0; i--) {
    /* Evolve quadrature TS solution to compute integrals */
    ierr = TSComputeRHSFunction(quadts,rk->ptime+rk->time_step*c[i],Y[i],ts->vec_costintegrand);CHKERRQ(ierr);
    ierr = VecAXPY(quadts->vec_sol,rk->time_step*b[i],ts->vec_costintegrand);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdjointCostIntegral_RK(TS ts)
{
  TS_RK           *rk = (TS_RK*)ts->data;
  RKTableau       tab = rk->tableau;
  TS              quadts = ts->quadraturets;
  const PetscInt  s = tab->s;
  const PetscReal *b = tab->b,*c = tab->c;
  Vec             *Y = rk->Y;
  PetscInt        i;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  for (i=s-1; i>=0; i--) {
    /* Evolve quadrature TS solution to compute integrals */
    ierr = TSComputeRHSFunction(quadts,ts->ptime+ts->time_step*(1.0-c[i]),Y[i],ts->vec_costintegrand);CHKERRQ(ierr);
    ierr = VecAXPY(quadts->vec_sol,-ts->time_step*b[i],ts->vec_costintegrand);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSRollBack_RK(TS ts)
{
  TS_RK           *rk = (TS_RK*)ts->data;
  TS              quadts = ts->quadraturets;
  RKTableau       tab = rk->tableau;
  const PetscInt  s  = tab->s;
  const PetscReal *b = tab->b,*c = tab->c;
  PetscScalar     *w = rk->work;
  Vec             *Y = rk->Y,*YdotRHS = rk->YdotRHS;
  PetscInt        j;
  PetscReal       h;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  switch (rk->status) {
  case TS_STEP_INCOMPLETE:
  case TS_STEP_PENDING:
    h = ts->time_step; break;
  case TS_STEP_COMPLETE:
    h = ts->ptime - ts->ptime_prev; break;
  default: SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Invalid TSStepStatus");
  }
  for (j=0; j<s; j++) w[j] = -h*b[j];
  ierr = VecMAXPY(ts->vec_sol,s,w,YdotRHS);CHKERRQ(ierr);
  if (quadts && ts->costintegralfwd) {
    for (j=0; j<s; j++) {
      /* Revert the quadrature TS solution */
      ierr = TSComputeRHSFunction(quadts,rk->ptime+h*c[j],Y[j],ts->vec_costintegrand);CHKERRQ(ierr);
      ierr = VecAXPY(quadts->vec_sol,-h*b[j],ts->vec_costintegrand);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSForwardStep_RK(TS ts)
{
  TS_RK           *rk = (TS_RK*)ts->data;
  RKTableau       tab = rk->tableau;
  Mat             J,*MatsFwdSensipTemp = rk->MatsFwdSensipTemp;
  const PetscInt  s = tab->s;
  const PetscReal *A = tab->A,*c = tab->c,*b = tab->b;
  Vec             *Y = rk->Y;
  PetscInt        i,j;
  PetscReal       stage_time,h = ts->time_step;
  PetscBool       zero;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatCopy(ts->mat_sensip,rk->MatFwdSensip0,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = TSGetRHSJacobian(ts,&J,NULL,NULL,NULL);CHKERRQ(ierr);

  for (i=0; i<s; i++) {
    stage_time = ts->ptime + h*c[i];
    zero = PETSC_FALSE;
    if (b[i] == 0 && i == s-1) zero = PETSC_TRUE;
    /* TLM Stage values */
    if (!i) {
      ierr = MatCopy(ts->mat_sensip,rk->MatsFwdStageSensip[i],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    } else if (!zero) {
      ierr = MatZeroEntries(rk->MatsFwdStageSensip[i]);CHKERRQ(ierr);
      for (j=0; j<i; j++) {
        ierr = MatAXPY(rk->MatsFwdStageSensip[i],h*A[i*s+j],MatsFwdSensipTemp[j],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      }
      ierr = MatAXPY(rk->MatsFwdStageSensip[i],1.,ts->mat_sensip,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    } else {
      ierr = MatZeroEntries(rk->MatsFwdStageSensip[i]);CHKERRQ(ierr);
    }

    ierr = TSComputeRHSJacobian(ts,stage_time,Y[i],J,J);CHKERRQ(ierr);
    ierr = MatMatMult(J,rk->MatsFwdStageSensip[i],MAT_REUSE_MATRIX,PETSC_DEFAULT,&MatsFwdSensipTemp[i]);CHKERRQ(ierr);
    if (ts->Jacprhs) {
      ierr = TSComputeRHSJacobianP(ts,stage_time,Y[i],ts->Jacprhs);CHKERRQ(ierr); /* get f_p */
      if (ts->vecs_sensi2p) { /* TLM used for 2nd-order adjoint */
        PetscScalar *xarr;
        ierr = MatDenseGetColumn(MatsFwdSensipTemp[i],0,&xarr);CHKERRQ(ierr);
        ierr = VecPlaceArray(rk->VecDeltaFwdSensipCol,xarr);CHKERRQ(ierr);
        ierr = MatMultAdd(ts->Jacprhs,ts->vec_dir,rk->VecDeltaFwdSensipCol,rk->VecDeltaFwdSensipCol);CHKERRQ(ierr);
        ierr = VecResetArray(rk->VecDeltaFwdSensipCol);CHKERRQ(ierr);
        ierr = MatDenseRestoreColumn(MatsFwdSensipTemp[i],&xarr);CHKERRQ(ierr);
      } else {
        ierr = MatAXPY(MatsFwdSensipTemp[i],1.,ts->Jacprhs,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
      }
    }
  }

  for (i=0; i<s; i++) {
    ierr = MatAXPY(ts->mat_sensip,h*b[i],rk->MatsFwdSensipTemp[i],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  rk->status = TS_STEP_COMPLETE;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSForwardGetStages_RK(TS ts,PetscInt *ns,Mat **stagesensip)
{
  TS_RK     *rk = (TS_RK*)ts->data;
  RKTableau tab  = rk->tableau;

  PetscFunctionBegin;
  if (ns) *ns = tab->s;
  if (stagesensip) *stagesensip = rk->MatsFwdStageSensip;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSForwardSetUp_RK(TS ts)
{
  TS_RK          *rk = (TS_RK*)ts->data;
  RKTableau      tab  = rk->tableau;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* backup sensitivity results for roll-backs */
  ierr = MatDuplicate(ts->mat_sensip,MAT_DO_NOT_COPY_VALUES,&rk->MatFwdSensip0);CHKERRQ(ierr);

  ierr = PetscMalloc1(tab->s,&rk->MatsFwdStageSensip);CHKERRQ(ierr);
  ierr = PetscMalloc1(tab->s,&rk->MatsFwdSensipTemp);CHKERRQ(ierr);
  for (i=0; i<tab->s; i++) {
    ierr = MatDuplicate(ts->mat_sensip,MAT_DO_NOT_COPY_VALUES,&rk->MatsFwdStageSensip[i]);CHKERRQ(ierr);
    ierr = MatDuplicate(ts->mat_sensip,MAT_DO_NOT_COPY_VALUES,&rk->MatsFwdSensipTemp[i]);CHKERRQ(ierr);
  }
  ierr = VecDuplicate(ts->vec_sol,&rk->VecDeltaFwdSensipCol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSForwardReset_RK(TS ts)
{
  TS_RK          *rk = (TS_RK*)ts->data;
  RKTableau      tab  = rk->tableau;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&rk->MatFwdSensip0);CHKERRQ(ierr);
  if (rk->MatsFwdStageSensip) {
    for (i=0; i<tab->s; i++) {
      ierr = MatDestroy(&rk->MatsFwdStageSensip[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(rk->MatsFwdStageSensip);CHKERRQ(ierr);
  }
  if (rk->MatsFwdSensipTemp) {
    for (i=0; i<tab->s; i++) {
      ierr = MatDestroy(&rk->MatsFwdSensipTemp[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(rk->MatsFwdSensipTemp);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&rk->VecDeltaFwdSensipCol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSStep_RK(TS ts)
{
  TS_RK           *rk  = (TS_RK*)ts->data;
  RKTableau       tab  = rk->tableau;
  const PetscInt  s = tab->s;
  const PetscReal *A = tab->A,*c = tab->c;
  PetscScalar     *w = rk->work;
  Vec             *Y = rk->Y,*YdotRHS = rk->YdotRHS;
  PetscBool       FSAL = tab->FSAL;
  TSAdapt         adapt;
  PetscInt        i,j;
  PetscInt        rejections = 0;
  PetscBool       stageok,accept = PETSC_TRUE;
  PetscReal       next_time_step = ts->time_step;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (ts->steprollback || ts->steprestart) FSAL = PETSC_FALSE;
  if (FSAL) { ierr = VecCopy(YdotRHS[s-1],YdotRHS[0]);CHKERRQ(ierr); }

  rk->status = TS_STEP_INCOMPLETE;
  while (!ts->reason && rk->status != TS_STEP_COMPLETE) {
    PetscReal t = ts->ptime;
    PetscReal h = ts->time_step;
    for (i=0; i<s; i++) {
      rk->stage_time = t + h*c[i];
      ierr = TSPreStage(ts,rk->stage_time);CHKERRQ(ierr);
      ierr = VecCopy(ts->vec_sol,Y[i]);CHKERRQ(ierr);
      for (j=0; j<i; j++) w[j] = h*A[i*s+j];
      ierr = VecMAXPY(Y[i],i,w,YdotRHS);CHKERRQ(ierr);
      ierr = TSPostStage(ts,rk->stage_time,i,Y);CHKERRQ(ierr);
      ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
      ierr = TSAdaptCheckStage(adapt,ts,rk->stage_time,Y[i],&stageok);CHKERRQ(ierr);
      if (!stageok) goto reject_step;
      if (FSAL && !i) continue;
      ierr = TSComputeRHSFunction(ts,t+h*c[i],Y[i],YdotRHS[i]);CHKERRQ(ierr);
    }

    rk->status = TS_STEP_INCOMPLETE;
    ierr = TSEvaluateStep(ts,tab->order,ts->vec_sol,NULL);CHKERRQ(ierr);
    rk->status = TS_STEP_PENDING;
    ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
    ierr = TSAdaptCandidatesClear(adapt);CHKERRQ(ierr);
    ierr = TSAdaptCandidateAdd(adapt,tab->name,tab->order,1,tab->ccfl,(PetscReal)tab->s,PETSC_TRUE);CHKERRQ(ierr);
    ierr = TSAdaptChoose(adapt,ts,ts->time_step,NULL,&next_time_step,&accept);CHKERRQ(ierr);
    rk->status = accept ? TS_STEP_COMPLETE : TS_STEP_INCOMPLETE;
    if (!accept) { /* Roll back the current step */
      ierr = TSRollBack_RK(ts);CHKERRQ(ierr);
      ts->time_step = next_time_step;
      goto reject_step;
    }

    if (ts->costintegralfwd) { /* Save the info for the later use in cost integral evaluation */
      rk->ptime     = ts->ptime;
      rk->time_step = ts->time_step;
    }

    ts->ptime += ts->time_step;
    ts->time_step = next_time_step;
    break;

    reject_step:
    ts->reject++; accept = PETSC_FALSE;
    if (!ts->reason && ++rejections > ts->max_reject && ts->max_reject >= 0) {
      ts->reason = TS_DIVERGED_STEP_REJECTED;
      ierr = PetscInfo2(ts,"Step=%D, step rejections %D greater than current TS allowed, stopping solve\n",ts->steps,rejections);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdjointSetUp_RK(TS ts)
{
  TS_RK          *rk  = (TS_RK*)ts->data;
  RKTableau      tab = rk->tableau;
  PetscInt       s   = tab->s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ts->adjointsetupcalled++) PetscFunctionReturn(0);
  ierr = VecDuplicateVecs(ts->vecs_sensi[0],s*ts->numcost,&rk->VecsDeltaLam);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(ts->vecs_sensi[0],ts->numcost,&rk->VecsSensiTemp);CHKERRQ(ierr);
  if (ts->vecs_sensip) {
    ierr = VecDuplicate(ts->vecs_sensip[0],&rk->VecDeltaMu);CHKERRQ(ierr);
  }
  if (ts->vecs_sensi2) {
    ierr = VecDuplicateVecs(ts->vecs_sensi[0],s*ts->numcost,&rk->VecsDeltaLam2);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(ts->vecs_sensi2[0],ts->numcost,&rk->VecsSensi2Temp);CHKERRQ(ierr);
  }
  if (ts->vecs_sensi2p) {
    ierr = VecDuplicate(ts->vecs_sensi2p[0],&rk->VecDeltaMu2);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  Assumptions:
    - TSStep_RK() always evaluates the step with b, not bembed.
*/
static PetscErrorCode TSAdjointStep_RK(TS ts)
{
  TS_RK            *rk = (TS_RK*)ts->data;
  TS               quadts = ts->quadraturets;
  RKTableau        tab = rk->tableau;
  Mat              J,Jpre,Jquad;
  const PetscInt   s = tab->s;
  const PetscReal  *A = tab->A,*b = tab->b,*c = tab->c;
  PetscScalar      *w = rk->work,*xarr;
  Vec              *Y = rk->Y,*VecsDeltaLam = rk->VecsDeltaLam,VecDeltaMu = rk->VecDeltaMu,*VecsSensiTemp = rk->VecsSensiTemp;
  Vec              *VecsDeltaLam2 = rk->VecsDeltaLam2,VecDeltaMu2 = rk->VecDeltaMu2,*VecsSensi2Temp = rk->VecsSensi2Temp;
  Vec              VecDRDUTransCol = ts->vec_drdu_col,VecDRDPTransCol = ts->vec_drdp_col;
  PetscInt         i,j,nadj;
  PetscReal        t = ts->ptime;
  PetscReal        h = ts->time_step;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  rk->status = TS_STEP_INCOMPLETE;

  ierr = TSGetRHSJacobian(ts,&J,&Jpre,NULL,NULL);CHKERRQ(ierr);
  if (quadts) {
    ierr = TSGetRHSJacobian(quadts,&Jquad,NULL,NULL,NULL);CHKERRQ(ierr);
  }
  for (i=s-1; i>=0; i--) {
    if (tab->FSAL && i == s-1) {
      /* VecsDeltaLam[nadj*s+s-1] are initialized with zeros and the values never change.*/
      continue;
    }
    rk->stage_time = t + h*(1.0-c[i]);
    ierr = TSComputeSNESJacobian(ts,Y[i],J,Jpre);CHKERRQ(ierr);
    if (quadts) {
      ierr = TSComputeRHSJacobian(quadts,rk->stage_time,Y[i],Jquad,Jquad);CHKERRQ(ierr); /* get r_u^T */
    }
    if (ts->vecs_sensip) {
      ierr = TSComputeRHSJacobianP(ts,rk->stage_time,Y[i],ts->Jacprhs);CHKERRQ(ierr); /* get f_p */
      if (quadts) {
        ierr = TSComputeRHSJacobianP(quadts,rk->stage_time,Y[i],quadts->Jacprhs);CHKERRQ(ierr); /* get f_p for the quadrature */
      }
    }

    if (b[i]) {
      for (j=i+1; j<s; j++) w[j-i-1] = A[j*s+i]/b[i]; /* coefficients for computing VecsSensiTemp */
    } else {
      for (j=i+1; j<s; j++) w[j-i-1] = A[j*s+i]; /* coefficients for computing VecsSensiTemp */
    }

    for (nadj=0; nadj<ts->numcost; nadj++) {
      /* Stage values of lambda */
      if (b[i]) {
        /* lambda_{n+1} + \sum_{j=i+1}^s a_{ji}/b[i]*lambda_{s,j} */
        ierr = VecCopy(ts->vecs_sensi[nadj],VecsSensiTemp[nadj]);CHKERRQ(ierr); /* VecDeltaLam is an vec array of size s by numcost */
        ierr = VecMAXPY(VecsSensiTemp[nadj],s-i-1,w,&VecsDeltaLam[nadj*s+i+1]);CHKERRQ(ierr);
        ierr = MatMultTranspose(J,VecsSensiTemp[nadj],VecsDeltaLam[nadj*s+i]);CHKERRQ(ierr); /* VecsSensiTemp will be reused by 2nd-order adjoint */
        ierr = VecScale(VecsDeltaLam[nadj*s+i],-h*b[i]);CHKERRQ(ierr);
        if (quadts) {
          ierr = MatDenseGetColumn(Jquad,nadj,&xarr);CHKERRQ(ierr);
          ierr = VecPlaceArray(VecDRDUTransCol,xarr);CHKERRQ(ierr);
          ierr = VecAXPY(VecsDeltaLam[nadj*s+i],-h*b[i],VecDRDUTransCol);CHKERRQ(ierr);
          ierr = VecResetArray(VecDRDUTransCol);CHKERRQ(ierr);
          ierr = MatDenseRestoreColumn(Jquad,&xarr);CHKERRQ(ierr);
        }
      } else {
        /* \sum_{j=i+1}^s a_{ji}*lambda_{s,j} */
        ierr = VecSet(VecsSensiTemp[nadj],0);CHKERRQ(ierr);
        ierr = VecMAXPY(VecsSensiTemp[nadj],s-i-1,w,&VecsDeltaLam[nadj*s+i+1]);CHKERRQ(ierr);
        ierr = MatMultTranspose(J,VecsSensiTemp[nadj],VecsDeltaLam[nadj*s+i]);CHKERRQ(ierr);
        ierr = VecScale(VecsDeltaLam[nadj*s+i],-h);CHKERRQ(ierr);
      }

      /* Stage values of mu */
      if (ts->vecs_sensip) {
        ierr = MatMultTranspose(ts->Jacprhs,VecsSensiTemp[nadj],VecDeltaMu);CHKERRQ(ierr);
        if (b[i]) {
          ierr = VecScale(VecDeltaMu,-h*b[i]);CHKERRQ(ierr);
          if (quadts) {
            ierr = MatDenseGetColumn(quadts->Jacprhs,nadj,&xarr);CHKERRQ(ierr);
            ierr = VecPlaceArray(VecDRDPTransCol,xarr);CHKERRQ(ierr);
            ierr = VecAXPY(VecDeltaMu,-h*b[i],VecDRDPTransCol);CHKERRQ(ierr);
            ierr = VecResetArray(VecDRDPTransCol);CHKERRQ(ierr);
            ierr = MatDenseRestoreColumn(quadts->Jacprhs,&xarr);CHKERRQ(ierr);
          }
        } else {
          ierr = VecScale(VecDeltaMu,-h);CHKERRQ(ierr);
        }
        ierr = VecAXPY(ts->vecs_sensip[nadj],1.,VecDeltaMu);CHKERRQ(ierr); /* update sensip for each stage */
      }
    }

    if (ts->vecs_sensi2 && ts->forward_solve) { /* 2nd-order adjoint, TLM mode has to be turned on */
      /* Get w1 at t_{n+1} from TLM matrix */
      ierr = MatDenseGetColumn(rk->MatsFwdStageSensip[i],0,&xarr);CHKERRQ(ierr);
      ierr = VecPlaceArray(ts->vec_sensip_col,xarr);CHKERRQ(ierr);
      /* lambda_s^T F_UU w_1 */
      ierr = TSComputeRHSHessianProductFunctionUU(ts,rk->stage_time,Y[i],VecsSensiTemp,ts->vec_sensip_col,ts->vecs_guu);CHKERRQ(ierr);
      if (quadts)  {
        /* R_UU w_1 */
        ierr = TSComputeRHSHessianProductFunctionUU(quadts,rk->stage_time,Y[i],NULL,ts->vec_sensip_col,ts->vecs_guu);CHKERRQ(ierr);
      }
      if (ts->vecs_sensip) {
        /* lambda_s^T F_UP w_2 */
        ierr = TSComputeRHSHessianProductFunctionUP(ts,rk->stage_time,Y[i],VecsSensiTemp,ts->vec_dir,ts->vecs_gup);CHKERRQ(ierr);
        if (quadts)  {
          /* R_UP w_2 */
          ierr = TSComputeRHSHessianProductFunctionUP(quadts,rk->stage_time,Y[i],NULL,ts->vec_sensip_col,ts->vecs_gup);CHKERRQ(ierr);
        }
      }
      if (ts->vecs_sensi2p) {
        /* lambda_s^T F_PU w_1 */
        ierr = TSComputeRHSHessianProductFunctionPU(ts,rk->stage_time,Y[i],VecsSensiTemp,ts->vec_sensip_col,ts->vecs_gpu);CHKERRQ(ierr);
        /* lambda_s^T F_PP w_2 */
        ierr = TSComputeRHSHessianProductFunctionPP(ts,rk->stage_time,Y[i],VecsSensiTemp,ts->vec_dir,ts->vecs_gpp);CHKERRQ(ierr);
        if (b[i] && quadts) {
          /* R_PU w_1 */
          ierr = TSComputeRHSHessianProductFunctionPU(quadts,rk->stage_time,Y[i],NULL,ts->vec_sensip_col,ts->vecs_gpu);CHKERRQ(ierr);
          /* R_PP w_2 */
          ierr = TSComputeRHSHessianProductFunctionPP(quadts,rk->stage_time,Y[i],NULL,ts->vec_dir,ts->vecs_gpp);CHKERRQ(ierr);
        }
      }
      ierr = VecResetArray(ts->vec_sensip_col);CHKERRQ(ierr);
      ierr = MatDenseRestoreColumn(rk->MatsFwdStageSensip[i],&xarr);CHKERRQ(ierr);

      for (nadj=0; nadj<ts->numcost; nadj++) {
        /* Stage values of lambda */
        if (b[i]) {
          /* J_i^T*(Lambda_{n+1}+\sum_{j=i+1}^s a_{ji}/b_i*Lambda_{s,j} */
          ierr = VecCopy(ts->vecs_sensi2[nadj],VecsSensi2Temp[nadj]);CHKERRQ(ierr);
          ierr = VecMAXPY(VecsSensi2Temp[nadj],s-i-1,w,&VecsDeltaLam2[nadj*s+i+1]);CHKERRQ(ierr);
          ierr = MatMultTranspose(J,VecsSensi2Temp[nadj],VecsDeltaLam2[nadj*s+i]);CHKERRQ(ierr);
          ierr = VecScale(VecsDeltaLam2[nadj*s+i],-h*b[i]);CHKERRQ(ierr);
          ierr = VecAXPY(VecsDeltaLam2[nadj*s+i],-h*b[i],ts->vecs_guu[nadj]);CHKERRQ(ierr);
          if (ts->vecs_sensip) {
            ierr = VecAXPY(VecsDeltaLam2[nadj*s+i],-h*b[i],ts->vecs_gup[nadj]);CHKERRQ(ierr);
          }
        } else {
          /* \sum_{j=i+1}^s a_{ji}*Lambda_{s,j} */
          ierr = VecSet(VecsDeltaLam2[nadj*s+i],0);CHKERRQ(ierr);
          ierr = VecMAXPY(VecsSensi2Temp[nadj],s-i-1,w,&VecsDeltaLam2[nadj*s+i+1]);CHKERRQ(ierr);
          ierr = MatMultTranspose(J,VecsSensi2Temp[nadj],VecsDeltaLam2[nadj*s+i]);CHKERRQ(ierr);
          ierr = VecScale(VecsDeltaLam2[nadj*s+i],-h);CHKERRQ(ierr);
          ierr = VecAXPY(VecsDeltaLam2[nadj*s+i],-h,ts->vecs_guu[nadj]);CHKERRQ(ierr);
          if (ts->vecs_sensip) {
            ierr = VecAXPY(VecsDeltaLam2[nadj*s+i],-h,ts->vecs_gup[nadj]);CHKERRQ(ierr);
          }
        }
        if (ts->vecs_sensi2p) { /* 2nd-order adjoint for parameters */
          ierr = MatMultTranspose(ts->Jacprhs,VecsSensi2Temp[nadj],VecDeltaMu2);CHKERRQ(ierr);
          if (b[i]) {
            ierr = VecScale(VecDeltaMu2,-h*b[i]);CHKERRQ(ierr);
            ierr = VecAXPY(VecDeltaMu2,-h*b[i],ts->vecs_gpu[nadj]);CHKERRQ(ierr);
            ierr = VecAXPY(VecDeltaMu2,-h*b[i],ts->vecs_gpp[nadj]);CHKERRQ(ierr);
          } else {
            ierr = VecScale(VecDeltaMu2,-h);CHKERRQ(ierr);
            ierr = VecAXPY(VecDeltaMu2,-h,ts->vecs_gpu[nadj]);CHKERRQ(ierr);
            ierr = VecAXPY(VecDeltaMu2,-h,ts->vecs_gpp[nadj]);CHKERRQ(ierr);
          }
          ierr = VecAXPY(ts->vecs_sensi2p[nadj],1,VecDeltaMu2);CHKERRQ(ierr); /* update sensi2p for each stage */
        }
      }
    }
  }

  for (j=0; j<s; j++) w[j] = 1.0;
  for (nadj=0; nadj<ts->numcost; nadj++) { /* no need to do this for mu's */
    ierr = VecMAXPY(ts->vecs_sensi[nadj],s,w,&VecsDeltaLam[nadj*s]);CHKERRQ(ierr);
    if (ts->vecs_sensi2) {
      ierr = VecMAXPY(ts->vecs_sensi2[nadj],s,w,&VecsDeltaLam2[nadj*s]);CHKERRQ(ierr);
    }
  }
  rk->status = TS_STEP_COMPLETE;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdjointReset_RK(TS ts)
{
  TS_RK          *rk = (TS_RK*)ts->data;
  RKTableau      tab = rk->tableau;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(tab->s*ts->numcost,&rk->VecsDeltaLam);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ts->numcost,&rk->VecsSensiTemp);CHKERRQ(ierr);
  ierr = VecDestroy(&rk->VecDeltaMu);CHKERRQ(ierr);
  ierr = VecDestroyVecs(tab->s*ts->numcost,&rk->VecsDeltaLam2);CHKERRQ(ierr);
  ierr = VecDestroy(&rk->VecDeltaMu2);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ts->numcost,&rk->VecsSensi2Temp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSInterpolate_RK(TS ts,PetscReal itime,Vec X)
{
  TS_RK            *rk = (TS_RK*)ts->data;
  PetscInt         s  = rk->tableau->s,p = rk->tableau->p,i,j;
  PetscReal        h;
  PetscReal        tt,t;
  PetscScalar      *b;
  const PetscReal  *B = rk->tableau->binterp;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (!B) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"TSRK %s does not have an interpolation formula",rk->tableau->name);

  switch (rk->status) {
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
  ierr = PetscMalloc1(s,&b);CHKERRQ(ierr);
  for (i=0; i<s; i++) b[i] = 0;
  for (j=0,tt=t; j<p; j++,tt*=t) {
    for (i=0; i<s; i++) {
      b[i]  += h * B[i*p+j] * tt;
    }
  }
  ierr = VecCopy(rk->Y[0],X);CHKERRQ(ierr);
  ierr = VecMAXPY(X,s,b,rk->YdotRHS);CHKERRQ(ierr);
  ierr = PetscFree(b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode TSRKTableauReset(TS ts)
{
  TS_RK          *rk = (TS_RK*)ts->data;
  RKTableau      tab = rk->tableau;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tab) PetscFunctionReturn(0);
  ierr = PetscFree(rk->work);CHKERRQ(ierr);
  ierr = VecDestroyVecs(tab->s,&rk->Y);CHKERRQ(ierr);
  ierr = VecDestroyVecs(tab->s,&rk->YdotRHS);CHKERRQ(ierr);
  ierr = VecDestroyVecs(tab->s*ts->numcost,&rk->VecsDeltaLam);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ts->numcost,&rk->VecsSensiTemp);CHKERRQ(ierr);
  ierr = VecDestroy(&rk->VecDeltaMu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSReset_RK(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSRKTableauReset(ts);CHKERRQ(ierr);
  if (ts->use_splitrhsfunction) {
    ierr = PetscTryMethod(ts,"TSReset_RK_MultirateSplit_C",(TS),(ts));CHKERRQ(ierr);
  } else {
    ierr = PetscTryMethod(ts,"TSReset_RK_MultirateNonsplit_C",(TS),(ts));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCoarsenHook_TSRK(DM fine,DM coarse,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRestrictHook_TSRK(DM fine,Mat restrct,Vec rscale,Mat inject,DM coarse,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSubDomainHook_TSRK(DM dm,DM subdm,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSubDomainRestrictHook_TSRK(DM dm,VecScatter gscat,VecScatter lscat,DM subdm,void *ctx)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSRKTableauSetUp(TS ts)
{
  TS_RK          *rk  = (TS_RK*)ts->data;
  RKTableau      tab = rk->tableau;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(tab->s,&rk->work);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(ts->vec_sol,tab->s,&rk->Y);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(ts->vec_sol,tab->s,&rk->YdotRHS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetUp_RK(TS ts)
{
  TS             quadts = ts->quadraturets;
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  ierr = TSCheckImplicitTerm(ts);CHKERRQ(ierr);
  ierr = TSRKTableauSetUp(ts);CHKERRQ(ierr);
  if (quadts && ts->costintegralfwd) {
    Mat Jquad;
    ierr = TSGetRHSJacobian(quadts,&Jquad,NULL,NULL,NULL);CHKERRQ(ierr);
  }
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMCoarsenHookAdd(dm,DMCoarsenHook_TSRK,DMRestrictHook_TSRK,ts);CHKERRQ(ierr);
  ierr = DMSubDomainHookAdd(dm,DMSubDomainHook_TSRK,DMSubDomainRestrictHook_TSRK,ts);CHKERRQ(ierr);
  if (ts->use_splitrhsfunction) {
    ierr = PetscTryMethod(ts,"TSSetUp_RK_MultirateSplit_C",(TS),(ts));CHKERRQ(ierr);
  } else {
    ierr = PetscTryMethod(ts,"TSSetUp_RK_MultirateNonsplit_C",(TS),(ts));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetFromOptions_RK(PetscOptionItems *PetscOptionsObject,TS ts)
{
  TS_RK          *rk = (TS_RK*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"RK ODE solver options");CHKERRQ(ierr);
  {
    RKTableauLink link;
    PetscInt      count,choice;
    PetscBool     flg,use_multirate = PETSC_FALSE;
    const char    **namelist;

    for (link=RKTableauList,count=0; link; link=link->next,count++) ;
    ierr = PetscMalloc1(count,(char***)&namelist);CHKERRQ(ierr);
    for (link=RKTableauList,count=0; link; link=link->next,count++) namelist[count] = link->tab.name;
    ierr = PetscOptionsBool("-ts_rk_multirate","Use interpolation-based multirate RK method","TSRKSetMultirate",rk->use_multirate,&use_multirate,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = TSRKSetMultirate(ts,use_multirate);CHKERRQ(ierr);
    }
    ierr = PetscOptionsEList("-ts_rk_type","Family of RK method","TSRKSetType",(const char*const*)namelist,count,rk->tableau->name,&choice,&flg);CHKERRQ(ierr);
    if (flg) {ierr = TSRKSetType(ts,namelist[choice]);CHKERRQ(ierr);}
    ierr = PetscFree(namelist);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)ts),NULL,"Multirate methods options","");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ts_rk_dtratio","time step ratio between slow and fast","",rk->dtratio,&rk->dtratio,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSView_RK(TS ts,PetscViewer viewer)
{
  TS_RK          *rk = (TS_RK*)ts->data;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    RKTableau       tab  = rk->tableau;
    TSRKType        rktype;
    const PetscReal *c;
    PetscInt        s;
    char            buf[512];
    PetscBool       FSAL;

    ierr = TSRKGetType(ts,&rktype);CHKERRQ(ierr);
    ierr = TSRKGetTableau(ts,&s,NULL,NULL,&c,NULL,NULL,NULL,&FSAL);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  RK type %s\n",rktype);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Order: %D\n",tab->order);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  FSAL property: %s\n",FSAL ? "yes" : "no");CHKERRQ(ierr);
    ierr = PetscFormatRealArray(buf,sizeof(buf),"% 8.6f",s,c);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Abscissa c = %s\n",buf);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSLoad_RK(TS ts,PetscViewer viewer)
{
  PetscErrorCode ierr;
  TSAdapt        adapt;

  PetscFunctionBegin;
  ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptLoad(adapt,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TSRKGetOrder - Get the order of RK scheme

  Not collective

  Input Parameter:
.  ts - timestepping context

  Output Parameter:
.  order - order of RK-scheme

  Level: intermediate

.seealso: TSRKGetType()
@*/
PetscErrorCode TSRKGetOrder(TS ts,PetscInt *order)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidIntPointer(order,2);
  ierr = PetscUseMethod(ts,"TSRKGetOrder_C",(TS,PetscInt*),(ts,order));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSRKSetType - Set the type of RK scheme

  Logically collective

  Input Parameters:
+  ts - timestepping context
-  rktype - type of RK-scheme

  Options Database:
.   -ts_rk_type - <1fe,2a,3,3bs,4,5f,5dp,5bs>

  Level: intermediate

.seealso: TSRKGetType(), TSRK, TSRKType, TSRK1FE, TSRK2A, TSRK3, TSRK3BS, TSRK4, TSRK5F, TSRK5DP, TSRK5BS, TSRK6VR, TSRK7VR, TSRK8VR
@*/
PetscErrorCode TSRKSetType(TS ts,TSRKType rktype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidCharPointer(rktype,2);
  ierr = PetscTryMethod(ts,"TSRKSetType_C",(TS,TSRKType),(ts,rktype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSRKGetType - Get the type of RK scheme

  Not collective

  Input Parameter:
.  ts - timestepping context

  Output Parameter:
.  rktype - type of RK-scheme

  Level: intermediate

.seealso: TSRKSetType()
@*/
PetscErrorCode TSRKGetType(TS ts,TSRKType *rktype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscUseMethod(ts,"TSRKGetType_C",(TS,TSRKType*),(ts,rktype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSRKGetOrder_RK(TS ts,PetscInt *order)
{
  TS_RK *rk = (TS_RK*)ts->data;

  PetscFunctionBegin;
  *order = rk->tableau->order;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSRKGetType_RK(TS ts,TSRKType *rktype)
{
  TS_RK *rk = (TS_RK*)ts->data;

  PetscFunctionBegin;
  *rktype = rk->tableau->name;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSRKSetType_RK(TS ts,TSRKType rktype)
{
  TS_RK          *rk = (TS_RK*)ts->data;
  PetscErrorCode ierr;
  PetscBool      match;
  RKTableauLink  link;

  PetscFunctionBegin;
  if (rk->tableau) {
    ierr = PetscStrcmp(rk->tableau->name,rktype,&match);CHKERRQ(ierr);
    if (match) PetscFunctionReturn(0);
  }
  for (link = RKTableauList; link; link=link->next) {
    ierr = PetscStrcmp(link->tab.name,rktype,&match);CHKERRQ(ierr);
    if (match) {
      if (ts->setupcalled) {ierr = TSRKTableauReset(ts);CHKERRQ(ierr);}
      rk->tableau = &link->tab;
      if (ts->setupcalled) {ierr = TSRKTableauSetUp(ts);CHKERRQ(ierr);}
      ts->default_adapt_type = rk->tableau->bembed ? TSADAPTBASIC : TSADAPTNONE;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_UNKNOWN_TYPE,"Could not find '%s'",rktype);
}

static PetscErrorCode  TSGetStages_RK(TS ts,PetscInt *ns,Vec **Y)
{
  TS_RK *rk = (TS_RK*)ts->data;

  PetscFunctionBegin;
  if (ns) *ns = rk->tableau->s;
  if (Y)   *Y = rk->Y;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDestroy_RK(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_RK(ts);CHKERRQ(ierr);
  if (ts->dm) {
    ierr = DMCoarsenHookRemove(ts->dm,DMCoarsenHook_TSRK,DMRestrictHook_TSRK,ts);CHKERRQ(ierr);
    ierr = DMSubDomainHookRemove(ts->dm,DMSubDomainHook_TSRK,DMSubDomainRestrictHook_TSRK,ts);CHKERRQ(ierr);
  }
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSRKGetOrder_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSRKGetType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSRKSetType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSRKGetTableau_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSRKSetMultirate_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSRKGetMultirate_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  This defines the nonlinear equation that is to be solved with SNES
  We do not need to solve the equation; we just use SNES to approximate the Jacobian
*/
static PetscErrorCode SNESTSFormFunction_RK(SNES snes,Vec x,Vec y,TS ts)
{
  TS_RK          *rk = (TS_RK*)ts->data;
  DM             dm,dmsave;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr   = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  /* DM monkey-business allows user code to call TSGetDM() inside of functions evaluated on levels of FAS */
  dmsave = ts->dm;
  ts->dm = dm;
  ierr   = TSComputeRHSFunction(ts,rk->stage_time,x,y);CHKERRQ(ierr);
  ts->dm = dmsave;
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESTSFormJacobian_RK(SNES snes,Vec x,Mat A,Mat B,TS ts)
{
  TS_RK          *rk = (TS_RK*)ts->data;
  DM             dm,dmsave;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr   = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  dmsave = ts->dm;
  ts->dm = dm;
  ierr   = TSComputeRHSJacobian(ts,rk->stage_time,x,A,B);CHKERRQ(ierr);
  ts->dm = dmsave;
  PetscFunctionReturn(0);
}

/*@C
  TSRKSetMultirate - Use the interpolation-based multirate RK method

  Logically collective

  Input Parameters:
+  ts - timestepping context
-  use_multirate - PETSC_TRUE enables the multirate RK method, sets the basic method to be RK2A and sets the ratio between slow stepsize and fast stepsize to be 2

  Options Database:
.   -ts_rk_multirate - <true,false>

  Notes:
  The multirate method requires interpolation. The default interpolation works for 1st- and 2nd- order RK, but not for high-order RKs except TSRK5DP which comes with the interpolation coeffcients (binterp).

  Level: intermediate

.seealso: TSRKGetMultirate()
@*/
PetscErrorCode TSRKSetMultirate(TS ts,PetscBool use_multirate)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(ts,"TSRKSetMultirate_C",(TS,PetscBool),(ts,use_multirate));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSRKGetMultirate - Gets whether to Use the interpolation-based multirate RK method

  Not collective

  Input Parameter:
.  ts - timestepping context

  Output Parameter:
.  use_multirate - PETSC_TRUE if the multirate RK method is enabled, PETSC_FALSE otherwise

  Level: intermediate

.seealso: TSRKSetMultirate()
@*/
PetscErrorCode TSRKGetMultirate(TS ts,PetscBool *use_multirate)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(ts,"TSRKGetMultirate_C",(TS,PetscBool*),(ts,use_multirate));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
      TSRK - ODE and DAE solver using Runge-Kutta schemes

  The user should provide the right hand side of the equation
  using TSSetRHSFunction().

  Notes:
  The default is TSRK3BS, it can be changed with TSRKSetType() or -ts_rk_type

  Level: beginner

.seealso:  TSCreate(), TS, TSSetType(), TSRKSetType(), TSRKGetType(), TSRK2D, TTSRK2E, TSRK3,
           TSRK4, TSRK5, TSRKPRSSP2, TSRKBPR3, TSRKType, TSRKRegister(), TSRKSetMultirate(), TSRKGetMultirate()

M*/
PETSC_EXTERN PetscErrorCode TSCreate_RK(TS ts)
{
  TS_RK          *rk;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSRKInitializePackage();CHKERRQ(ierr);

  ts->ops->reset          = TSReset_RK;
  ts->ops->destroy        = TSDestroy_RK;
  ts->ops->view           = TSView_RK;
  ts->ops->load           = TSLoad_RK;
  ts->ops->setup          = TSSetUp_RK;
  ts->ops->interpolate    = TSInterpolate_RK;
  ts->ops->step           = TSStep_RK;
  ts->ops->evaluatestep   = TSEvaluateStep_RK;
  ts->ops->rollback       = TSRollBack_RK;
  ts->ops->setfromoptions = TSSetFromOptions_RK;
  ts->ops->getstages      = TSGetStages_RK;

  ts->ops->snesfunction    = SNESTSFormFunction_RK;
  ts->ops->snesjacobian    = SNESTSFormJacobian_RK;
  ts->ops->adjointintegral = TSAdjointCostIntegral_RK;
  ts->ops->adjointsetup    = TSAdjointSetUp_RK;
  ts->ops->adjointstep     = TSAdjointStep_RK;
  ts->ops->adjointreset    = TSAdjointReset_RK;

  ts->ops->forwardintegral = TSForwardCostIntegral_RK;
  ts->ops->forwardsetup    = TSForwardSetUp_RK;
  ts->ops->forwardreset    = TSForwardReset_RK;
  ts->ops->forwardstep     = TSForwardStep_RK;
  ts->ops->forwardgetstages= TSForwardGetStages_RK;

  ierr = PetscNewLog(ts,&rk);CHKERRQ(ierr);
  ts->data = (void*)rk;

  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSRKGetOrder_C",TSRKGetOrder_RK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSRKGetType_C",TSRKGetType_RK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSRKSetType_C",TSRKSetType_RK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSRKGetTableau_C",TSRKGetTableau_RK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSRKSetMultirate_C",TSRKSetMultirate_RK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSRKGetMultirate_C",TSRKGetMultirate_RK);CHKERRQ(ierr);

  ierr = TSRKSetType(ts,TSRKDefault);CHKERRQ(ierr);
  rk->dtratio = 1;
  PetscFunctionReturn(0);
}
