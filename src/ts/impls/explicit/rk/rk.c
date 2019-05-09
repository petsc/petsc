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

     This method has ten stages with the First Same As Last (FSAL) property.

     Options database:
.     -ts_rk_type 7vr

     Level: advanced

     References: http://people.math.sfu.ca/~jverner/RKV76.IIa.Robust.000027015646.081206.CoeffsOnlyRAT

.seealso: TSRK, TSRKType, TSRKSetType()
M*/
/*MC
     TSRK8VR - Eigth order robust Verner RK scheme with seventh order embedded method.

     This method has thirteen stages with the First Same As Last (FSAL) property.

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

.keywords: TS, TSRK, register, all

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
      A[7][7]   = {{0,0,0,0,0,0,0},
                   {RC(1.0)/RC(5.0),0,0,0,0,0,0},
                   {RC(3.0)/RC(40.0),RC(9.0)/RC(40.0),0,0,0,0,0},
                   {RC(44.0)/RC(45.0),RC(-56.0)/RC(15.0),RC(32.0)/RC(9.0),0,0,0,0},
                   {RC(19372.0)/RC(6561.0),RC(-25360.0)/RC(2187.0),RC(64448.0)/RC(6561.0),RC(-212.0)/RC(729.0),0,0,0},
                   {RC(9017.0)/RC(3168.0),RC(-355.0)/RC(33.0),RC(46732.0)/RC(5247.0),RC(49.0)/RC(176.0),RC(-5103.0)/RC(18656.0),0,0},
                   {RC(35.0)/RC(384.0),0,RC(500.0)/RC(1113.0),RC(125.0)/RC(192.0),RC(-2187.0)/RC(6784.0),RC(11.0)/RC(84.0),0}},
      b[7]      =  {RC(35.0)/RC(384.0),0,RC(500.0)/RC(1113.0),RC(125.0)/RC(192.0),RC(-2187.0)/RC(6784.0),RC(11.0)/RC(84.0),0},
        bembed[7] =  {RC(5179.0)/RC(57600.0),0,RC(7571.0)/RC(16695.0),RC(393.0)/RC(640.0),RC(-92097.0)/RC(339200.0),RC(187.0)/RC(2100.0),RC(1.0)/RC(40.0)},
        binterp[7][5] =  {{RC(1.0),RC(-4034104133.0)/RC(1410260304.0),RC(105330401.0)/RC(33982176.0),RC(-13107642775.0)/RC(11282082432.0),RC(6542295.0)/RC(470086768.0)},
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
                   {RC(9.0)/RC(50.0),0,0,0,0,0,0,0,0},
                   {RC(29.0)/RC(324.0),RC(25.0)/RC(324.0),0,0,0,0,0,0,0},
                   {RC(1.0)/RC(16.0),0,RC(3.0)/RC(16.0),0,0,0,0,0,0},
                   {RC(79129.0)/RC(250000.0),0,RC(-261237.0)/RC(250000.0),RC(19663.0)/RC(15625.0),0,0,0,0,0},
                   {RC(1336883.0)/RC(4909125.0),0,RC(-25476.0)/RC(30875.0),RC(194159.0)/RC(185250.0),RC(8225.0)/RC(78546.0),0,0,0,0},
                   {RC(-2459386.0)/RC(14727375.0),0,RC(19504.0)/RC(30875.0),RC(2377474.0)/RC(13615875.0),RC(-6157250.0)/RC(5773131.0),RC(902.0)/RC(735.0),0,0,0},
                   {RC(2699.0)/RC(7410.0),0,RC(-252.0)/RC(1235.0),RC(-1393253.0)/RC(3993990.0),RC(236875.0)/RC(72618.0),RC(-135.0)/RC(49.0),RC(15.0)/RC(22.0),0,0},
                   {RC(11.0)/RC(144.0),0,0,RC(256.0)/RC(693.0),0,RC(125.0)/RC(504.0),RC(125.0)/RC(528.0),RC(5.0)/RC(72.0),0}},
      b[9]      =  {RC(11.0)/RC(144.0),0,0,RC(256.0)/RC(693.0),0,RC(125.0)/RC(504.0),RC(125.0)/RC(528.0),RC(5.0)/RC(72.0),0},
      bembed[9] =  {RC(28.0)/RC(477.0),0,0,RC(212.0)/RC(441.0),RC(-312500.0)/RC(366177.0),RC(2125.0)/RC(1764.0),0,RC(-2105.0)/RC(35532.0),RC(2995.0)/RC(17766.0)};
    ierr = TSRKRegister(TSRK6VR,6,9,&A[0][0],b,NULL,bembed,0,NULL);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[10][10]  = {{0,0,0,0,0,0,0,0,0,0},
                    {RC(1.0)/RC(200.0),0,0,0,0,0,0,0,0,0},
                    {RC(-4361.0)/RC(4050.0),RC(2401.0)/RC(2025.0),0,0,0,0,0,0,0,0},
                    {RC(49.0)/RC(1200.0),0,RC(49.0)/RC(400.0),0,0,0,0,0,0,0},
                    {RC(1781.0)/RC(2800.0),0,RC(-13689.0)/RC(5600.0),RC(507.0)/RC(224.0),0,0,0,0,0,0},
                    {RC(-275776923568321554889485313326460.0)/RC(108782544039075797415588764982099.0),0,RC(9576001512158705648097438730000.0)/RC(929765333667314507825544999847.0),RC(-22178538465642954902290458689600.0)/RC(2789296001001943523476634999541.0),RC(12323686750109283453854913779200.0)/RC(15540363434153685345084109283157.0),0,0,0,0,0},
                    {RC(8221508014471276464993.0)/RC(8206108584945090400000.0),0,RC(-9852144759099.0)/RC(2364568872400.0),RC(15322550932778398907299.0)/RC(3996134347464283007200.0),RC(-18338463121898520004.0)/RC(36506562121215938675.0),RC(23340475544602125119307511373519383499.0)/RC(34957329425893779598660175543672800000.0),0,0,0,0},
                    {RC(81088643022740545033730780169.0)/RC(2975182110231937140152800000.0),0,RC(-2837794586103.0)/RC(67559110640.0),RC(-14167575606881316095038341141.0)/RC(1344719188593468553337836000.0),RC(2395552232834276839307772.0)/RC(29760062864388068268875.0),RC(-4076715891031001341580357765362043260356514682697.0)/RC(60535801523558513633981092635987721507186400000.0),RC(36551527355459957808.0)/RC(2801171464968864455.0),0,0,0},
                    {RC(-3347747115771808477876752833599.0)/RC(1101327591307901549464211073280.0),0,RC(1214704878477625125.0)/RC(119815105452943264.0),RC(-65581118617864038124456452415.0)/RC(10200342297342072539428709672.0),RC(-133373082911575479273298406095.0)/RC(84070826916189821955373830704.0),RC(622515683654039386383701463758952267447736841050281950137693.0)/RC(328994218860140584540186142455568117669077094653332432085760.0),RC(46169188671551441399013763125.0)/RC(2343692704156841275628930358208.0),RC(18880867865877597493091875.0)/RC(3469664148196911643475533504.0),0,0},
                    {RC(-74309815528722196127526037.0)/RC(51427190037752065777334640.0),0,RC(913722369253875.0)/RC(113761793498234.0),RC(-440658227159292602060396890.0)/RC(58109996881684093545238767.0),RC(37290888293935763341814380.0)/RC(10411746696360295961914239.0),RC(-645152888113581065780360392105179310452496326847.0)/RC(264735425121804814898131042131367320451487910960.0),RC(473757486190086838681892500.0)/RC(556321100802942639360459887.0),0,0,0}},
      b[10]      =  {RC(9420080774669597.0)/RC(198627609019792680.0),0,0,RC(18658605936510000.0)/RC(72821569629535727.0),RC(296950875175030000.0)/RC(1101802245630054969.0),RC(18875276980274212686824835566143151189850553896330009.0)/RC(148780947139609706104394157596648357994575577036224440.0),RC(18663850606812500.0)/RC(74993696164706319.0),RC(179884749312500.0)/RC(58508928482581269.0),RC(349315176247648.0)/RC(7273791403140339.0),0},
      bembed[10] =  {RC(7362904929137.0)/RC(155056681514280.0),0,0,RC(7505178129270000.0)/RC(29317774785916981.0),RC(1851744839320000.0)/RC(6843492208882329.0),RC(750882778189818437512810407839645051128089.0)/RC(6004363295715536270735789992319176322209240.0),RC(27902602073000000.0)/RC(110704980052661709.0),0,0,RC(331667036438.0)/RC(6791588611709.0)};
    ierr = TSRKRegister(TSRK7VR,7,10,&A[0][0],b,NULL,bembed,0,NULL);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[13][13]  = {{0,0,0,0,0,0,0,0,0,0,0,0,0},
                    {RC(1.0)/RC(4.0),0,0,0,0,0,0,0,0,0,0,0,0},
                    {RC(25374388559.0)/RC(290322000000.0),RC(7399612441.0)/RC(290322000000.0),0,0,0,0,0,0,0,0,0,0,0},
                    {RC(86021.0)/RC(2032000.0),0,RC(258063.0)/RC(2032000.0),0,0,0,0,0,0,0,0,0,0},
                    {RC(394117287349.0)/RC(924951555125.0),0,RC(-1478808184872.0)/RC(924951555125.0),RC(1476870356896.0)/RC(924951555125.0),0,0,0,0,0,0,0,0,0},
                    {RC(21806069.0)/RC(429936000.0),0,0,RC(2085253894.0)/RC(8198887125.0),RC(67122964561.0)/RC(329119824000.0),0,0,0,0,0,0,0,0},
                    {RC(-10767669849309.0)/RC(37129416272000.0),0,0,RC(107549272780798434.0)/RC(80010624633743375.0),RC(-1571425354197.0)/RC(548533040000.0),RC(195974150961.0)/RC(73190382500.0),0,0,0,0,0,0,0},
                    {RC(122914197.0)/RC(1247416480.0),0,0,0,RC(2834577.0)/RC(12772576.0),RC(-1123911.0)/RC(6195548.0),RC(1003245.0)/RC(91667332.0),0,0,0,0,0,0},
                    {RC(7516815761873698650240937021297888610305781198156454914018066425686713809124.0)/RC(19417750920086037142705440148991358353524951455336635926349428617092008016799.0),0,0,RC(-2970280712345026029784493078033500603558797778620788610891609609625083520000000.0)/RC(2059197881322255811366824379732129118957063276855086357832879720044283034195609.0),RC(160883051099229570661648871967605637311447402857798976770144318499344176200.0)/RC(55373838844007112802284696674027443630749586849658020349496486226959885797.0),RC(-531291044569132007849980767445890096567679581535833821522948115776782916800.0)/RC(286600138103969709516639731800408929385020510549016787456022559956334619913.0),RC(92917760295507691123384296375592154091398238056386167422408760067912000.0)/RC(663525387387781548392930054085313627947883220329590736767284727992793459.0),RC(16582882008248100072444682934459438760351720266822956376058566080.0)/RC(28953624935338209620540938361130162616618239541632595427443431123.0),0,0,0,0,0},
                    {RC(-397907419247815925921130059686865048509124.0)/RC(2467734205602744466122026313943800290709375.0),0,0,RC(-12397380504027867540933244272640.0)/RC(71497487771289358131961402817451.0),RC(-938215172932764769802614036208.0)/RC(720988934849815411624418693625.0),RC(39294182008628611505961825730565883738506112.0)/RC(34530663960302355453963385168366837619171375.0),RC(-44522287344548287709730039702917122688.0)/RC(1402380579847472465941191457456761953225.0),RC(20222100246339043804092120397184.0)/RC(21662367405707901792558292907875.0),RC(-30521958981544768864151899144647092370777642374348985368652529884.0)/RC(364283329166067503011761408054691133551081505869170977618646228125.0),0,0,0,0},
                    {RC(-7874248153103427073301520825686819245245713313.0)/RC(410128949126757953376487799591079982676499635200.0),0,0,RC(592147767606693369421771800000.0)/RC(2166590538523919943392769782347.0),RC(-3777316336297553135210313111.0)/RC(5593126282471295314419732896.0),RC(160775746506894169042900490691204481389877265439.0)/RC(470767315337957626238784466576517627715500536844.0),RC(-507593940096706904938228151863398894238905405.0)/RC(7470102369519587845109530946292707809479335476.0),RC(173196146788876378135182681744256899.0)/RC(1793073867682496036448646518502714730.0),RC(937132687483187059957479559988037034595389945431346986697458481846645979573026347.0)/RC(7071054501415007346897024777503547537721817297910549006214731593062974677881100800.0),RC(116416656136756811965545.0)/RC(315877857843708025406464.0),0,0,0},
                    {RC(28191405932827173126309408200664465069287507375903.0)/RC(46277040828099807659589313374497236005056822803488.0),0,0,RC(-429380072388144504981858700675520000000.0)/RC(188940382517997566021881003968356031669.0),RC(24173887609985281514926963208103700.0)/RC(5080791111711202068770238453613777.0),RC(-54881449674479480016593038012420742831966782264181600.0)/RC(9949308922521596315581680829335134273237927944004721.0),RC(970969892075126686370480622782649944320946827348000.0)/RC(3347483649309697775724736894621937483768621235854151.0),RC(77411823900067580361972977027037723040.0)/RC(136014860952969119034946902128158999209.0),RC(656779806953299263849114640553093566236946159841995197013788762083697980273440465097410146645645.0)/RC(828556489672666107963907105193770544832899849879697116662334249174938654251472014083110713914032.0),RC(3147129425650261440272289698280046875.0)/RC(20338543888980253307598069078752387872.0),RC(425854850025102447819650719213220000.0)/RC(263691965699784622774911587349396263.0),0,0},
                    {RC(3621456146894606285894995339383850874108700148373.0)/RC(4081168693163363596994878822807031005774876048608.0),0,0,RC(-3817583686059663366913679467944640000000.0)/RC(1283023117642128560298752783599596583383.0),RC(193234496869635405437524415902310900.0)/RC(34501742641572006835386893540142539.0),RC(-87974988257716022018037206149145819377007277738400.0)/RC(14871674463411999056342756887683215806741009406229.0),RC(3275133743198491987936602396440511473300401684000.0)/RC(14866908561378488620343856456061248578257652060133.0),RC(8526827543558748199980436728262708480.0)/RC(83965981332240915696885276969279079833.0),RC(39009210105998257555064429930449313140180415214883261536546755024187088643232941505.0)/RC(33878790250537583879968593309051059917530985063051045330469429477091985229650821808.0),RC(266248736648182734031056196146328125.0)/RC(137973433143769650633804707559140704.0),0,0,0}},
      b[13]      =  {RC(105711409411029096011.0)/RC(2363345366733216286848.0),0,0,0,0,RC(49144903996079337500000000000.0)/RC(313203740909075359159534015851.0),RC(122908398183230837500000000000.0)/RC(665774200882383246774037878723.0),RC(14734515405327276428000.0)/RC(65439093723444914604579.0),RC(98855236015768950806532226039371426875067857784942240050406497922541356709067743917330653.0)/RC(668183874060994347945818441723205177438829897987551586564757651033104640132059340614221120.0),RC(80375815959623894921875.0)/RC(1056804190408540040777856.0),RC(48020889007901260000.0)/RC(391135894718289455007.0),RC(1146762646211206909.0)/RC(27426666521711241540.0),0},
      bembed[13] =  {RC(415143901557332051.0)/RC(9054963090931863168.0),0,0,0,0,RC(1392529918457821250000000000.0)/RC(5308537981509751850161593489.0),RC(2142381651020498750000000000.0)/RC(11176065722246241552561786933.0),RC(4735428542080120856000.0)/RC(21813031241148304868193.0),RC(1360794378607602525113722425939330174725374220595898325328629456869721330363.0)/RC(10682792599936408020924171586589409694572260337238697644239411829708789571520.0),RC(9357212880829926171875.0)/RC(81292630031426156982912.0),0,0,RC(7787234078438942063.0)/RC(191986665651978690780.0)};
    ierr = TSRKRegister(TSRK8VR,8,13,&A[0][0],b,NULL,bembed,0,NULL);CHKERRQ(ierr);
  }
#undef RC
  PetscFunctionReturn(0);
}

/*@C
   TSRKRegisterDestroy - Frees the list of schemes that were registered by TSRKRegister().

   Not Collective

   Level: advanced

.keywords: TSRK, register, destroy
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
    ierr = PetscFree3(t->A,t->b,t->c);  CHKERRQ(ierr);
    ierr = PetscFree (t->bembed);       CHKERRQ(ierr);
    ierr = PetscFree (t->binterp);      CHKERRQ(ierr);
    ierr = PetscFree (t->name);         CHKERRQ(ierr);
    ierr = PetscFree (link);            CHKERRQ(ierr);
  }
  TSRKRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  TSRKInitializePackage - This function initializes everything in the TSRK package. It is called
  from TSInitializePackage().

  Level: developer

.keywords: TS, TSRK, initialize, package
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

.keywords: Petsc, destroy, package
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

.keywords: TS, register

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
  ierr = PetscMemcpy(t->A,A,s*s*sizeof(A[0]));CHKERRQ(ierr);
  if (b)  { ierr = PetscMemcpy(t->b,b,s*sizeof(b[0]));CHKERRQ(ierr); }
  else for (i=0; i<s; i++) t->b[i] = A[(s-1)*s+i];
  if (c)  { ierr = PetscMemcpy(t->c,c,s*sizeof(c[0]));CHKERRQ(ierr); }
  else for (i=0; i<s; i++) for (j=0,t->c[i]=0; j<s; j++) t->c[i] += A[i*s+j];
  t->FSAL = PETSC_TRUE;
  for (i=0; i<s; i++) if (t->A[(s-1)*s+i] != t->b[i]) t->FSAL = PETSC_FALSE;

  if (bembed) {
    ierr = PetscMalloc1(s,&t->bembed);CHKERRQ(ierr);
    ierr = PetscMemcpy(t->bembed,bembed,s*sizeof(bembed[0]));CHKERRQ(ierr);
  }

  if (!binterp) { p = 1; binterp = t->b; }
  t->p = p;
  ierr = PetscMalloc1(s*p,&t->binterp);CHKERRQ(ierr);
  ierr = PetscMemcpy(t->binterp,binterp,s*p*sizeof(binterp[0]));CHKERRQ(ierr);

  link->next = RKTableauList;
  RKTableauList = link;
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
    if(!i) {
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
        ierr = VecResetArray(rk->VecDeltaFwdSensipCol);CHKERRQ(ierr);CHKERRQ(ierr);
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
  for(i=0; i<tab->s; i++) {
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
      ierr = TSPreStage(ts,rk->stage_time); CHKERRQ(ierr);
      ierr = VecCopy(ts->vec_sol,Y[i]);CHKERRQ(ierr);
      for (j=0; j<i; j++) w[j] = h*A[i*s+j];
      ierr = VecMAXPY(Y[i],i,w,YdotRHS);CHKERRQ(ierr);
      ierr = TSPostStage(ts,rk->stage_time,i,Y); CHKERRQ(ierr);
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
  if(ts->vecs_sensip) {
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
  Mat              J,Jquad;
  const PetscInt   s = tab->s;
  const PetscReal  *A = tab->A,*b = tab->b,*c = tab->c;
  PetscScalar      *w = rk->work,*xarr;
  Vec              *Y = rk->Y,*VecsDeltaLam = rk->VecsDeltaLam,VecDeltaMu = rk->VecDeltaMu,*VecsSensiTemp = rk->VecsSensiTemp;
  Vec              *VecsDeltaLam2 = rk->VecsDeltaLam2,VecDeltaMu2 = rk->VecDeltaMu2,*VecsSensi2Temp = rk->VecsSensi2Temp;
  Vec              VecDRDUTransCol = ts->vec_drdu_col,VecDRDPTransCol = ts->vec_drdp_col;
  PetscInt         i,j,nadj;
  PetscReal        t = ts->ptime;
  PetscReal        h = ts->time_step,stage_time;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  rk->status = TS_STEP_INCOMPLETE;

  ierr = TSGetRHSJacobian(ts,&J,NULL,NULL,NULL);CHKERRQ(ierr);
  if (quadts) {
    ierr = TSGetRHSJacobian(quadts,&Jquad,NULL,NULL,NULL);CHKERRQ(ierr);
  }
  for (i=s-1; i>=0; i--) {
    if (tab->FSAL && i == s-1) {
      /* VecsDeltaLam[nadj*s+s-1] are initialized with zeros and the values never change.*/
      continue;
    }
    stage_time = t + h*(1.0-c[i]);
    ierr = TSComputeRHSJacobian(ts,stage_time,Y[i],J,J);CHKERRQ(ierr);
    if (quadts) {
      ierr = TSComputeRHSJacobian(quadts,stage_time,Y[i],Jquad,Jquad);CHKERRQ(ierr); /* get r_u^T */
    }
    if (ts->vecs_sensip) {
      ierr = TSComputeRHSJacobianP(ts,stage_time,Y[i],ts->Jacprhs);CHKERRQ(ierr); /* get f_p */
      if (quadts) {
        ierr = TSComputeRHSJacobianP(quadts,stage_time,Y[i],quadts->Jacprhs);CHKERRQ(ierr); /* get f_p for the quadrature */
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
      ierr = TSComputeRHSHessianProductFunctionUU(ts,stage_time,Y[i],VecsSensiTemp,ts->vec_sensip_col,ts->vecs_guu);CHKERRQ(ierr);
      if (quadts)  {
        /* R_UU w_1 */
        ierr = TSComputeRHSHessianProductFunctionUU(quadts,stage_time,Y[i],NULL,ts->vec_sensip_col,ts->vecs_guu);CHKERRQ(ierr);
      }
      if (ts->vecs_sensip) {
        /* lambda_s^T F_UP w_2 */
        ierr = TSComputeRHSHessianProductFunctionUP(ts,stage_time,Y[i],VecsSensiTemp,ts->vec_dir,ts->vecs_gup);CHKERRQ(ierr);
        if (quadts)  {
          /* R_UP w_2 */
          ierr = TSComputeRHSHessianProductFunctionUP(quadts,stage_time,Y[i],NULL,ts->vec_sensip_col,ts->vecs_gup);CHKERRQ(ierr);
        }
      }
      if (ts->vecs_sensi2p) {
        /* lambda_s^T F_PU w_1 */
        ierr = TSComputeRHSHessianProductFunctionPU(ts,stage_time,Y[i],VecsSensiTemp,ts->vec_sensip_col,ts->vecs_gpu);CHKERRQ(ierr);
        /* lambda_s^T F_PP w_2 */
        ierr = TSComputeRHSHessianProductFunctionPP(ts,stage_time,Y[i],VecsSensiTemp,ts->vec_dir,ts->vecs_gpp);CHKERRQ(ierr);
        if (b[i] && quadts) {
          /* R_PU w_1 */
          ierr = TSComputeRHSHessianProductFunctionPU(quadts,stage_time,Y[i],NULL,ts->vec_sensip_col,ts->vecs_gpu);CHKERRQ(ierr);
          /* R_PP w_2 */
          ierr = TSComputeRHSHessianProductFunctionPP(quadts,stage_time,Y[i],NULL,ts->vec_dir,ts->vecs_gpp);CHKERRQ(ierr);
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
/*
static PetscErrorCode RKSetAdjCoe(RKTableau tab)
{
  PetscReal *A,*b;
  PetscInt        s,i,j;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  s    = tab->s;
  ierr = PetscMalloc2(s*s,&A,s,&b);CHKERRQ(ierr);

  for (i=0; i<s; i++)
    for (j=0; j<s; j++) {
      A[i*s+j] = (tab->b[s-1-i]==0) ? 0: -tab->A[s-1-i+(s-1-j)*s] * tab->b[s-1-j] / tab->b[s-1-i];
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Coefficients: A[%D][%D]=%.6f\n",i,j,A[i*s+j]);CHKERRQ(ierr);
    }

  for (i=0; i<s; i++) b[i] = (tab->b[s-1-i]==0)? 0: -tab->b[s-1-i];

  ierr  = PetscMemcpy(tab->A,A,s*s*sizeof(A[0]));CHKERRQ(ierr);
  ierr  = PetscMemcpy(tab->b,b,s*sizeof(b[0]));CHKERRQ(ierr);
  ierr  = PetscFree2(A,b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
*/

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
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Multirate methods options","");CHKERRQ(ierr);
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
    RKTableau tab  = rk->tableau;
    TSRKType  rktype;
    char      buf[512];
    ierr = TSRKGetType(ts,&rktype);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  RK type %s\n",rktype);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Order: %D\n",tab->order);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  FSAL property: %s\n",tab->FSAL ? "yes" : "no");CHKERRQ(ierr);
    ierr = PetscFormatRealArray(buf,sizeof(buf),"% 8.6f",tab->s,tab->c);CHKERRQ(ierr);
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

/*@C
  TSRKSetType - Set the type of RK scheme

  Logically collective

  Input Parameter:
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

  Logically collective

  Input Parameter:
.  ts - timestepping context

  Output Parameter:
.  rktype - type of RK-scheme

  Level: intermediate

.seealso: TSRKGetType()
@*/
PetscErrorCode TSRKGetType(TS ts,TSRKType *rktype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscUseMethod(ts,"TSRKGetType_C",(TS,TSRKType*),(ts,rktype));CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
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
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSRKGetType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSRKSetType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSRKSetMultirate_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSRKGetMultirate_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSRKSetMultirate - Use the interpolation-based multirate RK method

  Logically collective

  Input Parameter:
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

.seealso:  TSCreate(), TS, TSSetType(), TSRKSetType(), TSRKGetType(), TSRKSetFullyImplicit(), TSRK2D, TTSRK2E, TSRK3,
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

  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSRKGetType_C",TSRKGetType_RK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSRKSetType_C",TSRKSetType_RK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSRKSetMultirate_C",TSRKSetMultirate_RK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSRKGetMultirate_C",TSRKGetMultirate_RK);CHKERRQ(ierr);

  ierr = TSRKSetType(ts,TSRKDefault);CHKERRQ(ierr);
  rk->dtratio = 1;
  PetscFunctionReturn(0);
}
