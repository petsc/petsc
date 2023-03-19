#include "chebyshevimpl.h"

/* given the polynomial order, return tabulated beta coefficients for use in opt. 4th-kind Chebyshev smoother */
PetscErrorCode KSPChebyshevGetBetas_Private(KSP ksp)
{
  const PetscInt       order = ksp->max_it;
  const KSP_Chebyshev *cheb  = (KSP_Chebyshev *)ksp->data;

  PetscFunctionBegin;
  PetscCheck(order >= 0 && order <= 16, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "Chebyshev polynomial order %" PetscInt_FMT " must be in [0, 16]", order);

  switch (order) {
  case 0:
    /* no-op */
    break;
  case 1:
    cheb->betas[0] = 1.12500000000000;
    break;
  case 2:
    cheb->betas[0] = 1.02387287570313;
    cheb->betas[1] = 1.26408905371085;
    break;
  case 3:
    cheb->betas[0] = 1.00842544782028;
    cheb->betas[1] = 1.08867839208730;
    cheb->betas[2] = 1.33753125909618;
    break;
  case 4:
    cheb->betas[0] = 1.00391310427285;
    cheb->betas[1] = 1.04035811188593;
    cheb->betas[2] = 1.14863498546254;
    cheb->betas[3] = 1.38268869241000;
    break;
  case 5:
    cheb->betas[0] = 1.00212930146164;
    cheb->betas[1] = 1.02173711549260;
    cheb->betas[2] = 1.07872433192603;
    cheb->betas[3] = 1.19810065292663;
    cheb->betas[4] = 1.41322542791682;
    break;
  case 6:
    cheb->betas[0] = 1.00128517255940;
    cheb->betas[1] = 1.01304293035233;
    cheb->betas[2] = 1.04678215124113;
    cheb->betas[3] = 1.11616489419675;
    cheb->betas[4] = 1.23829020218444;
    cheb->betas[5] = 1.43524297106744;
    break;
  case 7:
    cheb->betas[0] = 1.00083464397912;
    cheb->betas[1] = 1.00843949430122;
    cheb->betas[2] = 1.03008707768713;
    cheb->betas[3] = 1.07408384092003;
    cheb->betas[4] = 1.15036186707366;
    cheb->betas[5] = 1.27116474046139;
    cheb->betas[6] = 1.45186658649364;
    break;
  case 8:
    cheb->betas[0] = 1.00057246631197;
    cheb->betas[1] = 1.00577427662415;
    cheb->betas[2] = 1.02050187922941;
    cheb->betas[3] = 1.05019803444565;
    cheb->betas[4] = 1.10115572984941;
    cheb->betas[5] = 1.18086042806856;
    cheb->betas[6] = 1.29838585382576;
    cheb->betas[7] = 1.46486073151099;
    break;
  case 9:
    cheb->betas[0] = 1.00040960072832;
    cheb->betas[1] = 1.00412439506106;
    cheb->betas[2] = 1.01460212148266;
    cheb->betas[3] = 1.03561113626671;
    cheb->betas[4] = 1.07139972529194;
    cheb->betas[5] = 1.12688273710962;
    cheb->betas[6] = 1.20785219140729;
    cheb->betas[7] = 1.32121930716746;
    cheb->betas[8] = 1.47529642820699;
    break;
  case 10:
    cheb->betas[0] = 1.00030312229652;
    cheb->betas[1] = 1.00304840660796;
    cheb->betas[2] = 1.01077022715387;
    cheb->betas[3] = 1.02619011597640;
    cheb->betas[4] = 1.05231724933755;
    cheb->betas[5] = 1.09255743207549;
    cheb->betas[6] = 1.15083376663972;
    cheb->betas[7] = 1.23172250870894;
    cheb->betas[8] = 1.34060802024460;
    cheb->betas[9] = 1.48386124407011;
    break;
  case 11:
    cheb->betas[0]  = 1.00023058595209;
    cheb->betas[1]  = 1.00231675024028;
    cheb->betas[2]  = 1.00817245396304;
    cheb->betas[3]  = 1.01982986566342;
    cheb->betas[4]  = 1.03950210235324;
    cheb->betas[5]  = 1.06965042700541;
    cheb->betas[6]  = 1.11305754295742;
    cheb->betas[7]  = 1.17290876275564;
    cheb->betas[8]  = 1.25288300576792;
    cheb->betas[9]  = 1.35725579919519;
    cheb->betas[10] = 1.49101672564139;
    break;
  case 12:
    cheb->betas[0]  = 1.00017947200828;
    cheb->betas[1]  = 1.00180189139619;
    cheb->betas[2]  = 1.00634861907307;
    cheb->betas[3]  = 1.01537864566306;
    cheb->betas[4]  = 1.03056942830760;
    cheb->betas[5]  = 1.05376019693943;
    cheb->betas[6]  = 1.08699862592072;
    cheb->betas[7]  = 1.13259183097913;
    cheb->betas[8]  = 1.19316273358172;
    cheb->betas[9]  = 1.27171293675110;
    cheb->betas[10] = 1.37169337969799;
    cheb->betas[11] = 1.49708418575562;
    break;
  case 13:
    cheb->betas[0]  = 1.00014241921559;
    cheb->betas[1]  = 1.00142906932629;
    cheb->betas[2]  = 1.00503028986298;
    cheb->betas[3]  = 1.01216910518495;
    cheb->betas[4]  = 1.02414874342792;
    cheb->betas[5]  = 1.04238158880820;
    cheb->betas[6]  = 1.06842008128700;
    cheb->betas[7]  = 1.10399010936759;
    cheb->betas[8]  = 1.15102748242645;
    cheb->betas[9]  = 1.21171811910125;
    cheb->betas[10] = 1.28854264865128;
    cheb->betas[11] = 1.38432619380991;
    cheb->betas[12] = 1.50229418757368;
    break;
  case 14:
    cheb->betas[0]  = 1.00011490538261;
    cheb->betas[1]  = 1.00115246376914;
    cheb->betas[2]  = 1.00405357333264;
    cheb->betas[3]  = 1.00979590573153;
    cheb->betas[4]  = 1.01941300472994;
    cheb->betas[5]  = 1.03401425035436;
    cheb->betas[6]  = 1.05480599606629;
    cheb->betas[7]  = 1.08311420301813;
    cheb->betas[8]  = 1.12040891660892;
    cheb->betas[9]  = 1.16833095655446;
    cheb->betas[10] = 1.22872122288238;
    cheb->betas[11] = 1.30365305707817;
    cheb->betas[12] = 1.39546814053678;
    cheb->betas[13] = 1.50681646209583;
    break;
  case 15:
    cheb->betas[0]  = 1.00009404750752;
    cheb->betas[1]  = 1.00094291696343;
    cheb->betas[2]  = 1.00331449056444;
    cheb->betas[3]  = 1.00800294833816;
    cheb->betas[4]  = 1.01584236259140;
    cheb->betas[5]  = 1.02772083317705;
    cheb->betas[6]  = 1.04459535422831;
    cheb->betas[7]  = 1.06750761206125;
    cheb->betas[8]  = 1.09760092545889;
    cheb->betas[9]  = 1.13613855366157;
    cheb->betas[10] = 1.18452361426236;
    cheb->betas[11] = 1.24432087304475;
    cheb->betas[12] = 1.31728069083392;
    cheb->betas[13] = 1.40536543893560;
    cheb->betas[14] = 1.51077872501845;
    break;
  case 16:
    cheb->betas[0]  = 1.00007794828179;
    cheb->betas[1]  = 1.00078126847253;
    cheb->betas[2]  = 1.00274487974401;
    cheb->betas[3]  = 1.00662291017015;
    cheb->betas[4]  = 1.01309858836971;
    cheb->betas[5]  = 1.02289448329337;
    cheb->betas[6]  = 1.03678321409983;
    cheb->betas[7]  = 1.05559875719896;
    cheb->betas[8]  = 1.08024848405560;
    cheb->betas[9]  = 1.11172607131497;
    cheb->betas[10] = 1.15112543431072;
    cheb->betas[11] = 1.19965584614973;
    cheb->betas[12] = 1.25865841744946;
    cheb->betas[13] = 1.32962412656664;
    cheb->betas[14] = 1.41421360695576;
    cheb->betas[15] = 1.51427891730346;
    break;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
