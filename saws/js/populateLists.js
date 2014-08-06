//populate fieldsplit list
function populateFieldsplitList(endtag)
{
    var list = "#pc_fieldsplit_type" + endtag;
    $(list).append("<option value=\"multiplicative\">multiplicative</option>")
    $(list).append("<option value=\"additive\">additive</option>")
    $(list).append("<option value=\"symmetric_multiplicative\">symmetric_multiplicative</option>")
    $(list).append("<option value=\"special\">special</option>")
    $(list).append("<option value=\"schur\">schur</option>")
}

//populate mg list
function populateMgList(endtag)
{
    var list = "#pc_mg_type" + endtag;
    $(list).append("<option value=\"multiplicative\">multiplicative</option>")
    $(list).append("<option value=\"additive\">additive</option>")
    $(list).append("<option value=\"full\">full</option>")
    $(list).append("<option value=\"kaskade\">kaskade</option>")
}

//populate gamg list
function populateGamgList(endtag)
{
    var list = "#pc_gamg_type" + endtag;
    $(list).append("<option value=\"multiplicative\">multiplicative</option>")
    $(list).append("<option value=\"additive\">additive</option>")
    $(list).append("<option value=\"full\">full</option>")
    $(list).append("<option value=\"kaskade\">kaskade</option>")
}

/*
  populateKspList - populate the ksp list
  input
    endtag - string
*/
function populateKspList(endtag)
{
    var list = "#ksp_type" + endtag;

    //all options without parenthesis are for nonsymmetric (and, therefore, non pd) KSP list
    $(list).append("<option value=\"bcgs\">bcgs</option>");
    $(list).append("<option value=\"bcgsl\">bcgsl</option>");
    $(list).append("<option value=\"bicg\">bicg</option>");
    $(list).append("<option value=\"cg\">cg (symm, positive definite)</option>");
    $(list).append("<option value=\"cgne\">cgne</option>");
    $(list).append("<option value=\"cgs\">cgs</option>");
    $(list).append("<option value=\"chebyshev\">chebyshev</option>");
    $(list).append("<option value=\"cr\">cr</option>");
    $(list).append("<option value=\"fgmres\">fgmres</option>");
    $(list).append("<option value=\"gltr\">gltr</option>");
    $(list).append("<option value=\"gmres\">gmres</option>");
    $(list).append("<option value=\"groppcg\">groppcg</option>");
    $(list).append("<option value=\"lsqr\">lsqr</option>");
    $(list).append("<option value=\"minres\">minres (symm, non-positive definite)</option>");
    $(list).append("<option value=\"nash\">nash</option>");
    $(list).append("<option value=\"pgmres\">pgmres</option>");
    $(list).append("<option value=\"pipecg\">pipecg</option>");
    $(list).append("<option value=\"pipecr\">pipecr</option>");
    $(list).append("<option value=\"preonly\">preonly</option>");
    $(list).append("<option value=\"qcg\">qcg (symm, positive definite)</option>");
    $(list).append("<option value=\"richardson\">richardson</option>");
    $(list).append("<option value=\"specest\">specest</option>");
    $(list).append("<option value=\"stcg\">stcg</option>");
    $(list).append("<option value=\"symmlq\">symmlq (symm, non-positive definite)</option>");
    $(list).append("<option value=\"tcqmr\">tcqmr</option>");
    $(list).append("<option value=\"tfqmr\">tfqmr</option>");
}

/*
  populatePcList - populate the pc list
  input:
    endtag - string
*/
function populatePcList(endtag)
{
    var list="#pc_type" + endtag;

    $(list).append("<option value=\"asa\">asa</option>");
    $(list).append("<option value=\"asm\">asm</option>");
    $(list).append("<option value=\"bjacobi\">bjacobi</option>");
    $(list).append("<option value=\"cholesky\">cholesky</option>");
    $(list).append("<option value=\"composite\">composite</option>");
    $(list).append("<option value=\"cp\">cp</option>");
    $(list).append("<option value=\"eisenstat\">eisenstat</option>");
    $(list).append("<option value=\"exotic\">exotic</option>");
    $(list).append("<option value=\"fieldsplit\">fieldsplit (block structured)</option>");
    $(list).append("<option value=\"galerkin\">galerkin</option>");
    $(list).append("<option value=\"gamg\">gamg</option>");
    $(list).append("<option value=\"gasm\">gasm</option>");
    $(list).append("<option value=\"hmpi\">hmpi</option>");
    $(list).append("<option value=\"icc\">icc</option>");
    $(list).append("<option value=\"ilu\">ilu</option>");
    $(list).append("<option value=\"jacobi\">jacobi</option>");
    $(list).append("<option value=\"ksp\">ksp</option>");
    $(list).append("<option value=\"lsc\">lsc</option>");
    $(list).append("<option value=\"lu\">lu</option>");
    $(list).append("<option value=\"mat\">mat</option>");
    $(list).append("<option value=\"mg\">mg</option>");
    $(list).append("<option value=\"nn\">nn</option>");
    $(list).append("<option value=\"none\">none</option>");
    $(list).append("<option value=\"pbjacobi\">pbjacobi</option>");
    $(list).append("<option value=\"redistribute\">redistribute</option>");
    $(list).append("<option value=\"redundant\">redundant</option>");
    $(list).append("<option value=\"shell\">shell</option>");
    $(list).append("<option value=\"sor\">sor</option>");
    $(list).append("<option value=\"svd\">svd</option>");
}
