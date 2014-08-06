//this file contains all the default settings

//returns an object containing default pc_type, ksp_type, etc. enter an empty string for solver if working on the root solver
//IMPORTANT: USE SOLVER == "" TO GENERATE DEFAULT OPTIONS FOR A GIVEN SET OF PROPERTIES
function getDefaults(solver,symm,posdef,logstruc,child_symm,child_posdef,child_logstruc) {

    var ret = new Object();

    if(solver == "") { //case 1: user did not override default solver. we simply return the default pc_type and ksp_type for the given options. this should really only be used on the root solver since other solvers have parents and should call this method on the parent instead to generate a better default option (in that case, we would take the sub_pc_type instead of just pc_type)
        if(logstruc) {
            ret.pc_type = "fieldsplit";
        }
        if(symm) {
            if(!logstruc)
                ret.pc_type = "icc";

            if(!posdef) { //symm && !posdef
                ret.ksp_type = "minres";
            }
            else { //symm && posdef
                ret.ksp_type = "cg";
            }
        }
        else { //!symm
            if(!logstruc)
                ret.pc_type = "bjacobi";
            ret.ksp_type = "gmres";
        }
    }

    //case 2: user did indeed override default solver. generate the default options for the solver they selected (for example, pc_mg_blocks for pc_type=mg) and also the default sub_pc_type and sub_ksp_type for that solver (if any)

    else if(solver == "mg") {
        ret = {
            sub_pc_type: "sor",
            sub_ksp_type: "chebyshev",
            pc_mg_levels: 2,
            pc_mg_type: "multiplicative"
        };
        if(child_logstruc)
            ret.sub_pc_type = "fieldsplit";
    }

    else if(solver == "gamg") {
        ret = {
            sub_pc_type: "sor",
            sub_ksp_type: "chebyshev",
            pc_gamg_levels: 2,
            pc_gamg_type: "multiplicative"
        }
        if(child_logstruc)
            ret.sub_pc_type = "fieldsplit";
    }

    else if(solver == "fieldsplit") {
        ret = {
            sub_pc_type: "sor",
            sub_ksp_type: "chebyshev",
            pc_fieldsplit_blocks: 2,
            pc_fieldsplit_type: "multiplicative"
        };
        if(child_logstruc)
            ret.sub_pc_type = "fieldsplit";
    }

    else if(solver == "bjacobi") {
        ret = {
            pc_bjacobi_blocks: 2,
            sub_ksp_type: "preonly"
        };
        if(symm)
            ret.sub_pc_type = "icc";
        else
            ret.sub_pc_type = "ilu";

        if(child_logstruc)
            ret.sub_pc_type = "fieldsplit";
    }

    else if(solver == "asm") {
        ret = {
            pc_asm_blocks: 2,
            pc_asm_overlap: 2,
            sub_ksp_type: "preonly"
        };
        if(symm)
            ret.sub_pc_type = "icc";
        else
            ret.sub_pc_type = "ilu";

        if(child_logstruc)
            ret.sub_pc_type = "fieldsplit";
    }

    else if(solver == "redundant") {
        ret = {
            pc_redundant_number: 2,
            sub_ksp_type: "preonly"
        };
        if(symm)
            ret.sub_pc_type = "cholesky";
        else
            ret.sub_pc_type = "lu";

        if(child_logstruc)
            ret.sub_pc_type = "fieldsplit";
    }

    else if(solver == "ksp") { //note: this is for pc_type = ksp
        ret = {
            sub_pc_type: "bjacobi",
            sub_ksp_type: "gmres"
        };
        if(child_logstruc)
            ret.sub_pc_type = "fieldsplit";
    }

    return ret;
}
