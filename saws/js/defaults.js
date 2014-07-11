//this file contains all the default settings

//returns an object containing default pc_type, ksp_type, etc. enter an empty string for solver if working on the root solver
function getDefaults(solver,symm,posdef,logstruc) {

    var ret = new Object();

    if(solver == "") {
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

    else if(solver == "mg") {
        ret = {
            pc_type: "sor", //this would be the sub pc_type
            ksp_type: "chebyshev", //and this would be the sub ksp_type
            pc_mg_levels: 2,
            pc_mg_type: "multiplicative"
        };
    }

    else if(solver == "fieldsplit") {
        ret = {
            pc_type: "sor", //this would be the sub pc_type
            ksp_type: "chebyshev", //this would be the sub ksp_type
            pc_fieldsplit_blocks: 2,
            pc_fieldsplit_type: "multiplicative"
        };
    }

    else if(solver == "bjacobi") {
        ret = {
            pc_bjacobi_blocks: 2,
            ksp_type: "preonly"
        };
        if(symm)
            ret.pc_type = "icc";
        else
            ret.pc_type = "ilu";
    }

    else if(solver == "asm") {
        ret = {
            pc_asm_blocks: 2,
            pc_asm_overlap: 2,
            ksp_type: "preonly"
        };
        if(symm)
            ret.pc_type = "icc";
        else
            ret.pc_type = "ilu";
    }

    else if(solver == "redundant") {
        ret = {
            pc_redundant_number: 2,
            ksp_type: "preonly"
        };
        if(symm)
            ret.pc_type = "cholesky";
        else
            ret.pc_type = "lu";
    }

    else if(solver == "ksp") { //note: this is for pc_type = ksp
        ret = {
            pc_type: "bjacobi",
            ksp_type: "gmres"
        };
    }

    return ret;
}
