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

    }

    else if(solver == "asm") {

    }

    else if(solver == "redundant") {

    }

    else if(solver == "ksp") { //note: this is for pc_type = ksp

    }

    return ret;
}
