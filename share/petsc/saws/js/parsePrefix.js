//this js file parses the prefix into the endtag format to store information in an array (allows for indefinite mg/fieldsplit nesting)
//requires the SAWs_prefix and the array of data that was already parsed

//parses the prefix and returns an object containing the proper endtag and the newly encountered fieldsplit word (if any)
function parsePrefix(data,SAWs_prefix) {

    var endtag  = "0";
    var newWord = "";

    while(SAWs_prefix != "") {//parse the entire prefix

        var indexFirstUnderscore = SAWs_prefix.indexOf("_");
        var chunk                = SAWs_prefix.substring(0,indexFirstUnderscore);//dont include the underscore

        if(chunk == "mg") {//mg_
            indexFirstUnderscore = SAWs_prefix.indexOf("_",3); //index of the second underscore
            chunk                = SAWs_prefix.substring(0,indexFirstUnderscore);//updated chunk

            //include more underscores here
            if(chunk == "mg_levels") {//need to include yet another underscore
                indexFirstUnderscore = SAWs_prefix.indexOf("_",10); //index of the third underscore
                chunk                = SAWs_prefix.substring(0,indexFirstUnderscore);//updated chunk
            }
            //otherwise, chunk == "mg_coarse" and we don't need to worry about it
        }

        else if(chunk == "fieldsplit") {//fieldsplit_

            var closest = SAWs_prefix.length;//the furthest a keyword could possibly be
            //find index of next keyword (pc, ksp, sub, smoothing, coarse). we have to do it this way because the name of the fieldsplit may include an underscore. for example, x_velocity_

            var keywords = ["pc","ksp","sub","redundant","mg","asm"];
            var loc      = SAWs_prefix.length;
            for(var i=0; i<keywords.length; i++) {
                loc = SAWs_prefix.indexOf(keywords[i]);
                if(loc < closest && loc != -1)
                    closest = loc;
            }

            var theword          = SAWs_prefix.substring(11,closest-1);//omit the first and last underscore
            var existingEndtag   = getEndtagByName(data, theword, endtag);//get the id (for example "001") associated with this fieldsplit word. need to pass in the existing endtag because we need to specify the parent of this fieldsplit

            if(existingEndtag == "-1") { //new fieldsplit. this word has not been encountered yet.
                var fieldsplitNumber = getNumChildren(data, endtag);//endtag = parent of this fieldsplit @TODO
                endtag               = endtag + "_" + fieldsplitNumber.toString();
                newWord              = theword;
            }

            else { //we have encountered this word before
                endtag = existingEndtag;
            }
        }

        SAWs_prefix = SAWs_prefix.substring(indexFirstUnderscore+1, SAWs_prefix.length);//dont include the first underscore

        if(chunk=="ksp" || chunk=="sub" || chunk=="mg_coarse" || chunk=="redundant")
            endtag += "_0";
        else if(chunk.substring(0,10)=="mg_levels_")
            endtag += "_" + chunk.substring(10,chunk.length);
    }

    var ret     = new Object();
    ret.endtag  = endtag;
    ret.newWord = newWord;

    return ret;
}
