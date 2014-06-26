//this js file contains methods copied from matt's original SAWs.js but modified to fit our needs

var divSave;//ignore this for now. I'm trying to get rid of the 1000ms delay

var successFunc = function(data, textStatus, jqXHR)//ignore this for now. I'm trying to get rid of the 1000ms delay
{
    console.log(data);
    jQuery(divSave).html("");
    window.setTimeout(PETSc.getAndDisplayDirectory,1000,null,divSave);
}


PETSc = {};

var sawsInfo = [];//this variable is used to organize all the data from SAWs

var init = false;//record if initialized the page (added appropriate divs for the diagrams and such)

var removedText = false;//record if the text at the top was removed

//This Function is called once (document).ready. The javascript for this was written by the PETSc code into index.html
PETSc.getAndDisplayDirectory = function(names,divEntry){

    if(!init) {
        $("head").append('<script src="js/parsePrefix2.js"></script>');//reuse the code for parsing thru the prefix
        $("head").append('<script src="js/fetchSawsData.js"></script>');//reuse the code for organizing data into sawsInfo
        $("head").append('<script src="js/utils.js"></script>');//necessary for the two js files above
        $("head").append('<script src="js/drawDiagrams.js"></script>');//contains the code to draw diagrams of the solver structure. in particular, fieldsplit and multigrid
        $("body").append("<div id=\"diagram\"></div>");
        init = true;
    }

    jQuery(divEntry).html("");
    SAWs.getDirectory(names,PETSc.displayDirectory,divEntry);
}

PETSc.displayDirectory = function(sub,divEntry)
{
    globaldirectory[divEntry] = sub;
    recordSawsData(sub);//records data into sawsInfo[]

    if (sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables._title.data == "Preconditioner (PC) options") {
        var SAWs_pcVal  = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-pc_type"].data[0];
        var SAWs_prefix = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["prefix"].data[0];

        if(SAWs_prefix == "(null)")
            SAWs_prefix = "";

        if(SAWs_prefix != "" && !removedText) {//remove the text at the top (the first 9 elements)
            for(var i=0; i<9; i++) {
                $("body").children().first().remove();
            }
            removedText=true;
        }

        var data = drawDiagrams("0",parsePrefix(SAWs_prefix).endtag,5,5);
        $("#diagram").html("<svg id=\"svgCanvas\" width='500' height='500' viewBox='0 0 1000 1000'></svg>");
        //IMPORTANT: Viewbox determines the coordinate system for drawing. width and height will rescale the SVG to the given width and height.
        $("#svgCanvas").html(data);
        $("body").html($("body").html()); //hacky refresh after appending to svg

    }

    PETSc.displayDirectoryRecursive(sub.directories,divEntry,0,"");//this method is recursive on itself and actually fills the div with text and dropdown lists

    if (sub.directories.SAWs_ROOT_DIRECTORY.variables.hasOwnProperty("__Block") && (sub.directories.SAWs_ROOT_DIRECTORY.variables.__Block.data[0] == "true")) {
        console.log("data fetched:");
        console.log(sub);
        jQuery(divEntry).append("<input type=\"button\" value=\"Continue\" id=\"continue\">");
        jQuery('#continue').on('click', function(){
            SAWs.updateDirectoryFromDisplay(divEntry);
            sub.directories.SAWs_ROOT_DIRECTORY.variables.__Block.data = ["false"];

            divSave = divEntry;
            //PETSc.postDirectory(sub, successFunc);//ignore this for now. I'm trying to get rid of 1000ms delay
            SAWs.postDirectory(sub);
            window.setTimeout(PETSc.getAndDisplayDirectory,1000,null,divEntry);
        });
    } else console.log("no block property or block property is false");
}

PETSc.postDirectory = function(directory, callback)//ignore this for now. I'm trying to get rid of the 1000ms delay
{
    var stringJSON = JSON.stringify(directory);
    jQuery.ajax({type: 'POST',dataType: 'json',url: '/SAWs/*',data: {input: stringJSON}, success: callback});
}

PETSc.displayDirectoryRecursive = function(sub,divEntry,tab,fullkey)
{
    jQuery.each(sub,function(key,value){
        fullkey = fullkey+key;//key contains things such as "PETSc" or "Options"
        if(jQuery("#"+fullkey).length == 0){
            jQuery(divEntry).append("<div id =\""+fullkey+"\"></div>")
            if (key != "SAWs_ROOT_DIRECTORY") {
                //SAWs.tab(fullkey,tab);
	        //jQuery("#"+fullkey).append("<b>"+ key +"<b><br>");//do not display "PETSc" nor "Options"
            }

            var descriptionSave = "";//saved description string because although the data is fetched: "description, -option, value" we wish to display it: "-option, value, description"
            var manualSave = ""; //saved manual text
            var mg_encountered = false;//record whether or not we have encountered pc=multigrid

            jQuery.each(sub[key].variables, function(vKey, vValue) {//for each variable...

                if (vKey.substring(0,2) == "__") // __Block variable
                    return;
                //SAWs.tab(fullkey,tab+1);
                if (vKey[0] != '_') {//this chunk of code adds the option name
                    if(vKey.indexOf("prefix") != -1 && sub[key].variables[vKey].data[0] == "(null)")
                        return;//do not display (null) prefix

                    if(vKey.indexOf("prefix") != -1) //prefix text
                        $("#"+fullkey).append(vKey + ":&nbsp;");
                    else if(vKey.indexOf("ChangedMethod") == -1) { //options text
                        //options text is a link to the appropriate manual page

                        var manualDirectory = "all"; //this directory does not exist yet so links will not work for now
                        $("#"+fullkey).append("<br><a href=\"http://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/" +  manualDirectory + "/" + manualSave + ".html\" title=\"" + descriptionSave + "\" id=\"data"+fullkey+vKey+j+"\">"+vKey+"&nbsp</a>");
                    }
                }

                for(j=0;j<sub[key].variables[vKey].data.length;j++){//vKey tells us a lot of information on what the data is. data.length is 1 most of the time. when it is more than 1, that results in 2 input boxes right next to each other

                    if(vKey.indexOf("man") != -1) {//do not display manual, but record the text
                        manualSave = sub[key].variables[vKey].data[j];
                        continue;
                    }

                    if(vKey.indexOf("title") != -1) {//display title in center
                        $("#"+fullkey).append("<center>"+"<span style=\"font-family: Courier\" size=\""+(sub[key].variables[vKey].data[j].toString().length+1)+"\" id=\"data"+fullkey+vKey+j+"\">"+sub[key].variables[vKey].data[j]+"</span>"+"</center>");
                        continue;
                    }

                    if(sub[key].variables[vKey].alternatives.length == 0) {//case where there are no alternatives
                        if(sub[key].variables[vKey].dtype == "SAWs_BOOLEAN") {
                            $("#"+fullkey).append("<select id=\"data"+fullkey+vKey+j+"\">");//make the boolean dropdown list.
                            $("#data"+fullkey+vKey+j).append("<option value=\"true\">True</option> <option value=\"false\">False</option>");
                            if(vKey == "ChangedMethod") {//do not show changedmethod to user
                                $("#data"+fullkey+vKey+j).attr("hidden",true);
                            }

                        } else {
                            if(sub[key].variables[vKey].mtype != "SAWs_WRITE") {

                                descriptionSave = sub[key].variables[vKey].data[j];

                                if(vKey.indexOf("prefix") != -1) //data of prefix so dont do manual and use immediately
                                    $("#"+fullkey).append("<a style=\"font-family: Courier\" size=\""+(sub[key].variables[vKey].data[j].toString().length+1)+"\" id=\"data"+fullkey+vKey+j+"\">"+sub[key].variables[vKey].data[j]+"</a><br>");

                            }
                            else {//can be changed (append dropdown list)
                                $("#"+fullkey).append("<input type=\"text\" style=\"font-family: Courier\" size=\""+(sub[key].variables[vKey].data[j].toString().length+1)+"\" id=\"data"+fullkey+vKey+j+"\" name=\"data\" \\>");
                            }
                            jQuery("#data"+fullkey+vKey+j).keyup(function(obj) {
                                console.log( "Key up called "+key+vKey );
                                sub[key].variables[vKey].selected = 1;
                                $("#data"+fullkey+"ChangedMethod0").find("option[value='true']").attr("selected","selected");//set changed to true automatically
                            });
                        }
                        jQuery("#data"+fullkey+vKey+j).val(sub[key].variables[vKey].data[j]);//set val from server
                        if(vKey != "ChangedMethod") {
                            jQuery("#data"+fullkey+vKey+j).change(function(obj) {
                                sub[key].variables[vKey].selected = 1;
                                $("#data"+fullkey+"ChangedMethod0").find("option[value='true']").attr("selected","selected");//set changed to true automatically
                            });
                        }
                    } else {//case where there are alternatives
                        jQuery("#"+fullkey).append("<select id=\"data"+fullkey+vKey+j+"\">");
                        jQuery("#data"+fullkey+vKey+j).append("<option value=\""+sub[key].variables[vKey].data[j]+"\">"+sub[key].variables[vKey].data[j]+"</option>");
                        for(var l=0;l<sub[key].variables[vKey].alternatives.length;l++) {
                            jQuery("#data"+fullkey+vKey+j).append("<option value=\""+sub[key].variables[vKey].alternatives[l]+"\">"+sub[key].variables[vKey].alternatives[l]+"</option>");
                        }
                        jQuery("#"+fullkey).append("</select>");

                        jQuery("#data"+fullkey+vKey+j).change(function(obj) {
                            sub[key].variables[vKey].selected = 1;
                            $("#data"+fullkey+"ChangedMethod0").find("option[value='true']").attr("selected","selected");//set changed to true automatically
                            var id = "data"+fullkey+vKey+j;
                            if(id.indexOf("type") != -1) {//if some type variable changed, then act as if continue button was clicked
                                $("#continue").trigger("click");
                            }
                        });
                    }
                }
            });

            if(typeof sub[key].directories != 'undefined'){
                PETSc.displayDirectoryRecursive(sub[key].directories,divEntry,tab+1,fullkey);
             }
        }
    });
}