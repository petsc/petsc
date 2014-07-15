//this js file contains methods copied from matt's original SAWs.js but modified to fit our needs

PETSc = {};

var sawsInfo = [];//this variable is used to organize all the data from SAWs

var init = false;//record if initialized the page (added appropriate divs for the diagrams and such)

var iteration = 0;//record what iteration we are on (remove text on second iteration)

//This Function is called once (document).ready. The javascript for this was written by the PETSc code into index.html
PETSc.getAndDisplayDirectory = function(names,divEntry){

    if(!init) {
        $("head").append('<script src="js/parsePrefix.js"></script>');//reuse the code for parsing thru the prefix
        $("head").append('<script src="js/recordSawsData.js"></script>');//reuse the code for organizing data into sawsInfo
        $("head").append('<script src="js/utils.js"></script>');//necessary for the two js files above
        $("head").append('<script src="js/drawDiagrams.js"></script>');//contains the code to draw diagrams of the solver structure. in particular, fieldsplit and multigrid
        $("body").append("<div id=\"leftDiv\" style=\"float:left;\"></div>");
        $(divEntry).appendTo("#leftDiv");
        $("body").append("<div id=\"diagram\"></div>");
        init = true;
    }

    jQuery(divEntry).html("");
    SAWs.getDirectory(names,PETSc.displayDirectory,divEntry);
}

PETSc.displayDirectory = function(sub,divEntry)
{
    globaldirectory[divEntry] = sub;
    recordSawsData(sawsInfo,sub);//records data into sawsInfo[]

    iteration ++;
    if(iteration == 2) { //remove text
        for(var i=0; i<9; i++) {
            $("body").children().first().remove();
        }
    }

    if($("#leftDiv").children(0).is("center")) //remove the title of the options if needed
        $("#leftDiv").children().get(0).remove();

    if (sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables._title.data == "Preconditioner (PC) options") {
        var SAWs_pcVal  = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-pc_type"].data[0];
        var SAWs_prefix = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["prefix"].data[0];

        if(SAWs_prefix == "(null)")
            SAWs_prefix = "";

        $("#diagram").html("");
        var data = drawDiagrams(sawsInfo,"0",parsePrefix(sawsInfo,SAWs_prefix).endtag,5,5);

        if(data != "") {
            $("#diagram").html("<svg id=\"svgCanvas\" width='700' height='700' viewBox='0 0 2000 2000'>"+data+"</svg>");
            //IMPORTANT: Viewbox determines the coordinate system for drawing. width and height will rescale the SVG to the given width and height. Things should NEVER be appended to an svg element because then we would need to use a hacky refresh which works in Chrome, but no other browsers that I know of.
        }
    }

    PETSc.displayDirectoryRecursive(sub.directories,divEntry,0,"");//this method is recursive on itself and actually fills the div with text and dropdown lists

    if (sub.directories.SAWs_ROOT_DIRECTORY.variables.hasOwnProperty("__Block") && (sub.directories.SAWs_ROOT_DIRECTORY.variables.__Block.data[0] == "true")) {
        jQuery(divEntry).after("<input type=\"button\" value=\"Continue\" id=\"continue\">");
        $("#continue").after("<input type=\"button\" value=\"Finish\" id=\"finish\">");
        jQuery('#continue').on('click', function(){
            $("#continue").remove();//remove self immediately
            $("#finish").remove();
            SAWs.updateDirectoryFromDisplay(divEntry);
            sub.directories.SAWs_ROOT_DIRECTORY.variables.__Block.data = ["false"];
            SAWs.postDirectory(sub);
            window.setTimeout(PETSc.getAndDisplayDirectory,1000,null,divEntry);
        });
        jQuery('#finish').on('click', function(){
            $("#finish").remove();//remove self immediately
            $("#continue").remove();
            SAWs.updateDirectoryFromDisplay(divEntry);
            sub.directories.SAWs_ROOT_DIRECTORY.variables.__Block.data = ["false"];
            sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables.StopAsking.data = ["true"];//this is hardcoded (bad)
            SAWs.postDirectory(sub);
            window.setTimeout(PETSc.getAndDisplayDirectory,1000,null,divEntry);
        });
    } else console.log("no block property or block property is false");
}


/*
 * This function appends DOM elements to divEntry based on the JSON data in sub
 *
 */

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
                    else if(vKey.indexOf("ChangedMethod") == -1 && vKey.indexOf("StopAsking") == -1) { //options text
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
                        $("#"+"leftDiv").prepend("<center>"+"<span style=\"font-family: Courier\" size=\""+(sub[key].variables[vKey].data[j].toString().length+1)+"\" id=\"data"+fullkey+vKey+j+"\">"+sub[key].variables[vKey].data[j]+"</span>"+"</center>");//used to be ("#"+fullkey).append
                        continue;
                    }

                    if(sub[key].variables[vKey].alternatives.length == 0) {//case where there are no alternatives
                        if(sub[key].variables[vKey].dtype == "SAWs_BOOLEAN") {
                            $("#"+fullkey).append("<select id=\"data"+fullkey+vKey+j+"\">");//make the boolean dropdown list.
                            $("#data"+fullkey+vKey+j).append("<option value=\"true\">True</option> <option value=\"false\">False</option>");
                            if(vKey == "ChangedMethod" || vKey == "StopAsking") {//do not show changedmethod nor stopasking to user
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