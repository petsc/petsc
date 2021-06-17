#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 14:09:21 2020

@author: jacobfaibussowitsch
"""
import os

# need to situate ourselves and find 2 things, the _static directory and the location of
# petsc_team.rst. All the paths of the images need to be the relative paths from
# /path/to/petsc_team.rst and /path/to/_static/images/.

# do this instead of os.gcwd() since getcwd() returns the location this file was imported
# from not the location of this file
curFile = os.path.abspath(__file__)
curDir  = os.path.dirname(curFile)

petscTeamRstFile = os.path.join(curDir,"community","petsc_team.rst")
assert os.path.exists(petscTeamRstFile), "Could not locate 'petsc_team.rst' at {}".format(petscTeamRstFile)
petscTeamRstDir  = os.path.dirname(petscTeamRstFile)

# should be petsc/doc/_static
staticDir = os.path.join(curDir,"_static")
htmlDir   = os.path.join(staticDir,"html")
imDir     = os.path.join(staticDir,"images")

ownerURL      = "https://gitlab.com/api/v4/groups/petsc/members/all"
integratorURL = "https://gitlab.com/api/v4/groups/5583565/members/all"
devURL        = "https://gitlab.com/api/v4/groups/5981367/members/all"

def generateDevDicts(imRelDir):
  # Keys (i.e. usernames) here are just lower full names with underscores. Not really
  # needed, just so that I don't have to write an extra "writeRst" function :)
  emeritusCoreDevs = {
    "william_gropp" : {
      "web_url"    : "https://cs.illinois.edu/directory/profile/wgropp",
      "avatar_url" : os.path.join(imRelDir, "bill.gif"),
      "name"       : "William Gropp"
    },
    "victor_eijkhout" : {
      "web_url"    : "https://www.tacc.utexas.edu/staff/victor-eijkhout",
      "avatar_url" : os.path.join(imRelDir,"victor.jpg"),
      "name"       : "Victor Eijkhout"
    },
    "peter_brune" : {
      "web_url"    : "",
      "avatar_url" : os.path.join(imRelDir,"peter.jpg"),
      "name"       : "Peter Brune"
    },
    "kris_buschelman" : {
      "web_url"    : "",
      "avatar_url" : os.path.join(imRelDir,"buschelman.jpg"),
      "name"       : "Kris Buschelman"
    },
    "sean_farley" : {
      "web_url"    : "https://farley.io/",
      "avatar_url" : os.path.join(imRelDir,"sean.jpg"),
      "name"       : "Sean Farley"
    },
    "dmitry_karpeev" : {
      "web_url"    : "https://www.ci.uchicago.edu/profile/224",
      "avatar_url" : os.path.join(imRelDir,"dmitry.jpg"),
      "name"       : "Dmitry Karpeev"
    },
    "dinesh_kaushik" : {
      "web_url"    : "",
      "avatar_url" : os.path.join(imRelDir,"dinesh.jpg"),
      "name"       : "Dinesh Kaushik"
    },
    "jason_sarich" : {
      "web_url"    : "https://www.anl.gov/mcs/person/jason-sarich",
      "avatar_url" : os.path.join(imRelDir,"sarich.jpg"),
      "name"       : "Jason Sarich"
    },
    "victor_minden" : {
      "web_url"    : "",
      "avatar_url" : os.path.join(imRelDir,"victorminden.jpg"),
      "name"       : "Victor Minden"
    }
  }

  # Tuple of all current active devs' gitlab usernames, only the people whose corresponding
  # username is in this list will appear on the table. When we pull the list of PETSc
  # constributers fromm gitlab we get a list of anyone who has ever contributed a MR to
  # PETSc, thus we maintain this extra list to filter out the repeat-devs.
  activeCoreDevUsernames = ("lois.curfman.mcinnes","sbalay","jedbrown","adener","blaisebourdin",
                            "dmay","fdkong","ghammond","hannah_mairs","hongzhangsun",
                            "jfaibussowitsch","caidao22","jczhang07","karlrupp","markadams4",
                            "knepley","oanam198","psanan","rtmills","abhyshr","stefanozampini",
                            "tmunson","haplav","prj-","wence","tisaac","krugers","dalcinl",
                            "joseroman","bwhitchurch","barrysmith")
  activeCoreDevs = {d:{"web_url":None,"avatar_url":None,"name":None} for d in activeCoreDevUsernames}
  # Some people don't have the right profile picture (or any profile picture) on their
  # gitlab account, or perhaps want a different name or URL. So we apply this special
  # treatment here.
  activeCoreDevs["lois.curfman.mcinnes"]["web_url"]    = "https://press3.mcs.anl.gov/curfman/"
  activeCoreDevs["lois.curfman.mcinnes"]["avatar_url"] = os.path.join(imRelDir,"lois.gif")
  activeCoreDevs["lois.curfman.mcinnes"]["name"]       = "Lois Curfman McInnes"
  activeCoreDevs["dmay"]["avatar_url"]                 = os.path.join(imRelDir,"dave.jpg")
  activeCoreDevs["tmunson"]["avatar_url"]              = os.path.join(imRelDir,"todd.jpg")

  return emeritusCoreDevs,activeCoreDevs

def checkRstIncludes(coreFile,emeritusFile):
  __doc__ = """Check that the path to the files also corresponds to the path that petsc_team.rst expects to include"""
  import re

  reFindInclude   = re.compile("(..\s+include::\s*)(.*)")
  relCorePath     = os.path.relpath(coreFile,petscTeamRstDir)
  relEmeritusPath = os.path.relpath(emeritusFile,petscTeamRstDir)
  relPaths        = {relCorePath,relEmeritusPath}
  with open(petscTeamRstFile,"r") as f:
    line = f.readline()
    while line:
      reinc = reFindInclude.search(line)
      if reinc:
        assert reinc.group(2) in relPaths, "Include directive path in petsc_team.rst '{}' is not a valid relative path '{}'".format(reinc.group(2),relPaths)
      line = f.readline()
  return

def getJson(url,token):
  __doc__ = """Retrieve the JSON lists from GitLab via the API, requires a valid token and will raise if it is invalid"""
  import requests

  headers = {"PRIVATE-TOKEN" : token}
  params  = {"per_page" : 100, "page" : 1}
  rpage   = requests.get(url,headers = headers,params = params)
  rpage.raise_for_status()
  numPages = int(rpage.headers["X-Total-Pages"])
  lst = []
  for _ in range(numPages):
    r = requests.get(url,headers = headers,params = params)
    r.raise_for_status()
    lst.extend(r.json())
    params["page"] += 1
  return lst

def updateCoreDevs(devDict,devJson):
  __doc__ = """Goes through the supplied JSON, updating the activeCoreDevs dictionary. Respects existing values in activeCoreDevs so any special treatment applied above is not overwritten"""
  for dev in devJson:
    ldev = dev["username"].lower()
    if ldev in devDict:
      for key,val in devDict[ldev].items():
        if not val:
          devDict[ldev][key] = dev[key]
  return devDict

def writeRst(fname,fallbackFname,devs):
  __doc__ = """Write a processed dev dictionary as raw html RST to the outputfile"""
  try:
    petscDir = os.environ["PETSC_DIR"]
    # if PETSC_DIR is originally /path/to/petsc then petscDir is now /path/to, so that
    # relPath below has "petsc" in it
    petscDir = os.path.abspath(os.path.join(petscDir,os.path.pardir))
    # relPath = petsc/doc/genteamtable.py
    relPath  = os.path.relpath(__file__,petscDir)
  except KeyError:
    # "relPath" = genteamtable.py
    relPath = os.path.basename(__file__)
  lines = [".. raw:: html\n\n",
           "   <!-- Generated by %s -->\n" % (relPath),
           "   <div class=\"petsc-team-container\">\n",
           "   <style>\n",
           "     img.avatar {border-radius: 10px;width: 60px;height: 60px;}\n",
           "   </style>\n"]
  for dev in sorted(devs.items(),key = lambda item: item[1]["name"].split(" ")[-1]):
    lines.append("    <div>\n")
    lines.append("    <a href='%s'><img src='%s' class='avatar' /></a> <br />\n" % (dev[1]["web_url"],dev[1]["avatar_url"]))
    lines.append("    <p>%s</p>\n" % (dev[1]["name"]))
    lines.append("    </div>\n")
  lines.append("    </div>\n")
  with open(fname,"w") as f:
    f.writelines(lines)
    print("Wrote table to",fname)
  with open(fallbackFname,"w") as f:
    f.writelines(lines)
    print("Wrote fallback table to",fallbackFname)
  return

def main(writeDirPath,token=None,builderName=None,overwriteDirPath=htmlDir):
  __doc__ = """Main entry point for the file.

  Arguments:
  writeDirPath -- full path to directory where the generated .rst files will be

  Optional Arguments:
  token        -- GitLab private token, if none is given tries $PETSC_GITLAB_PRIVATE_TOKEN, if this also fails prints a diagnostic but exits returns cleanly. (default: None)
  builderName  -- type of sphinx builder, this may be required as any paths to images in the rst need to be relative to the writeDirPath otherwise they wont render, so we need to alter imRelDir. (default: None)
  """
  print(os.path.basename(__file__),"located at",curFile)
  currentFile  = os.path.join(writeDirPath,"petsc-team-table.inc")
  emeritusFile = os.path.join(writeDirPath,"petsc-emeritus-table.inc")
  # the regular version when built with "html" builder
  currentFileFallback  = os.path.join(overwriteDirPath,"petsc-team-table-html.inc")
  emeritusFileFallback = os.path.join(overwriteDirPath,"petsc-emeritus-table-html.inc")
  assert os.path.isfile(currentFileFallback),"Fallback current dev HTML table at {} missing or moved".format(currentFileFallback)
  assert os.path.isfile(emeritusFileFallback),"Fallback emeritus dev HTML table at {} missing or moved".format(emeritusFileFallback)

  checkRstIncludes(currentFile,emeritusFile)
  try:
    os.mkdir(writeDirPath)
    print("Generate directory created at",writeDirPath)
  except OSError as e:
    import errno
    if e.errno != errno.EEXIST:
      raise
  imRelDir = os.path.relpath(imDir,petscTeamRstDir)
  if builderName:
    # dirhtml makes it so every rst file is built as __file__/index.html, so we must
    # prepend ".." so image paths are correct
    print("Using builder",builderName)
    if builderName == "dirhtml":
      imRelDir = os.path.join(os.path.pardir,imRelDir)
      currentFileFallback  = os.path.join(overwriteDirPath,"petsc-team-table-dirhtml.inc")
      emeritusFileFallback = os.path.join(overwriteDirPath,"petsc-emeritus-table-dirhtml.inc")
      assert os.path.isfile(currentFileFallback),"Fallback current dev DIRHTML table at {} missing or moved".format(currentFileFallback)
      assert os.path.isfile(emeritusFileFallback),"Fallback emeritus dev DIRHTML table at {} missing or moved".format(emeritusFileFallback)
  if token is None:
    try:
      token = os.environ["PETSC_GITLAB_PRIVATE_TOKEN"]
    except KeyError:
      # no good token is available, so we fall back on the commited generated tables,
      # this will happen in CI pipelines for example
      import shutil

      print("Did not pass a valid token and $PETSC_GITLAB_PRIVATE_TOKEN was not defined in your environment. Falling back to committed (possibly outdated!) PETSc team tables.")
      shutil.copy(currentFileFallback,currentFile)
      shutil.copy(emeritusFileFallback,emeritusFile)
      print("\nCopied {} to {}\nCopied {} to {}".format(currentFileFallback,currentFile,emeritusFileFallback,emeritusFile))
      print("\nPlease generate a GitLab private token and export it to your environment. See https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html for more information.")
      return
  print("Image directory relative to",petscTeamRstFile,"at",imDir)
  devJson        = getJson(devURL,token)
  ownerJson      = getJson(ownerURL,token)
  integratorJson = getJson(integratorURL,token)
  emeritus,core  = generateDevDicts(imRelDir)
  core           = updateCoreDevs(core,devJson+ownerJson+integratorJson)
  writeRst(currentFile,currentFileFallback,core)
  writeRst(emeritusFile,emeritusFileFallback,emeritus)
  return

if __name__ == "__main__":
  import argparse,pathlib

  try:
    defaultToken = os.environ["PETSC_GITLAB_PRIVATE_TOKEN"]
  except KeyError:
    defaultToken = None
  string = "<string>"
  path   = "<path/to/dir>"
  parser = argparse.ArgumentParser(description="Build Developer Table",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("-o","--output-dir",required=True,metavar=path,type=pathlib.Path,help="Specify the output directory",dest="writeDir")
  parser.add_argument("--overwrite-committed-tables",default=htmlDir,metavar=path,type=pathlib.Path,help="Overwrite committed output tables at [PATH]",dest="overwritePath")
  parser.add_argument("-t","--gitlab-token",required=False,metavar=string,default=defaultToken,help="Specify your private Gitlab Authentican Token",dest="token")
  parser.add_argument("-b","--sphinx-builder",required=False,default="html",help="Sphinx builder type to simulate",dest="sphinxBuilder")
  args = parser.parse_args()
  writeDirPath = os.path.realpath(args.writeDir)
  print("============================================")
  print("  GENERATING TEAM TABLE FROM COMMAND LINE   ")
  print("============================================")
  main(writeDirPath,token=args.token,builderName=args.sphinxBuilder,overwriteDirPath=args.overwritePath)
