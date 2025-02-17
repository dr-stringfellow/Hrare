import ROOT
import os
import json
from subprocess import call,check_output
import fnmatch
from correctionlib import _core

if "/functions.so" not in ROOT.gSystem.GetLibraries():
    ROOT.gSystem.CompileMacro("functions.cc","k")

def loadCorrectionSet(type,year):
    # Load CorrectionSet#

    fname = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/"
    if type=='MUO':
        fname += "MUO/"+year+"_UL/muon_Z.json.gz"
    if type=='ELE':
        fname += "EGM/"+year+"_UL/electron.json.gz"
    if type=='PH':
        fname += "EGM/"+year+"_UL/photon.json.gz"

    if fname.endswith(".json.gz"):
        import gzip
        with gzip.open(fname,'rt') as file:
            data = file.read().strip()
            evaluator = _core.CorrectionSet.from_string(data)
    else:
        evaluator = _core.CorrectionSet.from_file(fname)

def loadJSON(fIn):

    if not os.path.isfile(fIn):
        print("JSON file %s does not exist" % fIn)
        return

    if not hasattr(ROOT, "jsonMap"):
        print("jsonMap not found in ROOT dict")
        return

    info = json.load(open(fIn))
    print("JSON file %s loaded" % fIn)
    for k,v in info.items():

        vec = ROOT.std.vector["std::pair<unsigned int, unsigned int>"]()
        for combo in v:
            pair = ROOT.std.pair["unsigned int", "unsigned int"](*[int(c) for c in combo])
            vec.push_back(pair)
            ROOT.jsonMap[int(k)] = vec

def findDataset(name):

    DASclient = "dasgoclient -query '%(query)s'"
    cmd= DASclient%{'query':'file dataset=%s'%name}
    print(cmd)
    check_output(cmd,shell=True)
    fileList=[ 'root://xrootd-cms.infn.it//'+x for x in check_output(cmd,shell=True).split() ]

    files_ROOT = ROOT.vector('string')()
    for f in fileList: files_ROOT.push_back(f)

    return files_ROOT

def findDIR(directory):

    print(directory)

    counter = 0
    rootFiles = ROOT.vector('string')()
    for root, directories, filenames in os.walk(directory):
        for f in filenames:

            counter+=1
            filePath = os.path.join(os.path.abspath(root), f)
            if "failed/" in filePath: continue
            if "log/" in filePath: continue
            rootFiles.push_back(filePath)
#            if counter>100: break
#            if counter>50: break
#            if counter>5: break

    return rootFiles

def findMany(basedir, regex):

    if basedir[-1] == "/": basedir = basedir[:-1]
    regex = basedir + "/" + regex

    rootFiles = ROOT.vector('string')()
    for root, directories, filenames in os.walk(basedir):

        for f in filenames:

            filePath = os.path.join(os.path.abspath(root), f)
            if "failed/" in filePath: continue
            if "log/" in filePath: continue
            if fnmatch.fnmatch(filePath, regex): rootFiles.push_back(filePath)

    return rootFiles

def concatenate(result, tmp1):
    for f in tmp1:
        result.push_back(f)


def getMClist(year,sampleNOW):

    files = findDIR("{}".format(SwitchSample(sampleNOW)[0]))
    if sampleNOW==0:
            files1 = findDIR("{}".format(SwitchSample(1000)[0]))
            concatenate(files, files1)
    return files

def getDATAlist(year,type):

    if(year == 2018):
        loadJSON("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/Legacy_2018/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt")
    if(year == 2017):
        loadJSON("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt")
    if(year == 2016 or year == 12016):
        loadJSON("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/Legacy_2016/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt")

    if(year == 2018 and type == -1):
        files = findMany("/mnt/T2_US_MIT/hadoop/cms/store/user/paus/nanohr/D01/","SingleMuon+Run2018A*")
    if(year == 2018 and type == -2):
        files = findMany("/mnt/T2_US_MIT/hadoop/cms/store/user/paus/nanohr/D01/","SingleMuon+Run2018B*")
    if(year == 2018 and type == -3):
        files = findMany("/mnt/T2_US_MIT/hadoop/cms/store/user/paus/nanohr/D01/","SingleMuon+Run2018C*")
    if(year == 2018 and type == -4):
        files = findMany("/mnt/T2_US_MIT/hadoop/cms/store/user/paus/nanohr/D01/","SingleMuon+Run2018D*")

    if(year == 2018 and type == -31):
        files = findMany("/mnt/T2_US_MIT/hadoop/cms/store/user/paus/nanohr/D01/","EGamma+Run2018A*")
    if(year == 2018 and type == -32):
        files = findMany("/mnt/T2_US_MIT/hadoop/cms/store/user/paus/nanohr/D01/","EGamma+Run2018B*")
    if(year == 2018 and type == -33):
        files = findMany("/mnt/T2_US_MIT/hadoop/cms/store/user/paus/nanohr/D01/","EGamma+Run2018C*")
    if(year == 2018 and type == -34):
        files = findMany("/mnt/T2_US_MIT/hadoop/cms/store/user/paus/nanohr/D01/","EGamma+Run2018D*")

    return files

def SwitchSample(argument):

    # cross section from  https://cms-gen-dev.cern.ch/xsdb
    dirT2 = "/mnt/T2_US_MIT/hadoop/cms/store/user/paus/nanohr/D01/"
    dirLocal = "/work/submit/mariadlf/Hrare/D01/2018/"

    switch = {
        1000: (dirT2+"DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM",6067*1000), #NNLO
        0: (dirT2+"DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1_ext1-v1+MINIAODSIM",6067*1000), #NNLO
        1: (dirT2+"ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM", 51.1*1000), #LO
        2: (dirT2+"WGToLNuG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM", 191.0*1000), #LO
        3: (dirT2+"WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",53870.0*1000), #LO
        4: (dirT2+"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",88.2*1000), #NNLO
        5: (dirT2+"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM",365.3452*1000), #NNLO
##
        6: (dirT2+"GJets_DR-0p4_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",5031*1000), #LO
        7: (dirT2+"GJets_DR-0p4_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",1126*1000), #LO
        8: (dirT2+"GJets_DR-0p4_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",124.3*1000), #LO
        9: (dirT2+"GJets_DR-0p4_HT-600ToInf_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",40.76*1000), #LO
        20: (dirT2+"QCD_Pt-30to50_EMEnriched_TuneCP5_13TeV-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM",6447000.0*1000), #LO
        21: (dirT2+"QCD_Pt-50to80_EMEnriched_TuneCP5_13TeV-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM",1988000.0*1000), #LO
        22: (dirT2+"QCD_Pt-80to120_EMEnriched_TuneCP5_13TeV-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM",367500.0*1000), #LO
        23: (dirT2+"QCD_Pt-120to170_EMEnriched_TuneCP5_13TeV-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM",66590.0*1000), #LO
        24: (dirT2+"QCD_Pt-170to300_EMEnriched_TuneCP5_13TeV-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM",16620.0*1000), #LO
        25: (dirT2+"QCD_Pt-300toInf_EMEnriched_TuneCP5_13TeV-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM",1104.0*1000), #LO
##
        31: (dirT2+"WJetsToLNu_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM",53330.0*1000), #LO
        32: (dirT2+"WJetsToLNu_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",8875.0*1000), #LO
        33: (dirT2+"WJetsToLNu_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM",3338.0*1000), #LO

        34: (dirT2+"DYJetsToLL_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",5129.0*1000), #LO
        35: (dirT2+"DYJetsToLL_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",951.5*1000), #LO
        36: (dirT2+"DYJetsToLL_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",361.4*1000), #LO
#### signal
        100: (dirLocal+"vbf-hrhogamma-powheg",4.*1000), # xsec = 4pb * BR(Hphigamma)=1 BR(rho->pipi)=1.
        101: (dirLocal+"vbf-hphigamma-powheg",2.*1000), # xsec = 4pb * BR(Hphigamma)=1 BR(phi->kk)=0.49
    }
    return switch.get(argument, "BKGdefault, xsecDefault")


def computeWeigths(df, files, sampleNOW, isMC):

    nevents = df.Count().GetValue()
    print("%s entries in the dataset" %nevents)

    if not isMC:
        return 1.
    else:
        rdf = ROOT.RDataFrame("Runs", files)
        genEventSumWeight = rdf.Sum("genEventSumw").GetValue()
        genEventSumNoWeight = rdf.Sum("genEventCount").GetValue()

        weight = (SwitchSample(sampleNOW)[1] / genEventSumWeight)
        weightApprox = (SwitchSample(sampleNOW)[1] / genEventSumNoWeight)
        print('weight',weight )
        print('weightApprox',weightApprox)
        lumiEq = (genEventSumNoWeight / SwitchSample(sampleNOW)[1])
        print("lumi equivalent fb %s" %lumiEq)

        return weightApprox ## later with negative weights

def plot(h,filename,doLogX,color):

   ROOT.gStyle.SetOptStat(1);
   ROOT.gStyle.SetTextFont(42)
   c = ROOT.TCanvas("c", "", 800, 700)
   if doLogX: c.SetLogx();
#  c.SetLogy()

   h.SetTitle("")
   h.GetXaxis().SetTitleSize(0.04)
   h.GetYaxis().SetTitleSize(0.04)
   h.SetLineColor(color)

   h.Draw()

   label = ROOT.TLatex(); label.SetNDC(True)
   label.DrawLatex(0.175, 0.740, "#eta")
   label.DrawLatex(0.205, 0.775, "#rho,#omega")
   label.DrawLatex(0.270, 0.740, "#phi")
   label.SetTextSize(0.040); label.DrawLatex(0.100, 0.920, "#bf{CMS Simulation}")
   label.SetTextSize(0.030); label.DrawLatex(0.630, 0.920, "#sqrt{s} = 13 TeV, L_{int} = X fb^{-1}")

   print("saving file: {} ".format(filename))

   c.SaveAs(filename)
