from PhysicsTools.NanoAOD.common_cff import *
import FWCore.ParameterSet.Config as cms

# NOTE: 
#    All instances of FlatTableProducers must end with Table in their
#    names so that their product match keep patterns in the default
#    event content. Otherwise you need to modify outputCommands in
#    NanoAODEDMEventContent or provide a custom event content to the
#    output module

def merge_psets(*argv):
    result = cms.PSet()
    for pset in argv:
        if isinstance(pset, cms._Parameterizable):
            for name in pset.parameters_().keys():
                value = getattr(pset,name)
                type = value.pythonTypeName()
                setattr(result,name,value)
    return result

V0prod = cms.EDProducer(
    "MesonProducer",
    beamSpot=cms.InputTag("offlineBeamSpot"),
    vertexCollection=cms.InputTag("offlineSlimmedPrimaryVertices"),
    muonCollection = cms.InputTag("linkedObjects","muons"),
    PFCandCollection = cms.InputTag("packedPFCandidates"),
    packedGenParticleCollection = cms.InputTag("packedGenParticles"),
    minMuonPt  = cms.double(3.5),
    maxMuonEta = cms.double(1.4),
    minPionPt  = cms.double(1.0),
    maxPionEta = cms.double(2.4),
    minKsMass  = cms.double(0.45),
    maxKsMass  = cms.double(0.55),
    minKsPreselectMass = cms.double(0.4),
    maxKsPreselectMass = cms.double(0.6),
    minPhiMass  = cms.double(1.00), # rho true mass 1020
    maxPhiMass  = cms.double(1.04),
    minPhiPreselectMass = cms.double(0.9),
    maxPhiPreselectMass = cms.double(1.1),
    minRhosPreselectMass = cms.double(0.4),
    maxRhosPreselectMass = cms.double(1.1),
    minRhosMass = cms.double(0.5), # rho true mass 770
    maxRhosMass = cms.double(1.),
    minOmegasPreselectMass = cms.double(0.3),
    maxOmegasPreselectMass = cms.double(1.65),
    minDsMass  = cms.double(1.91),
    maxDsMass  = cms.double(2.03),
    minDsPreselectMass = cms.double(1.8),
    maxDsPreselectMass = cms.double(2.1),
    minD0Mass  = cms.double(1.8),
    maxD0Mass  = cms.double(1.9),
    minD0PreselectMass = cms.double(1.6),
    maxD0PreselectMass = cms.double(2.0),
    minLambdaMass  = cms.double(1.05),
    maxLambdaMass  = cms.double(1.15),
    minLambdaPreselectMass = cms.double(1.0),
    maxLambdaPreselectMass = cms.double(1.2),
    maxTwoTrackDOCA = cms.double(0.1),
    maxLxy = cms.double(999),
    minSigLxy = cms.double(5),
    minVtxProb = cms.double(0.001),
    minCosAlpha = cms.double(0.9),
    minDisplaceTrackSignificance = cms.double(1),
    isMC = cms.bool(False)
)

V0prodMC = V0prod.clone( isMC = cms.bool(True) )

# KsToPiPi

KsVariables = cms.PSet(
    mass         = Var("mass",                         float, doc = "Unfit invariant mass"),
    doca         = Var("userFloat('doca')",            float, doc = "Distance of closest approach of tracks"),
    iso          = Var("userFloat('iso')",             float, doc = "tracks isolation (pt/(pt+sum))"),
    trk1_pt      = Var("userFloat('trk1_pt')",         float, doc = "Track 1 pt"),
    trk1_eta     = Var("userFloat('trk1_eta')",        float, doc = "Track 1 eta"),
    trk1_phi     = Var("userFloat('trk1_phi')",        float, doc = "Track 1 phi"),
#    trk1_mu_index = Var("userInt('trk1_mu_index')",      int, doc = "Matched muon index for track 1"),
    trk2_pt      = Var("userFloat('trk2_pt')",         float, doc = "Track 2 pt"),
    trk2_eta     = Var("userFloat('trk2_eta')",        float, doc = "Track 2 eta"),
    trk2_phi     = Var("userFloat('trk2_phi')",        float, doc = "Track 2 phi"),
#    trk2_mu_index = Var("userInt('trk2_mu_index')",      int, doc = "Matched muon index for track 2"),
    trk1_sip     = Var("userFloat('trk1_sip')",        float, doc = "Track 1 2D impact parameter significance wrt Beam Spot"),
    trk2_sip     = Var("userFloat('trk2_sip')",        float, doc = "Track 2 2D impact parameter significance wrt Beam Spot"),
    kin_valid    = Var("userInt('kin_valid')",         int,   doc = "Kinematic fit: vertex validity"),
    kin_vtx_prob = Var("userFloat('kin_vtx_prob')",    float, doc = "Kinematic fit: vertex probability"),
    kin_vtx_chi2dof = Var("userFloat('kin_vtx_chi2dof')", float, doc = "Kinematic fit: vertex normalized Chi^2"),
    kin_mass     = Var("userFloat('kin_mass')",        float, doc = "Kinematic fit: vertex refitted mass"),
    kin_pt       = Var("userFloat('kin_pt')",          float, doc = "Kinematic fit: vertex refitted pt"),
    kin_eta      = Var("userFloat('kin_eta')",         float, doc = "Kinematic fit: vertex refitted eta"),
    kin_phi      = Var("userFloat('kin_phi')",         float, doc = "Kinematic fit: vertex refitted phi"),
    kin_massErr  = Var("userFloat('kin_massErr')",     float, doc = "Kinematic fit: vertex refitted mass error"),
    kin_lxy      = Var("userFloat('kin_lxy')",         float, doc = "Kinematic fit: vertex displacement in XY plane wrt Beam Spot"),
    kin_slxy     = Var("userFloat('kin_sigLxy')",      float, doc = "Kinematic fit: vertex displacement significance in XY plane wrt Beam Spot"),
    kin_cosAlphaXY = Var("userFloat('kin_cosAlphaXY')",    float, doc = "Kinematic fit: cosine of pointing angle in XY wrt BS"),
    kin_sipBS    = Var("userFloat('kin_sipBS')",       float, doc = "Kinematic fit: impact parameter significance of the candidate trajectory in XY wrt BS"),
    kin_sipPV    = Var("userFloat('kin_sipPV')",       float, doc = "Kinematic fit: impact parameter significance of the candidate trajectory in 3D wrt PV"),
)

KsVariablesMC = merge_psets(
    KsVariables,
    cms.PSet(
        gen_trk1_pdgId  = Var("userInt(  'gen_trk1_pdgId')",    int,   doc = "Gen match: first track pdg Id"),
        gen_trk1_mpdgId = Var("userInt(  'gen_trk1_mpdgId')",   int,   doc = "Gen match: first track mother pdg Id"),
        gen_trk1_pt     = Var("userFloat('gen_trk1_pt')",     float,   doc = "Gen match: first track pt"),
        gen_trk2_pdgId  = Var("userInt(  'gen_trk2_pdgId')",    int,   doc = "Gen match: second track pdg Id"),
        gen_trk2_mpdgId = Var("userInt(  'gen_trk2_mpdgId')",   int,   doc = "Gen match: second track mother pdg Id"),
        gen_trk2_pt     = Var("userFloat('gen_trk2_pt')",     float,   doc = "Gen match: second track pt"),
        gen_pdgId       = Var("userInt(  'gen_pdgId')",         int,   doc = "Gen match: ditrack pdg Id"),
        gen_mass        = Var("userFloat('gen_mass')",        float,   doc = "Gen match: ditrack mass"),
        gen_pt          = Var("userFloat('gen_pt')",          float,   doc = "Gen match: ditrack pt"),
        ),
)

KsTable=cms.EDProducer("SimpleCompositeCandidateFlatTableProducer",
    src=cms.InputTag("V0prod","Ks"),
    cut=cms.string(""),
    name=cms.string("ks"),
    doc=cms.string("Ks Variables"),
    singleton=cms.bool(False),
    extension=cms.bool(False),
    variables = KsVariables
)

KsMcTable=cms.EDProducer("SimpleCompositeCandidateFlatTableProducer",
    src=cms.InputTag("V0prodMC","Ks"),
    cut=cms.string(""),
    name=cms.string("ks"),
    doc=cms.string("Ks Variables"),
    singleton=cms.bool(False),
    extension=cms.bool(False),
    variables = KsVariablesMC
)

RhosTable=cms.EDProducer("SimpleCompositeCandidateFlatTableProducer",
    src=cms.InputTag("V0prod","Rho"),
    cut=cms.string(""),
    name=cms.string("rho"),
    doc=cms.string("Rhos Variables"),
    singleton=cms.bool(False),
    extension=cms.bool(False),
    variables = KsVariables # for now same variables rho, k to pipi
)

RhosMcTable=cms.EDProducer("SimpleCompositeCandidateFlatTableProducer",
    src=cms.InputTag("V0prodMC","Rho"),
    cut=cms.string(""),
    name=cms.string("rho"),
    doc=cms.string("Rhos Variables"),
    singleton=cms.bool(False),
    extension=cms.bool(False),
    variables = KsVariablesMC # for now same variables rho, k to pipi
)


# Omega ToPiPi + Pi0
OmegasVariables = merge_psets(
    KsVariables,
    cms.PSet(
        photon_pt     = Var("userFloat('photon_pt')",    float,   doc = " pt of the photon proxy of pi0"),
        photon_eta    = Var("userFloat('photon_eta')",   float,   doc = " eta of the photon proxy of pi0"),
        photon_phi    = Var("userFloat('photon_phi')",   float,   doc = " phi of the photon proxy of pi0"),
        photon_pdgId  = Var("userInt('photon_pdgId')",   int,     doc = " pdgId of the photon proxy of pi0"),
        Nphotons      = Var("userInt('Nphotons')",       int,     doc = " numbers of the photon proxy of pi0"),
        threemass     = Var("userFloat('3body_mass')",   float,   doc = "mass of 3 pions"),
    )
)

OmegasVariablesMC = merge_psets(
    KsVariablesMC,
    cms.PSet(
        photon_pt     = Var("userFloat('photon_pt')",   float,   doc = " pt of the photon proxy of pi0"),
        photon_eta    = Var("userFloat('photon_eta')",  float,   doc = " eta of the photon proxy of pi0"),
        photon_phi    = Var("userFloat('photon_phi')",  float,   doc = " phi of the photon proxy of pi0"),
        photon_pdgId  = Var("userInt('photon_pdgId')",  int,     doc = " pdgId of the photon proxy of pi0"),
        Nphotons      = Var("userInt('Nphotons')",      int,     doc = " numbers of the photon proxy of pi0"),
        threemass     = Var("userFloat('3body_mass')",  float,   doc = "mass of 3 pions"),
    )
)

OmegasTable=cms.EDProducer("SimpleCompositeCandidateFlatTableProducer",
    src=cms.InputTag("V0prod","Omega"),
    cut=cms.string(""),
    name=cms.string("omega"),
    doc=cms.string("Omegas Variables"),
    singleton=cms.bool(False),
    extension=cms.bool(False),
    variables = OmegasVariables # for now same variables rho, k to pipi
)

OmegasMcTable=cms.EDProducer("SimpleCompositeCandidateFlatTableProducer",
    src=cms.InputTag("V0prodMC","Omega"),
    cut=cms.string(""),
    name=cms.string("omega"),
    doc=cms.string("Omegas Variables"),
    singleton=cms.bool(False),
    extension=cms.bool(False),
    variables = OmegasVariablesMC # for now same variables rho, k to pipi
)

# PhiToKK and DsToPhiPi

PhiVariables = merge_psets(
    KsVariables,
    cms.PSet(
        ds_pion_pt   = Var("userFloat('ds_pion_pt')",      float, doc = "DsToPhiPi: pion pt"),
        ds_pion_eta  = Var("userFloat('ds_pion_eta')",     float, doc = "DsToPhiPi: pion eta"),
        ds_pion_phi  = Var("userFloat('ds_pion_phi')",     float, doc = "DsToPhiPi: pion phi"),
        #    ds_pion_mu_index = Var("userInt('ds_pion_mu_index')",     float, doc = "DsToPhiPi: pion muon index"),
        ds_mass      = Var("userFloat('ds_mass')",         float, doc = "DsToPhiPi: 3-body mass with vertex constraint"),
        ds_vtx_prob  = Var("userFloat('ds_vtx_prob')",     float, doc = "DsToPhiPi: vertex probability"),
        ds_vtx_chi2dof = Var("userFloat('ds_vtx_chi2dof')", float, doc = "DsToPhiPi: vertex normalized Chi^2"),
        ds_pt        = Var("userFloat('ds_pt')",           float, doc = "DsToPhiPi: vertex refitted pt"),
        ds_eta       = Var("userFloat('ds_eta')",          float, doc = "DsToPhiPi: vertex refitted eta"),
        ds_phi       = Var("userFloat('ds_phi')",          float, doc = "DsToPhiPi: vertex refitted phi"),
        ds_massErr   = Var("userFloat('ds_massErr')",      float, doc = "DsToPhiPi: vertex refitted mass error"),
        ds_lxy       = Var("userFloat('ds_lxy')",          float, doc = "DsToPhiPi: vertex displacement in XY plane wrt Beam Spot"),
        ds_slxy      = Var("userFloat('ds_sigLxy')",       float, doc = "DsToPhiPi: vertex displacement significance in XY plane wrt Beam Spot"),
        ds_cosAlphaXY = Var("userFloat('ds_cosAlphaXY')",    float, doc = "DsToPhiPi: cosine of pointing angle in XY wrt BS"),
        ds_sipBS     = Var("userFloat('ds_sipBS')",        float, doc = "DsToPhiPi: impact parameter significance of the candidate trajectory in XY wrt BS"),
        ds_sipPV     = Var("userFloat('ds_sipPV')",        float, doc = "DsToPhiPi: impact parameter significance of the candidate trajectory in 3D wrt PV"),
    )
)

PhiVariablesMC = merge_psets(
    PhiVariables,
    cms.PSet(
        gen_trk1_pdgId  = Var("userInt(  'gen_trk1_pdgId')",    int,   doc = "Gen match: first track pdg Id"),
        gen_trk1_mpdgId = Var("userInt(  'gen_trk1_mpdgId')",   int,   doc = "Gen match: first track mother pdg Id"),
        gen_trk1_pt     = Var("userFloat('gen_trk1_pt')",     float,   doc = "Gen match: first track pt"),
        gen_trk2_pdgId  = Var("userInt(  'gen_trk2_pdgId')",    int,   doc = "Gen match: second track pdg Id"),
        gen_trk2_mpdgId = Var("userInt(  'gen_trk2_mpdgId')",   int,   doc = "Gen match: second track mother pdg Id"),
        gen_trk2_pt     = Var("userFloat('gen_trk2_pt')",     float,   doc = "Gen match: second track pt"),
        gen_pdgId       = Var("userInt(  'gen_pdgId')",         int,   doc = "Gen match: ditrack pdg Id"),
        gen_mass        = Var("userFloat('gen_mass')",        float,   doc = "Gen match: ditrack mass"),
        gen_pt          = Var("userFloat('gen_pt')",          float,   doc = "Gen match: ditrack pt"),
        ),
)

PhiTable=cms.EDProducer("SimpleCompositeCandidateFlatTableProducer", 
    src=cms.InputTag("V0prod","Phi"),
    cut=cms.string(""),
    name=cms.string("phi"),
    doc=cms.string("Phi Variables"),
    singleton=cms.bool(False),
    extension=cms.bool(False),
    variables = PhiVariables
)

PhiMcTable=cms.EDProducer("SimpleCompositeCandidateFlatTableProducer", 
    src=cms.InputTag("V0prodMC","Phi"),
    cut=cms.string(""),
    name=cms.string("phi"),
    doc=cms.string("Phi Variables"),
    singleton=cms.bool(False),
    extension=cms.bool(False),
    variables = PhiVariablesMC
)


# D0ToKPi

D0Variables = cms.PSet(
    mass         = Var("mass",                         float, doc = "Unfit invariant mass"),
    doca         = Var("userFloat('doca')",            float, doc = "Distance of closest approach of tracks"),
    iso          = Var("userFloat('iso')",             float, doc = "tracks isolation (pt/(pt+sum))"),
    kaon_pt      = Var("userFloat('kaon_pt')",         float, doc = "Kaon pt"),
    kaon_eta     = Var("userFloat('kaon_eta')",        float, doc = "Kaon eta"),
    kaon_phi     = Var("userFloat('kaon_phi')",        float, doc = "Kaon phi"),
#    kaon_mu_index = Var("userInt('kaon_mu_index')",      int, doc = "Matched muon index for track 1"),
    pion_pt      = Var("userFloat('pion_pt')",         float, doc = "Pion pt"),
    pion_eta     = Var("userFloat('pion_eta')",        float, doc = "Pion eta"),
    pion_phi     = Var("userFloat('pion_phi')",        float, doc = "Pion phi"),
#    pion_mu_index = Var("userInt('pion_mu_index')",      int, doc = "Matched muon index for track 2"),
    kaon_sip     = Var("userFloat('kaon_sip')",        float, doc = "Kaon 2D impact parameter significance wrt Beam Spot"),
    pion_sip     = Var("userFloat('pion_sip')",        float, doc = "Pion 2D impact parameter significance wrt Beam Spot"),
    kin_valid    = Var("userInt('kin_valid')",         int,   doc = "Kinematic fit: vertex validity"),
    kin_vtx_prob = Var("userFloat('kin_vtx_prob')",    float, doc = "Kinematic fit: vertex probability"),
    kin_vtx_chi2dof = Var("userFloat('kin_vtx_chi2dof')", float, doc = "Kinematic fit: vertex normalized Chi^2"),
    kin_mass     = Var("userFloat('kin_mass')",        float, doc = "Kinematic fit: vertex refitted mass"),
    kin_pt       = Var("userFloat('kin_pt')",          float, doc = "Kinematic fit: vertex refitted pt"),
    kin_eta      = Var("userFloat('kin_eta')",         float, doc = "Kinematic fit: vertex refitted eta"),
    kin_phi      = Var("userFloat('kin_phi')",         float, doc = "Kinematic fit: vertex refitted phi"),
    kin_massErr  = Var("userFloat('kin_massErr')",     float, doc = "Kinematic fit: vertex refitted mass error"),
    kin_lxy      = Var("userFloat('kin_lxy')",         float, doc = "Kinematic fit: vertex displacement in XY plane wrt Beam Spot"),
    kin_slxy     = Var("userFloat('kin_sigLxy')",      float, doc = "Kinematic fit: vertex displacement significance in XY plane wrt Beam Spot"),
    kin_cosAlphaXY = Var("userFloat('kin_cosAlphaXY')",    float, doc = "Kinematic fit: cosine of pointing angle in XY wrt BS"),
    kin_sipBS    = Var("userFloat('kin_sipBS')",       float, doc = "Kinematic fit: impact parameter significance of the candidate trajectory in XY wrt BS"),
    kin_sipPV    = Var("userFloat('kin_sipPV')",       float, doc = "Kinematic fit: impact parameter significance of the candidate trajectory in 3D wrt PV"),
)

D0VariablesMC = merge_psets(
    D0Variables,
    cms.PSet(
        gen_kaon_pdgId  = Var("userInt(  'gen_kaon_pdgId')",    int,   doc = "Gen match: first track pdg Id"),
        gen_kaon_mpdgId = Var("userInt(  'gen_kaon_mpdgId')",   int,   doc = "Gen match: first track mother pdg Id"),
        gen_kaon_pt     = Var("userFloat('gen_kaon_pt')",     float,   doc = "Gen match: first track pt"),
        gen_pion_pdgId  = Var("userInt(  'gen_pion_pdgId')",    int,   doc = "Gen match: second track pdg Id"),
        gen_pion_mpdgId = Var("userInt(  'gen_pion_mpdgId')",   int,   doc = "Gen match: second track mother pdg Id"),
        gen_pion_pt     = Var("userFloat('gen_pion_pt')",     float,   doc = "Gen match: second track pt"),
        gen_pdgId       = Var("userInt(  'gen_pdgId')",         int,   doc = "Gen match: ditrack pdg Id"),
        gen_mass        = Var("userFloat('gen_mass')",        float,   doc = "Gen match: ditrack mass"),
        gen_pt          = Var("userFloat('gen_pt')",          float,   doc = "Gen match: ditrack pt"),
        ),
)

D0Table=cms.EDProducer("SimpleCompositeCandidateFlatTableProducer", 
    src=cms.InputTag("V0prod","D0"),
    cut=cms.string(""),
    name=cms.string("d0"),
    doc=cms.string("D0s Variables"),
    singleton=cms.bool(False),
    extension=cms.bool(False),
    variables = D0Variables
)

D0McTable=cms.EDProducer("SimpleCompositeCandidateFlatTableProducer", 
    src=cms.InputTag("V0prodMC","D0"),
    cut=cms.string(""),
    name=cms.string("d0"),
    doc=cms.string("D0 Variables"),
    singleton=cms.bool(False),
    extension=cms.bool(False),
    variables = D0VariablesMC
)

# LambdaToPPi

LambdaVariables = cms.PSet(
    mass         = Var("mass",                         float, doc = "Unfit invariant mass"),
    doca         = Var("userFloat('doca')",            float, doc = "Distance of closest approach of tracks"),
    iso          = Var("userFloat('iso')",             float, doc = "tracks isolation (pt/(pt+sum))"),
    proton_pt      = Var("userFloat('proton_pt')",         float, doc = "Proton pt"),
    proton_eta     = Var("userFloat('proton_eta')",        float, doc = "Proton eta"),
    proton_phi     = Var("userFloat('proton_phi')",        float, doc = "Proton phi"),
#    proton_mu_index = Var("userInt('proton_mu_index')",      int, doc = "Matched muon index for track 1"),
    pion_pt      = Var("userFloat('pion_pt')",         float, doc = "Pion pt"),
    pion_eta     = Var("userFloat('pion_eta')",        float, doc = "Pion eta"),
    pion_phi     = Var("userFloat('pion_phi')",        float, doc = "Pion phi"),
#    pion_mu_index = Var("userInt('pion_mu_index')",      int, doc = "Matched muon index for track 2"),
    proton_sip     = Var("userFloat('proton_sip')",        float, doc = "Proton 2D impact parameter significance wrt Beam Spot"),
    pion_sip     = Var("userFloat('pion_sip')",        float, doc = "Pion 2D impact parameter significance wrt Beam Spot"),
    kin_valid    = Var("userInt('kin_valid')",         int,   doc = "Kinematic fit: vertex validity"),
    kin_vtx_prob = Var("userFloat('kin_vtx_prob')",    float, doc = "Kinematic fit: vertex probability"),
    kin_vtx_chi2dof = Var("userFloat('kin_vtx_chi2dof')", float, doc = "Kinematic fit: vertex normalized Chi^2"),
    kin_mass     = Var("userFloat('kin_mass')",        float, doc = "Kinematic fit: vertex refitted mass"),
    kin_pt       = Var("userFloat('kin_pt')",          float, doc = "Kinematic fit: vertex refitted pt"),
    kin_eta      = Var("userFloat('kin_eta')",         float, doc = "Kinematic fit: vertex refitted eta"),
    kin_phi      = Var("userFloat('kin_phi')",         float, doc = "Kinematic fit: vertex refitted phi"),
    kin_massErr  = Var("userFloat('kin_massErr')",     float, doc = "Kinematic fit: vertex refitted mass error"),
    kin_lxy      = Var("userFloat('kin_lxy')",         float, doc = "Kinematic fit: vertex displacement in XY plane wrt Beam Spot"),
    kin_slxy     = Var("userFloat('kin_sigLxy')",      float, doc = "Kinematic fit: vertex displacement significance in XY plane wrt Beam Spot"),
    kin_cosAlphaXY = Var("userFloat('kin_cosAlphaXY')",    float, doc = "Kinematic fit: cosine of pointing angle in XY wrt BS"),
    kin_sipBS    = Var("userFloat('kin_sipBS')",       float, doc = "Kinematic fit: impact parameter significance of the candidate trajectory in XY wrt BS"),
    kin_sipPV    = Var("userFloat('kin_sipPV')",       float, doc = "Kinematic fit: impact parameter significance of the candidate trajectory in 3D wrt PV"),
)

LambdaVariablesMC = merge_psets(
    LambdaVariables,
    cms.PSet(
        gen_proton_pdgId  = Var("userInt(  'gen_proton_pdgId')",    int,   doc = "Gen match: first track pdg Id"),
        gen_proton_mpdgId = Var("userInt(  'gen_proton_mpdgId')",   int,   doc = "Gen match: first track mother pdg Id"),
        gen_proton_pt     = Var("userFloat('gen_proton_pt')",     float,   doc = "Gen match: first track pt"),
        gen_pion_pdgId  = Var("userInt(  'gen_pion_pdgId')",    int,   doc = "Gen match: second track pdg Id"),
        gen_pion_mpdgId = Var("userInt(  'gen_pion_mpdgId')",   int,   doc = "Gen match: second track mother pdg Id"),
        gen_pion_pt     = Var("userFloat('gen_pion_pt')",     float,   doc = "Gen match: second track pt"),
        gen_pdgId       = Var("userInt(  'gen_pdgId')",         int,   doc = "Gen match: ditrack pdg Id"),
        gen_mass        = Var("userFloat('gen_mass')",        float,   doc = "Gen match: ditrack mass"),
        gen_pt          = Var("userFloat('gen_pt')",          float,   doc = "Gen match: ditrack pt"),
        ),
)

LambdaTable=cms.EDProducer("SimpleCompositeCandidateFlatTableProducer", 
    src=cms.InputTag("V0prod","Lambda"),
    cut=cms.string(""),
    name=cms.string("lambda"),
    doc=cms.string("Lambdas Variables"),
    singleton=cms.bool(False),
    extension=cms.bool(False),
    variables = LambdaVariables
)

LambdaMcTable=cms.EDProducer("SimpleCompositeCandidateFlatTableProducer", 
    src=cms.InputTag("V0prodMC","Lambda"),
    cut=cms.string(""),
    name=cms.string("lambda"),
    doc=cms.string("Lambda Variables"),
    singleton=cms.bool(False),
    extension=cms.bool(False),
    variables = LambdaVariablesMC
)


V0Sequence   = cms.Sequence(V0prod)
V0McSequence = cms.Sequence(V0prodMC)
V0Tables     = cms.Sequence(KsTable + RhosTable + OmegasTable + PhiTable + D0Table + LambdaTable)
V0McTables   = cms.Sequence(KsMcTable + RhosMcTable + OmegasMcTable + PhiMcTable + D0McTable + LambdaMcTable)
