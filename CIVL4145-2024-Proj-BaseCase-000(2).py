# =============================================================================
# ===== CIVL4145 Course Project (2024)
# ===== MODFLOW 6
# ===== Base case model (version 000)
# ===== Preliminary model set-up for leak site
# ===== 
# =============================================================================

# =============================================================================
# ===== 0.0 PYTHON ENVIRONMENT SET-UP =========================================
# =============================================================================
import matplotlib.pyplot as plt
import flopy
import numpy as np

# Specifying executable location, working folder and simulation name
mf6exe = 'D:/Working/00-Reference/Software/MODFLOW/MODFLOW6/mf6.5.0_win64/bin/mf6.exe'
ws = './Run000'
sim_name = "Base"

# =============================================================================
# ===== 1.0 DEFINING BASIC FloPy MODEL PROPERTIES =============================
# =============================================================================

# ===== 1.1 Setting length and time units
length_units = "meters"
time_units = "days"
# Setting three stress periods (1000 days, 5 years, 3 years)
perlen = [1000, 6*365.0, 4*365.0]  # Simulation time [days]
nper = len(perlen)
nstp = [10, 6*365, 4*365]
tsmult = [1.0, 1.0, 1.0]

# ===== 1.2 Setting model geometry
Lx = 3000    # Length of domain in x-direction
Ly = 2000    # Length of domain in y-direction
delr = 100.0  # Row width (height) [m] (i.e., dy)
delc = 100.0  # Column width [m] (i.e., dx)
# Add extra 2 columns for north & south boundary cells (affects y plot locations)
nrow = int(np.round(Ly/delr)) # Number of rows
ncol = int(np.round(Lx/delc)) # Number of columns

nlay = 3  # Number of layers
delz = [2.0, 1.0, 1.0]  # Layer thickness [m] (starting at top layer)
top = 4.0  # Elevation of top of the model [m]
basez = 0.0  # Elevation of base of the model [m]
# Settng elevation of bottom of each layer
botz = []
botz.append(top-delz[0])
for k in range(1,nlay):
    botz.append(botz[k-1]-delz[k])
botm = np.zeros((nlay, nrow, ncol), dtype=float)
for k in np.arange(nlay):
    botm[k,:,:] = botz[k]

# ===== 1.3 Setting hydraulic and solute transport properties
# Porosity [-]
ngravel = 0.28  # Porosity of gravel (specific yield)
prsity = np.ones((nlay, nrow, ncol), dtype=float) # Establish variable
prsity[0,:,:]=ngravel   # Set layer 0 porosity to ngravel
prsity[1,:,:]=ngravel # Set layer 1 porosity to ngravel
prsity[2,:,:]=ngravel   # Set layer 2 porosity to ngravel

# Hydraulic conductivity [m/d]
kgravel = 150.0 # Horizontal hydraulic conductivity of gravel [m/d]
k11 = np.ones((nlay, nrow, ncol), dtype=float) # Establish variable
k11[0,:,:] = kgravel
k11[1,:,:] = kgravel
k11[2,:,:] = kgravel
k33 = k11.copy()  # Vertical hydraulic conductivity [m/d]

# Dispersivity (nitrate)
al = 10.0              # Longitudinal dispersivity [m]
trpt = 0.5             # Ratio of transverse to longitudinal dispersitivity
trpv = 0.05            # Ratio of vertical to longitudinal dispersitivity
ath1 = al * trpt       # Transverse dispersivity [m]
ath2 = al * trpv       # Vertical dispersivity [m]
sconc = 0.0            # Starting soute concentration (C(t=0))
dmcoef = 1.468e-4     # Molecular diffusion coefficient [m**2/d]

# Recharge
rech = 0.0 # Recharge rate [m/d]

# Specify saturated thickness method
# (0=constant (confined?), >0=varies with head in cell (unconfined), <0=variable)
icelltype = 1

# Advection solution options
mixelm = -1
# mixelem = 0, the standard finite-difference method with upstream
#              or central-in-space weighting, depending on the value of NADVFD;
# mixelem = 1, the forward-tracking method of characteristics (MOC);
# mixelem = 2, the backward-tracking modified method of characteristics (MMOC);
# mixelem = 3, the hybrid method of characteristics (HMOC) with MOC or MMOC
#              automatically and dynamically selected;
# mixelem = -1, the third-order TVD scheme (ULTIMATE).

# ===== 1.4 Set initial conditions
h1 = 3.2 # Hydraulic head on northern boundary
h2 = 2.7 # Hydraulic head on southern boundary
havg = (h1+h2)/2 # Average hydraulic head to use as initial cond
strt = np.ones((nlay, nrow, ncol), dtype=float)*havg
#    l, r, c 
strt[:, 0, :] = h1   # Specify heads on western boundary
strt[:, -1,:] = h2  # Specify heads on eastern boundary (Note not fully pen)

# Exlude cells from calculation using idomain flag
# idomain = 0 --> cells does not exist in simulation, but values are written to it
# idomain = 1 --> cell exists in simulation
# idomain = -1 --> cell does not exist in simulation, but flow passes through it
idomain = np.ones((nlay, nrow, ncol), dtype=int) 

# ===== 1.4 Define well information
# Define function to get row and column locations from x-y coordinates
def find_rowcol(x,y,Lx,Ly,delc,delr):
    # Inputs are
    # x = real world x coordinate
    # y = real world y coordinate
    # Lx = length of domain in x-direction
    # Ly = length of domain in y-direction
    # delc = column width (i.e., dx) 
    # delr = row width (i.e., dy)
    # Finding nearest cell
    ycb = np.linspace(0,Ly,num = int(Ly/delr)+1)
    irowy = np.where((Ly-y) <= ycb)[0][0]
    xcb = np.linspace(0,Lx,num = int(Lx/delc)+1)
    jcolx = np.where(x <= xcb)[0][0]
    # Note adding +1 to column add to account for east boundary cell
    return jcolx, irowy

# Define pumping rates
qwell_leak = 64.0   # Volumetric injection rate of leak [m^3/d]
cwell_leak = 100.0  # Concentration of leak [kg/m^3]
qwell_ex1 = -150.0  # Well 1 volumetric extraction rate [m^3/d]
qwell_ex2 = -175.0  # Well 2 volumetric extraction rate [m^3/d]
qwell_ex3 = -140.0  # Well 3 volumetric extraction rate [m^3/d]
qwell_ex4 = -155.0  # Well 4 volumetric extraction rate [m^3/d]

cwell_ex = 0.0      # Concentration of extracted

# Well locations (Note: realworld coordinates, not adjusted for bc cells)

# Extraction well 1
x_ex1 = 1064
y_ex1 = 1720
jcolx_ex1, irowy_ex1 = find_rowcol(x_ex1, y_ex1, Lx, Ly, delc, delr)

# Extraction well 2
x_ex2 = 1988
y_ex2 = 1720
jcolx_ex2, irowy_ex2 = find_rowcol(x_ex2, y_ex2, Lx, Ly, delc, delr)

# Extraction well 3
x_ex3 = 1295
y_ex3 = 280
jcolx_ex3, irowy_ex3 = find_rowcol(x_ex3, y_ex3, Lx, Ly, delc, delr)

# Extraction well 4
x_ex4 = 1701
y_ex4 = 280
jcolx_ex4, irowy_ex4 = find_rowcol(x_ex4, y_ex4, Lx, Ly, delc, delr)

# Pipeline leak
x_leak = 1500
y_leak = 1000
jcolx_leak, irowy_leak = find_rowcol(x_leak, y_leak, Lx, Ly, delc, delr)


# Specifying well details
#          sp  [(l,          r,          c), flow,  conc]
spd_mf6 = {0: [[(2, irowy_ex1, jcolx_ex1), qwell_ex1, cwell_ex],
               [(2, irowy_ex2, jcolx_ex2), qwell_ex2, cwell_ex],
               [(2, irowy_ex3, jcolx_ex3), qwell_ex3, cwell_ex],
               [(2, irowy_ex4, jcolx_ex4), qwell_ex4, cwell_ex],
               [(0, irowy_leak,jcolx_leak), 0,0]],
           1: [[(2, irowy_ex1, jcolx_ex1), qwell_ex1, cwell_ex],
               [(2, irowy_ex2, jcolx_ex2), qwell_ex2, cwell_ex],
               [(2, irowy_ex3, jcolx_ex3), qwell_ex3, cwell_ex],
               [(2, irowy_ex4, jcolx_ex4), qwell_ex4, cwell_ex],
               [(0, irowy_leak,jcolx_leak),qwell_leak,cwell_leak]],
           2: [[(2, irowy_ex1, jcolx_ex1), qwell_ex1, cwell_ex],
               [(2, irowy_ex2, jcolx_ex2), qwell_ex2, cwell_ex],
               [(2, irowy_ex3, jcolx_ex3), qwell_ex3, cwell_ex],
               [(2, irowy_ex4, jcolx_ex4), qwell_ex4, cwell_ex],
               [(0, irowy_leak,jcolx_leak),qwell_leak,cwell_leak]],}

# ===== 1.5 Define recharge information
# Recharge stress perdiod data needed if recharge method 1 (general recharge
# package is used).
rchspd = []
for i in range(nrow):
    for j in range(ncol):
        #             [(lay, row, col), recharge, conc]
        rchspd.append([(  0,   i,   j),     rech,  0.0])
rchspd = {0: rchspd, 1: rchspd, 2: rchspd}

# ===== 1.6 Define constant head boundaries
chdspd = []
# Loop through the left & right sides.
for k in np.arange(nlay):
    for j in np.arange(ncol):
        #              (l, r, c),          head, conc
        chdspd.append([(k, 0, j), strt[k, 0, j], 0.0])
        # chdspd.append([(k, i, ncol - 1), strt[k, i, ncol - 1], 0.0])
        chdspd.append([(k, nrow-1, j), strt[k, nrow-1, j], 0.0])

chdspd = {0: chdspd, 1: chdspd, 2: chdspd}

# ===== 1.7 Define solver settings
nouter, ninner = 100, 300
hclose, rclose, relax = 1e-6, 1e-6, 1.0
percel = 1.0  # HMOC parameters
itrack = 3
wd = 0.5
dceps = 1.0e-5
nplane = 1
npl = 0
nph = 16
npmin = 2
npmax = 32
dchmoc = 1.0e-3
nlsink = nplane
npsink = nph

# ====== Set static temporal data used by tdis file
tdis_rc = []
tdis_rc.append((perlen, nstp, 1.0))

# =============================================================================
# ===== 2.0 CREATE FLOW MODEL OBJECTS AND DEFINE FLOW PACKAGES ================
# =============================================================================
name = "Base"
gwfname = "gwf-" + name
sim_ws = ws
sim = flopy.mf6.MFSimulation(sim_name=sim_name,
                             sim_ws=sim_ws,
                             exe_name=mf6exe)

# ===== 2.1 Defining MODFLOW 6 time discretization
tdis_rc = []
for i in range(nper):
    tdis_rc.append((perlen[i], nstp[i], tsmult[i]))

flopy.mf6.ModflowTdis(sim,
                      nper=nper,
                      perioddata=tdis_rc,
                      time_units=time_units)
        
# ===== 2.2 Defining MODFLOW 6 groundwater flow model
gwf = flopy.mf6.ModflowGwf(sim,
                           modelname=gwfname,
                           save_flows=True,
                           model_nam_file="{}.nam".format(gwfname),)       
        
# ===== 2.3 Defining MODFLOW 6 solver for flow model
imsgwf = flopy.mf6.ModflowIms(sim,
                              print_option="SUMMARY",
                              outer_dvclose=hclose,
                              outer_maximum=nouter,
                              under_relaxation="NONE",
                              inner_maximum=ninner,
                              inner_dvclose=hclose,
                              rcloserecord=rclose,
                              linear_acceleration="CG",
                              scaling_method="NONE",
                              reordering_method="NONE",
                              relaxation_factor=relax,
                              filename="{}.ims".format(gwfname),)

sim.register_ims_package(imsgwf, [gwf.name])

# ===== 2.4 Defining MODFLOW 6 discretization package
flopy.mf6.ModflowGwfdis(gwf,
                        length_units=length_units,
                        nlay=nlay,
                        nrow=nrow,
                        ncol=ncol,
                        delr=delr,
                        delc=delc,
                        top=top,
                        botm=botm,
                        idomain=idomain,
                        filename="{}.dis".format(gwfname),)

# ===== 2.5 Defining MODFLOW 6 node-property flow package
flopy.mf6.ModflowGwfnpf(gwf,
                        save_flows=False,
                        icelltype=icelltype,
                        k=k11,
                        k33=k33,
                        save_specific_discharge=True,
                        filename="{}.npf".format(gwfname),)

# ===== 2.6 Defining MODFLOW 6 initial conditions package for flow model
flopy.mf6.ModflowGwfic(gwf,
                       strt=strt,
                       filename="{}.ic".format(gwfname),)

# # ===== 2.7 Define MODFLOW 6 storage package
# sto = flopy.mf6.ModflowGwfsto(gwf,
#                               ss=0,
#                               sy=0,
#                               filename="{}.sto".format(gwfname),)

# ===== 2.8 Defining MODFLOW 6 constant head package
flopy.mf6.ModflowGwfchd(gwf,
                        maxbound=len(chdspd),
                        stress_period_data=chdspd,
                        save_flows=False,
                        auxiliary="CONCENTRATION",
                        pname="CHD-1",
                        filename="{}.chd".format(gwfname),)

# ===== 2.9 Defining MODFLOW well package
flopy.mf6.ModflowGwfwel(gwf,
                        print_input=True,
                        print_flows=True,
                        stress_period_data=spd_mf6,
                        save_flows=False,
                        auxiliary="CONCENTRATION",
                        pname="WEL-1",
                        filename="{}.wel".format(gwfname),)

# ===== 2.10 Defining MODFLOW recharge package
# Method 1 - Using general recharge package
flopy.mf6.ModflowGwfrch(gwf,
                        print_input=True,
                        print_flows=True,
                        stress_period_data=rchspd,
                        save_flows=False,
                        auxiliary="CONCENTRATION",
                        pname="RCH-1",
                        filename="{}.rch".format(gwfname),)

# Method 2 - Using array-based recharge package
# flopy.mf6.ModflowGwfrcha(gwf,
#                         print_input=True,
#                         print_flows=True,
#                         recharge = rech,
#                         save_flows=False,
#                         auxiliary="CONCENTRATION",
#                         pname="RCH-1",
#                         filename="{}.rch".format(gwfname),)

# ===== 2.11 Defining MODFLOW 6 output control package for flow model
flopy.mf6.ModflowGwfoc(gwf,
                       head_filerecord="{}.hds".format(gwfname),
                       budget_filerecord="{}.bud".format(gwfname),
                       headprintrecord=[("COLUMNS", 10, "WIDTH", 15,
                                         "DIGITS", 6, "GENERAL")],
                       saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
                       printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],)

# =============================================================================
# ===== 3.0 CREATE TRANSPORT MODEL OBJECTS AND DEFINE TRANSPORT PACKAGES ======
# =============================================================================
gwtname = "gwt_" + name
gwt = flopy.mf6.MFModel(sim,
                        model_type="gwt6",
                        modelname=gwtname,
                        model_nam_file="{}.nam".format(gwtname),)
gwt.name_file.save_flows = True

# ===== 3.1 Create iterative model solution and register the gwt model with it
imsgwt = flopy.mf6.ModflowIms(sim,
                              print_option="SUMMARY",
                              outer_dvclose=hclose,
                              outer_maximum=nouter,
                              under_relaxation="NONE",
                              inner_maximum=ninner,
                              inner_dvclose=hclose,
                              rcloserecord=rclose,
                              linear_acceleration="BICGSTAB",
                              scaling_method="NONE",
                              reordering_method="NONE",
                              relaxation_factor=relax,
                              filename="{}.ims".format(gwtname),)
sim.register_ims_package(imsgwt, [gwt.name])

# ===== 3.2 Defining MODFLOW 6 transport discretization package
flopy.mf6.ModflowGwtdis(gwt,
                        nlay=nlay,
                        nrow=nrow,
                        ncol=ncol,
                        delr=delr,
                        delc=delc,
                        top=top,
                        botm=botm,
                        idomain=1,
                        filename="{}.dis".format(gwtname),)

# ===== 3.3 Defining MODFLOW 6 transport initial concentrations
flopy.mf6.ModflowGwtic(gwt,
                       strt=sconc,
                       filename="{}.ic".format(gwtname),)

# ===== 3.4 Defining MODFLOW 6 transport advection package
if mixelm == 0:
    scheme = "UPSTREAM"
elif mixelm == -1:
    scheme = "TVD"
else:
    raise Exception()

flopy.mf6.ModflowGwtadv(gwt,
                        scheme=scheme,
                        filename="{}.adv".format(gwtname),)


# ===== 3.5 Defining MODFLOW 6 transport dispersion package
if al != 0:
    flopy.mf6.ModflowGwtdsp(gwt,
                            xt3d_off=True,
                            alh=al,
                            ath1=ath1,
                            ath2=ath2,
                            diffc=dmcoef,
                            filename="{}.dsp".format(gwtname),)

# ===== 3.6 Defining MODFLOW 6 transport mass storage package
#      (formerly "reaction" package in MT3DMS)
flopy.mf6.ModflowGwtmst(gwt,
                        porosity=prsity,
                        first_order_decay=False,
                        decay=None,
                        decay_sorbed=None,
                        sorption=None,
                        bulk_density=None,
                        distcoef=None,
                        filename="{}.mst".format(gwtname),)

# ===== 3.7 Defining MODFLOW 6 transport constant concentration package
# flopy.mf6.ModflowGwtcnc(gwt,
#                         maxbound=len(cncspd),
#                         stress_period_data=cncspd,
#                         save_flows=False,
#                         pname="CNC-1",
#                         filename="{}.cnc".format(gwtname),)

# ===== 3.8 Defining MODFLOW 6 transport source-sink mixing package
sourcearray = [("WEL-1", "AUX", "CONCENTRATION"),
                  ("CHD-1", "AUX", "CONCENTRATION"),
                  ("RCH-1", "AUX", "CONCENTRATION")]
flopy.mf6.ModflowGwtssm(gwt,
                        sources=sourcearray,
                        filename="{}.ssm".format(gwtname),)

# ===== 3.9 Defining MODFLOW 6 transport output control package
flopy.mf6.ModflowGwtoc(gwt,
                       budget_filerecord="{}.cbc".format(gwtname),
                       concentration_filerecord="{}.ucn".format(gwtname),
                       concentrationprintrecord=[
                           ("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6,
                            "GENERAL")],
                       saverecord=[("CONCENTRATION", "ALL"),
                                   ("BUDGET", "LAST")],
                       printrecord=[("CONCENTRATION", "LAST"),
                                    ("BUDGET", "LAST")],)

# Defining MODFLOW 6 flow-transport exchange mechanism
flopy.mf6.ModflowGwfgwt(sim,
                        exgtype="GWF6-GWT6",
                        exgmnamea=gwfname,
                        exgmnameb=gwtname,
                        filename="{}.gwfgwt".format(name),)

# =============================================================================
# ===== 4.0 CREATE MODFLOW 6 INPUT FILES AND RUN THE SIMULATION ===============
# =============================================================================

# ===== 4.1 Write input files
sim.write_simulation()

# ===== 4.2 Run the simulation
success, buff = sim.run_simulation()
assert success, "MODFLOW 6 did not terminate normally."

# =============================================================================
# ===== 5.0 POST-PROCESS SIMULATION RESULTS ===================================
# =============================================================================
# ===== 5.1 Extracting simulation data ========================================

# Get the MF6 flow model output
# ===== Output method
ucnobj_mf6_head = gwf.output.head()
head = ucnobj_mf6_head.get_alldata()
# timesteps_h = ucnobj_mf6_head.times
# head_end = head[-1]
# ===== Binary file method
# fname = 'Run000/gwf-Base.hds'
# hdobj = flopy.utils.HeadFile(fname)
# head = hdobj.get_data()

# ===== Get the MF6 transport model output (concentration)
ucnobj_mf6_conc = gwt.output.concentration()
conc = ucnobj_mf6_conc.get_alldata()
timesteps = ucnobj_mf6_conc.times

# ===== Set timestep to use
targetTime = 4600.0 # Days from start of simulation
timeIDX = [i for i, v in enumerate(timesteps) if v >= targetTime][0]
dispTime = timesteps[timeIDX]
    
# ===== Set layer to plot
klay = 0 # 0 = top layer

# ===== 5.2 Plotting map of head data =========================================
fig = plt.figure(figsize=(8,8),dpi=300, tight_layout=True)
ax = fig.add_subplot(1, 1, 1, aspect="equal")
ax.set_title("Hydrualic Head: Layer " + "{}".format(klay) +
             " (Time = " + "{}".format(dispTime) + " "
             "{}".format(time_units) + ")",
             loc='left')
plt.xlabel("Distance along x-axis [m]")
plt.ylabel("Distance along y-axis [m]")
mapview = flopy.plot.PlotMapView(model=gwf)
theLevels = np.linspace(2,4,num=21)
contour_set = mapview.contour_array(head[timeIDX,klay,:,:],
                                    levels=theLevels,
                                    colors='k',
                                    linestyles='--')
plt.clabel(contour_set, inline=1, fontsize=10)
# plt.colorbar(contour_set, shrink=0.250)
quadmesh = mapview.plot_array(head[timeIDX,klay,:,:], alpha=1)
cb = plt.colorbar(quadmesh, shrink=0.25)
quadmesh = mapview.plot_bc("CHD",color='pink')
quadmesh = mapview.plot_bc("WEL-1",color='grey')
linecollection = mapview.plot_grid()

# ===== 5.3 Plotting map of solute data =======================================
minC = 0.0 # Minium concentration to display
maxC = 100.0 # Maximum concentration to diplay
fig = plt.figure(figsize=(8, 8),dpi=300, tight_layout=True)
ax = fig.add_subplot(1, 1, 1, aspect="equal")
ax.set_title("Solute plume: Layer " + "{}".format(klay) +
             " (Time = " + "{}".format(dispTime) + " "
             "{}".format(time_units) + ")",
             loc='left')
plt.xlabel("Distance along x-axis [m]")
plt.ylabel("Distance along y-axis [m]")
mapview = flopy.plot.PlotMapView(model=gwf,
                                 layer=klay)
quadmesh = mapview.plot_array(conc[timeIDX,klay,:,:],
                              vmin=minC,vmax=maxC,
                              cmap='jet')
cb = plt.colorbar(quadmesh, shrink=0.5)
contour_set = mapview.contour_array(conc[timeIDX,klay,:,:],
                                    levels=[0.1, 1.0, 10.0, 50.0],
                                    colors="w",
                                    linestyles="--")
plt.clabel(contour_set, inline=1, fontsize=10)
quadmesh = mapview.plot_bc("CHD",color='pink')
quadmesh = mapview.plot_bc("WEL-1",color='grey')
# plt.colorbar(contour_set, shrink=0.50)
linecollection = mapview.plot_grid()


# ===== 5.4 Plotting cross-section of head and solute data
colID = jcolx_leak # Specify the row to use for cross section

fig = plt.figure(figsize=(12, 12),dpi=300, tight_layout=True)
# Hydraluic heads vertical section
ax = fig.add_subplot(2, 1, 1,aspect=90)
ax.set_title("(a) Hydraulic head: Vertical section - Column " +
             "{}".format(colID) + " (Time = " + "{}".format(dispTime) + " "
             "{}".format(time_units) + ")", loc='left')
plt.xlabel("Distance along y-axis [m]")
plt.ylabel("Elevation (z-axis) [m]")
xsect = flopy.plot.PlotCrossSection(model=gwf, line={"column": colID})
csa = xsect.plot_array(head, head=head,alpha=0.5)
cb = plt.colorbar(csa, shrink=0.25)
contour_set = xsect.contour_array(head,
                                  head=head,
                                  levels=theLevels,
                                  colors="k",
                                  linestyles='--')
plt.clabel(contour_set, fmt="%.1f", colors="k", fontsize=11)
patches = xsect.plot_bc("CHD",color='pink')
patches = xsect.plot_bc("WEL-1",color='grey')
linecollection = xsect.plot_grid()

# Solute concetration vertical section
ax = fig.add_subplot(2, 1, 2,aspect=90)
ax.set_title("(b) Solute concentration: Vertical section - Column " +
             "{}".format(colID) + " (Time = " + "{}".format(dispTime) + " "
             "{}".format(time_units) + ")", loc='left')
plt.xlabel("Distance along y-axis [m]")
plt.ylabel("Elevation (z-axis) [m]")
xsect = flopy.plot.PlotCrossSection(model=gwf, line={"column": colID})
csa = xsect.plot_array(conc[timeIDX],
                              vmin=minC,vmax=maxC,
                              cmap='jet')
cb = plt.colorbar(csa, shrink=0.25)
patches = xsect.plot_bc("CHD",color='pink')
patches = xsect.plot_bc("WEL-1",color='grey')
linecollection = xsect.plot_grid()

# ===== 5.5 Plotting solute breakthrough curve
# ===== Setting plot formatting (global settings - applies to all plots)
mS = 10 # Used to set marker size
lW = 3 # Used to set linewidth
fS = 16 # Used to set font size
plt.rcParams['font.family'] = 'Times New Roman' # Globally sets the font type
plt.rc('font',size=fS)
plt.rc('axes',titlesize=fS)
plt.rc('axes',labelsize=fS)
plt.rc('xtick',labelsize=fS)
plt.rc('ytick',labelsize=fS)
plt.rc('legend',fontsize=fS)
plt.rc('figure',titlesize=fS)


fig = plt.figure(figsize=(8, 8),dpi=300, tight_layout=True)
ax = fig.add_subplot(1, 1, 1)
ax.set_title("Solute concentration: Various sites",loc='left')
plt.xlabel("Time [d]")
plt.ylabel("Concentration [kg m$^-$$^3$]")
# Logarityhmic y-axis plots
plt.semilogy(timesteps,conc[:,0,irowy_leak,jcolx_leak],
             color='red',label='Leak site',linewidth=lW)

plt.semilogy(timesteps,conc[:,0,1,jcolx_leak],
             color='black',linestyle='solid',
             label='Northern boundary',linewidth=lW)

plt.semilogy(timesteps,conc[:,0,-2,jcolx_leak],
             color='black',linestyle='dashed',
             label='Southern boundary',linewidth=lW)

plt.semilogy(timesteps,conc[:,2,irowy_ex1,jcolx_ex1],
             color='yellow',linestyle='solid',
             label='Extraction Well 1',linewidth=lW)

plt.semilogy(timesteps,conc[:,2,irowy_ex2,jcolx_ex2],
             color='yellow',linestyle='dashed',
             label='Extraction Well 2',linewidth=lW)

plt.semilogy(timesteps,conc[:,2,irowy_ex3,jcolx_ex3],
             color='blue',linestyle='solid',
             label='Extraction Well 3',linewidth=lW)

plt.semilogy(timesteps,conc[:,2,irowy_ex4,jcolx_ex4],
             color='blue',linestyle='dashed',
             label='Extraction Well 4',linewidth=lW)
# Natural y-axis plots
# plt.plot(timesteps,conc[:,0,irowy_leak,jcolx_leak],label='Leak site')
# plt.plot(timesteps,conc[:,0,irowy_leak,0],label='Lake')
# plt.plot(timesteps,conc[:,0,irowy_leak,-1],label='River')
# plt.plot(timesteps,conc[:,1,irowy_ex1,jcolx_ex1],label='Extraction Well 1')
# plt.plot(timesteps,conc[:,1,irowy_ex2,jcolx_ex2],label='Extraction Well 2')
plt.ylim([10e-3,100])
plt.grid()
plt.legend()

# =============================================================================
# ======= END OF SCRIPT ======= END OF SCRIPT ======= END OF SCRIPT ===========
# =============================================================================
