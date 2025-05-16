##

class Default_parameters:

    def __init__(self):
        # number of photon packages
        self.nbr_photons_eq_th = "1.28e5"
        self.nbr_photons_lambda = "1.28e3"
        self.nbr_photons_image = "1.28e5"

        # wavelength
        # do not change these unless you know what you're doing
        self.n_lambda = "50"
        self.lambda_min = "0.1"
        self.lambda_max = "3000.0"  # mum

        self.compute_temp = 'T'
        self.compute_sed = 'T'
        self.default_wavelength = 'T'  # use default wavelength grid for output?

        self.wavelength_file = "IMLup.lambda"  # if previous parameter is F

        self.separation_of_different_contributions = 'F'
        self.stokes_parameters = 'T'

        # grid geometry and size
        self.geometry = '2'  # 1 = cylindrical, 2 = spherical, 3 = Voronoi tesselation (this is in beta, please ask Christophe)

        self.n_rad = "100"  # log distribution
        self.nz = "70"  # (or n_theta)
        self.n_az = "70"
        self.n_rad_in = "20"

        # maps
        self.grid_nx = "300"
        self.grid_ny = "300"
        self.size = "300."  # AU

        # RT
        self.imin = "0."
        self.imax = "0."
        self.n_incl = "1"
        self.centered = 'F'

        self.az_min = "0"
        self.az_max = "0."
        self.n_az_angles = "1"

        self.distance = "1000.0"  # pc
        self.disk_PA = "0."

        # Scattering method
        self.scattering_method = "0"  # 0=auto, 1=grain prop, 2=cell prop
        self.mie_hg = "1"  # 1=Mie, 2=hg (2 implies the loss of polarizarion)

        # Symmetries
        self.image_symmetry = 'F'
        self.central_symmetry = 'F'
        self.axial_symmetry = 'F'  # (important only if N_phi > 1)

        # Disk physics
        self.dust_settling = "0"  # (0=no settling, 1=parametric, 2=Dubrulle, 3=Fromang)
        self.exp_strat = "0.50"
        self.a_strat = "1.0"  # (for parametric settling)

        self.dust_radial_migration = 'F'
        self.sublimate_dust = 'F'
        self.hydrostatic_equilibrium = 'F'

        self.viscous_heating = 'F'
        self.alpha_viscosity = "1e-5"

        # Number of zones: 1 zone = 1 density structure + corresponding grain properties
        self.number_of_zones = "1"

        # Density structure
        self.zone_type = "3"  # zone type : 1 = disk, 2 = tappered-edge disk, 3 = envelope, 4 = debris disk, 5 = wall

        self.dust_mass = "1.e-7"
        self.gas_to_dust_mass_ratio = "100."

        self.scale_height = "10."
        self.reference_radius = "100.0"  # AU, unused for envelope
        self.vertical_profile_exponent = "2"  # only for debris disk

        self.Rin = "0.931" # what unit is this in
        self.edge = "0.0"
        self.Rout = "50."
        self.Rc = "100."  # AU, Rc is only used for tappered-edge & debris disks (Rout set to 8*Rc if Rout==0)

        self.flaring_exponent = "1.125"  # unused for envelope
        self.surface_density_exponent = "-0.5"  # (or -gamma for tappered-edge disk or volume density for envelope), usually < 0
        self.negative_gamma_exp = "0.0"  # or alpha_in & alpha_out for debris disk

        # Grain properties
        self.number_of_species = "1"

        self.grain_type = "Mie"  # Mie of DHS
        self.N_components = "1"
        self.mixing_rule = "2"  # 1 = EMT or 2 = coating
        self.porosity = "0.0"
        self.max_fraction = "1.0"
        self.Vmax = "0.9"  # for DHS

        self.optical_indicies_file = "Draine_Si_sUV.dat"
        self.volume_fraction = "1.0"

        self.heating_method = "1"  # 1 = RE + LTE, 2 = RE + NLTE, 3 = NRE

        self.amin = "0.03"  # mum
        self.amax = "1000"  # mum
        self.aexp = "3.5"
        self.n_grains = "100"  # log distribution

        # Molecular RT settings
        self.lpop = 'T'
        self.laccurate_pop = 'T'
        self.LTE = 'T'
        self.profile_width = "15."  # km.s^-1

        self.v_turb = "0.2"  # delta

        self.nmol = "1"

        self.molecular_data_filename = "co@xpol.dat"
        self.level_max = "6"

        self.vmax = "1000.0"  # km.s^-1
        self.n_speed = "101"

        self.cst_molecule_abundance = 'T'
        self.abundance = "1.e-6"
        self.abundance_file = "abundance.fits.gz"

        self.ray_tracing = 'T'
        self.number_lines_in_RT = "2"

        self.transition_number_1 = "1"
        self.transition_number_2 = "2"
        self.transition_number_3 = "3"



        self.number_of_atoms = '1'
        self.all_levels = 'H_6.atom'
        self.non_LTE = 'F'
        self.init_solution = '0'

        self.vmax = '10'
        self.n_points = '101'
        self.images = 'T'
        self.num_lines = '1'
        self.upper_level = '3'
        self.lower_level = '2'




        # Star properties
        self.number_of_stars = "1"
        self.temp = "3750.0"
        self.radius = "1000"  # solar radius
        self.mass = "25"  # solar mass
        self.x = "0.0"  # AU
        self.y = "0.0"  # AU
        self.z = "0.0"  # AU
        self.is_blackbody = 'F'
        self.fUV = "0.1"
        self.slope_FUV = "2.2"