import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
plt.rcParams["image.origin"] = 'lower'
from matplotlib.ticker import LogLocator
from scipy.spatial import cKDTree
import random


class custom_MCFOST_density():
#
    def __init__(self, grid_fits, mode):


        if mode == 'MCFOST':

            hdul_grid = fits.open(grid_fits)
            self.data_grid = hdul_grid[0].data
            self.num_dim = np.shape(self.data_grid)[0]
            self.mas_to_AU = 0.1
            self.pixel_ratio = 1

            if self.num_dim == 3:

                self.r = self.data_grid[0, ]
                self.z = self.data_grid[1,]
                self.theta = self.data_grid[2,]

            elif self.num_dim == 2:

                self.r = self.data_grid[0,]
                self.z = self.data_grid[1,]
                self.theta = np.pi/2

            self.x = self.r*np.cos(self.theta)
            self.y = self.r*np.sin(self.theta)
            self.z = self.z


        elif mode == 'manual':

            hdul_grid = fits.open(grid_fits)
            self.data_grid = hdul_grid[0].data
            self.num_dim = np.shape(self.data_grid)[0]

            if self.num_dim == 3:

                self.r = self.data_grid[0, ]
                self.z = self.data_grid[1,]
                self.theta = self.data_grid[2,]

                self.r = np.arange(np.min(self.r), np.max(self.r), (np.max(self.r) - np.min(self.r))/np.shape(self.r)[2])
                self.r = np.tile(self.r, (140, 1))
                self.r = np.expand_dims(self.r, axis = 0)

                self.z = np.arange(np.min(self.z), np.max(self.z), (np.max(self.z) - np.min(self.z))/np.shape(self.z)[1])
                self.z = np.tile(np.expand_dims(self.z, axis = 1),  (1, 100))
                self.z = np.expand_dims(self.z, axis = 0)

                self.theta = self.data_grid[2,]


            self.x = self.r * np.cos(self.theta)
            self.y = self.r * np.sin(self.theta)
            self.z = self.z

        self.rad_dist_cent = np.sqrt(self.x**2 + self.y**2 + self.z**2)

        self.final_gauss_plot = 0


    def blob(self, xo, yo, zo, rblobx, rbloby, rblobz, beta, blob_contrast, star_radius_au):

        sigmax = rblobx
        sigmay = rbloby
        sigmaz = rblobz

        def_gauss = np.exp(-((self.x - xo) ** 2 / (2 * sigmax ** 2) + (self.y - yo) ** 2 / (2 * sigmay ** 2) + (self.z - zo) ** 2 / (
                    2 * sigmaz ** 2))) ** (1 / beta)

        def_gauss[self.rad_dist_cent < star_radius_au] = 0

        final_gauss = def_gauss/def_gauss.max()
        final_gauss = final_gauss*blob_contrast

        return final_gauss

    def ellipse(self, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z, beta, alpha, dust_pl_exp, ellipse_cont, star_radius_au):

        angle_radians = np.deg2rad(alpha)
        x_rotated = self.x * np.cos(angle_radians) - self.y * np.sin(angle_radians)
        y_rotated = self.x * np.sin(angle_radians) + self.y * np.cos(angle_radians)

        exponent = -0.5 * ((x_rotated - mu_x) ** 2 / sigma_x ** 2 + (y_rotated - mu_y) ** 2 / sigma_y ** 2  + (self.z - mu_z) ** 2 / sigma_z ** 2  )  ** (1 / beta)
        gauss =  1 - np.exp(exponent)

        rad_stuff = np.sqrt(self.r**2 + self.z**2)
        rad_stuff_sq = rad_stuff**(-dust_pl_exp)

        final_gauss = gauss #* rad_stuff_sq # this doesnt look like its working properly
        final_gauss[rad_stuff < 2*sigma_x] = 0 # i think this unti is in AU
        final_gauss[rad_stuff > (2*sigma_x + 0.1)] = 0
        final_gauss = final_gauss/final_gauss.max()
        final_gauss = final_gauss * ellipse_cont

        return final_gauss


    def spiral(self, turns, max_radius, sigma, density, star_radius):

        x_grid = self.x
        y_grid = self.y
        z_grid = self.z

        threshold = 0.9        #0.35

        z_center = 0    # Center the spiral at z = 0
        z_range = sigma * 2  # Confine the spiral within +/- one standard deviation

        # Generate the spiral path within the defined z-range
        num_points = int(z_range * 100)  # Densely sample within z_range
        zs = np.linspace(z_center - sigma, z_center + sigma, num=num_points)
        theta = zs * (2 * np.pi * turns / z_range)
        radius = (zs - zs[0]) / z_range * max_radius
        spiral_x = radius * np.cos(theta)
        spiral_y = radius * np.sin(theta)
        spiral_z = zs

        # Build KD-tree for efficient nearest neighbor search
        spiral_points = np.column_stack((spiral_x, spiral_y, spiral_z))
        tree = cKDTree(spiral_points)

        # Flatten the grids and query the closest distance to any spiral point
        query_points = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))
        distances, _ = tree.query(query_points)

        # Compute Gaussian values
        gaussian_values = np.exp(-distances**2 / (2 * sigma**2))
        gaussian_values = gaussian_values.reshape(x_grid.shape)

        # Apply threshold
        gaussian_values[gaussian_values < threshold] = 0
        gaussian_values[self.rad_dist_cent < star_radius] = 0
        gaussian_values_first = (gaussian_values/gaussian_values.max()) * density



 ###################

        x_grid = -self.x
        y_grid = -self.y
        z_grid = self.z

        threshold = 0.9 #0.35

        z_center = 0  # Center the spiral at z = 0
        z_range = sigma * 2  # Confine the spiral within +/- one standard deviation

        # Generate the spiral path within the defined z-range
        num_points = int(z_range * 100)  # Densely sample within z_range
        zs = np.linspace(z_center - sigma, z_center + sigma, num=num_points)
        theta = zs * (2 * np.pi * turns / z_range)
        radius = (zs - zs[0]) / z_range * max_radius
        spiral_x = radius * np.cos(theta)
        spiral_y = radius * np.sin(theta)
        spiral_z = zs

        # Build KD-tree for efficient nearest neighbor search
        spiral_points = np.column_stack((spiral_x, spiral_y, spiral_z))
        tree = cKDTree(spiral_points)

        # Flatten the grids and query the closest distance to any spiral point
        query_points = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))
        distances, _ = tree.query(query_points)

        # Compute Gaussian values
        gaussian_values = np.exp(-distances ** 2 / (2 * sigma ** 2))
        gaussian_values = gaussian_values.reshape(x_grid.shape)

        # Apply threshold
        gaussian_values[gaussian_values < threshold] = 0
        gaussian_values[self.rad_dist_cent < star_radius] = 0
        gaussian_values_second = (gaussian_values / gaussian_values.max()) * density


        gaussian_values = gaussian_values_first + gaussian_values_second

        return  gaussian_values



    def cart_spiral(self,  turns, max_radius, sigma, density, star_radius_pix):

        threshold = 0.35
        size = 101
        x, y, z = np.mgrid[-np.floor(size / 2): np.floor(size / 2) + 1,
                  -np.floor(size / 2): np.floor(size / 2) + 1,
                  -np.floor(size / 2): np.floor(size / 2) + 1]

        # Convert to float32 for consistency with your requirement
        x_grid = x.astype(np.float32)
        y_grid = y.astype(np.float32)
        z_grid = z.astype(np.float32)

        rad_stuff = np.sqrt(x_grid**2 + y_grid**2 + z_grid**2)

        # Define the center and range for the spiral in the z-axis
        z_center = 0  # Center the spiral at z = 0
        z_range = sigma * 2  # Confine the spiral within +/- one standard deviation

        # Generate the spiral path within the defined z-range
        num_points = int(z_range * 100)  # Densely sample within z_range
        zs = np.linspace(z_center - sigma, z_center + sigma, num=num_points)
        theta = zs * (2 * np.pi * turns / z_range)
        radius = (zs - zs[0]) / z_range * max_radius
        spiral_x = radius * np.cos(theta)
        spiral_y = radius * np.sin(theta)
        spiral_z = zs

        # Build KD-tree for efficient nearest neighbor search
        spiral_points = np.column_stack((spiral_x, spiral_y, spiral_z))
        tree = cKDTree(spiral_points)

        # Flatten the grids and query the closest distance to any spiral point
        query_points = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))
        distances, _ = tree.query(query_points)

        # Compute Gaussian values
        gaussian_values = np.exp(-distances ** 2 / (2 * sigma ** 2))
        gaussian_values = gaussian_values.reshape(x_grid.shape)


        gaussian_values[gaussian_values < threshold] = 0
        gaussian_values[rad_stuff < star_radius_pix] = 0
        gaussian_values = (gaussian_values/gaussian_values.max()) * density

        return gaussian_values



    def what_mcfost_sees(self, obj, grid, i):

        # convert these to pixels?
        # you can plot plus the sd and minus the sd


        ind = np.argwhere((self.z > (obj.zo_AU[i] - obj.model_params['sigmaz_AU'][i]/2)) & (self.z < (obj.zo_AU[i] + obj.model_params['sigmaz_AU'][i]/2))) # zo is in pixels, and self.z is in AU. # converf

        #
        # relz = self.z[zind[:,0], zind[:,1], zind[:,2]]
        # arg_ind = np.argsort(relz)[0] # index of where z is zo
        #
        #
        #
        # # why doesnt this work.
        # indx = zind[arg_ind, 1] # index 0, 1 , 2
        # indx = 50

        ind_i = ind[:,0]
        ind_j = ind[:,1]
        ind_k = ind[:,2]

            # we need to convert y and x back to cartesian before this;

        ycart = self.y[ind_i, ind_j, ind_k] / self.mas_to_AU * self.pixel_ratio  # check units
        xcart = self.x[ind_i, ind_j, ind_k] / self.mas_to_AU * self.pixel_ratio
        gridcart = grid[ind_i, ind_j, ind_k] / self.mas_to_AU * self.pixel_ratio

        return  ycart, xcart , gridcart



    def cart_blob(self, xo, yo, zo, rblobx, rbloby, rblobz, beta, blob_contrast, star_rad_pix):


        sigmax = rblobx
        sigmay = rbloby
        sigmaz = rblobz


        x, y, z = np.ogrid[- np.floor(101 / 2): np.floor(101 / 2) +1 ,
                  - np.floor(101 / 2): np.floor(101 / 2) +1 ,
                  - np.floor(101 / 2): np.floor(101 / 2) +1] # changed from 101, got rid of +1

        xx = x.astype(np.float32)
        yy = y.astype(np.float32)
        zz = z.astype(np.float32)

        def_gaussplot = np.exp(-((xx - xo) ** 2 / (2 * sigmax ** 2) + (yy - yo) ** 2 / (2 * sigmay ** 2) + (zz - zo) ** 2 / (
                2 * sigmaz ** 2))) ** (1 / beta)

        rad_dist_cent = np.sqrt(xx**2 + yy**2 + zz**2)
        def_gaussplot[rad_dist_cent < star_rad_pix] = 0

        self.final_gauss_plot = self.final_gauss_plot + (def_gaussplot / def_gaussplot.max()) * blob_contrast
        return self.final_gauss_plot


    def cart_ellipse(self, xo, yo, zo, rellipsex, rellipsey, rellipsez, beta, alpha, dustplexp, dens, starrad):

        sigma_x = rellipsex
        sigma_y = rellipsey
        sigma_z = rellipsez

        mu_x = xo
        mu_y = yo
        mu_z = zo

        x, y, z = np.ogrid[- np.floor(101 / 2): np.floor(101 / 2) + 1,
                  - np.floor(101 / 2): np.floor(101 / 2) + 1,
                  - np.floor(101 / 2): np.floor(101 / 2) + 1]

        xx = x.astype(np.float32)
        yy = y.astype(np.float32)
        zz = z.astype(np.float32)

        angle_radians = np.deg2rad(alpha)
        x_rotated = xx * np.cos(angle_radians) - yy * np.sin(angle_radians)
        y_rotated = xx * np.sin(angle_radians) + yy * np.cos(angle_radians)

        exponent = -0.5 * ((x_rotated - mu_x) ** 2 / sigma_x ** 2 + (y_rotated - mu_y) ** 2 / sigma_y ** 2 + (
                    zz - mu_z) ** 2 / sigma_z ** 2) ** (1 / beta)
        gauss = 1 - np.exp(exponent)

        rad_stuff = np.sqrt(xx ** 2 + yy ** 2 + zz**2)
        rad_stuff_sq = rad_stuff ** (-dustplexp)

        final_gauss = gauss #* rad_stuff_sq
        final_gauss[rad_stuff  < 2*sigma_x] = 0
        final_gauss[rad_stuff  > 2*sigma_x + 0.1] = 0

        self.final_gauss_plot = self.final_gauss_plot + (final_gauss / final_gauss.max()) * dens

        return self.final_gauss_plot




    def cart_cylinder(self, xo, yo, zo, rellipsex, rellipsey, rellipsez, beta, alpha, dustplexp, dens, starrad):


        sigma_x = rellipsex
        sigma_y = rellipsey
        sigma_z = rellipsez

        mu_x = xo
        mu_y = yo
        mu_z = zo

        x, y, z = np.ogrid[- np.floor(101 / 2): np.floor(101 / 2) + 1,
                  - np.floor(101 / 2): np.floor(101 / 2) + 1,
                  - np.floor(101 / 2): np.floor(101 / 2) + 1]

        xx = x.astype(np.float32)
        yy = y.astype(np.float32)
        zz = z.astype(np.float32)

        angle_radians = np.deg2rad(alpha)
        x_rotated = xx * np.cos(angle_radians) - yy * np.sin(angle_radians)
        y_rotated = xx * np.sin(angle_radians) + yy * np.cos(angle_radians)

        exponent = -0.5 * ((x_rotated - mu_x) ** 2 / sigma_x ** 2 + (y_rotated - mu_y) ** 2 / sigma_y ** 2 + (
                zz - mu_z) ** 2 / sigma_z ** 2) ** (1 / beta)
        gauss = 1 - np.exp(exponent)

        rad_stuff = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
        rad_stuff_sq = rad_stuff ** (-dustplexp)

        final_gauss = gauss * rad_stuff_sq

        final_gauss[rad_stuff < starrad] = 0 # nothing in central star
        red_rad = np.sqrt(xx**2 + yy**2 + 0*zz)
        final_gauss[red_rad > starrad] = 0 # only stuff in front or behind of star
        zzonly = 0*xx + 0*yy + zz
        final_gauss[zzonly > 0] = 0 # only stuff in front of star.

        self.final_gauss_plot = self.final_gauss_plot + (final_gauss / final_gauss.max()) * dens

        return self.final_gauss_plot



    def cylinder(self, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z, beta, alpha, dust_pl_exp, ellipse_cont, star_radius_au, star_radius_pix):


        angle_radians = np.deg2rad(alpha)
        x_rotated = self.x * np.cos(angle_radians) - self.y * np.sin(angle_radians)
        y_rotated = self.x * np.sin(angle_radians) + self.y * np.cos(angle_radians)

        exponent = -0.5 * ((x_rotated - mu_x) ** 2 / sigma_x ** 2 + (y_rotated - mu_y) ** 2 / sigma_y ** 2 + (
                    self.z - mu_z) ** 2 / sigma_z ** 2) ** (1 / beta)
        gauss = 1 - np.exp(exponent)

        rad_stuff = np.sqrt(self.r ** 2 + self.z ** 2)
        rad_stuff_sq = rad_stuff ** (-dust_pl_exp)

        final_gauss = gauss * rad_stuff_sq
        final_gauss[self.rad_dist_cent < star_radius_au] = 0

        raddist = np.sqrt(self.x**2 + self.y**2 + 0*self.z)
        final_gauss[raddist > star_radius_pix] = 0
        final_gauss[self.z > 0] = 0


        final_gauss = final_gauss / final_gauss.max()
        final_gauss = final_gauss * ellipse_cont
        return final_gauss




    def write_to_densityfile(self, grid_array, save_path):

        if len(np.shape(grid_array)) < 4:
            grid_array = np.expand_dims(grid_array, axis = 0)

        hdu1 = fits.PrimaryHDU(grid_array)
        density_file = fits.HDUList([hdu1])
        density_file[0].header["read_n_a"] = 0 # this means that it will normalise itself
        density_file.writeto(save_path, overwrite=True)

