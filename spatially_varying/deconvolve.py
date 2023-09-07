
from scipy.signal import convolve
import numpy as np

def richardson_lucy_varying_1PC(
    small_image,
    cropping_grid,
    eigen_psf,
    pc_component_weight_map,
    mu,
    iterations=25,
    show_image=True,
    ):
    y = small_image
    grid_psf = cropping_grid
    x_t = np.ones_like(y) * 0.5

    x_t = y.copy()

    c_0 = mu
    c_1 = eigen_psf

    c_0_T = np.flip(c_0)
    c_1_T = np.flip(c_1)

    d_0 = np.ones_like(x_t)
    d_1 = pc_component_weight_map

    d_0_T = d_0
    d_1_T = d_1
    # TODO make this smarter
    x_t_crop = x_t[grid_psf[0], grid_psf[1], grid_psf[2]]
    # xyz_viewer(x_t_crop)
    # plt.show()

    ones = np.ones_like(y)

    print("Begin deconvolution")
    for i in range(iterations):
        c_0_x_t = convolve(x_t, c_0, mode="same")
        c_1_x_t = convolve(x_t, c_1, mode="same")

        d_0_c_0_x_t = np.multiply(d_0, c_0_x_t)
        d_1_c_1_x_t = np.multiply(d_1, c_1_x_t)

        d_c_x_t = np.add(d_0_c_0_x_t, d_1_c_1_x_t)
        y_over_d_c_x_t = np.divide(y, d_c_x_t)

        c_0_T_y_over_d_c_x_t = convolve(y_over_d_c_x_t, c_0_T, mode="same")
        c_1_T_y_over_d_c_x_t = convolve(y_over_d_c_x_t, c_1_T, mode="same")

        d_0_T_c_0_y_over_d_c_x_t = np.multiply(d_0_T, c_0_T_y_over_d_c_x_t)
        d_1_T_c_1_y_over_d_c_x_t = np.multiply(d_1_T, c_1_T_y_over_d_c_x_t)

        d_c_y_over_d_c_x_t = np.add(d_0_T_c_0_y_over_d_c_x_t, d_1_T_c_1_y_over_d_c_x_t)

        c_0_T_ones = convolve(ones, c_0_T, mode="same")
        c_1_T_ones = convolve(ones, c_1_T, mode="same")

        d_0_T_c_0_T_ones = np.multiply(d_0_T, c_0_T_ones)
        d_1_T_c_1_T_ones = np.multiply(d_1_T, c_1_T_ones)

        d_T_c_T_ones = np.add(d_0_T_c_0_T_ones, d_1_T_c_1_T_ones)
        d_c_y_over_d_c_x_t_over_d_T_c_T_ones = np.divide(
            d_c_y_over_d_c_x_t, d_T_c_T_ones
        )
        x_t = np.multiply(x_t, d_c_y_over_d_c_x_t_over_d_T_c_T_ones)
        x_t_crop = x_t[grid_psf[0], grid_psf[1], grid_psf[2]]

        # xyz_viewer(x_t)
        if show_image:
            # xyz_viewer(x_t_crop)
            plt.show()
        print(f"Iteration {str(i)}")

    return x_t_crop
