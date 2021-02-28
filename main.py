# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import helpers.optimizer as op
import constants.constants_2d_0 as const
import functions.functions_2d_fix as func

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


optimizer = op.BetasOptimizer(const.ALPHAS_INIT,const.FLAT_IMAGES[0],
                           const.SIGMA_G_INV, const.TEMPLATE_SD2,
                           const.SD_INIT,
                           func.PIXEL_G_CENTERS_MATRIX,
                           const.P_CENTERS)



res = op.test_gradient(5,const.KG,const.KP,const.IMAGE1,
                 func.PIXEL_G_CENTERS_MATRIX,
                 const.P_CENTERS)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
