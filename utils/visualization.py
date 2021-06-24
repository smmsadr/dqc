import imageio
import glob


def animator(location="."):
    number_of_files = len(glob.glob(location + '/*.png'))

    filenames = [location + '/step_' + str(i) + '.png' for i in range(number_of_files)]

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(location + '/' + 'animation.gif', images, fps=8)