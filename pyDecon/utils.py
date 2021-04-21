
def scaleImage(image, dtype=np.uint8):
    image = np.array(image)
    scaled = image / image.max()
    scaled_255 = scaled * (np.iinfo(dtype).max)

    scaled_255_8bit = scaled_255.astype(dtype)
    # output = scaled_255_8bit
    return scaled_255_8bit


def cropND(img, centre, window):

    centre = np.array(centre)
    centre_dist = np.divide(window, 2)
    shape = img.shape
    crops = []
    # for i, dim in enumerate(shape):
    #     l = (centre[-(i + 1)] - np.floor(centre_dist[-(i + 1)])).astype(int)
    #     r = (centre[-(i + 1)] + np.ceil(centre_dist[-(i + 1)])).astype(int)

    x_l = (centre[-1] - np.floor(centre_dist[-1])).astype(int)
    x_r = (centre[-1] + np.ceil(centre_dist[-1])).astype(int)

    y_l = (centre[-2] - np.floor(centre_dist[-2])).astype(int)
    y_r = (centre[-2] + np.ceil(centre_dist[-2])).astype(int)

    z_l = (centre[-3] - np.floor(centre_dist[-3])).astype(int)
    z_r = (centre[-3] + np.ceil(centre_dist[-3])).astype(int)
    # try:
    #     return util.crop(img,((z_l,z_r),(y_l,y_r),(x_l,x_r)))
    # except :
    #     return

    return img[z_l:z_r, y_l:y_r, x_l:x_r]


def cropNDv(img, centres, window=[20, 40, 40]):
    cropped_list = np.full((len(centres), *window), np.nan)
    i = 0
    centres_list = []
    for centre in centres:
        try:
            cropped_list[i, :, :, :] = cropND(img, centre, window=window)
            centres_list.append(centres[i])
            i += 1
        except:
            None
    return cropped_list, centres_list

def xyz_viewer(im):
    fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    ax[0].imshow(np.max(im, axis=0))
    ax[1].imshow(np.max(im, axis=1))
    ax[2].imshow(np.max(im, axis=2))
    plt.show()
