from astropy.io import fits
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
import scipy.ndimage as ndi

# Paramètres ajustables
attenuation_factor = 0.4  # Facteur d'atténuation pour les grosses étoiles (0.0 = aucune, 1.0 = pleine intensité)
mask_dilate_size = 5  # Taille du kernel de dilatation pour le masque (plus petit = moins de couverture)
blur_kernel_size = 21  # Taille du kernel de flou gaussien (plus grand = plus doux)
blur_sigma = 5  # Sigma pour le flou gaussien (plus grand = plus diffus)
fits_file = './examples/HorseHead.fits'
hdul = fits.open(fits_file)

# Display information about the file
hdul.info()

# Access the data from the primary HDU
data = hdul[0].data

# Access header information
header = hdul[0].header

# Handle both monochrome and color images
if data.ndim == 3:
    # Color image - need to transpose to (height, width, channels)
    if data.shape[0] == 3:  # If channels are first: (3, height, width)
        data = np.transpose(data, (1, 2, 0))
    # If already (height, width, 3), no change needed

    # Normalize the entire image to [0, 1] for matplotlib
    data_normalized = (data - data.min()) / (data.max() - data.min())

    # Save the data as a png image (no cmap for color images)
    plt.imsave('./results/original.png', data_normalized)

    # Normalize each channel separately to [0, 255] for OpenCV
    image = np.zeros_like(data, dtype='uint8')
    for i in range(data.shape[2]):
        channel = data[:, :, i]
        image[:, :, i] = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype('uint8')

    # Convert to grayscale for star detection
    data = np.mean(data, axis=2)
else:
    # Monochrome image
    plt.imsave('./results/original.png', data, cmap='gray')

    # Convert to uint8 for OpenCV
    image = ((data - data.min()) / (data.max() - data.min()) * 255).astype('uint8')

# Étape A : Création du masque d'étoiles
# Utiliser DAOStarFinder pour détecter les étoiles
mean, median, std = sigma_clipped_stats(data, sigma=3.0)

daofind = DAOStarFinder(fwhm=5.0, threshold=median + 0.5*std)  # Baisser le seuil
sources = daofind(data)

# Créer un masque binaire pour les étoiles
mask = np.zeros_like(data, dtype=np.uint8)
if sources is not None:
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    for pos in positions:
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < data.shape[1] and 0 <= y < data.shape[0]:
            mask[y, x] = 255  # Marquer les étoiles

# Ajouter un seuillage pour les grosses étoiles (valeurs élevées)
thresh_value = median + 3 * std
bright_mask = (data > thresh_value).astype(np.uint8) * 255

# Combiner les masques : DAO + seuillage sur valeurs élevées
mask = cv.bitwise_or(mask, bright_mask)

# Dilater plus largement le masque pour couvrir les étoiles
kernel_dilate = np.ones((mask_dilate_size, mask_dilate_size), np.uint8)  # Plus grand kernel
mask = cv.dilate(mask, kernel_dilate, iterations=2)

# Dilater le masque des étoiles brillantes pour la réintégration
dilated_bright = cv.dilate(bright_mask, kernel_dilate, iterations=1)

# Étape B : Réduction localisée
# 1. Créer une version érodée de l'image originale (Ierode)
kernel_erosion = np.ones((5,5), np.uint8)
Ierode = cv.erode(image, kernel_erosion, iterations=2)

# 2. Créer un masque d'étoiles (M) avec bords adoucis par flou gaussien
M = mask.astype(np.float32) / 255.0  # Normaliser à [0, 1]
M = cv.GaussianBlur(M, (blur_kernel_size, blur_kernel_size), blur_sigma)  # Flou gaussien pour adoucir les bords
cv.imwrite('./results/blurred_mask.png', (M * 255).astype(np.uint8))  # Sauvegarder le masque flouté

# Pour les images couleur, étendre le masque à tous les canaux
if image.ndim == 3:
    M = np.stack([M] * image.shape[2], axis=2)
    dilated_bright = np.stack([dilated_bright] * image.shape[2], axis=2)

# 3. Calculer l'image finale Ifinal = (M * Ierode) + ((1 - M) * image)
Ifinal = (M * Ierode.astype(np.float32) + (1 - M) * image.astype(np.float32)).astype(np.uint8)

# 4. Calculer l'image finale avec réintégration des grosses étoiles atténuées
final = Ifinal.copy()
if sources is not None:
    threshold_flux = np.percentile(sources['flux'], 90)  # Seulement les 10% les plus brillantes
    for source in sources:
        if source['flux'] > threshold_flux:
            x, y = int(source['xcentroid']), int(source['ycentroid'])
            half_size = 15  # Taille du patch autour de l'étoile
            x1, x2 = max(0, x - half_size), min(image.shape[1], x + half_size)
            y1, y2 = max(0, y - half_size), min(image.shape[0], y + half_size)
            # Extraire le patch de l'image originale
            star_patch = image[y1:y2, x1:x2].astype(np.float32)
            # Atténuer l'étoile
            attenuated_star = star_patch * attenuation_factor
            # Ajouter au résultat final (maximum pour éviter les artefacts)
            final[y1:y2, x1:x2] = np.maximum(final[y1:y2, x1:x2], attenuated_star.astype(np.uint8))

# Sauvegarder les résultats
cv.imwrite('./results/eroded.png', Ierode)
cv.imwrite('./results/final.png', final)
cv.imwrite('./results/mask.png', mask)

# Afficher les informations
print(f"Nombre d'étoiles détectées : {len(sources) if sources is not None else 0}")

# Close the file
hdul.close()