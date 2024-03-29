import cv2 as cv
import numpy as np

def kmeans(slika, k=3, iteracije=10):
    '''Izvede segmentacijo slike z uporabo metode k-means.'''
    # Spremenimo obliko slike v 2D
    slika_2d = slika.reshape((-1, 3))

    # Izračunamo centre
    centri = izracunaj_centre(slika_2d, k, slika_2d.shape[1], iteracije)

    for _ in range(iteracije):
        # Dodamo vsak pixel do najbližjega centra
        labele = np.argmin(np.linalg.norm(slika_2d[:, None] - centri, axis=-1), axis=-1)

        # Posodobimo centre
        for i in range(k):
            centri[i] = np.mean(slika_2d[labele == i], axis=0)

    # Spremenimo vsak piksel v sliki na njegov center
    segmentirana_slika_2d = centri[labele]

    # Spremenimo obliko slike nazaj v 3D
    segmentirana_slika = segmentirana_slika_2d.reshape(slika.shape)
    

def meanshift(slika, velikost_okna, dimenzija):
    '''Izvede segmentacijo slike z uporabo metode mean-shift.'''
    pass

def izracunaj_centre(slika, izbira, dimenzija_centra, T):
    '''Izračuna centre za metodo kmeans.'''
    pass

if __name__ == "__main__":
    pass