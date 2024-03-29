import cv2 as cv
import numpy as np

def kmeans(slika, k=3, iteracije=10):
    '''Izvede segmentacijo slike z uporabo metode k-means.'''
    # Spremenimo obliko slike v 2D
    slika_2d = slika.reshape((-1, 3))

    # Izračunamo centre
    centri = izracunaj_centre(slika_2d, k, slika_2d.shape[1], iteracije)

    # Začnemo z iteracijami algoritma K-means. Število iteracij je določeno z vrednostjo 'iteracije'.
    for _ in range(iteracije):
    
        # Izračunamo Manhattanovo razdaljo med vsakim pikslom slike (slika_2d) in trenutnimi centri gruč.
        # np.abs(slika_2d[:, None] - centri).sum(axis=-1) izračuna razdaljo med vsakim pikslom in vsakim centrom.
        # np.argmin() nato vrne indeks centra, ki je najbližji vsakemu pikslu. Te indekse shranimo v 'labele'.
        labele = np.argmin(np.abs(slika_2d[:, None] - centri).sum(axis=-1), axis=-1)

        # Posodobimo centre gruč. Za vsako gručo (od 0 do k-1) izračunamo povprečje vseh pikslov, ki so bili dodeljeni tej gruči.
        # To povprečje postane nov center za to gručo.
        for i in range(k):
            centri[i] = np.mean(slika_2d[labele == i], axis=0)

    # Spremenimo vsak piksel v sliki na njegov center
    segmentirana_slika_2d = centri[labele]

    # Spremenimo obliko slike nazaj v 3D
    segmentirana_slika = segmentirana_slika_2d.reshape(slika.shape)

    # Vrnemo segmentirano sliko
    return segmentirana_slika.astype(slika.dtype)


def meanshift(slika, velikost_okna, dimenzija):
    '''Izvede segmentacijo slike z uporabo metode mean-shift.'''
    pass

def izracunaj_centre(slika, izbira, dimenzija_centra, T):
    '''Izračuna centre za metodo kmeans.'''
    pass

if __name__ == "__main__":
    pass