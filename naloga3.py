import cv2 as cv
import numpy as np
centri = []

def click_event(event, x, y, flags, param):
    # Preverimo, če je uporabnik kliknil levi gumb miške
    if event == cv.EVENT_LBUTTONDOWN:
        centri.append([x, y])

def kmeans(slika, k=3, iteracije=10):
    '''Izvede segmentacijo slike z uporabo metode k-means.'''
    # Spremenimo obliko slike v 2D
    slika_2d = slika.reshape((-1, 3))

    # Izračunamo centre
    centri = izracunaj_centre(slika_2d, "nakljucno", slika_2d.shape[1], iteracije)

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

def gaussovo_jedro(velikost, sigma):
    '''Vrne 2D Gaussovo jedro.'''
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (velikost / sigma) ** 2)

def meanshift(slika, velikost_okna, dimenzija):
    '''Izvede segmentacijo slike z uporabo metode mean-shift.'''
    # Spremenimo obliko slike v 2D
    tocke = slika.reshape((-1), dimenzija)
    
    nove_tocke = np.copy(tocke)

    for _ in range(100):
        for i, tocka in enumerate(tocke):
            razdalje = np.linalg.norm(tocka - tocke, axis=1)
            utezi = gaussovo_jedro(razdalje, velikost_okna)
            nova_tocka = np.sum(tocke * utezi[:, None], axis=0) / np.sum(utezi)
            if np.linalg.norm(nova_tocka - nove_tocke[i]) < 1e-3:  # konvergenca
                break
            nove_tocke[i] = nova_tocka
    
    return nove_tocke.reshape(slika.shape)

def izracunaj_centre(slika, izbira, dimenzija_centra, T):
    '''Izračuna centre za metodo kmeans.'''
    # Definiramo globalno spremenljivko centri
    global centri
    if izbira == 'rocno':
        # Ročna izbira centrov
        centri = []
        cv.imshow('Slika', slika)
        # Počakamo na klik uporabnika
        cv.setMouseCallback('Slika', click_event)
        cv.waitKey(0)
        cv.destroyAllWindows()
    elif izbira == 'nakljucno':
        # Naključna izbira centrov
        centri = []
        while len(centri) < dimenzija_centra:
            # Naključno izberemo center
            center = np.random.rand(3) * 255
            # Preverimo, če je center dovolj oddaljen od ostalih centrov
            if all(np.linalg.norm(centri - center) > T for center in centri):
                centri.append(center)
    else:
        raise ValueError('Izbira centrov ni veljavna.')
    
    return np.array(centri)

if __name__ == "__main__":
    # Preberemo sliko
    slika = cv.imread('.utils/zelenjava.jpg')

    # Izvedemo segmentacijo slike z metodo k-means
    segmentirana_slika = kmeans(slika, k=3, iteracije=10)

    # Izvedemo segmentacijo slike z metodo mean-shift
    segmentirana_slika_ms = meanshift(slika, velikost_okna=50, dimenzija=3)

    # Prikažemo originalno in segmentirano sliko
    cv.imshow('Originalna slika', slika)
    cv.imshow('Segmentirana slika (K-means)', segmentirana_slika)
    cv.imshow('Segmentirana slika (Mean-shift)', segmentirana_slika_ms)
    cv.waitKey(0)
    cv.destroyAllWindows()


    