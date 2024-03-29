import cv2 as cv

def kmeans(slika, k=3, iteracije=10):
    '''Izvede segmentacijo slike z uporabo metode k-means.'''
    # Spremenimo obliko slike v 2D
    slika_2d = slika.reshape((-1, 3))

def meanshift(slika, velikost_okna, dimenzija):
    '''Izvede segmentacijo slike z uporabo metode mean-shift.'''
    pass

def izracunaj_centre(slika, izbira, dimenzija_centra, T):
    '''Izračuna centre za metodo kmeans.'''
    pass

if __name__ == "__main__":
    pass