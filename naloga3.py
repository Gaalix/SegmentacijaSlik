import cv2 as cv
import numpy as np

def kmeans(slika, k=3, iteracije=10):
    # Izberite med 3 ali 5 za dimenzijo centra, odvisno od tega, ali želite upoštevati koordinate pikslov
    dimenzija_centra = 3
    T = 50  # Prag za izbiro naključnih centrov
    izbira = 'rocno'  # Način izbire centrov: 'rocno' ali 'nakljucno'

    # Inicializacija centrov glede na izbrano metodo in kriterije
    centri = izracunaj_centre(slika, izbira, dimenzija_centra, T, k)

    # Glavna zanka algoritma k-means
    for _ in range(iteracije):
        # Inicializacija matrike oznak za shranjevanje indeksa najbližjega centra za vsak piksel
        labels = np.zeros((slika.shape[0], slika.shape[1]), dtype=int)

        # Dodeljevanje vsakega piksla najbližjemu centru
        for i in range(slika.shape[0]):
            for j in range(slika.shape[1]):
                # Izberemo piksel in, če je potrebno, dodamo njegove koordinate
                piksel = slika[i, j] if dimenzija_centra == 3 else np.append(slika[i, j], [i, j])
                # Izračun razdalj od piksla do vseh centrov
                razdalje = np.linalg.norm(centri - piksel, axis=1)
                # Dodelitev piksla najbližjemu centru
                labels[i, j] = np.argmin(razdalje)

        # Posodobitev položajev centrov na podlagi povprečja pikslov, ki spadajo v posamezni segment
        for c in range(k):
            tocke = [slika[i, j] if dimenzija_centra == 3 else np.append(slika[i, j], [i, j])
                     for i in range(slika.shape[0]) for j in range(slika.shape[1]) if labels[i, j] == c]
            if tocke:
                centri[c] = np.mean(tocke, axis=0)

    # Ustvarjanje segmentirane slike na podlagi dodeljenih centrov
    segmentirana_slika = np.zeros_like(slika)
    for i in range(slika.shape[0]):
        for j in range(slika.shape[1]):
            segmentirana_slika[i, j] = centri[labels[i, j]][:3]  # Uporabi barve centrov, brez koordinat

    return segmentirana_slika

def gaussian_kernel(distance, bandwidth):
    # Gaussova funkcija za izračun uteži na podlagi razdalje
    return np.exp(-0.5 * ((distance / bandwidth) ** 2)) / (bandwidth * np.sqrt(2 * np.pi))

def calculate_distances(points, dimension):
    '''Izračuna razdalje med točkami glede na dimenzijo (3 za barvo, 5 za barvo in prostor).'''
    
    # Izračunamo razlike med točkami glede na dimenzijo
    if dimension == 3:
        # Izračunamo razlike samo na podlagi barvnih razlik
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        distance = np.sqrt(np.sum(diff ** 2, axis=2))  # Izračunamo evklidsko razdaljo
    elif dimension == 5:
        # Izračunamo razlike na podlagi barvnih in prostorskih razlik
        spatial_diff = points[:, np.newaxis, :2] - points[np.newaxis, :, :2]  # Prostorske razlike
        color_diff = points[:, np.newaxis, 2:] - points[np.newaxis, :, 2:]  # Barvne razlike
        spatial_distance = np.sqrt(np.sum(spatial_diff ** 2, axis=2))  # Evklidska razdalja za prostorske razlike
        color_distance = np.sqrt(np.sum(color_diff ** 2, axis=2))  # Evklidska razdalja za barvne razlike
        distance = spatial_distance + color_distance  # Skupna razdalja je vsota prostorske in barvne razdalje
    return distance

def find_clusters(points, distance_treshold):
    '''Najde skupine točk, ki so bližje kot določen prag.'''
    
    clusters = []  # Seznam za shranjevanje skupin
    for point in points:  # Za vsako točko
        found = False  # Zastavica za označevanje, ali je točka že v skupini
        for cluster in clusters:  # Za vsako skupino
            if np.linalg.norm(cluster[0] - point) < distance_treshold:  # Če je točka bližje skupini kot prag
                cluster.append(point)  # Dodamo točko v skupino
                found = True  # Označimo, da smo našli skupino za točko
                break  # Prekinemo iskanje skupin
        if not found:  # Če nismo našli skupine za točko
            clusters.append([point])  # Ustvarimo novo skupino s to točko
    print(f'Najdenih {len(clusters)} skupin točk.')  


def meanshift(slika, h, dimenzija):
    '''Izvede segmentacijo slike z uporabo metode mean-shift.'''
    # Preoblikujemo sliko v točke glede na dimenzijo
    if dimenzija == 3:
        points = slika.reshape((-1, 3))  # Za 3D sliko, preoblikujemo v obliko (-1, 3)
    elif dimenzija == 5:
        x = np.arange(slika.shape[1])  # Ustvarimo mrežo točk za 5D sliko
        y = np.arange(slika.shape[0])
        xv, yv = np.meshgrid(x, y)
        points = np.stack([yv.ravel(), xv.ravel(), slika[:,:,0].ravel(), slika[:,:,1].ravel(), slika[:,:,2].ravel()], axis=-1)

    # Izračunamo razdalje med točkami in uporabimo Gaussovo jedro za uteži
    distances = calculate_distances(points, dimenzija)
    weights = gaussian_kernel(distances, h)

    # Izvedemo mean-shift iteracijo
    for _ in range(5):
        # Izračunamo uteženo vsoto točk in nove točke
        weighted_sum = np.dot(weights, points)
        sum_weights = np.sum(weights, axis=1, keepdims=True)
        new_points = weighted_sum / sum_weights

        # Ponovno izračunamo razdalje in uteži za nove točke
        if dimenzija == 3:
            distances = calculate_distances(new_points, dimenzija)
        elif dimenzija == 5:
            distances = calculate_distances(new_points[:, 2:], 3)
        weights = gaussian_kernel(distances, h)

    # Najdemo gruče v novih točkah
    find_clusters(new_points, h * 0.1)

    # Vrnemo rezultat v obliki slike
    if dimenzija == 3:
        return new_points.reshape(slika.shape)  # Za 3D sliko, preoblikujemo nazaj v obliko slike
    elif dimenzija == 5:
        new_slika = np.zeros_like(slika)  # Za 5D sliko, ustvarimo novo sliko in dodelimo vrednosti novih točk
        for i, point in enumerate(new_points):
            x, y = int(points[i, 0]), int(points[i, 1])
            new_slika[x, y] = point[2:]
        return new_slika

def izracunaj_centre(slika, izbira, dimenzija_centra, T, k):
    '''Izračuna centre za metodo kmeans.'''
    if dimenzija_centra != 3 and dimenzija_centra != 5:
        raise ValueError('Centri morajo biti ali 3D ali 5D')
    centri = []

    def izberi_centre(event, x, y, flags, param):
        # Callback funkcija za ročni izbor centrov
        nonlocal k
        if event == cv.EVENT_LBUTTONDOWN and k > 0:
            # Dodajanje centra ob kliku leve miškine tipke
            if dimenzija_centra == 3:
                centri.append(slika[y, x])
            else:
                centri.append(np.append(slika[y, x], np.array([y, x])))
            k -= 1
            print(f'Preostalo {k} centrov.')

    if izbira == 'rocno':
        
        cv.namedWindow('Izberi centre')
        cv.setMouseCallback('Izberi centre', izberi_centre)
        cv.imshow('Izberi centre', slika)
        while cv.waitKey(0) & 0xFF != ord('q'):
            pass
        cv.destroyAllWindows()
    elif izbira == 'nakljucno':
        
        for _ in range(k):
            while True:
                y = np.random.randint(0, slika.shape[0])
                x = np.random.randint(0, slika.shape[1])
                nov_center = (
                    slika[y, x] if dimenzija_centra == 3 else
                    np.append(slika[y, x], np.array([y, x]))
                )
                valid = True
                for center in centri:
                    if np.linalg.norm(center - nov_center) < T:
                        valid = False
                        break
                if valid:
                    centri.append(nov_center)
                    break
    return np.array(centri)



if __name__ == "__main__":
    # Preberemo sliko
    slika = cv.imread('.utils/slika.jpg')
    shape = slika.shape

    k=3
    iteracije=10
    # Spremenimo velikost slike
    slika = cv.resize(slika, (96, 96))

    # Izvedemo segmentacijo slike z metodo k-meanss
    #segmentirana_slika = kmeans(slika, k, iteracije)

    # Izvedemo segmentacijo slike z metodo mean-shift
    segmentirana_slika_ms = meanshift(slika, h=10, dimenzija=5)
    
    # Prikažemo originalno in segmentirano slikoine inicia
    #cv.imshow('Originalna slika', slika)

    #segmentirana_slika_img = cv.resize(segmentirana_slika, shape[:2][::-1], interpolation=cv.INTER_NEAREST)
    #cv.imshow('Segmentirana slika (K-means)', segmentirana_slika_img)

    segmentirana_slika_ms_img = cv.resize(segmentirana_slika_ms, shape[:2][::-1], interpolation=cv.INTER_NEAREST)
    cv.imshow('Segmentirana slika (Mean-shift)', segmentirana_slika_ms_img)

    while cv.waitKey(0) & 0xFF != ord('q'):
            pass
    cv.destroyAllWindows()


