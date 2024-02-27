#!/usr/bin/env python
# coding: utf-8

# # $Série$ $n°1:$ $compression$ $par$ $RLE$ $et$ $Huffman$

# ## $Exercice$ $1:$ $Compression$ $par$ $codage$ $RLE$

# In[1]:


import numpy as np
from PIL import Image


# In[2]:


data = "000110010111111100000000001111111"

with open('file2.txt', 'r') as fichier:
    text = fichier.read()
    
display(Image.open("./img/eren_rgb.png"))
    
img_gray = "./img/eren_gray.png"
img_gray = Image.open(img_gray)
display(img_gray)

img_bin = "./img/eren_bin.png"
img_bin = Image.open(img_bin)
display(img_bin)


# In[3]:


print(np.array(Image.open("./img1/eren_rgb.jpg")))
print("\n\n")
print(np.array(img_gray))
print("\n\n")
print(np.array(img_bin))


# ### (1). Retourne la suite des tuples, $(symbol$ $et$ $occur)$ d’une suite de symboles passée en argument ; où $symbol$, $occur$ font référence respectivement à un nouveau symbole trouvé dans la source et sa fréquence d’occurrence.

# In[4]:


def sym_occ(data):
    encoded_data = []
    #mat = []

    for char in data:
        if not encoded_data:
            encoded_data.append((char, 1))
        else:
            last = encoded_data[-1]

            if last[0] == char:
                encoded_data[-1] = (last[0], last[1] + 1)
            else:
                encoded_data.append((char, 1))

    #max_index1 = max(encoded_data, key=lambda x: len(str(x[0])))
    max_index2 = max(encoded_data, key=lambda x: len(str(x[1])))

    #indice1 = len(str(max_index1[0]))
    maxim2 = len(str(max_index2[1]))

    '''
    for char, count in encoded_data:
        char_str = '{:<{width}}'.format(char, width=indice1)
        count_str = '{:0>{width}}'.format(count, width=indice2)
        mat.append((char_str, count_str))
    '''

    return encoded_data, maxim2


# In[5]:


def RLE_binaire(data):
    chaine = ''
    if type(data) == list:
        data = ''.join(str(x) for x in data)  # Convertir chaque entier en chaîne de caractères
        
    symbole_occ, max2 = sym_occ(data)
    first = symbole_occ[0]

    if first[0] == '0':
        chaine += str(first[1]).zfill(max2)
    else:
        chaine += '0' + str(first[1]).zfill(max2)

    for i, j in symbole_occ[1:]:
        if j == 1:
            chaine += i.zfill(max2)
        else:
            chaine += str(j).zfill(max2)

    return chaine, max2


# In[6]:


data1 = '11010000001011100111'
print(data1)
print(list(data1))
print(RLE_binaire(list(data1)))
print(RLE_binaire(data1))


# In[7]:


print(data)
print(list(data))
print(RLE_binaire(list(data)))
print(RLE_binaire(data))


# In[8]:


encoded_data, _ = sym_occ(data)
print(encoded_data)


# ### (2). Évalue le code $RLE$ de la suite de symboles initiale.

# In[9]:


def code_RLE(data):
    encoded_data, maxim2 = sym_occ(data)
    last = encoded_data[-1]
    code = ""

    for index, (i, j) in enumerate(encoded_data):
        code += i + str(j)
        if index < len(encoded_data) - 1:  # Vérifie si ce n'est pas le dernier élément
            code += '*'

    return code, maxim2


# In[10]:


print(data)
print(code_RLE(data)[0])


# ### (3). Calcule le taux de compression.

# In[11]:


def taux_compression(data, compressed_data):
  taux = 1 - (len(compressed_data) / len(data))
  return taux * 100


# In[12]:


taux_compression(data, code_RLE(data)[0])


# ### (4). Implémente l’algorithme de décodage $RLE$.

# In[13]:


def decodage_RLE2(data):
    chaine = ""
    i = 0
    while i < len(data):
        # Prendre un caractère de la chaîne
        caractere = data[i]
        taille_sequence = ''
        j = i + 1
        while j < len(data) and data[j] != '*':
            taille_sequence += data[j]
            j += 1
            
        taille_sequence = int(taille_sequence)
        chaine += caractere * taille_sequence
        # Se déplacer à la position suivante après la séquence
        i = j + 1
    return chaine


# In[14]:


decoded_data = decodage_RLE2(code_RLE(data)[0])
print(decoded_data)


# ## $Test$ $de$ $compression$ $et$ $de$ $décompression$

# In[15]:


def vecteur_ligne(matrice):
  vecteur = np.array(matrice)
  return vecteur.flatten().tolist()


# In[16]:


def inverse_vecteur_ligne(vecteur, lignes, colonnes):
  mat = np.array(vecteur)
  mat = mat.reshape((lignes, colonnes))
  
  return mat


# In[17]:


def vecteur_colonne(matrice):
  vecteur = np.ravel(matrice, order = 'F')
  return vecteur.tolist()


# In[18]:


def inverse_vecteur_colonne(vecteur, lignes, colonnes):
  aux = lignes
  lignes = colonnes
  colonnes = aux
  mat = np.array(vecteur)
  mat = mat.reshape((lignes, colonnes))
  
  return mat.T


# In[19]:


def vecteur_zigzag(matrice):
    matrice = np.array(matrice)
    lignes, colonnes = matrice.shape
    solution = [[] for i in range(lignes + colonnes - 1)]

    for i in range(lignes):
        for j in range(colonnes):
            somme = i + j
            if somme % 2 == 0:
                solution[somme].insert(0, matrice[i, j])
            else:
                solution[somme].append(matrice[i, j])
    
    resultat = np.concatenate(solution)
    
    return resultat.tolist()


# In[20]:


def vecteur_zigzag2(arr):
    arr_np = np.array(arr)
    arr_np = np.fliplr(arr_np)
    mat = []
    
    for i in range(arr_np.shape[0] - 1, -(arr_np.shape[1]), -1):
        diagonal = np.diagonal(arr_np, offset=i).tolist()
        mat.append(diagonal)
    
    for i in range(len(mat)):
        if i % 2 == 0:
            mat[i] = mat[i][::-1]
    
    flat_list = [item for sublist in mat for item in sublist]
    
    return flat_list


# In[21]:


arr = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]] 

# Appeler la fonction
result = vecteur_zigzag2(arr)

# Afficher le résultat
print(result)


# In[22]:


def inverse_vecteur_zigzag(vecteur, lignes, colonnes):
    aux = lignes
    lignes = colonnes
    colonnes = aux
    nb_elements = lignes * colonnes
    if len(vecteur) != nb_elements:
        raise ValueError("La longueur du vecteur ne correspond pas au nombre total d'éléments de la matrice.")

    matrice = np.zeros((lignes, colonnes), dtype=np.int32)
    solution = [[] for i in range(lignes + colonnes - 1)]

    for i in range(lignes + colonnes - 1):
        nb_elements = min(i + 1, lignes, colonnes, lignes + colonnes - 1 - i)
        for j in range(nb_elements):
            if i % 2 == 0:
                solution[i].append(vecteur.pop(0))
            else:
                solution[i].insert(0, vecteur.pop(0))

    for i in range(lignes):
        for j in range(colonnes):
            matrice[i, j] = solution[i + j].pop(0)
    
    return matrice.T


# In[23]:


matrice1 = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

matrice2 = [
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8]
]

matrice3 = [
    [1, 2, 3, 4],
    [5, 6, 7, 8]
]


# In[24]:


print(matrice1)
print()
print(vecteur_ligne(matrice1))
print()
print(inverse_vecteur_ligne(vecteur_ligne(matrice1), 3, 3))


# In[25]:


print(matrice2)
print()
print(vecteur_ligne(matrice2))
print()
print(inverse_vecteur_ligne(vecteur_ligne(matrice2), 4, 2))


# In[26]:


print(matrice3)
print()
print(vecteur_ligne(matrice3))
print()
print(inverse_vecteur_ligne(vecteur_ligne(matrice3), 2, 4))


# In[27]:


print(matrice1)
print()
print(vecteur_colonne(matrice1))
print()
print(inverse_vecteur_colonne(vecteur_colonne(matrice1), 3, 3))


# In[28]:


print(matrice2)
print()
print(vecteur_colonne(matrice2))
print()
print(inverse_vecteur_colonne(vecteur_colonne(matrice2), 4, 2))


# In[29]:


print(matrice3)
print()
print(vecteur_colonne(matrice3))
print()
print(inverse_vecteur_colonne(vecteur_colonne(matrice3), 2, 4))


# In[30]:


print(matrice1)
print()
print(vecteur_zigzag(matrice1))
print()
print(inverse_vecteur_zigzag(vecteur_zigzag(matrice1), 3, 3))


# In[31]:


print(matrice1)
print()
print(vecteur_zigzag2(matrice1))
print()
print(inverse_vecteur_zigzag(vecteur_zigzag2(matrice1), 3, 3))


# In[32]:


print(matrice2)
print()
print(vecteur_zigzag(matrice2))
print()
print(inverse_vecteur_zigzag(vecteur_zigzag(matrice2), 4, 2))


# In[33]:


print(matrice3)
print()
print(vecteur_zigzag(matrice3))
print()
print(inverse_vecteur_zigzag(vecteur_zigzag(matrice3), 2, 4))


# In[34]:


print(vecteur_ligne(matrice1))
print(vecteur_ligne(matrice2))
print(vecteur_ligne(matrice3))


# In[35]:


print(vecteur_colonne(matrice1))
print(vecteur_colonne(matrice2))
print(vecteur_colonne(matrice3))


# In[36]:


print(vecteur_zigzag(matrice1))
print(vecteur_zigzag(matrice2))
print(vecteur_zigzag(matrice3))


# In[37]:


def ASCII_binaire(chaine):
  chaine_binaire = ""
  for caractere in chaine:
    caractere = (bin(ord(caractere)))[2:].zfill(8)
    chaine_binaire += caractere
  return chaine_binaire


# In[38]:


def binaire_liste(liste):
  chaine_binaire = ""
  for caractere in liste:
    caractere = (bin(caractere))[2:].zfill(8)
    chaine_binaire += caractere
  return chaine_binaire


# In[39]:


print(binaire_liste(vecteur_ligne(matrice1)))


# In[40]:


def RLE_image_gray(image):
  image = vecteur_ligne(image)
  image = binaire_liste(image)
  image, indice = code_RLE(image)
  return image, indice


# In[41]:


print(RLE_image_gray(img_gray)[0])


# In[42]:


def RLE_texte(chaine):
  chaine = ASCII_binaire(chaine)
  chaine, indice = code_RLE(chaine)
  return chaine, indice


# In[43]:


print(RLE_texte(text)[0])


# In[44]:


def compression_RLE_binaire(image):
    with open('image_compressée_binaire.txt', 'w') as fichier:
        L, C = image.size
        image = vecteur_ligne(image)  # Assurez-vous que cette fonction est correctement définie
        mat = []
        
        for i in image:
          if i == 255 or i:
            mat.append(1)
          else:
            mat.append(0)
        
        code, max2 = RLE_binaire(mat)  # Assurez-vous que cette fonction est correctement définie
        
        # Écrire les données dans le fichier
        fichier.write(str(L).zfill(5))
        fichier.write(str(C).zfill(5))
        fichier.write(str(max2))  # Convertir max2 en chaîne si ce n'est pas déjà le cas
        fichier.write(code)  # Convertir chaque élément de code en chaîne

compression_RLE_binaire(img_bin)
print(img_bin.size)

