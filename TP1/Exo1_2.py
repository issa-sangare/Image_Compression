#!/usr/bin/env python
# coding: utf-8

# # $Série$ $n°1:$ $compression$ $par$ $RLE$ $et$ $Huffman$

# ## $Exercice$ $1:$ $Compression$ $par$ $codage$ $RLE$

# In[126]:

from IPython.display import display
import numpy as np
from PIL import Image


# In[127]:


data = "000110010111111100000000001111111"

with open('file2.txt', 'r') as fichier:
    text = fichier.read()
    
display(Image.open("./img2/eren.jpg"))
    
img_gray = "./img2/eren_gray.jpg"
img_gray = Image.open(img_gray)
display(img_gray)

img_bin = "./img2/eren_bin.jpg"
img_bin = Image.open(img_bin)
display(img_bin)


# In[128]:


print(np.array(Image.open("./img1/eren_rgb.jpg")))
print("\n\n")
print(np.array(img_gray))
print("\n\n")
print(np.array(img_bin))


# ### (1). Retourne la suite des tuples, $(symbol$ $et$ $occur)$ d’une suite de symboles passée en argument ; où $symbol$, $occur$ font référence respectivement à un nouveau symbole trouvé dans la source et sa fréquence d’occurrence.

# In[129]:


def sym_occ(data):
    encoded_data = []
    mat = []

    for char in data:
        if not encoded_data:
            encoded_data.append((char, 1))
        else:
            last = encoded_data[-1]

            if last[0] == char:
                encoded_data[-1] = (last[0], last[1] + 1)
            else:
                encoded_data.append((char, 1))

    max_index1 = max(encoded_data, key=lambda x: len(str(x[0])))
    max_index2 = max(encoded_data, key=lambda x: len(str(x[1])))

    indice1 = len(str(max_index1[0]))
    indice2 = len(str(max_index2[1]))

    '''
    for char, count in encoded_data:
        char_str = '{:<{width}}'.format(char, width=indice1)
        count_str = '{:0>{width}}'.format(count, width=indice2)
        mat.append((char_str, count_str))
    '''

    return encoded_data, indice2


# In[130]:


encoded_data, _ = sym_occ(data)
print(encoded_data)


# ### (2). Évalue le code $RLE$ de la suite de symboles initiale.

# In[131]:


def code_RLE(data):
    encoded_data, indice = sym_occ(data)
    last = encoded_data[-1]
    code = ""

    for index, (i, j) in enumerate(encoded_data):
        code += i + str(j)
        if index < len(encoded_data) - 1:  # Vérifie si ce n'est pas le dernier élément
            code += '*'

    return code, indice


# In[132]:


print(data)
print(code_RLE(data)[0])


# ### (3). Calcule le taux de compression.

# In[133]:


def taux_compression(data, compressed_data):
  taux = 1 - (len(compressed_data) / len(data))
  return taux * 100


# In[134]:


taux_compression(data, code_RLE(data)[0])


# ### (4). Implémente l’algorithme de décodage $RLE$.

# In[135]:


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


# In[136]:


decoded_data = decodage_RLE2(code_RLE(data)[0])
print(decoded_data)


# ## $Test$ $de$ $compression$ $et$ $de$ $décompression$

# In[137]:


def vecteur_ligne(matrice):
  vecteur = np.array(matrice)
  return vecteur.flatten().tolist()


# In[138]:


def inverse_vecteur_ligne(vecteur, lignes, colonnes):
  mat = np.array(vecteur)
  mat = mat.reshape((lignes, colonnes))
  
  return mat


# In[139]:


def vecteur_colonne(matrice):
  vecteur = np.ravel(matrice, order = 'F')
  return vecteur.tolist()


# In[140]:


def inverse_vecteur_colonne(vecteur, lignes, colonnes):
  aux = lignes
  lignes = colonnes
  colonnes = aux
  mat = np.array(vecteur)
  mat = mat.reshape((lignes, colonnes))
  
  return mat.T


# In[141]:


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


# In[142]:


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


# In[143]:


arr = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]] 

# Appeler la fonction
result = vecteur_zigzag2(arr)

# Afficher le résultat
print(result)


# In[144]:


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


# In[146]:


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


# In[147]:


print(matrice1)
print()
print(vecteur_ligne(matrice1))
print()
print(inverse_vecteur_ligne(vecteur_ligne(matrice1), 3, 3))


# In[148]:


print(matrice2)
print()
print(vecteur_ligne(matrice2))
print()
print(inverse_vecteur_ligne(vecteur_ligne(matrice2), 4, 2))


# In[149]:


print(matrice3)
print()
print(vecteur_ligne(matrice3))
print()
print(inverse_vecteur_ligne(vecteur_ligne(matrice3), 2, 4))


# In[150]:


print(matrice1)
print()
print(vecteur_colonne(matrice1))
print()
print(inverse_vecteur_colonne(vecteur_colonne(matrice1), 3, 3))


# In[151]:


print(matrice2)
print()
print(vecteur_colonne(matrice2))
print()
print(inverse_vecteur_colonne(vecteur_colonne(matrice2), 4, 2))


# In[152]:


print(matrice3)
print()
print(vecteur_colonne(matrice3))
print()
print(inverse_vecteur_colonne(vecteur_colonne(matrice3), 2, 4))


# In[153]:


print(matrice1)
print()
print(vecteur_zigzag(matrice1))
print()
print(inverse_vecteur_zigzag(vecteur_zigzag(matrice1), 3, 3))


# In[154]:


print(matrice1)
print()
print(vecteur_zigzag2(matrice1))
print()
print(inverse_vecteur_zigzag(vecteur_zigzag2(matrice1), 3, 3))


# In[155]:


print(matrice2)
print()
print(vecteur_zigzag(matrice2))
print()
print(inverse_vecteur_zigzag(vecteur_zigzag(matrice2), 4, 2))


# In[156]:


print(matrice3)
print()
print(vecteur_zigzag(matrice3))
print()
print(inverse_vecteur_zigzag(vecteur_zigzag(matrice3), 2, 4))


# In[157]:


print(vecteur_ligne(matrice1))
print(vecteur_ligne(matrice2))
print(vecteur_ligne(matrice3))


# In[158]:


print(vecteur_colonne(matrice1))
print(vecteur_colonne(matrice2))
print(vecteur_colonne(matrice3))


# In[159]:


print(vecteur_zigzag(matrice1))
print(vecteur_zigzag(matrice2))
print(vecteur_zigzag(matrice3))


# In[160]:


def ASCII_binaire(chaine):
  chaine_binaire = ""
  for caractere in chaine:
    caractere = (bin(ord(caractere)))[2:].zfill(8)
    chaine_binaire += caractere
  return chaine_binaire


# In[161]:


def binaire_liste(liste):
  chaine_binaire = ""
  for caractere in liste:
    caractere = (bin(caractere))[2:].zfill(8)
    chaine_binaire += caractere
  return chaine_binaire


# In[162]:


print(binaire_liste(vecteur_ligne(matrice1)))


# In[163]:


def RLE_image_gray(image):
  image = vecteur_ligne(image)
  image = binaire_liste(image)
  image, indice = code_RLE(image)
  return image, indice


# In[164]:


print(RLE_image_gray(img_gray)[0])


# In[165]:


def RLE_texte(chaine):
  chaine = ASCII_binaire(chaine)
  chaine, indice = code_RLE(chaine)
  return chaine, indice


# In[166]:


print(RLE_texte(text)[0])


# In[167]:


def RLE_image_bin(image):
  image = vecteur_ligne(image)
  mat = []
  for i in image:
      if i == 0:
        mat.append(i)
      else:
        mat.append(1)
  mat, indice = code_RLE(mat)
  return mat, indice


# In[168]:


print(RLE_image_bin(img_bin)[0])


# In[ ]: