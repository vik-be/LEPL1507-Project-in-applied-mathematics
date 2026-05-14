# Projet 4 en Mathématiques Appliquées
## Classification robuste sous attaques adversariales
#### Antonutti Victor, Desmette Mateo, Djoukouo Marie Pascale, Hertoghe Adèle, Lepoutre Florian, Mbengue Abdou

### Arborescence du projet
```
./
├── graphs
│   ├── attackers
│   │   ├── attacker_group_03.py
│   │   └── attacker_group_03_DeepFool.py
│   ├── classifiers
│   │   ├── classifier_S4_group_A.pt
│   │   ├── classifier_S4_group_B.pt
│   │   ├── classifier_S4_group_C.pt
│   │   ├── classifier_S4_group_D.pt
│   │   ├── classifier_S4_group_E.pt
│   │   ├── classifier_S4_group_F.pt
│   │   └── classifier_S11_group_03.pt
│   ├── attack_success_benchmark.py
│   ├── benchmark.py
│   ├── clean_accuracy_benchmark.py
│   ├── confusion_matrix.py
│   ├── dessin.py
│   └── dessin_epsilon.py (nom à confirmer - interface graphique)
├── attacker_group_03.py
├── classifier_S11_group_03.pt
├── classifier_S11_group_03.py
├── README.md
└── report_group_03.pdf

```

### Détails
#### Fichiers principaux

* ``attacker_group_03.py``

Fichier python contenant notre attaquant, et donc une fonction ``attack(x, f_string, eps)``, avec :

``x`` : une image à attaquer, enregistrée comme un ``tensor.FloatTensor`` de dimension 28x28
dans l’intervalle [0.0, 1.0].\
``f_string`` : un string qui contient le chemin vers le classifieur enregistré sous format ``f.pt`` à attaquer (potentiellement jamais vu auparavant).\
``eps`` : une taille de perturbation encodée comme un ``np.float`` entre 0.0 et 2.0.


* ``classifier_group_03.pt``

L'*oracle* contenant notre classifieur final. Il permet donc de faire de l’inférence et de calculer des dérivées par rapport aux pixels de l’image d’entrée sans divulguer ni l’architecture, ni les poids du réseau de notre classifieur. Il contient une fonction ``forward`` définie comme ceci :

Input : une image ``x`` enregistrée comme un ``tensor.FloatTensor`` de dimension 28x28 dans l’intervalle [0.0, 1.0].\
Output : un vecteur ``y`` de taille 47 enregistré comme un ``tensor.FloatTensor`` avec les probabilités de classification associées à chaque classe.


* ``classifier_group_03.py``

Le fichier python contenant l'architecture du réseau de notre classifieur, ainsi que notre méthode d'entrainement.


#### Dossier graphs
Ce dossier contient les fichiers ``.py`` qui ont permis de réaliser les expériences et investigations menées. Pour exécuter tous ces fichiers, il faut d'abord se placer dans le dossier ``graphs`` de notre projet.

* sous-dossier ``attackers``

Contient les deux attaquants que nous avons développés, et que nous avons donc testé lors de nos investigations.


* sous-dossier ``classifiers``

Contient les classifieurs testés dans le cadre de nos expériences.


* ``attack_success_benchmark.py``

Ce script sert à tester un ou plusieurs attaquants sur un ou plusieurs classifieurs. Il génère un graphe pour chaque classifieur, montrant l'efficacité des attaques en fonction de la taille de perturbation autorisée (epsilon).

Pour lancer ce fichier, il suffit d'exécuter ``python attack_success_benchmark.py``. Par défaut, tous les classifieurs et tous les attaquants sont testés. Cependant, il suffit de mettre en commentaire les classifieurs de ``MODEL_PATHS`` ou attaquants de ``ATTACKER_PATHS`` que vous ne voulez pas tester pour une exécution personnalisée. Les graphes générés sont dénommés ``attack_success_vs_epsilon_<nom_du_modèle>.pdf``. 
Par défaut, les valeurs d'epsilon testées sont réparties entre 0 et 2 en 11 points et chaque valeur d'epsilon est testée 200 fois. Ceci peut être modifié en fonction de la précision attendue et/ou du temps de calcul disponible.


* ``benchmark.py``

Ce script sert également à tester un ou plusieurs attaquants sur un ou plusieurs classifieurs. Il génère un graphe pour chaque attaquant, montrant la précision des classifieurs après attaque en fonction de epsilon.

Pour lancer ce fichier, il suffit d'exécuter ``python benchmark.py``. Par défaut, tous les classifieurs et tous les attaquants sont testés. Cependant, il suffit de mettre en commentaire les classifieurs de ``MODEL_PATHS`` ou attaquants de ``ATTACKER_PATHS`` que vous ne voulez pas tester pour une exécution personnalisée. Les graphes générés sont dénommés ``benchmark_<nom_attacker>.pdf``.


* ``clean_accuracy_benchmark.py``

Ce script évalue la clean accuracy de tous les classifieurs ``.pt`` présents dans le dossier ``classifiers/``, sur un sous-ensemble du jeu de test EMNIST balanced.

Le script se lance avec ``python clean_accuracy_benchmark.py``. Le script n’enregistre pas de fichier et n’affiche pas de graphe : il imprime simplement les résultats dans le terminal.


* ``confusion_matrix.py``

Ce script permet de générer la matrice de confusion d'un classifieur sur le jeu de test EMNIST Balanced, ainsi qu'une matrice de confusion revisitée, mettant en évidence les K classes avec le plus d'erreurs (et en supprimant la diagonale). Les graphes sont nommés ``confusion_<nom_du_modèle>.pdf`` et ``<nom_du_modèle>-misclassification.pdf`` respectivement.

Le script peut être exécuté par ``python confusion_matrix.py``. Le classifieur testé peut être changé en modifiant la variable ``filename`` dans le code. Le nombre de classes K du deuxième graphe peut être également modifié.


* ``dessin.py`` et ``dessin_eps.py``

Interfaces graphiques permettant de visualiser la classification et les attaques d'un caractère dessiné à la main. Des détails concernant ces fichiers peuvent être trouvés dans la section 7 du rapport. Concernant leur exécution, il suffit d'utiliser les commandes ``python dessin.py`` ou ``python dessin_eps.py``. Dans les deux cas, une fenêtre s'ouvrira, vous permettant de dessiner le caractère de votre choix.

Pour ``dessin.py``, il vous est possible d'effacer votre caractère avec le bouton ``Effacer``, mais aussi de changer l'épaisseur de votre pinceau, en modifiant la valeur de ``Brush``. De plus, appuyer sur le bouton ``Prédire`` affichera les prédictions de chaque modèle du dossier ``classifiers``. Finalement, le bouton ``Attaquer`` vous permet d'attaquer chacun de ces classifieurs avec notre attaquant. Une fenêtre s'ouvrira alors, affichant les nouvelles prédictions, ainsi que les images perturbées. La taille de perturbation autorisée, epsilon, est modifiable depuis la fenêtre de base.

Concernant ``dessin_eps.py``, les mêmes options principales peuvent être retrouvées. La différence est que ce programme cherche le plus petit epsilon qui change la prédiction des classifieurs, grâce au bouton ``Chercher eps``. Il vous est par ailleurs demandé d'encoder manuellement le vrai label de votre dessin. Le pas de recherche du epsilon est modifiable en changeant la valeur du ``pas``.