# SudokuBot — Termux + AutoJs6 (Redmi Note 13)

Bot qui joue au sudoku (`easy.sudoku.puzzle.solver.free`) à ta place, de façon humaine.

```
[Téléphone]
  Jeu sudoku  ⇄  Agent AutoJs6 (autojs/agent.js) = yeux + mains, aucune décision
                    ⇅ HTTP long-polling 127.0.0.1:8765
  Termux : serveur Python (server/sudokubot) = cerveau (vision, solveur, rythme)
           page web de pilotage (web/) servie par le même serveur
  Barre flottante en haut : ▶/❚❚ · ■ · [vitesse ▾] · état
```

## Installation (une fois)
1. Copier le dossier `sudokubot/` dans Termux, par ex. `~/sudokubot` :
   `cp -r /sdcard/Download/sudokubot ~/ && cd ~/sudokubot`
2. `bash install_termux.sh`
3. AutoJs6 : activer **Accessibilité**, **Afficher par-dessus les autres applis**,
   et désactiver l'optimisation batterie pour AutoJs6 et Termux.

## Utilisation
- `./start.sh` : lance le serveur, l'agent et la page `http://127.0.0.1:8765`.
- À la demande de capture d'écran : choisir **Écran entier**.
- Ouvre le jeu (ou laisse `game.auto_launch`), le bot appuie sur « Nouvelle partie » → « Débutant »,
  attend le chargement, lit la grille, la résout, la remplit, puis recommence.
- `./start.sh --test` : analyse seule (aucun appui) — idéal pour vérifier la détection.
- `./stop.sh` : arrêt.

## Barre flottante
- ▶ / ❚❚ : lecture / pause (interrompt une salve immédiatement, jamais au milieu d'un coup)
- ■ : stop (appui long = fermer l'agent)
- Menu déroulant : Auto (adaptatif, défaut), Lent, Normal, Moyen, Rapide, Très rapide, Turbo
- ⇅ : déplacer la barre · ✕ : masquer (réactivable dans config `popup.enabled`)
- Secours : touche Volume + = pause / reprise

## Robustesse
- Grille détectée par « crêtes » de lignes + réseau régulier : 4×4, 6×6 (2×3), 8×8, 9×9, 12×12, 16×16…
- Blocs déduits de l'épaisseur des lignes (sinon table `vision.box_table`, ou `vision.force_box`).
- Coordonnées toujours relatives à la capture : fonctionne sur n'importe quelle résolution.
- Chiffres lus par gabarits : pavé du bas + polices Android + gabarits appris à chaque partie.
- Lecture incohérente → réparation par alternatives, puis OCR AutoJs6 en secours.
- Vérification après chaque salve ; une case ratée est rejouée.

## Vitesses
Dès « Moyen », les coups partent en **salves** (3, 8, 18, 81 coups) avec une seule capture de vérification
par salve. « Auto » démarre à Moyen, monte si le téléphone suit (retard < 45 ms, zéro raté), redescend sinon,
dans les bornes `speed.auto.min` / `speed.auto.max`.

## Tout est réglable
`config.json` (ou l'onglet Réglages de la page web) : textes des menus, temps d'attente, seuils de vision,
dispersion des appuis, durées, paliers, ordre de remplissage, etc.
