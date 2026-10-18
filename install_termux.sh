#!/data/data/com.termux/files/usr/bin/bash
# SudokuBot - installation Termux (a lancer une seule fois depuis le dossier du projet)
set -e
cd "$(dirname "$0")"
echo "== SudokuBot : installation =="

pkg update -y || true
pkg install -y python python-numpy python-pillow curl termux-api || pkg install -y python python-numpy python-pillow curl

if [ ! -d "$HOME/storage/shared" ]; then
  echo "-> Autorise l'acces au stockage (fenetre Android)..."
  termux-setup-storage || true
  sleep 3
fi

[ -f config.json ] || { cp config.example.json config.json; echo "-> config.json cree"; }
mkdir -p data

DEST=""
for d in /sdcard/Scripts "$HOME/storage/shared/Scripts"; do
  if mkdir -p "$d" 2>/dev/null && [ -w "$d" ]; then DEST="$d"; break; fi
done
if [ -n "$DEST" ]; then
  cp autojs/*.js "$DEST"/ && echo "-> agent copie dans $DEST"
else
  echo "!! Impossible d'ecrire dans /sdcard/Scripts : copie autojs/agent.js a la main."
fi

chmod +x start.sh stop.sh 2>/dev/null || true

echo "-> Verification des modules Python"
python - <<'EOF'
import numpy, PIL
print("numpy", numpy.__version__, "| Pillow", PIL.__version__)
EOF

echo "-> Tests unitaires (sans PYTHONPATH manuel, via run.py)"
python run.py --check || echo "!! certains tests ont echoue (voir ci-dessus)"

cat <<'TXT'

== Installation terminee ==
1. AutoJs6 : activer le service d'accessibilite + "afficher par-dessus les autres applis".
2. Termux  : ./start.sh          (joue)
             ./start.sh --test   (analyse seule, aucun appui)
             ./start.sh --beta   (attend "Demarrer" sur la page web)
3. Dans AutoJs6, lancer Scripts/agent.js si start.sh ne l'a pas lance seul.
   A la demande de capture, choisir "Ecran entier".
TXT
