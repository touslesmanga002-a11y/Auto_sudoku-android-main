#!/data/data/com.termux/files/usr/bin/bash
# SudokuBot - demarrage : serveur Python (nohup) + agent AutoJs6 + page web
# Options : --test (aucun appui) --beta (attend "Demarrer") --disable-popup --fg (premier plan) --no-web
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
# PYTHONPATH absolu (ceinture + bretelles : run.py ajoute aussi server/ lui-meme).
# Corrige le « ModuleNotFoundError: No module named 'sudokubot' » quand on
# lance depuis un autre dossier ou sans environnement prepare.
export PYTHONPATH="$SCRIPT_DIR/server${PYTHONPATH:+:$PYTHONPATH}"
PORT=${SUDOKUBOT_PORT:-8765}
DATA=./data
mkdir -p "$DATA"

ARGS=""; FG=0; WEB=1
for a in "$@"; do
  case "$a" in
    --test|--beta|--disable-popup|--auto) ARGS="$ARGS $a" ;;
    --fg) FG=1 ;;
    --no-web) WEB=0 ;;
    *) echo "option inconnue : $a" ;;
  esac
done

./stop.sh >/dev/null 2>&1

# copie de l'agent a jour
SCRIPTS=""
for d in /sdcard/Scripts "$HOME/storage/shared/Scripts"; do
  if mkdir -p "$d" 2>/dev/null && [ -w "$d" ]; then SCRIPTS="$d"; break; fi
done
[ -n "$SCRIPTS" ] && cp autojs/*.js "$SCRIPTS"/ 2>/dev/null

termux-wake-lock 2>/dev/null || true

if [ "$FG" = "1" ]; then
  exec python -u "$SCRIPT_DIR/run.py" --port "$PORT" $ARGS
fi

nohup python -u "$SCRIPT_DIR/run.py" --port "$PORT" $ARGS > "$DATA/server.out" 2>&1 &
echo $! > "$DATA/server.pid"

echo -n "Serveur en demarrage"
for i in $(seq 1 40); do
  if curl -s "http://127.0.0.1:$PORT/api/status" >/dev/null 2>&1; then echo " OK"; break; fi
  echo -n "."; sleep 0.25
done

# lancement de l'agent AutoJs6 (si 'am' est disponible)
if command -v am >/dev/null 2>&1 && [ -n "$SCRIPTS" ]; then
  am start -n org.autojs.autojs6/org.autojs.autojs.external.open.RunIntentActivity \
     -d "file://$SCRIPTS/agent.js" -t application/x-javascript >/dev/null 2>&1 \
  || echo "-> Lance Scripts/agent.js a la main dans AutoJs6"
else
  echo "-> Lance Scripts/agent.js dans AutoJs6"
fi

if [ "$WEB" = "1" ]; then
  termux-open-url "http://127.0.0.1:$PORT" 2>/dev/null \
  || am start -a android.intent.action.VIEW -d "http://127.0.0.1:$PORT" >/dev/null 2>&1 || true
fi
echo "Page de pilotage : http://127.0.0.1:$PORT   |   journal : tail -f $DATA/server.out"
