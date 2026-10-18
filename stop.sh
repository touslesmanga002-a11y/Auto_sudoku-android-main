#!/data/data/com.termux/files/usr/bin/bash
# SudokuBot - arret du serveur
cd "$(dirname "$0")"
if [ -f data/server.pid ]; then
  kill "$(cat data/server.pid)" 2>/dev/null && echo "Serveur arrete"
  rm -f data/server.pid
fi
pkill -f "sudokubot.server" 2>/dev/null || true
termux-wake-unlock 2>/dev/null || true
