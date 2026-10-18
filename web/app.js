/* SudokuBot - page de pilotage (vanilla JS, servie par le serveur Termux) */
(function () {
  var $ = function (s) { return document.querySelector(s); };
  var $$ = function (s) { return Array.prototype.slice.call(document.querySelectorAll(s)); };
  var ST = {}, CFG = null, LOGSEQ = 0, REV = -1, PICK = null, EDIT = null;

  function api(path, body) {
    var o = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
    return fetch(path, o).then(function (r) { return r.json(); });
  }
  function toast(t) {
    var el = $("#toast"); el.textContent = t; el.style.display = "block";
    clearTimeout(el._t); el._t = setTimeout(function () { el.style.display = "none"; }, 1800);
  }
  function show(id) {
    $$(".tab").forEach(function (s) { s.classList.toggle("on", s.id === id); });
    $$("#tabs button").forEach(function (b) { b.classList.toggle("on", b.dataset.tab === id); });
    if (id === "settings") loadCfg();
    if (id === "speed") renderLevels();
    if (id === "spots") loadSpots();
  }
  $$("#tabs button").forEach(function (b) { b.onclick = function () { show(b.dataset.tab); }; });

  // ---- commandes
  $$("[data-act]").forEach(function (b) {
    b.onclick = function () { api("/api/control", { action: b.dataset.act }).then(function (s) { ST = s; render(); toast(b.textContent.trim()); }); };
  });
  $("#testMode").onchange = function (e) { api("/api/control", { action: "test", value: e.target.checked }); };

  // ---- etat
  var STATE_FR = { running: "Lecture", paused: "Pause", stopped: "Arrêté", idle: "En attente" };
  function render() {
    $("#kState").textContent = STATE_FR[ST.state] || ST.state || "—";
    $("#kGames").textContent = ST.games || 0;
    $("#kMoves").textContent = ST.moves || 0;
    $("#kSpeed").textContent = ST.speed_label || "—";
    $("#msg").textContent = ST.msg || "";
    $("#testMode").checked = !!ST.test;
    var ai = ST.agent_info || {};
    $("#agentInfo").innerHTML =
      "<dt>Connecté</dt><dd>" + (ST.agent ? "oui" : "non — lancez agent.js dans AutoJs6") + "</dd>" +
      "<dt>Modèle</dt><dd>" + (ai.model || "—") + "</dd>" +
      "<dt>Écran</dt><dd>" + (ai.w ? ai.w + " × " + ai.h : "—") + "</dd>" +
      "<dt>Appli</dt><dd>" + (ai.pkg || "—") + "</dd>" +
      "<dt>Capture</dt><dd>" + (ST.shot_ms || 0) + " ms</dd>" +
      "<dt>Gabarits</dt><dd>" + (ST.templates ? ST.templates.learned + " appris · " + ST.templates.keypad + " pavé · " + ST.templates.font + " police" : "—") + "</dd>";
    var seg = $("#choices");
    if (ST.choices && seg.childElementCount !== ST.choices.length) {
      seg.innerHTML = "";
      ST.choices.forEach(function (c) {
        var b = document.createElement("button"); b.textContent = c.label; b.dataset.id = c.id;
        b.onclick = function () { api("/api/control", { action: "speed", level: c.id }).then(function (s) { ST = s; render(); }); };
        seg.appendChild(b);
      });
    }
    $$("#choices button").forEach(function (b) { b.classList.toggle("on", b.dataset.id === ST.speed_choice); });
    $$(".lvl").forEach(function (l) { l.classList.toggle("on", l.dataset.id === ST.speed_effective); });
    var tb = $("#perf tbody"); tb.innerHTML = "";
    (ST.perf || []).slice().reverse().forEach(function (p) {
      tb.insertAdjacentHTML("beforeend", "<tr><td>" + p.level + "</td><td>" + p.taps + "</td><td>" + p.lag + " ms</td><td>" + p.miss + "</td><td>" + p.shot + " ms</td></tr>");
    });
    if (ST.rev !== REV) {
      REV = ST.rev;
      var src = "/api/debug/latest.jpg?r=" + REV;
      $("#overlay").src = src; $("#ovMini").src = src;
    }
  }
  function poll() {
    api("/api/status").then(function (s) { ST = s; render(); }).catch(function () { $("#msg").textContent = "Serveur injoignable"; })
      .finally(function () { setTimeout(poll, 1000); });
  }
  function pollLogs() {
    api("/api/logs?since=" + LOGSEQ).then(function (list) {
      var box = $("#logBox"), atEnd = box.scrollTop + box.clientHeight >= box.scrollHeight - 30;
      list.forEach(function (l) {
        LOGSEQ = l.seq;
        var d = document.createElement("div"); d.className = l.lvl;
        d.textContent = l.t + "  " + l.msg; box.appendChild(d);
      });
      while (box.childElementCount > 800) box.removeChild(box.firstChild);
      if (atEnd) box.scrollTop = box.scrollHeight;
    }).catch(function () {}).finally(function () { setTimeout(pollLogs, 1200); });
  }

  // ---- config
  function loadCfg() {
    return api("/api/config").then(function (c) {
      CFG = c;
      $("#cfgText").value = JSON.stringify(c, null, 2);
      $("#order").value = c.human.order;
      $("#forceSize").value = c.vision.force_size || "";
      $("#forceBox").value = c.vision.force_box ? c.vision.force_box.join("x") : "";
      $("#jitter").value = c.human.jitter_sigma; $("#jitterOut").textContent = c.human.jitter_sigma;
      $("#detectDelay").value = (c.game && c.game.detect_delay_ms != null) ? c.game.detect_delay_ms : 2500;
      $("#findTimeout").value = (c.menus && c.menus.find_timeout_s != null) ? c.menus.find_timeout_s : 10;
      $("#fusedScan").checked = !(c.menus && c.menus.fused_scan === false);
      renderLevels();
    });
  }
  $("#jitter").oninput = function (e) { $("#jitterOut").textContent = e.target.value; };
  function patch(path, value) { return api("/api/config/patch", { path: path, value: value }); }
  $("#saveQuick").onclick = function () {
    var fb = $("#forceBox").value.trim(), m = fb.match(/^(\d+)\s*[x×*]\s*(\d+)$/);
    Promise.all([
      patch("human.order", $("#order").value),
      patch("vision.force_size", $("#forceSize").value ? +$("#forceSize").value : null),
      patch("vision.force_box", m ? [+m[1], +m[2]] : null),
      patch("human.jitter_sigma", +$("#jitter").value),
      patch("game.detect_delay_ms", Math.max(0, +$("#detectDelay").value || 0)),
      patch("menus.find_timeout_s", Math.min(25, Math.max(3, +$("#findTimeout").value || 10))),
      patch("menus.fused_scan", !!$("#fusedScan").checked)
    ]).then(function () { toast("Réglages enregistrés"); loadCfg(); });
  };
  $("#saveCfg").onclick = function () {
    try { var c = JSON.parse($("#cfgText").value); } catch (e) { return toast("JSON invalide : " + e.message); }
    api("/api/config", c).then(function () { toast("config.json enregistré"); loadCfg(); });
  };
  $("#reloadCfg").onclick = function () { loadCfg().then(function () { toast("Rechargé"); }); };

  // ---- paliers
  function renderLevels() {
    if (!CFG) { loadCfg(); return; }
    var box = $("#levels"); box.innerHTML = "";
    CFG.speed.order.forEach(function (id) {
      var L = CFG.speed.levels[id]; if (!L) return;
      var d = document.createElement("div"); d.className = "lvl"; d.dataset.id = id;
      d.innerHTML = "<b>" + L.label + "</b><small>appui " + L.tap_ms[0] + "-" + L.tap_ms[1] + " · case→chiffre " + L.cell_to_key_ms[0] +
        " · entre " + L.between_ms[0] + " · salve " + L.batch + "</small>";
      d.onclick = function () { openSheet(id); };
      box.appendChild(d);
    });
    render();
  }
  function openSheet(id) {
    EDIT = id; var L = CFG.speed.levels[id];
    $("#shTitle").textContent = L.label;
    $("#shTap0").value = L.tap_ms[0]; $("#shTap1").value = L.tap_ms[1];
    $("#shCk0").value = L.cell_to_key_ms[0]; $("#shCk1").value = L.cell_to_key_ms[1];
    $("#shBt0").value = L.between_ms[0]; $("#shBt1").value = L.between_ms[1];
    $("#shBatch").value = L.batch; $("#shThink").value = L.think_factor;
    $("#sheet").classList.add("on");
  }
  $("#shClose").onclick = function () { $("#sheet").classList.remove("on"); };
  $("#shSave").onclick = function () {
    var L = Object.assign({}, CFG.speed.levels[EDIT], {
      tap_ms: [+$("#shTap0").value, +$("#shTap1").value],
      cell_to_key_ms: [+$("#shCk0").value, +$("#shCk1").value],
      between_ms: [+$("#shBt0").value, +$("#shBt1").value],
      batch: Math.max(1, +$("#shBatch").value), think_factor: +$("#shThink").value
    });
    patch("speed.levels." + EDIT, L).then(function () { $("#sheet").classList.remove("on"); toast("Palier enregistré"); loadCfg(); });
  };

  // ---- analyse : lecture de coordonnees + test d'image
  $("#overlay").onclick = function (e) {
    var r = e.target.getBoundingClientRect();
    PICK = [+((e.clientX - r.left) / r.width).toFixed(4), +((e.clientY - r.top) / r.height).toFixed(4)];
    $("#pickOut").textContent = "x = " + PICK[0] + " · y = " + PICK[1];
  };
  $$("[data-pick]").forEach(function (b) {
    b.onclick = function () {
      if (!PICK) return toast("Touchez d'abord l'image");
      var k = b.dataset.pick;
      var p = k === "keypad" ? patch("vision.keypad_manual", { y_frac: PICK[1], x0_frac: 0.06, x1_frac: 0.94 })
                             : patch("menus.fallback_taps." + k, PICK);
      p.then(function () { toast("Enregistré : " + k); });
    };
  });
  $("#upload").onchange = function (e) {
    var f = e.target.files[0]; if (!f) return;
    $("#anaOut").textContent = "Analyse…";
    fetch("/api/analyze", { method: "POST", body: f }).then(function (r) { return r.json(); }).then(function (j) {
      var out = j.grid ? ("Grille " + j.grid.n + "x" + j.grid.n + " · blocs " + j.grid.br + "x" + j.grid.bc + " (" + j.grid.box_src + ") · " + j.ms + " ms\n" +
        j.values.map(function (r) { return r.join(" ").replace(/0/g, "."); }).join("\n") + "\nPavé : " + (j.keypad.length || "introuvable"))
        : ("Aucune grille détectée (" + (j.error || j.ms + " ms") + ")");
      $("#anaOut").textContent = out;
      $("#overlay").src = "/api/debug/latest.jpg?r=" + Date.now();
    });
  };

  // ---- emplacements : hitboxes sur captures de reference
  var SP = { spots: {}, refs: {} };
  var SP_LABELS = { grid: "grille", keys: "pavé", toggle: "placement rapide",
                    astuce: "astuce", nouvelle_partie: "nouvelle partie", debutant: "débutant" };
  function loadSpots() {
    return api("/api/spots").then(function (j) {
      SP.spots = j.spots || {}; SP.refs = j.refs || {};
      var st = "Mode : " + (j.auto ? "AUTO (tout OCR, emplacements ignorés)" :
        (j.enabled === false ? "désactivé (menus.spots_enabled)" : "hitboxes actives")) +
        " · références : " + (Object.keys(SP.refs).join(", ") || "aucune — ajoutez game/fin_partie/niveaux.* dans emplacements/") +
        (j.stale ? " · à recalibrer (images plus neuves)" : "") +
        " · calibrés : " + (Object.keys(SP.spots).filter(function (k) { return k[0] !== "_"; }).join(", ") || "—");
      $("#spotsStatus").textContent = st;
      renderSpots();
    }).catch(function () { $("#spotsStatus").textContent = "Serveur injoignable"; });
  }
  function spBoxes(ref) {
    // -> [{key, idx, label, x, y, w, h}] en % (centre+taille), grille convertie
    var s = SP.spots[ref] || {}, out = [];
    ["nouvelle_partie", "debutant", "toggle", "astuce"].forEach(function (k) {
      if (s[k]) out.push({ key: k, idx: -1, label: SP_LABELS[k] || k,
                           x: s[k].x * 100, y: s[k].y * 100, w: s[k].w * 100, h: s[k].h * 100 });
    });
    if (s.grid && s.grid.x1 != null) {
      var g = s.grid;
      out.push({ key: "grid", idx: -1, label: "grille",
                 x: (g.x0 + g.x1) / 2 * 100, y: (g.y0 + g.y1) / 2 * 100,
                 w: (g.x1 - g.x0) * 100, h: (g.y1 - g.y0) * 100 });
    }
    (s.keys || []).forEach(function (k, i) {
      out.push({ key: "keys", idx: i, label: "touche " + k.v,
                 x: k.x * 100, y: k.y * 100, w: k.w * 100, h: k.h * 100 });
    });
    return out;
  }
  function renderSpots() {
    var box = $("#spotsRefs"); box.innerHTML = "";
    ["game", "fin_partie", "niveaux"].forEach(function (ref) {
      var card = document.createElement("div"); card.className = "card";
      var title = document.createElement("h3");
      title.textContent = ref + (SP.refs[ref] ? " (" + SP.refs[ref].file + ")" : " — image manquante");
      card.appendChild(title);
      if (!SP.refs[ref]) {
        var p = document.createElement("p"); p.className = "hint";
        p.textContent = "Ajoutez " + ref + ".jpg dans emplacements/ puis Recalibrer.";
        card.appendChild(p); box.appendChild(card); return;
      }
      var wrap = document.createElement("div"); wrap.className = "spots-wrap";
      var img = document.createElement("img");
      img.src = "/spots/img/" + ref + "?r=" + Date.now(); img.alt = ref;
      wrap.appendChild(img);
      var boxes = spBoxes(ref);
      boxes.forEach(function (b) {
        var d = document.createElement("div");
        d.className = "spot-rect";
        d.style.left = (b.x - b.w / 2) + "%"; d.style.top = (b.y - b.h / 2) + "%";
        d.style.width = b.w + "%"; d.style.height = b.h + "%";
        d.innerHTML = "<span>" + b.label + "</span>";
        wrap.appendChild(d);
      });
      card.appendChild(wrap);
      // Liste des elements : position X/Y + dimensions L/H (%), modifiables.
      var list = document.createElement("div"); list.className = "spots-list";
      boxes.forEach(function (b) {
        var row = document.createElement("div"); row.className = "spots-row";
        row.innerHTML = "<b>" + b.label + "</b>";
        [["x", "X"], ["y", "Y"], ["w", "L"], ["h", "H"]].forEach(function (pair) {
          var lab = document.createElement("label");
          lab.textContent = pair[1];
          var inp = document.createElement("input");
          inp.type = "number"; inp.step = "0.1"; inp.value = b[pair[0]].toFixed(1);
          inp.dataset.ref = ref; inp.dataset.key = b.key; inp.dataset.idx = b.idx; inp.dataset.f = pair[0];
          inp.addEventListener("change", function () {
            writeSpotBox(ref, b.key, +b.idx, pair[0], +inp.value || 0);
            renderSpots();
          });
          lab.appendChild(inp); row.appendChild(lab);
        });
        list.appendChild(row);
      });
      if (!boxes.length) {
        var p2 = document.createElement("p"); p2.className = "hint";
        p2.textContent = "Rien de détecté — lancez Recalibrer.";
        list.appendChild(p2);
      }
      card.appendChild(list); box.appendChild(card);
    });
  }
  function writeSpotBox(ref, key, idx, f, val) {
    var s = SP.spots[ref] || (SP.spots[ref] = {});
    val = val / 100;
    if (key === "keys") {
      var k = (s.keys || [])[idx]; if (!k) return;
      if (f === "x" || f === "y") k[f] = val;
      else k[f] = Math.max(0.002, val);
    } else if (key === "grid") {
      var g = s.grid || (s.grid = { n: 9, br: 3, bc: 3 });
      var cur = { x: (g.x0 + g.x1) / 2, y: (g.y0 + g.y1) / 2,
                  w: g.x1 - g.x0, h: g.y1 - g.y0 };
      if (f === "x" || f === "y") cur[f] = val;
      else cur[f] = Math.max(0.002, val);
      g.x0 = cur.x - cur.w / 2; g.y0 = cur.y - cur.h / 2;
      g.x1 = cur.x + cur.w / 2; g.y1 = cur.y + cur.h / 2;
    } else {
      var b = s[key] || (s[key] = {});
      if (f === "x" || f === "y") b[f] = val;
      else b[f] = Math.max(0.002, val);
    }
  }
  $("#spotsSave").onclick = function () {
    api("/api/spots", { spots: SP.spots }).then(function (j) {
      if (j.ok) { SP.spots = j.spots; toast("Emplacements enregistrés"); renderSpots(); }
      else toast("Erreur : " + (j.error || "?"));
    });
  };
  $("#spotsCalib").onclick = function () {
    toast("Calibration…");
    api("/api/spots/calibrate", {}).then(function (j) {
      if (j.ok) { SP.spots = j.spots; toast("Calibré"); loadSpots(); }
      else toast("Erreur : " + (j.error || "?"));
    });
  };

  poll(); pollLogs(); loadCfg();
})();
