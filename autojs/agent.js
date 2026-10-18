/**
 * SudokuBot - Agent AutoJs6 ("yeux et mains", AUCUNE decision).
 * - Long-polling vers le serveur Termux : http://127.0.0.1:8765/api/agent/exchange
 * - Primitives : screenshot, tap, tap_batch, find_any, ui_scan, ocr_region, launch, back, sleep, toast
 * - Barre flottante en haut : [lecture/pause] [stop] [vitesse v] | etat | [deplacer] [x]
 *
 * Installation : copier dans /sdcard/Scripts (fait par install_termux.sh / start.sh) puis lancer dans AutoJs6.
 * Autorisations : Accessibilite AutoJs6 + capture d'ecran (choisir "Ecran entier") + affichage par-dessus.
 */
auto.waitFor();

var SERVER = "http://127.0.0.1:8765";
var VERSION = "1.2.0";
var RUNNING = true;
var LOCAL_PAUSE = 0;          // horodatage pose par la barre : interrompt immediatement une salve (0 = aucun)
function setPause(on) { LOCAL_PAUSE = on ? Date.now() : 0; }
var POPUP = null;             // fenetre flottante
var POPUP_STATE = {};
var POPUP_DEAD_SCR = 0;       // bas de la barre (coord. ecran) : zone ou aucun appui n'est envoye
var POPUP_HIDDEN = false;
var POPUP_Y = 0;
var CAPMAP = { sx: 1, sy: 1, cw: 0, ch: 0 };

function dm() { return context.getResources().getDisplayMetrics(); }
function SW() { return device.width || dm().widthPixels; }
function SH() { return device.height || dm().heightPixels; }

// ------------------------------------------------------------------ coordonnees capture <-> ecran
function updateCapMap(img) {
    var cw = img.getWidth(), ch = img.getHeight();
    CAPMAP = { sx: SW() / cw, sy: SH() / ch, cw: cw, ch: ch };
}
function cap2scr(x, y) {
    var X = Math.round(x * CAPMAP.sx), Y = Math.round(y * CAPMAP.sy);
    X = Math.max(1, Math.min(SW() - 2, X));
    Y = Math.max(1, Math.min(SH() - 2, Y));
    return [X, Y];
}
function scr2cap(x, y) { return [Math.round(x / CAPMAP.sx), Math.round(y / CAPMAP.sy)]; }
function rectScr2Cap(b) {
    var a = scr2cap(b.left, b.top), c = scr2cap(b.right, b.bottom);
    return [a[0], a[1], c[0], c[1]];
}

// ------------------------------------------------------------------ texte
function norm(s) {
    s = String(s || "").toLowerCase();
    try { s = s.normalize("NFD").replace(/[\u0300-\u036f]/g, ""); } catch (e) {
        var map = { "é": "e", "è": "e", "ê": "e", "ë": "e", "à": "a", "â": "a", "î": "i", "ï": "i", "ô": "o", "ù": "u", "û": "u", "ç": "c" };
        s = s.replace(/[éèêëàâîïôùûç]/g, function (m) { return map[m]; });
    }
    return s.replace(/\s+/g, " ").trim();
}
function blacklisted(t, bl) {
    var n = norm(t);
    for (var i = 0; i < (bl || []).length; i++) if (n.indexOf(norm(bl[i])) >= 0) return true;
    return false;
}

// ------------------------------------------------------------------ appuis
function pressRobust(x, y, d) {
    var ok = false;
    try { ok = press(x, y, Math.max(1, d)); } catch (e) { ok = false; }
    if (!ok) {   // un seul renvoi, uniquement si Android a refuse le geste
        sleep(30);
        try { ok = press(x, y, Math.max(1, d)); } catch (e2) { ok = false; }
    }
    return ok;
}
function withPopupHidden(yScr, fn) {
    if (POPUP && !POPUP_HIDDEN && yScr <= POPUP_DEAD_SCR + 4) {
        hidePopup(true); sleep(160);
        try { return fn(); } finally { hidePopup(false); }
    }
    return fn();
}
function cmdTap(c) {
    var p = cap2scr(c.x, c.y);
    var ok = withPopupHidden(p[1], function () { return pressRobust(p[0], p[1], c.duration || 60); });
    return { ok: ok };
}
function cmdTapBatch(c) {
    var taps = c.taps || [];
    // le serveur n'envoie une salve que s'il est en lecture : un vieux verrou (> 1,5 s) est perime
    if (LOCAL_PAUSE && Date.now() - LOCAL_PAUSE > 1500) setPause(false);
    var t0 = Date.now(), lag = 0, lagMax = 0, done = 0, failed = 0, aborted = false;
    for (var i = 0; i < taps.length; i++) {
        if (i % 2 === 0 && (LOCAL_PAUSE || !RUNNING)) { aborted = true; break; }   // jamais au milieu d'un coup
        var tp = taps[i];
        var due = t0 + tp[2] + lag;
        var now = Date.now();
        if (now < due) sleep(due - now);
        else { var extra = now - due; lag += extra; if (extra > lagMax) lagMax = extra; }
        var p = cap2scr(tp[0], tp[1]);
        var ok = withPopupHidden(p[1], function () { return pressRobust(p[0], p[1], tp[3]); });
        if (!ok) failed++;
        done++;
    }
    return { ok: true, done: done, total: taps.length, failed: failed, lag_max: lagMax, lag_total: lag, aborted: aborted };
}

// ------------------------------------------------------------------ captures
function cmdScreenshot(c) {
    var img = images.captureScreen();
    if (!img) return { ok: false, error: "captureScreen null" };
    updateCapMap(img);
    var out = img, scaled = null;
    if (c.scale && c.scale < 1) { scaled = images.scale(img, c.scale, c.scale); out = scaled; }
    var b64 = images.toBase64(out, "jpg", c.quality || 88);
    var r = { ok: true, image: b64, width: out.getWidth(), height: out.getHeight(),
              dead: POPUP && !POPUP_HIDDEN ? scr2cap(0, POPUP_DEAD_SCR)[1] : 0,
              pkg: safePkg(), popup_visible: !!POPUP };
    if (scaled) scaled.recycle();
    return r;
}
function safePkg() { try { return currentPackage(); } catch (e) { return ""; } }

// ------------------------------------------------------------------ texte a l'ecran (accessibilite puis OCR)
function nodeVisible(n) {
    try { var b = n.bounds(); return b.width() > 4 && b.height() > 4 && b.top >= 0 && b.bottom <= SH() + 2; } catch (e) { return false; }
}
function findNode(t, exact) {
    var sels = exact ? [text(t), desc(t)] : [textContains(t), descContains(t)];
    for (var i = 0; i < sels.length; i++) {
        try {
            var list = sels[i].find();
            for (var j = 0; j < list.size(); j++) { var n = list.get(j); if (nodeVisible(n)) return n; }
        } catch (e) {}
    }
    return null;
}
function ocrItems(img) {
    var out = [];
    try {
        if (typeof ocr === "undefined") return out;
        var res = ocr.detect(img);
        var n = res.length !== undefined ? res.length : (res.size ? res.size() : 0);
        for (var i = 0; i < n; i++) {
            var r = res[i] !== undefined ? res[i] : res.get(i);
            var b = r.bounds;
            out.push({ text: String(r.label), b: [b.left, b.top, b.right, b.bottom], conf: r.confidence });
        }
    } catch (e) { log("ocr: " + e); }
    return out;
}
function cmdFindAny(c) {
    // Groupes cherches en UNE passe : renvoie `hits` (1er hit de CHAQUE groupe
    // trouve, trie par priorite = n° de groupe croissant) + les champs du 1er
    // hit (group/text/via/b) pour compatibilite avec les anciens serveurs.
    // - skip_ocr=true : accessibilite seule, AUCUNE capture (< 1 s). Ideal pour
    //   les balayages rapides a chaque cycle (popups par-dessus la grille).
    // - agent_timeout : accepte et ignore (aucune attente : un seul passage
    //   accessibilite + un seul OCR ; les cycles restent rapides).
    var groups = c.groups || [], ex = c.exact || [], bl = c.blacklist || [];
    var skipOcr = !!c.skip_ocr;
    var hits = [], seen = {};
    var g, k, i;
    for (g = 0; g < groups.length; g++) {
        for (k = 0; k < groups[g].length; k++) {
            var n = findNode(groups[g][k], !!ex[g]);
            if (n) {
                var t = (n.text() || n.desc() || "") + "";
                if (blacklisted(t, bl)) continue;
                hits.push({ group: g, text: t, via: "accessibilite", b: rectScr2Cap(n.bounds()) });
                seen[g] = true;
                break;
            }
        }
    }
    var ocrCount = 0;
    if (!skipOcr) {
        var img = images.captureScreen();
        updateCapMap(img);
        var items = ocrItems(img);
        ocrCount = items.length;
        for (g = 0; g < groups.length; g++) {
            if (seen[g]) continue;
            var done = false;
            for (k = 0; k < groups[g].length && !done; k++) {
                var want = norm(groups[g][k]);
                if (!want) continue;
                for (i = 0; i < items.length; i++) {
                    var got = norm(items[i].text);
                    var hit = ex[g] ? got === want : got.indexOf(want) >= 0;
                    if (hit && !blacklisted(items[i].text, bl)) {
                        hits.push({ group: g, text: items[i].text, via: "ocr", b: items[i].b });
                        seen[g] = true;
                        done = true;
                        break;
                    }
                }
            }
        }
    }
    hits.sort(function (a, b) { return a.group - b.group; });
    if (hits.length) {
        var h0 = hits[0];
        return { ok: true, found: true, group: h0.group, text: h0.text, via: h0.via,
                 b: h0.b, hits: hits, ocr_count: ocrCount };
    }
    return { ok: true, found: false, hits: [], ocr_count: ocrCount };
}
function cmdUiScan(c) {
    var out = [], max = c.max_items || 300;
    try {
        var list = selector().visibleToUser(true).find();
        for (var i = 0; i < list.size() && out.length < max; i++) {
            var n = list.get(i);
            var t = (n.text() || n.desc() || "") + "";
            if (!t) continue;
            out.push({ text: t, b: rectScr2Cap(n.bounds()), id: (n.id() || "") + "", via: "accessibilite" });
        }
    } catch (e) {}
    return { ok: true, items: out };
}
function cmdOcrRegion(c) {
    var img = images.captureScreen();
    updateCapMap(img);
    var r = c.region || [0, 0, img.getWidth(), img.getHeight()];
    var clip = images.clip(img, r[0], r[1], r[2] - r[0], r[3] - r[1]);
    var items = ocrItems(clip);
    clip.recycle();
    for (var i = 0; i < items.length; i++) {
        var b = items[i].b;
        items[i].b = [b[0] + r[0], b[1] + r[1], b[2] + r[0], b[3] + r[1]];
    }
    return { ok: true, items: items };
}
function cmdOcrImage(c) {
    // OCR d'une image fournie en base64 (ex. reference emplacements/) :
    // sert a calibrer les boutons sans passer par l'ecran live.
    // Renvoie les blocs en pixels DE L'IMAGE (le serveur convertit en fractions).
    if (typeof ocr === "undefined") return { ok: false, error: "ocr indisponible" };
    var img = null;
    try {
        try { img = images.fromBase64(c.image); }
        catch (e1) {
            var data = android.util.Base64.decode(c.image, 0);
            img = images.fromBytes(data);
        }
        if (!img) return { ok: false, error: "image illisible" };
        // fromBase64 peut rendre une image mise a l'echelle : on travaille
        // dans ses pixels natifs, les boites restent coherentes.
        var items = ocrItems(img);
        var r = { ok: true, items: items, w: img.getWidth(), h: img.getHeight() };
        try { img.recycle(); } catch (e2) {}
        return r;
    } catch (e) { return { ok: false, error: String(e && e.message || e) }; }
}

// ------------------------------------------------------------------ dispatch
function execute(cmd) {
    var r;
    try {
        switch (cmd.type) {
            case "screenshot": r = cmdScreenshot(cmd); break;
            case "tap": r = cmdTap(cmd); break;
            case "tap_batch": r = cmdTapBatch(cmd); break;
            case "find_any": r = cmdFindAny(cmd); break;
            case "ui_scan": r = cmdUiScan(cmd); break;
            case "ocr_region": r = cmdOcrRegion(cmd); break;
            case "ocr_image": r = cmdOcrImage(cmd); break;
            case "launch": app.launch(cmd.package); r = { ok: true }; break;
            case "back": back(); r = { ok: true }; break;
            case "sleep": sleep(cmd.ms || 100); r = { ok: true }; break;
            case "toast": toast(cmd.msg || ""); r = { ok: true }; break;
            case "open_url": app.openUrl(cmd.url); r = { ok: true }; break;
            case "quit": RUNNING = false; r = { ok: true }; break;
            default: r = { ok: false, error: "commande inconnue " + cmd.type };
        }
    } catch (e) { r = { ok: false, error: String(e) }; }
    r.id = cmd.id;
    return r;
}
function screenInfo() {
    return { version: VERSION, w: SW(), h: SH(), model: device.model, pkg: safePkg(),
             dead: POPUP_DEAD_SCR, features: ["tap", "tap_batch", "screenshot", "find_any", "ui_scan", "ocr_region", "ocr_image"] };
}

// ------------------------------------------------------------------ requetes popup
function api(method, path, body) {
    try {
        var r = method === "GET" ? http.get(SERVER + path) : http.postJson(SERVER + path, body || {});
        return r.statusCode === 200 ? r.body.json() : null;
    } catch (e) { return null; }
}
function control(action, extra) {
    threads.start(function () {
        var b = extra || {}; b.action = action;
        api("POST", "/api/control", b);
    });
}

// ------------------------------------------------------------------ barre flottante
var PU_XML =
    '<vertical id="root" bg="#EB0D0D0D" padding="6 5 6 5">' +
    '  <horizontal gravity="center_vertical">' +
    '    <text id="bPlay" text="❚❚" textColor="#C6FF00" textSize="17sp" w="46" h="40" gravity="center" bg="#262626"/>' +
    '    <text id="bStop" text="■" textColor="#FF5A5A" textSize="17sp" w="46" h="40" gravity="center" bg="#262626" marginLeft="5"/>' +
    '    <text id="bSpeed" text="Auto ▾" textColor="#FFFFFF" textSize="13sp" textStyle="bold" h="40" padding="10 0 10 0" gravity="center" bg="#5B2BFF" marginLeft="5"/>' +
    '    <text id="tState" text="Connexion…" textColor="#E6E6E6" textSize="11sp" layout_weight="1" marginLeft="8" maxLines="2"/>' +
    '    <text id="bMove" text="⇅" textColor="#FFFFFF" textSize="16sp" w="34" h="40" gravity="center"/>' +
    '    <text id="bClose" text="✕" textColor="#999999" textSize="15sp" w="30" h="40" gravity="center"/>' +
    '  </horizontal>' +
    '  <vertical id="lst" visibility="gone" marginTop="5"/>' +
    '</vertical>';

function statusBarH() {
    try {
        var res = context.getResources();
        return res.getDimensionPixelSize(res.getIdentifier("status_bar_height", "dimen", "android"));
    } catch (e) { return 60; }
}
function hidePopup(h) {
    if (!POPUP) return;
    POPUP_HIDDEN = h;
    ui.run(function () { try { POPUP.setPosition(h ? -SW() * 2 : 0, h ? -2000 : POPUP_Y); } catch (e) {} });
}
function measureDead() {
    try { POPUP_DEAD_SCR = POPUP_Y + POPUP.getHeight(); } catch (e) { POPUP_DEAD_SCR = POPUP_Y + 140; }
}
function buildList(choices, current) {
    var list = POPUP.lst;
    list.removeAllViews();
    for (var i = 0; i < choices.length; i++) {
        (function (ch) {
            var tv = new android.widget.TextView(context);
            tv.setText((ch.id === current ? "● " : "○ ") + ch.label);
            tv.setTextColor(android.graphics.Color.parseColor(ch.id === current ? "#C6FF00" : "#FFFFFF"));
            tv.setTextSize(14);
            tv.setPadding(24, 18, 24, 18);
            tv.setBackgroundColor(android.graphics.Color.parseColor("#1A1A1A"));
            tv.setOnClickListener(new android.view.View.OnClickListener({
                onClick: function () {
                    control("speed", { level: ch.id });
                    POPUP.lst.setVisibility(8);
                    setTimeout(measureDead, 150);
                }
            }));
            var lp = new android.widget.LinearLayout.LayoutParams(-1, -2);
            lp.setMargins(0, 3, 0, 0);
            list.addView(tv, lp);
        })(choices[i]);
    }
}
var POPUP_TOASTED = false;
function safeStartPopup(quiet) {
    if (POPUP || !RUNNING) return;
    try {
        if (typeof floaty !== "undefined" && floaty.checkPermission) {
            try {
                if (!floaty.checkPermission()) {
                    if (!quiet && !POPUP_TOASTED) {
                        toast("Autorise « Afficher par-dessus » pour la barre flottante");
                        POPUP_TOASTED = true;
                    }
                    try { floaty.requestPermission(); } catch (e2) {}
                }
            } catch (e3) {}
        }
        startPopup();
        POPUP_TOASTED = false;
    } catch (e) {
        log("popup impossible : " + e);
        if (!quiet && !POPUP_TOASTED) { toast("Barre flottante impossible : " + e); POPUP_TOASTED = true; }
        POPUP = null;
    }
}
function startPopupWatcher() {
    // Si la barre est fermee (clic ✕) mais que le serveur la veut affichee,
    // elle revient toute seule sans redemarrer l'agent (ex. reactivation web).
    threads.start(function () {
        while (RUNNING) {
            sleep(2000);
            if (POPUP || !RUNNING) continue;
            try {
                var st = api("GET", "/api/popup/state");
                if (!st || st.enabled !== false) safeStartPopup(true);
            } catch (e) {}
        }
    });
}
function startPopup() {
    if (POPUP) return;
    try {
        POPUP = floaty.rawWindow(PU_XML);
    } catch (e) { log("floaty.rawWindow : " + e); POPUP = null; throw e; }
    POPUP_Y = statusBarH();
    ui.run(function () {
        POPUP.setSize(SW(), -2);
        POPUP.setPosition(0, POPUP_Y);
        POPUP.bPlay.on("click", function () {
            if (POPUP_STATE.state === "running") { setPause(true); control("pause"); POPUP.bPlay.setText("▶"); }
            else { setPause(false); control("start"); POPUP.bPlay.setText("❚❚"); }
        });
        POPUP.bStop.on("click", function () { setPause(true); control("stop"); toast("Arrêt (appui long = quitter l'agent)"); });
        POPUP.bStop.on("long_click", function () {
            setPause(true); control("stop"); toast("Agent SudokuBot fermé");
            RUNNING = false;
            threads.start(function () { sleep(400); engines.myEngine().forceStop(); });
            return true;
        });
        POPUP.bSpeed.on("click", function () {
            var vis = POPUP.lst.getVisibility() === 0;
            if (!vis) buildList(POPUP_STATE.choices || [], POPUP_STATE.choice);
            POPUP.lst.setVisibility(vis ? 8 : 0);
            setTimeout(measureDead, 150);
        });
        POPUP.bMove.on("click", function () {   // bascule haut <-> sous la barre d'etat + 25 %
            POPUP_Y = POPUP_Y === statusBarH() ? Math.round(SH() * 0.25) : statusBarH();
            POPUP.setPosition(0, POPUP_Y);
            setTimeout(measureDead, 150);
        });
        POPUP.bClose.on("click", function () {
            // Clic = masquage local seulement : la barre revient au prochain
            // lancement (ou toute seule si le watcher la reveut). Pas d'ecriture config.
            closePopup();
            toast("Barre masquee (relance l'agent pour la revoir)");
        });
        POPUP.bClose.on("long_click", function () {
            // Appui long = desactivation persistante (popup.enabled=false).
            threads.start(function () { api("POST", "/api/popup/close", {}); });
            closePopup();
            toast("Barre desactivee (popup.enabled=false)");
            return true;
        });
    });
    setTimeout(measureDead, 400);
    threads.start(function () {
        var lastLabel = "";
        while (RUNNING && POPUP) {
            var st = api("GET", "/api/popup/state");
            if (st) {
                POPUP_STATE = st;
                if (st.state === "running" && LOCAL_PAUSE && Date.now() - LOCAL_PAUSE > 1500) setPause(false);
                if (st.enabled === false) { closePopup(); break; }
                ui.run(function () {
                    try {
                        POPUP.bPlay.setText(st.state === "running" ? "❚❚" : "▶");
                        POPUP.bSpeed.setText(st.label + " ▾");
                        POPUP.tState.setText(st.msg + "\nParties : " + st.games + " · coups : " + st.moves);
                        if (POPUP.lst.getVisibility() === 0 && st.label !== lastLabel) buildList(st.choices || [], st.choice);
                    } catch (e) {}
                });
                lastLabel = st.label;
            } else {
                ui.run(function () { try { POPUP.tState.setText("Serveur Termux injoignable…"); } catch (e) {} });
            }
            sleep((st && st.poll_ms) || 600);
            measureDead();
        }
    });
}
function closePopup() {
    var w = POPUP; POPUP = null; POPUP_DEAD_SCR = 0;
    if (w) ui.run(function () { try { w.close(); } catch (e) {} });
}

// ------------------------------------------------------------------ secours : Volume + = pause / reprise
function startKeys() {
    try {
        events.observeKey();
        events.onKeyDown("volume_up", function () {
            if (POPUP_STATE.state === "running") { setPause(true); control("pause"); toast("Pause"); }
            else { setPause(false); control("start"); toast("Reprise"); }
        });
    } catch (e) { log("touches: " + e); }
}

// ------------------------------------------------------------------ demarrage
function main() {
    toast("SudokuBot agent " + VERSION);
    // La barre doit etre visible meme en mode --beta (attente « Demarrer »)
    // et meme si le serveur est injoignable : demarrage tente d'emblee.
    try {
        var st0 = api("GET", "/api/popup/state");
        if (!st0 || st0.enabled !== false) safeStartPopup();
    } catch (e) { try { safeStartPopup(); } catch (e2) {} }
    startPopupWatcher();
    // porte de depart (mode --beta) : attend « Demarrer » sur la page web
    var gate = api("GET", "/api/agent/gate");
    while (RUNNING && gate && gate.go === false) { sleep(1000); gate = api("GET", "/api/agent/gate"); }
    if (!images.requestScreenCapture(false)) { toast("Capture d'écran refusée : arrêt"); exit(); }
    sleep(300);
    try {
        var st = api("GET", "/api/popup/state");
        if (!st || st.enabled !== false) safeStartPopup();
    } catch (e2b) { try { safeStartPopup(); } catch (e3) {} }
    startKeys();

    var pending = null;
    var fails = 0;
    while (RUNNING) {
        var data = null;
        try {
            var resp = http.postJson(SERVER + "/api/agent/exchange", { info: screenInfo(), result: pending, wait: 4 });
            data = resp.body.json();
            pending = null;
            fails = 0;
        } catch (e) {
            fails++;
            if (fails === 1 || fails % 15 === 0) log("serveur injoignable : " + e);
            setPause(true);              // securite : aucune salve sans serveur
            sleep(2000);
            continue;
        }
        if (data && data.cmd) pending = execute(data.cmd);
    }
    closePopup();
}
main();
