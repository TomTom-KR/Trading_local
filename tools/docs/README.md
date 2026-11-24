# Screenshots für die Dokumentation

`screenshots.mjs` fotografiert jede Ansicht der Oberfläche reproduzierbar und legt die PNG-Dateien in
`docs/screenshots/` ab. Es steuert das **installierte Microsoft Edge** (headless) über `playwright-core`;
ein Browser-Download ist nicht nötig.

Das Skript ist rein lesend: Es lädt Seiten, wechselt Reiter, sortiert die Marktliste, öffnet die
Signal-Schublade und tippt Beispielwerte ins Orderticket – es schickt **nie** ein Formular ab und ändert
nichts am Server oder an der Datenbank.

## Einmalig

```powershell
cd tools/docs
npm install
```

`node_modules/` ist über die `.gitignore` des Projekts ausgeschlossen.

## Echte Daten (laufende Plattform)

```powershell
node tools/docs/screenshots.mjs                      # alle Ansichten von http://127.0.0.1:8000
node tools/docs/screenshots.mjs --only=overview,ribbon
node tools/docs/screenshots.mjs --list               # alle Ansichts-IDs
```

## Beispieldaten (Demo-Server)

Ansichten, die auf einem frischen System noch leer sind (Signale, Depot, Validierung), zeigt der
Demo-Server mit erfundenen, in sich stimmigen Daten:

```powershell
.venv/Scripts/python.exe tools/mock_api.py --port 8765     # eigenes Terminal
node tools/docs/screenshots.mjs --base=http://127.0.0.1:8765
```

Der Demo-Server wird an seiner Versionskennung erkannt; seine Dateien bekommen automatisch die Endung
`-demo`. In der Dokumentation sind sie immer als **Beispieldaten** zu kennzeichnen.

## Optionen

| Option | Bedeutung | Standard |
| --- | --- | --- |
| `--base=<url>` | Server | `http://127.0.0.1:8000` |
| `--only=<ids>` | nur diese Ansichten (Komma-Liste) | alle (Demo: Auswahl) |
| `--out=<ordner>` | Zielordner | `docs/screenshots` |
| `--suffix=<text>` | Endung des Dateinamens | `-demo` beim Demo-Server |
| `--crypto=<symbol>` / `--equity=<symbol>` | Werte für die Instrument-Ansichten | `BTC-EUR` / `SAP.DE` |
| `--timeout=<ms>` | Wartezeit je Ansicht | `45000` |
| `--no-optimize` | PNGs nicht verlustfrei nachkomprimieren | – |

Dateinamen: `<ansicht>-<dark|light>-<desktop|phone>[-demo].png`. Desktop = 1440 × 900 (lange Seiten als
ganze Seite), Telefon = 390 × 844 bei doppelter Pixeldichte (PNG 780 px breit).

## Was „fertig gerendert“ heißt

Das Design wird über den Schalter der App gesetzt (`localStorage['tl-theme']`) und zusätzlich über
`prefers-color-scheme`. Fotografiert wird erst, wenn die App gestartet ist, die WebSocket-Verbindung steht,
kein Lade-Platzhalter mehr zu sehen ist, jeder Kurschart Kerzen und Legende hat, alle Chart-Flächen gezeichnet
und die Schriften geladen sind und 600 ms lang keine Anfrage mehr läuft. Bei einem geöffneten Handelsplatz
wartet die Instrument-Ansicht außerdem (höchstens 75 s) auf die erste Live-Kerze über die WebSocket-Verbindung,
damit der Chart aktuell ist – der erste Abruf liefert nur abgeschlossene Kerzen, der Server schickt jede neu
abgeschlossene Minute. Die Krypto-Ansichten zeigen deshalb den 1-Minuten-Chart (dort ist jede Live-Kerze
vollständig), die Aktie den voreingestellten 15-Minuten-Chart. Fehlerzustände, leere Kurscharts,
Konsolenfehler, fehlgeschlagene Anfragen oder HTTP-Status ab 400 lassen die Ansicht scheitern: Die Datei wird
gelöscht, der Fehler ausgegeben, der Exit-Code ist 1.

## Video-Tutorial „Das Dashboard einrichten“

`tutorial.mjs` nimmt das Video-Tutorial reproduzierbar auf (`docs/video/Trading_Local_Dashboard_Tutorial.mp4`
und die Untertitel `….de.srt`). Das Drehbuch mit dem Sprechertext steht in `tutorial-script.mjs`; gesprochen
wird er mit der Windows-Stimme „Microsoft Hedda Desktop“ (`tutorial-speak.ps1`). Das Skript bedient die
Oberfläche im installierten Edge, nimmt den Bildschirm über das DevTools-Protokoll auf und setzt Bild und Ton
mit **ffmpeg** zusammen (nicht Teil des Projekts: Pfad mit `--ffmpeg=` oder über die Umgebungsvariable `FFMPEG`).

Es klickt „Speichern“ in den Einstellungen und läuft deshalb **nur gegen den Demo-Server**; gegen eine echte
Plattform bricht es ab. Starten Sie den Demo-Server vor jeder Aufnahme neu, damit die Einstellungen im
Ausgangszustand sind:

```powershell
.venv/Scripts/python.exe tools/mock_api.py --port 8765     # eigenes Terminal
node tools/docs/tutorial.mjs --ffmpeg=C:\Pfad\zu\ffmpeg.exe
node tools/docs/tutorial.mjs --audio-only --ffmpeg=...      # nur Ton erzeugen, Längen der Abschnitte ausgeben
```

Die Aufnahme läuft in Echtzeit (rund 5 Minuten). Texte ändern Sie in `tutorial-script.mjs`, Abläufe
(Mausbewegungen, Klicks) in `tutorial.mjs` unter derselben Abschnitts-ID. Das Video zeigt ausschließlich
Beispieldaten.
