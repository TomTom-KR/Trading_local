# Trading Local

Lokale Trading-Plattform zum Forschen und Üben: Minutenkurse, minütliche Nachrichtenanalyse mit einem
lokalen Sprachmodell (Ollama `qwen3.5:9b`), automatische Long- und Short-Empfehlungen per E-Mail und eine
Handelssimulation, in der ein KI-Depot gegen dein eigenes Depot antritt.

**Wichtig:** Trading Local ist eine Simulation. Es werden keine echten Orders ausgeführt, und die
Empfehlungen sind keine Anlageberatung.

Die ausführliche Dokumentation mit Screenshots (Deutsch und Englisch) steht in
[Trading_Local_Dokumentation.html](Trading_Local_Dokumentation.html).

## Was die Plattform macht

- **Marktdaten:** 1-Minuten-Balken und Kurse von 71 Aktien aus Asien, Europa und den USA sowie 17 Indizes,
  12 Devisenpaaren, 5 Futures und 3 Kryptowährungen (108 Werte, in der Oberfläche erweiterbar). Quelle ist
  Yahoo Finance über einen Live-Stream und regelmäßigen Abgleich. Viele
  Börsen liefert Yahoo 15 bis 20 Minuten verzögert; die Plattform zeigt die Verzögerung überall an.
- **Handelszeiten:** Kalender für 17 Börsen mit Feiertagen, verkürzten Handelstagen und Mittagspausen.
  Signale und Ausführungen gibt es nur während der regulären Handelszeit.
- **Nachrichten:** 147 RSS-Feeds werden jede Minute abgefragt. Relevante Meldungen bewertet das lokale
  Sprachmodell nach Relevanz, Stimmung und betroffenen Werten.
- **Signale:** Fünf klassische Setups mit festen Lehrbuch-Parametern erzeugen Long- und Short-Kandidaten.
  Das Sprachmodell kann einen Kandidaten bestätigen, eine kleinere Position empfehlen oder ihn ablehnen;
  Kurse, Stopps und Ziele berechnet immer der Code.
- **Simulation:** Zwei Papierdepots mit identischen Regeln und Kosten. Das KI-Depot handelt die
  Empfehlungen automatisch, das Depot „Ich“ bedienst du selbst.
- **Ehrliche Auswertung:** Jedes erkannte Signal wird verfolgt, auch die verworfenen. Die Validierung
  spielt die gespeicherte Historie ohne Blick in die Zukunft nach und vergleicht mit Zufallseinstiegen.

## Voraussetzungen

- Windows 10/11
- Python 3.15
- [Ollama](https://ollama.com) mit dem Modell `qwen3.5:9b` und der Benutzer-Umgebungsvariable
  `OLLAMA_CONTEXT_LENGTH=4096`
- Eine Grafikkarte mit mindestens 8 GB Speicher wird empfohlen

## Installation

```powershell
uv venv --python C:\Python315\python.exe .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
Copy-Item .env.example .env   # danach .env öffnen und ausfüllen
```

Ohne `uv` geht es auch mit `python -m venv .venv` und `.venv\Scripts\pip install -r requirements.txt`.

## Starten

```powershell
.venv\Scripts\python.exe -m trading_local
```

Der Browser öffnet `http://127.0.0.1:8000/`. Beim ersten Start lädt die Plattform im Hintergrund die
Kurshistorie (rund 30 Tage Minutenbalken und 5 Jahre Tageskurse). Das dauert etwa 10 Minuten; die
Oberfläche ist sofort bedienbar. Beenden mit `Strg+C`.

| Option | Wirkung |
|---|---|
| `--offline` | keine Datenabrufe und Analysen, nur die Oberfläche mit den gespeicherten Daten |
| `--port 8001` | anderer Port |
| `--no-browser` | Browser nicht automatisch öffnen |

Die Validierung lässt sich auch ohne Oberfläche starten:
`.venv\Scripts\python.exe -m trading_local.validate --days 30`.

## Konfiguration

Alle Einstellungen stehen in `.env` (Vorlage: [.env.example](.env.example)). Die wichtigsten:

| Einstellung | Standard | Bedeutung |
|---|---|---|
| `TL_MAIL_MODE` | `file` | `smtp` verschickt Mails, `file` legt sie als `.eml` in `data/outbox` ab, `off` schaltet sie ab |
| `TL_SMTP_HOST`, `TL_SMTP_PORT`, `TL_SMTP_SECURITY` | `mail.gmx.net`, `465`, `ssl` | Mailserver; Port 587 mit `starttls` geht auch |
| `TL_SMTP_USER`, `TL_SMTP_PASSWORD`, `TL_MAIL_FROM`, `TL_MAIL_TO` | – | Zugangsdaten, Absender und Empfänger |
| `TL_OLLAMA_MODEL` | `qwen3.5:9b` | Sprachmodell |
| `TL_OLLAMA_NUM_CTX` | `4096` | Kontextlänge; gleicher Wert wie `OLLAMA_CONTEXT_LENGTH` |
| `TL_OLLAMA_TEMPERATURE` | `0.1` | Temperatur für alle KI-Aufgaben |
| `TL_START_CAPITAL`, `TL_BASE_CURRENCY` | `100000`, `EUR` | Startkapital beider Depots |
| `TL_HOST`, `TL_PORT` | `127.0.0.1`, `8000` | Adresse des Webservice |

Bei GMX muss in den Postfach-Einstellungen der Zugriff über POP3/IMAP/SMTP erlaubt sein. Laufende
Einstellungen wie Mail-Limits, Regionen oder das Risiko pro Trade änderst du in der Oberfläche unter
**Einstellungen**.

## Daten und Sicherheit

- Alle Daten liegen in `data/`: die SQLite-Datenbank, Logs und der Mail-Ausgang. Minutenbalken bleiben
  dauerhaft gespeichert. Für ein Backup kopierst du den Ordner bei gestoppter Plattform.
- Der Webservice hat keine Anmeldung und lauscht deshalb nur auf `127.0.0.1`. Gib den Port nicht im
  Netzwerk frei.
- `.env` und `data/` werden nie nach Git eingecheckt.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest              # alle Offline-Tests
.venv\Scripts\python.exe -m pytest -m network   # Tests gegen Yahoo
.venv\Scripts\python.exe -m pytest -m ollama    # Tests gegen das lokale Modell
```

## Projektstruktur

| Ordner | Inhalt |
|---|---|
| `trading_local/markets` | Handelskalender aller Börsen |
| `trading_local/data` | Marktdaten: Yahoo-Stream, Abgleich, Minutenbalken |
| `trading_local/news` | Feeds, Duplikaterkennung, Zuordnung zu Werten |
| `trading_local/llm` | Anbindung an Ollama mit Prioritäts-Warteschlange |
| `trading_local/analysis` | Indikatoren, Setups, Score, Kalibrierung, Signal-Engine, Validierung |
| `trading_local/sim` | Papier-Broker für die Depots KI und Ich |
| `trading_local/alerts`, `trading_local/charts` | E-Mail-Alarme und Chartbilder |
| `trading_local/web` | REST-API, WebSocket und die Oberfläche (`static/`) |
| `trading_local/db` | Datenbankschema und Zugriffsfunktionen |
| `trading_local/runtime` | Verdrahtung und Minutenzyklus |
| `tests` | Tests je Paket |
| `tools` | Demo-Server und Werkzeuge für die Dokumentation |

## Lizenzhinweise

Die Kurscharts der Oberfläche nutzen [TradingView Lightweight Charts](https://www.tradingview.com/lightweight-charts/)
(Apache-2.0). Weitere Lizenzen und Quellen stehen in der Dokumentation.
