// tutorial-script.mjs - Drehbuch des Video-Tutorials "Das Dashboard einrichten".
//
// Jeder Eintrag ist ein Abschnitt der Sprechertonspur: `text` erscheint als Untertitel, `say` (falls gesetzt)
// ist die Schreibweise, die die Windows-Stimme spricht (Lautschrift für Wörter, die sie sonst falsch liest).
// Die Handlungen dazu (Mausbewegungen, Klicks, Seitenwechsel) stehen in tutorial.mjs unter derselben ID.

/** Kapitel -> Beschriftung im Untertitel */
export const CHAPTERS = {
  intro: 'Einleitung',
  views: '1 · Die Ansichten',
  settings: '2 · Einstellungen',
  design: '3 · Design und Farben',
  outro: 'Zusammenfassung',
};

/** @type {Array<{id: string, chapter: keyof typeof CHAPTERS, text: string, say?: string}>} */
export const SEGMENTS = [
  {
    id: 'intro', chapter: 'intro',
    text: 'Willkommen bei Trading Local. Dieses Tutorial zeigt, wie Sie das Dashboard einrichten: Ansichten, Einstellungen und das Design in Hell und Dunkel.',
    say: 'Willkommen bei Trading Lokal. Dieses Tutorial zeigt, wie Sie das Dashboard einrichten: Ansichten, Einstellungen und das Design in Hell und Dunkel.',
  },
  {
    id: 'nav', chapter: 'views',
    text: 'Wir beginnen auf der Übersicht. Oben links liegt die Hauptnavigation mit sieben Ansichten.',
  },
  {
    id: 'overview', chapter: 'views',
    text: 'Das Band zeigt, welche Börsen gerade offen sind. Darunter folgen Empfehlungen, der Vergleich von KI-Depot und eigenem Depot, die Märkte und Nachrichten.',
    say: 'Das Band zeigt, welche Börsen gerade offen sind. Darunter folgen Empfehlungen, der Vergleich von Ka-I-Depot und eigenem Depot, die Märkte und Nachrichten.',
  },
  {
    id: 'lights', chapter: 'views',
    text: 'Die drei Ampeln oben rechts melden den Zustand von Daten, KI und Mail. Fahren Sie mit der Maus darüber, sehen Sie die Einzelheiten.',
    say: 'Die drei Ampeln oben rechts melden den Zustand von Daten, Ka-I und Mehl. Fahren Sie mit der Maus darüber, sehen Sie die Einzelheiten.',
  },
  {
    id: 'markets', chapter: 'views',
    text: 'Unter Märkte steht die ganze Beobachtungsliste. Ein Klick auf eine Spaltenüberschrift sortiert, zum Beispiel nach der Tagesveränderung.',
  },
  {
    id: 'filters', chapter: 'views',
    text: 'Suche und Filter grenzen die Liste ein. Die Schalter Aktiv und Handelbar bestimmen, welche Werte beobachtet und gehandelt werden.',
  },
  {
    id: 'instrument', chapter: 'views',
    text: 'Ein Klick auf einen Wert öffnet Chart, Chartanalyse und Orderticket. Den Zeitrahmen wählen Sie über der Grafik.',
  },
  {
    id: 'signals', chapter: 'views',
    text: 'Signale zeigt alle Empfehlungen. Die Reiter trennen offene und abgeschlossene, ein Klick auf eine Zeile öffnet die Details.',
    say: 'Signale zeigt alle Empfehlungen. Die Reiter trennen offene und abgeschlossene, ein Klick auf eine Zeile öffnet die Details.',
  },
  {
    id: 'depot', chapter: 'views',
    text: 'Im Depot treten das KI-Depot und Ihr Depot gegeneinander an. Positionen, Orders und Trades haben eigene Reiter.',
    say: 'Im Depot treten das Ka-I-Depot und Ihr Depot gegeneinander an. Positionen, Orders und Trädes haben eigene Reiter.',
  },
  {
    id: 'more', chapter: 'views',
    text: 'Nachrichten, Validierung und System ergänzen das Bild: bewertete Meldungen, die ehrliche Prüfung der Strategie und der Zustand aller Dienste.',
  },

  {
    id: 'open-settings', chapter: 'settings',
    text: 'Nun zur Konfiguration. Das Regler-Symbol oben rechts öffnet die Einstellungen.',
  },
  {
    id: 'groups', chapter: 'settings',
    text: 'Die Einstellungen sind in fünf Gruppen geordnet. Die erste regelt die E-Mail-Alarme.',
    say: 'Die Einstellungen sind in fünf Gruppen geordnet. Die erste regelt die Imehl-Alarme.',
  },
  {
    id: 'mail-regions', chapter: 'settings',
    text: 'Wählen wir bei den Regionen für Mails Asien ab, bleibt es nachts ruhig. Unten zählt die Leiste die offenen Änderungen.',
    say: 'Wählen wir bei den Regionen für Mehls Asien ab, bleibt es nachts ruhig. Unten zählt die Leiste die offenen Änderungen.',
  },
  {
    id: 'signal-group', chapter: 'settings',
    text: 'In der Gruppe Signale legen Sie fest, welche Setups gesucht werden und ob Long- und Short-Empfehlungen erlaubt sind.',
    say: 'In der Gruppe Signale legen Sie fest, welche Setaps gesucht werden und ob Long- und Schort-Empfehlungen erlaubt sind.',
  },
  {
    id: 'score', chapter: 'settings',
    text: 'Der Mindestscore bestimmt, ab wann ein Kandidat zur Empfehlung wird. Höher heißt weniger, aber strengere Signale.',
    say: 'Der Mindest-Skor bestimmt, ab wann ein Kandidat zur Empfehlung wird. Höher heißt weniger, aber strengere Signale.',
  },
  {
    id: 'risk-error', chapter: 'settings',
    text: 'Bei Risiko KI-Depot steuern Sie die Positionsgröße. Ein unzulässiger Wert wird sofort gemeldet, und Speichern bleibt gesperrt.',
    say: 'Bei Risiko Ka-I-Depot steuern Sie die Positionsgröße. Ein unzulässiger Wert wird sofort gemeldet, und Speichern bleibt gesperrt.',
  },
  {
    id: 'risk-fix', chapter: 'settings',
    text: 'Mit einem gültigen Wert verschwindet die Warnung.',
  },
  {
    id: 'benchmarks', chapter: 'settings',
    text: 'Unter Vergleich wählen Sie Indizes, gegen die sich die Depotkurve messen lässt. Nehmen wir den Nasdaq hundert dazu.',
    say: 'Unter Vergleich wählen Sie Indizes, gegen die sich die Depotkurve messen lässt. Nehmen wir den Nasdäk hundert dazu.',
  },
  {
    id: 'save', chapter: 'settings',
    text: 'Mit Speichern gelten die Änderungen ab dem nächsten Minutentakt. Verwerfen setzt alles zurück.',
  },
  {
    id: 'fixed', chapter: 'settings',
    text: 'Bewusst nicht einstellbar sind Strategieregeln und Indikatoren. Zugangsdaten, Datenordner und Port stehen in der Datei Punkt env.',
    say: 'Bewusst nicht einstellbar sind Strategieregeln und Indikatoren. Zugangsdaten, Datenordner und Port stehen in der Datei Punkt E N V.',
  },

  {
    id: 'theme', chapter: 'design',
    text: 'Jetzt das Design. Der Schalter ganz rechts wechselt zwischen dunkel und hell. Ohne eigene Wahl folgt die Oberfläche Ihrem Betriebssystem.',
  },
  {
    id: 'colors-light', chapter: 'design',
    text: 'Die Farben haben feste Bedeutungen: Grün für steigend und Long, Rot für fallend und Short, Blauviolett für die KI, Bernstein für Sie selbst.',
    say: 'Die Farben haben feste Bedeutungen: Grün für steigend und Long, Rot für fallend und Schort, Blauviolett für die Ka-I, Bernstein für Sie selbst.',
  },
  {
    id: 'light-markets', chapter: 'design',
    text: 'Im hellen Design bleiben Tabellen und Charts gut lesbar. Hier die Märkte und ein Kerzenchart.',
  },
  {
    id: 'light-depot', chapter: 'design',
    text: 'Und das Depot: Die Kurve zeigt die KI und mich in genau diesen festen Farben.',
    say: 'Und das Depot: Die Kurve zeigt die Ka-I und mich in genau diesen festen Farben.',
  },
  {
    id: 'colors-dark', chapter: 'design',
    text: 'Zurück zum dunklen Design, wieder mit einem Klick. Die Farben passen sich an, ihre Bedeutung bleibt gleich. Der Browser merkt sich die Wahl.',
  },
  {
    id: 'phone', chapter: 'design',
    text: 'Auf schmalen Bildschirmen wie dem Telefon stapeln sich die Bereiche untereinander, und Tabellen werden zu Karten.',
  },

  {
    id: 'outro', chapter: 'outro',
    text: 'Das war es: Ansichten über die Kopfleiste, Einstellungen über das Regler-Symbol, das Design über den Schalter. Viel Erfolg!',
  },
];
